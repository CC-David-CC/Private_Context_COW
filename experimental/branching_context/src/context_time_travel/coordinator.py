"""One request per selected GPU, shared serialized RAM/disk checkpoints.

Only the worker thread calls its native backend. The writer and reader operate
on immutable serialized bytes and canonical files, never on a native lease.
"""
from collections import deque
import base64
from concurrent.futures import CancelledError, Future, ThreadPoolExecutor
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, asdict
import fcntl
import hashlib
import json
import os
from pathlib import Path
import queue
import stat
import threading
import time

from .pools import SharedCheckpointPool, DiskCheckpointPool, PoolFull, PoolTransitionTrace, checkpoint_bytes, normalize_gpu_selection, worker_key
from .real import PreparedCheckpoint, supports_consumed_selection
from .checkpoint_cache import CheckpointCache
from .store import Store, BranchHead, Conflict, IntegrityError, InvalidOperation, NotFound, _id, _json_bytes, _name, _stage, _sync_dir, _private_directory
from .presentation_request import detach_request, request_digest, binding_for_snapshot
from .strata_backend import StrataImage
from .input_spans import InputHistory, history_for_snapshot, history_segment, history_digest, generated_text
from .grammar_first import _valid_name as _valid_grammar_state_name
from .prefix_scheduler import order_jobs, select_worker
from .cache_inventory import InventoryJournal, PoolNotifications
from .transfer_costs import TransferCosts


class Request:
    def __init__(self,operation,signature,*,pause_supported=False):
        self.operation,self.signature = operation,signature
        self.snapshot,self.durable = Future(),Future()
        self.events = queue.SimpleQueue()
        self.state = 'queued'
        self.pause_supported = pause_supported
        self.pause_event = threading.Event()
        self.created = time.monotonic()

    def request_pause(self):
        """Transient control only; the normal durable future proves settlement."""
        if self.durable.done():
            return False
        if not self.pause_supported:
            raise InvalidOperation('Request does not support consumed-token pause')
        self.pause_event.set()
        return True

    def emit(self,event,**fields):
        self.events.put(dict(event=event,operation=self.operation,time=time.monotonic(),**fields))


@dataclass
class _Job:
    request: Request
    branch: str
    head: object
    ids: tuple | InputHistory
    max_steps: int
    kind: str
    principal: object = None
    source_lease: object = None
    cancelled: object = None
    presentation: object = None
    reconstructed_source: object = None
    cache_keys_before: object = None
    sampling: object = None
    resume: bool = False
    source_head: object = None
    generation_stream: str | None = None
    presentation_source_operation: str | None = None
    grammar_registry: object = None
    source_grammar_engine: object = None
    grammar_state: str | None = None
    seed: int | None = None
    text_stops: tuple = ()
    input_request: object = None
    input_preparer_identity: str | None = None
    input_admission: object = None
    input_artifacts: object = None
    prefill_key: str | None = None
    prefill_claim: str | None = None


@dataclass
class _FailedPublication:
    """Small retry metadata; checkpoint bytes remain owned only by the RAM pool."""
    model: object
    branch: str
    head: object
    ids: tuple | InputHistory
    max_steps: int
    kind: str
    principal: object
    presentation: object
    snapshot: str
    key: tuple
    envelope: bytes
    sampling: object = None
    resume: bool = False
    source_head: object = None
    generation_stream: str | None = None
    presentation_source_operation: str | None = None
    seed: int | None = None
    text_stops: tuple = ()


class ContextCoordinator:
    def __init__(self,directory,config,model_factory,*,admission_timeout=30,transition_trace=None,
                 input_preparer=None):
        if input_preparer is not None:
            self._preparer_identity(input_preparer)
            if not all(callable(getattr(input_preparer,name,None)) for name in ('validate','prepare')):
                raise InvalidOperation('Input preparer requires validation and worker preparation')
        self._input_preparer = input_preparer
        self.config,self._factory = config,model_factory
        self.store = Store(directory)
        if transition_trace is not None and type(transition_trace) is not PoolTransitionTrace:
            raise TypeError('transition_trace must be a PoolTransitionTrace')
        self._pool_notifications=PoolNotifications() if config.cache_inventory else None
        self.pool = SharedCheckpointPool(config.ram_bytes,transition_trace=transition_trace,
            checkpoint_idle_seconds=config.checkpoint_idle_seconds,notifications=self._pool_notifications)
        self.disk = DiskCheckpointPool(self.store,config.disk_bytes,
            checkpoint_idle_seconds=config.checkpoint_idle_seconds,notifications=self._pool_notifications)
        self._process_lock = (self.store.root/'pool.lock').open('a+b')
        try: fcntl.flock(self._process_lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        except OSError:
            self._process_lock.close();self.pool.close()
            raise InvalidOperation('Another coordinator owns this context pool')
        self._condition = threading.Condition(threading.RLock())
        self._pending = deque()
        self._requests = {}
        self._input_preparations = {}
        self._branches = set()
        self._prefills = {}
        self._active_groups = {}
        self._service_rate = {}
        self._costs=TransferCosts() if config.scheduling=='transfer-aware' else None
        self._last_placement=None
        self._inventory=InventoryJournal(os.uname().nodename) if config.cache_inventory else None
        self._inventory_lock=threading.Lock()
        self._inventory_pool_seq=None; self._inventory_scan_at=0.
        self._inventory_disk={}; self._inventory_ram=[]; self._inventory_models=()
        self._workers = {gpu:dict(queue=queue.Queue(),ready=Future(),busy=False,model=None,
            active=0,max_active=0,last_root=None,health='starting',restarts=0,
            failure_reason=None,model_contract_id=None,execution_profile_id=None,native_inventory=[])
            for gpu in config.gpus}
        self._writer = ThreadPoolExecutor(max_workers=1,thread_name_prefix='cow-write')
        self._accepting,self._stopping,self._closed = True,False,False
        self._admission_timeout = admission_timeout
        self._failure_payloads = {}
        self._completed = self._failed = self._preloads = self._preload_failures = 0
        self._cancelled = 0
        self._qualified = False
        self._intents = self.store.root/'pool-operations'
        _private_directory(self._intents);_sync_dir(self.store.root)
        self._threads = []
        for gpu in config.gpus:
            t = threading.Thread(target=self._worker,args=(gpu,),name='cow-gpu-'+str(gpu))
            self._threads.append(t);t.start()
        self._dispatcher = threading.Thread(target=self._dispatch,name='cow-dispatch')
        self._dispatcher.start()
        self._expiry_stop = threading.Event()
        self._expiry_lock = threading.Lock()
        self._expiry_thread = None
        self._expiry_runs = 0
        self._expiry_last_error = None
        self._expiry_last_result = dict(ram=0,disk=0)
        if config.checkpoint_idle_seconds is not None:
            self._expiry_thread = threading.Thread(target=self._expire_idle,
                name='cow-idle-expiry',daemon=True)
            self._expiry_thread.start()

    def collect_expired(self):
        """Collect optional RAM/disk copies; never call a native GPU backend."""
        with self._expiry_lock:
            result=dict(ram=self.pool.collect_expired(),disk=self.disk.collect_expired())
            collect=getattr(self._input_preparer,'collect_expired',None)
            if callable(collect):result['images']=collect(pool=self.pool,disk=self.disk)
        with self._condition:
            self._expiry_runs += 1
            self._expiry_last_result = result
            self._expiry_last_error = None
        return result

    def _expire_idle(self):
        interval=max(.01,min(self.config.checkpoint_idle_seconds,240)/4)
        while not self._expiry_stop.wait(interval):
            try:
                self.collect_expired()
            except Exception as error:
                # Retain a bounded diagnostic, not a traceback that may own
                # checkpoint objects. A later successful sweep clears it.
                with self._condition:
                    self._expiry_last_error = type(error).__name__+': '+str(error)

    def _touch_idle_source(self,job):
        source_head=job.source_head or job.head
        if self.config.checkpoint_idle_seconds is not None and source_head is not None:
            try:
                # A successful resident request also used this computation.
                # Refresh its optional disk copy before releasing the source
                # read lease, without rereading payload bytes on a hot request.
                self.disk.touch(source_head.tip)
            except FileNotFoundError:
                pass  # Numerical copies are optional; history remains canonical.

    def ready(self,timeout=600):
        deadline = time.monotonic()+timeout
        models = [w['ready'].result(max(0,deadline-time.monotonic())) for w in self._workers.values()]
        identities = {(m.profile['model_contract_id'],m.profile['execution_profile_id']) for m in models}
        if len(identities) != 1: raise InvalidOperation('Workers do not share a qualified execution profile')
        with self._condition:
            self._qualified = True;self._condition.notify_all()
        return self

    def _intent(self,operation,signature,head,presentation=None,generation=None):
        with self.store._locked():
            self.store._assert_operation_id_available(operation, allow_evaluation_kind='pool')
            return self._intent_locked(operation, signature, head, presentation, generation)

    def _intent_locked(self,operation,signature,head,presentation=None,generation=None):
        path = self._intents/(operation+'.json')
        seeded_presentation = bool(presentation and 'seed' in presentation)
        controls = generation.get('controls') if isinstance(generation,dict) else None
        has_generation_controls = isinstance(controls,dict)
        schema = (5 if generation is not None and has_generation_controls else
                  3 if generation is not None else
                  4 if seeded_presentation else
                  1 if presentation is None else 2)
        if path.exists():
            envelope = json.loads(path.read_bytes())
            value = envelope['intent']
            stored_schema = value.get('schema')
            if type(stored_schema) is not int:
                raise IntegrityError('Operation intent schema is corrupt')
            fields = ({'schema','signature','head'} if stored_schema == 1 else
                      {'schema','signature','head','presentation'} if stored_schema == 2 else
                      {'schema','signature','head','presentation','generation'} if stored_schema in (3,5) else
                      {'schema','signature','head','presentation'} if stored_schema == 4 else set())
            if (envelope.get('checksum') != hashlib.sha256(_json_bytes(value)).hexdigest()
                    or set(value) != fields):
                raise IntegrityError('Operation intent is corrupt')
            if value['signature'] != signature: raise ValueError('Operation ID reused for different input')
            if stored_schema != schema: raise IntegrityError('Operation intent schema differs from signed request')
            if generation is not None and value.get('generation') != generation:
                raise IntegrityError('Operation generation descriptor differs from signed request')
            return value
        value = dict(schema=schema,signature=signature,head=asdict(head) if head is not None else None)
        if presentation is not None: value['presentation'] = presentation
        if generation is not None: value['generation'] = generation
        raw = _json_bytes(dict(intent=value,checksum=hashlib.sha256(_json_bytes(value)).hexdigest()))
        stage = _stage(self._intents,raw)
        try: os.link(stage,path);_sync_dir(self._intents)
        finally: stage.unlink(missing_ok=True)
        return value

    def request_source_head(self,branch,revision,operation):
        """Resolve the source head originally bound to an idempotent request.

        A retry after publication must validate its complete prompt against the
        original immutable source snapshot, while submit() remains responsible
        for signature validation, receipt recovery, and the current-head CAS.
        """
        _name(branch);_name(operation)
        if type(revision) is not int or revision < 0:
            raise Conflict('Requested branch revision changed')
        with self._condition:
            path = self._intents/(operation+'.json')
            if path.exists():
                envelope = json.loads(path.read_bytes())
                value = envelope.get('intent',{})
                schema = value.get('schema')
                if type(schema) is not int:
                    raise IntegrityError('Operation intent schema is corrupt')
                fields = ({'schema','signature','head'} if schema == 1 else
                          {'schema','signature','head','presentation'} if schema == 2 else
                          {'schema','signature','head','presentation','generation'} if schema in (3,5) else
                          {'schema','signature','head','presentation'} if schema == 4 else set())
                if (envelope.get('checksum') != hashlib.sha256(_json_bytes(value)).hexdigest()
                        or set(value) != fields):
                    raise IntegrityError('Operation intent is corrupt')
                if value['head'] is None:
                    raise Conflict('Operation has no append source head')
                generation=value.get('generation') if schema in (3,5) else None
                head = BranchHead(**(generation['source_head'] if generation is not None
                                     else value['head']))
                if head.name != branch or head.revision != revision:
                    raise Conflict('Operation ID is bound to a different source head')
                return head
            head = self.store.head(branch)
            if head.revision != revision:
                raise Conflict('Requested branch revision changed')
            return head

    def request_source_state(self,branch,revision,operation):
        """Read immutable request source history and controller under one lease."""
        head = self.request_source_head(branch,revision,operation)
        with self.store.read_lease(head.tip):
            record = self.store.get_snapshot(head.tip)
            history = history_for_snapshot(self.store,head.tip)
            return head,history,record['controller']

    def committed_generation(self,operation):
        """Validate a committed retry chain iteratively and reject cycles."""
        current=_name(operation);seen=set();chain=[]
        while True:
            if current in seen:
                raise IntegrityError('Original generation retry chain contains a cycle')
            seen.add(current)
            record=self._committed_generation_one(current)
            chain.append(record)
            generation=record['generation']
            parent=None if generation is None else generation.get('retry_source_operation')
            if parent is None:
                break
            if not isinstance(parent,str):
                raise IntegrityError('Retry source operation is malformed')
            current=_name(parent)
        for child,parent in zip(chain,chain[1:]):
            generation=child['generation']
            parent_generation=parent['generation']
            parent_stream=(parent_generation['stream_operation_id']
                           if parent_generation is not None else parent['operation_id'])
            if (parent['source_head']!=child['source_head']
                    or parent_stream!=generation['stream_operation_id']
                    or parent['presentation'].get('namespace_id')!=child['presentation'].get('namespace_id')):
                raise IntegrityError('Retry lineage changed the original generation source')
        return chain[0]

    def _committed_generation_one(self,operation):
        """Return a hash-checked original generation descriptor for retry.

        A presentation receipt proves the answer was committed; the intent
        binds the exact inputs and policy that produced it. Older intents can
        still be replayed, but cannot be changed and regenerated safely.
        """
        operation=_name(operation)
        receipt=self.store.presentation_receipt(operation)
        path=self._intents/(operation+'.json')
        try:
            envelope=json.loads(path.read_bytes())
            value=envelope['intent']
            generation=value.get('generation')
        except (OSError,KeyError,TypeError,ValueError) as error:
            raise InvalidOperation('Original request has no retryable generation descriptor') from error
        # Schema-2 requests predate retry descriptors. Their immutable
        # presentation receipt still supports exact answer replay, but it
        # cannot safely authorize changed-policy regeneration.
        if (isinstance(value,dict) and value.get('schema')==2
                and envelope.get('checksum')==hashlib.sha256(_json_bytes(value)).hexdigest()
                and set(value)=={'schema','signature','head','presentation'}):
            stored=receipt
            presentation=stored.get('presentation')
            if not isinstance(presentation,dict) or presentation.get('kind') not in (
                    'answer','initial_turn','user_turn','resume'):
                raise InvalidOperation('Legacy operation has no committed replayable answer')
            source_record=presentation.get('source')
            result=stored.get('result')
            if (not isinstance(source_record,dict) or not isinstance(result,dict)
                    or source_record.get('revision') is None
                    or source_record.get('snapshot') is None):
                raise IntegrityError('Legacy committed answer source is malformed')
            source=BranchHead(result['name'],source_record['revision'],
                              source_record['snapshot'],result['root'])
            return dict(operation_id=operation,intent=value,generation=None,
                        source_head=source,presentation=presentation,receipt=stored)
        if (not isinstance(value,dict) or value.get('schema') not in (3,5)
                or set(value)!={'schema','signature','head','presentation','generation'}
                or envelope.get('checksum')!=hashlib.sha256(_json_bytes(value)).hexdigest()
                or not isinstance(generation,dict)):
            raise IntegrityError('Original generation descriptor is corrupt')
        required={'schema','source_head','source_controller_sha256','source_history_sha256',
                  'ids','max_steps','sampling','resume','stream_operation_id','kind',
                  'retry_source_operation'}
        generation_schema=generation.get('schema') if isinstance(generation,dict) else None
        expected_fields=(required if generation_schema==1 else
                         required|{'grammar'} if generation_schema==2 else
                         required|{'controls'} if generation_schema==7 else set())
        if generation_schema==7 and 'grammar' in generation:
            expected_fields=expected_fields|{'grammar'}
        if generation_schema in (8,9,10,11):
            expected_fields=required|{'source_input_kind'}
            if generation_schema in (10,11):expected_fields|={'grammar'}
        if generation_schema in (12,13):
            expected_fields=required|{'source_input_kind','grammar','controls'}
        # Schema 8 binds a mixed source with text continuation; schema 9 binds
        # resolved mixed input spans. Schemas 10/11 add a checked grammar
        # descriptor to those respective input kinds. All are greedy-only.
        # Schema 7 is the seed/text-stop descriptor; legacy
        # text schemas 1/2 remain readable unchanged.
        controls = generation.get('controls') if generation_schema in (7,12,13) else None
        controls_seed = controls.get('seed') if isinstance(controls,dict) else None
        if (set(generation)!=expected_fields or type(generation_schema) is not int
                or (generation_schema in (8,9,10,11) and (generation.get('source_input_kind')!='canonical-mixed-spans-v1'
                    or generation.get('sampling') is not None))
                or (generation_schema in (12,13) and generation.get('source_input_kind')!='canonical-mixed-spans-v1')
                or type(generation.get('ids')) is not list
                or (generation_schema not in (9,11,13) and any(type(token) is not int or token<0 for token in generation['ids']))
                or type(generation.get('max_steps')) is not int or generation['max_steps']<1
                or type(generation.get('resume')) is not bool
                or (generation_schema in (7,12,13) and
                    (not isinstance(controls,dict)
                     or set(controls)!={'schema','seed','text_stops'}
                     or type(controls.get('schema')) is not int or controls['schema']!=1
                     or (controls_seed is not None and (type(controls_seed) is not int
                         or not 0 <= controls_seed < (1<<64)))
                     or not isinstance(controls.get('text_stops'),list)
                     or any(not isinstance(item,str) or not item for item in controls['text_stops'])
                     or len(set(controls['text_stops']))!=len(controls['text_stops'])))
                or generation.get('kind') not in ('answer','initial_turn','user_turn','resume')):
            raise IntegrityError('Original generation descriptor has invalid fields')
        try:
            delta=InputHistory(generation['ids']) if generation_schema in (9,11,13) else tuple(generation['ids'])
        except InvalidOperation as error:
            raise IntegrityError('Original typed generation inputs are malformed') from error
        if generation_schema in (9,11,13) and (delta.spans!=generation['ids'] or not delta or generation['resume']
                or generation['kind'] not in ('initial_turn','user_turn')):
            raise IntegrityError('Original typed generation descriptor has invalid boundaries')
        if ((value['schema']==3 and generation_schema in (7,12,13))
                or (value['schema']==5 and generation_schema not in (7,12,13))):
            raise IntegrityError('Operation intent schema differs from seed descriptor')
        grammar_descriptor=generation.get('grammar')
        has_grammar=(generation_schema in (2,10,11) or (generation_schema==7 and 'grammar' in generation)
                     or (generation_schema in (12,13) and grammar_descriptor is not None))
        if has_grammar:
            if (not isinstance(grammar_descriptor,dict)
                    or set(grammar_descriptor)!={'schema','agent_state','engine_id','spec_sha256',
                        'tokenizer_sha256','llguidance_version'}
                    or type(grammar_descriptor.get('schema')) is not int
                    or grammar_descriptor['schema']!=1):
                raise IntegrityError('Original grammar generation descriptor is malformed')
            if not _valid_grammar_state_name(grammar_descriptor['agent_state']):
                raise IntegrityError('Original grammar state name is malformed')
            for key in ('engine_id','spec_sha256','tokenizer_sha256'):
                digest=grammar_descriptor.get(key)
                if (not isinstance(digest,str) or len(digest)!=64
                        or any(char not in '0123456789abcdef' for char in digest)):
                    raise IntegrityError('Original grammar identity digest is malformed')
            version=grammar_descriptor.get('llguidance_version')
            if not isinstance(version,str) or not version or len(version)>64:
                raise IntegrityError('Original grammar compiler version is malformed')
        controls=generation.get('controls') if generation_schema in (7,12,13) else None
        if generation_schema in (7,12,13):
            if (not isinstance(controls,dict) or set(controls)!={'schema','seed','text_stops'}
                    or type(controls.get('schema')) is not int or controls['schema']!=1):
                raise IntegrityError('Original generation controls are malformed')
            seed=controls.get('seed')
            if seed is not None and (type(seed) is not int or not 0<=seed<2**64):
                raise IntegrityError('Original generation seed is malformed')
            try:
                from .text_stops import normalize_stop_sequences
                stops=normalize_stop_sequences(controls.get('text_stops'))
            except (TypeError,ValueError) as error:
                raise IntegrityError('Original text-stop controls are malformed') from error
            if (list(stops)!=controls.get('text_stops') or
                    (seed is None and not stops and (generation_schema==7 or generation['sampling'] is None))):
                raise IntegrityError('Empty generation controls must use a legacy descriptor')
        retry_source=generation.get('retry_source_operation')
        if retry_source is None:
            if generation.get('stream_operation_id')!=operation:
                raise IntegrityError('Original stream identity differs from operation ID')
        elif not isinstance(retry_source,str):
            raise IntegrityError('Retry source operation is malformed')
        try:
            source=BranchHead(**generation['source_head'])
            stored=receipt
            presentation=value['presentation']
            if stored.get('presentation') is None or not isinstance(presentation,dict):
                raise IntegrityError('Original presentation intent is unavailable')
            with self.store.read_lease(source.tip):
                record=self.store.get_snapshot(source.tip)
                history=history_for_snapshot(self.store,source.tip)
                if (generation_schema in (8,9,10,11,12,13))!=isinstance(history,InputHistory):
                    raise IntegrityError('Generation descriptor differs from canonical input kind')
            binding=stored.get('presentation')
            input_span=binding.get('input_span') if isinstance(binding,dict) else None
            output_span=binding.get('output_span') if isinstance(binding,dict) else None
            target=stored['result'].get('tip')
            if not isinstance(input_span,dict) or not isinstance(output_span,dict):
                raise IntegrityError('Committed generation lacks canonical input/output spans')
            with self.store.read_lease(target):
                target_history=history_for_snapshot(self.store,target)
                target_controller=self.store.get_snapshot(target)['controller']
            input_start=len(history)
            output_start=input_start+input_span.get('count',-1)
            if (binding.get('kind')!=generation['kind']
                    or input_span.get('start')!=input_start
                    or input_span.get('count')!=len(delta)
                    or type(input_span.get('count')) is not int
                    or history_segment(target_history,0,input_start)!=history
                    or (history_segment(target_history,input_start,output_start)!=delta if generation_schema in (9,11,13) else
                        list(generated_text(history_segment(target_history,input_start,output_start)))!=generation['ids'])
                    or output_span.get('start')!=output_start
                    or output_span.get('count')!=len(target_history)-output_start
                    or type(output_span.get('count')) is not int
                    or output_span.get('count')<0
                    or output_span.get('count')>generation['max_steps']):
                raise IntegrityError('Generation descriptor differs from committed presentation spans')
            if (hashlib.sha256(_json_bytes(record['controller'])).hexdigest()!=generation['source_controller_sha256']
                    or history_digest(history)!=generation['source_history_sha256']
                    or binding.get('source',{}).get('snapshot')!=source.tip
                    or binding.get('source',{}).get('revision')!=source.revision
                    or stored['result'].get('root')!=source.root):
                raise IntegrityError('Original generation source no longer matches its committed receipt')
            if has_grammar:
                constraint=target_controller.get('constraint') if isinstance(target_controller,dict) else None
                if (not isinstance(constraint,dict)
                        or any(constraint.get(key)!=grammar_descriptor[key]
                               for key in ('agent_state','engine_id','spec_sha256',
                                           'tokenizer_sha256','llguidance_version'))):
                    raise IntegrityError('Committed grammar controller differs from its generation descriptor')
                if generation['resume']:
                    source_constraint=record['controller'].get('constraint')
                    if (not isinstance(source_constraint,dict)
                            or any(source_constraint.get(key)!=grammar_descriptor[key]
                                   for key in ('agent_state','engine_id','spec_sha256',
                                               'tokenizer_sha256','llguidance_version'))):
                        raise IntegrityError('Original grammar resume changed the saved parser identity')
            if generation_schema in (12,13):
                from .min_p import validate_policy
                sampling=generation['sampling']
                if sampling is not None:
                    try:checked=validate_policy(sampling)
                    except (TypeError,ValueError) as error:
                        raise IntegrityError('Original mixed sampling policy is malformed') from error
                    if checked!=sampling or any(target_controller.get('policy',{}).get(key)!=setting for key,setting in checked.items()):
                        raise IntegrityError('Committed controller differs from its mixed sampling policy')
                elif target_controller.get('policy',{}).get('temperature')!=0:
                    raise IntegrityError('Mixed greedy generation has a sampled controller')
                if not has_grammar and target_controller.get('constraint') is not None:
                    raise IntegrityError('Mixed generation omitted its grammar binding')
                draw_base=(record['controller'].get('next_draw',0) if generation['resume'] and controls_seed is None else 0)
                # The canonical target controller preserves the exact cursor
                # committed by consume-validated proposals. Token count alone
                # cannot reconstruct XTC's support-dependent one/two draws.
                # Check its tight policy bounds here; runtime/native receipt
                # validation remains responsible for each exact draw count.
                from .min_p import xtc_enabled
                from .sampling_proposal import MAX_PROPOSAL_DRAWS
                minimum=output_span['count'] if sampling is not None and sampling['temperature']>0 else 0
                maximum=minimum*(MAX_PROPOSAL_DRAWS if sampling is not None and xtc_enabled(sampling) else 1)
                cursor=target_controller.get('next_draw')
                if type(cursor) is not int or not draw_base+minimum<=cursor<=draw_base+maximum:
                    raise IntegrityError('Mixed sampling draw cursor is outside its policy bounds')
            seed_value=controls_seed if generation_schema in (7,12,13) else None
            if seed_value is not None:
                expected_stream=hashlib.sha256(
                    f"cow-seed-v1:{seed_value}".encode('utf-8')).hexdigest()
                if (binding.get('generation_seed')!=seed_value
                        or target_controller.get('stream_key')!=expected_stream):
                    raise IntegrityError('Committed generation seed differs from its controller binding')
            if generation_schema in (7,12,13):
                target_stops=tuple(target_controller.get('text_stop',{}).get('stop_sequences',()))
                if target_stops!=stops:
                    raise IntegrityError('Committed controller differs from its text-stop controls')
                if generation['resume']:
                    source_stops=tuple(record['controller'].get('text_stop',{}).get('stop_sequences',()))
                    if source_stops!=stops:
                        raise IntegrityError('Resume changed its saved text-stop controls')
            if generation['resume'] != (generation['kind']=='resume'):
                raise IntegrityError('Original resume descriptor differs from presentation kind')
            if generation['kind']=='resume':
                if generation['ids'] or record['controller'].get('status')!='active':
                    raise IntegrityError('Original resume source is not an active empty-suffix request')
            elif record['controller'].get('status') not in ('idle','stopped'):
                raise IntegrityError('Original generation source controller is not resumable')
        except (KeyError,TypeError,ValueError) as error:
            raise IntegrityError('Original generation source is malformed') from error
        return dict(operation_id=operation,intent=value,generation=generation,
                    source_head=source,presentation=presentation,receipt=stored)

    def submit_retry_generation(self,source_operation,branch,expected_revision,operation,
                                sampling,presentation,*,principal=None,cancelled=None,seed=None,
                                grammar_registry=None,text_stops=None):
        """Run a changed-policy answer from a committed request's exact source."""
        original=self.committed_generation(source_operation)
        generation=original['generation']
        if generation is not None and generation.get('schema') in (9,11):
            raise InvalidOperation('Changed image-input generation controls are not yet qualified')
        grammar_descriptor=None if generation is None else generation.get('grammar')
        grammar_state=None
        if grammar_descriptor is None:grammar_registry=None
        if grammar_descriptor is not None:
            if grammar_registry is None:
                raise InvalidOperation('Retry requires the original server-owned grammar registry')
            try:
                grammar_state=grammar_descriptor['agent_state']
                if grammar_registry.descriptor(grammar_state)!=grammar_descriptor:
                    raise InvalidOperation('Retry grammar registry differs from original operation')
            except (KeyError,TypeError,ValueError,AttributeError) as error:
                raise InvalidOperation('Retry grammar registry is unavailable or changed') from error
        if sampling is None and seed is None:
            raise InvalidOperation('Changed retry requires an explicit sampled policy')
        source=original['source_head']
        controls=generation.get('controls',{}) if generation is not None else {}
        if seed is None and isinstance(controls,dict):
            # A changed retry inherits the committed generation's explicit
            # seed, including continuations; only a supplied seed can replace it.
            seed=controls.get('seed')
        if type(expected_revision) is not int or expected_revision<0:
            raise Conflict('Retry requires an explicit current branch revision')
        bound=self.store if principal is None else self.store.for_principal(principal)
        current=bound.head(branch)
        # _submit owns revision checking for new work and signature/receipt
        # recovery for an existing operation. A committed retry has already
        # advanced this head: rejecting its original revision here would make
        # a lost acknowledgment impossible to recover with the same ID.
        if current.root!=source.root:
            raise InvalidOperation('Retry destination belongs to another computational root')
        namespace=bound.branch_namespace(branch)
        if namespace!=original['presentation']['namespace_id']:
            raise InvalidOperation('Retry destination belongs to another presentation namespace')
        ids=tuple(generation['ids']);artifacts=None
        if generation.get('schema')==13:
            from .vision_artifacts import ImageArtifacts
            ids=InputHistory(generation['ids']);images=[span for span in ids.spans if span['kind']=='image']
            if images:
                profiles={(span['encoder_profile_id'],span['width']) for span in images}
                if len(profiles)!=1:raise IntegrityError('Original image delta has incompatible artifact profiles')
                encoder_id,width=next(iter(profiles))
                # Admission retains originals; the owning GPU worker loads or
                # regenerates derived bytes using the same artifact files/pools.
                artifacts=ImageArtifacts(self.store,self.pool,self.disk,encoder_profile_id=encoder_id,
                    expected_width=width,original_bytes_limit=None)
        return self._submit(branch,ids,generation['max_steps'],operation,
            'request',expected_revision,principal,cancelled,presentation,sampling,
            generation['resume'],source_head=source,
            generation_stream=generation['stream_operation_id'],
            presentation_source_operation=source_operation,
            retry_source_operation=source_operation,
            seed=seed,
            grammar_registry=grammar_registry,grammar_state=grammar_state,
            text_stops=text_stops,artifacts=artifacts)

    def _recover(self,job,receipt):
        """The Store receipt is authoritative; intent only binds caller input."""
        head = BranchHead(**receipt['result'])
        if head.name != job.branch: raise InvalidOperation('Operation receipt belongs to another branch')
        model = next(w['model'] for w in self._workers.values())
        bound = model if job.principal is None else model.for_principal(job.principal)
        with self.store.read_lease(head.tip):
            move_options = {}
            if 'checkpoint_receipt' in receipt:
                committed = self.store.checkpoint_receipt(job.request.operation)
                if committed['result'] != receipt['result']:
                    raise IntegrityError('Committed checkpoint receipt result differs')
                metadata = committed['checkpoint_receipt']
                if (metadata['snapshot'] != head.tip
                        or metadata['model_contract_id'] != model.profile['model_contract_id']
                        or metadata['execution_profile_id'] != model.profile['execution_profile_id']):
                    raise IntegrityError('Committed checkpoint receipt execution profile differs')
                move_options['checkpoint_receipt'] = metadata
            else:
                # Old operations did not bind a checkpoint summary to their
                # CAS receipt. Preserve their checked numerical-file recovery
                # path; never invent a payload identity after eviction.
                manifest = bound.checkpoint_manifest(head.tip)
                size = manifest['state_bytes']+4*manifest['logits_count']
                with self.pool.reserve(2*min(8*1024*1024,size)+4*manifest['logits_count'],timeout=self._admission_timeout):
                    bound.validate_checkpoint(head.tip)
                self.disk.touch(head.tip)
                metadata = manifest
            if job.presentation is not None and job.kind != 'create':
                source_head=job.source_head or job.head
                expected = binding_for_snapshot(bound.store,job.presentation,
                    operation_id=job.request.operation,source_head=source_head,
                    target_snapshot=head.tip,input_ids=job.ids,
                    generation_operation_id=job.generation_stream)
                retry_source_options = {}
                if job.presentation_source_operation is not None:
                    retry_source_options['presentation_source_operation'] = job.presentation_source_operation
                confirmed = bound.store.move(job.branch,head.revision-1,head.tip,
                    job.request.operation,presentation=expected,
                    **retry_source_options, **move_options)
            else:
                expected = None
                confirmed = bound.store.move(job.branch,head.revision-1,head.tip,job.request.operation,**move_options)
            history = bound.history(head.tip)
            if job.input_request is not None:
                # The signed intent plus committed receipt bind this root.
                # Replaying an acknowledged operation must not encode or
                # evaluate its inputs again, even after derived eviction.
                if job.kind != 'create' or job.head is not None:
                    raise IntegrityError('Prepared root receipt has an unexpected source')
                record=bound.store.get_snapshot(head.tip)
                if record['parent'] is not None or record['controller'] != {'status':'idle'}:
                    raise IntegrityError('Prepared root receipt is not an idle canonical root')
                if not isinstance(history,InputHistory):
                    raise IntegrityError('Prepared root receipt lacks canonical mixed inputs')
                job.ids=history
            source_head=job.source_head or job.head
            prior = ((InputHistory([]) if isinstance(history,InputHistory) else ())
                     if source_head is None else bound.history(source_head.tip))
            # Even plain text roots can use canonical typed spans. Add the
            # request to the empty canonical view instead of comparing that
            # view with a legacy token tuple. Images retain their span identity.
            expected_inputs=prior+job.ids
            if history_segment(history,0,len(prior)+len(job.ids)) != expected_inputs:
                raise IntegrityError('Committed request history differs from recorded inputs')
            if job.presentation is not None and job.kind != 'create':
                stored = bound.store.presentation_receipt(job.request.operation)
                if stored['presentation'] != expected:
                    raise IntegrityError('Committed presentation differs from operation intent')
            elif job.presentation is not None:
                root_operation = 'root-'+hashlib.sha256(job.request.operation.encode()).hexdigest()[:32]
                root_receipt = bound.store.presentation_receipt(root_operation)
                root_source = BranchHead(job.branch,0,head.tip,head.root)
                root_binding = binding_for_snapshot(bound.store,job.presentation,
                    operation_id=root_operation,source_head=root_source,target_snapshot=head.tip,
                    input_ids=job.ids)
                if root_receipt['presentation'] != root_binding:
                    raise IntegrityError('Committed root presentation differs from operation intent')
            tokens = list(generated_text(history_segment(history,len(prior)+len(job.ids))))
            answer = dict(snapshot=head.tip,tokens=tokens,recovered=True,request_seconds=None,generation_seconds=None)
            result = dict(branch=asdict(confirmed),snapshot=head.tip,payload_sha256=metadata['payload_sha256'],
                model_contract_id=model.profile['model_contract_id'],execution_profile_id=model.profile['execution_profile_id'])
            job.request.state = 'durable'
            job.request.emit('snapshot',**answer);job.request.emit('durable',receipt=result,recovered=True)
            job.request.snapshot.set_result(answer);job.request.durable.set_result(result)

    def _key(self,model,snapshot):
        return (model.profile['model_contract_id'],model.profile['execution_profile_id'],snapshot)

    def _close_worker_backend(self,model,gpu):
        succeeded=False
        try:
            model.backend.close()
            succeeded=True
        finally:
            process=getattr(model.backend,'process',None)
            self.pool.record_transition('native_worker_closed',
                (model.profile['model_contract_id'],model.profile['execution_profile_id'],
                 'worker-'+str(gpu)),gpu_id=gpu,
                pid=getattr(process,'pid',None),returncode=getattr(process,'returncode',None),
                close_succeeded=succeeded)

    @staticmethod
    def _worker_process_stopped(model):
        backend=getattr(model,'backend',None)
        if backend is None: return False
        if not bool(getattr(backend,'_closed',False)): return False
        process=getattr(backend,'process',None)
        if process is None:
            return bool(getattr(backend,'_closed',False))
        try: return process.poll() is not None
        except BaseException: return False

    def _replace_poisoned_worker(self,gpu,old_model,w):
        """Replace a poisoned native child only after its exit is observable."""
        replacement=None
        try:
            try:
                self._close_worker_backend(old_model,gpu)
            except BaseException as close_error:
                if not self._worker_process_stopped(old_model):
                    raise InvalidOperation('Poisoned native worker did not stop') from close_error
            if not self._worker_process_stopped(old_model):
                raise InvalidOperation('Poisoned native worker exit was not observed')
            old_model.cache.clear()
            with self._condition:
                stopping=self._stopping
            if stopping:
                raise InvalidOperation('Coordinator closing during worker replacement')
            replacement=self._factory(gpu)
            replacement.gpu_id=gpu
            self._observe_worker_restores(replacement,gpu)
            capabilities=self._backend_capabilities(replacement)
            expected_gpu=normalize_gpu_selection(old_model.profile['execution'].get('GPU'))
            actual_gpu=normalize_gpu_selection(replacement.profile['execution'].get('GPU'))
            if (replacement.profile.get('model_contract_id') != w['model_contract_id']
                    or replacement.profile.get('execution_profile_id') != w['execution_profile_id']
                    or actual_gpu != expected_gpu or actual_gpu != self.config.gpus):
                raise InvalidOperation('Replacement worker profile differs from the qualified worker')
            # Diagnostics can fail too. Keep the new child owned locally until
            # validation and its lifecycle record have both completed.
            self.pool.record_transition('native_worker_replaced',
                (replacement.profile['model_contract_id'],replacement.profile['execution_profile_id'],
                 'worker-'+str(gpu)),gpu_id=gpu,
                old_pid=getattr(getattr(old_model.backend,'process',None),'pid',None),
                old_returncode=getattr(getattr(old_model.backend,'process',None),'returncode',None),
                new_pid=getattr(getattr(replacement.backend,'process',None),'pid',None),
                restart_count=w['restarts']+1)
            with self._condition:
                if self._stopping:
                    raise InvalidOperation('Coordinator closing during worker replacement')
                w['model']=replacement
                w['capabilities']=capabilities
                w['health']='healthy'
                w['busy']=False
                w['restarts']+=1
                w['failure_reason']=None
                w['native_inventory']=[]
                self._condition.notify_all()
            return replacement
        except BaseException as error:
            reason=str(error)
            cleanup_owner=None
            if replacement is not None:
                try: self._close_worker_backend(replacement,gpu)
                except BaseException as close_error:
                    reason+='; replacement close: '+str(close_error)
                if not self._worker_process_stopped(replacement):
                    # The worker must retain any child whose exit is unproven
                    # for final shutdown. Dispatch remains disabled; receipt
                    # lookup continues using the retired model's metadata.
                    cleanup_owner=replacement
                    reason+='; replacement process exit unproven'
            with self._condition:
                w['health']='unavailable'
                w['failure_reason']=reason
                w['busy']=False
                self._condition.notify_all()
            try:
                self.pool.record_transition('native_worker_unavailable',
                    (w.get('model_contract_id'),w.get('execution_profile_id'),'worker-'+str(gpu)),
                    gpu_id=gpu,reason=reason)
            except BaseException as trace_error:
                with self._condition:
                    w['failure_reason']+='; unavailable trace: '+str(trace_error)
            return cleanup_owner

    def _submit(self,branch,ids,max_steps,operation,kind,expected_revision=None,principal=None,cancelled=None,
                presentation=None,sampling=None,resume=False,*,source_head=None,
                generation_stream=None,presentation_source_operation=None,
                retry_source_operation=None,grammar_registry=None,grammar_state=None,seed=None,
                text_stops=None,input_request=None,input_preparer_identity=None,artifacts=None):
        _name(branch);_name(operation)
        typed_delta=isinstance(ids,InputHistory)
        if not typed_delta:
            ids=tuple(ids)
            if any(type(t) is not int or t < 0 for t in ids):raise ValueError('Invalid token IDs')
        elif (kind!='request' or not ids or resume or principal is not None
                or text_stops):
            raise InvalidOperation('Typed image requests currently require owner-controlled greedy input deltas')
        if type(max_steps) is not int or max_steps < (1 if kind == 'request' else 0):
            raise ValueError('Invalid output bound')
        if kind == 'reseed' and (ids or max_steps != 0 or sampling is not None or resume):
            raise InvalidOperation('Reseed is a controller-only transition')
        if cancelled is not None and not callable(cancelled):
            raise ValueError('Cancellation check must be callable')
        if seed is not None:
            from .seed_controls import seed_stream
            seed_stream(seed)
            if kind not in ('request','reseed'):
                raise InvalidOperation('Explicit generation seed applies only to text requests')
        if type(resume) is not bool or (resume and
                ((sampling is None and grammar_registry is None and seed is None) or ids)):
            raise InvalidOperation('Generation resume requires a saved policy or grammar and no new input')
        if sampling is not None:
            from .min_p import validate_policy
            if kind != 'request':
                raise InvalidOperation('Sampling applies only to generation requests')
            sampling = validate_policy(sampling)
        if text_stops is not None:
            from .text_stops import normalize_stop_sequences
            text_stops=normalize_stop_sequences(text_stops)
            if not text_stops:
                text_stops=None
        if presentation is not None:
            presentation = detach_request(presentation,expected_kind='root' if kind == 'create' else None)
            allowed = ({'root'} if kind == 'create' else
                       {'reseed'} if kind == 'reseed' else
                       {'answer','initial_turn','user_turn','resume'})
            if presentation['kind'] not in allowed:
                raise InvalidOperation('Presentation kind does not match coordinator operation')
            if kind == 'create' and max_steps != 0:
                raise InvalidOperation('A root request cannot generate output')
            if kind != 'create' and kind != 'reseed' and max_steps < 1:
                raise InvalidOperation('A presentation turn requires a positive output bound')
            if kind == 'reseed' and 'seed' not in presentation:
                raise InvalidOperation('Reseed presentation requires its seed')
        if kind=='reseed':
            bound=self.store if principal is None else self.store.for_principal(principal)
            namespace=bound.branch_namespace(branch)
            if (namespace is not None) != (presentation is not None):
                raise InvalidOperation('Namespaced reseed requires a typed presentation binding')
        # Resolve a dynamic catalog once at admission. Jobs retain the actual
        # immutable registry; later registrations cannot change queued work.
        source_registry_facade=grammar_registry
        resolver=getattr(grammar_registry,'resolve_registry',None)
        if callable(resolver):
            saved_controller=None
            if kind=='reseed':
                bound=self.store if principal is None else self.store.for_principal(principal)
                selected_head=source_head if source_head is not None else bound.head(branch)
                saved_controller=bound.get_snapshot(selected_head.tip)['controller']
            grammar_registry=resolver(grammar_state,controller=saved_controller)
        grammar_descriptor=None
        if grammar_registry is None:
            if grammar_state is not None:
                raise InvalidOperation('A grammar state requires an enabled server registry')
        else:
            engine=grammar_registry.engine
            if kind == 'reseed':
                if grammar_state is not None or not callable(getattr(engine,'restore',None)):
                    raise InvalidOperation('Grammar reseed requires its configured engine')
            else:
                if (kind!='request' or type(grammar_state) is not str or max_steps<1
                        or not callable(getattr(grammar_registry,'descriptor',None))
                        or not callable(getattr(engine,'start',None))
                        or not callable(getattr(engine,'restore',None))):
                    raise InvalidOperation('Grammar selection requires a named state and generation request')
                grammar_descriptor=grammar_registry.descriptor(grammar_state)
            if kind == 'request':
                for worker in self._workers.values():
                    model=worker.get('model')
                    if model is None:
                        raise InvalidOperation('Grammar operation has no ready model worker')
                    execution=model.profile.get('execution',{})
                    validator=getattr(model.backend,'validate_grammar_engine',None)
                    if (not supports_consumed_selection(model.profile)
                            or list(engine.eos_tokens)!=execution.get('eos_tokens')
                            or not callable(validator)
                            or not callable(getattr(model.backend,'consume_generated',None))):
                        raise InvalidOperation('Worker profile does not support checked grammar generation')
                    validator(engine)
        signature_fields = dict(branch=branch,ids=ids.spans if typed_delta else list(ids),max_steps=max_steps,
            kind=kind,expected_revision=expected_revision,principal=principal)
        if typed_delta:signature_fields['input_kind']='canonical-mixed-spans-v1'
        if input_request is not None:
            if kind != 'create' or ids or max_steps != 0:
                raise InvalidOperation('Input preparation applies only to a new root')
            signature_fields.update(input_request_sha256=hashlib.sha256(_json_bytes(input_request)).hexdigest(),
                input_preparer_identity=input_preparer_identity)
        if seed is not None: signature_fields['seed'] = seed
        if presentation is not None: signature_fields['presentation_digest'] = request_digest(presentation)
        if sampling is not None:
            signature_fields.update(sampling=sampling,resume=resume)
        if grammar_descriptor is not None:
            if presentation is None:
                raise InvalidOperation('Grammar generation requires a retry-bound presentation request')
            signature_fields.update(grammar=grammar_descriptor,grammar_resume=resume)
        if text_stops is not None:
            signature_fields['text_stops']=list(text_stops)
        if source_head is not None:
            signature_fields['generation_source_head']=asdict(source_head)
        if generation_stream is not None:
            signature_fields['generation_stream']=generation_stream
        if presentation_source_operation is not None:
            signature_fields['presentation_source_operation']=presentation_source_operation
        if retry_source_operation is not None:
            signature_fields['retry_source_operation']=retry_source_operation
        signature = hashlib.sha256(_json_bytes(signature_fields)).hexdigest()
        with self._condition:
            if not self._accepting: raise InvalidOperation('Coordinator is closing')
            if not self._qualified: raise InvalidOperation('Call ready() before admitting requests')
            previous = self._requests.get(operation)
            if previous is not None:
                if previous.signature != signature: raise ValueError('Operation ID reused for different input')
                return previous
            if grammar_registry is not None and kind == 'request':
                if any(worker.get('model') is None or not callable(
                       getattr(worker['model'],'run_state_request',None))
                       for worker in self._workers.values()):
                    raise InvalidOperation('Worker does not support checked grammar requests')
            if sampling is not None:
                from .min_p import (supports_policy, adaptive_enabled, mirostat_enabled,
                                    xtc_enabled, SAMPLING_PROTOCOL_V6, SAMPLING_PROTOCOL_V7)
                from .sampling_proposal import (
                    XTC_PROPOSAL_PROTOCOL, MIROSTAT2_PROPOSAL_PROTOCOL,
                    ADAPTIVE_P_PROPOSAL_PROTOCOL,
                )
                def supports_worker(worker):
                    model = worker.get('model')
                    if model is None or not callable(getattr(model,'run_sampled_request',None)):
                        return False
                    protocol = model.profile.get('execution',{}).get('sampling_protocol')
                    proposal_family = (adaptive_enabled(sampling) or mirostat_enabled(sampling)
                                       or xtc_enabled(sampling))
                    if proposal_family:
                        if adaptive_enabled(sampling):
                            protocols = (ADAPTIVE_P_PROPOSAL_PROTOCOL,)
                        elif mirostat_enabled(sampling):
                            protocols = (MIROSTAT2_PROPOSAL_PROTOCOL, SAMPLING_PROTOCOL_V7)
                        else:
                            protocols = (XTC_PROPOSAL_PROTOCOL, SAMPLING_PROTOCOL_V6,
                                         SAMPLING_PROTOCOL_V7)
                        return (protocol in protocols
                                and callable(getattr(model.backend,'sample_selection_proposal',None)))
                    ordinary = supports_policy(protocol, sampling)
                    proposal = (protocol in (XTC_PROPOSAL_PROTOCOL, MIROSTAT2_PROPOSAL_PROTOCOL,
                                             ADAPTIVE_P_PROPOSAL_PROTOCOL, SAMPLING_PROTOCOL_V6)
                                and callable(getattr(model.backend,'sample_selection_proposal',None)))
                    return ordinary or proposal
                if any(not supports_worker(worker) for worker in self._workers.values()):
                    raise InvalidOperation('Worker profile does not support full-vocabulary Min-P')
            existing = (self._intents/(operation+'.json')).exists()
            legacy_replay = False
            if existing:
                try:
                    prior_value=json.loads((self._intents/(operation+'.json')).read_bytes())['intent']
                    prior_generation=prior_value.get('generation')
                except (OSError,KeyError,TypeError,ValueError) as error:
                    raise IntegrityError('Operation intent is corrupt') from error
                value = self._intent(operation,signature,None,presentation,prior_generation)
                head = BranchHead(**value['head']) if value['head'] is not None else None
                stored_generation=value.get('generation')
                if stored_generation is not None:
                    source_head=BranchHead(**stored_generation['source_head'])
                    generation_stream=stored_generation['stream_operation_id']
                    retry_source_operation=stored_generation.get('retry_source_operation')
                    presentation_source_operation=(retry_source_operation or presentation_source_operation)
                legacy_replay = prior_value.get('schema') == 2 and stored_generation is None
            elif kind == 'create':
                if (self.store.references/(branch+'.json')).exists(): raise Conflict('Branch already exists')
                head = None
            else:
                bound = self.store if principal is None else self.store.for_principal(principal)
                head = bound.head(branch)
                bound.authorize(head.tip,'append')
                if expected_revision is not None and head.revision != expected_revision:
                    raise Conflict('Requested branch revision changed')
            generation_source_head=source_head or head
            validation_head=generation_source_head or head
            if typed_delta:
                from .vision_execution import validate_boundary,RESUME_CONTRACT
                with self.store.read_lease(validation_head.tip):
                    source_history=history_for_snapshot(self.store,validation_head.tip)
                if not isinstance(source_history,InputHistory):
                    raise InvalidOperation('Typed image deltas require canonical mixed source history')
                if presentation is not None and presentation['kind'] not in ('initial_turn','user_turn'):
                    raise InvalidOperation('Typed image deltas require a new user turn')
                for worker in self._workers.values():
                    model=worker.get('model');execution=model.profile.get('execution',{}) if model is not None else {}
                    if execution.get('resume_contract')!=RESUME_CONTRACT:
                        raise InvalidOperation('Worker does not support typed image deltas')
                    validate_boundary(source_history+ids,capacity=execution['capacity']-max_steps,
                        vocab=model.backend.vocab_size,width=execution['embedding_width'])
            if resume:
                with self.store.read_lease(validation_head.tip):
                    decoder=self.store.get_snapshot(validation_head.tip)['controller']
                    if (decoder.get('schema') not in (1,2,3,4,5,6)
                            or decoder.get('status')!='active'):
                        raise InvalidOperation('Resume requires an active generation controller')
                saved_constraint=decoder.get('constraint')
                if grammar_registry is not None:
                    if (not isinstance(saved_constraint,dict)
                            or saved_constraint.get('agent_state')!=grammar_state
                            or not grammar_registry.matches_controller(decoder)):
                        raise InvalidOperation('Grammar resume must restore the saved registry state')
                elif saved_constraint is not None:
                    raise InvalidOperation('Grammar continuation requires its server-owned registry')
                saved_stops=tuple(decoder.get('text_stop',{}).get('stop_sequences',()))
                if text_stops is not None and text_stops!=saved_stops:
                    raise InvalidOperation('Text stops cannot change while resuming a generation')
                if text_stops is None and saved_stops:
                    text_stops=saved_stops
                if decoder.get('schema') == 4:
                    from .sampling_proposal import proposal_protocol_supported
                    for worker in self._workers.values():
                        model = worker.get('model')
                        if (model is None
                                or not callable(getattr(model.backend, 'sample_selection_proposal', None))
                                or not proposal_protocol_supported(
                                    model.profile.get('execution', {}).get('sampling_protocol'), sampling)):
                            raise InvalidOperation('Schema-4 resume requires a compatible native proposal selector')
            if presentation is not None:
                bound = self.store if principal is None else self.store.for_principal(principal)
                bound.get_namespace(presentation['namespace_id'])
                if kind == 'create':
                    if (self.store.references/(branch+'.json')).exists():
                        if bound.branch_namespace(branch) != presentation['namespace_id']:
                            raise InvalidOperation('Branch belongs to another presentation namespace')
                else:
                    namespace = bound.branch_namespace(branch)
                    if namespace != presentation['namespace_id']:
                        raise InvalidOperation('Branch belongs to another presentation namespace')
                    with bound.read_lease(validation_head.tip):
                        controller_state = bound.get_snapshot(validation_head.tip)['controller']
                    status = controller_state.get('status')
                    p_kind = presentation['kind']
                    expected_status = {'answer':('idle',),'initial_turn':('idle',),
                        'user_turn':('stopped',),'resume':('active',),
                        'reseed':('active','stopped')}[p_kind]
                    if status not in expected_status:
                        raise InvalidOperation('Presentation kind does not match source controller state')
                    generation_settings=presentation.get('generation_settings')
                    if generation_settings is not None and generation_settings!=sampling:
                        raise InvalidOperation('Presentation generation settings differ from the request')
                    if p_kind in ('initial_turn','user_turn') and not ids:
                        raise InvalidOperation('A new user turn must consume input tokens')
                    if p_kind in ('answer','resume') and ids:
                        raise InvalidOperation('This presentation kind cannot add input tokens')
                    if p_kind == 'reseed' and (ids or max_steps != 0):
                        raise InvalidOperation('Reseed presentation cannot add inputs or generated tokens')
                    if p_kind == 'resume':
                        try:
                            prior_receipt = bound.presentation_receipt(operation)
                        except NotFound:
                            prior_receipt = None
                        if prior_receipt is not None:
                            prior = prior_receipt['presentation']
                            matches = (prior.get('kind') == 'resume'
                                and prior.get('namespace_id') == presentation['namespace_id']
                                and prior.get('output_span',{}).get('message_id') == presentation['output_message_id'])
                        else:
                            active = bound.active_presentation(branch, snapshot_id=validation_head.tip)
                            matches = (active.get('namespace_id') == presentation['namespace_id']
                                and active.get('message_id') == presentation['output_message_id'])
                        if not matches:
                            raise InvalidOperation('Resume assistant message differs from active presentation')
            generation=None
            if presentation is not None and kind=='request' and not legacy_replay:
                if generation_source_head is None:
                    raise InvalidOperation('A retryable generation requires a canonical source head')
                with self.store.read_lease(generation_source_head.tip):
                    source_record=self.store.get_snapshot(generation_source_head.tip)
                    source_history=history_for_snapshot(self.store,generation_source_head.tip)
                controls=(dict(schema=1,seed=seed,text_stops=list(text_stops or ()))
                          if seed is not None or text_stops is not None else None)
                mixed_source=isinstance(source_history,InputHistory)
                mixed_sampling=mixed_source and (controls is not None or sampling is not None)
                if mixed_sampling:
                    from .vision_controllers import require_basic_sampling
                    # VC07 deferred: keep admission aligned with real.py until
                    # mixed pending-decoder/stop state is durably qualified.
                    if text_stops:raise InvalidOperation('Mixed-input text stops are not yet qualified')
                    for worker in self._workers.values():
                        model=worker['model'];require_basic_sampling(model.profile.get('execution',{}),model.backend,sampling)
                    controls=dict(schema=1,seed=seed,text_stops=[])
                generation=dict(schema=(13 if typed_delta else 12) if mixed_sampling else
                                (11 if typed_delta else 10) if mixed_source and grammar_descriptor is not None else
                                9 if typed_delta else 8 if mixed_source else 7 if controls is not None else
                                (2 if grammar_descriptor is not None else 1),
                    source_head=asdict(generation_source_head),
                    source_controller_sha256=hashlib.sha256(_json_bytes(source_record['controller'])).hexdigest(),
                    source_history_sha256=history_digest(source_history),
                    # The descriptor records canonical continuation semantics.
                    # A legacy greedy continue executes through the greedy
                    # path (`resume=False`) while still continuing the active
                    # presentation/controller scope.
                    ids=ids.spans if typed_delta else list(ids),max_steps=max_steps,sampling=sampling,
                    resume=(presentation.get('kind')=='resume'),
                    stream_operation_id=generation_stream or operation,kind=presentation['kind'],
                    retry_source_operation=retry_source_operation)
                if mixed_source: generation['source_input_kind']='canonical-mixed-spans-v1'
                if grammar_descriptor is not None or mixed_sampling:
                    generation['grammar']=grammar_descriptor
                if controls is not None:
                    generation['controls']=controls
                if existing and generation!=prior_generation:
                    raise IntegrityError('Operation generation descriptor changed')
            with self.store._locked(): receipt = self.store._read_refs()['receipts'].get(operation)
            source_grammar_engine=None
            source_selection=generation_source_head or head
            if source_selection is not None:
                source_controller=self.store.get_snapshot(source_selection.tip)['controller']
                if source_controller.get('constraint') is not None and source_registry_facade is not None:
                    source_resolver=getattr(source_registry_facade,'resolve_registry',None)
                    source_registry=(source_resolver(controller=source_controller)
                                     if callable(source_resolver) else source_registry_facade)
                    if not source_registry.matches_controller(source_controller):
                        raise InvalidOperation('Saved source grammar is unavailable')
                    source_grammar_engine=source_registry.engine
            source_lease=None
            input_admission=None
            if typed_delta and receipt is None:
                from .vision_artifacts import ImageArtifacts
                images=[span for span in ids.spans if span['kind']=='image']
                if len(images)>8:raise InvalidOperation('A request supports at most eight new images')
                if images and (type(artifacts) is not ImageArtifacts
                        or artifacts.store.root.resolve()!=self.store.root.resolve()
                        or artifacts.ram is not self.pool or artifacts.disk is not self.disk):
                    raise InvalidOperation('Image deltas require artifacts owned by this coordinator and its pools')
                source_lease=ExitStack()
                try:
                    source_lease.enter_context(self.pool.reserve(2*len(_json_bytes(ids.spans)),timeout=0))
                    for span in images:
                        lease=artifacts.retain(span) if retry_source_operation is not None else artifacts.lease(span)
                        source_lease.enter_context(lease)
                except BaseException:
                    source_lease.close();raise
            if input_request is not None and receipt is None:
                # Admission must not wait while holding the dispatcher lock.
                # Account for both the detached JSON and worker staging copy.
                source_lease=ExitStack()
                try:
                    source_lease.enter_context(self.pool.reserve(2*len(_json_bytes(input_request)),timeout=0))
                    admit=getattr(self._input_preparer,'admit',None)
                    if callable(admit):
                        input_admission=source_lease.enter_context(admit(input_request,pool=self.pool,disk=self.disk,
                            presentation=presentation,operation_id=operation))
                except BaseException:
                    source_lease.close();raise
            try:
                if not existing: self._intent(operation,signature,head,presentation,generation)
            except BaseException:
                if source_lease is not None: source_lease.close()
                raise
            state_path = (grammar_registry is not None or sampling is not None
                          or (resume and seed is not None))
            pause_supported = (kind == 'request' and state_path and
                all(getattr(worker['model'],'supports_consumed_pause',False)
                    for worker in self._workers.values()))
            handle = Request(operation,signature,pause_supported=pause_supported)
            handle.branch = branch
            self._requests[operation] = handle
            job = _Job(handle,branch,head,ids,max_steps,kind,principal,cancelled=cancelled,
                       presentation=presentation,sampling=sampling,resume=resume,
                       source_head=generation_source_head,generation_stream=generation_stream or operation,
                       presentation_source_operation=presentation_source_operation,
                       grammar_registry=grammar_registry,grammar_state=grammar_state,
                       source_grammar_engine=source_grammar_engine,
                        seed=seed,text_stops=tuple(text_stops or ()),source_lease=source_lease,
                        input_request=input_request,input_preparer_identity=input_preparer_identity,
                        input_admission=input_admission,input_artifacts=artifacts if typed_delta else None)
            if receipt is not None:
                try: self._recover(job,receipt)
                except BaseException as exc:
                    self._fail(job,exc);raise
                return handle
            if not any(w['health'] in ('starting','recovering','healthy')
                       for w in self._workers.values()):
                self._fail(job,InvalidOperation('Native worker unavailable'))
                return handle
            lease_head=generation_source_head or head
            if lease_head is not None:
                bound = self.store if principal is None else self.store.for_principal(principal)
                try:
                    if job.source_lease is None:job.source_lease=ExitStack()
                    job.source_lease.enter_context(bound.read_lease(lease_head.tip))
                except BaseException as exc:
                    self._fail(job,exc);raise
            self._set_prefill_key(job)
            self._pending.append(job)
            self._condition.notify_all()
            return handle

    def create(self,branch,ids,operation,*,presentation=None,cancelled=None):
        return self._submit(branch,ids,0,operation,'create',presentation=presentation,cancelled=cancelled)

    @staticmethod
    def _preparer_identity(preparer):
        identity=getattr(preparer,'identity',None)
        if (type(identity) is not str or len(identity)!=64
                or any(c not in '0123456789abcdef' for c in identity)):
            raise InvalidOperation('Input preparer identity must be a pinned SHA-256')
        return identity

    def prepare_root(self,branch,input_request,operation,*,presentation=None,cancelled=None):
        """Queue a bounded request for trusted preparation on the model worker.

        Only immutable mixed inputs enter canonical history. The raw request
        remains transient; its digest and preparer identity bind retries.
        """
        preparer=self._input_preparer
        if preparer is None: raise InvalidOperation('Input preparation is not configured')
        identity=self._preparer_identity(preparer)
        try:
            validated=preparer.validate(input_request)
            if type(validated) is not dict:
                raise InvalidOperation('Input preparation request must be a JSON object')
            raw=_json_bytes(validated)
            if len(raw)>16*1024*1024:
                raise InvalidOperation('Input preparation request exceeds 16 MiB')
            detached=json.loads(raw)
        except (TypeError,ValueError,OverflowError) as error:
            raise InvalidOperation('Input preparation request is not finite JSON') from error
        if self._preparer_identity(preparer)!=identity:
            raise InvalidOperation('Input preparer identity changed during validation')
        return self._submit(branch,(),0,operation,'create',presentation=presentation,
            cancelled=cancelled,input_request=detached,input_preparer_identity=identity)

    def submit(self,branch,ids,max_steps,operation,*,expected_revision=None,principal=None,cancelled=None,
               presentation=None,sampling=None,resume=False,grammar_registry=None,
               grammar_state=None,seed=None,text_stops=None,artifacts=None):
        """Cancellation is transient request control, never canonical controller state.

        It abandons uncommitted work at a consumed boundary. Once a complete
        snapshot is handed to durable publication, that publication finishes.
        Existing cancel(handle) continues to mean removal of queued work only.
        """
        return self._submit(branch,ids,max_steps,operation,'request',expected_revision,principal,cancelled,presentation,
                            sampling,resume,grammar_registry=grammar_registry,
                            grammar_state=grammar_state,seed=seed,text_stops=text_stops,artifacts=artifacts)

    @contextmanager
    def prepare_suffix(self,branch,revision,input_request,operation,max_steps,*,presentation,cancelled=None):
        """Borrow worker-prepared derived inputs, without evaluating a prefix.

        The caller submits the yielded typed delta through submit(), which binds
        its canonical identity and owns publication. This context pins source,
        originals and embeddings across that admission. Preparation itself
        creates no history, operation intent, branch move or durable save.
        """
        _name(branch);_name(operation)
        if type(max_steps) is not int or max_steps<1:raise InvalidOperation('Invalid output bound')
        preparer=self._input_preparer
        if preparer is None:raise InvalidOperation('Input preparation is not configured')
        identity=self._preparer_identity(preparer)
        validated=preparer.validate(input_request)
        if validated.get('schema')!=2:raise InvalidOperation('Suffix preparation requires a suffix request')
        raw=_json_bytes(validated)
        if len(raw)>16*1024*1024:raise InvalidOperation('Input preparation request exceeds 16 MiB')
        detached=json.loads(raw)
        presentation=detach_request(presentation)
        stopped=threading.Event()
        handle=Request(operation,hashlib.sha256(raw).hexdigest());handle.branch=branch
        held=ExitStack();job=None
        try:
            with self._condition:
                if not self._accepting:raise InvalidOperation('Coordinator is closing')
                head=self.store.head(branch)
                if type(revision) is not int or head.revision!=revision:
                    raise Conflict('Requested branch revision changed')
                if self.store.branch_namespace(branch)!=detached['namespace_id']:
                    raise InvalidOperation('Image suffix namespace differs from its branch')
                held.enter_context(self.store.read_lease(head.tip))
                if not isinstance(history_for_snapshot(self.store,head.tip),InputHistory):
                    raise InvalidOperation('Image suffix requires canonical mixed source history')
                held.enter_context(self.pool.reserve(2*len(raw),timeout=0))
                admission=held.enter_context(preparer.admit(detached,pool=self.pool,disk=self.disk,
                    presentation=presentation,operation_id=operation))
                job=_Job(handle,branch,head,(),max_steps,'prepare_inputs',source_lease=held,
                    cancelled=lambda:stopped.is_set() or (cancelled is not None and cancelled()),
                    presentation=presentation,input_request=detached,input_preparer_identity=identity,
                    input_admission=admission)
                self._input_preparations[id(handle)]=job
                self._set_prefill_key(job)
                self._pending.append(job);self._condition.notify_all()
            while True:
                try:
                    prepared=handle.snapshot.result(.05);break
                except TimeoutError:
                    if handle.snapshot.done():raise
                    if cancelled is not None and cancelled():
                        stopped.set();self.cancel(handle)
            if cancelled is not None and cancelled():raise CancelledError('Image preparation cancelled')
            yield prepared
        finally:
            stopped.set()
            if job is not None:
                self.cancel(handle)
                # Do not release buffers while a running encoder still uses them.
                try:handle.durable.result()
                except BaseException:pass
                with self._condition:
                    self._release_source(job)
                    self._input_preparations.pop(id(handle),None)
            else:held.close()

    def reseed(self,branch,seed,operation,*,expected_revision=None,principal=None,presentation=None,
               grammar_registry=None):
        """Publish a controller-only RNG stream transition under branch CAS."""
        from .seed_controls import seed_stream
        seed_stream(seed)
        return self._submit(branch,(),0,operation,'reseed',expected_revision,principal,
                            presentation=presentation,seed=seed,grammar_registry=grammar_registry)

    def cancel(self,handle):
        with self._condition:
            for job in self._pending:
                if job.request is handle:
                    self._pending.remove(job)
                    handle.state = 'cancelled'
                    handle.snapshot.cancel();handle.durable.cancel()
                    self._release_source(job)
                    self._cancelled += 1
                    self._condition.notify_all()
                    return True
            return False

    @staticmethod
    def _is_ancestor(store,snapshot,descendant):
        cursor=descendant;visited=set()
        while cursor is not None and cursor not in visited:
            if cursor==snapshot: return True
            visited.add(cursor);cursor=store.get_snapshot(cursor)['parent']
        return False

    def _validate_optional_checkpoint(self,snapshot):
        """Check a present numerical copy; canonical references outlive eviction.

        Absence permits a history-only branch action, never an exact restore.
        A present incomplete, redirected or corrupt copy must still reject.
        """
        record=self.store.get_snapshot(snapshot)
        path=self.disk.directory/(snapshot+'.json')
        if not os.path.lexists(path):
            return None
        if path.is_symlink() or not path.is_file():
            raise IntegrityError('Checkpoint manifest is not a regular local file')
        with self._condition:
            models=[worker['model'] for worker in self._workers.values() if worker['model'] is not None]
        for model in models:
            if model.profile.get('model_contract_id')!=record['model_contract_id']: continue
            validate=getattr(model,'validate_checkpoint',None)
            if not callable(validate): continue
            try:
                manifest=validate(snapshot)
            except (OSError,KeyError,TypeError,ValueError) as exc:
                raise IntegrityError('Checkpoint validation failed') from exc
            if not isinstance(manifest,dict) or manifest.get('snapshot')!=snapshot:
                raise IntegrityError('Checkpoint validator returned a different snapshot')
            return manifest
        raise InvalidOperation('No CPU checkpoint validator is available for this model contract')

    def _namespace_savedpoint(self,namespace_id,snapshot):
        catalog=self.store.presentation_catalog(namespace_id)
        return any(isinstance(event.get('result'),dict) and event['result'].get('tip')==snapshot
                   for event in catalog['events'])

    def fork(self,name,snapshot,*,principal=None,source_branch=None,operation_id=None):
        _name(name)
        if source_branch is None:
            if operation_id is not None:
                raise InvalidOperation('A namespace fork operation requires its source branch')
            with self._condition:
                bound=self.store if principal is None else self.store.for_principal(principal)
                bound.authorize(snapshot,'fork')
                with bound.read_lease(snapshot):
                    # The model validator has a separate Store handle. Never
                    # call it while holding this Store's exclusive refs lock.
                    self._validate_optional_checkpoint(snapshot)
                    with self.store._locked():
                        point='fork-'+base64.urlsafe_b64encode(bytes.fromhex(snapshot)).decode().rstrip('=')
                        self.store.retain(point,snapshot)
                        return bound.fork(name,snapshot)

        _name(source_branch)
        if operation_id is None:
            raise InvalidOperation('A namespace fork requires a stable operation ID')
        _name(operation_id)
        with self._condition:
            bound=self.store if principal is None else self.store.for_principal(principal)
            bound.authorize(snapshot,'fork')
            namespace_id=bound.branch_namespace(source_branch)
            if namespace_id is None:
                raise InvalidOperation('Namespace fork source branch is not presentation-bound')
            catalog=self.store.presentation_catalog(namespace_id)
            prior=next((event for event in catalog['events']
                        if event.get('operation_id')==operation_id),None)
            if prior is not None:
                captured_source=prior.get('event',{}).get('source')
                if (prior.get('branch')!=name or prior.get('result',{}).get('tip')!=snapshot
                        or prior.get('event',{}).get('type')!='fork'
                        or not isinstance(captured_source,dict)
                        or captured_source.get('branch')!=source_branch):
                    raise InvalidOperation('Operation ID reused for a different fork request')
                return bound.fork(name,snapshot,namespace_id=namespace_id,operation_id=operation_id,
                                  source_branch=captured_source['branch'],
                                  source_revision=captured_source['revision'],
                                  source_tip=captured_source['tip'])
            try:
                bound.head(name)
            except NotFound:
                pass
            else:
                raise Conflict(f'Branch exists: {name}')
            source_head=bound.head(source_branch)
            with ExitStack() as leases:
                for identity in dict.fromkeys((source_head.tip,snapshot)):
                    leases.enter_context(bound.read_lease(identity))
                current=bound.head(source_branch)
                if current.revision!=source_head.revision or current.tip!=source_head.tip:
                    raise Conflict('Source branch moved while acquiring its snapshot lease')
                same_root=self.store._root_for(source_head.tip)==self.store._root_for(snapshot)
                valid=(same_root and self._is_ancestor(bound,snapshot,source_head.tip))
                if not valid:
                    valid=self._namespace_savedpoint(namespace_id,snapshot)
                if not valid:
                    raise InvalidOperation('Fork target is not a source ancestor or published namespace savedpoint')
                self._validate_optional_checkpoint(snapshot)
                point='fork-'+base64.urlsafe_b64encode(bytes.fromhex(snapshot)).decode().rstrip('=')
                self.store.retain(point,snapshot)
                return bound.fork(name,snapshot,namespace_id=namespace_id,operation_id=operation_id,
                                  source_branch=source_branch,source_revision=source_head.revision,
                                  source_tip=source_head.tip)

    def _namespace_renderer(self,namespace_id,root_snapshot):
        catalog=self.store.presentation_catalog(namespace_id)
        for event in catalog['events']:
            descriptor=event.get('presentation');result=event.get('result')
            if (isinstance(descriptor,dict) and descriptor.get('kind')=='root'
                    and isinstance(result,dict) and result.get('tip')==root_snapshot):
                renderer=descriptor.get('renderer_id')
                if isinstance(renderer,str) and len(renderer)==64:
                    return renderer
        raise InvalidOperation('Namespace has no published root renderer identity for this history')

    def rewind_presented(self,branch,revision,snapshot,operation):
        """Publish a typed namespace rewind from canonical history receipts."""
        _name(branch);_name(operation)
        if type(revision) is not int or revision<0:
            raise Conflict('Requested branch revision changed')
        with self._condition:
            if branch in self._branches:
                raise Conflict('Branch has a pending request/publication')
            try: receipt=self.store.presentation_receipt(operation)
            except NotFound: receipt=None
            if receipt is not None:
                descriptor=receipt['presentation'];result=receipt['result']
                source=descriptor.get('source',{})
                if (result.get('name')!=branch or result.get('tip')!=snapshot
                        or descriptor.get('kind')!='rewind' or source.get('revision')!=revision):
                    raise Conflict('Operation ID is bound to another typed rewind')
                self._validate_optional_checkpoint(snapshot)
                return self.store.move(branch,revision,snapshot,operation,presentation=descriptor)

            head=self.store.head(branch)
            if head.revision!=revision:
                raise Conflict('Requested branch revision changed')
            namespace_id=self.store.branch_namespace(branch)
            if namespace_id is None:
                raise InvalidOperation('Typed rewind requires a presentation-bound branch')
            with ExitStack() as leases:
                for identity in dict.fromkeys((head.tip,snapshot)):
                    leases.enter_context(self.store.read_lease(identity))
                current=self.store.head(branch)
                if current.revision!=head.revision or current.tip!=head.tip:
                    raise Conflict('Source branch moved while acquiring its snapshot lease')
                if not self._is_ancestor(self.store,snapshot,head.tip):
                    raise InvalidOperation('Typed rewind target must be an ancestor of the current branch tip')
                if snapshot==head.tip:
                    raise InvalidOperation('Typed rewind target must differ from the current branch tip')
                root=self.store._root_for(head.tip)
                if self.store._root_for(snapshot)!=root:
                    raise InvalidOperation('Typed rewind target belongs to another history root')
                self._validate_optional_checkpoint(snapshot)
                renderer_id=self._namespace_renderer(namespace_id,root)
                source_history=history_for_snapshot(self.store,head.tip)
                source_count=len(source_history)
                complete=self.store.get_snapshot(snapshot)['controller'].get('status')=='stopped'
                descriptor=dict(schema=3 if isinstance(source_history,InputHistory) else 1,
                    namespace_id=namespace_id,kind='rewind',messages=[],
                    source=dict(snapshot=head.tip,revision=head.revision,token_count=source_count),
                    input_span=None,output_span=None,renderer_id=renderer_id,turn_complete=complete)
                return self.store.move(branch,revision,snapshot,operation,presentation=descriptor)

    def rewind(self,branch,revision,snapshot,operation,*,principal=None):
        with self._condition:
            if branch in self._branches: raise Conflict('Branch has a pending request/publication')
            bound = self.store if principal is None else self.store.for_principal(principal)
            bound.authorize(snapshot,'move')
            with bound.read_lease(snapshot):
                self._validate_optional_checkpoint(snapshot)
                return bound.move(branch,revision,snapshot,operation)

    def adopt_evaluated_once(self, operation_id, branch, expected_revision,
                             evaluation_ids, mandatory, *,
                             presentation_source_operation=None):
        """Serialize evaluation adoption with this branch's live publications."""
        with self._condition:
            try:
                status = self.store.evaluation_operation_status(operation_id)
            except NotFound:
                status = None
            # A committed replay only returns its durable receipt; it does not
            # move the branch again, so it remains safe during active work. A
            # pending evaluation receipt may also follow an already-committed
            # Store.move if the process stopped between those two durable writes.
            # Permit that exact replay; Store.adopt_evaluated_once verifies the
            # full evaluation request and returns the existing move receipt.
            replay_move = self._pending_adoption_move_exists(
                operation_id, branch, expected_revision, evaluation_ids, mandatory,
                presentation_source_operation)
            if (status is None or status.get('state') != 'committed') and not replay_move:
                if branch in self._branches:
                    raise Conflict('Branch has a pending request/publication')
                head = self.store.head(branch)
                if head.revision != expected_revision:
                    raise Conflict('Requested branch revision changed')
            return self.store.adopt_evaluated_once(
                operation_id, branch, expected_revision, evaluation_ids, mandatory,
                presentation_source_operation=presentation_source_operation)

    def _pending_adoption_move_exists(self, operation_id, branch, expected_revision,
                                      evaluation_ids, mandatory,
                                      presentation_source_operation=None):
        """Check for the exact durable move preceding an adoption receipt."""
        if (type(expected_revision) is not int or expected_revision < 0
                or not isinstance(evaluation_ids, list) or not isinstance(mandatory, list)):
            return False
        try:
            operation_id = _name(operation_id)
            branch = _name(branch)
            identities = [_id(value) for value in evaluation_ids]
        except InvalidOperation:
            return False
        request = dict(kind='adoption', branch=branch,
            expected_revision=expected_revision, evaluation_ids=identities,
            mandatory=mandatory)
        if presentation_source_operation is not None:
            try: presentation_source_operation = _name(presentation_source_operation)
            except InvalidOperation: return False
            request['presentation_source_operation'] = presentation_source_operation
        request_digest = hashlib.sha256(_json_bytes(request)).hexdigest()
        with self.store._locked():
            operation = self.store._read_evaluation_operation(operation_id)
            if (operation is None or operation.get('kind') != 'adoption'
                    or operation.get('state') != 'pending'
                    or operation.get('request_digest') != request_digest
                    or operation.get('branch') != branch
                    or operation.get('expected_revision') != expected_revision
                    or not isinstance(operation.get('selected_snapshot_id'), str)):
                return False
            target = operation['selected_snapshot_id']
            receipt = self.store._read_refs()['receipts'].get(operation_id)
            if receipt is None:
                return False
            move_digest = hashlib.sha256(_json_bytes(dict(
                type='move', name=branch, expected_revision=expected_revision, target=target
            ))).hexdigest()
            source_operation = operation.get('adoption_presentation_source_operation')
            if source_operation is not None:
                move_digest = hashlib.sha256(_json_bytes(dict(
                    type='move', name=branch, expected_revision=expected_revision,
                    target=target,
                    adoption_presentation_source_operation=source_operation
                ))).hexdigest()
            result = receipt.get('result')
            return (receipt.get('digest') == move_digest and isinstance(result, dict)
                    and result.get('name') == branch
                    and result.get('revision') == expected_revision + 1
                    and result.get('tip') == target)

    def _headroom(self):
        if not self.config.min_available_bytes: return
        path = Path('/proc/meminfo')
        if not path.is_file(): raise InvalidOperation('Host available-memory witness is missing')
        rows = dict(line.split(':',1) for line in path.read_text().splitlines())
        if int(rows['MemAvailable'].split()[0])*1024 < self.config.min_available_bytes:
            raise PoolFull('Host memory headroom is below the configured minimum')

    @staticmethod
    def _release_source(job):
        try:
            if job.source_lease is not None:
                job.source_lease.__exit__(None,None,None)
        finally:
            job.source_lease = None
            job.input_request = None
            job.input_admission = None
            job.input_artifacts = None

    def _compatible(self,model,job):
        """CPU-only inspection of a retained image's backend layout.

        A matching model/profile is necessary but a grouped worker may also
        require a particular self-described native placement. No native lease
        or replay is acquired by this routing decision.
        """
        if job.kind=='prepare_inputs':
            return model.profile.get('execution',{}).get('resume_contract')=='identical-mixed-native-operators-v1'
        source_head=job.source_head or job.head
        if source_head is None: return True
        accepts = getattr(model.backend,'checkpoint_compatible',None)
        snapshot=source_head.tip
        manifest_path=model.checkpoint_directory/(snapshot+'.json')
        if not os.path.lexists(manifest_path):
            # A checkpoint that still has a RAM/native materialization can be
            # used directly. If every derived copy is absent, only a worker
            # with the explicit canonical-history rebuild capability may be
            # selected. A present path (including a dangling symlink) always
            # takes the strict manifest-validation path below.
            key=self._key(model,snapshot)
            cached=model.cache.get(snapshot)
            if (cached is not None and isinstance(getattr(cached,'blob',None),StrataImage)
                    and cached.model_contract_id==model.profile['model_contract_id']
                    and cached.execution_profile_id==model.profile['execution_profile_id']):
                return True
            try:
                pooled=self.pool.get(key)
            except KeyError:
                pooled=None
            if pooled is not None:
                return (pooled.model_contract_id==model.profile['model_contract_id']
                        and pooled.execution_profile_id==model.profile['execution_profile_id'])
            rebuild=getattr(model,'rebuild_materialization',None)
            if not callable(rebuild): return False
            record=self.store.get_snapshot(snapshot)
            return record.get('model_contract_id')==model.profile['model_contract_id']
        if accepts is None: return True
        manifest = model.checkpoint_manifest(snapshot)
        accepted = accepts(manifest,model.checkpoint_directory)
        if type(accepted) is not bool: raise InvalidOperation('Backend compatibility must return a boolean')
        return accepted

    def _set_prefill_key(self,job):
        if self.config.scheduling == 'fifo':return
        head=job.source_head or job.head
        if head is not None:
            value=('snapshot',head.tip)
        elif job.input_request is not None:
            value=('prepared-root',job.input_preparer_identity,job.input_request)
        else:
            # InputHistory retains image/position identities; tokens alone
            # must never stand in for a mixed-input prefill.
            value=('root',job.ids.spans if isinstance(job.ids,InputHistory) else list(job.ids))
        job.prefill_key=hashlib.sha256(_json_bytes(value)).hexdigest()

    def _release_prefill(self,job):
        with self._condition:
            key=job.prefill_claim
            if key is not None and self._prefills.get(key)==job.request.operation:
                del self._prefills[key]
            job.prefill_claim=None
            self._condition.notify_all()

    def _needs_prefill_claim(self,job,model):
        if job.kind=='create':return True
        if job.kind!='request':return False
        source=(job.source_head or job.head).tip
        return (self._key(model,source) not in self.pool
            and not os.path.lexists(model.checkpoint_directory/(source+'.json')))

    def _placement(self,eligible,job):
        """Estimate native restore plus bounded forward work, after compatibility."""
        head=job.source_head or job.head
        warm=[]; size=0; work=len(job.ids)+job.max_steps
        if head is not None:
            for gpu in eligible:
                cp=self._workers[gpu]['model'].cache.get(head.tip)
                if cp is not None and isinstance(cp.blob,StrataImage):warm.append(gpu)
            model=self._workers[eligible[0]]['model']
            size=self.pool.checkpoint_extent(self._key(model,head.tip))
            if size is None:
                path=model.checkpoint_directory/(head.tip+'.json')
                if os.path.lexists(path):
                    manifest=model.checkpoint_manifest(head.tip)
                    size=manifest['state_bytes']+4*manifest['logits_count']
                else:
                    size=0
                    work+=len(model.history(head.tip))
        choice=self._costs.choose(eligible,warm,size,work)
        self._last_placement=dict(worker=worker_key(choice.worker),reason=choice.reason,
            estimated_seconds=choice.estimated_seconds,checkpoint_bytes=size,work_bound=work)
        return choice.worker

    def _observe_worker_restores(self,model,gpu):
        if self._costs is None:return
        w=self._workers[gpu]
        w['restore_seconds']=0.
        def observe(nbytes,seconds):
            self._costs.observe_restore(gpu,nbytes,seconds)
            w['restore_seconds']+=seconds
        model._restore_observer=observe

    @staticmethod
    def _backend_capabilities(model):
        from .capabilities import NativeCapabilities
        describe=getattr(model.backend,'capabilities',None)
        if describe is None:return None
        value=describe()
        if (type(value) is not NativeCapabilities
                or value.model_contract_id!=model.profile['model_contract_id']
                or value.execution_profile_id!=model.profile['execution_profile_id']):
            raise InvalidOperation('Native capability identity differs from worker profile')
        return value

    def capabilities(self):
        """Observe already validated worker facts; no native leases or model work."""
        from .capabilities import feature
        with self._condition:
            workers=[dict(worker=str(worker_key(gpu)),health=w['health'],
                model_contract_id=w['model_contract_id'],execution_profile_id=w['execution_profile_id'],
                native=w['capabilities'].to_json() if w.get('capabilities') is not None else None)
                for gpu,w in self._workers.items()]
        native=[w['native'] for w in workers]
        def every(key):return bool(native) and all(n is not None and n[key] for n in native)
        features=dict(branches=feature(True),controllers=feature(True),
            checkpoint_ram=feature(True),checkpoint_disk=feature(True),
            native_checkpoints=feature(every('checkpoint_export') and every('checkpoint_import')),
            native_residency=feature(every('native_residency')),
            physical_prefix_sharing=feature(every('physical_prefix_sharing')),
            independent_workers=feature(True),tensor_parallel_restore=feature(False),
            pause_resume=feature(every('selected_token_protocol')),
            sampling=feature(every('sampling_protocol')),
            images=feature(bool(native) and all(n is not None and 'image' in n['inputs'] for n in native),
                           self._input_preparer is not None),
            cache_inventory=feature(True,self.config.cache_inventory,qualification='experimental'),
            cache_inventory_pages=feature(True,self.config.cache_inventory,qualification='experimental'),
            cache_notifications=feature(True,self.config.cache_inventory,qualification='experimental'),
            transfer_aware_scheduling=feature(True,self.config.scheduling=='transfer-aware',qualification='experimental'),
            prefill_coalescing=feature(True,self.config.scheduling!='fifo'),
            peer_transfer=feature(False),prefill_decode_placement=feature(False),
            tokenizer_fastokens=feature(False))
        return dict(workers=workers,features=features,limits=dict(
            ram_cache_bytes=self.config.ram_bytes,disk_cache_bytes=self.config.disk_bytes,
            minimum_available_ram_bytes=self.config.min_available_bytes),
            scheduling=self.config.scheduling)

    @staticmethod
    def _native_inventory(model,gpu):
        """Called only by the owning worker after native execution has settled."""
        return [dict(model=cp.model_contract_id,execution=cp.execution_profile_id,
                     snapshot=snapshot,tier='native',worker=str(worker_key(gpu)),
                     checksum=cp.checksum,bytes=len(cp.blob)+4*len(cp.frontier))
                for snapshot in tuple(model.cache) for cp in (model.cache.get(snapshot),)
                if cp is not None and isinstance(cp.blob,StrataImage)]

    def cache_inventory(self,**cursor):
        """Observe pool changes; rescan at startup, overrun or every 30 seconds.

        This is a hint, not a pin, checkpoint validation or a history export.
        Busy workers are omitted rather than inspected from another thread.
        """
        if self._inventory is None:raise InvalidOperation('Cache inventory is not enabled')
        with self._inventory_lock:
            if cursor.get('cursor') is not None:
                return self._inventory.read(**cursor)
            changes=self._pool_notifications.read(self._inventory_pool_seq)
            native_rows=[]
            with self._condition:
                models={}
                for w in self._workers.values():
                    model=w['model']
                    if w['health']=='healthy' and model is not None:
                        models[(model.profile['model_contract_id'],model.profile['execution_profile_id'])]=model
                        if not w['busy']:native_rows.extend(w['native_inventory'])
            now=time.monotonic()
            force=(changes['reset'] or tuple(sorted(models))!=self._inventory_models
                   or now-self._inventory_scan_at>=30)
            ram=(self.pool.inventory() if force or any(tier=='ram' for tier,_ in changes['events'])
                 else self._inventory_ram)
            disk={} if force else dict(self._inventory_disk)
            affected={key for tier,key in changes['events'] if tier=='disk'}
            for (contract,execution),model in models.items():
                paths=(model.checkpoint_directory.glob('*.json') if force else
                       (model.checkpoint_directory/(snapshot+'.json') for snapshot in sorted(affected)))
                for count,path in enumerate(paths,1):
                    if count>self._inventory.entry_limit:raise InvalidOperation('Cache inventory capacity exceeded')
                    key=(contract,execution,path.stem); disk.pop(key,None)
                    try:
                        manifest=model.checkpoint_manifest(path.stem)
                        size=manifest['state_bytes']+4*manifest['logits_count']
                        payload=model.checkpoint_directory/(manifest['payload_sha256']+'.state')
                        if payload.stat().st_size!=size:continue
                        disk[key]=dict(model=contract,execution=execution,snapshot=path.stem,
                            tier='disk',worker='pool',checksum=manifest['payload_sha256'],bytes=size)
                    except (OSError,ValueError,IntegrityError):continue
                    if len(disk)>self._inventory.entry_limit:
                        raise InvalidOperation('Cache inventory capacity exceeded')
            rows=ram+native_rows+list(disk.values())
            self._inventory.reconcile(rows)
            self._inventory_ram,self._inventory_disk=ram,disk
            self._inventory_models=tuple(sorted(models)); self._inventory_pool_seq=changes['seq']
            if force:self._inventory_scan_at=now
            return self._inventory.read(**cursor)

    def _dispatch(self):
        while True:
            lookahead = None
            with self._condition:
                if self._stopping: return
                idle = [gpu for gpu,w in self._workers.items() if self._qualified
                    if w['health']=='healthy' and w['ready'].done()
                    and w['ready'].exception() is None and not w['busy']]
                recovering = any(w['health'] in ('starting','recovering')
                                 for w in self._workers.values())
                if self._qualified and not recovering and not any(
                        w['health']=='healthy' for w in self._workers.values()):
                    for job in tuple(self._pending):
                        self._pending.remove(job)
                        self._fail(job,InvalidOperation('Native worker unavailable'))
                ordered=(order_jobs(self._pending,self._active_groups,now=time.monotonic())
                    if self.config.scheduling!='fifo' else tuple(self._pending))
                for job in ordered:
                    if (self.config.scheduling!='fifo' and job.cancelled is not None
                            and job.cancelled()):
                        self._pending.remove(job)
                        self._fail(job,CancelledError('Queued request cancelled'),release_branch=False)
                        continue
                    if not idle:
                        if self.config.scheduling!='fifo':continue
                        break
                    if job.branch in self._branches: continue
                    if job.prefill_key in self._prefills:continue
                    try:
                        compatible = [gpu for gpu,w in self._workers.items()
                            if w['health']=='healthy' and w['model'] is not None
                            and self._compatible(w['model'],job)]
                    except Exception as error:
                        self._pending.remove(job);self._fail(job,error);continue
                    if not compatible:
                        self._pending.remove(job)
                        self._fail(job,InvalidOperation('No compatible worker for retained checkpoint layout'))
                        continue
                    eligible = [gpu for gpu in idle if gpu in compatible]
                    if not eligible: continue
                    warm = [gpu for gpu in eligible if job.head and job.head.tip in self._workers[gpu]['model'].cache]
                    try:
                        gpu = (self._placement(eligible,job) if self._costs is not None else
                            select_worker(eligible,warm,self._service_rate)
                            if self.config.scheduling!='fifo' else (warm or eligible)[0])
                    except Exception as error:
                        self._pending.remove(job);self._fail(job,error);continue
                    idle.remove(gpu);self._pending.remove(job)
                    w = self._workers[gpu];w['busy'] = True
                    if self.config.scheduling!='fifo':
                        self._active_groups[job.prefill_key]=self._active_groups.get(job.prefill_key,0)+1
                        if self._needs_prefill_claim(job,w['model']):
                            self._prefills[job.prefill_key]=job.request.operation
                            job.prefill_claim=job.prefill_key
                    self._branches.add(job.branch)
                    job.request.state = 'loading'
                    w['queue'].put(job)
                # One load ahead, only for real queued/eligible work. It never
                # activates GPU state and does not give a later job queue priority.
                for job in self._pending:
                    if job.kind=='prepare_inputs':continue
                    if job.head and job.branch not in self._branches:
                        try:
                            w = next((w for w in self._workers.values() if w['health']=='healthy'
                                and w['model'] is not None
                                and self._compatible(w['model'],job)),None)
                        except Exception as error:
                            self._pending.remove(job);self._fail(job,error);break
                        if w is not None: lookahead = (w['model'],job.head.tip)
                        break
                self._condition.wait(.05)
            if lookahead is not None:
                try: self._preload(*lookahead)
                except (PoolFull,KeyError,FileNotFoundError): pass
                except Exception:
                    # Worker activation validates again and surfaces the failure
                    # to the request; a failed speculative read is never a pass.
                    self._preload_failures += 1

    def _preload(self,model,snapshot,*,timeout=0,extra_bytes=0):
        key = self._key(model,snapshot)
        if key in self.pool: return None
        manifest = model.checkpoint_manifest(snapshot)
        size = manifest['state_bytes']+4*manifest['logits_count']
        if type(extra_bytes) is not int or extra_bytes < 0:
            raise ValueError('Invalid disk-read staging extent')
        self._headroom()
        def load():
            with self.store.read_lease(snapshot):
                checkpoint = model._load_checkpoint(snapshot,model.history(snapshot))
                self.disk.touch(snapshot)
                return checkpoint
        # Keep the normal double-buffer allowance, plus any caller-specific
        # staging, reserved before the loader allocates the checkpoint object.
        future = self.pool.preload(key,2*size+extra_bytes,load,timeout=timeout,trace_disk_read=True)
        self._preloads += 1
        return future

    def _root_materialization(self,model,job,snapshot,history):
        """Borrow a root checkpoint through RAM admission and native restore."""
        if job.source_lease is None:
            job.source_lease = ExitStack()
        if not isinstance(job.source_lease,ExitStack):
            raise InvalidOperation('Root materialization already has a different lease')
        job.source_lease.enter_context(model.store.read_lease(snapshot))
        # A native image is already resident in its owning worker and needs no
        # host-pool staging. Keep it hot; the caller still validates its full
        # checkpoint/profile/input identity before accepting reuse.
        saved = model.cache.get(snapshot)
        if saved is not None and isinstance(getattr(saved,'blob',None),StrataImage):
            return saved
        key = self._key(model,snapshot)
        manifest = None
        deadline = time.monotonic() + self._admission_timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise PoolFull('Timed out pinning root checkpoint materialization')
            try:
                if key not in self.pool:
                    if manifest is None:
                        manifest_path = model.checkpoint_directory/(snapshot+'.json')
                        try:
                            info = manifest_path.lstat()
                        except FileNotFoundError:
                            return None
                        if not stat.S_ISREG(info.st_mode):
                            raise IntegrityError('Root checkpoint manifest is not a regular file')
                        manifest = model.checkpoint_manifest(snapshot,history)
                    # _load_checkpoint temporarily holds the encoded payload
                    # and copied blob while the pool owns the final immutable
                    # checkpoint. Reserve both plus later restore scratch.
                    extra = manifest['state_bytes'] + 12*len(history)
                    future = self._preload(model,snapshot,timeout=remaining,
                                           extra_bytes=extra)
                    if future is not None:
                        future.result(remaining)
            except PoolFull:
                continue
            try:
                cold = job.source_lease.enter_context(self.pool.pin(key))
                break
            except KeyError:
                # A read-ahead or concurrent admission may have evicted the
                # unpinned entry after preload completed; retry within the
                # same admission deadline rather than failing or bypassing RAM.
                continue
        job.source_lease.callback(self._drop_root_pool_alias,model,snapshot,cold)
        # Preload's reservation ends at RAM admission. Hold a separate extent
        # through native activation/capture, including an already pooled hit.
        # The source pin and this reservation share the worker's ExitStack.
        job.source_lease.enter_context(self.pool.reserve(
            3*checkpoint_bytes(cold)+12*cold.consumed,
            timeout=max(0,deadline-time.monotonic())))
        return cold

    @staticmethod
    def _drop_root_pool_alias(model,snapshot,cold):
        if model.cache.get(snapshot) is cold:
            model.cache.pop(snapshot,None)

    def _rebuild_missing_source(self,model,job,source,before):
        rebuild=getattr(model,'rebuild_materialization',None)
        if not callable(rebuild):
            raise InvalidOperation('Canonical source has no materialization and cannot be reconstructed')
        history=model.history(source)
        rebuild_options={}
        if job.source_grammar_engine is not None:
            rebuild_options['grammar_engine']=job.source_grammar_engine
        job.request.emit('source_recomputation_started',source_snapshot=source,
                         history_tokens=len(history))
        try:
            input_preparation=None
            if isinstance(history,InputHistory):
                prepare=getattr(self._input_preparer,'reconstruction',None)
                if callable(prepare):
                    with prepare(model,history,pool=self.pool,disk=self.disk,cancelled=job.cancelled) as inputs:
                        reconstruction=rebuild(source,artifacts=inputs.artifacts,reencode=inputs.reencode,cancelled=job.cancelled,**rebuild_options)
                        input_preparation=json.loads(_json_bytes(inputs.receipt))
                else:
                    from .vision_artifacts import RetainedImageArtifacts
                    artifacts=RetainedImageArtifacts(self.store,self.pool,self.disk,
                        expected_width=model.profile['execution']['embedding_width'],
                        max_rows=model.profile['execution']['capacity'])
                    reconstruction=rebuild(source,artifacts=artifacts,cancelled=job.cancelled,**rebuild_options)
                    input_preparation=dict(encoder_calls=0,encodings=[],producer_available=False)
            else:reconstruction=rebuild(source,**rebuild_options)
            job.reconstructed_source=source
            if not isinstance(reconstruction,dict):
                raise IntegrityError('History reconstruction returned no evidence')
            required=dict(snapshot=source,source_kind='recomputed-history',durable=False,
                prior_computation_preserved=False,consumed=len(history),
                forward_tokens=len(history),model_contract_id=model.profile['model_contract_id'],
                execution_profile_id=model.profile['execution_profile_id'])
            if (any(reconstruction.get(name)!=value for name,value in required.items())
                    or type(reconstruction.get('durable')) is not bool
                    or reconstruction.get('durable') is not False
                    or type(reconstruction.get('prior_computation_preserved')) is not bool
                    or reconstruction.get('prior_computation_preserved') is not False
                    or type(reconstruction.get('consumed')) is not int
                    or type(reconstruction.get('forward_tokens')) is not int):
                raise IntegrityError('History reconstruction evidence differs from canonical source')
            rebuilt=model.cache.get(source)
            if (rebuilt is None or rebuilt.model_contract_id!=model.profile['model_contract_id']
                    or rebuilt.execution_profile_id!=model.profile['execution_profile_id']
                    or rebuilt.consumed!=len(history)):
                raise IntegrityError('History reconstruction did not retain the expected native image')
            after_rebuild=model.backend.work_counters()
            if (type(after_rebuild.get('forward_tokens')) is not int
                    or type(before.get('forward_tokens')) is not int
                    or type(after_rebuild.get('consumed_tokens')) is not int
                    or type(before.get('consumed_tokens')) is not int):
                raise IntegrityError('History reconstruction lacks exact forward-work counters')
            actual_prefix=after_rebuild['forward_tokens']-before['forward_tokens']
            actual_consumed=after_rebuild['consumed_tokens']-before['consumed_tokens']
            if actual_prefix!=len(history) or actual_consumed!=len(history):
                raise IntegrityError('History reconstruction work differs from canonical history')
            if model.store.head(job.branch)!=job.head:
                raise Conflict('Queued branch revision changed during source reconstruction')
            if job.cancelled is not None and job.cancelled():
                raise CancelledError('Request cancelled after source reconstruction')
            result=dict(source_kind='recomputed-history',durable=False,
                prior_computation_preserved=False,prefix_history_tokens=len(history),
                prefix_forward_tokens=actual_prefix,prefix_consumed_tokens=actual_consumed,
                schedule_sha256=reconstruction.get('schedule_sha256'),
                controller_sha256=reconstruction.get('controller_sha256'),
                frontier_sha256=reconstruction.get('frontier_sha256'),
                seconds=reconstruction.get('seconds'))
            if isinstance(history,InputHistory):
                result.update(input_kind='canonical-mixed-spans-v1',consumed_cells=len(history),
                    text_tokens=len(history.text_tokens),image_rows=len(history)-len(history.text_tokens),
                    image_artifacts=reconstruction['image_artifacts'],
                    reencode_callback_calls=reconstruction['reencode_callback_calls'],input_preparation=input_preparation)
            job.request.emit('source_recomputed',source_snapshot=source,
                prefix_history_tokens=len(history),prefix_forward_tokens=actual_prefix,
                prefix_consumed_tokens=actual_consumed,schedule_sha256=result['schedule_sha256'])
            if job.prefill_claim is not None:
                self._share_rebuilt_source(model,job,source)
            return result,after_rebuild
        except BaseException as error:
            job.request.emit('source_recomputation_failed',source_snapshot=source,error=str(error))
            raise

    def _share_rebuilt_source(self,model,job,source):
        """Publish derived source bytes once before releasing waiting slots.

        No branch/controller move is performed. Source read leases and all
        capture, writer and transfer extents use the existing pool accounting.
        """
        saved=model.cache[source]
        estimate=3*len(saved.blob)+8*len(saved.frontier)
        self._headroom()
        with self.pool.reserve(estimate,timeout=self._admission_timeout) as reservation:
            prepared=model.prepare_checkpoint(source)
            key=self._key(model,source)
            reservation.admit(key,prepared.checkpoint,durable=False)
        with self.pool.pin(key):
            size=checkpoint_bytes(prepared.checkpoint)
            scratch=2*min(8*1024*1024,size)+4*len(prepared.checkpoint.frontier)
            try:
                with self.pool.reserve(scratch,timeout=self._admission_timeout):
                    with self.disk.reserve(len(prepared.envelope),timeout=self._admission_timeout,
                            payload_sha256=prepared.checkpoint.checksum) as disk_reservation:
                        if not (self.disk.directory/(prepared.checkpoint.checksum+'.state')).exists():
                            disk_reservation.grow(size,timeout=self._admission_timeout)
                        model.publish_prepared_checkpoint(prepared)
                        self.disk.touch(source)
                self.pool.mark_durable(key)
            except BaseException:
                self.pool.mark_failed(key)
                raise
        job.request.emit('source_checkpoint_shared',source_snapshot=source,
            checkpoint_bytes=size)
        self._release_prefill(job)

    def _run(self,model,job):
        if job.cancelled is not None and job.cancelled():
            raise CancelledError('Request cancelled before execution')
        if job.head is None:
            started=time.monotonic()
            prepared_inputs=None
            preparation_receipt=None
            if job.input_request is not None:
                preparer=self._input_preparer
                if self._preparer_identity(preparer)!=job.input_preparer_identity:
                    raise InvalidOperation('Input preparer identity changed before execution')
                if job.source_lease is None: job.source_lease=ExitStack()
                admission_options=({'admission':job.input_admission} if job.input_admission is not None else {})
                prepared_inputs=job.source_lease.enter_context(preparer.prepare(model,job.input_request,
                    pool=self.pool,disk=self.disk,cancelled=job.cancelled,**admission_options))
                if self._preparer_identity(preparer)!=job.input_preparer_identity:
                    raise InvalidOperation('Input preparer identity changed during execution')
                if not isinstance(getattr(prepared_inputs,'history',None),InputHistory):
                    raise InvalidOperation('Input preparation must produce canonical InputHistory')
                receipt_raw=_json_bytes(getattr(prepared_inputs,'receipt',None))
                if len(receipt_raw)>1024*1024:
                    raise InvalidOperation('Input preparation receipt exceeds 1 MiB')
                preparation_receipt=json.loads(receipt_raw)
                job.ids=prepared_inputs.history
                if job.cancelled is not None and job.cancelled():
                    raise CancelledError('Request cancelled after input preparation')
            before=model.backend.work_counters()
            create_with_reuse = getattr(model, 'create_with_reuse', None)
            if callable(create_with_reuse):
                input_options=({'artifacts':prepared_inputs.artifacts} if prepared_inputs is not None else {})
                sid, exact_materialization_reused = create_with_reuse(job.ids,**input_options,
                    materialization_loader=lambda snapshot,history:
                        self._root_materialization(model,job,snapshot,history))
                if type(exact_materialization_reused) is not bool:
                    raise IntegrityError('Root materialization reuse result is not boolean')
            else:
                if prepared_inputs is not None:
                    raise InvalidOperation('Worker does not support typed input preparation')
                sid = model.create(job.ids)
                exact_materialization_reused = False
            # Native prefill is not interruptible inside a batch. Observe a
            # pending cancellation before publishing its new root reference.
            if job.cancelled is not None and job.cancelled():
                raise CancelledError('Root preparation cancelled after consumed prefill')
            if (model.store.references/(job.branch+'.json')).exists():
                head = model.store.head(job.branch)
                if head.tip != sid or head.revision != 0: raise Conflict('Unacknowledged root branch changed')
                if job.presentation is not None:
                    root_operation = 'root-'+hashlib.sha256(job.request.operation.encode()).hexdigest()[:32]
                    receipt = model.store.presentation_receipt(root_operation)
                    expected = binding_for_snapshot(model.store,job.presentation,
                        operation_id=root_operation,source_head=head,target_snapshot=sid,input_ids=job.ids)
                    if receipt.get('presentation') != expected:
                        raise IntegrityError('Existing root presentation differs from request intent')
            elif job.presentation is None:
                head = model.store.fork(job.branch,sid)
            else:
                root_operation = 'root-'+hashlib.sha256(job.request.operation.encode()).hexdigest()[:32]
                root_head = BranchHead(job.branch,0,sid,model.store._root_for(sid))
                binding = binding_for_snapshot(model.store,job.presentation,operation_id=root_operation,
                    source_head=root_head,target_snapshot=sid,input_ids=job.ids)
                head = model.store.fork(job.branch,sid,namespace_id=job.presentation['namespace_id'],
                    operation_id=root_operation,presentation=binding)
            job.head = head
            after=model.backend.work_counters()
            expected_forward = 0 if exact_materialization_reused else len(job.ids)
            if after['forward_tokens']-before['forward_tokens'] != expected_forward:
                raise InvalidOperation('Root forward work differs from actual prefix tokens')
            w = self._workers[model.gpu_id if hasattr(model,'gpu_id') else model.gpu]
            w['last_root'] = head.root
            answer=dict(snapshot=sid,tokens=[],request_seconds=time.monotonic()-started,generation_seconds=0.,
                exact_materialization_reused=exact_materialization_reused,
                counter_deltas={k:after[k]-v for k,v in before.items()
                    if type(v) in (int,float) and type(after.get(k)) in (int,float)})
            if prepared_inputs is not None:
                answer['input_preparation']=preparation_receipt
            return answer
        if model.store.head(job.branch) != job.head: raise Conflict('Queued branch revision changed')
        source_head=job.source_head or job.head
        source = source_head.tip
        root = source_head.root
        if job.kind == 'reseed':
            from .seed_controls import seed_stream
            before=model.backend.work_counters()
            inserted_source=False
            source_pin=None
            try:
                if model.cache.get(source) is None:
                    manifest_path=model.checkpoint_directory/(source+'.json')
                    if not os.path.lexists(manifest_path) and self._key(model,source) not in self.pool:
                        raise InvalidOperation(
                            'Reseed needs a retained source checkpoint; submit a generation request to rebuild it')
                    future=self._preload(model,source,timeout=self._admission_timeout)
                    if future is not None: future.result(self._admission_timeout)
                    source_pin=self.pool.pin(self._key(model,source))
                    cold=source_pin.__enter__()
                    model.cache[source]=cold
                    inserted_source=True
                grammar_engine=None
                if job.grammar_registry is not None:
                    source_controller=self.store.get_snapshot(source)['controller']
                    if (isinstance(source_controller,dict)
                            and source_controller.get('constraint') is not None):
                        grammar_engine=job.grammar_registry.engine
                if grammar_engine is None:
                    sid=model.set_random_stream(source,seed_stream(job.seed))
                else:
                    sid=model.set_random_stream(
                        source,seed_stream(job.seed),grammar_engine=grammar_engine)
            finally:
                if source_pin is not None: source_pin.__exit__(None,None,None)
                if inserted_source: model.cache.pop(source,None)
            after=model.backend.work_counters()
            if after.get('forward_tokens') != before.get('forward_tokens'):
                raise InvalidOperation('Reseed unexpectedly performed native model work')
            return dict(snapshot=sid,tokens=[],request_seconds=0.,generation_seconds=0.,
                counter_deltas={k:after[k]-v for k,v in before.items()
                    if type(v) in (int,float) and type(after.get(k)) in (int,float)})
        w = self._workers[model.gpu_id if hasattr(model,'gpu_id') else model.gpu]
        # The bounded native image cache owns its LRU admission/eviction.
        # Changing canonical trees does not invalidate another owned image.
        # Preserve the previous narrow retention for unbounded adapters.
        if w['last_root'] != root and not isinstance(model.cache,CheckpointCache):
            model.cache.clear();model.backend.release_checkpoints()
        w['last_root'] = root
        before = model.backend.work_counters()
        job.request.state = 'running'
        job.request.emit('request_started',gpu=w['gpu'],source=source)
        transfer=None
        def token_ready(token):
            nonlocal transfer
            if job.cancelled is not None and job.cancelled():
                raise CancelledError('Request cancelled at consumed boundary')
            # Source load and suffix prefill have finished before consumed
            # progress arrives. Their tmpfs/native read buffers are gone.
            if transfer is not None: transfer.release();transfer=None
            # End-of-turn metadata comes from the pinned execution policy.
            # Generic coordinator models need not implement a native token API;
            # without that policy their original event contract is preserved.
            eos = model.profile.get('execution',{}).get('eos_tokens')
            fields = {} if eos is None else dict(turn_complete=token in eos)
            job.request.emit('token',token=token,**fields)
        def execute():
            from .seed_controls import seed_stream
            stream=(seed_stream(job.seed) if job.seed is not None and not job.resume
                    else job.generation_stream or job.request.operation)
            options=dict(stream=stream,on_token=token_ready)
            if job.request.pause_supported:
                if not getattr(model,'supports_consumed_pause',False):
                    raise InvalidOperation('Worker pause capability changed before execution')
                options['paused']=job.request.pause_event.is_set
            preparation=None
            if isinstance(job.ids,InputHistory):
                artifacts=job.input_artifacts
                if job.presentation_source_operation is not None:
                    # A changed retry consumes exactly the recorded delta. Do
                    # not rerender it or require an expired upload grant. Encoder
                    # callbacks run only here, on the serialized model worker,
                    # before acquiring a native execution lease.
                    prepare=getattr(self._input_preparer,'reconstruction',None)
                    if callable(prepare):
                        inputs=job.source_lease.enter_context(prepare(model,job.ids,
                            pool=self.pool,disk=self.disk,cancelled=job.cancelled))
                        artifacts=inputs.artifacts;reencode=inputs.reencode
                        preparation=inputs.receipt
                    else:
                        reencode=None
                        preparation=dict(encoder_calls=0,encodings=[],producer_available=False)
                    sources=[]
                    for span in job.ids.spans:
                        if job.cancelled is not None and job.cancelled():
                            raise CancelledError('Image retry preparation cancelled')
                        if span['kind']=='image':
                            item=job.source_lease.enter_context(artifacts.lease(span,reencode=reencode))
                            sources.append(dict(embedding_sha256=span['embedding_sha256'],source=item.source))
                    preparation=dict(preparation,images=sources)
                options['artifacts']=artifacts
            if job.cancelled is not None: options['cancelled']=job.cancelled
            if job.seed is not None and job.resume:
                options['reseed_seed']=job.seed
            if job.text_stops: options['text_stops']=job.text_stops
            if job.grammar_registry is not None:
                answer=model.run_state_request(source,job.ids,job.max_steps,
                    engine=job.grammar_registry.engine,state=job.grammar_state,
                    resume=job.resume,sampling=job.sampling,**options)
            elif job.sampling is not None:
                answer=model.run_sampled_request(source,job.ids,job.max_steps,
                    sampling=job.sampling,resume=job.resume,**options)
            elif job.resume and job.seed is not None:
                answer=model.run_state_request(source,job.ids,job.max_steps,
                    engine=None,state=None,resume=True,sampling=None,**options)
            else:answer=model.run_greedy_request(source,job.ids,job.max_steps,**options)
            if preparation is not None:answer['input_preparation']=json.loads(_json_bytes(preparation))
            return answer
        key = self._key(model,source)
        cached = model.cache.get(source)
        if cached is not None and isinstance(cached.blob,StrataImage):
            self.pool.record_transition('request_source_acquired',key,
                operation_id=job.request.operation,branch=job.branch,source_snapshot=source,
                source_kind='model_cache')
            if job.prefill_claim is not None:self._share_rebuilt_source(model,job,source)
            answer = execute()
        else:
            recomputation=None
            # Serialized aliases outside the shared pool are not accounted
            # materializations. Drop them before the explicit absence check;
            # native images were handled above and remain reusable.
            if cached is not None:
                model.cache.pop(source,None)
                cached=None
            manifest_path=model.checkpoint_directory/(source+'.json')
            if key not in self.pool and not os.path.lexists(manifest_path):
                recomputation,before=self._rebuild_missing_source(model,job,source,before)
            else:
                recomputation=None
            if recomputation is not None:
                # Rebuild already produced a private native image on this
                # worker. No disk/RAM checkpoint restore is involved.
                answer=execute()
                answer['source_recomputation']=recomputation
            else:
                # A serialized alias is not an independent resident image:
                # the shared pool owns and accounts for those bytes. Drop any
                # stale alias, then borrow the pinned pool value for execution.
                if cached is not None: model.cache.pop(source,None)
                deadline=time.monotonic()+self._admission_timeout
                while True:
                    remaining=deadline-time.monotonic()
                    if remaining<=0: raise PoolFull('Timed out admitting source checkpoint')
                    try:
                        future=self._preload(model,source,timeout=remaining)
                        if future is not None: future.result(remaining)
                        future=None
                        pin=self.pool.pin(key);cold=pin.__enter__()
                        self.pool.record_transition('request_source_acquired',key,
                            operation_id=job.request.operation,branch=job.branch,source_snapshot=source,
                            source_kind='shared_pool')
                        break
                    except FileNotFoundError:
                        # The source may have been a RAM-only entry when the
                        # outer routing check ran, then lose its last resident
                        # owner before preload acquires it. Rebuild only when
                        # both representations are now genuinely absent.
                        if key not in self.pool and not os.path.lexists(manifest_path):
                            recomputation,before=self._rebuild_missing_source(
                                model,job,source,before)
                            answer=execute()
                            answer['source_recomputation']=recomputation
                            break
                        raise
                    except IntegrityError as error:
                        # BranchingModel deliberately wraps manifest I/O
                        # failures as IntegrityError. Only the underlying
                        # missing-manifest case may transition to replay; a
                        # present malformed manifest or missing payload stays
                        # a strict integrity failure.
                        if (isinstance(error.__cause__,FileNotFoundError)
                                and not os.path.lexists(manifest_path)
                                and key not in self.pool):
                            recomputation,before=self._rebuild_missing_source(
                                model,job,source,before)
                            answer=execute()
                            answer['source_recomputation']=recomputation
                            break
                        raise
                    except (PoolFull,KeyError):
                        # A nonblocking read-ahead may have lost admission, or a
                        # clean image may be evicted before we acquire its pin.
                        continue
                if recomputation is None:
                    try:
                        # Account for tmpfs transport and native deserialization
                        # scratch in addition to the pinned shared payload.
                        transfer=self.pool.reserve(3*checkpoint_bytes(cold)+12*cold.consumed,
                            timeout=max(0,deadline-time.monotonic()))
                        model.cache[source] = cold
                        try:
                            answer = execute()
                        finally:
                            retained = model.cache.get(source)
                            if retained is not None and not isinstance(retained.blob,StrataImage):
                                model.cache.pop(source,None)
                    finally:
                        if transfer is not None: transfer.release();transfer=None
                        pin.__exit__(None,None,None)
        after = model.backend.work_counters()
        expected = len(job.ids)+len(answer['tokens'])
        if after['forward_tokens']-before['forward_tokens'] != expected:
            raise InvalidOperation('Request recomputed a saved prefix or omitted consumed work')
        answer['counter_deltas'] = {k:after[k]-v for k,v in before.items()
            if type(v) in (int,float) and type(after.get(k)) in (int,float)}
        return answer

    def _prepare_suffix_on_worker(self,model,job):
        from .vision_execution import validate_boundary
        if job.cancelled():raise CancelledError('Image preparation cancelled before execution')
        if model.store.head(job.branch)!=job.head:raise Conflict('Queued branch revision changed')
        preparer=self._input_preparer
        if self._preparer_identity(preparer)!=job.input_preparer_identity:
            raise InvalidOperation('Input preparer identity changed before execution')
        before=model.backend.work_counters()
        prepared=job.source_lease.enter_context(preparer.prepare(model,job.input_request,
            pool=self.pool,disk=self.disk,cancelled=job.cancelled,admission=job.input_admission,
            source_snapshot=job.head.tip))
        if self._preparer_identity(preparer)!=job.input_preparer_identity:
            raise InvalidOperation('Input preparer identity changed during execution')
        if not isinstance(getattr(prepared,'history',None),InputHistory):
            raise InvalidOperation('Input preparation must produce canonical InputHistory')
        source=history_for_snapshot(model.store,job.head.tip)
        execution=model.profile['execution']
        validate_boundary(source+prepared.history,capacity=execution['capacity']-job.max_steps,
            vocab=model.backend.vocab_size,width=execution['embedding_width'])
        raw=_json_bytes(prepared.receipt)
        if len(raw)>1024*1024:raise InvalidOperation('Input preparation receipt exceeds 1 MiB')
        after=model.backend.work_counters()
        if after['forward_tokens']!=before['forward_tokens']:
            raise IntegrityError('Input preparation unexpectedly evaluated model tokens')
        if model.store.head(job.branch)!=job.head:raise Conflict('Branch changed during image preparation')
        if job.cancelled():raise CancelledError('Image preparation cancelled after encoding')
        prepared.receipt=dict(json.loads(raw),forward_tokens=0,source_snapshot=job.head.tip)
        return prepared

    @staticmethod
    def _discard_job_created_images(model,job,error):
        baseline=job.cache_keys_before
        if baseline is None:
            return
        removed=False
        try:
            for snapshot in tuple(model.cache):
                if snapshot not in baseline:
                    model.cache.pop(snapshot,None)
                    removed=True
            backend=model.backend
            if removed and not getattr(backend,'_closed',False):
                backend.release_checkpoints()
        except BaseException as cleanup_error:
            try: error.add_note('Job-created native image cleanup also failed: '+str(cleanup_error))
            except Exception: pass
        finally:
            job.reconstructed_source=None
            job.cache_keys_before=None

    def _worker(self,gpu):
        w = self._workers[gpu];w['gpu'] = gpu
        model = None
        try:
            model = self._factory(gpu)
            model.gpu_id = gpu
            self._observe_worker_restores(model,gpu)
            w['capabilities']=self._backend_capabilities(model)
            if normalize_gpu_selection(model.profile['execution'].get('GPU')) != self.config.gpus:
                raise InvalidOperation('Worker profile GPU selection differs from coordinator')
            w['model_contract_id']=model.profile['model_contract_id']
            w['execution_profile_id']=model.profile['execution_profile_id']
            w['model'] = model;w['health']='healthy';w['ready'].set_result(model)
            with self._condition: self._condition.notify_all()
            while True:
                job = w['queue'].get()
                if job is None: break
                prepared=pin=lease=None
                key=None
                publication_handoff=False
                replace_after_job=False
                service_ok=False
                w['restore_seconds']=0.
                if self.config.scheduling!='fifo':
                    service_started=time.monotonic()
                    service_before=model.backend.work_counters().get('forward_tokens',0)
                try:
                    w['active'] += 1;w['max_active'] = max(w['max_active'],w['active'])
                    bound = model if job.principal is None else model.for_principal(job.principal)
                    job.cache_keys_before=frozenset(model.cache)
                    if job.kind=='prepare_inputs':
                        inputs=self._prepare_suffix_on_worker(bound,job)
                        with self._condition:
                            self._branches.discard(job.branch)
                            job.request.state='prepared'
                            # The caller's context owns only derived-artifact
                            # and canonical read leases, never a native lease.
                            publication_handoff=True
                            job.request.durable.set_result(None)
                            job.request.snapshot.set_result(inputs)
                            self._condition.notify_all()
                        inputs=None
                        continue
                    answer = self._run(bound,job)
                    service_ok=True
                    self._touch_idle_source(job)
                    sid = answer['snapshot']
                    self._headroom()
                    capture_started=time.monotonic()
                    estimate = 3*len(model.cache[sid].blob)+8*len(model.cache[sid].frontier)
                    with self.pool.reserve(estimate,timeout=self._admission_timeout) as reservation:
                        prepared = bound.prepare_checkpoint(sid)
                        key = self._key(model,sid)
                        reservation.admit(key,prepared.checkpoint,durable=False)
                    # The immutable bytes are now admitted and can be restored
                    # by another independent slot while this branch publishes.
                    self._release_prefill(job)
                    answer['capture_seconds']=time.monotonic()-capture_started
                    answer['ram_reservation_bytes']=estimate
                    # Protect canonical records across publication and GC. No
                    # exclusive native lease crosses into the writer thread.
                    lease = self.store.read_lease(sid);lease.__enter__()
                    pin = self.pool.pin(key);pin.__enter__()
                    job.request.state = 'persisting'
                    job.request.emit('snapshot',gpu=gpu,**answer)
                    job.request.snapshot.set_result(answer)
                    # Bounded native caches retain hot trees within their
                    # existing byte/entry budgets. Unbounded adapters retain
                    # only this source and tip, as before. Cold aliases are
                    # still removed before native drop requests are flushed.
                    if not isinstance(model.cache,CheckpointCache):
                        keep = {(job.source_head or job.head).tip,sid}
                        for old in tuple(model.cache):
                            if old not in keep: model.cache.pop(old,None)
                    # Finish all worker-owned native cleanup before transferring
                    # the pool pin and canonical read lease to the writer. If
                    # cleanup or writer submission fails, this worker still
                    # owns and can unwind both resources without racing a save.
                    model.backend.release_checkpoints()
                    self._submit_publication(bound,job,prepared,key,pin,lease)
                    publication_handoff=True
                    pin=lease=None
                    # Worker-created source/tip images now belong to the
                    # asynchronous publication handoff; this worker no longer
                    # owns failure cleanup for them.
                    job.reconstructed_source=None
                    job.cache_keys_before=None
                except BaseException as exc:
                    if not publication_handoff:
                        # A completed capture remains retryable if its bytes
                        # were admitted. Release transient ownership first;
                        # the failed-publication record then owns only the
                        # unpinned, ordinary-LRU pool entry and envelope.
                        if pin is not None:
                            try: pin.__exit__(None,None,None)
                            except BaseException as cleanup_error:
                                try: exc.add_note('Pre-submit pool pin release also failed: '+str(cleanup_error))
                                except Exception: pass
                            pin=None
                        if lease is not None:
                            try: lease.__exit__(None,None,None)
                            except BaseException as cleanup_error:
                                try: exc.add_note('Pre-submit store lease release also failed: '+str(cleanup_error))
                                except Exception: pass
                            lease=None
                        retained = prepared is not None and key is not None and key in self.pool
                        durable_alias = self.pool.is_durable(key) if retained else False
                        if retained and not durable_alias:
                            self.pool.mark_failed(key)
                        if retained:
                            retry=_FailedPublication(bound,job.branch,job.head,job.ids,job.max_steps,
                                job.kind,job.principal,job.presentation,prepared.snapshot,key,
                                bytes(prepared.envelope),job.sampling,job.resume,
                                job.source_head,job.generation_stream,job.presentation_source_operation,job.seed,
                                job.text_stops)
                            with self._condition:
                                self._failure_payloads[job.request.operation]=retry
                    self._discard_job_created_images(model,job,exc)
                    replace_after_job=bool(getattr(getattr(model,'backend',None),
                        '_protocol_failed',False))
                    if replace_after_job:
                        with self._condition:
                            w['health']='recovering'
                            w['failure_reason']='native protocol poisoned'
                            self._condition.notify_all()
                    # Request Futures retain their exception objects until a
                    # caller observes them. Never retain this worker traceback:
                    # it can own rebuilt checkpoints/native images through locals.
                    self._fail(job,self._detached_error(exc))
                finally:
                    if not publication_handoff: self._release_source(job)
                    # The long-lived worker frame otherwise keeps its last
                    # prepared image and lease context managers alive while
                    # blocked waiting for the next queued job.
                    prepared=pin=lease=None
                    with self._condition:
                        self._release_prefill(job)
                        if self._inventory is not None:
                            w['native_inventory']=self._native_inventory(model,gpu)
                        if self.config.scheduling!='fifo':
                            count=self._active_groups.get(job.prefill_key,1)-1
                            if count:self._active_groups[job.prefill_key]=count
                            else:self._active_groups.pop(job.prefill_key,None)
                            work=model.backend.work_counters().get('forward_tokens',0)-service_before
                            if work>0:
                                sample=(time.monotonic()-service_started)/work
                                self._service_rate[gpu]=.75*self._service_rate.get(gpu,sample)+.25*sample
                                if service_ok and self._costs is not None:
                                    elapsed=max(0.,time.monotonic()-service_started-w['restore_seconds'])
                                    self._costs.observe_forward(gpu,work,elapsed)
                        w['active'] -= 1
                        if not replace_after_job: w['busy'] = False
                        self._condition.notify_all()
                if replace_after_job:
                    replacement=self._replace_poisoned_worker(gpu,model,w)
                    if replacement is not None: model=replacement
        except BaseException as exc:
            with self._condition:
                w['health']='unavailable'
                w['failure_reason']=str(exc)
                if not w['ready'].done(): w['ready'].set_exception(exc)
                self._condition.notify_all()
        finally:
            if model is not None:
                try: self._close_worker_backend(model,gpu)
                except BaseException as exc:
                    with self._condition:
                        w['health']='unavailable'
                        w['failure_reason']=str(exc)
                        self._condition.notify_all()
            with self._condition:
                if w['health'] != 'unavailable': w['health']='closed'
                self._condition.notify_all()

    def _submit_publication(self,model,job,prepared,key,pin,lease):
        # ThreadPoolExecutor retains its last work item on the worker thread.
        # Pass a one-shot holder and clear it before entering _publish so a
        # completed writer cannot keep a large PreparedCheckpoint alive. The
        # decision gate makes executor submission the exact ownership boundary:
        # submit() can enqueue work and then raise while starting a thread, so
        # queued work must wait until the caller confirms the handoff.
        holder=[prepared]
        decided=threading.Event()
        accepted=[False]
        def guarded_publish():
            decided.wait()
            if not accepted[0]:
                holder.clear()
                return None
            return self._publish_held(model,job,holder,key,pin,lease)
        try:
            future=self._writer.submit(guarded_publish)
        except BaseException:
            holder.clear()
            decided.set()
            raise
        accepted[0]=True
        decided.set()
        return future

    def _publish_held(self,model,job,holder,key,pin,lease):
        prepared=holder.pop()
        return self._publish(model,job,prepared,key,pin,lease)

    def _publish(self,model,job,prepared,key,pin,lease):
        failed = False
        was_durable=self.pool.is_durable(key)
        try:
            size = checkpoint_bytes(prepared.checkpoint)
            # Streaming SHA reads can hold two chunks plus a packed frontier.
            scratch = 2*min(8*1024*1024,size)+4*len(prepared.checkpoint.frontier)
            with self.pool.reserve(scratch,timeout=self._admission_timeout):
                admission_started=time.monotonic()
                # Pin the digest before probing deduplicated payload bytes.
                # Otherwise a concurrent collector can remove an old copy
                # after exists() and make a manifest-only reservation too small.
                with self.disk.reserve(len(prepared.envelope),timeout=self._admission_timeout,
                                       payload_sha256=prepared.checkpoint.checksum) as disk_reservation:
                    if not (self.disk.directory/(prepared.checkpoint.checksum+'.state')).exists():
                        remaining=max(0,self._admission_timeout-(time.monotonic()-admission_started))
                        disk_reservation.grow(size,timeout=remaining)
                    if job.presentation is None or job.kind == 'create':
                        receipt = model.save_prepared_branch(job.branch,job.head.revision,prepared,job.request.operation)
                    else:
                        source_head=job.source_head or job.head
                        binding = binding_for_snapshot(model.store,job.presentation,
                            operation_id=job.request.operation,source_head=source_head,
                            target_snapshot=prepared.snapshot,input_ids=job.ids,
                            generation_operation_id=job.generation_stream)
                        retry_source_options = {}
                        if job.presentation_source_operation is not None:
                            retry_source_options['presentation_source_operation'] = job.presentation_source_operation
                        receipt = model.save_prepared_branch(job.branch,job.head.revision,prepared,
                            job.request.operation,presentation=binding,
                            **retry_source_options)
                    self.disk.touch(prepared.snapshot)
            self.pool.mark_durable(key)
            job.request.state = 'durable'
            job.request.emit('durable',gpu=model.gpu_id,receipt=receipt)
            job.request.durable.set_result(receipt)
            with self._condition: self._completed += 1
        except BaseException as exc:
            failed = True
            snapshot=prepared.snapshot
            if not was_durable:
                self.pool.mark_failed(key)
                if model.cache.get(snapshot) is not None:
                    model.cache.pop(snapshot,None)
            retry=_FailedPublication(model,job.branch,job.head,job.ids,job.max_steps,
                job.kind,job.principal,job.presentation,snapshot,key,bytes(prepared.envelope),job.sampling,job.resume,
                job.source_head,job.generation_stream,job.presentation_source_operation,job.seed,job.text_stops)
            # Drop every lease/pin before publishing retry state. The failed
            # bytes are now ordinary unpinned RAM LRU data.
            try: pin.__exit__(None,None,None)
            finally: lease.__exit__(None,None,None)
            with self._condition:
                self._failure_payloads[job.request.operation] = retry
            safe_error=self._detached_error(exc)
            self._fail(job,safe_error)
        finally:
            if not failed:
                try:
                    pin.__exit__(None,None,None);lease.__exit__(None,None,None)
                finally:
                    self._release_source(job)
                    with self._condition:
                        self._branches.discard(job.branch);self._condition.notify_all()

    def _fail(self,job,exc,*,release_branch=True):
        stopped = isinstance(exc,CancelledError) and not job.request.snapshot.done()
        job.request.state = 'cancelled' if stopped else 'failed'
        with self._condition:
            self._release_prefill(job)
            self._release_source(job)
            if stopped: self._cancelled += 1
            else: self._failed += 1
            if release_branch: self._branches.discard(job.branch)
            self._condition.notify_all()
        failure_fields={'error':str(exc)}
        if isinstance(exc,OSError) and exc.errno is not None:
            failure_fields['errno']=exc.errno
        job.request.emit('cancelled' if stopped else 'failure',**failure_fields)
        for future in (job.request.snapshot,job.request.durable):
            if not future.done():
                if stopped: future.cancel()
                else: future.set_exception(exc)

    @staticmethod
    def _detached_error(error):
        """Retain a useful error without keeping writer-frame tracebacks alive."""
        notes=getattr(error,'__notes__',())
        if isinstance(error,OSError):
            try:
                if error.errno is None:
                    result=type(error)(str(error))
                else:
                    result=type(error)(error.errno,error.strerror or str(error),error.filename)
            except BaseException:
                if error.errno is None: result=OSError(str(error))
                else: result=OSError(error.errno,error.strerror or str(error),error.filename)
            result.__traceback__=None
            result.__cause__=None
            result.__context__=None
            for note in notes: result.add_note(note)
            return result
        try:
            # Keep only the rendered message; exception args can themselves
            # reference request/checkpoint objects supplied by custom backends.
            result=type(error)(str(error))
        except BaseException:
            result=InvalidOperation(str(error))
        result.__traceback__=None
        result.__cause__=None
        result.__context__=None
        for note in notes: result.add_note(note)
        return result

    def _publication_receipt_matches(self,operation,record):
        with self.store._locked():
            receipt=self.store._read_refs()['receipts'].get(operation)
        if not isinstance(receipt,dict): return False
        result=receipt.get('result')
        return (isinstance(result,dict) and result.get('name')==record.branch
                and result.get('tip')==record.snapshot)

    def _publication_history_available(self,record):
        """Require the exact failed target and its full canonical ancestry."""
        cursor=record.snapshot
        seen=set()
        root=None
        try:
            with self.store._locked():
                while cursor is not None:
                    if cursor in seen: return False
                    seen.add(cursor)
                    root=cursor
                    cursor=self.store.get_snapshot(cursor).get('parent')
        except (NotFound,InvalidOperation,IntegrityError,OSError):
            return False
        expected=record.head.root if record.head is not None else record.snapshot
        return root==expected

    def _publication_retryable(self,operation,record):
        if record is None or record.key not in self.pool or record.branch in self._branches:
            return False
        if not self._publication_history_available(record): return False
        if self._publication_receipt_matches(operation,record): return True
        try:
            return self.store.head(record.branch)==record.head
        except (NotFound,InvalidOperation,Conflict):
            return False

    def discard_failed(self,operation):
        """Explicitly abandon an unacknowledged computation, preserving its old branch."""
        with self._condition:
            self._failure_payloads.pop(operation)
            self._condition.notify_all()

    def retry_failed(self,operation):
        """Retry publication only while the original failed bytes remain resident."""
        with self._condition:
            if not self._accepting: raise InvalidOperation('Coordinator is closing')
            record=self._failure_payloads.get(operation)
            if record is None: raise InvalidOperation('Operation has no retained publication to retry')
            if record.branch in self._branches:
                raise Conflict('Branch has a pending request/publication')
            if (self.store.head(record.branch)!=record.head
                    and not self._publication_receipt_matches(operation,record)):
                self._failure_payloads.pop(operation,None)
                self._condition.notify_all()
                raise Conflict('Failed publication source head is stale')
            if record.key not in self.pool:
                self._failure_payloads.pop(operation,None)
                self._condition.notify_all()
                raise InvalidOperation('Failed checkpoint bytes were evicted; retry cannot recompute')
            if not self._publication_history_available(record):
                self._failure_payloads.pop(operation,None)
                self._condition.notify_all()
                raise InvalidOperation('Failed checkpoint history was collected; retry cannot recompute')
            request=self._requests[operation]
            if not request.durable.done():
                raise Conflict('Publication failure is still being reported')
            lease=self.store.read_lease(record.snapshot)
            try:
                lease.__enter__()
            except (NotFound,IntegrityError,OSError):
                self._failure_payloads.pop(operation,None)
                self._condition.notify_all()
                raise InvalidOperation('Failed checkpoint history was collected; retry cannot recompute') from None
            lease_entered=True
            pin=None
            history_unavailable=False
            try:
                if not self._publication_history_available(record):
                    history_unavailable=True
                    raise InvalidOperation('Failed checkpoint history was collected; retry cannot recompute')
                if (self.store.head(record.branch)!=record.head
                        and not self._publication_receipt_matches(operation,record)):
                    raise Conflict('Failed publication source head is stale')
                pin=self.pool.pin(record.key);pin.__enter__()
                checkpoint=self.pool.get(record.key)
                prepared=PreparedCheckpoint(record.snapshot,checkpoint,record.envelope)
                job=_Job(request,record.branch,record.head,record.ids,record.max_steps,
                         record.kind,record.principal,presentation=record.presentation,
                         sampling=record.sampling,resume=record.resume,
                         source_head=record.source_head or record.head,
                         generation_stream=record.generation_stream or operation,
                         presentation_source_operation=record.presentation_source_operation,
                         seed=record.seed,text_stops=record.text_stops)
                self._failure_payloads.pop(operation,None)
                self._branches.add(record.branch)
                request.durable=Future()
                request.state='persisting'
                self._condition.notify_all()
            except BaseException as exc:
                if pin is not None: pin.__exit__(None,None,None)
                if lease_entered: lease.__exit__(None,None,None)
                if history_unavailable:
                    self._failure_payloads.pop(operation,None)
                    self._condition.notify_all()
                raise
            try:
                self._submit_publication(record.model,job,prepared,record.key,pin,lease)
            except BaseException as exc:
                # Submission is the ownership-transfer boundary. If the
                # executor rejects the job, retain the original bytes and
                # operation while restoring a terminal failed request.
                cleanup_errors=[]
                try: pin.__exit__(None,None,None)
                except BaseException as cleanup_error: cleanup_errors.append(cleanup_error)
                try: lease.__exit__(None,None,None)
                except BaseException as cleanup_error: cleanup_errors.append(cleanup_error)
                for cleanup_error in cleanup_errors:
                    try: exc.add_note('Retry ownership release also failed: '+str(cleanup_error))
                    except Exception: pass
                with self._condition:
                    self._failure_payloads[operation]=record
                self._fail(job,self._detached_error(exc))
                return request.durable
            return request.durable

    def committed_root_status(self, operation):
        """Prove a modern root publication from existing immutable records.

        No checkpoint payload or native lease is needed to acknowledge prior
        commitment. Missing records never prove abort or authorize replay.
        """
        _name(operation)
        try:
            envelope = json.loads((self._intents / (operation + '.json')).read_bytes())
        except FileNotFoundError:
            raise NotFound('No retained root intent') from None
        except (OSError, ValueError) as error:
            raise IntegrityError('Root intent is unreadable') from error
        value = envelope.get('intent') if isinstance(envelope, dict) else None
        if not isinstance(value, dict):
            raise IntegrityError('Root intent is corrupt')
        presentation = value.get('presentation')
        if (value.get('head') is not None or not isinstance(presentation, dict)
                or presentation.get('kind') != 'root'):
            raise NotFound('Operation is not a root publication')
        if (type(value.get('schema')) is not int or value['schema'] != 2
                or set(value) != {'schema', 'signature', 'head', 'presentation'}
                or envelope.get('checksum') != hashlib.sha256(_json_bytes(value)).hexdigest()):
            raise IntegrityError('Root intent is corrupt')
        _id(value['signature'])
        committed = self.store.checkpoint_receipt(operation)
        head = BranchHead(**committed['result'])
        metadata = committed['checkpoint_receipt']
        if head.tip != head.root or metadata['snapshot'] != head.tip:
            raise IntegrityError('Root receipt is not a canonical root')
        with self.store.read_lease(head.tip):
            record = self.store.get_snapshot(head.tip)
            if (record['parent'] is not None or record['controller'] != {'status': 'idle'}
                    or record.get('model_contract_id') != metadata['model_contract_id']):
                raise IntegrityError('Root snapshot differs from publication')
            history = history_for_snapshot(self.store, head.tip)
            root_operation = 'root-' + hashlib.sha256(operation.encode()).hexdigest()[:32]
            original = self.store.presentation_receipt(root_operation)
            root_head = BranchHead(head.name, 0, head.tip, head.root)
            expected = binding_for_snapshot(self.store, presentation,
                operation_id=root_operation, source_head=root_head,
                target_snapshot=head.tip, input_ids=history)
            if (original['presentation'] != expected
                    or original['result'] != asdict(root_head)):
                raise IntegrityError('Root presentation differs from original intent')
        receipt = dict(branch=asdict(head), snapshot=head.tip,
            payload_sha256=metadata['payload_sha256'],
            model_contract_id=metadata['model_contract_id'],
            execution_profile_id=metadata['execution_profile_id'])
        return dict(operation_id=operation, branch=head.name, state='durable',
                    scope='canonical-root', durable=True, receipt=receipt)

    def operation_status(self,operation):
        """Describe this process's admitted request; never infer success from output."""
        _name(operation)
        with self._condition:
            handle=self._requests.get(operation)
            if handle is None: raise NotFound('Operation is not retained by this coordinator process')
            durable=(handle.durable.done() and not handle.durable.cancelled()
                     and handle.durable.exception() is None)
            retryable=self._publication_retryable(operation,self._failure_payloads.get(operation))
            result=dict(operation_id=operation,branch=handle.branch,state=handle.state,
                scope='coordinator-process',durable=durable,
                publication_retryable=(retryable and handle.durable.done() and not durable))
            if durable: result['receipt']=handle.durable.result()
            return result

    def pending_request(self,branch):
        """Transient request or retryable publication status for a branch."""
        _name(branch)
        with self._condition:
            for handle in reversed(tuple(self._requests.values())):
                if handle.branch!=branch: continue
                if handle.state in ('queued','loading','persisting'):
                    return self.operation_status(handle.operation)
                if (handle.state=='failed'
                        and handle.operation in self._failure_payloads):
                    status=self.operation_status(handle.operation)
                    if status['publication_retryable']: return status
                if handle.state=='durable': return None
            return None

    def retry_publication(self,operation):
        """Idempotently finish an existing save; never admit model computation."""
        _name(operation)
        with self._condition:
            handle=self._requests.get(operation)
            if handle is None: raise NotFound('Operation is not retained by this coordinator process')
            if operation in self._failure_payloads:
                if not handle.durable.done():
                    raise Conflict('Publication failure is still being reported; refresh before retrying')
                return self.retry_failed(operation)
            if handle.state in ('persisting','durable'):
                return handle.durable
            raise InvalidOperation('Operation has no retained publication to retry')

    def metrics(self):
        with self._condition:
            result=dict(GPU=list(self.config.gpus),worker_count=self.config.worker_count,
                physical_gpu_count=len(self.config.physical_gpus),
                active_per_gpu={worker_key(gpu):w['active'] for gpu,w in self._workers.items()},
                max_active_per_gpu={worker_key(gpu):w['max_active'] for gpu,w in self._workers.items()},
                queued=len(self._pending),pending_branches=len(self._branches),
                failed_publications=len(self._failure_payloads),completed=self._completed,failed=self._failed,
                cancelled=self._cancelled,
                preload_failures=self._preload_failures,
                ram=self.pool.metrics(),disk=self.disk.metrics())
            result['idle_expiry']=dict(seconds=self.config.checkpoint_idle_seconds,
                runs=self._expiry_runs,last_result=dict(self._expiry_last_result),
                last_error=self._expiry_last_error,
                running=self._expiry_thread is not None and self._expiry_thread.is_alive())
            result['workers']={worker_key(gpu):dict(health=w['health'],restarts=w['restarts'],
                failure_reason=w['failure_reason'],active=w['active'],max_active=w['max_active'],
                pid=getattr(getattr(getattr(w.get('model'),'backend',None),'process',None),'pid',None))
                for gpu,w in self._workers.items()}
            result['scheduling']=dict(policy=self.config.scheduling,
                inflight_prefills=len(self._prefills),
                waiting_on_prefill=sum(job.prefill_key in self._prefills for job in self._pending),
                queued_per_prefill={key:sum(job.prefill_key==key for job in self._pending)
                    for key in {job.prefill_key for job in self._pending} if key is not None},
                active_per_prefill=dict(self._active_groups),
                seconds_per_forward_token={worker_key(gpu):rate for gpu,rate in self._service_rate.items()})
            if self._costs is not None:
                rates=self._costs.metrics()
                result['scheduling']['transfer_costs']={name:{str(worker_key(gpu)):value for gpu,value in values.items()}
                    for name,values in rates.items()}
                result['scheduling']['last_placement']=self._last_placement
            if self._inventory is not None:
                result['cache_inventory']=dict(enabled=True,event_limit=1024,entry_limit=self._inventory.entry_limit)
            return result

    def close(self):
        with self._condition:
            if self._closed: return
            self._accepting = False
            for job in tuple(self._pending): self.cancel(job.request)
            handles = list(self._requests.values())+[job.request for job in self._input_preparations.values()]
            self._expiry_stop.set()
        if self._expiry_thread is not None:
            self._expiry_thread.join()
        for handle in handles:
            try: handle.durable.result(600)
            except Exception: pass
        with self._condition:
            self._stopping = True;self._condition.notify_all()
        self._dispatcher.join()
        self._writer.shutdown(wait=True)
        for w in self._workers.values(): w['queue'].put(None)
        for t in self._threads: t.join()
        with self._condition:
            for job in self._input_preparations.values():self._release_source(job)
            self._input_preparations.clear()
        # Failed publications were never acknowledged. Closing deliberately
        # abandons their volatile copies; files/history remain available to audit.
        for operation in tuple(self._failure_payloads): self.discard_failed(operation)
        self.pool.close();self._process_lock.close();self._closed = True
