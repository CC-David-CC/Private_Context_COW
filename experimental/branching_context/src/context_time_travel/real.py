"""Canonical immutable API with checked native backend materializations."""
from contextlib import contextmanager, ExitStack, nullcontext
from concurrent.futures import CancelledError
import ctypes as C
from dataclasses import asdict, dataclass
import hashlib
import fcntl
import json
import os
from pathlib import Path
import stat
import struct
import copy
import math
import time
import tempfile
from types import MappingProxyType

from .oracle import FreshOracle
from .store import Store, IntegrityError, InvalidOperation, _stage, _sync_dir, _json_bytes, _decode, _private_directory
from . import controller
from .digest_attestation import _attest_factory_result, verified_checkpoint_digest
from .input_spans import InputHistory
from .vision_execution import RESUME_CONTRACT, POSITION_CONVENTION, input_binding, validate_boundary


def digest_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(8*1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


class NativeBackend:
    def __init__(self, profile):
        self.profile = profile
        self.validate_execution_profile()
        if digest_file(profile['model_path']) != profile['model_weights_sha256']:
            raise IntegrityError('Model weight fingerprint mismatch')
        self.lib = C.CDLL(profile['bridge_library'])
        signatures = {
            'error': (C.c_char_p, []), 'load': (C.c_void_p, [C.c_char_p]),
            'unload': (None, [C.c_void_p]), 'vocab': (C.c_int, [C.c_void_p]),
            'meta': (C.c_int, [C.c_void_p,C.c_char_p,C.c_void_p,C.c_size_t]),
            'piece': (C.c_int,[C.c_void_p,C.c_int,C.c_void_p,C.c_int]),
            'eog': (C.c_bool,[C.c_void_p,C.c_int]),
            'new': (C.c_void_p, [C.c_void_p]), 'free': (None, [C.c_void_p]),
            'position': (C.c_int, [C.c_void_p]),
            'consume': (C.c_int, [C.c_void_p,C.POINTER(C.c_int32),C.c_int]),
            'logits': (C.POINTER(C.c_float), [C.c_void_p]),
            'work': (C.c_uint64,[C.c_void_p]), 'state_size': (C.c_size_t,[C.c_void_p]),
            'export': (C.c_size_t,[C.c_void_p,C.c_void_p,C.c_size_t]),
            'import': (C.c_size_t,[C.c_void_p,C.c_void_p,C.c_size_t]),
        }
        for name, (result,args) in signatures.items():
            fn = getattr(self.lib,'ctt_'+name)
            fn.restype = result
            fn.argtypes = args
        self.model = self.lib.ctt_load(os.fsencode(profile['model_path']))
        if not self.model:
            raise RuntimeError(self.lib.ctt_error().decode())
        self.vocab_size = self.lib.ctt_vocab(self.model)
        self.stats = dict(consumed_tokens=0, checkpoint_export_bytes=0,
                          checkpoint_import_bytes=0, contexts_created=0,
                          contexts_freed=0, shared_prefix_pages=None)

    def validate_execution_profile(self):
        execution = self.profile['execution']
        actual_math = os.environ.get('GGML_CUDA_CUBLAS_COMPUTE_TYPE','auto')
        actual_tf32 = os.environ.get('NVIDIA_TF32_OVERRIDE','unset')
        if actual_math != execution['cuda_cublas_compute_type'] or actual_tf32 != execution['nvidia_tf32_override']:
            raise IntegrityError('Execution math profile mismatch')
        for path,expected in self.profile['native_build']['sha256'].items():
            if digest_file(path) != expected:
                raise IntegrityError('Native binary/header fingerprint mismatch: '+path)
        expected_id = hashlib.sha256(json.dumps(dict(build=self.profile['native_build'],
            execution=execution,hardware=self.profile['hardware']),sort_keys=True).encode()).hexdigest()
        if expected_id != self.profile['execution_profile_id']:
            raise IntegrityError('Execution profile fingerprint mismatch')

    def metadata(self, key):
        data = C.create_string_buffer(8192)
        n = self.lib.ctt_meta(self.model,key.encode(),data,len(data))
        return data.value.decode() if n>=0 else None

    def contract_fields(self):
        return dict(attention='causal-full',state_kind='full-attention-transformer')

    def new_cache(self):
        return {}

    def park(self, lease):
        # This adapter has no independent native history cache.
        return None

    def restore_cached(self, lease, history):
        return False

    def clear_cached_histories(self):
        return 0

    def release_checkpoints(self):
        # Owned byte images are released by dropping their Python references.
        pass

    def position(self, lease):
        return self.lib.ctt_position(lease)

    def is_eog(self, token):
        return self.lib.ctt_eog(self.model,token)

    @contextmanager
    def lease(self):
        p = self.lib.ctt_new(self.model)
        if not p:
            raise RuntimeError(self.lib.ctt_error().decode())
        self.stats['contexts_created'] += 1
        try:
            yield p
        finally:
            self.lib.ctt_free(p)
            self.stats['contexts_freed'] += 1

    def consume(self, lease, ids):
        if not ids or any(type(t) is not int or not 0<=t<self.vocab_size for t in ids):
            raise InvalidOperation('Invalid token IDs')
        if self.lib.ctt_position(lease)+1+len(ids)>2048:
            raise InvalidOperation('Certified execution context exceeded')
        native_ids = (C.c_int32*len(ids))(*ids)
        before = self.lib.ctt_work(lease)
        if self.lib.ctt_consume(lease,native_ids,len(ids)):
            raise RuntimeError(self.lib.ctt_error().decode())
        delta = self.lib.ctt_work(lease)-before
        if delta != len(ids):
            raise AssertionError('Native consumed-work counter mismatch')
        self.stats['consumed_tokens'] += delta

    def logits(self, lease):
        data = self.lib.ctt_logits(lease)
        if not data:
            raise IntegrityError('Missing consumed frontier')
        return tuple(data[:self.vocab_size])

    def export(self, lease):
        size = self.lib.ctt_state_size(lease)
        data = C.create_string_buffer(size)
        if self.lib.ctt_export(lease,data,size) != size:
            raise IntegrityError('Incomplete state export')
        self.stats['checkpoint_export_bytes'] += size
        return data.raw

    def piece(self, token):
        data = C.create_string_buffer(4096)
        n = self.lib.ctt_piece(self.model,token,data,len(data))
        if n < 0:
            data = C.create_string_buffer(-n)
            n = self.lib.ctt_piece(self.model,token,data,len(data))
        if n < 0:
            raise IntegrityError('Token output conversion failed')
        return data.raw[:n]

    def restore(self, lease, blob, *, history=None, frontier=None):
        data = C.create_string_buffer(blob,len(blob))
        if getattr(self.lib,'ctt_import')(lease,data,len(blob)) != len(blob):
            raise IntegrityError('Incomplete state import')
        self.stats['checkpoint_import_bytes'] += len(blob)
        return blob

    def serialize(self, lease, state):
        return state

    def checkpoint_metrics(self, values):
        unique = {id(item.blob):len(item.blob) for item in values}
        return dict(native_counters=dict(self.stats),owned_checkpoint_bytes=sum(unique.values()))

    def close(self):
        if self.model:
            self.lib.ctt_unload(self.model)
            self.model = None


@dataclass(frozen=True)
class Checkpoint:
    model_contract_id: str
    execution_profile_id: str
    consumed: int
    blob: object
    frontier: tuple
    checksum: str
    input_identity: str = ''
    execution_schedule: tuple = ()

    @staticmethod
    def digest(blob, frontier):
        from .digest_attestation import _compute_digest
        return _compute_digest(blob,frontier)

    @classmethod
    def create(cls, model_contract_id, execution_profile_id, consumed, blob, frontier,
               input_identity='', execution_schedule=()):
        """Create and certify an exact owner after hashing its RAM bytes."""
        owner = cls(model_contract_id,execution_profile_id,consumed,blob,frontier,
                    '',input_identity,tuple(execution_schedule))
        _attest_factory_result(owner)
        return owner


@dataclass(frozen=True)
class PreparedCheckpoint:
    """Existing checkpoint bytes captured under an owner, ready for disk I/O.

    A transport object only: no native handle, authoritative history or promise
    of durability. The encoded manifest is immutable across writer threads.
    """
    snapshot: str
    checkpoint: Checkpoint
    envelope: bytes


@dataclass(frozen=True)
class Generation:
    snapshot_id: str
    accepted_tokens: tuple
    boundaries: tuple


class BranchingModel:
    # State/sampled requests support an early consumed work boundary.
    supports_consumed_pause = True

    def slot_queue(self, count):
        from .slots import BranchSlots
        return BranchSlots(self, count)

    @staticmethod
    def load_backend(profile):
        kind = profile.get('backend','llama.cpp')
        if kind == 'strata':
            from .strata_backend import StrataBackend
            return StrataBackend(profile)
        if kind != 'llama.cpp':
            raise IntegrityError('Unsupported backend profile')
        return NativeBackend(profile)
    durability_scope = MappingProxyType(dict(platform='local-posix',
        process_crash_tested=True,power_loss_tested=False,
        publication='fsync-file; atomic-link-or-rename; fsync-directory'))

    def __init__(self, directory, backend, profile):
        if profile['model_contract_id'] != backend.profile['model_contract_id'] or profile['execution_profile_id'] != backend.profile['execution_profile_id']:
            raise IntegrityError('Backend profile does not match requested contract')
        self.store = Store(directory)
        self.backend = backend
        self.profile = profile
        self.cache = backend.new_cache()
        self.checkpoint_directory = self.store.root/'checkpoints'
        _private_directory(self.checkpoint_directory)
        self._checkpoint_fault = lambda point: None
        self._lease_fault = lambda point: None
        self._allocation_fault = lambda point, lease: None
        architecture = backend.metadata('general.architecture')
        fields = dict(weights_sha256=profile['model_weights_sha256'],
            architecture=architecture,rope_base=backend.metadata(architecture+'.rope.freq_base'),
            yarn='model-default',tokenizer_identity=profile['model_weights_sha256'],
            template_identity='exact-token-ids/no-template',
            positions='absolute-zero-based',input_kind='token_ids',adapters=())
        fields.update(backend.contract_fields())
        self.contract = MappingProxyType(fields)

    def history(self, snapshot):
        if self.store.get_snapshot(snapshot)['schema'] == 2:
            return InputHistory(self.store.input_lineage(snapshot)[1])
        return tuple(self.store.lineage(snapshot)[1])

    def for_principal(self, principal):
        bound = copy.copy(self)
        bound.store = self.store.for_principal(principal)
        return bound

    def snapshot_bytes(self, snapshot):
        self.store.authorize(snapshot, 'read')
        return (self.store.objects/(snapshot+'.json')).read_bytes()

    def _save(self, lease, history, schedule=()):
        self._allocation_fault('before_native_export',lease)
        blob = self.backend.export(lease)
        frontier = self.backend.logits(lease)
        # Bound native-operator histories restore their owned COW image or exact
        # disk checkpoint; they never enter the legacy history-cache replay path.
        # A second deep host copy is neither canonical state nor a prerequisite
        # for this contract. Other profiles retain their existing cache behavior.
        if self.profile.get('execution',{}).get('resume_contract') not in ('identical-native-operators-v1',RESUME_CONTRACT):
            self.backend.park(lease)
        return Checkpoint.create(self.profile['model_contract_id'],self.profile['execution_profile_id'],
            self.backend.position(lease)+1,blob,frontier,self._input_identity(history),tuple(schedule))

    @contextmanager
    def materialize(self, snapshot):
        self.store.authorize(snapshot, 'materialize')
        with self.store.read_lease(snapshot):
            with self._materialize(snapshot) as state:
                yield state

    @contextmanager
    def _materialize(self, snapshot):
        record = self.store.get_snapshot(snapshot)
        if record['model_contract_id'] != self.profile['model_contract_id']:
            raise IntegrityError('Snapshot model contract mismatch')
        history = self.history(snapshot)
        input_identity = self._input_identity(history)
        with self.backend.lease() as lease:
            saved = self.cache.get(snapshot)
            if saved is None and (self.checkpoint_directory/(snapshot+'.json')).exists():
                saved = self._load_checkpoint(snapshot,history)
                self.cache[snapshot] = saved
            if saved is not None and saved.input_identity != input_identity:
                # The cache is only an accelerator. Never restore a same-sized
                # image for another canonical input history; rebuild from the
                # immutable snapshot inputs instead.
                self.cache.pop(snapshot, None)
                self.backend.clear_cached_histories()
                saved = None
            if saved:
                if saved.model_contract_id != self.profile['model_contract_id'] or saved.execution_profile_id != self.profile['execution_profile_id']:
                    raise IntegrityError('Checkpoint contract/profile mismatch')
                if saved.consumed != len(history) or saved.checksum != verified_checkpoint_digest(saved):
                    raise IntegrityError('Checkpoint integrity failure')
                if isinstance(history,InputHistory):
                    from .execution_receipt import validate_input_schedule
                    try:
                        validate_input_schedule(saved.execution_schedule,history,
                                                self.profile['execution']['prefill_chunk'])
                    except (ValueError,TypeError) as error:
                        raise IntegrityError('Mixed checkpoint execution binding is invalid') from error
                resident = self.backend.restore(lease,saved.blob,history=history,frontier=saved.frontier)
                if resident is not saved.blob:
                    saved = Checkpoint.create(saved.model_contract_id,saved.execution_profile_id,saved.consumed,
                        resident,saved.frontier,saved.input_identity,saved.execution_schedule)
                    self.cache[snapshot] = saved
            else:
                if self._schedule_contract() is not None:
                    raise IntegrityError('Exact batched materialization requires a retained numerical checkpoint')
                if not self.backend.restore_cached(lease,history):
                    self.backend.consume(lease,history)
                saved = self._save(lease,history)
                self.cache[snapshot] = saved
            if self.backend.position(lease) != len(history)-1:
                raise IntegrityError('Restored consumed position mismatch')
            yield lease,saved.frontier

    def create(self, ids, *, contract=None):
        snapshot, _ = self.create_with_reuse(ids, contract=contract)
        return snapshot

    def create_inputs(self, spans, *, artifacts, reencode=None):
        """Create one consumed mixed prefix using retained, leased artifacts."""
        snapshot,_ = self.create_with_reuse(InputHistory(spans),artifacts=artifacts,reencode=reencode)
        return snapshot

    @contextmanager
    def _input_artifacts(self, history, artifacts, reencode, activity=None):
        images = [span for span in history.spans if span['kind']=='image']
        if images and (artifacts is None or artifacts.store.root.resolve()!=self.store.root.resolve()):
            raise InvalidOperation('Image input requires artifacts owned by the canonical store')
        with ExitStack() as held:
            payloads = {}
            for span in images:
                item = held.enter_context(artifacts.lease(span,reencode=reencode))
                payloads[span['embedding_sha256']] = item.embedding
                if activity is not None:
                    activity.append(dict(original_sha256=span['original_sha256'],
                        embedding_sha256=item.embedding.sha256,embedding_bytes=item.embedding.nbytes,
                        rows=item.embedding.rows,source=item.source))
            yield payloads

    def create_with_reuse(self, ids, *, contract=None, materialization_loader=None,
                          artifacts=None, reencode=None):
        """Create an idle root, reusing only its validated exact materialization.

        The root identity is the existing content-addressed canonical snapshot;
        this adds no second cache key or operation-ID-based shortcut. The bool
        result is evidence for the coordinator's prefix-work check.
        """
        if self.store._principal is not None:
            raise PermissionError('Only the local owner can create model histories')
        if contract is not None and dict(contract) != dict(self.contract):
            raise IntegrityError('Unsupported or incompatible model/input semantics')
        mixed = self.profile.get('execution',{}).get('resume_contract') == RESUME_CONTRACT
        if mixed:
            if not isinstance(ids,InputHistory):
                ids = InputHistory([dict(kind='text',tokens=list(ids))])
            validate_boundary(ids,capacity=self.profile['execution']['capacity'],
                vocab=self.backend.vocab_size,width=self.profile['execution']['embedding_width'])
        else:
            if isinstance(ids,InputHistory):
                raise InvalidOperation('Pinned execution profile does not support mixed input')
            ids = tuple(ids)
            if (not ids or any(type(token) is not int or not 0 <= token < self.backend.vocab_size
                               for token in ids)):
                raise InvalidOperation('Invalid token IDs')

        # Canonicalize before computing so an already-materialized identical
        # root can be found by its existing snapshot identity. This object is
        # only immutable history; it does not imply a numerical result exists.
        sid = (self.store.create_input_root(self.profile['model_contract_id'],ids.spans,{'status':'idle'})
               if mixed else self.store.create_root(self.profile['model_contract_id'],ids,{'status':'idle'}))
        identity = self._input_identity(ids)
        loaded_from_disk = False
        # Protect both the canonical identity lookup and any disk validation
        # from GC. The coordinator loader also owns a longer lease through
        # worker capture; this nested short lease is safe and keeps standalone
        # callers equally protected.
        with self.store.read_lease(sid):
            record = self.store.get_snapshot(sid)
            if (record['parent'] is not None or record['model_contract_id'] != self.profile['model_contract_id']
                    or (InputHistory(record['inputs']) if mixed else tuple(record['inputs'])) != ids
                    or record['controller'] != {'status':'idle'}):
                raise IntegrityError('Canonical root differs from requested model/input/controller')
            if materialization_loader is not None:
                saved = materialization_loader(sid,ids)
                if saved is None:
                    self.cache.pop(sid,None)
                else:
                    loaded_from_disk = self.cache.get(sid) is not saved
                    self.cache[sid] = saved
            else:
                saved = self.cache.get(sid)
                if saved is None:
                    manifest = self.checkpoint_directory / (sid + '.json')
                    try:
                        info = manifest.lstat()
                    except FileNotFoundError:
                        info = None
                    if info is not None:
                        if not stat.S_ISREG(info.st_mode):
                            raise IntegrityError('Root checkpoint manifest is not a regular file')
                        # A present manifest is authoritative: corruption or a
                        # profile mismatch is never a reason to compute over it.
                        saved = self._load_checkpoint(sid,ids)
                        loaded_from_disk = True
            if saved is not None:
                if (not isinstance(saved,Checkpoint)
                        or saved.model_contract_id != self.profile['model_contract_id']
                        or saved.execution_profile_id != self.profile['execution_profile_id']
                        or saved.consumed != len(ids) or saved.input_identity != identity
                        or type(saved.frontier) is not tuple or len(saved.frontier) != self.backend.vocab_size
                        or not all(map(math.isfinite,saved.frontier))
                        or verified_checkpoint_digest(saved) != saved.checksum):
                    raise IntegrityError('Existing root materialization differs from exact input/profile')
                schedule_contract = self._schedule_contract()
                if schedule_contract is not None:
                    schema, validate = schedule_contract
                    try:
                        digest = validate(saved.execution_schedule,ids,self.profile['execution']['prefill_chunk'])
                    except (TypeError,ValueError) as error:
                        raise IntegrityError('Existing root operator schedule is invalid') from error
                    # _load_checkpoint validates its manifest binding; a resident
                    # image has no manifest read, so validate its schedule here.
                    if schema not in (2,3,4) or not digest:
                        raise IntegrityError('Existing root operator schedule is invalid')
                if loaded_from_disk:
                    self.cache[sid] = saved
        if saved is not None:
            return sid, True

        with self.store.read_lease(sid), ExitStack() as held:
            payloads = held.enter_context(self._input_artifacts(ids,artifacts,reencode)) if mixed else None
            with self.backend.lease() as lease:
                if mixed:
                    self.backend.consume_inputs(lease,ids.spans,embeddings=payloads)
                else:
                    self.backend.consume(lease,ids)
                schedule = self._completed_schedule((),ids)
                saved = self._save(lease,ids,schedule)
        self.cache[sid] = saved
        return sid, False

    def feed(self, snapshot, ids):
        return self._feed(snapshot,ids,self.controller(snapshot))

    def feed_inputs(self,snapshot,spans,*,artifacts=None,reencode=None):
        """Append leased model inputs to one existing canonical mixed history."""
        self.store.authorize(snapshot,'append')
        if (self.profile.get('execution',{}).get('resume_contract')!=RESUME_CONTRACT
                or not callable(getattr(self.backend,'consume_inputs',None))):
            raise InvalidOperation('Pinned backend does not support mixed input deltas')
        delta=InputHistory(spans)
        with self.store.read_lease(snapshot):
            prior=self.history(snapshot)
            if not isinstance(prior,InputHistory):
                raise InvalidOperation('Image deltas require canonical mixed history')
            if not delta:return snapshot
            history=prior+delta
            validate_boundary(history,capacity=self.profile['execution']['capacity'],
                vocab=self.backend.vocab_size,width=self.profile['execution']['embedding_width'])
            # Image rows are model input cells, never sampled text. Feeding
            # actual text retains the existing declared controller semantics.
            # Detach once per batch, including after a completed turn where
            # the saved policy may hold a full-vocabulary allowed-token list.
            decoder=controller.consume_many(self.controller(snapshot),
                ((token,self.backend.piece(token),self.backend.is_eog(token)) for token in delta.text_tokens))
            with self._input_artifacts(delta,artifacts,reencode) as payloads:
                with self.materialize(snapshot) as (lease,_):
                    self.backend.consume_inputs(lease,delta.spans,embeddings=payloads)
                    self._lease_fault('after_token_consume')
                    saved=self._save(lease,history,self._completed_schedule(self.cache[snapshot].execution_schedule,history))
                return self._publish_feed(snapshot,delta,decoder,saved)

    def run_greedy_request(self, snapshot, ids, max_steps, *, stream, on_token=None, cancelled=None,
                           artifacts=None,reencode=None):
        """One owned native request; progress is provisional until boundary publication.

        The returned snapshot is canonical consumed state, not a disk-save
        acknowledgment. Call save_branch for durable publication and a CAS move.
        General sampling and the legacy per-token snapshot stream are separate.
        """
        if cancelled is not None and not callable(cancelled):
            raise InvalidOperation('Cancellation check must be callable')

        def check_cancelled():
            if cancelled is not None and cancelled():
                raise CancelledError()

        check_cancelled()
        started = time.monotonic()
        self.store.authorize(snapshot,'append')
        execution = self.profile.get('execution',{})
        if (execution.get('resume_contract') not in ('identical-native-operators-v1',RESUME_CONTRACT)
                or execution.get('generated_operator') != 'normal-verifier-decode'
                or execution.get('native_request_protocol') != 'consumed-greedy-ack-v1'
                or not callable(getattr(self.backend,'generate_request',None))):
            raise InvalidOperation('Resident greedy request requires the pinned native operator contract')
        mixed_delta=isinstance(ids,InputHistory)
        if mixed_delta:
            if (execution.get('resume_contract')!=RESUME_CONTRACT
                    or not callable(getattr(self.backend,'consume_inputs',None))):
                raise InvalidOperation('Pinned backend does not support mixed input deltas')
        else:ids=tuple(ids)
        if ((not mixed_delta and any(type(t) is not int or not 0 <= t < self.backend.vocab_size for t in ids))
                or type(max_steps) is not int or max_steps <= 0):
            raise InvalidOperation('Invalid resident request tokens or output bound')
        with self.store.read_lease(snapshot),ExitStack() as held:
            check_cancelled()
            history = self.history(snapshot)
            if mixed_delta and not isinstance(history,InputHistory):
                raise InvalidOperation('Image deltas require canonical mixed history')
            if len(history)+len(ids)+max_steps > execution['capacity']:
                raise InvalidOperation('Pinned execution context exceeded')
            prompt_history = history+ids
            if mixed_delta:
                validate_boundary(prompt_history,capacity=execution['capacity'],vocab=self.backend.vocab_size,
                    width=execution['embedding_width'])
            decoder = controller._validated_state(self.controller(snapshot))
            if decoder.get('schema') == 2:
                raise InvalidOperation('Use the explicit grammar-first request for this controller')
            if ids or decoder['status'] == 'idle':
                decoder = controller.start(dict(temperature=0,eos_tokens=execution['eos_tokens']),
                                           stream,self._text_history(prompt_history),self.backend.vocab_size)
            else:
                policy = decoder['policy']
                if (decoder['status'] != 'active' or policy['temperature'] != 0
                        or policy['repetition_penalty'] != 1 or policy['grammar'] is not None
                        or policy['stop_sequences'] or policy['rejection_threshold'] != 1
                        or policy['allowed'] != list(range(self.backend.vocab_size))
                        or policy['eos_tokens'] != execution['eos_tokens']):
                    raise InvalidOperation('Native request cannot resume this controller policy')
            payloads=held.enter_context(self._input_artifacts(ids,artifacts,reencode)) if mixed_delta else None
            with self._materialize(snapshot) as (lease,_):
                check_cancelled()
                schedule = self.cache[snapshot].execution_schedule
                if ids:
                    if mixed_delta:self.backend.consume_inputs(lease,ids.spans,embeddings=payloads)
                    else:self.backend.consume(lease,ids)
                    schedule = self._completed_schedule(schedule,prompt_history)
                generation_started = time.monotonic()
                check_cancelled()
                outputs = []
                with self.backend.generate_request(lease,max_steps) as request:
                    for token in request:
                        if len(outputs)>=max_steps:
                            raise InvalidOperation('Native output exceeded the requested token bound')
                        outputs.append(token)
                        if on_token is not None:
                            on_token(token)
                        # The iterator yields only at its consumed boundary. If
                        # cancelled here, leaving its context sends native
                        # cancel/ack before releasing the lease, without saving
                        # this provisional request as a canonical snapshot.
                        check_cancelled()
                generation_seconds = time.monotonic()-generation_started
                consumed_history = prompt_history+tuple(outputs)
                schedule = self._completed_schedule(schedule,consumed_history)
                accepted_before=decoder['accepted_in_scope']
                decoder = controller.consume_many(decoder,
                    ((token,self.backend.piece(token),self.backend.is_eog(token)) for token in outputs))
                if decoder['accepted_in_scope']-accepted_before!=len(outputs):
                    raise InvalidOperation('Native output extends beyond the stopped controller')
                self._lease_fault('after_token_consume')
                saved = self._save(lease,consumed_history,schedule)
                if saved.consumed != len(consumed_history):
                    raise IntegrityError('Resident request capture omitted consumed inputs')
            sid = self._publish_feed(snapshot,ids+tuple(outputs),decoder,saved)
        return dict(snapshot=sid,tokens=outputs,generation_seconds=generation_seconds,
                    request_seconds=time.monotonic()-started)

    def run_state_request(self, snapshot, ids, max_steps, *, engine, state, stream,
                          resume=False, on_token=None, cancelled=None, sampling=None,
                          reseed_seed=None, text_stops=None,artifacts=None,reencode=None,paused=None):
        """Host-selected grammar, legal-token greedy selection, one owned lease.

        The grammar controls output syntax only. The caller validates semantics,
        authorizes tools and executes them. Progress remains provisional until
        the final consumed-state snapshot is published; save_branch provides
        the separate durable branch acknowledgment. A bounded partial result
        can resume its grammar from canonical history with resume=True.
        """
        from .grammar_first import GrammarEngine
        from .sampled_controller import PlainSamplingSession, start as start_sampled
        from .min_p import validate_policy, supports_policy
        started = time.monotonic()
        plain=engine is None
        if reseed_seed is not None:
            from .seed_controls import seed_stream
            seed_stream(reseed_seed)
            if not resume:
                raise InvalidOperation('A controller stream transition requires resume')
        if (plain and (state is not None or (sampling is None and not resume))) or (not plain and not isinstance(engine, GrammarEngine)):
            raise InvalidOperation('A checked grammar engine is required')
        supplied_sampling=None
        if sampling is not None:
            try:
                supplied=dict(sampling)
                supplied.setdefault('repetition_penalty',1.0)
                supplied_sampling=validate_policy(supplied)
            except (TypeError,ValueError) as exc:
                raise InvalidOperation('Unsupported Min-P request policy') from exc
        from .text_stops import normalize_stop_sequences
        if text_stops is not None:
            text_stops = normalize_stop_sequences(text_stops)
        if type(resume) is not bool or (cancelled is not None and not callable(cancelled)):
            raise InvalidOperation('Invalid grammar request controls')
        if on_token is not None and not callable(on_token):
            raise InvalidOperation('Progress callback must be callable')
        if paused is not None and not callable(paused):
            raise InvalidOperation('Pause callback must be callable')

        def check_cancelled():
            if cancelled is not None and cancelled():
                raise CancelledError()

        check_cancelled()
        self.store.authorize(snapshot, 'append')
        execution = self.profile.get('execution', {})
        mixed=execution.get('resume_contract')==RESUME_CONTRACT
        if mixed:
            from .vision_controllers import require_basic_sampling
            # VC07 deferred: mixed text stops require the canonical pending UTF8
            # and stop-watcher controller to survive fork/restart. Image rows
            # must never enter decoder text; retain this refusal until tested.
            if text_stops:raise InvalidOperation('Mixed-input text stops are not yet qualified')
            if plain or sampling is not None or reseed_seed is not None:
                require_basic_sampling(execution,self.backend,supplied_sampling)
        if (execution.get('resume_contract') not in ('identical-native-operators-v1',RESUME_CONTRACT)
                or execution.get('generated_operator') != 'normal-verifier-decode'
                or execution.get('selected_token_protocol') != 'consumed-selected-receipt-v1'
                or not callable(getattr(self.backend, 'consume_generated', None))
                or (not plain and list(engine.eos_tokens) != execution.get('eos_tokens'))):
            raise InvalidOperation('Grammar request requires the pinned native operator and stop tokens')
        validate_grammar = getattr(self.backend, 'validate_grammar_engine', None)
        if not plain and not callable(validate_grammar):
            raise InvalidOperation('Backend cannot bind a grammar tokenizer to its model')
        if not plain:validate_grammar(engine)
        selection = execution.get('grammar_selection_protocol')
        if selection is not None and (selection != 'finite-mask-argmax-v1'
                or not callable(getattr(self.backend, 'choose_legal', None))):
            raise InvalidOperation('Unsupported native grammar selection protocol')
        mixed_delta=isinstance(ids,InputHistory)
        if mixed_delta:
            if not mixed or not callable(getattr(self.backend,'consume_inputs',None)):
                raise InvalidOperation('Pinned backend does not support mixed grammar input deltas')
        else:ids=tuple(ids)
        if ((not mixed_delta and any(type(t) is not int or not 0 <= t < self.backend.vocab_size for t in ids))
                or type(max_steps) is not int or max_steps <= 0 or (resume and ids)):
            raise InvalidOperation('Invalid grammar request inputs or resume suffix')
        with self.store.read_lease(snapshot),ExitStack() as held:
            history = self.history(snapshot)
            if mixed_delta and not isinstance(history,InputHistory):
                raise InvalidOperation('Image deltas require canonical mixed history')
            if len(history) + len(ids) + max_steps > execution['capacity']:
                raise InvalidOperation('Pinned execution context exceeded')
            prompt_history = history + ids
            if isinstance(prompt_history,InputHistory):
                validate_boundary(prompt_history,capacity=execution['capacity'],vocab=self.backend.vocab_size,
                    width=execution['embedding_width'])
            # Parser/repetition progress counts actual text, never image pad
            # rows. Native receipts still bind every mixed input cell.
            controller_history=self._text_history(history)
            prompt_controller_history=self._text_history(prompt_history)
            if resume:
                decoder = controller._validated_state(self.controller(snapshot))
                if mixed and decoder.get('schema')!=2:
                    if decoder.get('schema') not in (1,3,4):
                        raise InvalidOperation('Mixed-input sampler/controller is unavailable; use qualified greedy controls')
                    require_basic_sampling(execution,self.backend,decoder['policy'])
                saved_stops = (tuple(decoder['text_stop']['stop_sequences'])
                               if decoder.get('schema') in (5, 6) else ())
                if text_stops is not None and text_stops != saved_stops:
                    raise InvalidOperation('Text stops cannot change while resuming a generation')
                if supplied_sampling is not None and decoder.get('schema') in (1,2):
                    from .sampled_controller import sampler_order
                    # An explicit generation-boundary opt-in changes only the
                    # policy; existing RNG/parser/decoder progress is retained.
                    decoder = dict(decoder, schema=3, constraint=decoder.get('constraint'))
                    decoder['policy'] = dict(decoder['policy'], **supplied_sampling)
                    decoder['sampler_order'] = sampler_order(supplied_sampling)
                    from .min_p import xtc_enabled, mirostat_enabled, adaptive_enabled
                    if xtc_enabled(supplied_sampling):
                        decoder['schema'] = 4
                        decoder['sampler_state'] = None
                    elif mirostat_enabled(supplied_sampling):
                        from .mirostat2 import start_state
                        decoder['schema'] = 4
                        decoder['sampler_state'] = start_state(supplied_sampling['mirostat_tau'])
                    elif adaptive_enabled(supplied_sampling):
                        from .adaptive_p import initial_state
                        decoder['schema'] = 4
                        decoder['sampler_state'] = initial_state(
                            supplied_sampling['adaptive_target'], supplied_sampling['adaptive_decay'])
                if plain:
                    if decoder['policy']['eos_tokens'] != execution['eos_tokens']:
                        raise InvalidOperation('Saved sampling stop tokens differ from the execution profile')
                    session=PlainSamplingSession.restore(decoder,controller_history,self.backend.vocab_size,
                        self.backend.piece,self.backend.is_eog)
                else:
                    session = engine.restore(decoder, controller_history)
                if not plain and decoder['constraint']['agent_state'] != state:
                    raise InvalidOperation('Resume cannot change the host-selected agent state')
                if supplied_sampling is not None:
                    if decoder.get('schema') not in (3,4,5,6):
                        raise InvalidOperation('Sampler override requires a sampled controller')
                    session.configure_generation(supplied_sampling)
                    decoder = session.current_controller
                if reseed_seed is not None:
                    from .seed_controls import reseed_controller
                    decoder = reseed_controller(decoder,seed=reseed_seed,
                        history=controller_history,vocab_size=self.backend.vocab_size)
                    session=(PlainSamplingSession.restore(decoder,controller_history,self.backend.vocab_size,
                        self.backend.piece,self.backend.is_eog) if plain else engine.restore(decoder,controller_history))
            else:
                if plain:
                    decoder=start_sampled(supplied_sampling,stream,prompt_controller_history,self.backend.vocab_size,
                                          execution['eos_tokens'],text_stops=text_stops or ())
                    session=PlainSamplingSession.restore(decoder,prompt_controller_history,self.backend.vocab_size,
                        self.backend.piece,self.backend.is_eog)
                else:
                    decoder = engine.start(state, stream, prompt_controller_history, self.backend.vocab_size,
                                           sampling=supplied_sampling,text_stops=text_stops or ())
                    session = engine.restore(decoder, prompt_controller_history)
            sampled=decoder.get('schema') in (3,4,5,6)
            protocol = execution.get('sampling_protocol')
            proposal_sampled = decoder.get('schema') in (4,6)
            if proposal_sampled:
                if mixed:
                    from .vision_controllers import require_proposal_sampling
                    require_proposal_sampling(execution,self.backend,decoder['policy'])
                from .sampling_proposal import proposal_protocol_supported
                if (not proposal_protocol_supported(protocol, decoder['policy'])
                        or not callable(getattr(self.backend,'sample_selection_proposal',None))):
                    raise InvalidOperation('Schema-4 sampler requires its pinned native selection-proposal protocol')
            elif sampled and (not supports_policy(protocol, decoder['policy'])
                    or not callable(getattr(self.backend,'sample_legal',None))):
                raise InvalidOperation('Min-P requires the pinned full-vocabulary native sampler')
            native_selection=sampled or selection is not None
            full_mask=None
            if plain:
                width=(self.backend.vocab_size+7)//8
                mask=bytearray(b'\xff'*width)
                if self.backend.vocab_size%8:mask[-1]&=(1<<(self.backend.vocab_size%8))-1
                full_mask=bytes(mask)
            if session.is_stopped():
                raise InvalidOperation('Grammar scope has already stopped')
            check_cancelled()
            payloads=held.enter_context(self._input_artifacts(ids,artifacts,reencode)) if mixed_delta else None
            check_cancelled()
            with self._materialize(snapshot) as (lease, frontier):
                schedule = self.cache[snapshot].execution_schedule
                if ids:
                    if mixed_delta:self.backend.consume_inputs(lease,ids.spans,embeddings=payloads)
                    else:self.backend.consume(lease, ids)
                    schedule = self._completed_schedule(schedule, prompt_history)
                    if not native_selection:
                        frontier = self.backend.logits(lease)
                schedule_cursor = None
                if isinstance(prompt_history, InputHistory):
                    from .execution_receipt import _InputScheduleCursor
                    try:
                        schedule_cursor = _InputScheduleCursor(schedule,prompt_history,execution['prefill_chunk'])
                    except (ValueError,TypeError) as error:
                        raise IntegrityError('Mixed request execution schedule is invalid') from error
                outputs = []
                consumed_history = prompt_history
                generation_started = time.monotonic()
                for _ in range(max_steps):
                    check_cancelled()
                    if sampled:
                        if proposal_sampled:
                            select=lambda mask,temp,p,**filters:self.backend.sample_selection_proposal(
                                lease,mask,temp,p,**filters)
                            token=(session.choose(full_mask,select) if plain else session._choose_proposal(select))
                        else:
                            select=lambda mask,temp,p,draw,**filters:self.backend.sample_legal(lease,mask,temp,p,draw,**filters)
                            token=(session.choose(full_mask,select) if plain else session._choose_min_p(select))
                    else:
                        token = (session._choose_masked(lambda mask: self.backend.choose_legal(lease, mask))
                                 if selection is not None else session.choose(frontier))
                    # Advance only private controller/parser state. No snapshot
                    # is published unless native ingestion also succeeds.
                    piece = self.backend.piece(token)
                    is_eog = self.backend.is_eog(token)
                    if proposal_sampled:
                        # A schema-4 proposal's cursor/feedback is accepted only
                        # after the native selected-token acknowledgment.
                        self.backend.consume_generated(lease, (token,))
                        session._consume_selected(token, piece, is_eog)
                    else:
                        session._consume_selected(token, piece, is_eog)
                        self.backend.consume_generated(lease, (token,))
                    consumed_history += (token,)
                    if schedule_cursor is None:
                        schedule = self._completed_schedule(schedule, consumed_history)
                    else:
                        try:
                            schedule_cursor.append(self.backend.last_execution_receipt,consumed_history)
                        except (ValueError,TypeError) as error:
                            raise IntegrityError('Native execution extension does not match consumed history') from error
                    outputs.append(token)
                    if on_token is not None:
                        on_token(token)
                    check_cancelled()
                    # A pause is a smaller work bound, never destructive
                    # cancellation. Consume at least one token, keep its full
                    # controller/parser state, then capture/publish normally.
                    # Publication failures remain failures, not pause receipts.
                    if session.is_stopped() or (paused is not None and paused()):
                        break
                    if not native_selection:
                        frontier = self.backend.logits(lease)
                generation_seconds = time.monotonic() - generation_started
                decoder = session.current_controller
                if schedule_cursor is not None:
                    try:
                        schedule = schedule_cursor.finish()
                    except (ValueError,TypeError) as error:
                        raise IntegrityError('Mixed request publication schedule is invalid') from error
                self._lease_fault('after_token_consume')
                saved = self._save(lease, consumed_history, schedule)
                if saved.consumed != len(consumed_history):
                    raise IntegrityError('Grammar request capture omitted consumed inputs')
            sid = self._publish_feed(snapshot, ids + tuple(outputs), decoder, saved)
        return dict(snapshot=sid, tokens=outputs, grammar_complete=session.is_complete(),
                    stopped=session.is_stopped(), generation_seconds=generation_seconds,
                    request_seconds=time.monotonic() - started)

    def run_sampled_request(self,snapshot,ids,max_steps,*,sampling,stream,resume=False,
                            on_token=None,cancelled=None,reseed_seed=None,text_stops=None,artifacts=None,reencode=None,paused=None):
        """Plain Min-P under the same consumed-state and branch-owned contract."""
        return self.run_state_request(snapshot,ids,max_steps,engine=None,state=None,
            stream=stream,sampling=sampling,resume=resume,on_token=on_token,cancelled=cancelled,paused=paused,
            reseed_seed=reseed_seed,text_stops=text_stops,artifacts=artifacts,reencode=reencode)

    def _feed(self, snapshot, ids, decoder, *, generated=False):
        self.store.authorize(snapshot, 'append')
        if not ids:
            return snapshot
        history,next_decoder = self._prepare_feed(snapshot,ids,decoder)
        with self.materialize(snapshot) as (lease,_):
            saved = self._consume_feed(lease,snapshot,ids,history,generated=generated)
        return self._publish_feed(snapshot,ids,next_decoder,saved)

    def _prepare_feed(self, snapshot, ids, decoder):
        history = self.history(snapshot) + tuple(ids)
        next_decoder = copy.deepcopy(decoder)
        for token in ids:
            if type(token) is not int or not 0<=token<self.backend.vocab_size:
                raise InvalidOperation('Invalid token ID')
            next_decoder = controller.consume(next_decoder,token,self.backend.piece(token),
                self.backend.is_eog(token))
        return history,next_decoder

    def _consume_feed(self, lease, snapshot, ids, history, *, generated):
        previous_schedule = self.cache[snapshot].execution_schedule
        consume = (self.backend.consume_generated if generated and
            hasattr(self.backend, 'consume_generated') else self.backend.consume)
        consume(lease,tuple(ids))
        self._lease_fault('after_token_consume')
        return self._save(lease,history,self._completed_schedule(previous_schedule,history))

    def _publish_feed(self, snapshot, ids, next_decoder, saved):
        spans=ids.spans if isinstance(ids,InputHistory) else [dict(kind='text',tokens=list(ids))]
        sid = (self.store.append_inputs(snapshot,spans,next_decoder)
               if self.store.get_snapshot(snapshot)['schema']==2
               else self.store.append(snapshot,ids,next_decoder))
        self.cache[sid] = saved
        return sid

    def controller(self, snapshot):
        return self.store.get_snapshot(snapshot)['controller']

    def _schedule_contract(self):
        from .execution_receipt import validate_schedule, validate_operator_schedule, validate_input_schedule
        contract = self.profile.get('execution',{}).get('resume_contract')
        if contract == 'identical-native-batches-v1':
            return 2, validate_schedule
        if contract == 'identical-native-operators-v1':
            return 3, validate_operator_schedule
        if contract == RESUME_CONTRACT:
            return 4, validate_input_schedule
        return None

    def _completed_schedule(self, prior, history):
        contract = self._schedule_contract()
        if contract is None:
            return ()
        _, validate = contract
        receipt = getattr(self.backend,'last_execution_receipt',None)
        schedule = tuple(prior)+(copy.deepcopy(receipt),)
        try:
            validate(schedule,history,self.profile['execution']['prefill_chunk'])
        except (ValueError,TypeError) as error:
            raise IntegrityError('Native execution schedule does not match consumed history') from error
        return schedule

    def _controller_transition(self, snapshot, decoder):
        sid = (self.store.append_inputs(snapshot,(),decoder)
               if self.store.get_snapshot(snapshot)['schema']==2 else self.store.append(snapshot,(),decoder))
        saved = self.cache.get(snapshot)
        if saved is not None:
            self.cache[sid] = saved
        return sid

    def start_generation(self, snapshot, policy, stream):
        decoder = controller.start(policy,stream,self._text_history(self.history(snapshot)),self.backend.vocab_size)
        return self._controller_transition(snapshot,decoder)

    @staticmethod
    def _text_history(history):
        return history.text_tokens if isinstance(history,InputHistory) else history

    def set_random_stream(self, snapshot, stream, *, grammar_engine=None):
        from .seed_controls import validate_reseed_source
        source_controller = self.controller(snapshot)
        history=self.history(snapshot)
        if isinstance(history,InputHistory):
            from .vision_controllers import require_basic_sampling
            require_basic_sampling(self.profile.get('execution',{}),self.backend,source_controller.get('policy'))
        if source_controller.get('status') == 'idle':
            raise InvalidOperation('No generation scope')
        if grammar_engine is not None:
            execution = self.profile.get('execution', {})
            validator = getattr(self.backend, 'validate_grammar_engine', None)
            if (execution.get('resume_contract') not in ('identical-native-operators-v1',RESUME_CONTRACT)
                    or execution.get('generated_operator') != 'normal-verifier-decode'
                    or execution.get('selected_token_protocol') != 'consumed-selected-receipt-v1'
                    or list(grammar_engine.eos_tokens) != execution.get('eos_tokens')
                    or not callable(validator)
                    or not callable(getattr(self.backend, 'consume_generated', None))):
                raise InvalidOperation('Grammar reseed requires its pinned execution profile')
            validator(grammar_engine)
        decoder = validate_reseed_source(
            source_controller, history=history.text_tokens if isinstance(history,InputHistory) else history,
            vocab_size=self.backend.vocab_size, token_bytes=self.backend.piece,
            is_eog=self.backend.is_eog, grammar_engine=grammar_engine)
        decoder['stream_key'] = hashlib.sha256(stream.encode()).hexdigest()
        decoder['next_draw'] = 0
        return self._controller_transition(snapshot,decoder)

    def stream(self, snapshot, max_steps):
        if type(max_steps) is not int or max_steps<0:
            raise InvalidOperation('Invalid generation work bound')
        for _ in range(max_steps):
            decoder = self.controller(snapshot)
            if decoder['status'] == 'stopped':
                break
            self.store.authorize(snapshot,'append')
            with self.materialize(snapshot) as (lease,frontier):
                token,next_decoder = controller.choose(frontier,decoder)
                history,next_decoder = self._prepare_feed(snapshot,(token,),next_decoder)
                saved = self._consume_feed(lease,snapshot,(token,),history,generated=True)
            snapshot = self._publish_feed(snapshot,(token,),next_decoder,saved)
            # Store.append has fsynced and published the canonical object.
            yield snapshot,token

    def generate(self, snapshot, max_steps):
        boundaries,tokens = [],[]
        for snapshot,token in self.stream(snapshot,max_steps):
            boundaries.append(snapshot)
            tokens.append(token)
        return Generation(snapshot,tuple(tokens),tuple(boundaries))

    def next_logits(self, snapshot):
        with self.materialize(snapshot) as (_,frontier):
            return frontier

    def consumed(self, snapshot):
        with self.materialize(snapshot) as (lease,_):
            return self.backend.position(lease)+1

    def position_max(self, snapshot):
        return self.consumed(snapshot)-1

    @staticmethod
    def _input_identity(history):
        if isinstance(history,InputHistory):
            return history.digest
        return hashlib.sha256(_json_bytes(list(history))).hexdigest()

    @staticmethod
    def _checkpoint_position_binding(history):
        if isinstance(history,InputHistory):
            return dict(positions=POSITION_CONVENTION,**input_binding(history))
        return dict(positions='absolute-zero-based')

    @contextmanager
    def _checkpoint_stage(self):
        """Keep staging inode ownership until publication or failure cleanup.

        Creation/locking shares the Store lock with disk collection, closing
        the create-before-flock race. Kernel locks disappear on process death;
        live file locks also survive the flush/rename publication boundary.
        """
        with self.store._locked():
            fd,name=tempfile.mkstemp(prefix='.stage-',dir=self.checkpoint_directory)
            stage=Path(name)
            try:
                fcntl.flock(fd,fcntl.LOCK_EX)
                output=os.fdopen(fd,'wb')
            except BaseException:
                os.close(fd);stage.unlink(missing_ok=True);raise
        try:yield output,stage
        finally:
            # Keep the file lock until cleanup and never unlink a replacement
            # that acquired the old staging pathname after rename/unlink.
            try:
                try:current=stage.lstat()
                except FileNotFoundError:pass
                else:
                    owned=os.fstat(output.fileno())
                    if (current.st_dev,current.st_ino)==(owned.st_dev,owned.st_ino):stage.unlink()
            finally:output.close()

    def _publish_checkpoint_file(self, destination, data):
        kind = 'payload' if destination.suffix == '.state' else 'manifest'
        with self._checkpoint_stage() as (output,stage):
            middle = max(1, len(data)//2)
            output.write(data[:middle])
            output.flush()
            self._checkpoint_fault(kind+'_during_write')
            output.write(data[middle:])
            output.flush()
            os.fsync(output.fileno())
            self._checkpoint_fault(kind+'_before_publish')
            os.replace(stage,destination)
            _sync_dir(self.checkpoint_directory)
            self._checkpoint_fault(kind+'_durable')

    def park_checkpoint(self, snapshot):
        """Retain the existing native serialization in the bounded host cache.

        This is optional derived state, not a new canonical snapshot or a
        durable save. Native serializers may use temporary files; their actual
        I/O counters remain visible. No history/controller/reference is moved.
        """
        started = time.monotonic()
        self.store.authorize(snapshot,'materialize')
        if not callable(getattr(self.backend,'work_counters',None)):
            raise IntegrityError('Host parking requires native forward-work instrumentation')
        with self.store.read_lease(snapshot):
            before = self.backend.work_counters()
            if type(before.get('forward_tokens')) is not int:
                raise IntegrityError('Missing native forward-work instrumentation')
            with self._materialize(snapshot) as (lease,_):
                saved = self.cache[snapshot]
                payload = self.backend.serialize(lease,saved.blob)
                if type(payload) is not bytes or not payload:
                    raise IntegrityError('Native serialization must return complete immutable bytes')
                after = self.backend.work_counters()
                if (type(after.get('forward_tokens')) is not int
                        or after['forward_tokens'] != before['forward_tokens']):
                    raise IntegrityError('Host parking performed unexpected forward computation')
                parked = Checkpoint.create(saved.model_contract_id,saved.execution_profile_id,saved.consumed,
                    payload,saved.frontier,saved.input_identity,saved.execution_schedule)
                # Admission may reject the payload; the existing hot entry is
                # unchanged until the cache accepts the complete replacement.
                self.cache[snapshot] = parked
            from .execution_receipt import schedule_digest
            return dict(snapshot=snapshot,tier='host-serialized',state_bytes=len(payload),
                state_sha256=hashlib.sha256(payload).hexdigest(),consumed=saved.consumed,
                execution_profile_id=saved.execution_profile_id,
                schedule_sha256=schedule_digest(saved.execution_schedule) if saved.execution_schedule else None,
                forward_tokens=after['forward_tokens']-before['forward_tokens'],
                seconds=time.monotonic()-started,
                counter_deltas={key:after[key]-before[key] for key in after
                    if type(after[key]) in (int,float) and type(before.get(key)) in (int,float)})

    def checkpoint(self, snapshot):
        self.store.authorize(snapshot, 'checkpoint')
        with self.materialize(snapshot) as (lease,_):
            saved = self.cache[snapshot]
            blob = self.backend.serialize(lease,saved.blob)
        payload = blob+struct.pack('<'+str(len(saved.frontier))+'f',*saved.frontier)
        payload_hash = hashlib.sha256(payload).hexdigest()
        payload_path = self.checkpoint_directory/(payload_hash+'.state')
        self._publish_checkpoint_file(payload_path,payload)
        manifest = dict(schema=1,snapshot=snapshot,input_identity=self._input_identity(self.history(snapshot)),
            model_contract_id=saved.model_contract_id,execution_profile_id=saved.execution_profile_id,
            consumed=saved.consumed,**self._checkpoint_position_binding(self.history(snapshot)),state_bytes=len(blob),
            logits_count=len(saved.frontier),payload_sha256=payload_hash)
        contract = self._schedule_contract()
        if contract is not None:
            schema, validate = contract
            digest = validate(saved.execution_schedule,self.history(snapshot),
                                       self.profile['execution']['prefill_chunk'])
            manifest.update(schema=schema,execution_schedule=list(saved.execution_schedule),schedule_sha256=digest)
        envelope = dict(manifest=manifest,checksum=hashlib.sha256(_json_bytes(manifest)).hexdigest())
        self._checkpoint_fault('before_manifest_publish')
        manifest_path = self.checkpoint_directory/(snapshot+'.json')
        self._publish_checkpoint_file(manifest_path,_json_bytes(envelope))
        return dict(manifest_path=manifest_path,payload_path=payload_path)

    def _prepared_envelope(self, snapshot, cold, history):
        if (type(cold.blob) is not bytes or not cold.blob or not cold.frontier
                or not all(map(math.isfinite,cold.frontier))):
            raise IntegrityError('Incomplete prepared numerical checkpoint')
        if (cold.model_contract_id != self.profile['model_contract_id']
                or cold.execution_profile_id != self.profile['execution_profile_id']
                or cold.consumed != len(history)
                or cold.input_identity != self._input_identity(history)):
            raise IntegrityError('Prepared checkpoint does not match canonical history/profile')
        checksum = verified_checkpoint_digest(cold)
        if checksum != cold.checksum:
            raise IntegrityError('Prepared checkpoint checksum mismatch')
        manifest = dict(schema=1,snapshot=snapshot,input_identity=cold.input_identity,
            model_contract_id=cold.model_contract_id,execution_profile_id=cold.execution_profile_id,
            consumed=cold.consumed,**self._checkpoint_position_binding(history),state_bytes=len(cold.blob),
            logits_count=len(cold.frontier),payload_sha256=checksum)
        contract = self._schedule_contract()
        if contract is not None:
            schema,validate = contract
            digest = validate(cold.execution_schedule,history,self.profile['execution']['prefill_chunk'])
            manifest.update(schema=schema,execution_schedule=list(cold.execution_schedule),schedule_sha256=digest)
        envelope = dict(manifest=manifest,checksum=hashlib.sha256(_json_bytes(manifest)).hexdigest())
        return _json_bytes(envelope)

    def _prepared_checkpoint(self, snapshot, saved, blob):
        if type(blob) is not bytes or not blob or not all(map(math.isfinite,saved.frontier)):
            raise IntegrityError('Incomplete prepared numerical checkpoint')
        history = self.history(snapshot)
        cold = Checkpoint.create(saved.model_contract_id,saved.execution_profile_id,saved.consumed,
            blob,saved.frontier,saved.input_identity,saved.execution_schedule)
        return PreparedCheckpoint(snapshot,cold,self._prepared_envelope(snapshot,cold,history))

    def prepare_checkpoint(self, snapshot):
        """Owner-only native capture. Callers reserve RAM before entering here."""
        self.store.authorize(snapshot,'checkpoint')
        with self.materialize(snapshot) as (lease,_):
            saved = self.cache[snapshot]
            before = self.backend.work_counters()
            blob = self.backend.serialize(lease,saved.blob)
            after = self.backend.work_counters()
            if after['consumed_tokens'] != before['consumed_tokens']:
                raise IntegrityError('Preparing a checkpoint recomputed saved tokens')
            return self._prepared_checkpoint(snapshot,saved,blob)

    def _publish_checkpoint_parts(self, destination, parts):
        """Same atomic publication, without concatenating a second full blob."""
        total = sum(map(len,parts))
        count,signalled = 0,False
        with self._checkpoint_stage() as (output,stage):
            for part in parts:
                view = memoryview(part)
                for start in range(0,len(view),1024*1024):
                    chunk = view[start:start+1024*1024]
                    output.write(chunk)
                    count += len(chunk)
                    if not signalled and count >= max(1,total//2):
                        output.flush()
                        self._checkpoint_fault('payload_during_write')
                        signalled = True
            output.flush();os.fsync(output.fileno())
            self._checkpoint_fault('payload_before_publish')
            os.replace(stage,destination)
            _sync_dir(self.checkpoint_directory)
            self._checkpoint_fault('payload_durable')

    def publish_prepared_checkpoint(self, prepared):
        """CPU/filesystem-only writer; never acquires a native execution lease."""
        if not isinstance(prepared,PreparedCheckpoint): raise IntegrityError('Invalid prepared checkpoint')
        self.store.authorize(prepared.snapshot,'checkpoint')
        cold = prepared.checkpoint
        history = self.history(prepared.snapshot)
        expected = PreparedCheckpoint(prepared.snapshot,cold,
            self._prepared_envelope(prepared.snapshot,cold,history))
        if expected != prepared: raise IntegrityError('Prepared checkpoint manifest/payload changed')
        payload = self.checkpoint_directory/(cold.checksum+'.state')
        manifest = self.checkpoint_directory/(prepared.snapshot+'.json')
        if payload.exists():
            if payload.stat().st_size != len(cold.blob)+4*len(cold.frontier) or digest_file(payload) != cold.checksum:
                raise IntegrityError('Existing immutable checkpoint payload is corrupt')
        else:
            self._publish_checkpoint_parts(payload,(cold.blob,struct.pack('<'+'f'*len(cold.frontier),*cold.frontier)))
        self._checkpoint_fault('before_manifest_publish')
        if manifest.exists():
            if manifest.read_bytes() != prepared.envelope: raise IntegrityError('Existing checkpoint manifest differs')
        else: self._publish_checkpoint_file(manifest,prepared.envelope)
        # Validate actual saved bytes, including a new write, before a branch
        # can acknowledge this checkpoint. RAM hashes alone do not certify I/O.
        self.validate_checkpoint(prepared.snapshot)
        return dict(manifest_path=manifest,payload_path=payload)

    def save_prepared_branch(self, name, expected_revision, prepared, operation_id, *, presentation=None,
                             presentation_source_operation=None):
        """Disk and CAS half of save_branch, using already captured bytes."""
        with self.store.read_lease(prepared.snapshot):
            self.publish_prepared_checkpoint(prepared)
            _sync_dir(self.checkpoint_directory)
            self._checkpoint_fault('before_branch_publish')
            options = {} if presentation is None else {'presentation': presentation}
            if presentation_source_operation is not None:
                options['presentation_source_operation'] = presentation_source_operation
            metadata = dict(schema=1,snapshot=prepared.snapshot,
                payload_sha256=prepared.checkpoint.checksum,
                model_contract_id=prepared.checkpoint.model_contract_id,
                execution_profile_id=prepared.checkpoint.execution_profile_id)
            with self.store._locked():
                prior = self.store._read_refs()['receipts'].get(operation_id)
                # Existing legacy operations keep their original request
                # digest. New publications bind this versioned metadata in
                # the same atomic CAS record as the resulting branch head.
                if prior is None or 'checkpoint_receipt' in prior:
                    options['checkpoint_receipt'] = metadata
                head = self.store.move(name,expected_revision,prepared.snapshot,operation_id,**options)
            self._checkpoint_fault('branch_committed_before_ack')
            return dict(branch=asdict(head),snapshot=prepared.snapshot,
                payload_sha256=prepared.checkpoint.checksum,
                model_contract_id=prepared.checkpoint.model_contract_id,
                execution_profile_id=prepared.checkpoint.execution_profile_id)

    def save_branch(self, name, expected_revision, snapshot, operation_id):
        """Durable checkpoint first, then atomic reference and retry receipt."""
        self.store.authorize(snapshot, 'checkpoint')
        with self.store.read_lease(snapshot):
            path = self.checkpoint_directory/(snapshot+'.json')
            if not path.exists():
                self.checkpoint(snapshot)
            # Even retries must never acknowledge a vanished/corrupt disk image.
            saved = self._load_checkpoint(snapshot, self.history(snapshot))
            # The previous rename may be visible despite a failed directory
            # fsync. A retry must finish that durability step before moving refs.
            _sync_dir(self.checkpoint_directory)
            self._checkpoint_fault('before_branch_publish')
            head = self.store.move(name, expected_revision, snapshot, operation_id)
            self._checkpoint_fault('branch_committed_before_ack')
            return dict(branch=asdict(head), snapshot=snapshot, payload_sha256=saved.checksum,
                model_contract_id=saved.model_contract_id, execution_profile_id=saved.execution_profile_id)

    def restore_checkpoint(self, snapshot, *, expected_schedule_sha256=None):
        """Require an exact durable image; never fall back to cached or replayed inputs."""
        self.store.authorize(snapshot, 'materialize')
        started = time.monotonic()
        with self.store.read_lease(snapshot):
            record = self.store.get_snapshot(snapshot)
            if record['model_contract_id'] != self.profile['model_contract_id']:
                raise IntegrityError('Snapshot model contract mismatch')
            decoder = controller._validated_state(record['controller'])
            if decoder['status'] not in ('idle','active','stopped'):
                raise IntegrityError('Unsupported controller status')
            history = self.history(snapshot)
            saved = self._load_checkpoint(snapshot, history)
            from .execution_receipt import schedule_digest
            digest = schedule_digest(saved.execution_schedule) if saved.execution_schedule else None
            if expected_schedule_sha256 is not None and digest != expected_schedule_sha256:
                raise IntegrityError('Requested execution schedule differs from checkpoint')
            # Native measurements are required; a backend without a work witness
            # cannot promise strict no-recompute restoration.
            if not callable(getattr(self.backend, 'work_counters', None)):
                raise IntegrityError('Backend lacks native forward-work instrumentation for strict restore')
            with self.backend.lease() as lease:
                before = self.backend.work_counters()
                if 'consumed_tokens' not in before:
                    raise IntegrityError('Missing native forward-work instrumentation')
                try:
                    resident = self.backend.restore(lease, saved.blob,
                        history=history, frontier=saved.frontier)
                    actual = self.backend.logits(lease)
                    after = self.backend.work_counters()
                    if after['consumed_tokens'] != before['consumed_tokens']:
                        raise IntegrityError('Strict checkpoint restore performed forward work')
                    if self.backend.position(lease) != len(history)-1 or tuple(actual) != saved.frontier:
                        raise IntegrityError('Restored position/frontier differs from durable checkpoint')
                    replacement = Checkpoint(saved.model_contract_id, saved.execution_profile_id,
                        saved.consumed, resident, saved.frontier,
                        Checkpoint.digest(resident, saved.frontier), saved.input_identity)
                    replacement = Checkpoint(replacement.model_contract_id,replacement.execution_profile_id,
                        replacement.consumed,replacement.blob,replacement.frontier,replacement.checksum,
                        replacement.input_identity,saved.execution_schedule)
                    self.cache[snapshot] = replacement
                except BaseException:
                    self.cache.pop(snapshot, None)
                    # A failed transfer can leave partially written native state.
                    # Destroy this backend before any later lease can reuse it.
                    self.backend.close()
                    raise
            return dict(snapshot=snapshot, payload_sha256=saved.checksum,
                schedule_sha256=digest,
                consumed=saved.consumed, **self._checkpoint_position_binding(history),
                controller_sha256=hashlib.sha256(_json_bytes(record['controller'])).hexdigest(),
                model_contract_id=saved.model_contract_id, execution_profile_id=saved.execution_profile_id,
                native_pid=getattr(getattr(self.backend, 'process', None), 'pid', None),
                forward_tokens=after['consumed_tokens']-before['consumed_tokens'],
                counters_before=before, counters_after=after,
                counter_deltas={key: value-before.get(key,0) for key,value in after.items()
                    if type(value) is int and type(before.get(key,0)) is int},
                seconds=time.monotonic()-started)

    def rebuild_materialization(self, snapshot, *, artifacts=None, reencode=None,cancelled=None,grammar_engine=None):
        """Explicitly recompute an evicted history using a new batched schedule.

        This does not restore the previous computation or publish a durable save.
        Callers must report the prefix work and preserve the returned arrangement.
        Strict restore and ordinary materialization never call this implicitly.
        """
        if cancelled is not None and not callable(cancelled):raise InvalidOperation('Cancellation check must be callable')
        if cancelled is not None and cancelled():raise CancelledError('History reconstruction cancelled before execution')
        self.store.authorize(snapshot,'materialize')
        started = time.monotonic()
        execution = self.profile.get('execution',{})
        mixed = execution.get('resume_contract') == RESUME_CONTRACT
        if (execution.get('resume_contract') not in ('identical-native-operators-v1',RESUME_CONTRACT)
                or execution.get('generated_operator') != 'normal-verifier-decode'
                or execution.get('native_request_protocol') != 'consumed-greedy-ack-v1'):
            raise InvalidOperation('History reconstruction requires the native greedy execution profile')
        with self.store.read_lease(snapshot):
            record = self.store.get_snapshot(snapshot)
            if record['model_contract_id'] != self.profile['model_contract_id']:
                raise IntegrityError('Snapshot model contract mismatch')
            manifest = self.checkpoint_directory/(snapshot+'.json')
            if snapshot in self.cache or os.path.lexists(manifest):
                raise InvalidOperation('An existing materialization must be validated and reused, not bypassed')
            decoder = controller._validated_state(record['controller'])
            history = self.history(snapshot)
            constrained = decoder.get('constraint') is not None
            if constrained:
                # Replaying canonical input does not resample earlier tokens.
                # Validate the retained parser/progress before native work, then
                # leave every controller byte unchanged. This is a new measured
                # prefill arrangement, never a claim of exact old-compute restore.
                from .grammar_first import GrammarEngine
                policy = decoder['policy']
                if (not isinstance(grammar_engine,GrammarEngine)
                        or decoder.get('schema') not in (2,3)
                        or policy.get('temperature') != 0
                        or policy.get('dry_multiplier',0) != 0
                        or policy.get('eos_tokens') != execution['eos_tokens']):
                    raise InvalidOperation('History reconstruction requires its saved greedy grammar engine')
                validator=getattr(self.backend,'validate_grammar_engine',None)
                if callable(validator):validator(grammar_engine)
                grammar_engine.restore(decoder,list(history.text_tokens if isinstance(history,InputHistory) else history))
            elif grammar_engine is not None:
                raise InvalidOperation('Plain controller cannot use a reconstruction grammar')
            if decoder['status'] != 'idle' and not constrained:
                policy = decoder.get('policy',{})
                policy_fields = {'allowed','temperature','repetition_penalty','eos_tokens',
                                 'stop_sequences','grammar','rejection_threshold'}
                if (decoder['status'] not in ('active','stopped') or type(policy) is not dict
                        or set(policy) != policy_fields
                        or any(type(policy.get(field)) not in (int,float)
                               for field in ('temperature','repetition_penalty','rejection_threshold'))
                        or policy.get('temperature') != 0
                        or policy.get('repetition_penalty') != 1
                        or policy.get('grammar') is not None
                        or type(policy.get('stop_sequences')) is not list or policy['stop_sequences']
                        or policy.get('rejection_threshold') != 1
                        or any(type(policy.get(field)) is not list
                               or any(type(token) is not int for token in policy[field])
                               for field in ('allowed','eos_tokens'))
                        or policy.get('allowed') != list(range(self.backend.vocab_size))
                        or policy.get('eos_tokens') != execution['eos_tokens']):
                    raise InvalidOperation('History reconstruction supports the tested greedy controller only')
            history = self.history(snapshot)
            if mixed:
                validate_boundary(history,capacity=execution['capacity'],vocab=self.backend.vocab_size,
                                  width=execution['embedding_width'])
            elif (not history or len(history) > execution['capacity']
                    or any(type(token) is not int or not 0 <= token < self.backend.vocab_size
                           for token in history)):
                raise InvalidOperation('Invalid history for batched reconstruction')
            activity = []
            reencode_calls = 0
            def tracked_reencode(original):
                nonlocal reencode_calls
                reencode_calls += 1
                return reencode(original)
            # Artifact leases and reconstruction validation precede native reset.
            # The trusted caller schedules any encoder on the worker's GPU queue.
            inputs = (self._input_artifacts(history,artifacts,
                        tracked_reencode if reencode is not None else None,activity)
                      if mixed else nullcontext(None))
            @contextmanager
            def checked_inputs():
                with inputs as payloads:
                    if cancelled is not None and cancelled():
                        raise CancelledError('History reconstruction cancelled before native computation')
                    yield payloads
            native_started = False
            try:
                with checked_inputs() as payloads, self.backend.lease() as lease:
                    # Another caller may have rebuilt while this caller waited
                    # for the owned backend. Do not overwrite its checked image.
                    if snapshot in self.cache or os.path.lexists(manifest):
                        raise InvalidOperation('A materialization became available while waiting for execution')
                    before = self.backend.work_counters()
                    if type(before.get('forward_tokens')) is not int:
                        raise IntegrityError('Reconstruction requires native forward-work instrumentation')
                    native_started = True
                    if mixed:
                        self.backend.consume_inputs(lease,history.spans,embeddings=payloads)
                    else:
                        self.backend.consume(lease,history)
                    saved = self._save(lease,history,self._completed_schedule((),history))
                    after = self.backend.work_counters()
                    if (type(after.get('forward_tokens')) is not int
                            or after['forward_tokens']-before['forward_tokens'] != len(history)
                            or saved.consumed != len(history)
                            or self.backend.position(lease) != len(history)-1
                            or len(saved.frontier) != self.backend.vocab_size
                            or not all(map(math.isfinite,saved.frontier))):
                        raise IntegrityError('Incomplete reconstructed work, position or frontier')
                    self.cache[snapshot] = saved
            except BaseException as error:
                if native_started:
                    self.cache.pop(snapshot,None)
                    try:
                        self.backend.close()
                    except BaseException as close_error:
                        error.add_note('Backend cleanup also failed: '+str(close_error))
                raise
            from .execution_receipt import schedule_digest
            result = dict(snapshot=snapshot,source_kind='recomputed-history',durable=False,
                prior_computation_preserved=False,consumed=len(history),**self._checkpoint_position_binding(history),
                model_contract_id=saved.model_contract_id,execution_profile_id=saved.execution_profile_id,
                schedule_sha256=schedule_digest(saved.execution_schedule),
                controller_sha256=hashlib.sha256(_json_bytes(record['controller'])).hexdigest(),
                frontier_sha256=hashlib.sha256(struct.pack('<'+'f'*len(saved.frontier),*saved.frontier)).hexdigest(),
                forward_tokens=after['forward_tokens']-before['forward_tokens'],
                counter_deltas={key:value-before[key] for key,value in after.items()
                    if type(value) is int and type(before.get(key)) is int},
                native_pid=getattr(getattr(self.backend,'process',None),'pid',None),
                seconds=time.monotonic()-started)
            if mixed:
                result.update(reencode_callback_calls=reencode_calls,image_artifacts=activity)
            return result

    def checkpoint_manifest(self, snapshot, history=None):
        """Validate canonical binding before allocating a disk-read buffer."""
        from .checkpoint_cache import validate_manifest
        self.store.authorize(snapshot,'materialize')
        history = self.history(snapshot) if history is None else history
        try:
            envelope = _decode((self.checkpoint_directory/(snapshot+'.json')).read_bytes())
            required = dict(schema=1,snapshot=snapshot,input_identity=self._input_identity(history),
                model_contract_id=self.profile['model_contract_id'],execution_profile_id=self.profile['execution_profile_id'],
                consumed=len(history),**self._checkpoint_position_binding(history),logits_count=self.backend.vocab_size)
            contract = self._schedule_contract()
            if contract is not None: required['schema'] = contract[0]
            manifest = validate_manifest(envelope, required)
            schedule = ()
            if contract is not None:
                _, validate = contract
                schedule = tuple(manifest['execution_schedule'])
                digest = validate(schedule,history,self.profile['execution']['prefill_chunk'])
                if digest != manifest['schedule_sha256']:
                    raise IntegrityError('Checkpoint execution schedule checksum mismatch')
            return manifest
        except (OSError,KeyError,TypeError,ValueError) as error:
            raise IntegrityError('Invalid or incomplete checkpoint manifest') from error

    def validate_checkpoint(self, snapshot):
        """Stream validation for retry receipts; no native load or full blob copy."""
        manifest = self.checkpoint_manifest(snapshot)
        path = self.checkpoint_directory/(manifest['payload_sha256']+'.state')
        size,width = manifest['state_bytes'],4*self.backend.vocab_size
        try:
            if path.stat().st_size != size+width or digest_file(path) != manifest['payload_sha256']:
                raise IntegrityError('Checkpoint payload corruption')
            with path.open('rb') as stream:
                stream.seek(size)
                logits = struct.unpack('<'+'f'*self.backend.vocab_size,stream.read(width))
            if not all(map(math.isfinite,logits)): raise IntegrityError('Nonfinite checkpoint frontier')
            return manifest
        except (OSError,ValueError,struct.error) as exc:
            raise IntegrityError('Invalid checkpoint payload') from exc

    def _load_checkpoint(self, snapshot, history):
        try:
            manifest = self.checkpoint_manifest(snapshot,history)
            schedule = tuple(manifest.get('execution_schedule',()))
            sha = manifest['payload_sha256']
            payload = (self.checkpoint_directory/(sha+'.state')).read_bytes()
            size = manifest['state_bytes']
            if type(size) is not int or size<=0 or len(payload)!=size+4*self.backend.vocab_size or hashlib.sha256(payload).hexdigest()!=sha:
                raise IntegrityError('Checkpoint payload corruption')
            logits = struct.unpack('<'+str(self.backend.vocab_size)+'f',payload[size:])
            if not all(map(math.isfinite, logits)):
                raise IntegrityError('Nonfinite checkpoint frontier')
            return Checkpoint(manifest['model_contract_id'],manifest['execution_profile_id'],len(history),
                payload[:size],logits,sha,manifest['input_identity'],schedule)
        except (OSError,KeyError,TypeError,ValueError) as error:
            raise IntegrityError('Invalid or incomplete checkpoint') from error

    def evict_all(self):
        if self.store._principal is not None:
            raise PermissionError('Global eviction requires local owner access')
        self.cache.clear()
        self.backend.release_checkpoints()
        for path in self.checkpoint_directory.iterdir():
            if path.is_file():
                path.unlink()
        _sync_dir(self.checkpoint_directory)

    def measure(self, report, snapshot, history):
        path = os.environ.get('CTT_MEASUREMENTS')
        if path:
            with open(path,'a') as output:
                output.write(json.dumps(dict(snapshot=snapshot,history=history,numerics=asdict(report),
                    **self.backend.checkpoint_metrics(self.cache.values())))+'\n')
