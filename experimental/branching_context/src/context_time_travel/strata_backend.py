"""Strata's private pipe behind the canonical history/controller API.

Hot images own native immutable handles. Cold images use the existing native
file format; only the canonical checkpoint manifest binds file, frontier,
history and model. The pipe itself never publishes an API snapshot.
"""
from contextlib import contextmanager
from dataclasses import dataclass
from types import MappingProxyType
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path, PurePosixPath
import queue
import re
import selectors
import struct
import subprocess
import sys
import tempfile
import threading
import time
import uuid
import weakref

from .store import IntegrityError, InvalidOperation
from .pools import normalize_gpu_selection
from .input_spans import InputHistory
from .vision_execution import (VISION_PROTOCOL, RESUME_CONTRACT, POSITION_CONVENTION,
                               placements, validate_boundary)
from .input_projection import cached_input_binding as input_binding


def _tmpfs_mount(mounts, directory):
    if not directory.is_dir():
        return False
    target = directory.resolve()
    for line in mounts.splitlines():
        fields = line.split()
        if len(fields) < 3 or fields[2] != 'tmpfs':
            continue
        mount = re.sub(r'\\([0-7]{3})',lambda match:chr(int(match.group(1),8)),fields[1])
        if Path(mount).resolve() == target:
            return True
    return False


def _state_scratch():
    # This is transient transport, never the durable checkpoint directory.
    # Use the existing native format on a verified RAM filesystem when present.
    # Other platforms retain their ordinary private temporary directory.
    candidate = Path('/dev/shm')
    mounts = Path('/proc/mounts')
    if sys.platform == 'linux' and mounts.is_file() and _tmpfs_mount(mounts.read_text(),candidate):
        return tempfile.TemporaryDirectory(prefix='ctt-strata-',dir=candidate),'tmpfs'
    return tempfile.TemporaryDirectory(prefix='ctt-strata-'),'default-temporary-filesystem'


def _sha(path):
    result = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def _environment():
    return {key: value for key, value in os.environ.items()
            if key.startswith(('STRATA_', 'CUDA_', 'NVIDIA_', 'GGML_'))}


def verify_runtime_source_closure(profile, runtime_root=None):
    """Verify the declared source bytes at startup, not on the token hot path.

    This checks the declared set, not its completeness or future filesystem
    immutability. Legacy profiles without a declaration remain explicitly
    unverified. Relocation changes no identity when the selected bytes match.
    """
    closure = profile.get('native_build', {}).get('selected_runtime_source_closure')
    if closure is None:
        return dict(status='not_declared', files=0)
    if not isinstance(closure, dict):
        raise IntegrityError('Invalid runtime source declaration')
    pins = closure.get('files_sha256')
    if not isinstance(pins, dict) or not 1 <= len(pins) <= 4096:
        raise IntegrityError('Invalid runtime source file set')
    if runtime_root is None:
        runtime_root = Path(__file__).resolve().parents[4]
    root = Path(runtime_root).resolve(strict=True)
    checked = {}
    for declared, expected in pins.items():
        if (type(declared) is not str or not declared or len(declared) > 4096
                or type(expected) is not str or len(expected) != 64
                or any(c not in '0123456789abcdef' for c in expected)):
            raise IntegrityError('Invalid runtime source hash declaration')
        parts = PurePosixPath(declared.replace('\\', '/')).parts
        markers = [parts.index(marker) for marker in ('serve', 'experimental', 'tools')
                   if marker in parts]
        if not markers or '..' in parts:
            raise IntegrityError('Invalid runtime source path: ' + declared)
        suffix = Path(*parts[min(markers):])
        relative = suffix.as_posix()
        if relative in checked:
            raise IntegrityError('Duplicate runtime source path: ' + relative)
        target = root
        for part in suffix.parts:
            target = target / part
            if target.is_symlink():
                raise IntegrityError('Linked runtime source path: ' + relative)
        try:
            if not target.is_file() or not target.resolve(strict=True).is_relative_to(root):
                raise IntegrityError('Missing runtime source file: ' + relative)
            actual = _sha(target)
        except OSError as error:
            raise IntegrityError('Unreadable runtime source file: ' + relative) from error
        if actual != expected:
            raise IntegrityError('Runtime source fingerprint mismatch: ' + relative)
        checked[relative] = actual
    digest = hashlib.sha256(json.dumps(checked, sort_keys=True,
                                      separators=(',', ':')).encode()).hexdigest()
    return dict(status='verified', files=len(checked), files_sha256=digest,
                declared_source_commit=closure.get('source_commit'))


def _validate_execution(profile, *, environment=None):
    verify_runtime_source_closure(profile)
    actual = _environment() if environment is None else {
        k:v for k,v in environment.items() if k.startswith(('STRATA_','CUDA_','NVIDIA_','GGML_'))}
    if actual != profile['execution']['environment']:
        raise IntegrityError('Execution environment differs from pinned Strata profile')
    for name, expected in profile['native_build']['sha256'].items():
        if _sha(name) != expected:
            raise IntegrityError('Native executable/helper fingerprint mismatch: ' + name)
    identity = _digest(dict(build=profile['native_build'], execution=profile['execution'], hardware=profile['hardware']))
    if identity != profile['execution_profile_id']:
        raise IntegrityError('Strata execution profile fingerprint mismatch')
    hardware = subprocess.check_output(['nvidia-smi', '--query-gpu=name,uuid,driver_version,memory.total',
                                        '--format=csv,noheader'], text=True).strip()
    if hardware != profile['hardware']:
        raise IntegrityError('Strata execution hardware differs from pinned profile')


def worker_command(profile,gpu_id):
    """Bind one physical GPU, from a list included in the pinned profile.

    --gpu is applied by generate.cpp before its first CUDA call. This is not
    Strata's layer-split multi-GPU mode. Never rebind a legacy profile silently.
    """
    command = list(profile['execution']['sut_command'])
    if gpu_id is None: return command
    try:
        ids = normalize_gpu_selection(profile['execution'].get('GPU'))
        gpu_id = normalize_gpu_selection([gpu_id])[0]
    except ValueError as exc:
        raise InvalidOperation('Invalid worker GPU topology') from exc
    if gpu_id not in ids:
        raise InvalidOperation('Worker GPU must be explicitly selected in its pinned profile')
    grouped = isinstance(gpu_id,tuple)
    value = ','.join(map(str,gpu_id)) if grouped else str(gpu_id)
    if '--gpus' in command or (not grouped and ('--split-device' in command or '--layer-split' in command)):
        raise InvalidOperation('Independent workers cannot use a layer-split command')
    if '--gpu' in command:
        i = command.index('--gpu')
        if command.count('--gpu') != 1 or i+1 == len(command) or command[i+1] != value:
            raise InvalidOperation('Worker command selects another GPU or multiple GPUs')
    else: command += ['--gpu',value]
    return command


def _validate_model(profile):
    receipt_path = Path(profile['model_hash_receipt'])
    if _sha(receipt_path) != profile['model_hash_receipt_sha256']:
        raise IntegrityError('Executed model hash receipt changed')
    receipt = json.loads(receipt_path.read_text())
    hashed_bytes, reused = 0, 0
    for name, item in profile['model_files'].items():
        stat = Path(name).stat()
        same = (stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns) == (item['bytes'], item['mtime_ns'], item['ctime_ns'])
        measured = receipt['model_files'].get(name, {})
        if same and measured.get('sha256') == item['sha256'] and measured.get('bytes') == item['bytes']:
            reused += 1
        else:
            if _sha(name) != item['sha256']:
                raise IntegrityError('Model weight/pack fingerprint mismatch: ' + name)
            hashed_bytes += stat.st_size
    weights = {name: dict(bytes=item['bytes'], sha256=item['sha256']) for name, item in profile['model_files'].items()}
    if _digest(weights) != profile['model_weights_sha256'] or _digest(profile['model_contract']) != profile['model_contract_id']:
        raise IntegrityError('Strata model contract fingerprint mismatch')
    if profile['model_contract']['weights_sha256'] != profile['model_weights_sha256']:
        raise IntegrityError('Strata model contract references different weights')
    return dict(model_bytes_hashed_this_process=hashed_bytes, model_file_hash_receipts_reused=reused)


def _trace(variable, row):
    path = os.environ.get(variable)
    if path:
        with open(path, 'a') as stream:
            stream.write(json.dumps(dict(time=time.time(), **row), allow_nan=False) + '\n')


def _unpack_frontier(raw, vocab):
    if len(raw) != vocab * 4:
        raise IntegrityError('Incomplete full-vocabulary frontier')
    values = struct.unpack('<' + str(vocab) + 'f', raw)
    if not all(map(math.isfinite, values)):
        raise IntegrityError('Nonfinite native frontier')
    return values


@dataclass(frozen=True)
class StrataImage:
    """Process-local ownership, never a persisted history or portable file."""
    identity: bytes
    process_identity: bytes
    handle: int
    consumed: int
    running_bytes: int
    frontier_bytes: int
    pages: tuple
    input_identity: str = ''

    def __len__(self):
        # Referenced state extent; deduplicated ownership is reported separately.
        return self.running_bytes + self.frontier_bytes + self.consumed * 4 + sum(p[3] for p in self.pages)


@dataclass
class _Lease:
    backend: object
    bound: bool = False
    consumed: int = 0
    image_identity: bytes | None = None
    state: str = 'NEW'
    _owner_thread: int | None = None
    inputs: InputHistory | None = None
    _input_binding_cache: tuple | None = None

    def __enter__(self):
        if self.state != 'NEW':
            raise InvalidOperation('Execution lease cannot be entered twice')
        self.backend._lock.acquire()
        try:
            if self.state != 'NEW' or self.backend._closed:
                raise InvalidOperation('Unavailable execution lease')
            if self.backend._active is not None:
                raise InvalidOperation('Nested private leases are not supported')
            self.backend._flush_drops()
            self._owner_thread = threading.get_ident()
            self.state = 'ACTIVE'
            self.backend._active = self
            self.backend.stats['lease_acquires'] += 1
            return self
        except BaseException:
            self.backend._lock.release()
            raise

    def __exit__(self, exc_type, exc, traceback):
        # Check before changing the active lease or releasing its owner's lock.
        self.backend._check(self)
        if getattr(self.backend, '_request_active', None) is not None:
            raise InvalidOperation('Close the owned native request before releasing its lease')
        self._input_binding_cache = None
        self.backend._active = None
        self.state = 'CLOSED'
        self.backend.stats['lease_releases'] += 1
        self.backend._lock.release()
        return False


def _drop_image(backend_ref, handle):
    backend = backend_ref()
    if backend is not None:
        backend._drops.put(handle)


class _NativeRequest:
    """Owned request iterator. Tokens are consumed progress, not published snapshots."""
    def __init__(self, backend, lease, limit):
        self.backend, self.lease, self.limit = backend, lease, limit
        self.initial = lease.consumed
        self.initial_inputs = lease.inputs
        self.count = 0
        self.tokens = []
        self.started = self.awaiting_ack = False
        self.closed = limit == 0

    def __iter__(self):
        return self

    def __enter__(self):
        self.backend._check(self.lease)
        return self

    def __exit__(self, *_):
        self.close()

    def _send(self, command):
        self.backend.process.stdin.write(command+'\n')
        self.backend.process.stdin.flush()

    def _finish(self, response, *, cancelled=False):
        if (response.get('status') != 'ok' or response.get('operator') != 'normal-verifier-decode'
                or type(response.get('generated')) is not int or response['generated'] != self.count
                or type(response.get('consumed')) is not int or response['consumed'] != self.initial+self.count
                or type(response.get('forward_tokens')) is not int
                or response['forward_tokens']-self.before_work != self.count
                or (self.count < self.limit and not cancelled and self.last_token not in self.eos)):
            raise IntegrityError('Incomplete or inconsistent native request completion')
        from .execution_receipt import tokens_digest, validate_operator_schedule
        receipt = {key: response.get(key) for key in ('operator','position','count','chunks')}
        receipt['tokens_sha256'] = tokens_digest(self.tokens)
        if (type(receipt['position']) is not int or receipt['position'] != self.initial
                or type(receipt['count']) is not int or receipt['count'] != self.count):
            raise IntegrityError('Native request receipt position/count mismatch')
        try:
            local = dict(receipt, position=0,
                         chunks=[[p-self.initial,n] for p,n in receipt['chunks']])
            validate_operator_schedule([local],self.tokens,1)
        except (ValueError,TypeError) as error:
            raise IntegrityError('Missing or invalid native request receipt') from error
        self.backend.last_execution_receipt = receipt
        self.backend.last_native_request = response
        self.backend._request_active = None
        self.closed = True
        if self.backend._vision_enabled():
            if not isinstance(self.initial_inputs, InputHistory):
                raise IntegrityError('Native request lost its mixed input binding')
            history = self.initial_inputs + tuple(self.tokens)
            self.backend._bind_input_receipt(self.lease, history, receipt)

    def _fail(self):
        self.backend._protocol_failed = True
        self.backend._request_active = None
        self.closed = True

    def __next__(self):
        self.backend._check(self.lease)
        if self.closed:
            raise StopIteration
        if not self.started and getattr(self.backend, '_request_active', None) is not None:
            raise InvalidOperation('Another native request already owns the pipe')
        try:
            if not self.started:
                self.before_work = self.backend._refresh_stats()['forward_tokens']
                self.eos = tuple(self.backend.profile['execution']['eos_tokens'])
                self.last_token = None
                self.backend._request_active = self
                self._send('generate '+str(self.limit)+' '+str(len(self.eos))+
                           ''.join(' '+str(token) for token in self.eos))
                self.started = True
            elif self.awaiting_ack:
                self._send('advance')
                self.awaiting_ack = False
            response = self.backend._read()
            if (response.get('status') == 'error' and set(response) == {'status','message'}
                    and isinstance(response.get('message'), str) and response['message'].strip()):
                # The native frame carries a useful terminal cause but no
                # consumed/work receipt. Surface the cause while the enclosing
                # failure path still poisons this pipe; do not reuse it.
                raise IntegrityError(response['message'])
            if response.get('status') == 'ok':
                self._finish(response)
                raise StopIteration
            token = response.get('token')
            if (response.get('status') != 'token' or type(token) is not int
                    or not 0 <= token < self.backend.vocab_size or self.count >= self.limit
                    or self.last_token in self.eos
                    or type(response.get('consumed')) is not int
                    or response['consumed'] != self.initial+self.count+1):
                raise IntegrityError('Invalid consumed native token event')
            self.count += 1
            self.tokens.append(token)
            self.last_token = token
            self.lease.consumed = response['consumed']
            self.lease.image_identity = None
            self.awaiting_ack = True
            return token
        except StopIteration:
            raise
        except BaseException:
            self._fail()
            raise

    def close(self):
        # Check before entering iterator cleanup: foreign close cannot strand
        # the original owner by closing the Python generator underneath it.
        self.backend._check(self.lease)
        if self.closed:
            return
        if not self.started:
            self.closed = True
            return
        try:
            if not self.awaiting_ack:
                raise IntegrityError('Native cancellation has no consumed boundary')
            self._send('cancel')
            self.awaiting_ack = False
            self._finish(self.backend._read(), cancelled=True)
        except BaseException:
            self._fail()
            raise


def _validate_selected_token_capability(profile, ready):
    expected = profile['execution'].get('selected_token_protocol')
    if expected is not None and (expected != 'consumed-selected-receipt-v1'
                                 or ready.get('selected_token_protocol') != expected):
        raise IntegrityError('Native selected-token receipt capability differs from pinned profile')


def _validate_grammar_selection_capability(profile, ready):
    expected = profile['execution'].get('grammar_selection_protocol')
    if expected is not None and (expected != 'finite-mask-argmax-v1'
                                 or ready.get('grammar_selection_protocol') != expected):
        raise IntegrityError('Native grammar selection capability differs from pinned profile')


def _validate_sampling_capability(profile,ready):
    from .min_p import MIN_P_PROTOCOL,SAMPLING_PROTOCOL_V2,DRY_PROTOCOL,SAMPLING_PROTOCOL_V4,SAMPLING_PROTOCOL_V6,SAMPLING_PROTOCOL_V7
    from .sampling_proposal import XTC_PROPOSAL_PROTOCOL,MIROSTAT2_PROPOSAL_PROTOCOL,ADAPTIVE_P_PROPOSAL_PROTOCOL
    expected=profile['execution'].get('sampling_protocol')
    if expected is not None and (expected not in (MIN_P_PROTOCOL,SAMPLING_PROTOCOL_V2,DRY_PROTOCOL,SAMPLING_PROTOCOL_V4,
                                                    XTC_PROPOSAL_PROTOCOL,SAMPLING_PROTOCOL_V6,
                                                    MIROSTAT2_PROPOSAL_PROTOCOL,SAMPLING_PROTOCOL_V7,
                                                    ADAPTIVE_P_PROPOSAL_PROTOCOL)
            or ready.get('sampling_protocol')!=expected):
        raise IntegrityError('Native Min-P capability differs from pinned profile')
    if expected in (SAMPLING_PROTOCOL_V7, ADAPTIVE_P_PROPOSAL_PROTOCOL):
        if ready.get('adaptive_p_protocol') != ADAPTIVE_P_PROPOSAL_PROTOCOL:
            raise IntegrityError('Native Adaptive-P capability differs from pinned profile')
        retained = {
            'sampling_v1_protocol': MIN_P_PROTOCOL,
            'sampling_v2_protocol': SAMPLING_PROTOCOL_V2,
            'sampling_v3_protocol': DRY_PROTOCOL,
            'sampling_v4_protocol': SAMPLING_PROTOCOL_V4,
            'sampling_v5_protocol': XTC_PROPOSAL_PROTOCOL,
            'sampling_v6_protocol': MIROSTAT2_PROPOSAL_PROTOCOL,
            'mirostat2_protocol': MIROSTAT2_PROPOSAL_PROTOCOL,
        }
        if any(ready.get(key) != value for key, value in retained.items()):
            raise IntegrityError('Native V7 profile is missing a retained sampling command')
        if (ready.get('adaptive_p_feedback_protocol') != 'adaptive-p-state-v1'
                or type(ready.get('adaptive_p_feedback_version')) is not int
                or ready.get('adaptive_p_feedback_version') != 1
                or ready.get('mirostat2_feedback_protocol') != 'mirostat2-state-v1'
                or type(ready.get('mirostat2_feedback_version')) is not int
                or ready.get('mirostat2_feedback_version') != 1):
            raise IntegrityError('Native Adaptive-P feedback capability differs from pinned profile')


class StrataBackend:
    def __init__(self, profile, *, gpu_id=None):
        self.profile = profile
        self.gpu_id = gpu_id
        command = worker_command(profile,gpu_id)
        if isinstance(gpu_id,(tuple,list)) and len(gpu_id)>1:
            # generate.cpp supports layer splitting, but this pinned private
            # BranchSession protocol has no verified composite checkpoint.
            raise InvalidOperation('Native Context COW lacks verified multi-device capture/restore')
        self._process_environment = None
        if gpu_id is not None:
            # Routing and numerical settings are private to the native child;
            # creating worker 5 must not change worker 1's process environment.
            self._process_environment = {k:v for k,v in os.environ.items()
                if not k.startswith(('STRATA_','CUDA_','NVIDIA_','GGML_'))}
            self._process_environment.update(profile['execution']['environment'])
        self.validate_execution_profile()
        self.stats = dict(consumed_tokens=0, checkpoint_export_bytes=0, checkpoint_import_bytes=0,
            contexts_created=0, contexts_freed=0, lease_acquires=0, lease_releases=0,
            frontier_read_bytes=0, **_validate_model(profile))
        helper = Path(profile['gguf_reader'])
        spec = importlib.util.spec_from_file_location('ctt_strata_gguf', helper)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        self._metadata = module.GGUFFile(Path(profile['model_path'])).metadata
        self._tokens = self._metadata['tokenizer.ggml.tokens']
        self.vocab_size = len(self._tokens)
        if self._metadata['general.architecture'] != 'qwen4exp' or self._metadata['tokenizer.ggml.model'] != 'gpt2':
            raise IntegrityError('Unsupported Strata model/tokenizer architecture')
        # The inspected tools/strata_tokenizer.py uses this exact GPT-2 byte map.
        direct = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
        remaining = [byte for byte in range(256) if byte not in direct]
        self._byte_map = dict(zip(map(chr, direct + list(range(256, 256 + len(remaining)))), direct + remaining))
        self._lock = threading.RLock()
        self._active = None
        self._drops = queue.SimpleQueue()
        self._process_identity = uuid.uuid4().bytes
        self._scratch, self.state_staging_storage = _state_scratch()
        self._directory = Path(self._scratch.name)
        self._log = tempfile.TemporaryFile()
        self._sequence = 0
        self._closed = False
        self.process = subprocess.Popen(command, stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, stderr=self._log, text=True, bufsize=1,
            env=self._process_environment)
        self.stats['contexts_created'] = 1
        try:
            ready = self._read()
            if ready.get('status') != 'ready' or ready.get('vocab') != self.vocab_size or ready.get('capacity') != profile['execution']['capacity']:
                raise IntegrityError('Native geometry differs from pinned profile')
            _validate_selected_token_capability(profile, ready)
            _validate_grammar_selection_capability(profile, ready)
            _validate_sampling_capability(profile,ready)
            self._ready = dict(ready)
            protocol = profile['execution'].get('native_request_protocol')
            if protocol is not None and ready.get('native_request_protocol') != protocol:
                raise IntegrityError('Native request capability differs from pinned profile')
            self._refresh_stats()
            if self._vision_enabled():
                if (profile['execution'].get('vision_protocol') != VISION_PROTOCOL
                        or profile['execution'].get('embedding_width') != self._metadata.get('qwen4exp.embedding_length')):
                    raise IntegrityError('Mixed-input profile differs from loaded model geometry')
                info = self._request('vision_info')
                if (info.get('vision_protocol') != VISION_PROTOCOL or info.get('enabled') is not True
                        or info.get('hot_image_snapshots') is not True
                        or info.get('cold_image_snapshots') is not True):
                    raise IntegrityError('Native worker lacks pinned image input/restore capability')
                expected_dry=profile['execution'].get('vision_dry_protocol')
                if expected_dry is not None:
                    from .vision_dry import PROTOCOL as MIXED_DRY_PROTOCOL
                    if expected_dry!=MIXED_DRY_PROTOCOL or info.get('vision_dry_protocol')!=expected_dry:
                        raise IntegrityError('Native mixed DRY capability differs from pinned profile')
                    self._ready['vision_dry_protocol']=expected_dry
                    combined=info.get('vision_dry_sampling_protocol')
                    if combined is not None:
                        from .min_p import SAMPLING_PROTOCOL_V4
                        from .vision_dry import SAMPLING_PROTOCOL as MIXED_SAMPLING_PROTOCOL
                        families=info.get('vision_sampling_protocols')
                        if (combined!=MIXED_SAMPLING_PROTOCOL or type(families) is not list or not families
                                or any(type(p) is not str or p!=SAMPLING_PROTOCOL_V4 for p in families)
                                or len(set(families))!=len(families)):
                            raise IntegrityError('Native mixed DRY sampler families are incompatible')
                        self._ready.update(vision_dry_sampling_protocol=combined,vision_sampling_protocols=list(families))
        except BaseException:
            self.close()
            raise

    def validate_execution_profile(self):
        _validate_execution(self.profile,environment=self._process_environment)

    def metadata(self, key):
        value = self._metadata.get(key)
        return None if value is None else str(value)

    def contract_fields(self):
        fields = dict(attention=self.profile['model_contract']['attention'],
                      state_kind=self.profile['model_contract']['state_kind'])
        if self._vision_enabled():
            fields.update(positions=POSITION_CONVENTION,input_kind='canonical-mixed-spans-v1')
        return fields

    def _vision_enabled(self):
        return self.profile.get('execution',{}).get('resume_contract') == RESUME_CONTRACT

    def _expected_input_binding(self, history):
        """One derived digest per immutable history on the currently owned lease.

        Native receipts are still queried and compared on every check. Cache
        only the exact detached InputHistory type backed by immutable bytes;
        mutable subclasses use the original uncached computation. Include the
        encoded object identity so even a private-field replacement cannot hit
        a former binding. Retention is bounded to one entry per active lease.
        """
        lease = getattr(self, '_active', None)
        if (type(history) is not InputHistory or type(history._encoded) is not bytes
                or not isinstance(lease, _Lease) or lease.backend is not self
                or lease.state != 'ACTIVE' or lease._owner_thread != threading.get_ident()):
            return input_binding(history)
        cached = lease._input_binding_cache
        if cached is not None and cached[0] is history and cached[1] is history._encoded:
            return cached[2]
        expected = MappingProxyType(input_binding(history))
        # Publish the complete immutable entry with one reference assignment;
        # interruption cannot expose new keys paired with a former digest.
        lease._input_binding_cache = (history, history._encoded, expected)
        return expected

    def _check_input_binding(self, history):
        info = self._request('vision_info')
        expected = dict(consumed=len(history), **self._expected_input_binding(history))
        if (info.get('vision_protocol') != VISION_PROTOCOL or info.get('enabled') is not True
                or any(info.get(key) != value for key,value in expected.items())):
            raise IntegrityError('Native image input/position binding differs from canonical history')
        return info

    def _bind_input_receipt(self, lease, history, receipt):
        from .execution_receipt import tokens_digest, validate_operator_schedule
        start = receipt.get('position')
        if type(start) is not int or not 0 <= start < len(history):
            raise IntegrityError('Mixed receipt has an invalid start position')
        ids = history.native_tokens[start:]
        local = dict(receipt, position=0,
                     chunks=[[p-start,n] for p,n in receipt['chunks']])
        if receipt['tokens_sha256'] != tokens_digest(ids):
            raise IntegrityError('Mixed receipt token projection differs from request')
        try:
            validate_operator_schedule([local],ids,self.profile['execution']['prefill_chunk'])
        except (TypeError,ValueError) as exc:
            raise IntegrityError('Invalid mixed native execution receipt') from exc
        self._check_input_binding(history)
        lease.inputs = history
        self.last_execution_receipt = dict(receipt,inputs_sha256=history.digest,**self._expected_input_binding(history))

    def consume_inputs(self, lease, spans, *, embeddings):
        """Consume explicit input spans while the caller holds artifact leases.

        Embeddings are validated immutable SVE values, never caller file paths.
        Scratch files transport them to the existing native bounded parser.
        """
        from .vision_embeddings import Embedding
        from .execution_receipt import tokens_digest
        self._check(lease)
        if not self._vision_enabled():
            raise InvalidOperation('Pinned profile does not support mixed inputs')
        delta = InputHistory(spans)
        prior = lease.inputs if lease.bound else InputHistory([])
        if not isinstance(prior,InputHistory) or len(prior) != lease.consumed:
            raise IntegrityError('Active mixed history is unavailable')
        history = InputHistory(prior.spans+delta.spans)
        validate_boundary(history,capacity=self.profile['execution']['capacity'],
            vocab=self.vocab_size,width=self.profile['execution']['embedding_width'])
        images = placements(delta)
        if not images:
            return self._consume(lease,delta.native_tokens,'feed')
        if len(images)>8:
            raise InvalidOperation('One native request supports at most eight images')
        payloads = []
        for span in delta.spans:
            if span['kind'] != 'image': continue
            value = embeddings.get(span['embedding_sha256'])
            if (not isinstance(value,Embedding) or value.sha256 != span['embedding_sha256']
                    or value.nbytes != span['embedding_bytes']
                    or (value.rows,value.width,value.grid_width,value.grid_height) !=
                       (span['rows'],span['width'],span['grid_width'],span['grid_height'])):
                raise IntegrityError('Missing or incompatible leased image embedding')
            payloads.append(value.payload)
        # All request validation precedes reset, forward computation or COW writes.
        with tempfile.TemporaryDirectory(prefix='vision-',dir=self._directory) as staging:
            folder = Path(staging)
            lines = [f'COW_VISION_INPUT_V1 {len(prior)} {len(delta)} {len(images)}',
                     ' '.join(map(str,delta.native_tokens))]
            for index,(image,payload) in enumerate(zip(images,payloads)):
                path = folder/(str(index)+'.sve'); path.write_bytes(payload)
                fields = [image['start']+len(prior)] + [image[key] for key in
                    ('rows','grid_width','grid_height','width','pad_token','embedding_sha256')]
                lines.append(' '.join(map(str,fields))+' '+json.dumps(str(path)))
            manifest = folder/'input.manifest'; manifest.write_text('\n'.join(lines)+'\n')
            if not lease.bound:
                self._request('reset'); lease.bound=True
            before = self._refresh_stats()['forward_tokens']
            try:
                response = self._request('feed_image '+json.dumps(str(manifest)))
                after = self._refresh_stats()['forward_tokens']
                if (after-before != len(delta) or response.get('consumed') != len(history)
                        or response.get('position') != len(prior)
                        or response.get('operator') != 'batched-prefill'):
                    raise IntegrityError('Mixed native work/position receipt mismatch')
                receipt = {key:response.get(key) for key in ('operator','position','count','chunks')}
                receipt['tokens_sha256'] = tokens_digest(delta.native_tokens)
                self._bind_input_receipt(lease,history,receipt)
                lease.consumed = len(history)
                lease.image_identity = None
            except BaseException:
                self._protocol_failed = True
                raise

    def validate_grammar_engine(self, engine):
        """Bind grammar token IDs to this already-verified model, once per backend."""
        from .grammar_first import GrammarEngine
        from tools.strata_tokenizer import Tokenizer
        if not isinstance(engine, GrammarEngine):
            raise InvalidOperation('Expected a checked grammar engine')
        expected = getattr(self, '_grammar_tokenizer_sha256', None)
        if expected is None:
            md = self._metadata
            tokenizer = Tokenizer(list(self._tokens), list(md['tokenizer.ggml.merges']),
                list(md.get('tokenizer.ggml.token_type') or []) or None,
                md.get('tokenizer.ggml.pre', 'qwen35'),
                {key: int(value) for key, value in md.items()
                 if key.startswith('tokenizer.ggml.') and key.endswith('_token_id')})
            expected = GrammarEngine._tokenizer_hash(tokenizer)
            self._grammar_tokenizer_sha256 = expected
        if engine.tokenizer_sha256 != expected:
            raise IntegrityError('Grammar tokenizer differs from the loaded model')

    def new_cache(self):
        from .checkpoint_cache import CheckpointCache
        # Conservative: count every image's referenced pages, even when shared.
        # This bounds this cache, not externally held images or the model arena.
        policy = self.profile['execution'].get('checkpoint_cache',
            dict(byte_limit=1024*1024*1024,entry_limit=256))
        if (not isinstance(policy,dict) or set(policy) != {'byte_limit','entry_limit'}
                or type(policy['byte_limit']) is not int or not 0 < policy['byte_limit'] <= 32*1024**3
                or type(policy['entry_limit']) is not int or not 0 < policy['entry_limit'] <= 256):
            raise IntegrityError('Invalid bounded derived-checkpoint resource policy')
        return CheckpointCache(policy['byte_limit'],policy['entry_limit'])

    def piece(self, token):
        if type(token) is not int or not 0 <= token < self.vocab_size:
            raise InvalidOperation('Invalid token ID')
        try:
            return bytes(self._byte_map[c] for c in self._tokens[token])
        except KeyError as error:
            raise IntegrityError('Token contains a character outside the pinned byte alphabet') from error

    def is_eog(self, token):
        return token in self.profile['execution']['eos_tokens']

    def _read(self):
        with selectors.DefaultSelector() as selector:
            selector.register(self.process.stdout, selectors.EVENT_READ)
            if not selector.select(timeout=18000):
                raise TimeoutError('Strata protocol request exceeded execution limit')
        line = self.process.stdout.readline()
        if not line:
            self._log.seek(0)
            raise RuntimeError('Strata process closed: ' + self._log.read().decode(errors='replace')[-4000:])
        return json.loads(line)

    def _request(self, command):
        if getattr(self, '_protocol_failed', False):
            raise IntegrityError('Native request protocol failed; restart the backend')
        if getattr(self, '_request_active', None) is not None:
            raise InvalidOperation('A native generation request still owns the pipe')
        if self._closed or self.process.poll() is not None:
            raise IntegrityError('Native Strata context is unavailable')
        self.process.stdin.write(command + '\n')
        self.process.stdin.flush()
        response = self._read()
        _trace('CTT_NATIVE_TRACE', dict(pid=self.process.pid, command=command, response=response))
        if response.get('status') not in ('ok', 'closed'):
            message = response.get('message', 'Invalid native response')
            if 'bad_alloc' in message:
                raise MemoryError(message)
            raise IntegrityError(message)
        return response

    def _refresh_stats(self):
        raw = self._request('stats')
        self.stats.update({key: value for key, value in raw.items() if key != 'status'})
        self.stats['consumed_tokens'] = raw['forward_tokens']
        self.stats['checkpoint_export_bytes'] = raw['saved_running_bytes'] + raw['written_file_bytes']
        self.stats['checkpoint_import_bytes'] = raw['restored_running_bytes'] + raw['loaded_file_bytes']
        return raw

    def work_counters(self):
        """Read actual execution counters from the native pipe."""
        self._refresh_stats()
        return dict(self.stats)

    def _flush_drops(self):
        while True:
            try:
                handle = self._drops.get_nowait()
            except queue.Empty:
                return
            self._request('drop ' + str(handle))

    def release_checkpoints(self):
        with self._lock:
            self._flush_drops()
            self._request('ram_clear')
            self._refresh_stats()

    def _check(self, lease):
        if self._active is not lease or lease.state != 'ACTIVE' or lease._owner_thread != threading.get_ident():
            raise InvalidOperation('Inactive or foreign private lease')

    def lease(self):
        return _Lease(self)

    def position(self, lease):
        self._check(lease)
        if not lease.bound:
            return -1
        actual = self._refresh_stats()['consumed']
        if actual != lease.consumed:
            raise IntegrityError('Native consumed position differs from private lease')
        return actual - 1

    def consume(self, lease, ids):
        return self._consume(lease, ids, 'feed')

    def consume_generated(self, lease, ids):
        if self.profile['execution'].get('generated_operator') == 'normal-verifier-decode':
            if len(ids) != 1:
                raise InvalidOperation('Generated decode requires one consumed token')
            return self._consume(lease, ids, 'decode')
        return self.consume(lease, ids)

    def generate_request(self, lease, max_steps):
        self._check(lease)
        execution = self.profile['execution']
        if (execution.get('native_request_protocol') != 'consumed-greedy-ack-v1'
                or execution.get('generated_operator') != 'normal-verifier-decode'):
            raise InvalidOperation('Pinned backend does not support native greedy requests')
        if getattr(self, '_protocol_failed', False):
            raise IntegrityError('Native request protocol failed; restart the backend')
        if getattr(self, '_request_active', None) is not None:
            raise InvalidOperation('A native generation request is already active')
        if type(max_steps) is not int or max_steps < 0 or max_steps > execution['capacity']-lease.consumed:
            raise InvalidOperation('Invalid native generation bound')
        if not lease.bound or lease.consumed <= 0:
            raise InvalidOperation('Native generation requires a consumed prefix')
        eos = execution.get('eos_tokens')
        if not isinstance(eos, list) or any(type(t) is not int or not 0 <= t < self.vocab_size for t in eos):
            raise IntegrityError('Invalid pinned stop-token identities')
        return _NativeRequest(self, lease, max_steps)

    def _consume(self, lease, ids, command):
        self._check(lease)
        ids = tuple(ids)
        if not ids or any(type(t) is not int or not 0 <= t < self.vocab_size for t in ids):
            raise InvalidOperation('Invalid token IDs')
        if lease.consumed + len(ids) > self.profile['execution']['capacity']:
            raise InvalidOperation('Pinned execution context exceeded')
        mixed = self._vision_enabled()
        if mixed:
            prior = lease.inputs if lease.bound else InputHistory([])
            if not isinstance(prior,InputHistory) or len(prior) != lease.consumed:
                raise IntegrityError('Active mixed history is unavailable')
            history = prior + ids
        if not lease.bound:
            self._request('reset')
            lease.bound = True
        before = self._refresh_stats()['forward_tokens']
        try:
            self._consume_completed(lease,ids,command,before,history if mixed else None)
        except BaseException:
            if mixed:
                # A failed completed receipt cannot leave partially advanced
                # native state available under the former canonical binding.
                self._protocol_failed = True
            raise

    def _consume_completed(self, lease, ids, command, before, history):
        mixed = history is not None
        try:
            response = self._request(command + ' ' + ' '.join(map(str, ids)))
        finally:
            after = self._refresh_stats()['forward_tokens']
        if after - before != len(ids) or response['consumed'] != lease.consumed + len(ids):
            raise IntegrityError('Completed native work/frontier mismatch')
        contract = self.profile['execution'].get('resume_contract')
        if contract in ('identical-native-batches-v1','identical-native-operators-v1',RESUME_CONTRACT):
            from .execution_receipt import tokens_digest, validate_schedule, validate_operator_schedule
            validate = validate_schedule if contract == 'identical-native-batches-v1' else validate_operator_schedule
            receipt = {key: response.get(key) for key in ('operator','position','count','chunks')}
            receipt['tokens_sha256'] = tokens_digest(ids)
            expected_operator = 'normal-verifier-decode' if command == 'decode' else 'batched-prefill'
            if receipt['operator'] != expected_operator:
                raise IntegrityError('Native command operator receipt mismatch')
            # Validate the native response against the request before publishing
            # its derivative receipt. Positions remain absolute in storage.
            if receipt['position'] != lease.consumed:
                raise IntegrityError('Native batch position mismatch')
            try:
                local = dict(receipt, position=0,
                             chunks=[[p-lease.consumed,n] for p,n in receipt['chunks']])
                validate([local],ids,self.profile['execution']['prefill_chunk'])
            except (ValueError,TypeError) as error:
                raise IntegrityError('Missing or invalid native batch receipt') from error
            self.last_execution_receipt = receipt
            if mixed:
                self._bind_input_receipt(lease,history,receipt)
        lease.consumed = response['consumed']
        lease.image_identity = None

    def logits(self, lease):
        self._check(lease)
        self._sequence += 1
        path = self._directory / ('frontier-' + str(self._sequence) + '.f32')
        response = self._request('logits ' + str(path))
        raw = path.read_bytes()
        path.unlink()
        if response['bytes'] != len(raw):
            raise IntegrityError('Native frontier byte counter differs from file')
        self.stats['frontier_read_bytes'] += len(raw)
        _trace('CTT_NATIVE_TRACE', dict(pid=self.process.pid, logits_sha256=hashlib.sha256(raw).hexdigest(), consumed=lease.consumed))
        return _unpack_frontier(raw, self.vocab_size)

    def choose_legal(self, lease, mask):
        """Read-only finite-logit argmax on the native consumed frontier."""
        self._check(lease)
        width = (self.vocab_size + 7) // 8
        remainder = self.vocab_size % 8
        if (self.profile['execution'].get('grammar_selection_protocol') != 'finite-mask-argmax-v1'
                or not lease.bound or lease.consumed <= 0
                or not isinstance(mask, bytes) or len(mask) != width or not any(mask)
                or (remainder and mask[-1] >> remainder)):
            raise InvalidOperation('Invalid native grammar selection capability, frontier or mask')
        self._sequence += 1
        path = self._directory / ('grammar-mask-' + str(self._sequence) + '.bin')
        try:
            path.write_bytes(mask)
            response = self._request('select_mask ' + str(lease.consumed) + ' ' + str(path))
        finally:
            path.unlink(missing_ok=True)
        token = response.get('token')
        fields = {'consumed': lease.consumed, 'logits_count': self.vocab_size, 'mask_bytes': width}
        if (response.get('selection_protocol') != 'finite-mask-argmax-v1'
                or any(type(response.get(key)) is not int or response[key] != value
                       for key, value in fields.items())
                or type(token) is not int or not 0 <= token < self.vocab_size
                or not (mask[token >> 3] & (1 << (token & 7)))):
            raise IntegrityError('Native grammar selection receipt differs from the consumed frontier/mask')
        _trace('CTT_NATIVE_TRACE', dict(pid=self.process.pid, selected_token=token,
            grammar_mask_sha256=hashlib.sha256(mask).hexdigest(), consumed=lease.consumed))
        return token

    def sample_legal(self,lease,mask,temperature,min_p,draw53,*,top_k=0,top_p=1.0,
                     dry_multiplier=0.,dry_base=1.75,dry_allowed_length=2,
                     dry_penalty_last_n=256,dry_history=(),dry_breakers=(),
                     typical_p=1.,dynatemp_range=0.,dynatemp_exponent=1.):
        """Read-only full-support Min-P choice; branch RNG remains on the host."""
        from .min_p import (validate_policy,MIN_P_PROTOCOL,SAMPLING_PROTOCOL_V2,DRY_PROTOCOL,
                            SAMPLING_PROTOCOL_V4,SAMPLING_PROTOCOL_V6,SAMPLING_PROTOCOL_V7,
                            dry_enabled,entropy_enabled,supports_policy)
        from .sampling_proposal import XTC_PROPOSAL_PROTOCOL
        self._check(lease)
        width=(self.vocab_size+7)//8
        remainder=self.vocab_size%8
        capability = self.profile['execution'].get('sampling_protocol')
        if (capability not in (MIN_P_PROTOCOL,SAMPLING_PROTOCOL_V2,DRY_PROTOCOL,SAMPLING_PROTOCOL_V4,
                               XTC_PROPOSAL_PROTOCOL,SAMPLING_PROTOCOL_V6,SAMPLING_PROTOCOL_V7)
                or not lease.bound or lease.consumed<=0
                or type(mask) is not bytes or len(mask)!=width or not any(mask)
                or (remainder and mask[-1]>>remainder)
                or type(draw53) is not int or not 0<=draw53<(1<<53)):
            raise InvalidOperation('Invalid Min-P capability, frontier, draw or mask')
        try:
            policy=validate_policy(dict(temperature=temperature,min_p=min_p,
                top_k=top_k,top_p=top_p,repetition_penalty=1.0,
                dry_multiplier=dry_multiplier,dry_base=dry_base,
                dry_allowed_length=dry_allowed_length,dry_penalty_last_n=dry_penalty_last_n,
                typical_p=typical_p,dynatemp_range=dynatemp_range,dynatemp_exponent=dynatemp_exponent))
        except (ValueError,TypeError) as exc:
            raise InvalidOperation('Invalid native Min-P numeric policy') from exc
        if not supports_policy(capability, policy):
            raise InvalidOperation('Native sampler does not support this generation policy')
        if policy['temperature'] == 0 and draw53 != 0:
            raise InvalidOperation('Greedy selection requires the zero draw sentinel')
        active_dry=dry_enabled(policy)
        active_entropy=entropy_enabled(policy)
        mixed_window=None
        if active_dry and self.profile['execution'].get('resume_contract')==RESUME_CONTRACT:
            from .vision_dry import sample,sampling_window
            if not active_entropy:return sample(self,lease,mask,policy,draw53,dry_history,dry_breakers)
            mixed_window=sampling_window(self,lease,policy,dry_history,dry_breakers,SAMPLING_PROTOCOL_V4)
        if active_entropy and self.profile['execution'].get('resume_contract')==RESUME_CONTRACT:
            from .vision_controllers import has_entropy_command
            if not has_entropy_command(self):
                raise InvalidOperation('Mixed entropy sampling requires the checked native V4 command')
        if not isinstance(dry_history,(tuple,list)) or not isinstance(dry_breakers,(tuple,list)):
            raise InvalidOperation('DRY inputs must be token sequences')
        if (any(type(t) is not int or not 0<=t<self.vocab_size for t in (*dry_history,*dry_breakers))
                or list(dry_breakers)!=sorted(set(dry_breakers))
                or not set(dry_breakers)<=set(dry_history)
                or (active_dry and len(dry_history)!=(len(mixed_window['tokens']) if mixed_window is not None
                    else min(lease.consumed,policy['dry_penalty_last_n'])))
                or (not active_dry and (dry_history or dry_breakers))):
            raise InvalidOperation('DRY inputs differ from the bounded consumed-token window')
        protocol=(SAMPLING_PROTOCOL_V4 if active_entropy else DRY_PROTOCOL if active_dry else
                  SAMPLING_PROTOCOL_V2 if capability in (DRY_PROTOCOL,SAMPLING_PROTOCOL_V4,
                                                           XTC_PROPOSAL_PROTOCOL,SAMPLING_PROTOCOL_V6,
                                                           SAMPLING_PROTOCOL_V7) else capability)
        self._sequence+=1
        path=self._directory/('sample-mask-'+str(self._sequence)+'.bin')
        history_path=path.with_suffix('.history')
        breakers_path=path.with_suffix('.breakers')
        try:
            path.write_bytes(mask)
            command = ('sample_mask_v4 ' if active_entropy else 'sample_mask_v3 ' if active_dry else
                       'sample_mask_v2 ' if protocol == SAMPLING_PROTOCOL_V2 else 'sample_mask ')
            command += str(lease.consumed)+' '+repr(policy['temperature'])+' '+repr(policy['min_p'])
            if protocol in (SAMPLING_PROTOCOL_V2,DRY_PROTOCOL,SAMPLING_PROTOCOL_V4):
                command += ' '+str(policy['top_k'])+' '+repr(policy['top_p'])
            command+=' '+str(draw53)+' '+json.dumps(str(path))
            if active_dry or active_entropy:
                history_path.write_bytes(struct.pack('<'+'I'*len(dry_history),*dry_history))
                breakers_path.write_bytes(struct.pack('<'+'I'*len(dry_breakers),*dry_breakers))
                command+=' '+repr(policy['dry_multiplier'])+' '+repr(policy['dry_base'])
                command+=' '+str(policy['dry_allowed_length'])+' '+str(policy['dry_penalty_last_n'])
                command+=' '+json.dumps(str(history_path))+' '+json.dumps(str(breakers_path))
            if active_entropy:
                command+=' '+repr(policy['typical_p'])+' '+repr(policy['dynatemp_range'])
                command+=' '+repr(policy['dynatemp_exponent'])
            if mixed_window is None:response=self._request(command)
            else:
                from .vision_dry import sampling_request
                response=sampling_request(self,lease,command,mixed_window)
        finally:
            path.unlink(missing_ok=True)
            history_path.unlink(missing_ok=True)
            breakers_path.unlink(missing_ok=True)
        token=response.get('token')
        legal_count=int.from_bytes(mask,'little').bit_count()
        fields=dict(consumed=lease.consumed,logits_count=self.vocab_size,
                    mask_bytes=width,legal_count=legal_count,draw53=draw53)
        kept=response.get('kept_count')
        if active_dry or active_entropy:
            fields.update(dry_history_count=len(dry_history),dry_breaker_count=len(dry_breakers),
                          dry_allowed_length=policy['dry_allowed_length'],dry_penalty_last_n=policy['dry_penalty_last_n'])
            if any(type(response.get(key)) not in (int,float) or response[key]!=policy[key]
                   for key in ('dry_multiplier','dry_base')):
                raise IntegrityError('Native DRY receipt differs from the generation policy')
            affected=response.get('dry_penalized_candidates')
            if type(affected) is not int or not 0<=affected<=len(set(dry_history)):
                raise IntegrityError('Native DRY receipt has an invalid affected-candidate count')
        if active_entropy:
            for key in ('typical_p','dynatemp_range','dynatemp_exponent'):
                if type(response.get(key)) not in (int,float) or response[key]!=policy[key]:
                    raise IntegrityError('Native entropy receipt differs from the generation policy')
            effective=response.get('effective_temperature')
            typical_count=response.get('typical_kept_count')
            if (type(effective) not in (int,float) or not math.isfinite(effective)
                    or not max(0.,policy['temperature']-policy['dynatemp_range'])<=effective<=
                               policy['temperature']+policy['dynatemp_range']
                    or (policy['dynatemp_range']==0 and effective!=policy['temperature'])
                    or type(typical_count) is not int or not 1<=typical_count<=legal_count
                    or type(kept) is not int or not 1<=kept<=typical_count):
                raise IntegrityError('Native entropy receipt has invalid temperature or support')
        if (response.get('protocol')!=protocol or response.get('operator')!='full-vocab-min-p'
                or (protocol in (SAMPLING_PROTOCOL_V2,DRY_PROTOCOL,SAMPLING_PROTOCOL_V4) and
                    (type(response.get('top_k')) is not int or response['top_k'] != policy['top_k']
                     or type(response.get('top_p')) not in (int,float) or response['top_p'] != policy['top_p']))
                or any(type(response.get(key)) is not int or response[key]!=value for key,value in fields.items())
                or type(kept) is not int or not 1<=kept<=legal_count
                or type(token) is not int or not 0<=token<self.vocab_size
                or not (mask[token>>3]&(1<<(token&7)))):
            raise IntegrityError('Native Min-P receipt differs from frontier, draw or support')
        event=dict(pid=self.process.pid,selected_token=token,
            sampling_protocol=protocol,draw53=draw53,consumed=lease.consumed,
            top_k=policy['top_k'],top_p=policy['top_p'],
            legal_count=legal_count,kept_count=kept,mask_sha256=hashlib.sha256(mask).hexdigest())
        if active_dry:
            event.update({key:policy[key] for key in ('dry_multiplier','dry_base','dry_allowed_length','dry_penalty_last_n')})
            event.update(dry_history_sha256=hashlib.sha256(struct.pack('<'+'I'*len(dry_history),*dry_history)).hexdigest(),
                         dry_breakers=list(dry_breakers),dry_penalized_candidates=affected)
        if active_entropy:
            event.update({key:policy[key] for key in ('typical_p','dynatemp_range','dynatemp_exponent')})
            event.update(effective_temperature=effective,typical_kept_count=typical_count)
        _trace('CTT_NATIVE_TRACE',event)
        return token

    def sample_selection_proposal(self, lease, mask, temperature, min_p, *,
                                  draw_start, draw_candidates, source_digest,
                                  top_k=0, top_p=1.0,
                                  mirostat=0, mirostat_tau=5.0, mirostat_eta=0.1,
                                  adaptive_target=-1.0, adaptive_decay=0.90,
                                  sampler_state=None,
                                  dry_multiplier=0.0, dry_base=1.75,
                                  dry_allowed_length=2, dry_penalty_last_n=256,
                                  dry_history=(), dry_breakers=(),
                                  typical_p=1.0, dynatemp_range=0.0,
                                  dynatemp_exponent=1.0,
                                  xtc_probability=0.0, xtc_threshold=0.1):
        """Return a checked V5/V6/V7 selection proposal without mutating state."""
        from .min_p import (
            validate_policy, supports_policy, dry_enabled, xtc_enabled,
            SAMPLING_PROTOCOL_V6, SAMPLING_PROTOCOL_V7, mirostat_enabled,
            adaptive_enabled,
        )
        from .sampling_proposal import (
            DRAW_LIMIT, XTC_PROPOSAL_PROTOCOL, MIROSTAT2_PROPOSAL_PROTOCOL,
            ADAPTIVE_P_PROPOSAL_PROTOCOL,
            SelectionProposal,
        )

        self._check(lease)
        width = (self.vocab_size + 7) // 8
        remainder = self.vocab_size % 8
        capability = self.profile['execution'].get('sampling_protocol')
        if (capability not in (XTC_PROPOSAL_PROTOCOL, SAMPLING_PROTOCOL_V6,
                               SAMPLING_PROTOCOL_V7, ADAPTIVE_P_PROPOSAL_PROTOCOL)
                or not lease.bound or lease.consumed <= 0
                or type(mask) is not bytes or len(mask) != width or not any(mask)
                or (remainder and mask[-1] >> remainder)):
            raise InvalidOperation('Invalid V5 sampler capability, frontier or mask')
        if (type(draw_start) is not int or not 0 <= draw_start < DRAW_LIMIT
                or type(draw_candidates) is not tuple or len(draw_candidates) > 2
                or any(type(draw) is not int or not 0 <= draw < (1 << 53)
                       for draw in draw_candidates)
                or draw_start + len(draw_candidates) >= DRAW_LIMIT
                or type(source_digest) is not str or len(source_digest) != 64
                or any(char not in '0123456789abcdef' for char in source_digest)):
            raise InvalidOperation('Invalid V5 proposal draw binding')
        try:
            policy = validate_policy(dict(
                temperature=temperature, min_p=min_p, top_k=top_k, top_p=top_p,
                repetition_penalty=1.0,
                mirostat=mirostat, mirostat_tau=mirostat_tau, mirostat_eta=mirostat_eta,
                dry_multiplier=dry_multiplier, dry_base=dry_base,
                dry_allowed_length=dry_allowed_length,
                dry_penalty_last_n=dry_penalty_last_n,
                typical_p=typical_p, dynatemp_range=dynatemp_range,
                dynatemp_exponent=dynatemp_exponent,
                xtc_probability=xtc_probability, xtc_threshold=xtc_threshold,
                adaptive_target=adaptive_target, adaptive_decay=adaptive_decay))
        except (TypeError, ValueError) as exc:
            raise InvalidOperation('Invalid V5 sampler policy') from exc
        if not supports_policy(capability, policy):
            raise InvalidOperation('Pinned native sampler does not support this policy')
        if self.profile['execution'].get('resume_contract')==RESUME_CONTRACT:
            if dry_enabled(policy):
                # VC07 deferred: native text/image projection must replace the
                # model-cell tail for this family before this guard can open.
                # Preserve the existing variable-draw/feedback validation below.
                # Future community PRs are welcome for image DRY+XTC: reuse the
                # existing samplers and isolated native fixture, then prove
                # exact draw cursors, image barriers and restart isolation.
                raise InvalidOperation('Mixed DRY proposal combinations are not yet qualified')
            from .vision_controllers import has_xtc_command,has_mirostat_command,has_adaptive_command
            if adaptive_enabled(policy) and not has_adaptive_command(self):
                raise InvalidOperation('Mixed Adaptive-P sampling requires the checked native V7 feedback command')
            if mirostat_enabled(policy) and not has_mirostat_command(self):
                raise InvalidOperation('Mixed Mirostat sampling requires the checked native V6 feedback command')
            if not any(check(policy) for check in (mirostat_enabled,adaptive_enabled)) and not has_xtc_command(self):
                raise InvalidOperation('Mixed proposal sampling requires the checked native V5 command')
        active_mirostat = mirostat_enabled(policy)
        active_adaptive = adaptive_enabled(policy)
        if active_mirostat:
            if (capability not in (SAMPLING_PROTOCOL_V6, SAMPLING_PROTOCOL_V7)
                    or (capability == SAMPLING_PROTOCOL_V7
                        and self._ready.get('mirostat2_protocol') != MIROSTAT2_PROPOSAL_PROTOCOL)):
                raise InvalidOperation('Mirostat v2 requires the pinned V6 native sampler')
            return self._sample_mirostat2_proposal(
                lease, mask, policy, draw_start=draw_start,
                draw_candidates=draw_candidates, source_digest=source_digest,
                sampler_state=sampler_state, width=width,
                dry_history=dry_history, dry_breakers=dry_breakers)
        if active_adaptive:
            if (capability not in (SAMPLING_PROTOCOL_V7, ADAPTIVE_P_PROPOSAL_PROTOCOL)
                    or self._ready.get('adaptive_p_protocol') != ADAPTIVE_P_PROPOSAL_PROTOCOL):
                raise InvalidOperation('Adaptive-P requires the pinned V7 native sampler')
            return self._sample_adaptive_p_proposal(
                lease, mask, policy, draw_start=draw_start,
                draw_candidates=draw_candidates, source_digest=source_digest,
                sampler_state=sampler_state, width=width,
                dry_history=dry_history, dry_breakers=dry_breakers)
        if (capability == SAMPLING_PROTOCOL_V7
                and self._ready.get('sampling_v5_protocol') != XTC_PROPOSAL_PROTOCOL):
            raise InvalidOperation('V7 profile lacks its retained V5 proposal command capability')
        if sampler_state is not None:
            raise InvalidOperation('Stateless sampling does not accept Mirostat state')
        if capability == SAMPLING_PROTOCOL_V6 and mirostat != 0:
            raise InvalidOperation('Unsupported Mirostat mode')
        if capability in (SAMPLING_PROTOCOL_V7, ADAPTIVE_P_PROPOSAL_PROTOCOL) and mirostat != 0:
            raise InvalidOperation('Unsupported Mirostat mode for stateless V7 proposal')
        if type(dry_history) not in (tuple, list) or type(dry_breakers) not in (tuple, list):
            raise InvalidOperation('DRY inputs must be token sequences')
        if (any(type(token) is not int or not 0 <= token < self.vocab_size
                for token in (*dry_history, *dry_breakers))
                or list(dry_breakers) != sorted(set(dry_breakers))
                or not set(dry_breakers) <= set(dry_history)
                or (dry_enabled(policy)
                    and len(dry_history) != min(lease.consumed, policy['dry_penalty_last_n']))
                or (not dry_enabled(policy) and (dry_history or dry_breakers))):
            raise InvalidOperation('DRY inputs differ from bounded consumed-token window')
        if policy['temperature'] == 0:
            token = self.sample_legal(
                lease, mask, 0.0, policy['min_p'], 0,
                top_k=policy['top_k'], top_p=policy['top_p'],
                dry_multiplier=policy['dry_multiplier'], dry_base=policy['dry_base'],
                dry_allowed_length=policy['dry_allowed_length'],
                dry_penalty_last_n=policy['dry_penalty_last_n'],
                dry_history=dry_history, dry_breakers=dry_breakers,
                typical_p=policy['typical_p'],
                dynatemp_range=policy['dynatemp_range'],
                dynatemp_exponent=policy['dynatemp_exponent'])
            return SelectionProposal(token, draw_start, 0, None, source_digest)
        if not draw_candidates:
            raise InvalidOperation('Positive-temperature V5 selection requires a first draw')

        first_draw = draw_candidates[0]
        second_draw = draw_candidates[1] if len(draw_candidates) == 2 else 1 << 53
        active_dry = dry_enabled(policy)
        configured_xtc = xtc_enabled(policy)
        self._sequence += 1
        path = self._directory / ('sample-mask-v5-' + str(self._sequence) + '.bin')
        history_path = path.with_suffix('.history')
        breakers_path = path.with_suffix('.breakers')
        try:
            path.write_bytes(mask)
            history_path.write_bytes(struct.pack('<' + 'I' * len(dry_history), *dry_history))
            breakers_path.write_bytes(struct.pack('<' + 'I' * len(dry_breakers), *dry_breakers))
            command = 'sample_mask_v5 ' + str(lease.consumed)
            command += ' ' + repr(policy['temperature']) + ' ' + repr(policy['min_p'])
            command += ' ' + str(policy['top_k']) + ' ' + repr(policy['top_p'])
            command += ' ' + str(first_draw) + ' ' + json.dumps(str(path))
            command += ' ' + repr(policy['dry_multiplier']) + ' ' + repr(policy['dry_base'])
            command += ' ' + str(policy['dry_allowed_length']) + ' ' + str(policy['dry_penalty_last_n'])
            command += ' ' + json.dumps(str(history_path)) + ' ' + json.dumps(str(breakers_path))
            command += ' ' + repr(policy['typical_p']) + ' ' + repr(policy['dynatemp_range'])
            command += ' ' + repr(policy['dynatemp_exponent'])
            command += ' ' + repr(policy['xtc_probability']) + ' ' + repr(policy['xtc_threshold'])
            command += ' ' + str(second_draw)
            response = self._request(command)
        finally:
            path.unlink(missing_ok=True)
            history_path.unlink(missing_ok=True)
            breakers_path.unlink(missing_ok=True)

        legal_count = int.from_bytes(mask, 'little').bit_count()
        token = response.get('token')
        fields = dict(
            consumed=lease.consumed, logits_count=self.vocab_size,
            mask_bytes=width, legal_count=legal_count, draw53=first_draw,
            dry_history_count=len(dry_history), dry_breaker_count=len(dry_breakers),
            dry_allowed_length=policy['dry_allowed_length'],
            dry_penalty_last_n=policy['dry_penalty_last_n'],
        )
        for key in ('dry_multiplier', 'dry_base', 'typical_p',
                    'dynatemp_range', 'dynatemp_exponent',
                    'xtc_probability', 'xtc_threshold'):
            value = policy[key]
            received = response.get(key)
            if (type(received) not in (int, float) or not math.isfinite(received)
                    or received != value):
                raise IntegrityError('Native V5 receipt differs from generation policy: ' + key)
        affected = response.get('dry_penalized_candidates')
        if type(affected) is not int or not 0 <= affected <= len(set(dry_history)):
            raise IntegrityError('Native V5 receipt has invalid DRY affected-candidate count')
        pre_xtc_count = response.get('pre_xtc_count')
        kept = response.get('kept_count')
        removed = response.get('removed_count')
        activation_draws = response.get('activation_draws')
        activated = response.get('activated')
        draw_second = response.get('draw53_second')
        draw_count = response.get('draw_count')
        effective = response.get('effective_temperature')
        typical_count = response.get('typical_kept_count')
        min_p_count = response.get('min_p_kept_count')
        top_k_count = response.get('top_k_kept_count')
        if (type(effective) not in (int, float) or not math.isfinite(effective)
                or not max(0.0, policy['temperature'] - policy['dynatemp_range']) <= effective
                <= policy['temperature'] + policy['dynatemp_range']
                or (policy['dynatemp_range'] == 0 and effective != policy['temperature'])
                or type(min_p_count) is not int or not 1 <= min_p_count <= legal_count
                or type(top_k_count) is not int or not 1 <= top_k_count <= min_p_count
                or (policy['top_k'] == 0 and top_k_count != min_p_count)
                or (policy['top_k'] > 0 and top_k_count > policy['top_k'])
                or type(typical_count) is not int or not 1 <= typical_count <= top_k_count
                or type(pre_xtc_count) is not int or not 1 <= pre_xtc_count <= typical_count
                or type(kept) is not int or type(removed) is not int
                or not 1 <= kept <= pre_xtc_count or removed < 0
                or kept != pre_xtc_count - removed
                or type(activated) is not bool
                or type(activation_draws) is not int or activation_draws not in (0, 1)
                or type(draw_count) is not int or draw_count != 1 + activation_draws
                or draw_count > len(draw_candidates)
                or type(draw_second) is not int or draw_second != second_draw
                or type(response.get('total_weight')) not in (int, float)
                or not math.isfinite(response.get('total_weight'))
                or response.get('total_weight') <= 0
                or (activation_draws != int(configured_xtc and effective > 0.0
                                             and pre_xtc_count > 1))
                or (activated and len(draw_candidates) != 2)
                or (not activated and removed != 0)):
            raise IntegrityError('Native V5 receipt has invalid support or draw accounting')
        should_activate = (
            configured_xtc and effective > 0.0 and pre_xtc_count > 1
            and first_draw / float(1 << 53) <= policy['xtc_probability'])
        if activated != should_activate:
            raise IntegrityError('Native V5 XTC activation differs from first-draw probability')
        if (response.get('status') != 'ok'
                or response.get('protocol') != XTC_PROPOSAL_PROTOCOL
                or response.get('operator') != 'full-vocab-min-p'
                or any(type(response.get(key)) is not int or response[key] != value
                       for key, value in fields.items())
                or type(response.get('top_k')) is not int
                or response['top_k'] != policy['top_k']
                or type(response.get('top_p')) not in (int, float)
                or response['top_p'] != policy['top_p']
                or type(token) is not int or not 0 <= token < self.vocab_size
                or not (mask[token >> 3] & (1 << (token & 7)))):
            raise IntegrityError('Native V5 receipt differs from frontier, policy or legal support')
        event = dict(
            pid=self.process.pid, sampling_protocol=XTC_PROPOSAL_PROTOCOL,
            selected_token=token, consumed=lease.consumed,
            draw_start=draw_start, draw_count=draw_count,
            draw53=first_draw, draw53_second=draw_second,
            xtc_probability=policy['xtc_probability'], xtc_threshold=policy['xtc_threshold'],
            activated=activated, activation_draws=activation_draws,
            pre_xtc_count=pre_xtc_count, removed_count=removed, kept_count=kept,
            mask_sha256=hashlib.sha256(mask).hexdigest(),
            source_digest=source_digest, native_receipt=dict(response))
        _trace('CTT_NATIVE_TRACE', event)
        return SelectionProposal(token, draw_start, draw_count, None, source_digest)

    def _sample_mirostat2_proposal(self, lease, mask, policy, *, draw_start,
                                   draw_candidates, source_digest, sampler_state, width,
                                   dry_history, dry_breakers):
        """Validate one V6 Mirostat selection and return uncommitted feedback."""
        from .min_p import SAMPLING_PROTOCOL_V6, SAMPLING_PROTOCOL_V7, dry_enabled
        from .mirostat2 import validate_state
        from .sampling_proposal import DRAW_LIMIT, MIROSTAT2_PROPOSAL_PROTOCOL, SelectionProposal

        try:
            state = validate_state(sampler_state)
        except (TypeError, ValueError) as exc:
            raise InvalidOperation('Invalid V6 Mirostat feedback state') from exc
        mu = state['mu']
        capability = self.profile['execution'].get('sampling_protocol')
        if (capability not in (SAMPLING_PROTOCOL_V6, SAMPLING_PROTOCOL_V7)
                or (capability == SAMPLING_PROTOCOL_V7
                    and self._ready.get('mirostat2_protocol') != MIROSTAT2_PROPOSAL_PROTOCOL)
                or policy.get('mirostat') != 2
                or policy['temperature'] <= 0
                or type(draw_start) is not int or not 0 <= draw_start < DRAW_LIMIT - 1
                or type(draw_candidates) is not tuple or not 1 <= len(draw_candidates) <= 2
                or any(type(draw) is not int or not 0 <= draw < (1 << 53)
                       for draw in draw_candidates)
                or type(source_digest) is not str or len(source_digest) != 64
                or any(char not in '0123456789abcdef' for char in source_digest)):
            raise InvalidOperation('Invalid V6 Mirostat proposal inputs')
        if (type(policy.get('mirostat')) is not int or policy['mirostat'] != 2
                or type(policy['mirostat_tau']) not in (int, float)
                or type(policy['mirostat_eta']) not in (int, float)):
            raise InvalidOperation('Invalid V6 Mirostat parameters')
        tau, eta = float(policy['mirostat_tau']), float(policy['mirostat_eta'])
        if not math.isfinite(tau) or tau <= 0 or not math.isfinite(eta) or eta < 0:
            raise InvalidOperation('Invalid V6 Mirostat parameters')
        if type(dry_history) not in (tuple, list) or type(dry_breakers) not in (tuple, list):
            raise InvalidOperation('DRY inputs must be token sequences')
        active_dry = dry_enabled(policy)
        if (any(type(token) is not int or not 0 <= token < self.vocab_size
                for token in (*dry_history, *dry_breakers))
                or list(dry_breakers) != sorted(set(dry_breakers))
                or not set(dry_breakers) <= set(dry_history)
                or (active_dry and len(dry_history) != min(lease.consumed, policy['dry_penalty_last_n']))
                or (not active_dry and (dry_history or dry_breakers))):
            raise InvalidOperation('DRY inputs differ from bounded consumed-token window')

        draw = draw_candidates[0]
        self._sequence += 1
        path = self._directory / ('sample-mask-v6-' + str(self._sequence) + '.bin')
        history_path = path.with_suffix('.history')
        breakers_path = path.with_suffix('.breakers')
        try:
            path.write_bytes(mask)
            history_path.write_bytes(struct.pack('<' + 'I' * len(dry_history), *dry_history))
            breakers_path.write_bytes(struct.pack('<' + 'I' * len(dry_breakers), *dry_breakers))
            command = 'sample_mask_mirostat2 ' + str(lease.consumed)
            command += ' ' + repr(policy['temperature']) + ' ' + str(draw)
            command += ' ' + json.dumps(str(path)) + ' ' + repr(tau) + ' ' + repr(eta)
            command += ' ' + repr(mu) + ' ' + repr(policy['dry_multiplier'])
            command += ' ' + repr(policy['dry_base']) + ' ' + str(policy['dry_allowed_length'])
            command += ' ' + str(policy['dry_penalty_last_n'])
            command += ' ' + json.dumps(str(history_path)) + ' ' + json.dumps(str(breakers_path))
            response = self._request(command)
        finally:
            path.unlink(missing_ok=True)
            history_path.unlink(missing_ok=True)
            breakers_path.unlink(missing_ok=True)

        legal_count = int.from_bytes(mask, 'little').bit_count()
        token = response.get('token')

        def integer(name, expected):
            value = response.get(name)
            if type(value) is not int or value != expected:
                raise IntegrityError('Native V6 receipt differs from ' + name)

        def number(name, expected=None, *, minimum=None, maximum=None):
            value = response.get(name)
            if type(value) not in (int, float):
                raise IntegrityError('Native V6 receipt has invalid ' + name)
            try:
                value = float(value)
            except (OverflowError, ValueError) as exc:
                raise IntegrityError('Native V6 receipt has invalid ' + name) from exc
            if not math.isfinite(value):
                raise IntegrityError('Native V6 receipt has invalid ' + name)
            if ((expected is not None and value != expected)
                    or (minimum is not None and value < minimum)
                    or (maximum is not None and value > maximum)):
                raise IntegrityError('Native V6 receipt differs from ' + name)
            return value

        if (response.get('status') != 'ok'
                or response.get('protocol') != MIROSTAT2_PROPOSAL_PROTOCOL
                or response.get('operator') != 'full-vocab-mirostat2'
                or response.get('feedback_protocol') != 'mirostat2-state-v1'):
            raise IntegrityError('Native V6 Mirostat receipt has the wrong protocol')
        integer('feedback_version', 1)
        integer('consumed', lease.consumed)
        integer('logits_count', self.vocab_size)
        integer('mask_bytes', width)
        integer('legal_count', legal_count)
        integer('draw53', draw)
        integer('draw_count', 1)
        integer('dry_history_count', len(dry_history))
        integer('dry_breaker_count', len(dry_breakers))
        integer('dry_allowed_length', policy['dry_allowed_length'])
        integer('dry_penalty_last_n', policy['dry_penalty_last_n'])
        for key, expected in (('temperature', policy['temperature']), ('tau', tau),
                              ('eta', eta), ('mu', mu),
                              ('dry_multiplier', policy['dry_multiplier']),
                              ('dry_base', policy['dry_base'])):
            number(key, float(expected))
        affected = response.get('dry_penalized_candidates')
        if type(affected) is not int or not 0 <= affected <= len(set(dry_history)):
            raise IntegrityError('Native V6 receipt has invalid DRY adjustment count')
        legal_total = number('legal_total_weight', minimum=0.0)
        total_weight = number('total_weight', minimum=0.0)
        if legal_total <= 0.0 or total_weight <= 0.0:
            raise IntegrityError('Native V6 receipt has empty categorical mass')
        kept = response.get('kept_count')
        if type(kept) is not int or not 1 <= kept <= legal_count:
            raise IntegrityError('Native V6 receipt has invalid retained support')
        logp = number('selected_log_probability', maximum=0.0)
        selected_probability = number('selected_probability', minimum=0.0, maximum=1.0)
        expected_probability = math.exp(logp)
        if selected_probability != expected_probability:
            raise IntegrityError('Native V6 probability does not match selected log probability')
        surprise = number('surprise_bits', minimum=0.0)
        expected_surprise = -logp / math.log(2.0)
        if surprise != expected_surprise:
            raise IntegrityError('Native V6 surprise does not match selected log probability')
        next_mu = number('next_mu')
        update_error = surprise - tau
        update = eta * update_error
        expected_mu = mu - update
        if next_mu != expected_mu:
            raise IntegrityError('Native V6 feedback update differs from Mirostat v2')
        if (type(token) is not int or not 0 <= token < self.vocab_size
                or not (mask[token >> 3] & (1 << (token & 7)))):
            raise IntegrityError('Native V6 selected token is outside legal support')

        feedback = validate_state(dict(protocol='mirostat2-state-v1', version=1, mu=next_mu))
        event = dict(pid=self.process.pid, sampling_protocol=MIROSTAT2_PROPOSAL_PROTOCOL,
                     selected_token=token, consumed=lease.consumed, draw_start=draw_start,
                     draw_count=1, draw53=draw, source_digest=source_digest,
                     legal_count=legal_count, kept_count=kept,
                     legal_total_weight=legal_total, total_weight=total_weight,
                     selected_log_probability=logp, selected_probability=selected_probability,
                     surprise_bits=surprise, mu=mu, next_mu=next_mu,
                     dry_history_sha256=hashlib.sha256(struct.pack('<' + 'I' * len(dry_history), *dry_history)).hexdigest(),
                     dry_breakers=list(dry_breakers), dry_penalized_candidates=affected,
                     mask_sha256=hashlib.sha256(mask).hexdigest(), native_receipt=dict(response))
        _trace('CTT_NATIVE_TRACE', event)
        return SelectionProposal(token, draw_start, 1, feedback, source_digest)

    def _sample_adaptive_p_proposal(self, lease, mask, policy, *, draw_start,
                                    draw_candidates, source_digest, sampler_state, width,
                                    dry_history, dry_breakers):
        """Validate one V7 Adaptive-P selection and return private EMA feedback."""
        from .adaptive_p import validate_state
        from .min_p import SAMPLING_PROTOCOL_V7, dry_enabled, xtc_enabled
        from .sampling_proposal import (
            ADAPTIVE_P_PROPOSAL_PROTOCOL, DRAW_LIMIT, SelectionProposal,
        )

        try:
            state = validate_state(sampler_state)
        except (TypeError, ValueError) as exc:
            raise InvalidOperation('Invalid V7 Adaptive-P feedback state') from exc
        target = policy.get('adaptive_target')
        decay = policy.get('adaptive_decay')
        weighted_sum = state['weighted_sum']
        total_state_weight = state['total_weight']
        if (self.profile['execution'].get('sampling_protocol') not in
                (SAMPLING_PROTOCOL_V7, ADAPTIVE_P_PROPOSAL_PROTOCOL)
                or self._ready.get('adaptive_p_protocol') != ADAPTIVE_P_PROPOSAL_PROTOCOL
                or self._ready.get('adaptive_p_feedback_protocol') != 'adaptive-p-state-v1'
                or type(self._ready.get('adaptive_p_feedback_version')) is not int
                or self._ready.get('adaptive_p_feedback_version') != 1
                or type(policy.get('adaptive_target')) not in (int, float)
                or not math.isfinite(target) or target < 0.0 or target > 1.0
                or type(decay) not in (int, float) or not math.isfinite(decay)
                or not 0.0 <= decay <= 0.99
                or policy['temperature'] <= 0.0
                or type(draw_start) is not int or not 0 <= draw_start < DRAW_LIMIT - 1
                or type(draw_candidates) is not tuple or not 1 <= len(draw_candidates) <= 2
                or any(type(draw) is not int or not 0 <= draw < (1 << 53)
                       for draw in draw_candidates)
                or type(source_digest) is not str or len(source_digest) != 64
                or any(char not in '0123456789abcdef' for char in source_digest)):
            raise InvalidOperation('Invalid V7 Adaptive-P proposal inputs')
        # Adaptive-P is terminal: the native V7 selector first applies the
        # supported Min-P/Top-K/Top-P/Typical/Dyna/DRY pipeline, then adapts
        # that resulting distribution. Only other terminal feedback samplers
        # are mutually exclusive with it.
        if policy.get('mirostat', 0) != 0 or xtc_enabled(policy):
            raise InvalidOperation('Adaptive-P cannot be combined with Mirostat or active XTC')
        if type(dry_history) not in (tuple, list) or type(dry_breakers) not in (tuple, list):
            raise InvalidOperation('DRY inputs must be token sequences')
        active_dry = dry_enabled(policy)
        if (any(type(token) is not int or not 0 <= token < self.vocab_size
                for token in (*dry_history, *dry_breakers))
                or list(dry_breakers) != sorted(set(dry_breakers))
                or not set(dry_breakers) <= set(dry_history)
                or (active_dry and len(dry_history) != min(lease.consumed, policy['dry_penalty_last_n']))
                or (not active_dry and (dry_history or dry_breakers))):
            raise InvalidOperation('DRY inputs differ from bounded consumed-token window')

        draw = draw_candidates[0]
        self._sequence += 1
        path = self._directory / ('sample-mask-v7-' + str(self._sequence) + '.bin')
        history_path = path.with_suffix('.history')
        breakers_path = path.with_suffix('.breakers')
        try:
            path.write_bytes(mask)
            history_path.write_bytes(struct.pack('<' + 'I' * len(dry_history), *dry_history))
            breakers_path.write_bytes(struct.pack('<' + 'I' * len(dry_breakers), *dry_breakers))
            command = 'sample_mask_adaptive_p ' + str(lease.consumed)
            command += ' ' + repr(policy['temperature']) + ' ' + repr(policy['min_p'])
            command += ' ' + str(policy['top_k']) + ' ' + repr(policy['top_p'])
            command += ' ' + str(draw) + ' ' + json.dumps(str(path))
            command += ' ' + repr(policy['dry_multiplier']) + ' ' + repr(policy['dry_base'])
            command += ' ' + str(policy['dry_allowed_length']) + ' ' + str(policy['dry_penalty_last_n'])
            command += ' ' + json.dumps(str(history_path)) + ' ' + json.dumps(str(breakers_path))
            command += ' ' + repr(policy['typical_p']) + ' ' + repr(policy['dynatemp_range'])
            command += ' ' + repr(policy['dynatemp_exponent']) + ' ' + repr(target) + ' ' + repr(decay)
            command += ' ' + repr(weighted_sum) + ' ' + repr(total_state_weight)
            response = self._request(command)
        finally:
            path.unlink(missing_ok=True)
            history_path.unlink(missing_ok=True)
            breakers_path.unlink(missing_ok=True)

        def integer(name, expected):
            value = response.get(name)
            if type(value) is not int or value != expected:
                raise IntegrityError('Native V7 receipt differs from ' + name)

        def number(name, *, expected=None, minimum=None, maximum=None):
            value = response.get(name)
            if type(value) not in (int, float):
                raise IntegrityError('Native V7 receipt has invalid ' + name)
            try:
                value = float(value)
            except (OverflowError, ValueError) as exc:
                raise IntegrityError('Native V7 receipt has invalid ' + name) from exc
            if (not math.isfinite(value)
                    or (expected is not None and value != expected)
                    or (minimum is not None and value < minimum)
                    or (maximum is not None and value > maximum)):
                raise IntegrityError('Native V7 receipt differs from ' + name)
            return value

        token = response.get('token')
        legal_count = int.from_bytes(mask, 'little').bit_count()
        if (response.get('status') != 'ok'
                or response.get('protocol') != ADAPTIVE_P_PROPOSAL_PROTOCOL
                or response.get('operator') != 'full-vocab-adaptive-p'
                or response.get('feedback_protocol') != 'adaptive-p-state-v1'):
            raise IntegrityError('Native V7 Adaptive-P receipt has the wrong protocol')
        integer('feedback_version', 1)
        integer('consumed', lease.consumed)
        integer('logits_count', self.vocab_size)
        integer('mask_bytes', width)
        integer('legal_count', legal_count)
        integer('draw53', draw)
        integer('draw_count', 1)
        integer('dry_history_count', len(dry_history))
        integer('dry_breaker_count', len(dry_breakers))
        integer('dry_allowed_length', policy['dry_allowed_length'])
        integer('dry_penalty_last_n', policy['dry_penalty_last_n'])
        integer('top_k', policy['top_k'])
        for key, expected in (
                ('temperature', policy['temperature']), ('min_p', policy['min_p']),
                ('top_p', policy['top_p']), ('typical_p', policy['typical_p']),
                ('dynatemp_range', policy['dynatemp_range']),
                ('dynatemp_exponent', policy['dynatemp_exponent']),
                ('dry_multiplier', policy['dry_multiplier']), ('dry_base', policy['dry_base']),
                ('adaptive_target', target), ('adaptive_decay', decay),
                ('state_weighted_sum', weighted_sum), ('state_total_weight', total_state_weight)):
            number(key, expected=float(expected))
        affected = response.get('dry_penalized_candidates')
        if type(affected) is not int or not 0 <= affected <= len(set(dry_history)):
            raise IntegrityError('Native V7 receipt has invalid DRY adjustment count')
        min_p_count = response.get('min_p_kept_count')
        top_k_count = response.get('top_k_kept_count')
        typical_count = response.get('typical_kept_count')
        kept = response.get('kept_count')
        if (type(min_p_count) is not int or not 1 <= min_p_count <= legal_count
                or type(top_k_count) is not int or not 1 <= top_k_count <= min_p_count
                or (policy['top_k'] == 0 and top_k_count != min_p_count)
                or (policy['top_k'] > 0 and top_k_count > policy['top_k'])
                or type(typical_count) is not int or not 1 <= typical_count <= top_k_count
                or type(kept) is not int or not 1 <= kept <= typical_count):
            raise IntegrityError('Native V7 receipt has invalid support counts')
        effective = number('effective_temperature', minimum=0.0)
        if (not max(0.0, policy['temperature'] - policy['dynatemp_range']) <= effective
                <= policy['temperature'] + policy['dynatemp_range']
                or (policy['dynatemp_range'] == 0.0 and effective != policy['temperature'])):
            raise IntegrityError('Native V7 effective temperature differs from policy bounds')
        original_probability = number('original_probability', minimum=0.0, maximum=1.0)
        adapted_target = number('adapted_target', minimum=0.0, maximum=1.0)
        expected_target = min(1.0, max(0.0, 2.0 * target - weighted_sum / total_state_weight))
        if adapted_target != expected_target:
            raise IntegrityError('Native V7 adapted target differs from feedback state')
        categorical_weight = number('categorical_total_weight', minimum=0.0)
        if categorical_weight <= 0.0:
            raise IntegrityError('Native V7 categorical mass is empty')
        next_weighted_sum = number('next_weighted_sum')
        next_total_weight = number('next_total_weight', minimum=0.0)
        expected_weighted_sum = original_probability + decay * weighted_sum
        expected_total_weight = 1.0 + decay * total_state_weight
        if (next_weighted_sum != expected_weighted_sum
                or next_total_weight != expected_total_weight):
            raise IntegrityError('Native V7 feedback update differs from Adaptive-P')
        if (type(token) is not int or not 0 <= token < self.vocab_size
                or not (mask[token >> 3] & (1 << (token & 7)))):
            raise IntegrityError('Native V7 selected token is outside legal support')

        feedback = validate_state(dict(version='adaptive-p-state-v1',
                                       weighted_sum=next_weighted_sum,
                                       total_weight=next_total_weight))
        event = dict(pid=self.process.pid, sampling_protocol=ADAPTIVE_P_PROPOSAL_PROTOCOL,
                     selected_token=token, consumed=lease.consumed, draw_start=draw_start,
                     draw_count=1, draw53=draw, source_digest=source_digest,
                     legal_count=legal_count, min_p_kept_count=min_p_count,
                     top_k_kept_count=top_k_count, typical_kept_count=typical_count,
                     kept_count=kept, categorical_total_weight=categorical_weight,
                     original_probability=original_probability, adapted_target=adapted_target,
                     adaptive_target=target, adaptive_decay=decay,
                     state_weighted_sum=weighted_sum, state_total_weight=total_state_weight,
                     next_weighted_sum=next_weighted_sum, next_total_weight=next_total_weight,
                     dry_history_sha256=hashlib.sha256(struct.pack('<' + 'I' * len(dry_history), *dry_history)).hexdigest(),
                     dry_breakers=list(dry_breakers), dry_penalized_candidates=affected,
                     mask_sha256=hashlib.sha256(mask).hexdigest(), native_receipt=dict(response))
        _trace('CTT_NATIVE_TRACE', event)
        return SelectionProposal(token, draw_start, 1, feedback, source_digest)

    def export(self, lease):
        self._check(lease)
        input_identity = ''
        if self._vision_enabled():
            if not isinstance(lease.inputs,InputHistory) or len(lease.inputs) != lease.consumed:
                raise IntegrityError('Capture requires checked canonical mixed inputs')
            self._check_input_binding(lease.inputs)
            input_identity = lease.inputs.digest
        before = self._refresh_stats()['saved_running_bytes']
        response = self._request('capture')
        after = self._refresh_stats()['saved_running_bytes']
        if response['consumed'] != lease.consumed or after <= before:
            raise IntegrityError('Incomplete consumed native capture')
        identity = self._process_identity + struct.pack('<QQ', response['handle'], response['consumed'])
        pages = tuple(tuple(row) for row in self._request('pages')['pages'])
        image = StrataImage(identity, self._process_identity, response['handle'], response['consumed'],
                            after - before, 4 * self.vocab_size, pages, input_identity)
        weakref.finalize(image, _drop_image, weakref.ref(self), image.handle)
        lease.image_identity = identity
        return image

    def park(self, lease):
        self._check(lease)
        if not lease.bound or lease.consumed <= 0:
            raise InvalidOperation('RAM parking requires a consumed native lease')
        response = self._request('park')
        if response['consumed'] != lease.consumed:
            raise IntegrityError('Parked native history differs from checked lease')
        return response

    def restore_cached(self, lease, history):
        self._check(lease)
        history = tuple(history)
        if not history or len(history) > self.profile['execution']['capacity'] or any(
                type(t) is not int or not 0 <= t < self.vocab_size for t in history):
            raise InvalidOperation('Invalid canonical RAM-cache history')
        path = self._directory / 'ram-history.ids'
        path.write_text(' '.join(map(str, history)))
        try:
            response = self._request('activate ' + json.dumps(str(path)))
        finally:
            path.unlink(missing_ok=True)
        if not response['hit']:
            return False
        if response['consumed'] != len(history):
            raise IntegrityError('RAM restored history differs from canonical inputs')
        lease.bound, lease.consumed = True, response['consumed']
        lease.image_identity = None
        self._refresh_stats()
        return True

    def clear_cached_histories(self):
        with self._lock:
            return self._request('ram_clear')['removed']

    def restore(self, lease, state, *, history, frontier):
        self._check(lease)
        mixed = self._vision_enabled()
        before_work = None
        if mixed:
            if (not isinstance(history,InputHistory)
                    or not isinstance(state,(StrataImage,bytes))
                    or (isinstance(state,StrataImage) and state.input_identity != history.digest)):
                raise IntegrityError('Mixed restore requires a matching native image input binding')
            validate_boundary(history,capacity=self.profile['execution']['capacity'],
                vocab=self.vocab_size,width=self.profile['execution']['embedding_width'])
            if (len(frontier) != self.vocab_size or not all(map(math.isfinite,frontier))
                    or (isinstance(state,bytes) and not state)):
                raise IntegrityError('Incomplete mixed cold state or frontier')
            before_work = self._refresh_stats()['forward_tokens']
        try:
            return self._restore_completed(lease,state,history,frontier,mixed,before_work)
        except BaseException:
            if mixed:
                # A completed native load may already have changed private GPU
                # state, even when its consumed-count or frontier receipt fails.
                self._protocol_failed = True
            raise

    def _restore_completed(self, lease, state, history, frontier, mixed, before_work):
        if isinstance(state, StrataImage):
            if state.process_identity != self._process_identity or state.consumed != len(history):
                raise IntegrityError('Unavailable or incompatible native hot image')
            response = self._request('restore ' + str(state.handle))
            resident = state
        elif isinstance(state, bytes):
            path = self._directory / 'cold.state'
            ids = self._directory / 'cold.ids'
            logits = self._directory / 'cold.f32'
            try:
                path.write_bytes(state)
                if mixed:
                    images = placements(history)
                    lines = [f'COW_VISION_RESTORE_V1 {len(history)} {len(images)}',
                             ' '.join(map(str,history.native_tokens))]
                    for image in images:
                        lines.append(' '.join(str(image[key]) for key in
                            ('start','rows','grid_width','grid_height','width','pad_token','embedding_sha256')))
                    ids.write_text('\n'.join(lines)+'\n')
                else:
                    ids.write_text(' '.join(map(str, history)))
                logits.write_bytes(struct.pack('<' + str(len(frontier)) + 'f', *frontier))
                command = 'load_image ' if mixed else 'load '
                response = self._request(command + ' '.join(json.dumps(str(p)) for p in (path, ids, logits)))
            except BaseException:
                if mixed: self._protocol_failed = True
                raise
            finally:
                for p in (path, ids, logits):
                    p.unlink(missing_ok=True)
            resident = None
        else:
            raise IntegrityError('Unsupported derived state representation')
        if response['consumed'] != len(history):
            raise IntegrityError('Native restore did not consume the expected history')
        lease.bound, lease.consumed = True, response['consumed']
        self._refresh_stats()
        if mixed:
            try:
                self._check_input_binding(history)
                if self.stats['forward_tokens'] != before_work:
                    raise IntegrityError('Image restore performed forward computation')
                lease.inputs = history
            except BaseException:
                self._protocol_failed = True
                raise
        if resident is None:
            resident = self.export(lease)
        lease.image_identity = resident.identity
        return resident

    def serialize(self, lease, state):
        self._check(lease)
        if not isinstance(state, StrataImage) or lease.image_identity != state.identity:
            raise IntegrityError('Cold serialization requires the checked active image')
        if self._vision_enabled():
            if not isinstance(lease.inputs,InputHistory) or state.input_identity != lease.inputs.digest:
                raise IntegrityError('Cold serialization lost its canonical mixed input binding')
            self._check_input_binding(lease.inputs)
        path = self._directory / 'export.state'
        response = self._request('write ' + json.dumps(str(path)))
        try:
            blob = path.read_bytes()
        finally:
            path.unlink(missing_ok=True)
        if len(blob) != response['bytes']:
            raise IntegrityError('Native saved byte count differs from actual file')
        self._refresh_stats()
        return blob

    def checkpoint_metrics(self, values):
        images = {id(item.blob): item.blob for item in values}
        pages = {}
        for image in images.values():
            if not isinstance(image, StrataImage):
                continue
            for row in image.pages:
                if len(row) != 6 or row[4] not in ('k', 'v', 'k_scale', 'v_scale', 'indexer_pool') or row[5] < 0:
                    raise IntegrityError('Native ownership omits the allocation pool kind')
                allocation = (row[3], row[4], row[5])
                previous = pages.setdefault(row[0], allocation)
                if previous != allocation:
                    raise IntegrityError('One physical allocation has conflicting pool metadata')
        kv = [row[0] for row in pages.values() if row[1] != 'indexer_pool']
        index = [row[0] for row in pages.values() if row[1] == 'indexer_pool']
        host_bytes = sum(image.running_bytes if isinstance(image, StrataImage) else len(image) for image in images.values())
        return dict(owned_checkpoint_bytes=host_bytes, checkpoint_unique_kv_bytes=sum(kv),
                    checkpoint_unique_kv_pages=len(kv), checkpoint_unique_index_bytes=sum(index),
                    checkpoint_unique_index_pages=len(index),
                    checkpoint_unique_physical_bytes=sum(row[0] for row in pages.values()),
                    checkpoint_unique_physical_pages=len(pages), native_counters=dict(self.stats))

    def audit(self, state):
        """Hash bytes read from a retained native image without activating it."""
        if not isinstance(state, StrataImage) or state.process_identity != self._process_identity:
            raise IntegrityError('Audit requires an image owned by this native process')
        with self._lock:
            path = self._directory / 'image-audit.bin'
            try:
                report = self._request('audit ' + str(state.handle) + ' ' + json.dumps(str(path)))
                if report['consumed'] != state.consumed or report['bytes'] != path.stat().st_size:
                    raise IntegrityError('Audit consumed count or file extent differs')
                whole = hashlib.sha256()
                offset = 0
                with path.open('rb') as stream:
                    for region in report['regions']:
                        if region['offset'] != offset or type(region['bytes']) is not int or region['bytes'] <= 0:
                            raise IntegrityError('Invalid audit region coverage')
                        if not 0 <= region['filled_prefix_bytes'] <= region['bytes']:
                            raise IntegrityError('Invalid consumed-prefix audit coverage')
                        section = hashlib.sha256()
                        remaining = region['bytes']
                        while remaining:
                            chunk = stream.read(min(remaining, 1024 * 1024))
                            if not chunk:
                                raise IntegrityError('Incomplete physical audit bytes')
                            whole.update(chunk)
                            section.update(chunk)
                            remaining -= len(chunk)
                        region['sha256'] = section.hexdigest()
                        offset += region['bytes']
                    if offset != report['bytes'] or stream.read(1):
                        raise IntegrityError('Unaccounted physical audit bytes')
                report['sha256'] = whole.hexdigest()
                self._refresh_stats()
                _trace('CTT_NATIVE_TRACE', dict(pid=self.process.pid, retained_image_audit=report))
                return report
            finally:
                path.unlink(missing_ok=True)

    def close(self):
        if self._closed:
            return
        with self._lock:
            try:
                if self.process.poll() is None:
                    self._flush_drops()
                    self._refresh_stats()
                    self._request('quit')
                    self.process.wait(timeout=30)
            finally:
                if self.process.poll() is None:
                    self.process.kill()
                    self.process.wait()
                self.stats['contexts_freed'] = 1
                self._closed = True
                self.process.stdin.close()
                self.process.stdout.close()
                trace = os.environ.get('CTT_NATIVE_TRACE')
                if trace:
                    self._log.seek(0)
                    Path(trace + '.stderr').write_bytes(self._log.read())
                _trace('CTT_NATIVE_TRACE', dict(pid=self.process.pid, closed=True, counters=dict(self.stats),
                    native_counter_phase='last measurement before native teardown; process exit observed separately'))
                self._log.close()
                self._scratch.cleanup()


class StrataFreshOracle:
    """One new native process/empty session per full-history request."""
    def __init__(self, profile):
        _validate_execution(profile)
        self.model_verification = _validate_model(profile)
        self.profile = profile
        self.requests = 0
        self.tokens_computed = 0
        self._scratch = tempfile.TemporaryDirectory(prefix='ctt-fresh-')
        self._directory = Path(self._scratch.name)

    def forward(self, tokens, *, chunk=1):
        ids = tuple(tokens)
        if chunk != 1:
            raise InvalidOperation('Pinned Strata oracle consumes one token per operator call')
        if not ids or len(ids) > self.profile['execution']['capacity'] or any(type(t) is not int or t < 0 or t >= 248320 for t in ids):
            raise InvalidOperation('Invalid fresh oracle history')
        _validate_execution(self.profile)
        self.model_verification = _validate_model(self.profile)
        prefix = self._directory / ('forward-' + str(self.requests))
        token_path = prefix.with_suffix('.tokens')
        logits_path = prefix.with_suffix('.f32')
        token_path.write_text(','.join(map(str, ids)))
        command = self.profile['execution']['oracle_command'] + ['--tokens-file', str(token_path)]
        environment = dict(os.environ, STRATA_DUMP_FIRST_LOGITS=str(logits_path))
        start = time.monotonic()
        result = subprocess.run(command, env=environment, capture_output=True, timeout=18000)
        trace = os.environ.get('CTT_ORACLE_TRACE')
        if trace:
            Path(trace + '.' + str(self.requests) + '.stderr').write_bytes(result.stderr)
        if result.returncode:
            raise RuntimeError('Independent fresh process failed: ' + result.stderr.decode(errors='replace')[-4000:])
        work = json.loads(result.stdout)
        if work['forward_tokens'] != len(ids) or work['consumed'] != len(ids) or work['position_max'] != len(ids) - 1:
            raise IntegrityError('Fresh oracle did not compute every input')
        raw = logits_path.read_bytes()
        if work['logits_bytes'] != len(raw):
            raise IntegrityError('Fresh oracle output size differs from measured bytes')
        values = _unpack_frontier(raw, 248320)
        self.requests += 1
        self.tokens_computed += work['forward_tokens']
        self.last = dict(tokens=ids, computed_tokens=work['forward_tokens'], position_max=work['position_max'],
                         logits_sha256=hashlib.sha256(raw).hexdigest())
        _trace('CTT_ORACLE_TRACE', dict(command=command, work=work, result=self.last,
                                      seconds=time.monotonic() - start, returncode=result.returncode))
        token_path.unlink()
        logits_path.unlink()
        return values

    def close(self):
        self._scratch.cleanup()
