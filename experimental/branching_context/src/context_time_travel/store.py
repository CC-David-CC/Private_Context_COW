"""Tiny POSIX, stdlib reference for *logical* branching-context persistence.

This module deliberately does NOT implement model evaluation, sampler advancement,
KV storage, numerical checkpoint materialization or zero-copy sharing.
It provides one authoritative immutable snapshot DAG plus atomic movable heads.

Durability model: local POSIX filesystem with atomic same-directory rename,
fsync(file), fsync(directory), and POSIX flock. Not validated on network FS,
Windows, untrusted multiuser paths, or power-failure simulators.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional
from contextlib import contextmanager
from collections.abc import Mapping
from collections import OrderedDict
from types import MappingProxyType
from functools import wraps
import copy
import fcntl
import hashlib
import json
import os
import re
import stat
import tempfile
import threading
import time


class StoreError(Exception):
    """Base store error."""


class Conflict(StoreError):
    """CAS conflict or duplicate branch identity."""


class IntegrityError(StoreError):
    """Object or metadata was missing, corrupted, or invalid."""


class InvalidOperation(StoreError):
    """Invalid API arguments, lineage, or operation-id reuse."""


class NotFound(StoreError):
    """Unknown snapshot or branch."""


@dataclass(frozen=True)
class BranchHead:
    name: str
    revision: int
    tip: str
    root: str


@dataclass(frozen=True)
class PresentationNamespace:
    """Metadata-only presentation identity; it has no model/history head."""

    namespace_id: str
    title: str
    operation_id: str
    created_at_ns: int


_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_-]{0,127}$")
_HEX = re.compile(r"^[0-9a-f]{64}$")
_NAMESPACE_HEX = re.compile(r"^[0-9a-f]{32}$")
_MAX_TOKEN = 2**32 - 1


def _pairs_unique(pairs: list[tuple[str, Any]]) -> dict:
    out: dict[str, Any] = {}
    for k, v in pairs:
        if k in out:
            raise IntegrityError(f"Duplicate JSON key: {k}")
        out[k] = v
    return out


def _json_bytes(obj: Any) -> bytes:
    # Reject NaN/Infinity and non-JSON data. Deterministic canonical byte layout
    # for our strictly constrained schema; this is NOT a general JSON canonicalization RFC.
    try:
        return json.dumps(obj, sort_keys=True, separators=(",", ":"),
                          ensure_ascii=True, allow_nan=False).encode("ascii") + b"\n"
    except (TypeError, ValueError, OverflowError) as exc:
        raise InvalidOperation(f"Unserializable store data: {exc}") from exc


def _decode(raw: bytes) -> dict:
    try:
        obj = json.loads(raw.decode("ascii"), object_pairs_hook=_pairs_unique,
                         parse_constant=lambda c: (_ for _ in ()).throw(IntegrityError(f"Forbidden value: {c}")))
    except (UnicodeError, ValueError, TypeError) as exc:
        raise IntegrityError(f"Invalid metadata JSON: {exc}") from exc
    try:
        canonical = _json_bytes(obj)
    except InvalidOperation as exc:
        raise IntegrityError("Invalid parsed metadata value") from exc
    if not isinstance(obj, dict) or canonical != raw:
        raise IntegrityError("Object is not canonical JSON")
    return obj


def _hash(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _tokens(values: tuple[int, ...] | list[int]) -> list[int]:
    if not isinstance(values, (tuple, list)):
        raise InvalidOperation("Token delta must be a tuple/list of exact integers")
    if any(type(x) is not int or x < 0 or x > _MAX_TOKEN for x in values):
        raise InvalidOperation("Token IDs must be nonnegative int32-range Python ints")
    return list(values)


def _controller(value: Mapping[str, Any]) -> dict:
    if not isinstance(value, Mapping):
        raise InvalidOperation("Controller state must be a JSON-compatible mapping")
    clone = _decode(_json_bytes(dict(value)))
    # Freeze semantics by serializing now; callers' mutable dicts cannot alter the record.
    return clone


def _id(value: str) -> str:
    if not isinstance(value, str) or not _HEX.fullmatch(value):
        raise InvalidOperation("Expected lowercase 64-digit content identifier")
    return value


def _name(value: str) -> str:
    if not isinstance(value, str) or not _NAME.fullmatch(value):
        raise InvalidOperation("Expected 1..128 byte ASCII branch/operation ID")
    return value


def _checkpoint_identity(value: Any, field: str) -> str:
    """Validate a bounded printable identifier carried by checkpoint metadata."""
    if (not isinstance(value, str) or not 1 <= len(value) <= 256
            or any(ord(char) < 0x20 or 0xd800 <= ord(char) <= 0xdfff for char in value)):
        raise InvalidOperation(f"Invalid checkpoint {field}")
    return value


def _namespace_title(value: str) -> str:
    if not isinstance(value, str):
        raise InvalidOperation("Namespace title must be text")
    title = value.strip()
    if not title or len(title) > 120:
        raise InvalidOperation("Namespace title must contain 1..120 characters")
    return title


def _sync_dir(directory: Path) -> None:
    fd = os.open(directory, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _stage(directory: Path, data: bytes) -> Path:
    fd, name = tempfile.mkstemp(prefix=".stage-", dir=directory)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
    except BaseException:
        Path(name).unlink(missing_ok=True)
        raise
    return Path(name)


def _private_directory(path: Path, *, parents=False) -> None:
    path.mkdir(mode=0o700, parents=parents, exist_ok=True)
    if path.is_symlink() or path.stat().st_uid != os.geteuid():
        raise PermissionError('Private directory must be owned by the current OS principal')
    path.chmod(0o700)


def _history_transaction(method):
    @wraps(method)
    def run(self, *args, **kwargs):
        with self._locked():
            return method(self, *args, **kwargs)
    return run


class Store:
    """Authoritative snapshots + transactional named branch heads.

    Exposes a private failpoint only for fault-injection tests. Production code
    must treat a transaction that faults after rename as having *uncertain*
    outcome, then retry the same operation ID and inspect the receipt.
    """

    persistence_policy = MappingProxyType(dict(kind='owner-only-plaintext',encrypted=False,
                                               directories=0o700,files=0o600))

    def __init__(self, directory: str | os.PathLike[str],
                 fault_hook: Optional[Callable[[str], None]] = None):
        self._principal = None  # Trusted local owner; authentication belongs to the caller.
        self._transaction_lock = threading.RLock()
        self._transaction_state = threading.local()
        self._work_lock = threading.Lock()
        self._ancestry_metadata = OrderedDict()
        self._snapshot_parse_cache = dict(entries=OrderedDict(),bytes=0,lock=threading.Lock())
        self._work = dict(snapshot_read_bytes=0, snapshot_reads=0, snapshot_written_bytes=0,
                         reference_read_bytes=0, reference_reads=0, reference_written_bytes=0,
                         reference_publications=0, reference_entries_scanned=0,
                         root_index_read_bytes=0, root_index_written_bytes=0, root_index_repairs=0,
                         ancestry_nodes_visited=0, ancestry_metadata_hits=0, ancestry_metadata_misses=0,
                         snapshot_parse_hits=0, snapshot_parse_misses=0,
                         evaluation_read_bytes=0,
                         evaluation_reads=0, evaluation_written_bytes=0)
        self.root = Path(directory)
        self.objects = self.root / "objects"
        self.root_index = self.root / "root-index"
        self.presentation_namespaces = self.root / "presentation-namespaces"
        self.presentation_bindings = self.root / "presentation-bindings"
        self.evaluation_operations = self.root / 'evaluation-operations'
        self.evaluation_operation_locks = self.root / 'evaluation-operation-locks'
        self.refs_path = self.root / "refs.json"
        self.lock_path = self.root / "refs.lock"
        self.order_path = self.root / "ref-order.json"
        self.fault_hook = fault_hook
        _private_directory(self.root, parents=True)
        _private_directory(self.objects)
        _private_directory(self.root_index)
        self.lock_path.touch(mode=0o600, exist_ok=True)
        self.lock_path.chmod(0o600)
        with self._locked():
            if not self.refs_path.exists():
                self.references = self.root / 'references'
                _private_directory(self.references)
                _sync_dir(self.root)
                self._write_order(0)
                self._publish_format(self.references.name)
            else:
                marker = self._read_reference_json(self.refs_path)
                if type(marker.get('schema')) is not int:
                    raise IntegrityError('Invalid reference schema version')
                if marker.get('schema') == 1:
                    self._migrate_references(marker)
                else:
                    self._set_reference_directory(marker)
            self._reclaim_reference_stages_locked()

    def _reclaim_reference_stages_locked(self) -> None:
        """Retire unpublished reference files only under the transaction lock.

        Reference writers retain this same process/file lock from staging
        through rename and directory fsync. Once startup owns it, no live
        writer can own a regular stage in the selected reference directory.
        Checkpoint writers have a different lifetime and their own inode locks;
        do not generalize this cleanup to objects or checkpoint directories.
        """
        changed = False
        for path in self.references.glob('.stage-*'):
            try:
                entry = path.lstat()
            except FileNotFoundError:
                continue
            if not stat.S_ISREG(entry.st_mode) or entry.st_uid != os.geteuid():
                continue
            # Do not follow a symlink or unlink a replacement inode.
            try:
                current = path.lstat()
                if (current.st_dev, current.st_ino) != (entry.st_dev, entry.st_ino):
                    continue
                path.unlink()
                changed = True
            except FileNotFoundError:
                continue
        if changed:
            _sync_dir(self.references)

    def for_principal(self, principal: str):
        """Bind a trusted caller's authenticated identity to this same store.

        This does not sandbox Python code or replace OS filesystem permissions.
        A restricted handle cannot create another identity or administer grants.
        """
        if self._principal is not None:
            raise PermissionError('Only the local owner can bind principals')
        _name(principal)
        bound = copy.copy(self)
        bound._principal = principal
        bound._work_lock = threading.Lock()
        bound._work = dict.fromkeys(self._work, 0)
        return bound

    def _require_namespace_owner(self) -> None:
        if self._principal is not None:
            raise PermissionError('Presentation namespaces require the local owner')

    @staticmethod
    def _namespace_identity(operation_id: str) -> str:
        try:
            encoded = operation_id.encode('utf-8')
        except UnicodeEncodeError as error:
            raise InvalidOperation('Namespace operation ID must be valid UTF-8 text') from error
        return _hash(b'cow-presentation-namespace-v1\0' + encoded)[:32]

    @staticmethod
    def _namespace_request_digest(operation_id: str, title: str) -> str:
        return _hash(_json_bytes(dict(schema=1, operation_id=operation_id, title=title)))

    def _validate_presentation(self, head: BranchHead, target: str, binding: dict,
                               namespace_id: str, operation_id: str,
                               generation_operation_id: str | None = None) -> dict:
        self._require_namespace_owner()
        self._read_namespace(namespace_id)
        from .presentation_binding import validate_binding
        return validate_binding(self, head, target, binding, namespace_id=namespace_id,
                                operation_id=operation_id,
                                generation_operation_id=generation_operation_id)

    def _presentation_digest(self, binding: dict) -> tuple[dict, bytes, str]:
        if not isinstance(binding, dict):
            raise InvalidOperation('Presentation must be a JSON object')
        value = _decode(_json_bytes(binding))
        raw = _json_bytes(value)
        return value, raw, _hash(raw)

    def _validate_checkpoint_receipt(self, value: Any, target: str, *, stored: bool = False,
                                    validate_model: bool = True):
        """Return a detached descriptor and digest bound to a canonical snapshot."""
        try:
            if not isinstance(value, dict) or set(value) != {
                    'schema', 'snapshot', 'payload_sha256', 'model_contract_id', 'execution_profile_id'}:
                raise InvalidOperation('Checkpoint receipt has an invalid schema')
            if type(value['schema']) is not int or value['schema'] != 1:
                raise InvalidOperation('Unsupported checkpoint receipt schema')
            snapshot = _id(value['snapshot'])
            payload = _id(value['payload_sha256'])
            model = _checkpoint_identity(value['model_contract_id'], 'model contract ID')
            profile = _checkpoint_identity(value['execution_profile_id'], 'execution profile ID')
            if snapshot != _id(target):
                raise InvalidOperation('Checkpoint receipt snapshot differs from move target')
            if not stored and validate_model:
                canonical = self.get_snapshot(snapshot)
                if canonical.get('model_contract_id') != model:
                    raise InvalidOperation('Checkpoint receipt model differs from canonical snapshot')
            descriptor = {'schema': 1, 'snapshot': snapshot, 'payload_sha256': payload,
                          'model_contract_id': model, 'execution_profile_id': profile}
            return descriptor, _hash(_json_bytes(descriptor))
        except (InvalidOperation, IntegrityError, KeyError, TypeError, ValueError) as error:
            if stored:
                if isinstance(error, IntegrityError):
                    raise
                raise IntegrityError('Invalid checkpoint receipt metadata') from error
            if isinstance(error, InvalidOperation):
                raise
            raise InvalidOperation('Invalid checkpoint receipt metadata') from error

    def _write_presentation_blob(self, raw: bytes, digest: str) -> None:
        directory = self.presentation_bindings
        was_present = directory.exists()
        _private_directory(directory)
        if not was_present:
            _sync_dir(self.root)
        path = directory / (digest + '.json')
        try:
            info = path.lstat()
        except FileNotFoundError:
            info = None
        if info is not None:
            existing = self._read_presentation_blob(digest)
            if _json_bytes(existing) != raw:
                raise IntegrityError('Conflicting content-addressed presentation blob')
            _sync_dir(directory)
            return
        stage = _stage(directory, raw)
        try:
            self._fault('before_presentation_blob_publish')
            try:
                os.link(stage, path)
            except FileExistsError:
                if self._read_presentation_blob(digest) != _decode(raw):
                    raise IntegrityError('Conflicting content-addressed presentation blob')
            else:
                self._fault('before_presentation_blob_dir_fsync')
                _sync_dir(directory)
            self._fault('after_presentation_blob_publish')
        finally:
            stage.unlink(missing_ok=True)

    def _read_presentation_blob(self, digest: str) -> dict:
        digest = _id(digest)
        path = self.presentation_bindings / (digest + '.json')
        try:
            dir_info = self.presentation_bindings.lstat()
            info = path.lstat()
            if (stat.S_ISLNK(dir_info.st_mode) or not stat.S_ISDIR(dir_info.st_mode)
                    or dir_info.st_uid != os.geteuid() or stat.S_IMODE(dir_info.st_mode) != 0o700
                    or stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode)
                    or info.st_uid != os.geteuid() or stat.S_IMODE(info.st_mode) != 0o600):
                raise PermissionError('Presentation metadata must be owner-only')
            raw = path.read_bytes()
        except FileNotFoundError as error:
            raise IntegrityError('Missing referenced presentation blob') from error
        if _hash(raw) != digest:
            raise IntegrityError('Presentation blob digest mismatch')
        value = _decode(raw)
        if not isinstance(value, dict):
            raise IntegrityError('Presentation blob must contain an object')
        return value

    def _validate_stored_presentation(self, name: str, branch: dict, operation_id: str,
                                      receipt: dict, *, blob_cache: dict | None = None,
                                      refs: dict | None = None) -> dict:
        digest = receipt.get('presentation_digest')
        if not isinstance(digest, str):
            raise NotFound('Operation has no presentation receipt')
        if blob_cache is not None and digest in blob_cache:
            binding = copy.deepcopy(blob_cache[digest])
        else:
            binding = self._read_presentation_blob(digest)
            if blob_cache is not None:
                blob_cache[digest] = copy.deepcopy(binding)
        namespace_id = branch.get('namespace_id')
        if namespace_id is None or binding.get('namespace_id') != namespace_id:
            raise IntegrityError('Presentation receipt namespace differs from branch')
        result = receipt.get('result')
        if not isinstance(result, dict):
            raise IntegrityError('Presentation receipt lacks a result head')
        if binding.get('kind') == 'root':
            source_record = binding.get('source')
            if not isinstance(source_record, dict) or source_record.get('revision') != 0:
                raise IntegrityError('Root presentation has an invalid source revision')
            source = BranchHead(name, 0, result['tip'], result['root'])
        else:
            source_binding = binding.get('source')
            if not isinstance(source_binding, dict):
                raise IntegrityError('Presentation receipt lacks a source')
            source_id = source_binding.get('snapshot')
            if not isinstance(source_id, str):
                raise IntegrityError('Presentation source snapshot is missing')
            source = BranchHead(name, source_binding.get('revision'), source_id, result['root'])
        source_operation = receipt.get('presentation_source_operation', operation_id)
        if source_operation == operation_id:
            # Ordinary presentation receipts already own their stream ID.
            # Avoid rereading the complete canonical refs catalog on every
            # normal annotation validation.
            generation_operation = operation_id
        else:
            try:
                generation_operation,original = self._presentation_stream_operation(
                    source_operation, return_source=True, refs=refs)
            except (NotFound, InvalidOperation, IntegrityError) as error:
                raise IntegrityError('Retry presentation source receipt is unavailable') from error
            original_binding = original.get('presentation')
            if (not isinstance(original_binding, dict)
                    or original_binding.get('namespace_id') != namespace_id
                    or original.get('result', {}).get('root') != result.get('root')
                    or original_binding.get('source') != binding.get('source')):
                raise IntegrityError('Retry presentation differs from its committed source operation')
        try:
            return self._validate_presentation(source, result['tip'], binding,
                                               namespace_id, operation_id,
                                               generation_operation_id=generation_operation)
        except InvalidOperation as error:
            raise IntegrityError('Stored presentation annotation no longer validates') from error

    def _presentation_stream_operation(self, operation_id: str, *, return_source=False, refs=None):
        """Resolve a retry chain to the original generation's RNG identity."""
        if refs is None:
            with self._locked():
                refs=self._read_refs()
        seen=set();current=operation_id;expected=None
        first_raw=None
        while True:
            if current in seen:
                raise IntegrityError('Presentation retry source cycle')
            seen.add(current)
            raw=self._raw_presentation_receipt(current,refs=refs)
            if first_raw is None:
                first_raw=raw
            binding=raw['presentation'];result=raw['result']
            identity=(binding.get('namespace_id'),binding.get('source'),result.get('root'))
            if expected is not None and identity!=expected:
                raise IntegrityError('Presentation retry chain changed source identity')
            expected=identity
            parent=raw.get('presentation_source_operation')
            if parent is None:
                return (current,first_raw) if return_source else current
            if not isinstance(parent,str):
                raise IntegrityError('Presentation retry source operation is malformed')
            current=parent

    def _raw_presentation_receipt(self, operation_id: str, *, refs=None) -> dict:
        """Read a receipt and digest-checked presentation without recursive validation."""
        operation=_name(operation_id)
        with self._locked():
            if refs is None:
                refs=self._read_refs()
            receipt=refs['receipts'].get(operation)
            if receipt is None:
                raise NotFound(f'Unknown presentation operation: {operation}')
            digest=receipt.get('presentation_digest')
            if not isinstance(digest,str):
                raise NotFound('Operation has no presentation receipt')
            binding=self._read_presentation_blob(digest)
            result=receipt.get('result')
            if not isinstance(result,dict):
                raise IntegrityError('Presentation receipt lacks a result head')
            return dict(presentation=binding,result=copy.deepcopy(result),
                        presentation_source_operation=receipt.get('presentation_source_operation'))

    def _check_resume_message_id(self, binding: dict, namespace_id: str,
                                 source: BranchHead) -> None:
        if binding.get('kind') != 'resume':
            return
        stream_key = self.get_snapshot(source.tip)['controller'].get('stream_key')
        if not isinstance(stream_key, str):
            raise InvalidOperation('Resume source has no active stream identity')
        expected = self._published_assistant_message_id(namespace_id, source.tip, stream_key,
                                                        branch_name=source.name)
        output = binding.get('output_span')
        if expected is None or not isinstance(output, dict) or output.get('message_id') != expected:
            raise InvalidOperation('Resume must retain its published assistant message ID')

    def _published_assistant_message_id(self, namespace_id: str, snapshot_id: str,
                                        stream_key: str, *, branch_name: str | None = None,
                                        _visited=None) -> str | None:
        requested_branch=branch_name
        refs = self._read_refs()
        found = set()
        if self.get_snapshot(snapshot_id)['controller'].get('stream_key') != stream_key:
            raise InvalidOperation('Resume stream does not match its source snapshot')
        if branch_name is None:
            candidates=refs['receipts'].items()
        else:
            visited=set() if _visited is None else set(_visited)
            if branch_name in visited:
                raise IntegrityError('Fork presentation provenance contains a cycle')
            visited.add(branch_name)
            branch_record=self._read_branch(branch_name)
            if branch_record.get('namespace_id')!=namespace_id:
                raise InvalidOperation('Active presentation branch belongs to another namespace')
            # An evaluation adoption can intentionally choose one of several
            # operations that produced the same canonical snapshot. Its move
            # event is the branch-specific authority for that text identity.
            for event in reversed(branch_record.get('events', [])):
                if (event.get('type') == 'move' and event.get('to') == snapshot_id):
                    source_operation = event.get('adoption_presentation_source_operation')
                    if source_operation is not None:
                        return self._published_message_from_operation(
                            source_operation, namespace_id, snapshot_id, stream_key, refs)
                    break
            candidates=((operation,refs['receipts'].get(operation))
                        for operation in branch_record.get('receipts',{}))
        for operation, receipt in candidates:
            if not isinstance(receipt,dict):
                continue
            result = receipt.get('result')
            if not isinstance(result, dict) or result.get('tip') != snapshot_id:
                continue
            receipt_branch_name = result.get('name')
            if not isinstance(receipt_branch_name, str):
                raise IntegrityError('Presentation receipt has an invalid branch result')
            branch = self._read_branch(receipt_branch_name)
            if branch.get('namespace_id') != namespace_id:
                continue
            if not isinstance(receipt.get('presentation_digest'), str):
                continue
            descriptor = self._validate_stored_presentation(receipt_branch_name, branch, operation, receipt,
                                                            refs=refs)
            if descriptor.get('kind') not in ('answer', 'initial_turn', 'user_turn', 'resume', 'reseed'):
                continue
            output = descriptor.get('output_span')
            if not isinstance(output, dict):
                continue
            message_id = output.get('message_id')
            if isinstance(message_id, str):
                found.add(message_id)
        if len(found) > 1:
            if requested_branch is None:
                raise IntegrityError('One active stream has conflicting published assistant IDs')
            raise IntegrityError('One branch has conflicting published assistant IDs')
        if not found and requested_branch is not None:
            events=branch_record.get('events',[])
            if events and isinstance(events[0],dict) and events[0].get('type')=='fork':
                source=events[0].get('source')
                if (isinstance(source,dict) and isinstance(source.get('tip'),str)
                        and isinstance(source.get('branch'),str)
                        and self._snapshot_is_ancestor(snapshot_id, source['tip'])):
                    return self._published_assistant_message_id(namespace_id,snapshot_id,stream_key,
                        branch_name=source['branch'],_visited=visited)
                if source is None:
                    # Older namespace forks did not persist a source branch.
                    # A global lookup is safe only when exactly one verified
                    # operation owns this snapshot/stream in the namespace.
                    global_ids = self._global_published_message_ids(
                        namespace_id, snapshot_id, stream_key, refs)
                    if len(global_ids) > 1:
                        raise IntegrityError('Unprovenanced fork has ambiguous published assistant IDs')
                    return next(iter(global_ids), None)
        return next(iter(found), None)

    def _snapshot_is_ancestor(self, ancestor: str, descendant: str) -> bool:
        """Check canonical ancestry without materializing the token history."""
        cursor = _id(descendant)
        ancestor = _id(ancestor)
        seen = set()
        while cursor is not None:
            if cursor in seen:
                raise IntegrityError('Cycle in snapshot ancestry')
            seen.add(cursor)
            if cursor == ancestor:
                return True
            _, parent, _, _ = self._ancestry_record(cursor)
            cursor = parent
        return False

    def _global_published_message_ids(self, namespace_id, snapshot_id, stream_key, refs):
        found = set()
        for operation, receipt in refs['receipts'].items():
            if not isinstance(receipt, dict):
                continue
            result = receipt.get('result')
            if not isinstance(result, dict) or result.get('tip') != snapshot_id:
                continue
            branch_name = result.get('name')
            if not isinstance(branch_name, str):
                continue
            branch = self._read_branch(branch_name)
            if branch.get('namespace_id') != namespace_id:
                continue
            descriptor = self._validated_published_descriptor(
                operation, receipt, branch_name, branch, namespace_id, snapshot_id, refs)
            if descriptor is not None:
                found.add(descriptor['output_span']['message_id'])
        return found

    def _published_message_from_operation(self, operation, namespace_id,
                                          snapshot_id, stream_key, refs):
        receipt = refs['receipts'].get(operation)
        if not isinstance(receipt, dict):
            raise IntegrityError('Adoption source operation receipt is missing')
        result = receipt.get('result')
        branch_name = result.get('name') if isinstance(result, dict) else None
        if not isinstance(branch_name, str) or result.get('tip') != snapshot_id:
            raise IntegrityError('Adoption source does not publish the selected snapshot')
        branch = self._read_branch(branch_name)
        if branch.get('namespace_id') != namespace_id:
            raise IntegrityError('Adoption source belongs to another namespace')
        descriptor = self._validated_published_descriptor(
            operation, receipt, branch_name, branch, namespace_id, snapshot_id, refs)
        if (descriptor is None
                or self.get_snapshot(snapshot_id)['controller'].get('stream_key') != stream_key):
            raise IntegrityError('Adoption source has no matching published stream')
        return descriptor['output_span']['message_id']

    def _validated_published_descriptor(self, operation, receipt, branch_name,
                                        branch, namespace_id, snapshot_id, refs):
        if not isinstance(receipt.get('presentation_digest'), str):
            return None
        descriptor = self._validate_stored_presentation(
            branch_name, branch, operation, receipt, refs=refs)
        if (descriptor.get('namespace_id') != namespace_id
                or descriptor.get('kind') not in ('answer', 'initial_turn', 'user_turn', 'resume', 'reseed')):
            return None
        output = descriptor.get('output_span')
        if not isinstance(output, dict) or not isinstance(output.get('message_id'), str):
            return None
        return descriptor

    def _resolve_adoption_presentation_source(self, branch_name, snapshot_id,
                                              requested_operation, refs=None):
        """Resolve only a verified receipt that uniquely owns the selected text identity."""
        branch = self._read_branch(branch_name)
        namespace_id = branch.get('namespace_id')
        if namespace_id is None:
            if requested_operation is not None:
                raise InvalidOperation('Presentation source requires a namespaced adoption branch')
            return None
        controller = self.get_snapshot(snapshot_id)['controller']
        status = controller.get('status') if isinstance(controller, dict) else None
        if status not in ('active', 'stopped'):
            if requested_operation is not None:
                raise InvalidOperation('Selected snapshot has no published presentation scope')
            return None
        root = branch['head']['root']
        if self._root_for(snapshot_id) != root:
            raise InvalidOperation('Presentation source snapshot belongs to another branch root')
        if refs is None:
            refs = self._read_refs()
        candidates = []
        for operation, receipt in refs['receipts'].items():
            result = receipt.get('result') if isinstance(receipt, dict) else None
            if not isinstance(result, dict) or result.get('tip') != snapshot_id:
                continue
            source_branch = result.get('name')
            if not isinstance(source_branch, str):
                continue
            source_record = self._read_branch(source_branch)
            if (source_record.get('namespace_id') != namespace_id
                    or source_record['head']['root'] != root):
                continue
            descriptor = self._validated_published_descriptor(
                operation, receipt, source_branch, source_record,
                namespace_id, snapshot_id, refs)
            if descriptor is not None:
                candidates.append((operation, descriptor['output_span']['message_id']))
        if requested_operation is not None:
            if not any(operation == requested_operation for operation, _ in candidates):
                raise InvalidOperation('Presentation source is not a verified publication for the selected snapshot')
            return requested_operation
        message_ids = {message_id for _, message_id in candidates}
        if len(message_ids) > 1:
            raise InvalidOperation('Ambiguous presentation source; specify the published source operation')
        if candidates:
            return sorted(operation for operation, _ in candidates)[0]
        raise InvalidOperation('Selected active snapshot has no verified presentation source')

    def _check_namespace_directory(self) -> None:
        path = self.presentation_namespaces
        info = path.lstat()
        if (stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode)
                or info.st_uid != os.geteuid() or stat.S_IMODE(info.st_mode) != 0o700):
            raise PermissionError('Presentation namespace directory must be owner-only')

    def _read_namespace(self, namespace_id: str) -> PresentationNamespace:
        if not isinstance(namespace_id, str) or not _NAMESPACE_HEX.fullmatch(namespace_id):
            raise InvalidOperation('Expected lowercase 32-digit namespace identifier')
        path = self.presentation_namespaces / (namespace_id + '.json')
        try:
            self._check_namespace_directory()
            info = path.lstat()
            if (stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode)
                    or info.st_uid != os.geteuid() or stat.S_IMODE(info.st_mode) != 0o600):
                raise PermissionError('Presentation namespace record must be owner-only')
            raw = path.read_bytes()
        except FileNotFoundError as error:
            raise NotFound('Unknown presentation namespace') from error
        self._count(reference_reads=1, reference_read_bytes=len(raw))
        envelope = _decode(raw)
        value = envelope.get('value')
        if (set(envelope) != {'value', 'checksum'} or not isinstance(value, dict)
                or envelope['checksum'] != _hash(_json_bytes(value))):
            raise IntegrityError('Corrupt presentation namespace checksum')
        expected = {'schema', 'namespace_id', 'title', 'operation_id',
                    'request_digest', 'created_at_ns'}
        if (set(value) != expected or type(value['schema']) is not int or value['schema'] != 1
                or type(value['created_at_ns']) is not int or value['created_at_ns'] < 0
                or not isinstance(value['operation_id'], str)
                or not 1 <= len(value['operation_id']) <= 128):
            raise IntegrityError('Invalid presentation namespace record')
        try:
            if (not isinstance(value['namespace_id'], str)
                    or not _NAMESPACE_HEX.fullmatch(value['namespace_id'])):
                raise InvalidOperation('Invalid namespace identifier')
            title = _namespace_title(value['title'])
            expected_id = self._namespace_identity(value['operation_id'])
            expected_digest = self._namespace_request_digest(value['operation_id'], title)
            _id(value['request_digest'])
        except InvalidOperation as error:
            raise IntegrityError('Invalid presentation namespace identity or title') from error
        if (value['namespace_id'] != namespace_id or namespace_id != expected_id
                or value['title'] != title or value['request_digest'] != expected_digest):
            raise IntegrityError('Inconsistent presentation namespace record')
        return PresentationNamespace(namespace_id, title, value['operation_id'],
                                    value['created_at_ns'])

    @_history_transaction
    def create_namespace(self, title: str, operation_id: str) -> PresentationNamespace:
        """Create an empty presentation identity without creating COW state.

        The operation ID deterministically identifies its namespace, making a
        retry after an uncertain rename recover the original durable receipt.
        """
        self._require_namespace_owner()
        title = _namespace_title(title)
        if not isinstance(operation_id, str) or not 1 <= len(operation_id) <= 128:
            raise InvalidOperation('Namespace operation ID must contain 1..128 characters')
        namespace_id = self._namespace_identity(operation_id)
        digest = self._namespace_request_digest(operation_id, title)
        directory = self.presentation_namespaces
        was_present = directory.exists()
        _private_directory(directory)
        if not was_present:
            _sync_dir(self.root)
        path = directory / (namespace_id + '.json')
        try:
            path.lstat()
            path_present = True
        except FileNotFoundError:
            path_present = False
        if path_present:
            existing = self._read_namespace(namespace_id)
            if existing.operation_id != operation_id or self._namespace_request_digest(
                    existing.operation_id, existing.title) != digest:
                raise InvalidOperation('Namespace operation ID reused for different input')
            # Complete directory durability if a prior attempt faulted after
            # rename but before fsyncing the namespace directory.
            _sync_dir(directory)
            return existing

        with self._locked():
            self._assert_operation_id_available(operation_id, allow_evaluation_kind='namespace')

        value = dict(schema=1, namespace_id=namespace_id, title=title,
                     operation_id=operation_id, request_digest=digest,
                     created_at_ns=time.time_ns())
        raw_value = _json_bytes(value)
        raw = _json_bytes(dict(value=value, checksum=_hash(raw_value)))
        stage = _stage(directory, raw)
        try:
            self._fault('namespace_before_rename')
            os.replace(stage, path)
            self._fault('namespace_after_rename_before_dir_fsync')
            _sync_dir(directory)
        finally:
            stage.unlink(missing_ok=True)
        return PresentationNamespace(namespace_id, title, operation_id,
                                     value['created_at_ns'])

    @_history_transaction
    def get_namespace(self, namespace_id: str) -> PresentationNamespace:
        """Read and validate one owner-only metadata namespace record."""
        self._require_namespace_owner()
        return self._read_namespace(namespace_id)

    @_history_transaction
    def namespaces(self) -> tuple[PresentationNamespace, ...]:
        """List a consistent, read-only snapshot of presentation namespaces."""
        self._require_namespace_owner()
        if not self.presentation_namespaces.exists():
            return ()
        self._check_namespace_directory()
        records = []
        for path in sorted(self.presentation_namespaces.glob('*.json')):
            try:
                records.append(self._read_namespace(path.stem))
            except InvalidOperation as error:
                raise IntegrityError('Invalid presentation namespace filename') from error
        return tuple(records)

    def grant(self, root: str, principal: str, rights) -> None:
        if self._principal is not None:
            raise PermissionError('Only the local owner can change grants')
        _id(root)
        _name(principal)
        rights = set(rights)
        allowed = {'read','fork','move','append','evaluate','materialize','checkpoint','delete'}
        if rights - allowed:
            raise InvalidOperation('Unknown root permission')
        if self.get_snapshot(root)['parent'] is not None:
            raise InvalidOperation('Grants must identify a history root')
        with self._locked():
            directory = self.root/'grants'
            _private_directory(directory)
            path = directory/(root+'.json')
            value = _decode(path.read_bytes()) if path.exists() else {}
            value[principal] = sorted(rights)
            stage = _stage(directory,_json_bytes(value))
            try:
                os.replace(stage,path)
                _sync_dir(directory)
                _sync_dir(self.root)
            finally:
                stage.unlink(missing_ok=True)

    def authorize(self, snapshot: str, operation: str = 'read') -> None:
        if self._principal is None:
            return
        try:
            root = self._read_root_index(_id(snapshot))
            grants = _decode((self.root/'grants'/(root+'.json')).read_bytes())
        except FileNotFoundError as error:
            raise PermissionError('No root grant for principal') from error
        if operation not in grants.get(self._principal, []):
            raise PermissionError('Root operation is not granted')

    def _index_evaluation_access(self, identity, snapshot):
        directory = self.root/'evaluation-access'
        _private_directory(directory)
        path = directory/(identity+'.json')
        raw = _json_bytes({'snapshot':snapshot})
        stage = _stage(directory,raw)
        try:
            try:
                os.link(stage,path)
            except FileExistsError:
                if path.read_bytes() != raw:
                    raise IntegrityError('Conflicting evaluation access index')
            _sync_dir(directory)
            _sync_dir(self.root)
        finally:
            stage.unlink(missing_ok=True)

    def _count(self, **values: int) -> None:
        with self._work_lock:
            for key, value in values.items():
                self._work[key] += value

    def retain(self, name: str, snapshot: str) -> None:
        """Name an explicit historical pin without copying history."""
        if self._principal is not None:
            raise PermissionError('Retention administration requires the local owner')
        _name(name)
        with self._locked():
            self.get_snapshot(snapshot)
            directory=self.root/'retained'
            _private_directory(directory)
            stage=_stage(directory,_json_bytes({'snapshot':snapshot}))
            try:
                os.replace(stage,directory/(name+'.json'))
                _sync_dir(directory)
                _sync_dir(self.root)
            finally:
                stage.unlink(missing_ok=True)

    def release_retained(self, name: str) -> None:
        if self._principal is not None:
            raise PermissionError('Retention administration requires the local owner')
        with self._locked():
            path=self.root/'retained'/(_name(name)+'.json')
            path.unlink()
            _sync_dir(path.parent)

    @contextmanager
    def read_lease(self, snapshot: str):
        """An OS lock pins an active reader; process death releases it."""
        self.authorize(snapshot,'read')
        stream=None
        try:
            with self._locked():
                self.get_snapshot(snapshot)
                directory=self.root/'read-leases'
                _private_directory(directory)
                descriptor=os.open(directory/(_id(snapshot)+'.lock'),os.O_CREAT|os.O_RDWR,0o600)
                stream=os.fdopen(descriptor,'r+b')
                fcntl.flock(stream.fileno(),fcntl.LOCK_SH)
            yield snapshot
        finally:
            if stream is not None:
                stream.close()

    def _retained_snapshots_locked(self):
        refs=self._read_refs()
        seeds={s for h in refs['branches'].values() for s in (h['root'],h['tip'])}
        for event in refs['events']:
            source = event.get('source')
            if isinstance(source, dict) and isinstance(source.get('tip'), str):
                seeds.add(_id(source['tip']))
        for operation, receipt in refs['receipts'].items():
            digest = receipt.get('presentation_digest')
            if digest is None:
                continue
            binding = self._read_presentation_blob(digest)
            source = binding.get('source')
            if isinstance(source, dict) and isinstance(source.get('snapshot'), str):
                seeds.add(_id(source['snapshot']))
        for path in (self.root/'retained').glob('*.json'):
            seeds.add(_id(_decode(path.read_bytes())['snapshot']))
        for path in (self.root/'evaluations').glob('*.json'):
            record=self.get_evaluation(path.stem)
            seeds.update((record['snapshot_id'],record['task_root']))
        for path in (self.root/'read-leases').glob('*.lock'):
            with path.open('r+b') as stream:
                try:
                    fcntl.flock(stream.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
                except BlockingIOError:
                    seeds.add(_id(path.stem))
        retained=set()
        for seed in seeds:
            cursor=seed
            while cursor is not None and cursor not in retained:
                retained.add(cursor)
                cursor=self.get_snapshot(cursor)['parent']
        return retained

    def retained_snapshots(self):
        if self._principal is not None:
            raise PermissionError('Global retention inspection requires the local owner')
        with self._locked():
            return self._retained_snapshots_locked()

    def checkpoint_roots(self):
        """Live numerical read pins, independent of canonical retention roots.

        Branches, named history and evaluations retain canonical records, not
        permanent checkpoint copies. A reader/publication lease protects the
        numerical copy while it is in use; process death releases that pin.
        """
        if self._principal is not None:
            raise PermissionError('Checkpoint retention inspection requires local owner')
        with self._locked():
            seeds = set()
            for path in (self.root/'read-leases').glob('*.lock'):
                with path.open('r+b') as stream:
                    try: fcntl.flock(stream.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
                    except BlockingIOError: seeds.add(_id(path.stem))
            return seeds

    def release_history(self, snapshot):
        """Explicitly make an identity eligible for authorized collection."""
        self.authorize(snapshot,'delete')
        with self._locked():
            self.get_snapshot(snapshot)
            directory=self.root/'released-history'
            _private_directory(directory)
            stage=_stage(directory,_json_bytes({'snapshot':snapshot}))
            try:
                os.replace(stage,directory/(_id(snapshot)+'.json'))
                _sync_dir(directory)
                _sync_dir(self.root)
            finally:
                stage.unlink(missing_ok=True)

    def collect_history(self, root):
        """Sweep explicitly released, unreachable objects under one history root.

        Collection is an owner administration operation. Ordinary cache eviction
        never calls this. Delete descendants first so interruption cannot leave
        a surviving object with a missing parent.
        """
        if self._principal is not None:
            raise PermissionError('History collection requires the local owner')
        with self._locked():
            if self.get_snapshot(root)['parent'] is not None:
                raise InvalidOperation('Collection scope must be a history root')
            records={p.stem:self.get_snapshot(p.stem) for p in self.objects.glob('*.json')}
            scope={sid for sid in records if self._validated_root(sid) == root}
            released={p.stem for p in (self.root/'released-history').glob('*.json')
                      if _decode(p.read_bytes()) == {'snapshot':p.stem}}
            marked=self._retained_snapshots_locked() | (set(records)-released) | {root}
            # Retry receipts are durable references, even after a branch rewind.
            for receipt in self._read_refs()['receipts'].values():
                marked.add(receipt['result']['tip'])
            pending=list(marked)
            while pending:
                sid=pending.pop()
                parent=records[sid]['parent']
                if parent is not None and parent not in marked:
                    marked.add(parent)
                    pending.append(parent)
            def depth(sid):
                count=0
                while sid is not None:
                    count+=1
                    sid=records[sid]['parent']
                return count
            removable=scope-marked
            deleted=[]
            for sid in sorted(removable,key=lambda sid:(-depth(sid),sid)):
                self._fault('before_history_unlink')
                (self.objects/(sid+'.json')).unlink()
                _sync_dir(self.objects)
                self._fault('after_history_unlink')
                (self.root_index/(sid+'.json')).unlink(missing_ok=True)
                (self.root/'released-history'/(sid+'.json')).unlink(missing_ok=True)
                deleted.append(sid)
            _sync_dir(self.root_index)
            if (self.root/'released-history').exists():
                _sync_dir(self.root/'released-history')
            return {'deleted':sorted(deleted),'retained':sorted(scope & marked)}

    def work_counters(self) -> dict[str, int]:
        """Completed application-byte I/O and actual traversals, not disk traffic.

        Reads include cached filesystem reads; writes count staged/fsynced bytes.
        reference_publications counts branch-record renames, before directory
        fsync; reference byte counters also include the format and audit ordinal.
        These counters neither infer device traffic nor establish power-loss durability.
        """
        with self._work_lock:
            return dict(self._work)

    @staticmethod
    def _empty_refs() -> dict:
        return {"schema": 1, "branches": {}, "receipts": {}, "events": []}

    def _fault(self, step: str) -> None:
        if self.fault_hook is not None:
            self.fault_hook(step)

    @contextmanager
    def _locked(self):
        with self._transaction_lock:
            if getattr(self._transaction_state,'active',False):
                yield
                return
            with open(self.lock_path, "r+b") as fh:
                fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
                self._transaction_state.active=True
                try:
                    yield
                finally:
                    self._transaction_state.active=False
                    fcntl.flock(fh.fileno(), fcntl.LOCK_UN)

    def _operation_path(self, operation_id):
        return self.evaluation_operations / (_name(operation_id) + '.json')

    def _operation_lock(self, operation_id):
        _private_directory(self.evaluation_operation_locks)
        path = self.evaluation_operation_locks / (_name(operation_id) + '.lock')
        descriptor = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
        os.fchmod(descriptor, 0o600)
        return descriptor

    @contextmanager
    def _evaluation_operation_locked(self, operation_id):
        descriptor = self._operation_lock(operation_id)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            yield
        finally:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)

    def _assert_operation_id_available(self, operation_id, *, allow_evaluation_kind=None):
        """Reject conflicting IDs across operation records visible to this claim.

        Call under ``_locked`` so these checks serialize with evaluation,
        namespace, and coordinator-pool claims. Branch and pool publication
        paths intentionally allow their own operation class while committing;
        callers must use the owning coordinator path for those operations.
        An adoption may reuse its own ID for the single Store.move it authorizes.
        """
        try:
            operation = _name(operation_id)
        except InvalidOperation:
            # Namespace operation IDs deliberately retain their wider
            # 1..128-character Unicode contract. IDs outside the branch /
            # evaluation grammar cannot collide with those operation stores.
            if allow_evaluation_kind == 'namespace':
                return
            raise
        if allow_evaluation_kind not in ('branch', 'adoption', 'pool'):
            refs = self._read_refs()
        else:
            refs = None
        if refs is not None and operation in refs.get('receipts', {}):
            if allow_evaluation_kind == 'adoption':
                current = self._read_evaluation_operation(operation)
                if current is not None and current.get('kind') == 'adoption':
                    return
            raise InvalidOperation('Operation ID is already used by a branch operation')
        if (allow_evaluation_kind not in ('pool', 'branch')
                and (self.root / 'pool-operations' / (operation + '.json')).exists()):
            raise InvalidOperation('Operation ID is already used by a coordinator operation')
        if allow_evaluation_kind != 'namespace' and self.presentation_namespaces.exists():
            namespace_id = self._namespace_identity(operation)
            path = self.presentation_namespaces / (namespace_id + '.json')
            if path.exists() and self._read_namespace(namespace_id).operation_id == operation:
                raise InvalidOperation('Operation ID is already used by a namespace operation')
        existing = self._read_evaluation_operation(operation)
        if existing is not None:
            if allow_evaluation_kind is not None and existing.get('kind') == allow_evaluation_kind:
                return
            raise InvalidOperation('Operation ID is already used by an evaluation operation')

    def _read_evaluation_operation(self, operation_id):
        path = self._operation_path(operation_id)
        try:
            envelope = _decode(path.read_bytes())
        except FileNotFoundError:
            return None
        value = envelope.get('operation')
        if (not isinstance(value, dict) or set(envelope) != {'operation', 'checksum'}
                or envelope.get('checksum') != _hash(_json_bytes(value))
                or value.get('schema') != 1
                or value.get('kind') not in ('evaluate', 'adoption')
                or value.get('operation_id') != operation_id
                or value.get('state') not in ('pending', 'committed')
                or not isinstance(value.get('request_digest'), str)
                or not _HEX.fullmatch(value['request_digest'])):
            raise IntegrityError('Evaluation operation record is corrupt')
        common = {'schema','kind','operation_id','request_digest','state'}
        if value['kind'] == 'evaluate':
            expected = common | {'snapshot_id'}
            if value['state'] == 'committed': expected |= {'result','committed_at_ns'}
        else:
            expected = common | {'branch','expected_revision','evaluation_ids','mandatory'}
            if 'requested_presentation_source_operation' in value:
                expected.add('requested_presentation_source_operation')
            if 'adoption_presentation_source_operation' in value:
                expected.add('adoption_presentation_source_operation')
            if 'selected_evaluation_id' in value or 'selected_snapshot_id' in value:
                expected |= {'selected_evaluation_id','selected_snapshot_id'}
            if value['state'] == 'committed': expected |= {'result','committed_at_ns'}
            if value['state'] == 'pending' and 'selected_evaluation_id' not in value:
                if 'selected_snapshot_id' in value:
                    raise IntegrityError('Evaluation operation record is corrupt')
        if set(value) != expected:
            raise IntegrityError('Evaluation operation record is corrupt')
        if value['kind'] == 'evaluate':
            try: _id(value['snapshot_id'])
            except InvalidOperation as error: raise IntegrityError('Evaluation operation record is corrupt') from error
        else:
            if (not isinstance(value['branch'],str) or type(value['expected_revision']) is not int
                    or value['expected_revision'] < 0 or not isinstance(value['evaluation_ids'],list)
                    or not isinstance(value['mandatory'],list)):
                raise IntegrityError('Evaluation operation record is corrupt')
            if 'selected_evaluation_id' in value:
                try:
                    _id(value['selected_evaluation_id']);_id(value['selected_snapshot_id'])
                except InvalidOperation as error:
                    raise IntegrityError('Evaluation operation record is corrupt') from error
            for key in ('requested_presentation_source_operation',
                        'adoption_presentation_source_operation'):
                if key in value:
                    try: _name(value[key])
                    except InvalidOperation as error:
                        raise IntegrityError('Evaluation operation record is corrupt') from error
            if ('requested_presentation_source_operation' in value
                    and 'adoption_presentation_source_operation' in value
                    and value['requested_presentation_source_operation']
                        != value['adoption_presentation_source_operation']):
                raise IntegrityError('Evaluation operation record is corrupt')
        if value['state'] == 'committed':
            result = value.get('result')
            if (not isinstance(result,dict) or type(value.get('committed_at_ns')) is not int
                    or value['committed_at_ns'] <= 0):
                raise IntegrityError('Evaluation operation receipt is corrupt')
            if value['kind'] == 'evaluate':
                try: _id(result.get('evaluation_id'))
                except InvalidOperation as error:
                    raise IntegrityError('Evaluation operation receipt is corrupt') from error
            elif (result.get('evaluation_id') != value.get('selected_evaluation_id')
                  or result.get('snapshot_id') != value.get('selected_snapshot_id')
                  or result.get('adoption_presentation_source_operation')
                     != value.get('adoption_presentation_source_operation')):
                raise IntegrityError('Adoption operation receipt is corrupt')
        return value

    def _write_evaluation_operation(self, value):
        _private_directory(self.evaluation_operations)
        _sync_dir(self.root)
        raw = _json_bytes(dict(operation=value, checksum=_hash(_json_bytes(value))))
        path = self._operation_path(value['operation_id'])
        stage = _stage(self.evaluation_operations, raw)
        try:
            os.replace(stage, path)
            _sync_dir(self.evaluation_operations)
        finally:
            stage.unlink(missing_ok=True)

    def _claim_evaluation_operation(self, operation_id, kind, request_digest, details):
        if not _HEX.fullmatch(request_digest):
            raise InvalidOperation('Operation digest must be a lowercase SHA-256 value')
        with self._locked():
            existing = self._read_evaluation_operation(operation_id)
            if existing is not None:
                if existing.get('kind') != kind:
                    raise InvalidOperation('Operation ID is already used by another endpoint')
                if existing.get('request_digest') != request_digest:
                    raise InvalidOperation('Operation ID reused for different input')
                return existing
            self._assert_operation_id_available(operation_id)
            value = dict(schema=1, kind=kind, operation_id=_name(operation_id),
                         request_digest=request_digest, state='pending', **details)
            self._write_evaluation_operation(value)
            return value

    def _commit_evaluation_operation(self, value, result):
        committed = dict(value, state='committed', result=result,
                         committed_at_ns=time.time_ns())
        self._write_evaluation_operation(committed)
        return committed

    def evaluation_operation_status(self, operation_id):
        operation = _name(operation_id)
        with self._locked():
            value = self._read_evaluation_operation(operation)
        if value is None:
            raise NotFound('Unknown evaluation operation: ' + operation)
        result = {'operation_id':operation, 'kind':value['kind'], 'state':value['state']}
        if value['state'] == 'committed':
            result['result'] = copy.deepcopy(value['result'])
        return result

    def _read_reference_json(self, path: Path) -> dict:
        try:
            raw = path.read_bytes()
        except FileNotFoundError as exc:
            raise IntegrityError("Missing reference metadata") from exc
        self._count(reference_reads=1, reference_read_bytes=len(raw))
        return _decode(raw)

    def _publish_reference_json(self, path: Path, value: dict, *, branch: bool = False) -> None:
        raw = _json_bytes(value)
        stage = _stage(path.parent, raw)
        self._count(reference_written_bytes=len(raw))
        try:
            if branch:
                self._fault("after_refs_stage_fsync")
            os.replace(stage, path)
            if branch:
                self._count(reference_publications=1)
                self._fault("after_refs_rename_before_dir_fsync")
            _sync_dir(path.parent)
        finally:
            stage.unlink(missing_ok=True)

    def _publish_format(self, directory: str) -> None:
        self._publish_reference_json(self.refs_path, dict(schema=2, directory=directory))

    def _set_reference_directory(self, marker: dict) -> None:
        directory = marker.get('directory')
        if set(marker) != {'schema', 'directory'} or type(marker['schema']) is not int or marker['schema'] != 2 or not isinstance(directory, str):
            raise IntegrityError('Invalid reference directory schema')
        if directory != 'references' and not re.fullmatch(r'references-[0-9a-f]{64}', directory):
            raise IntegrityError('Invalid reference directory name')
        self.references = self.root / directory
        if not self.references.is_dir():
            raise IntegrityError('Missing reference directory')

    def _check_reference_format(self) -> None:
        self._set_reference_directory(self._read_reference_json(self.refs_path))

    def _write_order(self, next_order: int) -> None:
        self._publish_reference_json(self.order_path, dict(schema=1, next=next_order))

    def _reserve_order(self) -> int:
        # Reserve a fixed-size audit ordinal before publishing a branch. A
        # failed transaction may leave a gap; it never creates an audit event
        # or changes a branch. The global file lock serializes reservations.
        value = self._read_reference_json(self.order_path)
        if set(value) != {'schema', 'next'} or type(value['schema']) is not int or value['schema'] != 1 or type(value['next']) is not int or not 0 <= value['next'] < 2**64 - 1:
            raise IntegrityError('Invalid or exhausted reference order counter')
        self._write_order(value['next'] + 1)
        return value['next']

    def _read_branch(self, name: str, *, _source_chain=()) -> dict:
        if name in _source_chain:
            raise IntegrityError('Cycle in namespace fork source provenance')
        source_chain = _source_chain + (name,)
        path = self.references / (name + '.json')
        if not path.exists():
            raise NotFound(f'Unknown branch: {name}')
        value = self._read_reference_json(path)
        base_fields = {'schema', 'name', 'head', 'receipts', 'events', 'orders'}
        schema = value.get('schema')
        if (type(schema) is not int or schema not in (1, 2)
                or set(value) != (base_fields if schema == 1 else base_fields | {'namespace_id'})
                or value['name'] != name):
            raise IntegrityError('Invalid branch record')
        if schema == 2:
            namespace_id = value['namespace_id']
            if not isinstance(namespace_id, str) or not _NAMESPACE_HEX.fullmatch(namespace_id):
                raise IntegrityError('Invalid branch namespace identity')
            try:
                self._read_namespace(namespace_id)
            except (InvalidOperation, NotFound, PermissionError) as error:
                raise IntegrityError('Branch presentation namespace is unavailable') from error
        head = value['head']
        if not isinstance(head, dict) or set(head) != {'revision', 'tip', 'root'} or type(head['revision']) is not int or head['revision'] < 0:
            raise IntegrityError('Invalid branch head')
        try:
            _id(head['tip']); _id(head['root'])
        except InvalidOperation as error:
            raise IntegrityError('Invalid branch snapshot identity') from error
        if not isinstance(value['receipts'], dict) or not isinstance(value['events'], list) or not isinstance(value['orders'], list):
            raise IntegrityError('Invalid branch receipt/event containers')
        if len(value['events']) != len(value['orders']) or any(type(x) is not int or x < 0 for x in value['orders']):
            raise IntegrityError('Invalid branch event order')
        # Optional checkpoint evidence is inline with the existing CAS receipt.
        # This common validation covers both legacy and namespace branch files.
        events_by_operation = {
            event.get('operation_id'): event for event in value['events']
            if isinstance(event, dict) and event.get('type') == 'move'
        }
        for operation, receipt in value['receipts'].items():
            if not isinstance(receipt, dict):
                continue  # Existing schema-specific validation reports it below.
            event = events_by_operation.get(operation)
            annotated = ('checkpoint_receipt' in receipt
                         or 'checkpoint_receipt_digest' in receipt
                         or isinstance(event, dict) and 'checkpoint_receipt_digest' in event)
            if not annotated:
                continue
            if (not isinstance(event, dict) or 'checkpoint_receipt' not in receipt
                    or 'checkpoint_receipt_digest' not in receipt
                    or event.get('checkpoint_receipt_digest') != receipt.get('checkpoint_receipt_digest')):
                raise IntegrityError('Checkpoint receipt and move event differ')
            try:
                descriptor, digest = self._validate_checkpoint_receipt(
                    receipt['checkpoint_receipt'], event.get('to'), stored=True)
                if digest != receipt['checkpoint_receipt_digest']:
                    raise IntegrityError('Checkpoint receipt digest mismatch')
                result = receipt.get('result')
                from_revision = event.get('from_revision')
                to_revision = event.get('to_revision')
                event_target = _id(event.get('to'))
                event_source = _id(event.get('from'))
                if (event.get('branch') != name
                        or type(from_revision) is not int or from_revision < 0
                        or type(to_revision) is not int or to_revision != from_revision + 1
                        or not isinstance(result, dict)
                        or set(result) != {'name', 'revision', 'tip', 'root'}
                        or result.get('name') != name
                        or result.get('revision') != to_revision
                        or result.get('tip') != event_target
                        or result.get('root') != head['root']
                        or descriptor['snapshot'] != event_target):
                    raise IntegrityError('Checkpoint receipt result differs from its move event')
                request = {'type': 'move', 'name': name,
                           'expected_revision': from_revision, 'target': event_target}
                if 'presentation_digest' in receipt:
                    request['presentation_digest'] = receipt['presentation_digest']
                if 'presentation_source_operation' in receipt:
                    request['presentation_source_operation'] = receipt['presentation_source_operation']
                if 'adoption_presentation_source_operation' in receipt:
                    request['adoption_presentation_source_operation'] = receipt[
                        'adoption_presentation_source_operation']
                request['checkpoint_receipt_digest'] = digest
                if receipt.get('digest') != _hash(_json_bytes(request)):
                    raise IntegrityError('Checkpoint receipt move binding mismatch')
            except (InvalidOperation, KeyError, TypeError) as error:
                raise IntegrityError('Invalid stored checkpoint receipt') from error
        if schema == 2:
            events_by_operation = {}
            for event in value['events']:
                if not isinstance(event, dict):
                    raise IntegrityError('Invalid namespace branch event')
                if event.get('type') == 'fork':
                    operation = event.get('operation_id')
                    if not isinstance(operation, str):
                        raise IntegrityError('Invalid namespace fork event operation')
                    try:
                        _name(operation)
                        _id(event.get('tip'))
                    except InvalidOperation as error:
                        raise IntegrityError('Invalid namespace fork event identity') from error
                    if operation in events_by_operation:
                        raise IntegrityError('Duplicate namespace branch event operation')
                    events_by_operation[operation] = event
                elif event.get('type') == 'move':
                    operation = event.get('operation_id')
                    if not isinstance(operation, str) or operation in events_by_operation:
                        raise IntegrityError('Invalid namespace branch move event')
                    try:
                        _name(operation)
                        _id(event.get('from'))
                        _id(event.get('to'))
                    except InvalidOperation as error:
                        raise IntegrityError('Invalid namespace branch move identity') from error
                    events_by_operation[operation] = event
                else:
                    raise IntegrityError('Unsupported namespace branch event')
                if event.get('namespace_id') != value['namespace_id']:
                    raise IntegrityError('Branch event namespace differs from immutable branch namespace')
            fork_events = [event for event in value['events'] if event['type'] == 'fork']
            if len(fork_events) != 1:
                raise IntegrityError('Namespace branch must contain exactly one fork event')
            if set(events_by_operation) != set(value['receipts']):
                raise IntegrityError('Namespace branch events and receipts differ')
            fork_event = fork_events[0]
            fork_operation = fork_event.get('operation_id')
            if not isinstance(fork_operation, str) or fork_operation not in value['receipts']:
                raise IntegrityError('Namespace fork event lacks a retry receipt')
            fork_receipt = value['receipts'][fork_operation]
            fork_result = fork_receipt.get('result')
            if (fork_event.get('branch') != name or not isinstance(fork_result, dict) or fork_result.get('name') != name
                    or fork_result.get('revision') != 0 or fork_result.get('tip') != fork_event.get('tip')
                    or fork_result.get('root') != head['root']):
                raise IntegrityError('Namespace fork receipt differs from its event')
            fork_request = {'type': 'fork', 'name': name, 'snapshot': fork_event['tip'],
                            'namespace_id': value['namespace_id']}
            source = fork_event.get('source')
            if 'source' in fork_event:
                if not isinstance(source, dict) or set(source) != {'branch', 'revision', 'tip'}:
                    raise IntegrityError('Invalid namespace fork source provenance')
                try:
                    source_branch = _name(source['branch'])
                    source_tip = _id(source['tip'])
                except (KeyError, InvalidOperation) as error:
                    raise IntegrityError('Invalid namespace fork source identity') from error
                source_revision = source['revision']
                if type(source_revision) is not int or source_revision < 0 or source_branch == name:
                    raise IntegrityError('Invalid namespace fork source revision')
                try:
                    source_record = self._read_branch(source_branch, _source_chain=source_chain)
                except NotFound as error:
                    raise IntegrityError('Namespace fork source branch is unavailable') from error
                if (source_record.get('schema') != 2
                        or source_record.get('namespace_id') != value['namespace_id']
                        or source_record['head']['root'] != head['root']):
                    raise IntegrityError('Namespace fork source scope differs from target')
                source_events = source_record['events']
                if source_revision == 0:
                    revision_tip = next(event['tip'] for event in source_events if event['type'] == 'fork')
                else:
                    revision_events = [event for event in source_events
                                       if event['type'] == 'move' and event.get('to_revision') == source_revision]
                    if len(revision_events) != 1:
                        raise IntegrityError('Namespace fork source revision is unavailable')
                    revision_tip = revision_events[0]['to']
                if source_tip != revision_tip or self._root_for(source_tip) != head['root']:
                    raise IntegrityError('Namespace fork source tip differs from captured revision')
                fork_request['source'] = source
            fork_digest = fork_receipt.get('presentation_digest')
            if fork_digest is not None:
                try:
                    _id(fork_digest)
                except InvalidOperation as error:
                    raise IntegrityError('Invalid namespace fork presentation digest') from error
                fork_request['presentation_digest'] = fork_digest
            if fork_receipt.get('digest') != _hash(_json_bytes(fork_request)):
                raise IntegrityError('Namespace fork request digest mismatch')
            last_event = value['events'][-1]
            expected_revision = last_event.get('to_revision', 0)
            expected_tip = last_event.get('to', last_event.get('tip'))
            if head['revision'] != expected_revision or head['tip'] != expected_tip:
                raise IntegrityError('Namespace branch head differs from its event history')
            for operation, receipt in value['receipts'].items():
                if not isinstance(operation, str) or not isinstance(receipt, dict):
                    raise IntegrityError('Invalid namespace branch receipt')
                event = events_by_operation.get(operation)
                if event is None:
                    raise IntegrityError('Namespace branch receipt lacks an event')
                if event['type'] == 'move':
                    if (event.get('branch') != name or type(event.get('from_revision')) is not int
                            or type(event.get('to_revision')) is not int
                            or event.get('to_revision') != event.get('from_revision') + 1):
                        raise IntegrityError('Invalid namespace branch move event')
                    request = {'type': 'move', 'name': name,
                               'expected_revision': event['from_revision'], 'target': event.get('to')}
                    event_digest = receipt.get('presentation_digest')
                    if event_digest is not None:
                        request['presentation_digest'] = event_digest
                    checkpoint_digest = receipt.get('checkpoint_receipt_digest')
                    if checkpoint_digest is not None:
                        request['checkpoint_receipt_digest'] = checkpoint_digest
                    source_operation=receipt.get('presentation_source_operation')
                    if source_operation is not None:
                        if event.get('presentation_source_operation')!=source_operation:
                            raise IntegrityError('Namespace retry source receipt differs from its event')
                        request['presentation_source_operation']=source_operation
                    adoption_source=receipt.get('adoption_presentation_source_operation')
                    if (event.get('adoption_presentation_source_operation') != adoption_source
                            or ('adoption_presentation_source_operation' in event) != (adoption_source is not None)):
                        raise IntegrityError('Adoption presentation source receipt differs from its event')
                    if adoption_source is not None:
                        request['adoption_presentation_source_operation']=adoption_source
                        adoption=self._read_evaluation_operation(operation)
                        if (adoption is None or adoption.get('kind')!='adoption'
                                or adoption.get('branch')!=name
                                or adoption.get('selected_snapshot_id')!=event.get('to')
                                or adoption.get('adoption_presentation_source_operation')!=adoption_source):
                            raise IntegrityError('Adoption presentation source is not bound to its operation')
                    if receipt.get('digest') != _hash(_json_bytes(request)):
                        raise IntegrityError('Namespace move request digest mismatch')
                    result = receipt.get('result')
                    if (not isinstance(result, dict) or result.get('name') != name
                            or result.get('revision') != event['to_revision']
                            or result.get('tip') != event.get('to') or result.get('root') != head['root']):
                        raise IntegrityError('Namespace move receipt differs from its event')
                digest = receipt.get('presentation_digest')
                if digest is not None:
                    try:
                        _id(digest)
                    except InvalidOperation as error:
                        raise IntegrityError('Invalid presentation receipt digest') from error
                    if (event is None or event.get('presentation_digest') != digest
                            or event.get('namespace_id') != value['namespace_id']):
                        raise IntegrityError('Presentation receipt and branch event differ')
        return value

    def _read_refs(self) -> dict:
        # Full reference enumeration is for receipts/events, never fork. Move
        # still scans committed receipts to keep operation IDs store-global.
        self._check_reference_format()
        result = self._empty_refs()
        events = []
        for path in sorted(self.references.glob('*.json')):
            name = _name(path.stem)
            record = self._read_branch(name)
            self._count(reference_entries_scanned=1 + len(record['receipts']) + len(record['events']))
            result['branches'][name] = record['head']
            for operation, receipt in record['receipts'].items():
                if operation in result['receipts']:
                    raise IntegrityError('Duplicate committed operation ID')
                result['receipts'][operation] = receipt
            events.extend(zip(record['orders'], record['events']))
        orders = [order for order, _ in events]
        if len(orders) != len(set(orders)):
            raise IntegrityError('Duplicate committed audit ordinal')
        result['events'] = [event for _, event in sorted(events, key=lambda row: row[0])]
        return result

    def _migrate_references(self, old: dict) -> None:
        if set(old) != {'schema', 'branches', 'receipts', 'events'} or type(old['schema']) is not int or old['schema'] != 1 or not isinstance(old['branches'], dict) or not isinstance(old['receipts'], dict) or not isinstance(old['events'], list):
            raise IntegrityError('Invalid legacy reference index')
        # Snapshot objects keep their exact bytes/IDs. Only the reference index
        # changes layout. Until the final marker rename, the old index is the
        # authority; a restart can repeat the same deterministic migration.
        directory = self.root / ('references-' + _hash(_json_bytes(old)))
        _private_directory(directory)
        records = {}
        for name, head in old['branches'].items():
            _name(name)
            records[name] = dict(schema=1, name=name, head=head, receipts={}, events=[], orders=[])
        for operation, receipt in old['receipts'].items():
            _name(operation)
            try:
                records[receipt['result']['name']]['receipts'][operation] = receipt
            except (KeyError, TypeError) as error:
                raise IntegrityError('Invalid legacy retry receipt') from error
        for order, event in enumerate(old['events']):
            try:
                record = records[event['branch']]
            except (KeyError, TypeError) as error:
                raise IntegrityError('Invalid legacy audit event') from error
            record['events'].append(event)
            record['orders'].append(order)
        self.references = directory
        for name, record in records.items():
            self._publish_reference_json(directory / (name + '.json'), record)
            self._read_branch(name)
        _sync_dir(directory)
        _sync_dir(self.root)
        self._write_order(len(old['events']))
        self._publish_format(directory.name)

    def _publish_root_index(self, identity: str, root: str) -> None:
        value = dict(schema=1, snapshot=identity, root=root)
        raw = _json_bytes(dict(value=value, checksum=_hash(_json_bytes(value))))
        path = self.root_index / (identity + '.json')
        if path.exists():
            if self._read_root_index(identity) != root:
                raise IntegrityError('Conflicting snapshot root lookup')
            return
        stage = _stage(self.root_index, raw)
        self._count(root_index_written_bytes=len(raw))
        try:
            try:
                os.link(stage, path)
            except FileExistsError:
                if self._read_root_index(identity) != root:
                    raise IntegrityError('Conflicting snapshot root lookup')
            else:
                _sync_dir(self.root_index)
        finally:
            stage.unlink(missing_ok=True)

    def _read_root_index(self, identity: str) -> str:
        path = self.root_index / (identity + '.json')
        raw = path.read_bytes()
        self._count(root_index_read_bytes=len(raw))
        envelope = _decode(raw)
        value = envelope.get('value')
        if not isinstance(value, dict) or set(envelope) != {'value', 'checksum'} or envelope['checksum'] != _hash(_json_bytes(value)):
            raise IntegrityError('Corrupt snapshot root lookup')
        if set(value) != {'schema', 'snapshot', 'root'} or value['schema'] != 1 or value['snapshot'] != identity:
            raise IntegrityError('Incompatible snapshot root lookup')
        try:
            return _id(value['root'])
        except InvalidOperation as error:
            raise IntegrityError('Invalid root lookup identity') from error

    def _root_for(self, identity: str) -> str:
        # This derived lookup copies no inputs/controller. Canonical reads and
        # materialization still hash/validate every actual history object.
        try:
            root = self._read_root_index(identity)
        except FileNotFoundError:
            root = self._validated_root(identity)
            self._publish_root_index(identity, root)
            self._count(root_index_repairs=1)
        for retained in (identity, root):
            if not (self.objects / (retained + '.json')).is_file():
                raise NotFound(f'Missing retained snapshot: {retained}')
        return root

    def _put_object(self, record: dict, root: str | None = None) -> str:
        raw = _json_bytes(record)
        digest = _hash(raw)
        destination = self.objects / (digest + ".json")
        if destination.exists():
            self._verify_object_file(destination, digest)
            self._publish_root_index(digest, root or digest)
            return digest
        stage = _stage(self.objects, raw)
        self._count(snapshot_written_bytes=len(raw))
        try:
            self._fault("before_snapshot_publish")
            # link is atomic, and unlike replace will never overwrite an
            # existing immutable object (even in a concurrent writer race).
            try:
                os.link(stage, destination)
            except FileExistsError:
                self._verify_object_file(destination, digest)
            else:
                _sync_dir(self.objects)
        finally:
            stage.unlink(missing_ok=True)
        self._publish_root_index(digest, root or digest)
        return digest

    def _verified_object_bytes(self, path: Path, expected: str) -> bytes:
        try:
            raw = path.read_bytes()
        except FileNotFoundError as exc:
            raise NotFound(f"Missing object: {expected}") from exc
        self._count(snapshot_reads=1, snapshot_read_bytes=len(raw))
        if _hash(raw) != expected:
            raise IntegrityError(f"SHA-256 mismatch in object {expected}")
        return raw

    def _verify_object_file(self, path: Path, expected: str) -> dict:
        return _decode(self._verified_object_bytes(path,expected))

    def _ancestry_record(self, identity):
        # Only derived parent/input metadata is memoized. Every traversal still
        # reads and hashes the complete canonical object, including controller
        # bytes, so corruption/deletion cannot be hidden by this cache.
        self.authorize(identity)
        if identity in self._ancestry_metadata:
            self._verified_object_bytes(self.objects/(identity+'.json'),identity)
            self._ancestry_metadata.move_to_end(identity)
            self._count(ancestry_metadata_hits=1)
            return self._ancestry_metadata[identity]
        record = self.get_snapshot(identity)
        value = (record['model_contract_id'],record['parent'],tuple(record['inputs']),record['schema'])
        self._ancestry_metadata[identity] = value
        self._count(ancestry_metadata_misses=1)
        if len(self._ancestry_metadata) > 256:
            self._ancestry_metadata.popitem(last=False)
        return value

    def _validated_root(self, identity):
        """Verify canonical ancestry without assuming or flattening token inputs."""
        cursor = _id(identity)
        seen = set()
        expected = None
        while True:
            if cursor in seen:
                raise IntegrityError('Cycle in snapshot ancestry')
            seen.add(cursor)
            self._count(ancestry_nodes_visited=1)
            contract, parent, _inputs, schema = self._ancestry_record(cursor)
            if expected is None:
                expected = (contract, schema)
            elif expected != (contract, schema):
                raise IntegrityError('Cross-contract or cross-input-version parent reference')
            if parent is None:
                return cursor
            cursor = parent

    def get_snapshot(self, snapshot_id: str) -> dict:
        identity = _id(snapshot_id)
        self.authorize(identity)
        # The complete canonical file is checked on every read. The bounded
        # cache saves JSON parsing only; it cannot hide corruption or deletion.
        raw = self._verified_object_bytes(self.objects / (identity + ".json"), identity)
        cache = self._snapshot_parse_cache
        with cache['lock']:
            cached = cache['entries'].get(identity)
            if cached is not None: cache['entries'].move_to_end(identity)
        if cached is not None:
            self._count(snapshot_parse_hits=1)
            return self._copy_snapshot_record(cached[0],cached[1])
        self._count(snapshot_parse_misses=1)
        record = _decode(raw)
        if set(record) != {"kind", "schema", "model_contract_id", "parent", "inputs", "controller"}:
            raise IntegrityError("Unrecognized snapshot schema")
        if (record["kind"] != "snapshot" or type(record['schema']) is not int
                or record["schema"] not in (1, 2)):
            raise IntegrityError("Unsupported snapshot kind/version")
        if not isinstance(record["model_contract_id"], str) or not record["model_contract_id"]:
            raise IntegrityError("Invalid model contract identity")
        if record["parent"] is not None:
            try:
                _id(record["parent"])
            except InvalidOperation as exc:
                raise IntegrityError("Invalid parent reference") from exc
        try:
            if record['schema'] == 1:
                _tokens(record["inputs"])
            else:
                from .input_spans import canonical_inputs
                if canonical_inputs(record['inputs']) != record['inputs']:
                    raise InvalidOperation('Mixed input spans are not canonical')
            # _verify_object_file already decoded fresh, hash-checked canonical
            # JSON. Its controller is detached and JSON-valid; only its object
            # shape remains to check. Re-encoding a large policy adds no check.
            if not isinstance(record["controller"], dict):
                raise InvalidOperation("Controller state must be a JSON-compatible mapping")
        except InvalidOperation as exc:
            raise IntegrityError("Invalid stored snapshot fields") from exc
        policy = record['controller'].get('policy')
        allowed = policy.get('allowed') if isinstance(policy,dict) else None
        atomic_support = isinstance(allowed,list) and all(type(t) is int for t in allowed)
        # Bound retained serialized size; parsed Python objects cost more RAM.
        # One oversized object is returned normally without being cached.
        if len(raw) <= 32*1024*1024:
            with cache['lock']:
                previous = cache['entries'].pop(identity,None)
                if previous is not None: cache['bytes'] -= previous[2]
                cache['entries'][identity] = (record,atomic_support,len(raw))
                cache['bytes'] += len(raw)
                while cache['bytes'] > 32*1024*1024 or len(cache['entries']) > 256:
                    _,removed = cache['entries'].popitem(last=False)
                    cache['bytes'] -= removed[2]
        return self._copy_snapshot_record(record,atomic_support)

    @staticmethod
    def _copy_snapshot_record(record, atomic_support):
        # JSON input IDs and a validated integer support list have no mutable
        # children. Copy those lists in C; deepcopy all other fields normally.
        memo = ({id(record['inputs']):record['inputs'].copy()}
                if record['schema'] == 1 else {})
        if atomic_support:
            allowed = record['controller']['policy']['allowed']
            memo[id(allowed)] = allowed.copy()
        return copy.deepcopy(record,memo)

    @_history_transaction
    def put_evaluation(self, record: dict) -> str:
        """Publish a report without changing any history, controller or reference."""
        from .evaluation import validate_record
        value = validate_record(record)
        self.authorize(value['snapshot_id'], 'evaluate')
        cursor = value['snapshot_id']
        seen = set()
        contract = None
        while cursor is not None:
            if cursor in seen:
                raise IntegrityError('Cycle in evaluated snapshot ancestry')
            seen.add(cursor)
            snapshot = self.get_snapshot(cursor)
            if contract is None:
                contract = snapshot['model_contract_id']
            elif contract != snapshot['model_contract_id']:
                raise IntegrityError('Cross-contract evaluated ancestry')
            cursor = snapshot['parent']
        if value['task_root'] not in seen:
            raise InvalidOperation('Evaluation task root must be an ancestor of its candidate')
        raw = _json_bytes(value)
        identity = _hash(raw)
        directory = self.root / 'evaluations'
        _private_directory(directory)
        _sync_dir(self.root)
        destination = directory / (identity + '.json')
        if destination.exists():
            self._index_evaluation_access(identity, value['snapshot_id'])
            self.get_evaluation(identity)
            return identity
        stage = _stage(directory, raw)
        self._count(evaluation_written_bytes=len(raw))
        try:
            self._fault('before_evaluation_publish')
            try:
                os.link(stage, destination)
            except FileExistsError:
                self.get_evaluation(identity)
            else:
                _sync_dir(directory)
            self._index_evaluation_access(identity, value['snapshot_id'])
        finally:
            stage.unlink(missing_ok=True)
        return identity

    def get_evaluation(self, evaluation_id: str) -> dict:
        """Read a detached, hash-checked report, never a mutable model snapshot."""
        from .evaluation import validate_record
        identity = _id(evaluation_id)
        if self._principal is not None:
            try:
                index = _decode((self.root/'evaluation-access'/(identity+'.json')).read_bytes())
            except FileNotFoundError as error:
                raise PermissionError('No evaluation access index') from error
            self.authorize(index['snapshot'], 'evaluate')
            self.authorize(index['snapshot'], 'read')
        try:
            raw = (self.root / 'evaluations' / (identity + '.json')).read_bytes()
        except FileNotFoundError as error:
            raise NotFound('Missing evaluation: ' + identity) from error
        self._count(evaluation_reads=1, evaluation_read_bytes=len(raw))
        if _hash(raw) != identity:
            raise IntegrityError('SHA-256 mismatch in evaluation ' + identity)
        try:
            return validate_record(_decode(raw))
        except InvalidOperation as error:
            raise IntegrityError('Invalid stored evaluation') from error

    def evaluate_once(self, operation_id: str, snapshot_id: str, specification: dict,
                      fixtures: dict[str, bytes], compute: Callable[[], dict]) -> dict:
        """Run an evaluation and durably bind its report to one operation ID."""
        operation = _name(operation_id)
        snapshot = _id(snapshot_id)
        self.authorize(snapshot, 'evaluate')
        self.authorize(snapshot, 'read')
        if not callable(compute):
            raise InvalidOperation('Evaluation callback must be callable')
        if not isinstance(specification, dict) or 'fixtures' in specification:
            raise InvalidOperation('Evaluation specification must be an object without fixture bytes')
        if (not isinstance(fixtures, dict)
                or any(not isinstance(key, str) or type(value) is not bytes
                       for key, value in fixtures.items())):
            raise InvalidOperation('Evaluation fixtures must map names to exact bytes')
        from .evaluation import validate_request
        validate_request(self, snapshot, dict(specification, fixtures=fixtures))
        request_digest = _hash(_json_bytes(dict(
            kind='evaluate', snapshot_id=snapshot, specification=specification,
            fixtures={key: value.hex() for key, value in sorted(fixtures.items())})))
        with self._evaluation_operation_locked(operation), self.read_lease(snapshot):
            current = self._read_evaluation_operation(operation)
            if current is not None:
                if current.get('kind') != 'evaluate':
                    raise InvalidOperation('Operation ID is already used by another endpoint')
                if current.get('request_digest') != request_digest:
                    raise InvalidOperation('Operation ID reused for different input')
                if current.get('state') == 'committed':
                    identity = current.get('result', {}).get('evaluation_id')
                    return {'operation_id': operation, 'evaluation_id': identity,
                            'record': self.get_evaluation(identity)}
            current = self._claim_evaluation_operation(operation, 'evaluate', request_digest,
                                                       dict(snapshot_id=snapshot))
            record = compute()
            from .evaluation import validate_record
            record = validate_record(record)
            if record.get('snapshot_id') != snapshot:
                raise InvalidOperation('Evaluation callback returned a different snapshot')
            evaluator = specification.get('evaluator')
            if (not isinstance(evaluator,dict)
                    or record.get('task_root') != specification.get('task_root')
                    or record.get('metric') != specification.get('metric')
                    or record.get('seed') != specification.get('seed')
                    or record.get('evaluator',{}).get('name') != evaluator.get('name')
                    or record.get('evaluator',{}).get('version') != evaluator.get('version')
                    or record.get('evaluator',{}).get('source_sha256') != _hash(
                        evaluator.get('source','').encode('utf-8') if isinstance(evaluator.get('source'),str) else b'')
                    or record.get('fixtures') != {key:_hash(value) for key,value in fixtures.items()}):
                raise InvalidOperation('Evaluation callback output differs from the bound request')
            identity = self.put_evaluation(record)
            verified = self.get_evaluation(identity)
            self._fault('before_evaluation_operation_commit')
            with self._locked():
                latest = self._read_evaluation_operation(operation)
                if (latest is None or latest.get('kind') != 'evaluate'
                        or latest.get('request_digest') != request_digest
                        or latest.get('state') != 'pending'):
                    raise IntegrityError('Evaluation operation changed before commit')
                self._commit_evaluation_operation(latest, {'evaluation_id': identity})
            self._fault('after_evaluation_operation_commit')
            return {'operation_id': operation, 'evaluation_id': identity, 'record': verified}

    def adopt_evaluated_once(self, operation_id: str, branch: str,
                             expected_revision: int, evaluation_ids: list[str],
                             mandatory: list[str], *,
                             presentation_source_operation: str | None = None) -> dict:
        """Select from immutable reports, binding selection before the Store CAS."""
        operation = _name(operation_id)
        branch = _name(branch)
        if type(expected_revision) is not int or expected_revision < 0:
            raise InvalidOperation('Expected revision must be a nonnegative integer')
        if (not isinstance(evaluation_ids, list) or not evaluation_ids or len(evaluation_ids)>256
                or any(not isinstance(value, str) for value in evaluation_ids)):
            raise InvalidOperation('Evaluation IDs must be a list of 1 to 256 values')
        identities = [_id(value) for value in evaluation_ids]
        if len(set(identities)) != len(identities):
            raise InvalidOperation('Evaluation IDs must be unique')
        if (not isinstance(mandatory, list) or not mandatory or len(mandatory)>64
                or any(not isinstance(value, str) for value in mandatory)):
            raise InvalidOperation('Mandatory checks must be a list of 1 to 64 values')
        checks = list(mandatory)
        if presentation_source_operation is not None:
            presentation_source_operation = _name(presentation_source_operation)
        request = dict(kind='adoption', branch=branch,
            expected_revision=expected_revision, evaluation_ids=identities, mandatory=checks)
        if presentation_source_operation is not None:
            request['presentation_source_operation'] = presentation_source_operation
        request_digest = _hash(_json_bytes(request))
        with self._evaluation_operation_locked(operation):
            current = self._read_evaluation_operation(operation)
            new_operation = current is None
            if current is not None:
                if current.get('kind') != 'adoption':
                    raise InvalidOperation('Operation ID is already used by another endpoint')
                if current.get('request_digest') != request_digest:
                    raise InvalidOperation('Operation ID reused for different input')
                if current.get('state') == 'committed':
                    result = current.get('result')
                    head = self.move(branch, expected_revision, result['snapshot_id'], operation,
                        adoption_presentation_source_operation=
                            current.get('adoption_presentation_source_operation'),
                        _evaluation_operation=True)
                    return dict(operation_id=operation, evaluation_id=result['evaluation_id'],
                                snapshot_id=result['snapshot_id'], branch=head,
                                **({'presentation_source_operation':
                                    current['adoption_presentation_source_operation']}
                                   if current.get('adoption_presentation_source_operation') is not None else {}))
            else:
                current = {}
            selected_id = current.get('selected_evaluation_id')
            selected_snapshot = current.get('selected_snapshot_id')
            if selected_id is None:
                from .evaluation import choose_validated
                chosen = choose_validated(self, identities, mandatory=checks)
                if chosen is None:
                    raise InvalidOperation('No evaluation satisfies the mandatory checks')
                selected_id = next(identity for identity in identities
                                   if self.get_evaluation(identity) == chosen)
                selected_snapshot = chosen['snapshot_id']
            # Preserve an already-published legacy adoption move exactly. Old
            # pending intents did not bind presentation provenance; changing
            # their move digest during lost-ACK recovery would strand a valid
            # CAS receipt and rewrite the operation's semantics.
            if (not new_operation and current.get('state') == 'pending'
                    and 'adoption_presentation_source_operation' not in current
                    and 'requested_presentation_source_operation' not in current
                    and operation in self._read_refs()['receipts']):
                head = self.move(branch, expected_revision, selected_snapshot, operation,
                                 _evaluation_operation=True)
                legacy_result = dict(evaluation_id=selected_id, snapshot_id=selected_snapshot,
                    branch=dict(name=head.name, revision=head.revision,
                                tip=head.tip, root=head.root))
                with self._locked():
                    latest = self._read_evaluation_operation(operation)
                    if (latest is None or latest.get('state') != 'pending'
                            or latest.get('request_digest') != request_digest):
                        raise IntegrityError('Legacy adoption operation changed before receipt recovery')
                    self._commit_evaluation_operation(latest, legacy_result)
                return dict(operation_id=operation, evaluation_id=selected_id,
                            snapshot_id=selected_snapshot, branch=head)
            bound_source = current.get('adoption_presentation_source_operation')
            if bound_source is None:
                bound_source = self._resolve_adoption_presentation_source(
                    branch, selected_snapshot, presentation_source_operation)
            elif (presentation_source_operation is not None
                    and bound_source != presentation_source_operation):
                raise InvalidOperation('Operation ID reused for different presentation provenance')
            updated = dict(current, selected_evaluation_id=selected_id,
                           selected_snapshot_id=selected_snapshot)
            if bound_source is not None:
                updated['adoption_presentation_source_operation'] = bound_source
            if new_operation:
                details = dict(branch=branch, expected_revision=expected_revision,
                    evaluation_ids=identities, mandatory=checks,
                    selected_evaluation_id=selected_id, selected_snapshot_id=selected_snapshot)
                if presentation_source_operation is not None:
                    details['requested_presentation_source_operation'] = presentation_source_operation
                if bound_source is not None:
                    details['adoption_presentation_source_operation'] = bound_source
                current = self._claim_evaluation_operation(
                    operation, 'adoption', request_digest, details)
            elif updated != current:
                with self._locked():
                    latest = self._read_evaluation_operation(operation)
                    if (latest is None or latest.get('state') != 'pending'
                            or latest.get('request_digest') != request_digest):
                        raise IntegrityError('Adoption operation changed before selection was bound')
                    self._write_evaluation_operation(updated)
                current = updated
            self._fault('before_evaluation_adoption_move')
            with self.read_lease(selected_snapshot):
                head = self.move(branch, expected_revision, selected_snapshot, operation,
                    adoption_presentation_source_operation=bound_source,
                    _evaluation_operation=True)
                self._fault('after_evaluation_adoption_move')
                result = dict(evaluation_id=selected_id, snapshot_id=selected_snapshot,
                              branch=dict(name=head.name, revision=head.revision,
                                          tip=head.tip, root=head.root))
                if bound_source is not None:
                    result['adoption_presentation_source_operation'] = bound_source
                with self._locked():
                    latest = self._read_evaluation_operation(operation)
                    if (latest is None or latest.get('state') != 'pending'
                            or latest.get('request_digest') != request_digest):
                        raise IntegrityError('Adoption operation changed before receipt commit')
                    self._commit_evaluation_operation(latest, result)
                response = dict(operation_id=operation, evaluation_id=selected_id,
                                snapshot_id=selected_snapshot, branch=head)
                if bound_source is not None:
                    response['presentation_source_operation'] = bound_source
                return response

    @_history_transaction
    def create_root(self, model_contract_id: str,
                    inputs: tuple[int, ...] | list[int],
                    controller: Mapping[str, Any]) -> str:
        if self._principal is not None:
            raise PermissionError('Only the local owner can create history roots')
        if not isinstance(model_contract_id, str) or not model_contract_id:
            raise InvalidOperation("Model contract identity must be nonempty")
        return self._put_object({"kind": "snapshot", "schema": 1,
                                 "model_contract_id": model_contract_id,
                                 "parent": None,
                                 "inputs": _tokens(inputs),
                                 "controller": _controller(controller)})

    @_history_transaction
    def append(self, parent_id: str, inputs: tuple[int, ...] | list[int],
               controller: Mapping[str, Any]) -> str:
        self.authorize(parent_id, 'append')
        parent = self.get_snapshot(parent_id)
        if parent['schema'] != 1:
            raise InvalidOperation('Use append_inputs for a mixed-input history')
        root = self._root_for(_id(parent_id))
        return self._put_object({"kind": "snapshot", "schema": 1,
                                 "model_contract_id": parent["model_contract_id"],
                                 "parent": _id(parent_id),
                                 "inputs": _tokens(inputs),
                                 "controller": _controller(controller)}, root)

    @_history_transaction
    def lineage(self, snapshot_id: str) -> tuple[str, list[int]]:
        """Return (root_id, exact inputs); validate every ancestor and contract."""
        cursor = _id(snapshot_id)
        seen: set[str] = set()
        chunks: list[list[int]] = []
        contract = None
        while True:
            self._count(ancestry_nodes_visited=1)
            if cursor in seen:
                raise IntegrityError("Cycle in snapshot ancestry")
            seen.add(cursor)
            model_contract,parent,inputs,schema = self._ancestry_record(cursor)
            if schema != 1:
                raise InvalidOperation('Use input_lineage for a mixed-input history')
            if contract is None:
                contract = model_contract
            elif contract != model_contract:
                raise IntegrityError("Cross-contract parent reference")
            chunks.append(inputs)
            if parent is None:
                return cursor, [token for chunk in reversed(chunks) for token in chunk]
            cursor = parent

    @_history_transaction
    def create_input_root(self, model_contract_id: str, inputs,
                          controller: Mapping[str, Any]) -> str:
        """Create an explicitly mixed-input root without reinterpreting text IDs."""
        from .input_spans import canonical_inputs
        if self._principal is not None:
            raise PermissionError('Only the local owner can create history roots')
        if not isinstance(model_contract_id, str) or not model_contract_id:
            raise InvalidOperation('Model contract identity must be nonempty')
        return self._put_object(dict(kind='snapshot', schema=2,
            model_contract_id=model_contract_id, parent=None,
            inputs=canonical_inputs(inputs), controller=_controller(controller)))

    @_history_transaction
    def append_inputs(self, parent_id: str, inputs,
                      controller: Mapping[str, Any]) -> str:
        from .input_spans import canonical_inputs
        self.authorize(parent_id, 'append')
        parent = self.get_snapshot(parent_id)
        if parent['schema'] != 2:
            raise InvalidOperation('append_inputs requires an explicit mixed-input root')
        root = self._root_for(_id(parent_id))
        return self._put_object(dict(kind='snapshot', schema=2,
            model_contract_id=parent['model_contract_id'], parent=_id(parent_id),
            inputs=canonical_inputs(inputs), controller=_controller(controller)), root)

    @_history_transaction
    def input_lineage(self, snapshot_id: str):
        """Return canonical spans, projecting legacy tokens to text when needed.

        This is a read of the same parent-linked objects, not another history.
        Returned nested data never aliases the derived ancestry metadata cache.
        """
        from .input_spans import canonical_inputs
        cursor = _id(snapshot_id)
        seen = set()
        chunks = []
        contract = version = None
        while True:
            self._count(ancestry_nodes_visited=1)
            if cursor in seen:
                raise IntegrityError('Cycle in snapshot ancestry')
            seen.add(cursor)
            current_contract, parent, inputs, schema = self._ancestry_record(cursor)
            if contract is None:
                contract, version = current_contract, schema
            elif contract != current_contract or version != schema:
                raise IntegrityError('Cross-contract or cross-input-version parent reference')
            chunks.append([dict(kind='text', tokens=list(inputs))] if schema == 1 else inputs)
            if parent is None:
                return cursor, canonical_inputs([span for chunk in reversed(chunks) for span in chunk])
            cursor = parent

    @_history_transaction
    def fork(self, name: str, snapshot_id: str, *, namespace_id: str | None = None,
             operation_id: str | None = None, presentation: dict | None = None,
             source_branch: str | None = None, source_revision: int | None = None,
             source_tip: str | None = None) -> BranchHead:
        self.authorize(snapshot_id, 'fork')
        self.authorize(snapshot_id, 'read')
        name = _name(name)
        identity = _id(snapshot_id)
        root = self._root_for(identity)
        provided_source = (source_branch is not None, source_revision is not None, source_tip is not None)
        if any(provided_source) and not all(provided_source):
            raise InvalidOperation('Fork source provenance must be complete')
        source = None
        if all(provided_source):
            if namespace_id is None:
                raise InvalidOperation('Fork source provenance requires a namespace')
            source = {'branch': _name(source_branch), 'revision': source_revision,
                      'tip': _id(source_tip)}
            if type(source_revision) is not int or source_revision < 0 or source['branch'] == name:
                raise InvalidOperation('Invalid fork source provenance')
        if namespace_id is None:
            if operation_id is not None or presentation is not None:
                raise InvalidOperation('Fork operation and presentation require a namespace')
            operation = None
            namespace = None
            binding_raw = None
            binding_digest = None
            request_digest = None
        else:
            self._require_namespace_owner()
            if operation_id is None:
                raise InvalidOperation('Namespace-bound fork requires an operation ID')
            namespace = namespace_id
            self._read_namespace(namespace)
            operation = _name(operation_id)
            binding = None
            binding_raw = None
            binding_digest = None
            if presentation is not None:
                head = BranchHead(name, 0, identity, root)
                binding = self._validate_presentation(head, identity, presentation, namespace, operation)
                binding, binding_raw, binding_digest = self._presentation_digest(binding)
            request = {'type': 'fork', 'name': name, 'snapshot': identity,
                       'namespace_id': namespace}
            if source is not None:
                request['source'] = source
            if binding_digest is not None:
                request['presentation_digest'] = binding_digest
            request_digest = _hash(_json_bytes(request))
        with self._locked():
            self._check_reference_format()
            path = self.references / (name + '.json')
            if operation is not None:
                self._assert_operation_id_available(operation, allow_evaluation_kind='branch')
                refs = self._read_refs()
                old_receipt = refs['receipts'].get(operation)
                if old_receipt is not None:
                    if old_receipt.get('digest') != request_digest:
                        raise InvalidOperation('Operation ID reused for different fork request')
                    if old_receipt.get('presentation_digest') is not None:
                        self.presentation_receipt(operation)
                    result = old_receipt['result']
                    _sync_dir(self.references)
                    return BranchHead(result['name'], result['revision'], result['tip'], result['root'])
            if source is not None:
                try:
                    source_record = self._read_branch(source['branch'])
                except NotFound as error:
                    raise InvalidOperation('Fork source branch is unavailable') from error
                if (source_record.get('schema') != 2
                        or source_record.get('namespace_id') != namespace
                        or source_record['head']['root'] != root):
                    raise InvalidOperation('Fork source branch belongs to another namespace or root')
                if (source_record['head']['revision'] != source['revision']
                        or source_record['head']['tip'] != source['tip']):
                    raise Conflict('Fork source branch moved or changed namespace')
                if self._root_for(source['tip']) != root:
                    raise InvalidOperation('Fork source tip belongs to another root')
            if path.exists():
                raise Conflict(f"Branch exists: {name}")
            if binding_raw is not None:
                self._write_presentation_blob(binding_raw, binding_digest)
            if namespace is None:
                record = dict(schema=1, name=name, head=dict(revision=0, tip=identity, root=root),
                              receipts={}, events=[dict(type='fork', branch=name, tip=identity)],
                              orders=[self._reserve_order()])
            else:
                result = {'name': name, 'revision': 0, 'tip': identity, 'root': root}
                receipt = {'digest': request_digest, 'result': result}
                event = dict(type='fork', branch=name, tip=identity, operation_id=operation,
                             namespace_id=namespace)
                if source is not None:
                    event['source'] = source
                if binding_digest is not None:
                    receipt['presentation_digest'] = binding_digest
                    event['presentation_digest'] = binding_digest
                record = dict(schema=2, name=name, namespace_id=namespace,
                              head=dict(revision=0, tip=identity, root=root),
                              receipts={operation: receipt}, events=[event],
                              orders=[self._reserve_order()])
            if namespace is not None:
                self._fault('before_refs_stage')
            self._publish_reference_json(path, record, branch=True)
        return BranchHead(name, 0, identity, root)

    def head(self, name: str) -> BranchHead:
        name = _name(name)
        with self._locked():
            self._check_reference_format()
            entry = self._read_branch(name)['head']
        self.get_snapshot(entry["tip"])
        return BranchHead(name, entry["revision"], entry["tip"], entry["root"])

    @_history_transaction
    def branches(self, root: str | None = None) -> tuple[BranchHead, ...]:
        """Return one consistent, authorized snapshot of canonical branch heads.

        The catalog reads the per-branch records directly; it does not gather
        global receipts or audit events. For restricted principals, unreadable
        roots are omitted so their branch names and tips are not disclosed.
        A root-scoped query for an unreadable root is therefore indistinguishable
        from an empty catalog. The transaction lock makes all returned records
        correspond to one point in the serialized reference history.
        """
        requested_root = _id(root) if root is not None else None
        if requested_root is not None:
            try:
                self.authorize(requested_root, 'read')
            except PermissionError:
                return ()

        self._check_reference_format()
        result = []
        # Root and model-contract facts are enough for catalog integrity; a
        # public listing must not assemble every branch's potentially huge
        # token history. Memoizing each visited node also makes shared ancestry
        # cost once per query rather than once per branch.
        ancestry: dict[str, tuple[str, str]] = {}

        def root_contract(identity: str) -> tuple[str, str]:
            path: list[tuple[str, str, str | None]] = []
            seen: set[str] = set()
            cursor = identity
            while cursor not in ancestry:
                if cursor in seen:
                    raise IntegrityError('Cycle in snapshot ancestry')
                seen.add(cursor)
                self._count(ancestry_nodes_visited=1)
                contract, parent, _inputs, _schema = self._ancestry_record(cursor)
                path.append((cursor, contract, parent))
                if parent is None:
                    ancestry[cursor] = (cursor, contract)
                    path.pop()
                    break
                cursor = parent
            while path:
                node, contract, parent = path.pop()
                parent_root, parent_contract = ancestry[parent]
                if contract != parent_contract:
                    raise IntegrityError('Cross-contract parent reference')
                ancestry[node] = (parent_root, contract)
            return ancestry[identity]

        for path in sorted(self.references.glob('*.json')):
            name = _name(path.stem)
            record = self._read_branch(name)
            self._count(reference_entries_scanned=1)
            entry = record['head']
            if requested_root is not None and entry['root'] != requested_root:
                continue

            # Establish read access before traversing snapshots: lineage()
            # performs the same per-snapshot authorization checks as other
            # history reads. Filtering first keeps private branch identities
            # and contents out of a restricted principal's view.
            try:
                self.authorize(entry['root'], 'read')
                self.authorize(entry['tip'], 'read')
            except PermissionError:
                continue

            # Verify root and contract through the existing full-object
            # validator without materializing the concatenated token history.
            actual_root, _contract = root_contract(entry['tip'])
            if actual_root != entry['root']:
                raise IntegrityError('Branch root does not match snapshot ancestry')
            result.append(BranchHead(name, entry['revision'], entry['tip'], entry['root']))
        return tuple(result)

    @_history_transaction
    def branch_namespace(self, name: str) -> str | None:
        """Return a branch's immutable presentation namespace, if bound."""
        self._require_namespace_owner()
        record = self._read_branch(_name(name))
        return record.get('namespace_id')

    @_history_transaction
    def active_presentation(self, name: str, snapshot_id: str | None = None) -> dict:
        """Resolve the published assistant ID for an active namespaced branch.

        A shared snapshot can be reached from multiple chats, so the lookup
        requires both this branch's immutable namespace and the exact snapshot.
        """
        return self._presentation_for_scope(name,snapshot_id,allowed_statuses=('active',),label='Active')

    @_history_transaction
    def published_presentation(self, name: str, snapshot_id: str | None = None) -> dict:
        """Resolve a bound assistant identity for an active or stopped scope."""
        return self._presentation_for_scope(name,snapshot_id,
            allowed_statuses=('active','stopped'),label='Published')

    def _presentation_for_scope(self, name, snapshot_id, *, allowed_statuses, label):
        self._require_namespace_owner()
        branch = self._read_branch(_name(name))
        namespace_id = branch.get('namespace_id')
        if namespace_id is None:
            raise NotFound('Branch has no presentation namespace')
        target = _id(snapshot_id) if snapshot_id is not None else branch['head']['tip']
        if self._root_for(target) != branch['head']['root']:
            raise InvalidOperation(f'{label} presentation snapshot belongs to another root')
        controller=self.get_snapshot(target)['controller']
        stream_key = controller.get('stream_key')
        if controller.get('status') not in allowed_statuses or not isinstance(stream_key, str):
            raise InvalidOperation(f'Branch tip does not contain a {label.lower()} generation scope')
        message_id = self._published_assistant_message_id(namespace_id, target, stream_key,
                                                          branch_name=name)
        if message_id is None:
            raise NotFound('No published assistant annotation for active stream')
        refs = self._read_refs()
        matches = []
        for operation, receipt in refs['receipts'].items():
            result = receipt.get('result')
            if not isinstance(result, dict) or result.get('tip') != target:
                continue
            if not isinstance(receipt.get('presentation_digest'), str):
                continue
            branch_name = result.get('name')
            if not isinstance(branch_name, str):
                continue
            other_branch = self._read_branch(branch_name)
            if other_branch.get('namespace_id') != namespace_id:
                continue
            descriptor = self._validate_stored_presentation(branch_name, other_branch,
                                                            operation, receipt, refs=refs)
            output = descriptor.get('output_span')
            if (descriptor.get('kind') in ('answer', 'initial_turn', 'user_turn', 'resume', 'reseed')
                    and isinstance(output, dict) and output.get('message_id') == message_id):
                matches.append((operation, descriptor, receipt))
        if not matches:
            raise IntegrityError('Active assistant ID has no matching durable receipt')
        operation, descriptor, receipt = sorted(matches, key=lambda item: item[0])[-1]
        return {'operation_id': operation, 'namespace_id': namespace_id,
                'snapshot': target, 'stream_key': stream_key, 'message_id': message_id,
                'digest': receipt['presentation_digest'],
                'presentation': copy.deepcopy(descriptor),
                'result': copy.deepcopy(receipt['result'])}

    @_history_transaction
    def presentation_catalog(self, namespace_id: str) -> dict:
        """Return detached heads and ordered presentation receipts for one chat.

        The owner-only query scans branch records once under the history lock,
        filters by immutable namespace identity, and reads each referenced
        descriptor blob at most once. It does not enumerate the global receipt
        index, write metadata, or load numerical checkpoints.
        """
        self._require_namespace_owner()
        namespace = self._read_namespace(namespace_id)
        namespace_value = {'namespace_id': namespace.namespace_id,
                           'title': namespace.title,
                           'operation_id': namespace.operation_id,
                           'created_at_ns': namespace.created_at_ns}
        heads = []
        events = []
        blob_cache: dict[str, dict] = {}
        for path in sorted(self.references.glob('*.json')):
            name = _name(path.stem)
            record = self._read_branch(name)
            self._count(reference_entries_scanned=1 + len(record['events']))
            if record.get('namespace_id') != namespace_id:
                continue
            head = record['head']
            if self._root_for(head['tip']) != head['root']:
                raise IntegrityError('Namespace branch head root differs from canonical ancestry')
            heads.append({'name': name, 'revision': head['revision'],
                          'tip': head['tip'], 'root': head['root']})
            for ordinal, event in zip(record['orders'], record['events']):
                operation = event['operation_id']
                receipt = record['receipts'][operation]
                digest = receipt.get('presentation_digest')
                presentation = None
                if digest is not None:
                    presentation = self._validate_stored_presentation(
                        name, record, operation, receipt, blob_cache=blob_cache)
                events.append({
                    'ordinal': ordinal,
                    'branch': name,
                    'type': event['type'],
                    'operation_id': operation,
                    'event': copy.deepcopy(event),
                    'receipt': copy.deepcopy(receipt),
                    'digest': digest,
                    'presentation': copy.deepcopy(presentation),
                    'source': copy.deepcopy(presentation.get('source')) if presentation else None,
                    'result': copy.deepcopy(receipt['result']),
                })
        events.sort(key=lambda item: (item['ordinal'], item['branch'], item['operation_id']))
        ordinals = [item['ordinal'] for item in events]
        if len(ordinals) != len(set(ordinals)):
            raise IntegrityError('Duplicate namespace presentation audit ordinal')
        return {'namespace': namespace_value, 'heads': heads, 'events': events}

    @_history_transaction
    def root_presentations(self, namespace_id: str, root_snapshot: str) -> list[dict]:
        """Return verified root annotations targeting one canonical root.

        This is the narrow capability lookup used before rendering a request at
        an idle root. It scans branch metadata under the same history lock as
        ``presentation_catalog``, but filters receipts by their committed
        result tip before opening descriptor blobs or reading source snapshots.
        The full catalog remains the API for callers that need every event.
        """
        self._require_namespace_owner()
        namespace = self._read_namespace(namespace_id)
        root_id = _id(root_snapshot)
        root_record = self.get_snapshot(root_id)
        if root_record.get('parent') is not None:
            raise InvalidOperation('Root presentation query requires a canonical root snapshot')

        presentations = []
        blob_cache: dict[str, dict] = {}
        for path in sorted(self.references.glob('*.json')):
            name = _name(path.stem)
            branch = self._read_branch(name)
            self._count(reference_entries_scanned=1 + len(branch['events']))
            if branch.get('namespace_id') != namespace.namespace_id:
                continue
            for event in branch['events']:
                operation = event['operation_id']
                receipt = branch['receipts'][operation]
                result = receipt.get('result')
                if not isinstance(result, dict) or result.get('tip') != root_id:
                    continue
                digest = receipt.get('presentation_digest')
                if digest is None:
                    continue
                if not isinstance(digest, str):
                    # Preserve the catalog's fail-closed behavior for malformed
                    # receipt metadata before attempting any path lookup.
                    self._validate_stored_presentation(name, branch, operation, receipt,
                                                       blob_cache=blob_cache)
                    continue
                if digest not in blob_cache:
                    # Every receipt at the requested target is hash-checked,
                    # but only root descriptors need their source snapshot and
                    # controller fully validated. Rewind receipts may point
                    # back to a large future source while their result is root.
                    blob_cache[digest] = self._read_presentation_blob(digest)
                if blob_cache[digest].get('kind') != 'root':
                    continue
                descriptor = self._validate_stored_presentation(
                    name, branch, operation, receipt, blob_cache=blob_cache)
                presentations.append(copy.deepcopy(descriptor))
        return presentations

    @_history_transaction
    def fork_operation_status(self, operation_id: str) -> dict:
        """Read a committed namespace-fork receipt,including after process exit.

        The per-branch record is the authority. Never infer the old result from
        a branch's current tip,create another status journal,or perform model
        work. Source-less initial-root creation is not an external fork request.
        """
        self._require_namespace_owner()
        operation = _name(operation_id)
        refs = self._read_refs()
        receipt = refs['receipts'].get(operation)
        if receipt is None:
            raise NotFound('Unknown fork operation: ' + operation)
        matches = [event for event in refs['events']
                   if event.get('operation_id') == operation and event.get('type') == 'fork'
                   and isinstance(event.get('source'), dict)]
        if not matches:
            raise NotFound('Operation is not a namespace fork: ' + operation)
        if len(matches) != 1:
            raise IntegrityError('Duplicate committed fork operation event')
        event = matches[0]
        result = receipt.get('result')
        if (not isinstance(result, dict) or result.get('name') != event.get('branch')
                or result.get('revision') != 0 or result.get('tip') != event.get('tip')):
            raise IntegrityError('Fork operation result differs from its event')
        # _read_refs/_read_branch already validate the request digest,source
        # provenance,namespace and original root,not just these returned fields.
        self.authorize(result['tip'], 'read')
        durable = dict(operation_id=operation, branch=copy.deepcopy(result),
                       snapshot=result['tip'], source=copy.deepcopy(event['source']))
        return dict(operation_id=operation, kind='fork', state='durable',
                    scope='canonical-reference', durable=True, receipt=durable)

    @_history_transaction
    def presentation_receipt(self, operation_id: str) -> dict:
        """Read a committed, hash-checked presentation annotation receipt."""
        self._require_namespace_owner()
        operation = _name(operation_id)
        refs = self._read_refs()
        receipt = refs['receipts'].get(operation)
        if receipt is None:
            raise NotFound(f'Unknown presentation operation: {operation}')
        name = receipt.get('result', {}).get('name')
        if not isinstance(name, str):
            raise IntegrityError('Presentation operation has no branch result')
        branch = self._read_branch(name)
        annotation = self._validate_stored_presentation(name, branch, operation, receipt,
                                                        refs=refs)
        result={'operation_id': operation, 'digest': receipt['presentation_digest'],
                'presentation': annotation, 'result': copy.deepcopy(receipt['result'])}
        if 'presentation_source_operation' in receipt:
            result['presentation_source_operation']=receipt['presentation_source_operation']
        return result

    @_history_transaction
    def checkpoint_receipt(self, operation_id: str) -> dict:
        """Read one owner-authorized operation's checkpoint metadata only."""
        self._require_namespace_owner()
        operation = _name(operation_id)
        refs = self._read_refs()
        receipt = refs['receipts'].get(operation)
        if receipt is None:
            raise NotFound(f'Unknown checkpoint operation: {operation}')
        descriptor = receipt.get('checkpoint_receipt')
        digest = receipt.get('checkpoint_receipt_digest')
        if descriptor is None or digest is None:
            raise NotFound(f'Operation has no checkpoint receipt: {operation}')
        result = receipt.get('result')
        if not isinstance(result, dict) or not isinstance(result.get('name'), str):
            raise IntegrityError('Checkpoint operation has no branch result')
        self._read_branch(result['name'])
        return {'operation_id': operation, 'digest': digest,
                'checkpoint_receipt': copy.deepcopy(descriptor),
                'result': copy.deepcopy(result)}

    def move(self, name: str, expected_revision: int,
             target_snapshot_id: str, operation_id: str, *, presentation: dict | None = None,
             checkpoint_receipt: dict | None = None,
             presentation_source_operation: str | None = None,
             adoption_presentation_source_operation: str | None = None,
             _evaluation_operation: bool = False) -> BranchHead:
        name = _name(name)
        op_id = _name(operation_id)
        identity = _id(target_snapshot_id)
        if presentation_source_operation is not None:
            presentation_source_operation = _name(presentation_source_operation)
            if presentation is None or presentation_source_operation == op_id:
                raise InvalidOperation('A retry presentation requires a distinct committed source operation')
        if adoption_presentation_source_operation is not None:
            adoption_presentation_source_operation = _name(adoption_presentation_source_operation)
            if not _evaluation_operation or presentation_source_operation is not None:
                raise InvalidOperation('Adoption presentation provenance requires an adoption move')
        if type(expected_revision) is not int or expected_revision < 0:
            raise InvalidOperation("Expected revision must be a nonnegative integer")
        request = {"type": "move", "name": name, "expected_revision": expected_revision,
                   "target": identity}
        checkpoint, checkpoint_digest = None, None
        if checkpoint_receipt is not None:
            checkpoint, checkpoint_digest = self._validate_checkpoint_receipt(
                checkpoint_receipt, identity, validate_model=False)
            request['checkpoint_receipt_digest'] = checkpoint_digest
        if presentation_source_operation is not None:
            request['presentation_source_operation'] = presentation_source_operation
        if adoption_presentation_source_operation is not None:
            request['adoption_presentation_source_operation'] = adoption_presentation_source_operation
        binding = binding_raw = binding_digest = None
        if presentation is not None:
            binding, binding_raw, binding_digest = self._presentation_digest(presentation)
            request['presentation_digest'] = binding_digest
        request_digest = _hash(_json_bytes(request))
        with self._locked():
            if _evaluation_operation:
                evaluation_operation = self._read_evaluation_operation(op_id)
                if (evaluation_operation is None or evaluation_operation.get('kind') != 'adoption'
                        or evaluation_operation.get('state') not in ('pending', 'committed')):
                    raise InvalidOperation('No matching adoption operation authorizes this move')
                if (evaluation_operation.get('branch') != name
                        or evaluation_operation.get('expected_revision') != expected_revision
                        or evaluation_operation.get('selected_snapshot_id') != identity
                        or evaluation_operation.get('adoption_presentation_source_operation')
                            != adoption_presentation_source_operation):
                    raise InvalidOperation('Move differs from its bound adoption operation')
            else:
                self._assert_operation_id_available(op_id, allow_evaluation_kind='branch')
            refs = self._read_refs()
            self.authorize(identity, 'move')
            if name in refs['branches']:
                self.authorize(refs['branches'][name]['tip'], 'move')
            receipt = refs["receipts"].get(op_id)
            if receipt is not None:
                if receipt["digest"] != request_digest:
                    raise InvalidOperation("Operation ID reused for different request")
                if binding_digest is not None:
                    self.presentation_receipt(op_id)
                record = receipt["result"]
                _sync_dir(self.references)
                return BranchHead(record["name"], record["revision"],
                                  record["tip"], record["root"])
            if checkpoint is not None:
                canonical = self.get_snapshot(identity)
                if canonical.get('model_contract_id') != checkpoint['model_contract_id']:
                    raise InvalidOperation('Checkpoint receipt model differs from canonical snapshot')
            if name not in refs["branches"]:
                raise NotFound(f"Unknown branch: {name}")
            # Validate the ancestry only on a new operation. Retries must be
            # answered from their committed receipt even after later rewinds.
            root = self._validated_root(identity)
            entry = refs["branches"][name]
            if entry["root"] != root:
                raise InvalidOperation("Cannot move branch between unrelated snapshot roots")
            if entry["revision"] != expected_revision:
                raise Conflict(f"CAS revision mismatch: expected {expected_revision}, got {entry['revision']}")
            old_tip = entry["tip"]
            branch = self._read_branch(name)
            namespace_id = branch.get('namespace_id')
            if namespace_id is not None:
                self._require_namespace_owner()
            if adoption_presentation_source_operation is not None:
                if namespace_id is None:
                    raise InvalidOperation('Adoption presentation provenance requires a namespaced branch')
                self._resolve_adoption_presentation_source(
                    name, identity, adoption_presentation_source_operation, refs=refs)
            if binding is not None:
                if namespace_id is None:
                    raise InvalidOperation('Presentation move requires a namespace-bound branch')
                source = BranchHead(name, expected_revision, old_tip, entry['root'])
                generation_operation = op_id
                if presentation_source_operation is not None:
                    original = self.presentation_receipt(presentation_source_operation)
                    original_binding = original.get('presentation')
                    original_result = original.get('result')
                    if (not isinstance(original_binding, dict)
                            or not isinstance(original_result, dict)
                            or original_binding.get('namespace_id') != namespace_id
                            or original_result.get('root') != entry['root']):
                        raise InvalidOperation('Retry source operation is outside this branch namespace/root')
                    original_source = original_binding.get('source')
                    if not isinstance(original_source, dict) or not isinstance(original_source.get('snapshot'), str):
                        raise InvalidOperation('Retry source operation has no canonical source snapshot')
                    source = BranchHead(name, original_source['revision'],
                        original_source['snapshot'], entry['root'])
                    generation_operation = self._presentation_stream_operation(
                        presentation_source_operation)
                    if binding.get('source') != original_source:
                        raise InvalidOperation('Retry presentation does not use the original pre-answer source')
                binding = self._validate_presentation(source, identity, binding,
                    namespace_id, op_id, generation_operation_id=generation_operation)
                binding, binding_raw, binding_digest = self._presentation_digest(binding)
                self._check_resume_message_id(binding, namespace_id, source)
                # Validation may normalize a descriptor only by detaching it;
                # the request identity must still cover exactly these canonical bytes.
                request['presentation_digest'] = binding_digest
                request_digest = _hash(_json_bytes(request))
                if receipt is not None and receipt['digest'] != request_digest:
                    raise InvalidOperation('Operation ID reused for different request')
                self._write_presentation_blob(binding_raw, binding_digest)
            result = BranchHead(name, expected_revision + 1, identity, root)
            branch['head'] = {"revision": result.revision, "tip": identity, "root": root}
            new_receipt = {"digest": request_digest,
                           "result": {"name": name, "revision": result.revision,
                                      "tip": identity, "root": root}}
            event = {"type": "move", "branch": name, "operation_id": op_id,
                     "from": old_tip, "to": identity,
                     "from_revision": expected_revision, "to_revision": result.revision}
            if namespace_id is not None:
                event['namespace_id'] = namespace_id
            if binding_digest is not None:
                new_receipt['presentation_digest'] = binding_digest
                event['presentation_digest'] = binding_digest
            if presentation_source_operation is not None:
                new_receipt['presentation_source_operation'] = presentation_source_operation
                event['presentation_source_operation'] = presentation_source_operation
            if adoption_presentation_source_operation is not None:
                new_receipt['adoption_presentation_source_operation'] = adoption_presentation_source_operation
                event['adoption_presentation_source_operation'] = adoption_presentation_source_operation
            if checkpoint_digest is not None:
                new_receipt['checkpoint_receipt'] = copy.deepcopy(checkpoint)
                new_receipt['checkpoint_receipt_digest'] = checkpoint_digest
                event['checkpoint_receipt_digest'] = checkpoint_digest
            branch["receipts"][op_id] = new_receipt
            branch["events"].append(event)
            self._fault("before_refs_stage")
            branch['orders'].append(self._reserve_order())
            self._publish_reference_json(self.references / (name + '.json'), branch, branch=True)
            return result

    def events(self) -> list[dict]:
        if self._principal is not None:
            raise PermissionError('Global audit requires local owner access')
        with self._locked():
            return copy.deepcopy(self._read_refs()["events"])
