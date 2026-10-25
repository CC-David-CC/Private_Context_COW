"""Opt-in owner-side prospective working-storage reservations; never dispatches work.

A policy is qualified outside this primitive by an enforced allocation bound or
an audited hard bound. An observed completed-file size is not qualification.
See PHASE-RESERVATIONS.md for the cooperating-writer contract and limitations.
"""
from __future__ import annotations

import contextlib
from dataclasses import asdict, dataclass
import fcntl
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import tempfile


class ReservationError(RuntimeError):
    pass


class CapacityUnavailable(ReservationError):
    pass


class WriterBusy(ReservationError):
    pass


class StaleRevision(ReservationError):
    pass


def _integer(value, name):
    if type(value) is not int or not 0 <= value < 2**63:
        raise ValueError(name + ' must be a nonnegative bounded integer')
    return value


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def _sha(value):
    return type(value) is str and re.fullmatch('[0-9a-f]{64}', value) is not None


@dataclass(frozen=True)
class PhasePolicy:
    name: str
    maximum_additional_bytes: int
    qualification: str = 'UNQUALIFIED'
    evidence_sha256: str | None = None

    def validate(self):
        if type(self.name) is not str or not 1 <= len(self.name) <= 128:
            raise ValueError('Explicit owner phase name required')
        _integer(self.maximum_additional_bytes, 'maximum_additional_bytes')
        if self.qualification not in ('UNQUALIFIED', 'QUALIFIED'):
            raise ValueError('Unknown phase qualification')
        if self.qualification == 'QUALIFIED' and not _sha(self.evidence_sha256):
            raise ValueError('Qualified owner policy requires pinned bound evidence')
        if self.evidence_sha256 is not None and not _sha(self.evidence_sha256):
            raise ValueError('Invalid bound evidence digest')


@dataclass(frozen=True)
class Artifact:
    relative_path: str
    bytes: int
    sha256: str

    def validate(self):
        path = PurePosixPath(self.relative_path)
        if (type(self.relative_path) is not str or not self.relative_path
                or path.is_absolute() or any(x in ('.', '..') for x in path.parts)
                or str(path) != self.relative_path):
            raise ValueError('Canonical relative artifact path required')
        _integer(self.bytes, 'artifact bytes')
        if not _sha(self.sha256):
            raise ValueError('Exact expected artifact SHA256 required')


def _owned_directory(path):
    path = Path(path).absolute()
    if path.is_symlink() or path.resolve() != path:
        raise ValueError('Directory links are refused')
    info = path.stat()
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
        raise ValueError('Private owned directory required')
    return path


def _open_private(path, flags):
    fd = os.open(path, flags | os.O_NOFOLLOW | os.O_CLOEXEC)
    info = os.fstat(fd)
    if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid()
            or info.st_mode & 0o077 or info.st_nlink != 1):
        os.close(fd)
        raise ValueError('Private owned regular lock/record required')
    return fd


class PhaseReservations:
    """Cross-process phase claims for one owner-controlled working-storage pool.

    This is not GPU-time accounting, a scheduler, a native launcher, or a quota.
    Metadata must live on a different filesystem from the guarded pool.
    """
    MAX_LEDGER_BYTES = 4 << 20

    @classmethod
    def create(cls, metadata, pool, *, floor_bytes, policies):
        metadata = Path(metadata).absolute()
        pool = _owned_directory(pool)
        _integer(floor_bytes, 'floor_bytes')
        policies = tuple(policies)
        policy_map = cls._policies(policies)
        metadata.mkdir(mode=0o700, exist_ok=False)
        metadata = _owned_directory(metadata)
        if metadata.stat().st_dev == pool.stat().st_dev:
            raise ValueError('Metadata must be on a separate filesystem from the pool')
        (metadata/'leases').mkdir(mode=0o700)
        fd = os.open(metadata/'ledger.lock', os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        os.close(fd)
        obj = cls.__new__(cls)
        obj.metadata, obj.pool = metadata, pool
        obj._save(dict(schema=1, pool=str(pool), pool_device=pool.stat().st_dev,
                       pool_inode=pool.stat().st_ino, floor_bytes=floor_bytes,
                       policies=policy_map, active={}, completed={}))
        return cls(metadata, pool, floor_bytes=floor_bytes, policies=policies)

    @staticmethod
    def _policies(policies):
        result = {}
        for policy in policies:
            if not isinstance(policy, PhasePolicy):
                raise ValueError('Explicit owner PhasePolicy required')
            policy.validate()
            if policy.name in result:
                raise ValueError('Duplicate phase policy')
            result[policy.name] = asdict(policy)
        if not result:
            raise ValueError('Owner phase registry required')
        return result

    def __init__(self, metadata, pool, *, floor_bytes, policies):
        self.metadata, self.pool = _owned_directory(metadata), _owned_directory(pool)
        if self.metadata.stat().st_dev == self.pool.stat().st_dev:
            raise ValueError('Metadata must be on a separate filesystem from the pool')
        _integer(floor_bytes, 'floor_bytes')
        self._expected = dict(pool=str(self.pool), pool_device=self.pool.stat().st_dev,
                              pool_inode=self.pool.stat().st_ino, floor_bytes=floor_bytes,
                              policies=self._policies(policies))
        self.snapshot()

    def _save(self, data):
        raw = (json.dumps(data, sort_keys=True, allow_nan=False) + '\n').encode()
        if len(raw) > self.MAX_LEDGER_BYTES:
            raise ReservationError('Bounded ledger full; explicit archival required')
        fd, temporary = tempfile.mkstemp(prefix='.ledger-', suffix='.pending', dir=self.metadata)
        with os.fdopen(fd, 'wb') as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, self.metadata/'ledger.json')
        directory = os.open(self.metadata, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)

    @contextlib.contextmanager
    def _locked(self):
        # Writable descriptors are required for NFS's emulated exclusive flock.
        fd = _open_private(self.metadata/'ledger.lock', os.O_RDWR)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            yield
        finally:
            os.close(fd)

    def _read(self):
        fd = _open_private(self.metadata/'ledger.json', os.O_RDONLY)
        with os.fdopen(fd, 'rb') as stream:
            raw = stream.read(self.MAX_LEDGER_BYTES + 1)
        if len(raw) > self.MAX_LEDGER_BYTES:
            raise ReservationError('Ledger exceeds declared metadata bound')
        data = json.loads(raw)
        if data.get('schema') != 1 or any(data.get(k) != v for k, v in self._expected.items()):
            raise ReservationError('Pinned pool, floor or owner registry differs')
        info = self.pool.stat()
        if info.st_dev != data['pool_device'] or info.st_ino != data['pool_inode']:
            raise ReservationError('Working-storage pool identity changed')
        if type(data.get('active')) is not dict or type(data.get('completed')) is not dict:
            raise ReservationError('Malformed original operation registry')
        if set(data['active']) & set(data['completed']):
            raise ReservationError('Operation is both active and complete')
        try:
            for completed, records in ((False, data['active']), (True, data['completed'])):
                for operation, record in records.items():
                    self._validate_record(data, operation, record, completed)
        except (KeyError, TypeError, ValueError) as error:
            raise ReservationError('Malformed original operation provenance') from error
        return data

    def _validate_record(self, data, operation, record, completed):
        self._operation(operation)
        record_keys = {'operation', 'phase', 'revision', 'remaining_bytes',
                       'admitted_free_bytes', 'events', 'admission', 'admission_sha256'}
        if type(record) is not dict or set(record) != record_keys or record['operation'] != operation:
            raise ReservationError('Original operation binding differs')
        revision = _integer(record['revision'], 'revision')
        if type(record['events']) is not list or revision != len(record['events']):
            raise ReservationError('Original revision/event count differs')
        admission = record['admission']
        admission_keys = {'operation', 'phase', 'remaining_bytes', 'admitted_free_bytes',
                          'total_claim_bytes', 'floor_bytes', 'policy_registry_sha256',
                          'pool_device', 'pool_inode'}
        if (type(admission) is not dict or set(admission) != admission_keys
                or record['admission_sha256'] != _digest(admission)
                or admission['operation'] != operation
                or admission['floor_bytes'] != data['floor_bytes']
                or admission['policy_registry_sha256'] != _digest(data['policies'])
                or admission['pool_device'] != data['pool_device']
                or admission['pool_inode'] != data['pool_inode']):
            raise ReservationError('Pinned original admission differs')
        phase = admission['phase']
        remaining = self._qualified(data, phase)
        if (type(admission['remaining_bytes']) is not int or admission['remaining_bytes'] != remaining
                or record['admitted_free_bytes'] != admission['admitted_free_bytes']):
            raise ReservationError('Original admitted bound differs')
        total = _integer(admission['total_claim_bytes'], 'admitted total claims')
        free = _integer(admission['admitted_free_bytes'], 'admitted free')
        if total < remaining or free < data['floor_bytes'] + total:
            raise ReservationError('Original admission violates its capacity invariant')
        for number, event in enumerate(record['events']):
            if phase is None:
                raise ReservationError('Transition follows an already terminal phase')
            keys = {'from_revision', 'next_phase', 'artifacts', 'intent_sha256', 'operation',
                    'from_phase', 'previous_bytes', 'remaining_bytes', 'revision',
                    'observed_free_bytes', 'materialized_artifacts', 'prospective_total_bytes',
                    'event_sha256'}
            if type(event) is not dict or set(event) != keys:
                raise ReservationError('Malformed original transition')
            payload = {k: v for k, v in event.items() if k != 'event_sha256'}
            intent = {k: event[k] for k in ('from_revision', 'next_phase', 'artifacts')}
            if (event['event_sha256'] != _digest(payload) or event['intent_sha256'] != _digest(intent)
                    or event['operation'] != operation
                    or type(event['from_revision']) is not int or event['from_revision'] != number
                    or type(event['revision']) is not int or event['revision'] != number+1
                    or event['from_phase'] != phase or type(event['previous_bytes']) is not int
                    or event['previous_bytes'] != remaining):
                raise ReservationError('Original phase transition chain differs')
            artifacts = event['artifacts']
            if type(artifacts) is not list or not artifacts:
                raise ReservationError('Original materialization manifest is missing')
            paths = []
            for artifact in artifacts:
                if type(artifact) is not dict or set(artifact) != {'relative_path', 'bytes', 'sha256'}:
                    raise ReservationError('Malformed original materialization manifest')
                Artifact(**artifact).validate()
                paths.append(artifact['relative_path'])
            if paths != sorted(set(paths)) or event['materialized_artifacts'] != artifacts:
                raise ReservationError('Original materialization manifest differs')
            phase = event['next_phase']
            remaining = 0 if phase is None else self._qualified(data, phase)
            total = _integer(event['prospective_total_bytes'], 'prospective total claims')
            free = _integer(event['observed_free_bytes'], 'observed free')
            if (type(event['remaining_bytes']) is not int or event['remaining_bytes'] != remaining
                    or total < remaining or free < data['floor_bytes'] + total):
                raise ReservationError('Original transition bound/capacity differs')
        if (record['phase'] != phase or type(record['remaining_bytes']) is not int
                or record['remaining_bytes'] != remaining or completed != (phase is None)):
            raise ReservationError('Current phase/claim differs from the original transition chain')

    def snapshot(self):
        with self._locked():
            return self._read()

    def _qualified(self, data, phase):
        policy = data['policies'].get(phase)
        if policy is None or policy['qualification'] != 'QUALIFIED':
            raise ReservationError('Phase has no qualified future-allocation bound')
        return policy['maximum_additional_bytes']

    def _capacity(self, data, prospective):
        # Called only under ledger.lock: callers cannot provide stale snapshots.
        usage = os.statvfs(self.pool)
        free = usage.f_bavail * usage.f_frsize
        if free < data['floor_bytes'] + prospective:
            raise CapacityUnavailable('Fresh pool capacity cannot retain the owner floor')
        return free

    @staticmethod
    def _operation(operation):
        if type(operation) is not str or not 1 <= len(operation) <= 256:
            raise ValueError('Explicit bounded original operation ID required')
        return _digest(operation)

    def _lease(self, operation, *, create=False):
        path = self.metadata/'leases'/(self._operation(operation) + '.lock')
        if create:
            try:
                fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
                os.close(fd)
            except FileExistsError:
                pass  # Stable inode, including a crash before original admission.
        return _open_private(path, os.O_RDWR)

    def reserve(self, operation, phase):
        self._operation(operation)
        with self._locked():
            data = self._read()
            if operation in data['active'] or operation in data['completed']:
                raise ReservationError('Original operation already exists; reconcile, never redispatch')
            maximum = self._qualified(data, phase)
            total = maximum + sum(x['remaining_bytes'] for x in data['active'].values())
            free = self._capacity(data, total)
            os.close(self._lease(operation, create=True))
            admission = dict(operation=operation, phase=phase, remaining_bytes=maximum,
                             admitted_free_bytes=free, total_claim_bytes=total,
                             floor_bytes=data['floor_bytes'], policy_registry_sha256=_digest(data['policies']),
                             pool_device=data['pool_device'], pool_inode=data['pool_inode'])
            record = dict(operation=operation, phase=phase, revision=0,
                          remaining_bytes=maximum, admitted_free_bytes=free, events=[],
                          admission=admission, admission_sha256=_digest(admission))
            data['active'][operation] = record
            self._save(data)
            return record

    @contextlib.contextmanager
    def writer_lease(self, operation, *, revision):
        _integer(revision, 'revision')
        fd = self._lease(operation)
        try:
            try:
                fcntl.flock(fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise WriterBusy('Original phase is being sealed') from error
            with self._locked():
                record = self._read()['active'].get(operation)
                if record is None or record['revision'] != revision:
                    raise StaleRevision('Writer requires its exact admitted phase')
            # Pass this FD to every writing subprocess. Close, do not LOCK_UN:
            # inherited child descriptors must retain protection after owner death.
            yield fd
        finally:
            os.close(fd)

    def _materialized(self, artifacts):
        if not artifacts:
            raise ReservationError('Exact nonempty completed-artifact manifest required')
        result = []
        for artifact in artifacts:
            if not isinstance(artifact, Artifact):
                raise ValueError('Explicit Artifact required')
            artifact.validate()
            if artifact.relative_path in {x['relative_path'] for x in result}:
                raise ValueError('Duplicate artifact')
            parent = os.open(self.pool, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
            try:
                parts = PurePosixPath(artifact.relative_path).parts
                for part in parts[:-1]:
                    next_fd = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent)
                    os.close(parent)
                    parent = next_fd
                fd = os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW, dir_fd=parent)
            finally:
                os.close(parent)
            with os.fdopen(fd, 'rb') as stream:
                before = os.fstat(stream.fileno())
                if (not stat.S_ISREG(before.st_mode) or before.st_uid != os.getuid()
                        or before.st_mode & 0o277 or before.st_nlink != 1
                        or before.st_dev != self.pool.stat().st_dev or before.st_size != artifact.bytes):
                    raise ReservationError('Completed artifact must be exact, sealed, private and owned')
                actual = hashlib.file_digest(stream, 'sha256').hexdigest()
                os.fsync(stream.fileno())
                after = os.fstat(stream.fileno())
                fields = ('st_dev', 'st_ino', 'st_size', 'st_mtime_ns', 'st_ctime_ns', 'st_mode', 'st_nlink')
                if actual != artifact.sha256 or any(getattr(before, k) != getattr(after, k) for k in fields):
                    raise ReservationError('Completed artifact bytes/identity differ')
            result.append(asdict(artifact))
        # Files must also have durably named directory entries before a shrink.
        directories = {self.pool}
        for artifact in artifacts:
            directory = (self.pool/artifact.relative_path).parent
            while directory != self.pool:
                directories.add(directory)
                directory = directory.parent
        for directory in directories:
            fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
        return sorted(result, key=lambda x: x['relative_path'])

    def transition(self, operation, *, revision, next_phase, artifacts):
        return self._change(operation, revision=revision, next_phase=next_phase, artifacts=artifacts)

    def finish(self, operation, *, revision, artifacts):
        return self._change(operation, revision=revision, next_phase=None, artifacts=artifacts)

    def _change(self, operation, *, revision, next_phase, artifacts):
        _integer(revision, 'revision')
        artifacts = tuple(artifacts)
        requested = []
        for artifact in artifacts:
            if not isinstance(artifact, Artifact):
                raise ValueError('Explicit Artifact required')
            artifact.validate()
            requested.append(asdict(artifact))
        requested.sort(key=lambda x: x['relative_path'])
        intent = dict(from_revision=revision, next_phase=next_phase, artifacts=requested)
        fd = self._lease(operation)
        try:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise WriterBusy('A live inherited writer retains the original claim') from error
            with self._locked():
                data = self._read()
                record = data['active'].get(operation) or data['completed'].get(operation)
                if record is None:
                    raise StaleRevision('Unknown original operation')
                previous = next((x for x in record['events'] if x['from_revision'] == revision), None)
                if previous is not None:
                    if previous['intent_sha256'] != _digest(intent):
                        raise StaleRevision('Original transition intent differs')
                    return previous
                if operation not in data['active'] or record['revision'] != revision:
                    raise StaleRevision('Exact current original phase required')
                if next_phase is not None:
                    self._qualified(data, next_phase)
            materialized = self._materialized(artifacts)
            with self._locked():
                data = self._read()
                record = data['active'][operation]
                if record['revision'] != revision:
                    raise StaleRevision('Original phase changed during sealing')
                maximum = 0 if next_phase is None else self._qualified(data, next_phase)
                prospective = maximum + sum(x['remaining_bytes'] for k, x in data['active'].items() if k != operation)
                free = self._capacity(data, prospective)
                event = dict(intent, intent_sha256=_digest(intent), operation=operation,
                             from_phase=record['phase'], previous_bytes=record['remaining_bytes'],
                             remaining_bytes=maximum, revision=revision+1, observed_free_bytes=free,
                             materialized_artifacts=materialized, prospective_total_bytes=prospective)
                event['event_sha256'] = _digest(event)
                record['events'].append(event)
                record.update(phase=next_phase, revision=revision+1, remaining_bytes=maximum)
                if next_phase is None:
                    data['completed'][operation] = data['active'].pop(operation)
                self._save(data)
                return event
        finally:
            os.close(fd)
