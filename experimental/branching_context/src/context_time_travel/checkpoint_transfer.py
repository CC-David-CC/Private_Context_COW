"""Bounded transport of existing COW checkpoints, inspired by KVCR.

Only numerical materializations cross this boundary. Canonical histories,
controllers, branches and publication receipts remain the destination's authority.
No GPU lease or model operation occurs here.
"""
from concurrent.futures import CancelledError
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import math
import os
import stat
import struct
import threading
import time

from .pools import PoolFull
from .store import IntegrityError, _decode, _json_bytes, _sync_dir


@dataclass(frozen=True)
class OfferedCheckpoint:
    envelope: dict
    stream: object
    payload_bytes: int
    chunk_bytes: int
    check: object


class CheckpointTransfers:
    def __init__(self,pool,disk,*,chunk_bytes=1024**2,max_bytes=64*1024**3,
                 max_concurrent=2,timeout=300):
        if (type(chunk_bytes) is not int or not 4<=chunk_bytes<=1024**2 or chunk_bytes%4
                or type(max_bytes) is not int or max_bytes<=0
                or type(max_concurrent) is not int or not 1<=max_concurrent<=16
                or type(timeout) not in (int,float) or not math.isfinite(timeout) or timeout<=0):
            raise ValueError('Invalid checkpoint transfer limits')
        self.pool,self.disk=pool,disk
        self.chunk_bytes,self.max_bytes,self.timeout=chunk_bytes,max_bytes,timeout
        self._slots=threading.BoundedSemaphore(max_concurrent)
        self._lock=threading.Lock()
        self._active=self._completed=self._failed=self._received_bytes=0

    def metrics(self):
        with self._lock:
            return dict(active=self._active,completed=self._completed,failed=self._failed,
                        received_bytes=self._received_bytes,max_checkpoint_bytes=self.max_bytes)

    @contextmanager
    def _slot(self,cancelled):
        if not self._slots.acquire(blocking=False):raise PoolFull('Checkpoint transfer slots are full')
        deadline=time.monotonic()+self.timeout
        def check():
            if cancelled is not None and cancelled():raise CancelledError('Checkpoint transfer cancelled')
            if time.monotonic()>deadline:raise TimeoutError('Checkpoint transfer deadline exceeded')
        with self._lock:self._active+=1
        try:
            check();yield check
        except BaseException:
            with self._lock:self._failed+=1
            raise
        else:
            with self._lock:self._completed+=1
        finally:
            with self._lock:self._active-=1
            self._slots.release()

    def _metadata(self,model,snapshot,envelope):
        raw=_json_bytes(envelope)
        if len(raw)>4*1024**2:raise IntegrityError('Checkpoint manifest exceeds transfer limit')
        # Detach caller-owned objects before any blocking admission or I/O.
        envelope=_decode(raw)
        manifest=model.validate_checkpoint_envelope(snapshot,envelope)
        size=manifest['state_bytes']+4*manifest['logits_count']
        if size>self.max_bytes:raise PoolFull('Checkpoint exceeds transfer size limit')
        return envelope,raw,manifest,size

    @staticmethod
    def _stamp(stream):
        value=os.fstat(stream.fileno())
        if not stat.S_ISREG(value.st_mode):raise IntegrityError('Checkpoint payload is not a regular file')
        return value.st_dev,value.st_ino,value.st_size,value.st_mtime_ns,value.st_ctime_ns

    def _validate_stream(self,stream,manifest,check):
        size=manifest['state_bytes']+4*manifest['logits_count']
        digest=hashlib.sha256();remaining=size
        stream.seek(0)
        while remaining:
            check();chunk=stream.read(min(self.chunk_bytes,remaining))
            if not chunk:raise IntegrityError('Incomplete checkpoint payload')
            digest.update(chunk);remaining-=len(chunk)
        if stream.read(1) or digest.hexdigest()!=manifest['payload_sha256']:
            raise IntegrityError('Checkpoint payload corruption')
        stream.seek(manifest['state_bytes']);remaining=4*manifest['logits_count']
        while remaining:
            check();chunk=stream.read(min(self.chunk_bytes,remaining))
            if len(chunk)!=min(self.chunk_bytes,remaining):raise IntegrityError('Incomplete checkpoint frontier')
            if any(not math.isfinite(v[0]) for v in struct.iter_unpack('<f',chunk)):
                raise IntegrityError('Nonfinite checkpoint frontier')
            remaining-=len(chunk)
        stream.seek(0)

    @contextmanager
    def _file(self,path,manifest,check):
        try:
            descriptor=os.open(path,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK)
            with os.fdopen(descriptor,'rb') as stream:
                before=self._stamp(stream)
                if before[2]!=manifest['state_bytes']+4*manifest['logits_count']:
                    raise IntegrityError('Checkpoint payload size mismatch')
                self._validate_stream(stream,manifest,check)
                if self._stamp(stream)!=before:raise IntegrityError('Checkpoint changed during validation')
                yield stream
                if self._stamp(stream)!=before:raise IntegrityError('Checkpoint changed during transfer')
        except OSError as error:
            raise IntegrityError('Checkpoint payload is unavailable') from error

    @contextmanager
    def export(self,model,snapshot,model_contract_id,execution_profile_id,*,cancelled=None):
        if (model.profile['model_contract_id']!=model_contract_id
                or model.profile['execution_profile_id']!=execution_profile_id):
            raise IntegrityError('Peer checkpoint identity differs')
        with self._slot(cancelled) as check,model.store.read_lease(snapshot):
            manifest=model.checkpoint_manifest(snapshot)
            envelope=dict(manifest=manifest,checksum=hashlib.sha256(_json_bytes(manifest)).hexdigest())
            envelope,raw,manifest,size=self._metadata(model,snapshot,envelope)
            with self.pool.reserve(4*self.chunk_bytes+len(raw)),self.disk.reserve(
                    0,payload_sha256=manifest['payload_sha256']):
                path=model.checkpoint_directory/(manifest['payload_sha256']+'.state')
                with self._file(path,manifest,check) as stream:
                    yield OfferedCheckpoint(envelope,stream,size,self.chunk_bytes,check)

    def receive(self,model,snapshot,envelope,stream,*,cancelled=None):
        """Admit a checked derived cache only for history already present here.

        Retry is idempotent. A present incompatible/corrupt cache is an error,
        never an instruction to overwrite it. Failed staging owns no branch move.
        """
        published_payload=False
        try:
            with self._slot(cancelled) as check,model.store.read_lease(snapshot):
                envelope,raw,manifest,size=self._metadata(model,snapshot,envelope)
                digest=manifest['payload_sha256']
                payload=model.checkpoint_directory/(digest+'.state')
                target=model.checkpoint_directory/(snapshot+'.json')
                with self.pool.reserve(4*self.chunk_bytes+len(raw)),self.disk.reserve(
                        0,payload_sha256=digest) as reservation:
                    # Existing complete imports require no payload transfer.
                    if os.path.lexists(target):
                        if target.is_symlink() or target.read_bytes()!=raw:
                            raise IntegrityError('Existing checkpoint manifest differs')
                        with self._file(payload,manifest,check):pass
                        self.disk.touch(snapshot)
                        return dict(snapshot=snapshot,payload_sha256=digest,transferred_bytes=0)
                    reservation.grow(len(raw))
                    if os.path.lexists(payload):
                        with self._file(payload,manifest,check):pass
                        copied=0
                    else:
                        reservation.grow(size)
                        with model._checkpoint_stage() as (output,stage):
                            remaining=size;checksum=hashlib.sha256()
                            while remaining:
                                check();requested=min(self.chunk_bytes,remaining)
                                chunk=stream.read(requested)
                                if type(chunk) is not bytes or not 0<len(chunk)<=requested:
                                    raise IntegrityError('Incomplete or oversized checkpoint stream')
                                output.write(chunk);checksum.update(chunk);remaining-=len(chunk)
                            check()
                            if stream.read(1) or checksum.hexdigest()!=digest:
                                raise IntegrityError('Checkpoint transfer checksum or extent differs')
                            output.flush();os.fsync(output.fileno())
                            with self._file(stage,manifest,check):pass
                            # A hard link is an atomic, no-overwrite publication.
                            with model.store._locked():
                                try:os.link(stage,payload);published_payload=True
                                except FileExistsError:
                                    with self._file(payload,manifest,check):pass
                                stage.unlink();_sync_dir(model.checkpoint_directory)
                        copied=size
                    check()
                    with model._checkpoint_stage() as (output,stage):
                        output.write(raw);output.flush();os.fsync(output.fileno())
                        with model.store._locked():
                            try:os.link(stage,target)
                            except FileExistsError:
                                if target.is_symlink() or target.read_bytes()!=raw:
                                    raise IntegrityError('Concurrent checkpoint manifest differs')
                            stage.unlink();_sync_dir(model.checkpoint_directory)
                    self.disk.touch(snapshot)
                    with self._lock:self._received_bytes+=copied
                    return dict(snapshot=snapshot,payload_sha256=digest,transferred_bytes=copied)
        except BaseException:
            # Once reservations are released, the existing collector reclaims
            # only unreferenced derived payloads. Canonical files are untouched.
            if published_payload:self.disk.collect()
            raise
