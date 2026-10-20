"""Shared, bounded derived checkpoint/embedding bytes. Never owns history.

The byte limit includes admitted payload/frontier extents and reservations.
Python object overhead, model weights and native arenas are reported separately
by callers. No claim of physical prefix deduplication between different blobs.
"""
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, Future
from contextlib import contextmanager
from dataclasses import dataclass, replace
import json
import errno
import math
import os
from pathlib import Path
import fcntl
import stat
import threading
import time
import uuid
from .digest_attestation import verified_checkpoint_digest
from .vision_embeddings import Embedding, parse_sve


def _validate_checkpoint_idle_seconds(value):
    if value is None:
        return None
    if type(value) is int:
        if value > 0:
            return value
    elif type(value) is float and math.isfinite(value) and value > 0:
        return value
    raise ValueError('checkpoint_idle_seconds must be None or a positive finite number')


def _idle_age_exceeded(last_ns, now_ns, idle_seconds):
    """Use exact integer arithmetic for the elapsed-nanoseconds cutoff."""
    if now_ns < last_ns:
        return False
    elapsed = now_ns - last_ns
    if type(idle_seconds) is int:
        return elapsed > idle_seconds * 1_000_000_000
    numerator, denominator = idle_seconds.as_integer_ratio()
    return elapsed * denominator > numerator * 1_000_000_000


def _monotonic_value(clock):
    value=clock()
    if type(value) is int:
        return value
    if type(value) is float and math.isfinite(value):
        return value
    raise ValueError('monotonic clock must return a finite integer or float')


def _wall_clock_ns(clock):
    value=clock()
    if type(value) is not int or value < 0:
        raise ValueError('clock_ns must return a nonnegative integer')
    return value


class PoolFull(MemoryError):
    pass


class PoolTransitionTrace:
    """Opt-in bounded metadata trace for within-process RAM/disk transitions.

    `freed_bytes` is unique-payload pool accounting, not a claim about process
    RSS or external/native references.
    """
    def __init__(self,phase,*,process_id=None,phase_id=None,max_events=8192):
        if not isinstance(phase,str) or not phase: raise ValueError('Trace phase is required')
        if type(max_events) is not int or max_events <= 0: raise ValueError('Invalid trace event bound')
        self.phase=phase
        self.process_id=os.getpid() if process_id is None else process_id
        self.phase_id=phase_id or uuid.uuid4().hex
        self.max_events=max_events
        self._lock=threading.Lock()
        self._events=[]
        self._event_count=0
        self._overflowed=False

    def record(self,event,key,**fields):
        # This is an internal bounded append, never a caller-provided callback.
        with self._lock:
            sequence=self._event_count
            self._event_count+=1
            if len(self._events)>=self.max_events:
                self._overflowed=True
                return sequence
            row=dict(sequence=sequence,event=event,key=list(key),phase=self.phase,
                process_id=self.process_id,phase_id=self.phase_id,
                monotonic_ns=time.monotonic_ns())
            if event=='disk_preload_started': row['load_id']=sequence
            row.update(fields)
            self._events.append(row)
            return sequence

    def snapshot(self):
        with self._lock:
            return dict(phase=self.phase,process_id=self.process_id,phase_id=self.phase_id,
                max_events=self.max_events,event_count=self._event_count,
                overflowed=self._overflowed,events=[dict(row,key=list(row['key'])) for row in self._events])


def normalize_gpu_selection(selection):
    """Outer entries are workers; inner entries are disjoint physical devices."""
    if not isinstance(selection,(list,tuple)) or not selection:
        raise ValueError('GPU must be a nonempty list of IDs or device groups')
    workers,physical = [],[]
    for item in selection:
        ids = (item,) if type(item) is int else tuple(item) if isinstance(item,(list,tuple)) else ()
        if not ids or any(type(n) is not int or n < 0 for n in ids):
            raise ValueError('GPU IDs must be nonnegative integers')
        workers.append(ids[0] if len(ids)==1 else ids)
        physical.extend(ids)
    if len(set(physical))!=len(physical): raise ValueError('GPU worker groups cannot overlap')
    return tuple(workers)


def worker_key(gpu):
    return gpu if type(gpu) is int else ','.join(map(str,gpu))


@dataclass(frozen=True)
class PoolConfig:
    gpus: tuple = (0,)
    ram_bytes: int = 8 * 1024**3
    disk_bytes: int = 64 * 1024**3
    min_available_bytes: int = 16 * 1024**3
    checkpoint_idle_seconds: int | float | None = None

    def __post_init__(self):
        ids = normalize_gpu_selection(self.gpus)
        object.__setattr__(self,'gpus',ids)
        for name in ('ram_bytes','disk_bytes','min_available_bytes'):
            n = getattr(self,name)
            if type(n) is not int or n < (0 if name == 'min_available_bytes' else 1):
                raise ValueError('Invalid pool byte budget: '+name)
        object.__setattr__(self,'checkpoint_idle_seconds',
            _validate_checkpoint_idle_seconds(self.checkpoint_idle_seconds))

    @property
    def worker_count(self):
        return len(self.gpus)

    @property
    def physical_gpus(self):
        return tuple(n for item in self.gpus for n in ((item,) if type(item) is int else item))


def checkpoint_bytes(checkpoint):
    if (type(checkpoint.blob) is not bytes or not checkpoint.blob or not checkpoint.frontier
            or not all(map(math.isfinite,checkpoint.frontier))):
        raise ValueError('Shared RAM entries require immutable serialized bytes')
    return len(checkpoint.blob) + 4 * len(checkpoint.frontier)


def checkpoint_digest(checkpoint):
    return verified_checkpoint_digest(checkpoint)


def _pool_identity(value):
    # Domains prevent a payload hash from impersonating a checkpoint checksum.
    return ('embedding',value.sha256) if type(value) is Embedding else ('checkpoint',value.checksum)


def _pool_value_bytes(value):
    if type(value) is Embedding:
        return value.nbytes
    return checkpoint_bytes(value)


def _validate_pool_value(value):
    if type(value) is Embedding:
        checked=parse_sve(value.payload,expected_sha256=value.sha256,
            expected_width=value.width,max_rows=value.rows)
        if checked!=value:
            from .store import IntegrityError
            raise IntegrityError('Embedding metadata does not describe its payload')
        return value.nbytes
    size=checkpoint_bytes(value)
    if checkpoint_digest(value)!=value.checksum:
        raise ValueError('Checkpoint checksum does not bind RAM bytes and frontier')
    return size


def _discardable(entry):
    # An embedding can be regenerated and checked against canonical identities.
    # The caller owns retained originals and must pin inputs during consumption.
    # This does not assert that an embedding has a durable disk copy.
    return (entry['durable'] or entry.get('failed',False)
            or type(entry['checkpoint']) is Embedding)


class _Reservation:
    def __init__(self,pool,size):
        self.pool,self.size,self.closed = pool,size,False

    def __enter__(self): return self

    def __exit__(self,*_): self.release()

    def release(self):
        with self.pool._condition:
            if not self.closed:
                self.pool._reserved -= self.size
                self.closed = True
                self.pool._condition.notify_all()

    def admit(self,key,checkpoint,*,durable=False):
        return self.pool._admit(self,key,checkpoint,durable)


class SharedCheckpointPool:
    """One RAM budget for checkpoints and validated, evictable image embeddings.

    `durable` always describes the derived disk copy. Embeddings may be evicted
    without that copy; retained original inputs and exact re-encoding belong to
    the artifact owner. Dirty checkpoints still wait for publication or failure.
    """
    def __init__(self,byte_limit,*,transition_trace=None,checkpoint_idle_seconds=None,clock=None):
        if type(byte_limit) is not int or byte_limit <= 0: raise ValueError('Invalid RAM budget')
        if transition_trace is not None and type(transition_trace) is not PoolTransitionTrace:
            raise TypeError('transition_trace must be a PoolTransitionTrace')
        self.checkpoint_idle_seconds=_validate_checkpoint_idle_seconds(checkpoint_idle_seconds)
        self._idle_clock=time.monotonic if clock is None else clock
        if not callable(self._idle_clock): raise TypeError('clock must be callable')
        self.byte_limit = byte_limit
        self._transition_trace=transition_trace
        self._entries = OrderedDict()
        self._buffers = {}
        self._reserved = 0
        self._condition = threading.Condition(threading.RLock())
        self._reader = ThreadPoolExecutor(max_workers=1,thread_name_prefix='cow-read')
        self._loads = {}
        self._closed = False
        self._evictions = self._failed_evictions = self._idle_evictions = self._peak = self._waits = 0

    def _bytes(self): return sum(row['bytes'] for row in self._buffers.values())

    def _mark_used(self,entry):
        if self.checkpoint_idle_seconds is not None:
            entry['last_used']=_monotonic_value(self._idle_clock)

    def record_transition(self,event,key,**fields):
        """Append bounded opt-in diagnostics without changing pool ownership."""
        trace=self._transition_trace
        return None if trace is None else trace.record(event,key,**fields)

    def _drop(self,key):
        entry = self._entries.pop(key)
        digest = _pool_identity(entry['checkpoint'])
        row = self._buffers[digest]
        row['refs'] -= 1
        if not row['refs']:
            del self._buffers[digest]
            return row['bytes']
        return 0

    def reserve(self,size,*,timeout=0):
        if type(size) is not int or not 0 <= size <= self.byte_limit:
            raise PoolFull('Checkpoint/staging reservation exceeds RAM pool')
        deadline = time.monotonic()+timeout
        with self._condition:
            while True:
                if self._closed: raise RuntimeError('RAM pool is closed')
                while self._bytes()+self._reserved+size > self.byte_limit:
                    victim = next((k for k,e in self._entries.items()
                        if _discardable(e) and not e['pins']),None)
                    if victim is None: break
                    failed=bool(self._entries[victim].get('failed',False))
                    freed=self._drop(victim)
                    if failed:
                        self._failed_evictions += 1
                    else:
                        self._evictions += 1
                    # Internal bounded append at the same lock-protected
                    # linearization point as removing the key/buffer.
                    if self._transition_trace is not None:
                        self._transition_trace.record(
                            'ram_failed_eviction' if failed else 'ram_clean_eviction',
                            victim,freed_bytes=freed)
                if self._bytes()+self._reserved+size <= self.byte_limit:
                    self._reserved += size
                    self._peak = max(self._peak,self._bytes()+self._reserved)
                    return _Reservation(self,size)
                remaining = deadline-time.monotonic()
                if remaining <= 0: raise PoolFull('All RAM victims are pinned or awaiting durability')
                self._waits += 1
                self._condition.wait(remaining)

    def admit(self,key,checkpoint,*,durable=False):
        size = _validate_pool_value(checkpoint)
        with self._condition:
            extra = 0 if _pool_identity(checkpoint) in self._buffers else size
            with self.reserve(extra) as reservation:
                return reservation.admit(key,checkpoint,durable=durable)

    def _admit(self,reservation,key,checkpoint,durable):
        size = _validate_pool_value(checkpoint)
        identity = _pool_identity(checkpoint)
        with self._condition:
            if self._closed: raise RuntimeError('RAM pool is closed')
            if reservation.closed: raise RuntimeError('Closed RAM reservation')
            row = self._buffers.get(identity)
            if row is None and size > reservation.size: raise PoolFull('Unreserved checkpoint allocation')
            old = self._entries.get(key)
            if old and (old['pins'] or not _discardable(old)):
                if old['checkpoint'] == checkpoint:
                    old['durable'] |= durable
                    if durable: old['failed']=False
                    self._mark_used(old)
                    return old['checkpoint']
                raise PoolFull('Cannot replace protected checkpoint')
            preserve_durable=bool(old and old['durable'] and old['checkpoint']==checkpoint)
            if row is not None:
                source = row['checkpoint']
                if type(checkpoint) is Embedding:
                    checkpoint = replace(checkpoint,payload=source.payload)
                else:
                    checkpoint = replace(checkpoint,blob=source.blob,frontier=source.frontier)
            else:
                self._buffers[identity] = dict(checkpoint=checkpoint,bytes=size,refs=0)
            # Increment before dropping a same-payload alias.
            self._buffers[identity]['refs'] += 1
            if old: self._drop(key)
            entry=dict(checkpoint=checkpoint,durable=bool(durable or preserve_durable),pins=0,failed=False)
            self._mark_used(entry)
            self._entries[key] = entry
            reservation.release()
            self._peak = max(self._peak,self._bytes()+self._reserved)
            self._condition.notify_all()
            return checkpoint

    def __contains__(self,key):
        with self._condition: return key in self._entries

    def is_durable(self,key):
        """Return the pool's acknowledged durability state for a resident key."""
        with self._condition:
            entry=self._entries.get(key)
            return bool(entry and entry['durable'])

    def get(self,key):
        with self._condition:
            entry = self._entries[key]
            self._mark_used(entry)
            self._entries.move_to_end(key)
            return entry['checkpoint']

    @contextmanager
    def pin(self,key):
        with self._condition:
            if self._closed: raise RuntimeError('RAM pool is closed')
            entry = self._entries[key]
            self._mark_used(entry)
            entry['pins'] += 1
            self._entries.move_to_end(key)
        try: yield entry['checkpoint']
        finally:
            with self._condition:
                entry['pins'] -= 1
                try:
                    if self._closed and entry['pins'] == 0:
                        if self._entries.get(key) is entry:
                            self._drop(key)
                    elif not self._closed and self._entries.get(key) is entry:
                        self._mark_used(entry)
                finally:
                    self._condition.notify_all()

    def mark_durable(self,key):
        with self._condition:
            if key in self._entries:
                self._entries[key]['durable'] = True
                self._entries[key]['failed'] = False
            self._condition.notify_all()

    def mark_failed(self,key):
        """Mark resident derived bytes failed without changing durability.

        An unpinned failed entry becomes an ordinary LRU victim. Active pins
        and unpublished dirty entries remain protected. This transient marker
        never changes checkpoint manifests or canonical history.
        """
        with self._condition:
            if key in self._entries and not self._entries[key]['durable']:
                self._entries[key]['failed']=True
            self._condition.notify_all()

    def collect_expired(self):
        """Expire eligible unpinned copies, protecting unpublished checkpoints."""
        if self.checkpoint_idle_seconds is None:
            return 0
        now=_monotonic_value(self._idle_clock)
        expired=[]
        with self._condition:
            for key,entry in list(self._entries.items()):
                last=entry.get('last_used')
                if last is None or entry['pins'] or not _discardable(entry):
                    continue
                if now < last or now-last <= self.checkpoint_idle_seconds:
                    continue
                failed=bool(entry.get('failed',False))
                freed=self._drop(key)
                self._idle_evictions+=1
                if failed:
                    self._failed_evictions+=1
                else:
                    self._evictions+=1
                if self._transition_trace is not None:
                    self._transition_trace.record('ram_idle_eviction',key,
                        freed_bytes=freed,failed=failed)
                expired.append(key)
            if expired:
                self._condition.notify_all()
        return len(expired)

    def remove(self,key,*,discard=False):
        with self._condition:
            entry = self._entries[key]
            if entry['pins'] or (not _discardable(entry) and not discard):
                raise PoolFull('Cannot evict pinned or unsaved checkpoint')
            self._drop(key)
            self._condition.notify_all()

    def preload(self,key,size,loader,*,timeout=0,trace_disk_read=False):
        if type(size) is not int or not 0 <= size <= self.byte_limit:
            raise PoolFull('Disk-read staging exceeds RAM pool')
        with self._condition:
            if self._closed: raise RuntimeError('RAM pool is closed')
            if key in self._loads: return self._loads[key]
            future = Future()
            if key in self._entries:
                future.set_result(self._entries[key]['checkpoint'])
                return future
            # Publish the pending-load identity BEFORE capacity waiting, so
            # simultaneous workers cannot each reserve/read the same payload.
            self._loads[key] = future
            def read():
                if not future.set_running_or_notify_cancel(): return
                traced=bool(trace_disk_read and self._transition_trace is not None)
                load_id=(self._transition_trace.record('disk_preload_started',key) if traced else None)
                try:
                    with self.reserve(size,timeout=timeout) as reservation:
                        loaded=loader()
                        # Serialize successful-read evidence with admission and
                        # any later eviction under the pool condition lock.
                        with self._condition:
                            result = reservation.admit(key,loaded,durable=True)
                            if traced:
                                self._transition_trace.record('disk_preload_succeeded',key,
                                    success=True,bytes=_pool_value_bytes(result),load_id=load_id)
                    future.set_result(result)
                except BaseException as exc:
                    if traced: self._transition_trace.record('disk_preload_failed',key,
                        success=False,load_id=load_id)
                    # Futures retain exception tracebacks. The traceback here
                    # includes this frame's loaded checkpoint local, so detach
                    # frames and chained errors before exposing the failure.
                    # Keep the original exception to preserve its type, args,
                    # and fields such as OSError.errno.
                    exc.__traceback__ = None
                    exc.__context__ = None
                    exc.__cause__ = None
                    future.set_exception(exc)
            def done(_):
                with self._condition:
                    self._loads.pop(key,None)
            future.add_done_callback(done)
            try:
                task = self._reader.submit(read)
                task.add_done_callback(lambda t:future.cancel() if t.cancelled() else None)
            except BaseException as exc: future.set_exception(exc)
            return future

    def metrics(self):
        with self._condition:
            dirty = {_pool_identity(e['checkpoint']) for e in self._entries.values() if not e['durable']}
            failed = {_pool_identity(e['checkpoint']) for e in self._entries.values() if e.get('failed',False)}
            return dict(resident_bytes=self._bytes(),reserved_bytes=self._reserved,
                dirty_bytes=sum(self._buffers[k]['bytes'] for k in dirty),
                failed_entries=sum(1 for e in self._entries.values() if e.get('failed',False)),
                failed_bytes=sum(self._buffers[k]['bytes'] for k in failed),entries=len(self._entries),
                unique_payloads=len(self._buffers),pins=sum(e['pins'] for e in self._entries.values()),
                peak_accounted_bytes=self._peak,byte_limit=self.byte_limit,
                clean_evictions=self._evictions,failed_evictions=self._failed_evictions,
                idle_evictions=self._idle_evictions,
                admission_waits=self._waits,loads_pending=len(self._loads))

    def close(self):
        with self._condition:
            self._closed = True
            self._condition.notify_all()
        self._reader.shutdown(wait=True,cancel_futures=True)
        with self._condition:
            # The reader is now quiescent, so no load can race this cleanup.
            # Borrowed entries remain accounted until their final pin exits.
            for key, entry in list(self._entries.items()):
                if not entry['pins']:
                    self._drop(key)
            self._condition.notify_all()


class DiskCheckpointPool:
    """One budget/LRU for derived checkpoint and SVE files, never history.

    Counts manifest and payload file lengths, not block-device traffic or
    power-loss certification. Without idle expiry, recency is transient and
    seeded from manifest mtime then snapshot ID. With idle expiry enabled,
    validated touches persist last-use time in the derived manifest's mtime so
    other pool instances and restarts observe the same expiry age. Manifest
    contents and canonical history are not changed.
    """
    def __init__(self,store,byte_limit,*,checkpoint_idle_seconds=None,clock_ns=None):
        if type(byte_limit) is not int or byte_limit <= 0: raise ValueError('Invalid disk budget')
        self.checkpoint_idle_seconds=_validate_checkpoint_idle_seconds(checkpoint_idle_seconds)
        self._clock_ns=time.time_ns if clock_ns is None else clock_ns
        if not callable(self._clock_ns): raise TypeError('clock_ns must be callable')
        self.store,self.byte_limit = store,byte_limit
        self.directory = store.root/'checkpoints'
        self.directory.mkdir(mode=0o700,exist_ok=True)
        self._reservation_directory = store.root/'.checkpoint-pool-reservations'
        from .store import _private_directory
        _private_directory(self._reservation_directory)
        self._condition = threading.Condition(threading.RLock())
        self._reserved = self._evictions = self._waits = 0
        self._idle_evictions=0
        self._recency = {}
        self._protected_payloads = {}
        self._clock = 0
        self._seed_recency()
        # Reclaim dead publisher staging bytes before the first admission,
        # even when the configured pool has no pressure or expiry worker.
        # Do not scan/validate numerical manifests here: request-time checks
        # must still report corrupt checkpoints at their existing boundary.
        with self._condition,self.store._locked():
            if self._reclaim_checkpoint_stages_locked():
                from .store import _sync_dir
                _sync_dir(self.directory)

    def _seed_recency(self):
        rows=[]
        for path in self.directory.iterdir():
            if path.suffix not in ('.json','.sve'): continue
            if path.is_file() and not path.is_symlink():
                rows.append((path.stat().st_mtime_ns,self._recency_id(path)))
        rows.sort(key=lambda row:(row[0],row[1]))
        for stamp,snapshot in rows:
            self._recency[snapshot]=stamp
            self._clock=max(self._clock,stamp)

    def _bytes(self):
        total=0
        for path in self.directory.iterdir():
            try:
                if path.is_file(): total+=path.stat().st_size
            except FileNotFoundError:
                # A concurrent publication cleanup may unlink a staging file
                # between the type check and the size read.
                continue
        return total

    def touch(self,snapshot):
        """Mark a validated checkpoint read/publication as most recently used.

        The caller is responsible for invoking this only after canonical
        manifest and payload validation succeeded. When idle expiry is enabled,
        persist last-use time in the derived manifest mtime; file contents and
        canonical history remain unchanged.
        """
        self.store.authorize(snapshot,'materialize')
        path=self.directory/(snapshot+'.json')
        return self._touch_path(path,snapshot)

    def touch_embedding(self,digest):
        """Touch a validated derived image copy under the caller's digest lease."""
        from .store import _id
        path=self.directory/(_id(digest)+'.sve')
        return self._touch_path(path,path.name)

    def _touch_path(self,path,identity):
        with self._condition,self.store._locked():
            if not path.is_file() or path.is_symlink():
                raise FileNotFoundError('Derived cache record is absent')
            if self.checkpoint_idle_seconds is None:
                self._clock=max(self._clock+1,time.time_ns())
                self._recency[identity]=self._clock
                return self._clock
            flags=os.O_RDONLY|getattr(os,'O_NOFOLLOW',0)
            fd=os.open(path,flags)
            try:
                info=os.fstat(fd)
                if not stat.S_ISREG(info.st_mode):
                    raise FileNotFoundError('Checkpoint manifest is not a regular file')
                now=_wall_clock_ns(self._clock_ns)
                # Never move an existing access stamp backwards after a wall
                # clock rollback. A 1ns increment records a touch even when a
                # deterministic/coarse clock returns the same timestamp.
                stamp=max(now,info.st_mtime_ns+1)
                os.utime(fd,ns=(info.st_atime_ns,stamp))
                os.fsync(fd)
            finally:
                os.close(fd)
            self._clock=max(self._clock,stamp)
            self._recency[identity]=stamp
            return stamp

    @staticmethod
    def _recency_id(path):
        return path.name if path.suffix=='.sve' else path.stem

    def _recency_key(self,path):
        identity=self._recency_id(path)
        if self.checkpoint_idle_seconds is not None:
            stamp=path.stat().st_mtime_ns
            self._recency[identity]=stamp
            self._clock=max(self._clock,stamp)
            return (stamp,identity)
        stamp=self._recency.get(identity)
        if stamp is None:
            stamp=path.stat().st_mtime_ns
            self._recency[identity]=stamp
            self._clock=max(self._clock,stamp)
        return (stamp,identity)

    @staticmethod
    def _payload_id(path,retained):
        from .store import IntegrityError
        try:
            envelope=json.loads(path.read_bytes())
            digest=envelope['manifest']['payload_sha256']
            if (not isinstance(digest,str) or len(digest)!=64
                    or any(ch not in '0123456789abcdef' for ch in digest)):
                raise ValueError('invalid payload digest')
            return digest
        except (OSError,ValueError,KeyError,TypeError) as exc:
            if path.stem in retained:
                raise IntegrityError('Retained checkpoint manifest is unreadable') from exc
            return None

    @staticmethod
    def _reservation_bytes(fd,record):
        from .store import _json_bytes
        data=_json_bytes(record)
        os.lseek(fd,0,os.SEEK_SET)
        os.ftruncate(fd,0)
        view=memoryview(data)
        while view:
            written=os.write(fd,view)
            if written<=0: raise OSError('Could not write checkpoint reservation metadata')
            view=view[written:]
        os.fsync(fd)

    def _new_reservation_locked(self,payload_sha256):
        """Create a globally visible zero-byte digest pin while Store._locked is held."""
        from .store import _private_directory,_sync_dir
        _private_directory(self._reservation_directory)
        path=self._reservation_directory/(uuid.uuid4().hex+'.json')
        flags=os.O_CREAT|os.O_EXCL|os.O_RDWR|getattr(os,'O_NOFOLLOW',0)
        fd=os.open(path,flags,0o600)
        try:
            fcntl.flock(fd,fcntl.LOCK_SH)
            record=dict(schema=1,pid=os.getpid(),byte_limit=self.byte_limit,
                        reserved_bytes=0,payload_sha256=payload_sha256)
            self._reservation_bytes(fd,record)
            _sync_dir(self._reservation_directory)
            return fd,path,record
        except BaseException:
            os.close(fd)
            path.unlink(missing_ok=True)
            _sync_dir(self._reservation_directory)
            raise

    def _active_reservations_locked(self):
        """Read live reservation leases and unlink markers whose flock owner died.

        Caller holds the Store transaction lock, so active owners cannot be in the
        middle of changing their record while it is read.
        """
        from .store import IntegrityError,_sync_dir
        rows=[]
        changed=False
        for path in sorted(self._reservation_directory.glob('*.json')):
            if path.is_symlink():
                raise IntegrityError('Checkpoint reservation marker is a symlink')
            try:
                fd=os.open(path,os.O_RDWR|getattr(os,'O_NOFOLLOW',0))
            except FileNotFoundError:
                continue
            try:
                try:
                    fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
                except BlockingIOError:
                    try: record=json.loads(path.read_bytes())
                    except (OSError,ValueError) as exc:
                        raise IntegrityError('Live checkpoint reservation metadata is unreadable') from exc
                    if (not isinstance(record,dict) or set(record)!=
                            {'schema','pid','byte_limit','reserved_bytes','payload_sha256'}
                            or type(record['schema']) is not int or record['schema']!=1
                            or type(record['pid']) is not int or record['pid']<=0
                            or type(record['byte_limit']) is not int or record['byte_limit']<=0
                            or type(record['reserved_bytes']) is not int or record['reserved_bytes']<0
                            or record['reserved_bytes']>record['byte_limit']):
                        raise IntegrityError('Live checkpoint reservation metadata is invalid')
                    digest=record['payload_sha256']
                    if digest is not None and (type(digest) is not str or len(digest)!=64
                            or any(ch not in '0123456789abcdef' for ch in digest)):
                        raise IntegrityError('Live checkpoint reservation digest is invalid')
                    rows.append(record)
                else:
                    # No process holds this lock: a killed owner left only a
                    # transient marker, so it must not reserve capacity forever.
                    path.unlink(missing_ok=True)
                    changed=True
            finally:
                os.close(fd)
        if changed: _sync_dir(self._reservation_directory)
        return rows

    def _remove_reservation_locked(self,path):
        from .store import _sync_dir
        # The reservation owner closes ``fd`` exactly once after leaving the
        # Store transaction. Keeping descriptor ownership with the caller avoids
        # accidentally closing a reused descriptor number during cleanup.
        path.unlink(missing_ok=True)
        _sync_dir(self._reservation_directory)

    def _reclaim_checkpoint_stages_locked(self):
        """Reclaim abandoned temp inodes while preserving active publishers.

        The caller holds Store._locked(); checkpoint writers create and lock
        stages under that same lock, then retain an inode lock through rename.
        Image artifact stages are already created/published under Store's lock.
        """
        removed=0
        for path in sorted(self.directory.glob('.stage-*')):
            try:fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK)
            except OSError as exc:
                if exc.errno in (errno.ENOENT,errno.ELOOP):continue
                raise
            try:
                before=os.fstat(fd)
                if not stat.S_ISREG(before.st_mode) or before.st_uid!=os.geteuid():continue
                try:fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
                except BlockingIOError:continue
                try:current=path.lstat()
                except FileNotFoundError:continue
                if (current.st_dev,current.st_ino)!=(before.st_dev,before.st_ino):continue
                path.unlink();removed+=1;self._evictions+=1
            finally:os.close(fd)
        return removed

    def _collect_for(self,required=0,*,idle_now_ns=None):
        from .store import _sync_dir
        changed=False
        with self._condition,self.store._locked():
            reservations=self._active_reservations_locked()
            if any(row['byte_limit']!=self.byte_limit for row in reservations):
                raise PoolFull('Concurrent checkpoint pools use conflicting disk byte limits')
            reserved=sum(row['reserved_bytes'] for row in reservations)
            protected_payloads={row['payload_sha256'] for row in reservations
                                if row['payload_sha256'] is not None}
            changed=bool(self._reclaim_checkpoint_stages_locked())
            retained=self.store.checkpoint_roots()
            manifest_paths=sorted(self.directory.glob('*.json'))
            image_paths=sorted(self.directory.glob('*.sve'))
            present={path.stem for path in manifest_paths} | {path.name for path in image_paths}
            for snapshot in tuple(self._recency):
                if snapshot not in present:
                    self._recency.pop(snapshot,None)
            payload_by_manifest={}
            references={}
            eligible=[path for path in image_paths
                if path.stem not in protected_payloads and self._protected_payloads.get(path.stem,0)==0]
            used=self._bytes()
            for path in manifest_paths:
                digest=self._payload_id(path,retained)
                payload_by_manifest[path.stem]=digest
                if digest is not None: references[digest]=references.get(digest,0)+1
                if path.stem not in retained: eligible.append(path)
            # Reclaim payload files that have no surviving manifest reference.
            for path in sorted(self.directory.glob('*.state'),key=lambda p:p.name):
                if (path.stem not in references
                        and self._protected_payloads.get(path.stem,0)==0
                        and path.stem not in protected_payloads):
                    size=path.stat().st_size
                    path.unlink();self._evictions+=1;changed=True
                    used-=size
            eligible.sort(key=self._recency_key)
            for path in eligible:
                idle_expired=(idle_now_ns is not None and self.checkpoint_idle_seconds is not None
                    and _idle_age_exceeded(path.stat().st_mtime_ns,idle_now_ns,
                                           self.checkpoint_idle_seconds))
                if idle_now_ns is not None:
                    if not idle_expired: continue
                elif used+reserved+required <= self.byte_limit:
                    break
                digest=None if path.suffix=='.sve' else payload_by_manifest[path.stem]
                size=path.stat().st_size
                path.unlink();self._evictions+=1;changed=True
                if idle_expired: self._idle_evictions+=1
                used-=size
                self._recency.pop(self._recency_id(path),None)
                if digest is not None:
                    references[digest]-=1
                    if (references[digest]==0
                            and self._protected_payloads.get(digest,0)==0
                            and digest not in protected_payloads):
                        state=self.directory/(digest+'.state')
                        if state.exists() and state.is_file():
                            size=state.stat().st_size
                            state.unlink();self._evictions+=1;changed=True
                            used-=size
            if changed: _sync_dir(self.directory)

    def collect(self):
        """Reclaim orphan payloads and, if over budget, least-recent eligible files."""
        self._collect_for(0)

    def collect_expired(self):
        """Remove idle derived manifests outside canonical/read-lease roots.

        A strict elapsed-time comparison is used; a clock rollback retains
        future-stamped checkpoints conservatively. Payload aliases, global
        reservations and read leases are handled by the shared collector.
        """
        if self.checkpoint_idle_seconds is None:
            return 0
        before=self._idle_evictions
        self._collect_for(0,idle_now_ns=_wall_clock_ns(self._clock_ns))
        return self._idle_evictions-before

    def reserve(self,size,*,timeout=0,payload_sha256=None):
        if type(size) is not int or not 0 <= size <= self.byte_limit:
            raise PoolFull('Checkpoint exceeds disk pool')
        if type(timeout) not in (int,float) or not math.isfinite(timeout) or timeout < 0:
            raise ValueError('Invalid disk reservation timeout')
        if payload_sha256 is not None and (
                type(payload_sha256) is not str or len(payload_sha256)!=64
                or any(ch not in '0123456789abcdef' for ch in payload_sha256)):
            raise ValueError('Invalid protected checkpoint payload digest')
        deadline=time.monotonic()+timeout
        with self._condition:
            if payload_sha256 is not None:
                self._protected_payloads[payload_sha256]=(
                    self._protected_payloads.get(payload_sha256,0)+1)
            marker_fd=marker_path=None
            try:
                # Register the digest before waiting for space. Other processes'
                # collectors must see this protection even while bytes == 0.
                with self.store._locked():
                    rows=self._active_reservations_locked()
                    if any(row['byte_limit']!=self.byte_limit for row in rows):
                        raise PoolFull('Concurrent checkpoint pools use conflicting disk byte limits')
                    marker_fd,marker_path,marker_record=self._new_reservation_locked(payload_sha256)
                while True:
                    with self.store._locked():
                        rows=self._active_reservations_locked()
                        if any(row['byte_limit']!=self.byte_limit for row in rows):
                            raise PoolFull('Concurrent checkpoint pools use conflicting disk byte limits')
                        reserved=sum(row['reserved_bytes'] for row in rows)
                        if self._bytes()+reserved+size > self.byte_limit:
                            self._collect_for(size)
                            rows=self._active_reservations_locked()
                            if any(row['byte_limit']!=self.byte_limit for row in rows):
                                raise PoolFull('Concurrent checkpoint pools use conflicting disk byte limits')
                            reserved=sum(row['reserved_bytes'] for row in rows)
                        if self._bytes()+reserved+size <= self.byte_limit:
                            marker_record['reserved_bytes']=size
                            self._reservation_bytes(marker_fd,marker_record)
                            self._reserved+=size
                            return _DiskReservation(self,size,payload_sha256,marker_fd,
                                marker_path,marker_record)
                    remaining=deadline-time.monotonic()
                    if remaining<=0:
                        raise PoolFull('Retained checkpoints leave no space for publication')
                    self._waits+=1
                    # Store read-lease release has no callback into this pool.
                    # Poll briefly so an unpinned checkpoint can become eligible
                    # without waiting for an unrelated reservation release.
                    self._condition.wait(min(remaining,0.05))
            except BaseException:
                try:
                    if marker_fd is not None:
                        try:
                            with self.store._locked():
                                self._remove_reservation_locked(marker_path)
                        finally:
                            # If taking the Store lock itself failed, don't leak
                            # the file descriptor; later collectors reap its marker.
                            try: os.close(marker_fd)
                            except OSError: pass
                finally:
                    self._unprotect_payload(payload_sha256)
                raise

    def _grow_reservation(self,reservation,extra_bytes,*,timeout=0):
        if type(extra_bytes) is not int or extra_bytes<0:
            raise ValueError('Invalid reservation growth size')
        if type(timeout) not in (int,float) or not math.isfinite(timeout) or timeout<0:
            raise ValueError('Invalid reservation growth timeout')
        deadline=time.monotonic()+timeout
        with self._condition:
            if reservation.closed: raise RuntimeError('Reservation is closed')
            if reservation.pool is not self:
                raise ValueError('Reservation belongs to a different disk pool')
            if extra_bytes==0: return reservation.size
            if reservation.size+extra_bytes>self.byte_limit:
                raise PoolFull('Checkpoint exceeds disk pool')
            while True:
                if reservation.closed: raise RuntimeError('Reservation is closed')
                with self.store._locked():
                    rows=self._active_reservations_locked()
                    if any(row['byte_limit']!=self.byte_limit for row in rows):
                        raise PoolFull('Concurrent checkpoint pools use conflicting disk byte limits')
                    try:
                        current=json.loads(reservation.marker_path.read_bytes())
                        fd_stat=os.fstat(reservation.marker_fd)
                        path_stat=reservation.marker_path.stat()
                    except (OSError,ValueError) as exc:
                        from .store import IntegrityError
                        raise IntegrityError('Active checkpoint reservation is unavailable') from exc
                    if ((fd_stat.st_dev,fd_stat.st_ino)!=(path_stat.st_dev,path_stat.st_ino)
                            or current!=reservation.marker_record
                            or current.get('reserved_bytes')!=reservation.size
                            or current.get('payload_sha256')!=reservation.payload_sha256):
                        from .store import IntegrityError
                        raise IntegrityError('Active checkpoint reservation changed unexpectedly')
                    reserved=sum(row['reserved_bytes'] for row in rows)
                    if self._bytes()+reserved+extra_bytes>self.byte_limit:
                        self._collect_for(extra_bytes)
                        rows=self._active_reservations_locked()
                        if any(row['byte_limit']!=self.byte_limit for row in rows):
                            raise PoolFull('Concurrent checkpoint pools use conflicting disk byte limits')
                        reserved=sum(row['reserved_bytes'] for row in rows)
                    if self._bytes()+reserved+extra_bytes<=self.byte_limit:
                        updated=dict(reservation.marker_record)
                        updated['reserved_bytes']=reservation.size+extra_bytes
                        try:
                            self._reservation_bytes(reservation.marker_fd,updated)
                        except BaseException as write_error:
                            # Marker updates truncate in place. If writing or
                            # fsync fails, its old contents may already be gone;
                            # abort the lease instead of leaving a live, partial
                            # global reservation that peers cannot safely parse.
                            try:
                                self._remove_reservation_locked(reservation.marker_path)
                            except BaseException as cleanup_error:
                                write_error.add_note(
                                    f'Could not remove failed reservation marker: {cleanup_error!r}')
                            finally:
                                self._reserved-=reservation.size
                                self._unprotect_payload(reservation.payload_sha256)
                                try:
                                    os.close(reservation.marker_fd)
                                except OSError as close_error:
                                    write_error.add_note(
                                        f'Could not close failed reservation marker: {close_error!r}')
                                reservation.closed=True
                                self._condition.notify_all()
                            raise
                        reservation.marker_record=updated
                        reservation.size+=extra_bytes
                        self._reserved+=extra_bytes
                        return reservation.size
                remaining=deadline-time.monotonic()
                if remaining<=0:
                    raise PoolFull('Retained checkpoints leave no space to grow reservation')
                self._waits+=1
                self._condition.wait(min(remaining,0.05))

    def _unprotect_payload(self,digest):
        if digest is None: return
        count=self._protected_payloads.get(digest,0)
        if count<=1: self._protected_payloads.pop(digest,None)
        else: self._protected_payloads[digest]=count-1

    def metrics(self):
        with self._condition,self.store._locked():
            reservations=self._active_reservations_locked()
            reserved=sum(row['reserved_bytes'] for row in reservations)
            return dict(file_bytes=self._bytes(),reserved_bytes=reserved,
                local_reserved_bytes=self._reserved,byte_limit=self.byte_limit,
                files_evicted=self._evictions,admission_waits=self._waits,
                idle_evictions=self._idle_evictions,
                recency_entries=len(self._recency))


class _DiskReservation:
    def __init__(self,pool,size,payload_sha256=None,marker_fd=None,marker_path=None,
            marker_record=None):
        self.pool,self.size,self.payload_sha256=pool,size,payload_sha256
        self.marker_fd,self.marker_path,self.closed=marker_fd,marker_path,False
        self.marker_record=marker_record
    def __enter__(self): return self
    def __exit__(self,*_): self.release()
    def grow(self,extra_bytes,*,timeout=0):
        return self.pool._grow_reservation(self,extra_bytes,timeout=timeout)
    def release(self):
        with self.pool._condition:
            if not self.closed:
                self.pool._reserved-=self.size
                self.pool._unprotect_payload(self.payload_sha256)
                try:
                    with self.pool.store._locked():
                        self.pool._remove_reservation_locked(self.marker_path)
                finally:
                    # Even if unlink/fsync reports an error, release the kernel
                    # lease. A surviving marker is recognized as stale later.
                    try: os.close(self.marker_fd)
                    except OSError: pass
                    self.closed=True
                    self.pool._condition.notify_all()
