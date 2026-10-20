"""Retained image inputs and evictable embeddings under the canonical Store.

This is a trusted-owner byte/artifact layer. The serving adapter must validate
image formats and run the pinned normalizer/encoder; callbacks here do not
establish model correctness or authenticate a remote caller.
"""
from contextlib import contextmanager, ExitStack
from dataclasses import dataclass, replace
import errno
import fcntl
import hashlib
import json
import math
import os
import stat
import time

from .input_spans import canonical_inputs
from .pools import PoolFull
from .store import IntegrityError, InvalidOperation, NotFound, _hash, _id, _json_bytes
from .store import _private_directory, _stage, _sync_dir, _tokens, _decode
from .vision_embeddings import encoding_key, parse_sve, read_sve


@dataclass(frozen=True)
class ImageLease:
    span: dict
    embedding: object
    source: str


@contextmanager
def _opened_file(path,limit,*,buffering=-1):
    try:
        fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK)
    except OSError as exc:
        if exc.errno==errno.ELOOP:
            raise IntegrityError('Image artifact must not be a symlink') from exc
        raise
    try:
        # This context owns the descriptor even if wrapper construction fails.
        with os.fdopen(fd,'rb',buffering=buffering,closefd=False) as stream:
            before=os.fstat(stream.fileno())
            if not stat.S_ISREG(before.st_mode) or not 0 < before.st_size <= limit:
                raise IntegrityError('Invalid image artifact type or byte length')
            yield stream,before
    finally:os.close(fd)


def _check_read(stream,before,count):
    after=os.fstat(stream.fileno())
    if (count!=before.st_size or
            (before.st_size,before.st_mtime_ns,before.st_ctime_ns)!=
            (after.st_size,after.st_mtime_ns,after.st_ctime_ns)):
        raise IntegrityError('Image artifact changed during reading')


def _read_file(path,limit):
    with _opened_file(path,limit) as (stream,before):
        raw=stream.read(before.st_size+1)
        _check_read(stream,before,len(raw));return raw


class ImageArtifacts:
    def __init__(self,store,ram,disk,*,encoder_profile_id,expected_width,
                 original_bytes_limit,max_rows=1024,upload_idle_seconds=1800,clock_ns=None):
        if store._principal is not None:
            raise PermissionError('Image artifacts require the trusted local owner')
        _id(encoder_profile_id)
        for n in (expected_width,max_rows):
            if type(n) is not int or n<=0:raise InvalidOperation('Invalid image artifact bound')
        if original_bytes_limit is not None and (type(original_bytes_limit) is not int or original_bytes_limit<=0):
            raise InvalidOperation('Invalid retained original byte bound')
        if expected_width>=2**31 or max_rows>=2**31:
            raise InvalidOperation('Image dimensions exceed int32')
        if disk.store.root.resolve()!=store.root.resolve():
            raise InvalidOperation('Image and checkpoint disk owners must use the same store')
        if (type(upload_idle_seconds) not in (int,float) or not math.isfinite(upload_idle_seconds)
                or upload_idle_seconds<=0 or (clock_ns is not None and not callable(clock_ns))):
            raise InvalidOperation('Invalid upload expiry configuration')
        self.upload_idle_seconds=upload_idle_seconds
        self._clock_ns=time.time_ns if clock_ns is None else clock_ns
        self.store,self.ram,self.disk=store,ram,disk
        self.profile,self.width,self.max_rows=encoder_profile_id,expected_width,max_rows
        self.original_bytes_limit=original_bytes_limit
        self.originals=store.root/'image-originals'
        self.encodings=store.root/'image-encodings'
        self.leases=store.root/'image-input-leases'
        self.encoding_locks=store.root/'image-encoding-locks'
        self.upload_grants=store.root/'image-upload-grants'
        with store._locked():
            for folder in (self.originals,self.encodings,self.leases,self.encoding_locks,self.upload_grants):
                _private_directory(folder)
            _sync_dir(store.root)

    @contextmanager
    def _pin_original(self,digest):
        path=self.leases/(_id(digest)+'.lock')
        with self.store._locked():
            fd=os.open(path,os.O_CREAT|os.O_RDWR|os.O_NOFOLLOW,0o600)
            try:fcntl.flock(fd,fcntl.LOCK_SH)
            except BaseException:os.close(fd);raise
        try:yield
        finally:os.close(fd)

    @contextmanager
    def _encode_lock(self,key):
        fd=os.open(self.encoding_locks/(_id(key)+'.lock'),
                   os.O_CREAT|os.O_RDWR|os.O_NOFOLLOW,0o600)
        try:
            fcntl.flock(fd,fcntl.LOCK_EX)
            yield
        finally:os.close(fd)

    def _canonical_bytes(self):
        return sum(path.stat().st_size for folder in (self.originals,self.encodings,self.upload_grants)
                   for path in folder.iterdir() if path.is_file())

    def _read_upload_grant(self,path):
        try:
            envelope=_decode(_read_file(path,4096));value=envelope['grant']
            if (set(envelope)!={'grant','checksum'} or envelope['checksum']!=_hash(_json_bytes(value))
                    or set(value)!={'schema','namespace_id','image_id','bytes','expires_ns'}
                    or type(value['schema']) is not int or value['schema']!=1
                    or type(value['bytes']) is not int or not 0<value['bytes']<=16*1024**2
                    or type(value['expires_ns']) is not int or not 0<value['expires_ns']<2**63
                    or path.name!=value['namespace_id']+'-'+_id(value['image_id'])+'.json'):
                raise ValueError('upload fields/checksum')
            self.store.get_namespace(value['namespace_id'])
            return value
        except FileNotFoundError as error:raise NotFound('Image is not uploaded in this namespace') from error
        except (ValueError,TypeError,KeyError,InvalidOperation) as error:
            raise IntegrityError('Invalid image upload grant') from error

    def upload(self,namespace_id,original):
        """Publish owner-validated original bytes and an expiring namespace grant.

        This is artifact authorization/retention, not another model history.
        The serving layer must validate the image format before calling it.
        """
        if self.original_bytes_limit is None:raise InvalidOperation('Original admission requires a retention budget')
        self.store.get_namespace(namespace_id)
        if type(original) is not bytes or not 0<len(original)<=16*1024**2:
            raise InvalidOperation('Original image exceeds the 16 MiB input bound')
        identity=_hash(original)
        with self._pin_original(identity):
            self._publish(self.originals/(identity+'.input'),original,canonical=True)
            with self.store._locked():
                expires=self._clock_ns()+int(self.upload_idle_seconds*10**9)
                if type(expires) is not int or not 0<expires<2**63:
                    raise InvalidOperation('Upload expiry is outside the supported range')
                value=dict(schema=1,namespace_id=namespace_id,image_id=identity,
                    bytes=len(original),expires_ns=expires)
                raw=_json_bytes(dict(grant=value,checksum=_hash(_json_bytes(value))))
                path=self.upload_grants/(namespace_id+'-'+identity+'.json')
                present=os.path.lexists(path)
                old=len(_read_file(path,4096)) if present else 0
                if present:self._read_upload_grant(path)
                if self._canonical_bytes()+len(raw)-old>self.original_bytes_limit:
                    raise PoolFull('Retained image inputs exceed their configured capacity')
                stage=_stage(self.upload_grants,raw)
                try:
                    self.store._fault('image_before_upload_publish')
                    os.replace(stage,path);_sync_dir(self.upload_grants)
                    self.store._fault('image_after_upload_publish')
                finally:stage.unlink(missing_ok=True)
        return {key:value[key] for key in ('namespace_id','image_id','bytes','expires_ns')}

    @contextmanager
    def lease_uploaded(self,namespace_id,image_id):
        """Pin an admitted original through queueing/consumption/publication."""
        self.store.get_namespace(namespace_id);identity=_id(image_id)
        with self._pin_original(identity):
            with self.store._locked():
                value=self._read_upload_grant(self.upload_grants/(namespace_id+'-'+identity+'.json'))
                if value['expires_ns']<=self._clock_ns():raise NotFound('Image upload expired; upload its bytes again')
            yield {key:value[key] for key in ('namespace_id','image_id','bytes','expires_ns')}

    def _publish(self,path,raw,*,canonical=False):
        if canonical and self.original_bytes_limit is None:
            raise InvalidOperation('Original admission requires a retention budget')
        with self.store._locked():
            if path.exists() or path.is_symlink():
                if _read_file(path,len(raw))!=raw:
                    raise IntegrityError('Conflicting immutable image artifact')
                _sync_dir(path.parent)
                return
            if canonical and self._canonical_bytes()+len(raw)>self.original_bytes_limit:
                raise PoolFull('Retained image inputs exceed their configured capacity')
            stage=_stage(path.parent,raw)
            try:
                os.link(stage,path)
                _sync_dir(path.parent)
            finally:stage.unlink(missing_ok=True)

    @staticmethod
    def _metadata(span):
        return dict(schema=1,**{k:v for k,v in span.items()
                    if k not in ('kind','original_sha256','pad_token_id')})

    def _read_metadata(self,key):
        try:raw=_read_file(self.encodings/(_id(key)+'.json'),65536)
        except FileNotFoundError:return None
        try:
            envelope=json.loads(raw)
            value=envelope['value']
            if (set(envelope)!={'value','checksum'} or envelope['checksum']!=_hash(_json_bytes(value))
                    or type(value['schema']) is not int or value['schema']!=1):
                raise ValueError('schema/checksum')
            fields={k:v for k,v in value.items() if k!='schema'}
            span=canonical_inputs([dict(kind='image',original_sha256='0'*64,pad_token_id=0,**fields)])[0]
            if encoding_key(span['normalized_sha256'],span['encoder_profile_id'])!=key:
                raise ValueError('encoding identity')
            if span['encoder_profile_id']!=self.profile or span['width']!=self.width or span['rows']>self.max_rows:
                raise ValueError('encoding profile')
            return value
        except (ValueError,TypeError,KeyError,InvalidOperation) as exc:
            raise IntegrityError('Invalid retained image encoding record') from exc

    def _write_metadata(self,key,span):
        value=self._metadata(span)
        raw=_json_bytes(dict(value=value,checksum=_hash(_json_bytes(value))))
        self.store._fault('image_before_encoding_publish')
        self._publish(self.encodings/(key+'.json'),raw,canonical=True)
        self.store._fault('image_after_encoding_publish')

    def _validate_span(self,span):
        spans=canonical_inputs([span])
        if len(spans)!=1 or spans[0]['kind']!='image':raise InvalidOperation('Expected one image span')
        span=spans[0]
        if span['encoder_profile_id']!=self.profile or span['width']!=self.width or span['rows']>self.max_rows:
            raise IntegrityError('Image input uses an incompatible encoder/model profile')
        key=encoding_key(span['normalized_sha256'],self.profile)
        if self._read_metadata(key)!=self._metadata(span):
            raise IntegrityError('Image input differs from its retained encoding record')
        return span,key

    def _original(self,digest):
        try:raw=_read_file(self.originals/(_id(digest)+'.input'),16*1024**2)
        except FileNotFoundError as exc:raise NotFound('Retained original image is missing') from exc
        if _hash(raw)!=digest:raise IntegrityError('Retained original image digest mismatch')
        return raw

    def _verify_original(self,digest):
        """Verify all original bytes using admitted scratch, retaining no copy."""
        try:
            with _opened_file(self.originals/(_id(digest)+'.input'),16*1024**2,buffering=0) as (stream,before):
                size=min(64*1024,before.st_size,self.ram.byte_limit)
                while True:
                    try:reservation=self.ram.reserve(size);break
                    except PoolFull:
                        if size==1:raise
                        size=max(1,size//2)
                with reservation:
                    chunk=None;count=0;checksum=hashlib.sha256()
                    try:
                        while count<before.st_size:
                            chunk=stream.read(min(size,before.st_size-count))
                            if not chunk:break
                            checksum.update(chunk);count+=len(chunk);chunk=None
                        extra=bool(stream.read(1))
                        _check_read(stream,before,count+int(extra))
                        if checksum.hexdigest()!=digest:raise IntegrityError('Retained original image digest mismatch')
                    finally:chunk=None
        except FileNotFoundError as error:raise NotFound('Retained original image is missing') from error

    @contextmanager
    def _original_bytes(self,digest):
        """Admit the opened file's bounded read before allocating encoder input."""
        with ExitStack() as opened:
            try:stream,before=opened.enter_context(_opened_file(self.originals/(_id(digest)+'.input'),16*1024**2))
            except FileNotFoundError as error:raise NotFound('Retained original image is missing') from error
            with self.ram.reserve(before.st_size+1):
                raw=None
                try:
                    raw=stream.read(before.st_size+1);_check_read(stream,before,len(raw))
                    if _hash(raw)!=digest:raise IntegrityError('Retained original image digest mismatch')
                    yield raw
                finally:raw=None

    def _parse(self,payload,expected=None):
        if type(payload) is not bytes:raise InvalidOperation('Encoder output must be immutable bytes')
        return parse_sve(payload,expected_sha256=_hash(payload) if expected is None else expected,
                         expected_width=self.width,max_rows=self.max_rows)

    def _save_embedding(self,span,value):
        path=self.disk.directory/(value.sha256+'.sve')
        with self.disk.reserve(0,payload_sha256=value.sha256) as reservation:
            if not path.exists():reservation.grow(value.nbytes)
            self.store._fault('image_before_payload_publish')
            self._publish(path,value.payload)
            self.store._fault('image_after_payload_publish')
            self.disk.touch_embedding(value.sha256)
        self.ram.admit(('image',encoding_key(span['normalized_sha256'],self.profile)),value,durable=True)

    @contextmanager
    def prepare(self,*,original,normalized,encode,pad_token_id):
        """Keep an admission lease through the caller's canonical publication.

        Normalized bytes and encode() must come from the pinned producer. This
        layer records exact identities and checks repeat results, not semantics.
        """
        if self.original_bytes_limit is None:raise InvalidOperation('Original admission requires a retention budget')
        _tokens([pad_token_id])
        if type(original) is not bytes or not 0<len(original)<=16*1024**2:
            raise InvalidOperation('Original image exceeds the 16 MiB input bound')
        if type(normalized) is not bytes or not 0<len(normalized)<=64*1024**2:
            raise InvalidOperation('Normalized image exceeds the 64 MiB staging bound')
        original_id,normalized_id=_hash(original),_hash(normalized)
        key=encoding_key(normalized_id,self.profile)
        encoded=False
        with self._pin_original(original_id):
            self._publish(self.originals/(original_id+'.input'),original,canonical=True)
            with self._encode_lock(key):
                metadata=self._read_metadata(key)
                if metadata is None:
                    if encode is None:raise NotFound('Image encoding is absent and an encoder is required')
                    value=self._parse(encode(normalized))
                    span=canonical_inputs([dict(kind='image',original_sha256=original_id,
                        normalized_sha256=normalized_id,encoder_profile_id=self.profile,
                        embedding_sha256=value.sha256,embedding_bytes=value.nbytes,rows=value.rows,
                        grid_width=value.grid_width,grid_height=value.grid_height,width=value.width,
                        pad_token_id=pad_token_id)])[0]
                    self._write_metadata(key,span)
                    self._save_embedding(span,value)
                    encoded=True
                else:
                    span=canonical_inputs([dict(kind='image',original_sha256=original_id,
                        pad_token_id=pad_token_id,**{k:v for k,v in metadata.items() if k!='schema'})])[0]
            callback=None if encode is None else lambda _: (normalized,encode(normalized))
            with self.lease(span,reencode=callback) as held:
                yield replace(held,source='encoded') if encoded else held

    @contextmanager
    def retain(self,span):
        """Keep canonical inputs collectible only after queued work completes.

        No derived embedding is loaded and no encoder can run at admission.
        A worker later acquires the numerical lease under its own scheduler.
        """
        span,_=self._validate_span(span)
        with self._pin_original(span['original_sha256']):
            self._verify_original(span['original_sha256'])
            yield

    @contextmanager
    def lease(self,span,*,reencode=None):
        span,key=self._validate_span(span)
        ram_key=('image',key)
        with ExitStack() as held:
            held.enter_context(self._pin_original(span['original_sha256']))
            self._verify_original(span['original_sha256'])
            with self.ram._condition:
                if ram_key in self.ram:
                    value=held.enter_context(self.ram.pin(ram_key))
                else:value=None
            source='ram'
            if value is None:
                def load():
                    with self.disk.reserve(0,payload_sha256=span['embedding_sha256']):
                        value=read_sve(self.disk.directory/(span['embedding_sha256']+'.sve'),
                            expected_sha256=span['embedding_sha256'],expected_width=self.width,
                            max_rows=self.max_rows)
                        self.disk.touch_embedding(value.sha256)
                        return value
                try:
                    value=self.ram.preload(ram_key,span['embedding_bytes'],load).result()
                    source='disk'
                except FileNotFoundError:
                    if reencode is None:raise NotFound('Embedding was evicted; an encoder is required')
                    with self._encode_lock(key):
                        # Another producer may have completed while this one waited.
                        try:value=load();source='disk'
                        except FileNotFoundError:
                            with self._original_bytes(span['original_sha256']) as original:
                                normalized=payload=None
                                try:
                                    normalized,payload=reencode(original)
                                    if (type(normalized) is not bytes or not 0<len(normalized)<=64*1024**2
                                            or _hash(normalized)!=span['normalized_sha256']):
                                        raise IntegrityError('Re-encoding changed normalized image identity')
                                    value=self._parse(payload,span['embedding_sha256'])
                                    self._save_embedding(span,value);source='reencoded'
                                finally:original=normalized=payload=None
                with self.ram._condition:
                    self.ram.admit(ram_key,value,durable=True)
                    value=held.enter_context(self.ram.pin(ram_key))
            expected=(span['embedding_sha256'],span['rows'],span['width'],span['grid_width'],span['grid_height'])
            if (value.sha256,value.rows,value.width,value.grid_width,value.grid_height)!=expected:
                raise IntegrityError('Cached embedding differs from canonical input')
            yield ImageLease(dict(span),value,source)

    def collect(self):
        """Collect unreferenced originals/records; cache expiry is independent."""
        originals,encodings,active=set(),set(),set()
        with self.store._locked():
            now=self._clock_ns();expired_grants=[]
            # Validate every grant before deleting anything. Corrupt ownership
            # metadata fails closed, rather than making originals collectible.
            for path in self.upload_grants.glob('*.json'):
                value=self._read_upload_grant(path)
                if value['expires_ns']>now:originals.add(value['image_id'])
                else:expired_grants.append(path)
            for path in self.store.objects.glob('*.json'):
                snapshot=self.store.get_snapshot(path.stem)
                if snapshot['schema']!=2:continue
                for span in snapshot['inputs']:
                    if span['kind']=='image':
                        originals.add(span['original_sha256'])
                        encodings.add(encoding_key(span['normalized_sha256'],span['encoder_profile_id']))
            for path in self.leases.glob('*.lock'):
                fd=os.open(path,os.O_RDWR|os.O_NOFOLLOW)
                try:
                    try:fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
                    except BlockingIOError:active.add(path.stem)
                finally:os.close(fd)
            for path in expired_grants:path.unlink()
            if expired_grants:_sync_dir(self.upload_grants)
            removed=[]
            for path in self.originals.glob('*.input'):
                if path.stem not in originals|active:
                    path.unlink();removed.append(path.stem)
            # A producer can hold an original lease before publishing its new
            # encoding record. Conservatively retain small records during it.
            records=[]
            if not active:
                for path in self.encodings.glob('*.json'):
                    if path.stem not in encodings:
                        path.unlink();records.append(path.stem)
            if removed:_sync_dir(self.originals)
            if records:_sync_dir(self.encodings)
            return dict(originals_deleted=sorted(removed),encodings_deleted=sorted(records),
                        upload_grants_deleted=sorted(path.name for path in expired_grants))


class RetainedImageArtifacts:
    """Lease existing canonical inputs without configuring new-original admission.

    This view uses the same artifact files, validation and shared pools. Each
    span selects its recorded encoder profile. A configured producer may only
    regenerate its own profile; cached bytes need no available producer.
    """
    def __init__(self,store,ram,disk,*,expected_width,max_rows,reencode_profile_id=None):
        self.store,self.ram,self.disk=store,ram,disk
        self.width,self.max_rows=expected_width,max_rows
        if reencode_profile_id is not None:_id(reencode_profile_id)
        self.reencode_profile_id=reencode_profile_id

    @contextmanager
    def lease(self,span,*,reencode=None):
        span=canonical_inputs([span])[0]
        owner=ImageArtifacts(self.store,self.ram,self.disk,encoder_profile_id=span['encoder_profile_id'],
            expected_width=self.width,max_rows=self.max_rows,original_bytes_limit=None)
        callback=reencode
        if reencode is not None and span['encoder_profile_id']!=self.reencode_profile_id:
            def unavailable(_):raise InvalidOperation('No compatible pinned encoder for this retained image profile')
            callback=unavailable
        with owner.lease(span,reencode=callback) as item:yield item
