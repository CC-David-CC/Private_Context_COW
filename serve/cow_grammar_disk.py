"""Owner-only durable grammar definitions; canonical controllers remain elsewhere.

Linux/local filesystem fsync semantics. Registration publishes immutable definitions
before exposing compiled groups. A retry verifies an already-published definition.
This module grants no tools, execution authority, or HTTP admission.
"""
import hashlib
import fcntl
import json
import os
from pathlib import Path
import re
import stat
import uuid
from serve.cow_grammar_catalog import GrammarCatalog
from context_time_travel.store import InvalidOperation

_MAX_RECORD=5*1024*1024
def _bytes(value):
    return json.dumps(value,sort_keys=True,ensure_ascii=False,allow_nan=False,
                      separators=(',',':')).encode('utf-8')

def _open_root(path):
    path=Path(os.path.abspath(path))
    fd=os.open('/',os.O_RDONLY|os.O_DIRECTORY|os.O_CLOEXEC)
    try:
        for index,part in enumerate(path.parts[1:]):
            final=index==len(path.parts)-2
            if final:
                try:os.mkdir(part,0o700,dir_fd=fd);os.fsync(fd)
                except FileExistsError:pass
            child=os.open(part,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW|os.O_CLOEXEC,dir_fd=fd)
            os.close(fd);fd=child
        info=os.fstat(fd)
        if info.st_uid!=os.geteuid() or stat.S_IMODE(info.st_mode)!=0o700:
            raise InvalidOperation('Grammar directory must be owner-only mode0700')
        return fd
    except BaseException:
        os.close(fd);raise

class DurableGrammarCatalog(GrammarCatalog):
    def __init__(self,directory,tokenizer,eos_tokens,**limits):
        super().__init__(tokenizer,eos_tokens,**limits)
        self._fd=_open_root(directory)
        self._lock_fd=None
        try:
            self._lock_fd=os.open('.lock',os.O_RDWR|os.O_CREAT|os.O_NOFOLLOW|os.O_CLOEXEC,0o600,dir_fd=self._fd)
            info=os.fstat(self._lock_fd)
            if (not stat.S_ISREG(info.st_mode) or info.st_nlink!=1
                    or info.st_uid!=os.geteuid() or stat.S_IMODE(info.st_mode)!=0o600):
                raise InvalidOperation('Unsafe grammar lock')
            with self._lock:
                fcntl.flock(self._lock_fd,fcntl.LOCK_EX)
                try:self._load()
                finally:fcntl.flock(self._lock_fd,fcntl.LOCK_UN)
        except BaseException:
            self.close();raise

    def __enter__(self):return self
    def __exit__(self,*args):self.close()
    def close(self):
        for key in ('_lock_fd','_fd'):
            fd=getattr(self,key,None)
            if fd is not None:os.close(fd);setattr(self,key,None)

    def _read(self,name):
        fd=os.open(name,os.O_RDONLY|os.O_NOFOLLOW|os.O_CLOEXEC|os.O_NONBLOCK,dir_fd=self._fd)
        try:
            info=os.fstat(fd)
            if (not stat.S_ISREG(info.st_mode) or info.st_nlink!=1
                    or info.st_uid!=os.geteuid() or stat.S_IMODE(info.st_mode)!=0o600
                    or info.st_size>_MAX_RECORD):
                raise InvalidOperation('Unsafe or oversized grammar record')
            chunks=[];size=0
            while True:
                block=os.read(fd,min(1024*1024,_MAX_RECORD+1-size))
                if not block:break
                chunks.append(block);size+=len(block)
                if size>_MAX_RECORD:raise InvalidOperation('Oversized grammar record')
            return b''.join(chunks)
        finally:os.close(fd)

    def _load(self):
        for name in sorted(os.listdir(self._fd)):
            if name=='.lock' or name.startswith('.stage-'):continue
            if re.fullmatch(r'operation-[0-9a-f]{64}\.json',name):continue
            if not re.fullmatch(r'[0-9a-f]{64}\.json',name):
                raise InvalidOperation('Unknown grammar metadata entry')
            if name[:-5] in self._groups:continue
            try:record=json.loads(self._read(name).decode('utf-8'))
            except (ValueError,UnicodeError,RecursionError) as error:
                raise InvalidOperation('Invalid durable grammar JSON') from error
            if (type(record) is not dict or set(record)!={'schema','config','receipt'}
                    or type(record['schema']) is not int or record['schema']!=1
                    or type(record['receipt']) is not dict
                    or record['receipt'].get('group_id')!=name[:-5]):
                raise InvalidOperation('Grammar record identity mismatch')
            preview=GrammarCatalog(self._tokenizer,self._eos,
                max_groups=self._max_groups,max_source_bytes=self._max_bytes)
            receipt=preview.register(record['config'])
            if receipt!=record['receipt']:
                raise InvalidOperation('Durable grammar definition or execution identity changed')
            super().register(record['config'])

    def register(self,config):
        with self._lock:
            if self._lock_fd is None:raise InvalidOperation('Grammar catalog is closed')
            fcntl.flock(self._lock_fd,fcntl.LOCK_EX)
            try:
                self._load()
                return super().register(config)
            finally:fcntl.flock(self._lock_fd,fcntl.LOCK_UN)

    def _persist_group(self,config,receipt):
        payload=_bytes(dict(schema=1,config=config,receipt=receipt))
        self._publish_record(receipt['group_id']+'.json',payload)

    def _publish_record(self,name,payload):
        if len(payload)>_MAX_RECORD:raise InvalidOperation('Oversized durable grammar record')
        try:existing=self._read(name)
        except FileNotFoundError:existing=None
        if existing is not None:
            if existing!=payload:raise InvalidOperation('Conflicting durable grammar definition')
            os.fsync(self._fd)
            return
        staged='.stage-'+uuid.uuid4().hex
        fd=os.open(staged,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW|os.O_CLOEXEC,0o600,dir_fd=self._fd)
        try:
            os.fchmod(fd,0o600)
            view=memoryview(payload)
            while view:
                written=os.write(fd,view)
                if written<=0:raise OSError('Short grammar write')
                view=view[written:]
            os.fsync(fd);os.close(fd);fd=None
            # All catalog writers hold the same flock; the target was checked
            # absent under that lock. Rename avoids a two-hardlink crash window.
            os.rename(staged,name,src_dir_fd=self._fd,dst_dir_fd=self._fd)
            os.fsync(self._fd)
        finally:
            if fd is not None:os.close(fd)
            try:os.unlink(staged,dir_fd=self._fd)
            except FileNotFoundError:pass

    def register_operation(self,config,operation_id):
        """Bind one original registration ID durably before acknowledging it."""
        if type(operation_id) is not str or re.fullmatch('[A-Za-z][A-Za-z0-9_-]{0,63}',operation_id) is None:
            raise InvalidOperation('Bounded registration operation ID required')
        try:detached=json.loads(_bytes(config))
        except (ValueError,TypeError,UnicodeError,RecursionError) as error:
            raise InvalidOperation('Finite inline grammar JSON required') from error
        digest=hashlib.sha256(_bytes(detached)).hexdigest()
        name='operation-'+hashlib.sha256(operation_id.encode('ascii')).hexdigest()+'.json'
        with self._lock:
            if self._lock_fd is None:raise InvalidOperation('Grammar catalog is closed')
            fcntl.flock(self._lock_fd,fcntl.LOCK_EX)
            try:
                self._load()
                try:raw=self._read(name)
                except FileNotFoundError:raw=None
                if raw is not None:
                    try:record=json.loads(raw)
                    except (ValueError,UnicodeError,RecursionError) as error:
                        raise InvalidOperation('Corrupt grammar registration receipt') from error
                    if (type(record) is not dict or set(record)!={'schema','operation_id','definition_sha256','receipt'}
                            or type(record['schema']) is not int or record['schema']!=1
                            or record['operation_id']!=operation_id
                            or record['definition_sha256']!=digest
                            or type(record['receipt']) is not dict):
                        raise InvalidOperation('Registration ID has different or corrupt original definitions')
                    receipt=record['receipt']
                    group=self._groups.get(receipt.get('group_id'))
                    if group is None or json.loads(group[1])!=receipt:
                        raise InvalidOperation('Registration receipt references unavailable grammar identity')
                else:
                    receipt=super().register(detached)
                    record=dict(schema=1,operation_id=operation_id,definition_sha256=digest,receipt=receipt)
                    self._publish_record(name,_bytes(record))
                return dict(receipt,operation_id=operation_id,definition_sha256=digest)
            finally:fcntl.flock(self._lock_fd,fcntl.LOCK_UN)
