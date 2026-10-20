"""Pinned native image encoder with COW-owned staging and no hidden cache.

Invoke only on the selected inference worker. The native protocol is the same
ENC/QUIT transport used by ordinary serving; artifacts and reuse belong to COW.
"""
from concurrent.futures import CancelledError
from dataclasses import dataclass
import hashlib
import math
import os
from pathlib import Path
import subprocess
import signal
import tempfile
import time

from context_time_travel.store import IntegrityError, InvalidOperation, _json_bytes, _private_directory
from context_time_travel.vision_artifacts import _read_file
from context_time_travel.vision_embeddings import parse_sve


def _stamp(path):
    stat=path.stat()
    return stat.st_dev,stat.st_ino,stat.st_size,stat.st_mtime_ns,stat.st_ctime_ns


@dataclass(frozen=True)
class EncodedImage:
    embedding: object
    receipt: dict


class CowNativeVisionEncoder:
    def __init__(self,*,executable,projector,model,pinned_sha256,width,max_rows=1024,
                 threads=None,flash_attn='on',timeout=120):
        if (type(width) is not int or not 0<width<2**31 or type(max_rows) is not int
                or not 0<max_rows<=1024 or flash_attn not in ('on','off','auto')
                or type(timeout) not in (int,float) or not math.isfinite(timeout) or timeout<=0):
            raise InvalidOperation('Invalid pinned image encoder configuration')
        threads=os.cpu_count() if threads is None else threads
        if type(threads) is not int or threads<=0:raise InvalidOperation('Invalid encoder thread count')
        paths=[Path(path).absolute() for path in (executable,projector,model)]
        if not isinstance(pinned_sha256,dict) or set(pinned_sha256)!={str(path) for path in paths}:
            raise IntegrityError('Encoder executable/projector/model pins are required')
        self.files={};self._stamps={}
        for path in paths:
            expected=pinned_sha256[str(path)];before=_stamp(path)
            with path.open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
            if digest!=expected or _stamp(path)!=before:
                raise IntegrityError('Pinned encoder input changed: '+str(path))
            self.files[str(path)]=dict(bytes=before[2],sha256=digest);self._stamps[path]=before
        self.argv=(str(paths[0]),'--mmproj',str(paths[1]),'--model',str(paths[2]),'--gpu',
            '--threads',str(threads),'--max-tokens',str(max_rows),'--flash-attn',flash_attn)
        self.width,self.max_rows,self.timeout=width,max_rows,timeout
        self.max_output_bytes=20+max_rows*width*4
        self.identity=hashlib.sha256(_json_bytes(dict(schema='cow-native-encoder-v1',
            implementation_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            files=self.files,argv=self.argv,width=width,max_rows=max_rows))).hexdigest()

    def _validate_files(self):
        for path,stamp in self._stamps.items():
            try:actual=_stamp(path)
            except OSError as error:raise IntegrityError('Pinned encoder input is unavailable') from error
            if actual!=stamp:raise IntegrityError('Pinned encoder input changed: '+str(path))

    def encode(self,normalized,*,store,pool,disk,gpu_id,cancelled=None):
        if (type(normalized) is not bytes or not 0<len(normalized)<=64*1024**2
                or type(gpu_id) is not int or gpu_id<0):
            raise InvalidOperation('Encoder requires bounded normalized bytes and one physical GPU')
        if cancelled is not None and not callable(cancelled):raise InvalidOperation('Invalid cancellation check')
        if store._principal is not None or disk.store.root.resolve()!=store.root.resolve():
            raise InvalidOperation('Encoder staging must belong to the canonical owner')
        self._validate_files()
        if cancelled is not None and cancelled():raise CancelledError('Image encoding cancelled before launch')
        staging=store.root/'image-staging';_private_directory(staging)
        stdout_bound=64*1024;stderr_bound=1024**2
        disk_bytes=len(normalized)+self.max_output_bytes+stdout_bound+stderr_bound
        ram_bytes=len(normalized)+2*self.max_output_bytes+stdout_bound+stderr_bound
        # Reserve before creating files or starting GPU work. These are staging
        # extents, not another embedding cache; every path releases both pools.
        with pool.reserve(ram_bytes),disk.reserve(disk_bytes),tempfile.TemporaryDirectory(prefix='encode-',dir=staging) as folder:
            directory=Path(folder);image=directory/'input.img';target=directory/'output.sve'
            image.write_bytes(normalized);image.chmod(0o600)
            stdout_path=directory/'stdout';stderr_path=directory/'stderr'
            started=time.monotonic();process=None
            def check_bounds():
                for path,bound in ((stdout_path,stdout_bound),(stderr_path,stderr_bound),(target,self.max_output_bytes)):
                    if path.exists() and path.stat().st_size>bound:
                        raise InvalidOperation('Native encoder exceeded a staging bound')
            try:
                with stdout_path.open('wb') as stdout,stderr_path.open('wb') as stderr:
                    process=subprocess.Popen(self.argv,cwd=directory,stdin=subprocess.PIPE,
                        stdout=stdout,stderr=stderr,start_new_session=True,umask=0o077,
                        env={**os.environ,'CUDA_VISIBLE_DEVICES':str(gpu_id)})
                    pending=b'ENC input.img output.sve\nQUIT\n'
                    while True:
                        if cancelled is not None and cancelled():raise CancelledError('Image encoding cancelled')
                        elapsed=time.monotonic()-started
                        if elapsed>=self.timeout:raise TimeoutError('Native image encoder timed out')
                        check_bounds()
                        try:
                            process.communicate(pending,timeout=min(.1,self.timeout-elapsed));break
                        except subprocess.TimeoutExpired:pending=None
                    if process.returncode!=0:raise InvalidOperation('Native image encoder exited with code '+str(process.returncode))
                check_bounds()
                if cancelled is not None and cancelled():raise CancelledError('Image encoding cancelled at completion')
                output=_read_file(stdout_path,stdout_bound).decode('utf-8').splitlines()
                if len(output)!=2 or output[0]!=f'READY {self.width}':
                    raise IntegrityError('Unexpected encoder readiness or response count')
                fields=output[1].split()
                if len(fields)!=5 or fields[0]!='OK':raise InvalidOperation('Native encoder refused the image')
                try:rows,nx,ny=map(int,fields[1:4]);milliseconds=float(fields[4])
                except ValueError as error:raise IntegrityError('Malformed encoder dimensions/timing') from error
                if not math.isfinite(milliseconds) or milliseconds<0:raise IntegrityError('Invalid encoder timing')
                try:payload=_read_file(target,self.max_output_bytes)
                except FileNotFoundError as error:raise IntegrityError('Encoder output is missing') from error
                embedding=parse_sve(payload,expected_sha256=hashlib.sha256(payload).hexdigest(),
                    expected_width=self.width,max_rows=self.max_rows)
                if (rows,nx,ny)!=(embedding.rows,embedding.grid_width,embedding.grid_height):
                    raise IntegrityError('Encoder response differs from the complete payload')
                self._validate_files()
                receipt=dict(encoder_id=self.identity,pid=process.pid,exit_code=process.returncode,
                    process_absent=not Path(f'/proc/{process.pid}').exists(),gpu_id=gpu_id,
                    input_bytes=len(normalized),output_bytes=embedding.nbytes,output_sha256=embedding.sha256,
                    rows=rows,grid=[nx,ny],width=self.width,seconds=time.monotonic()-started,
                    native_milliseconds=milliseconds,ram_staging_bytes=ram_bytes,disk_staging_bytes=disk_bytes)
                if not receipt['process_absent']:raise InvalidOperation('Native encoder process did not exit')
                return EncodedImage(embedding,receipt)
            finally:
                if process is not None:
                    # start_new_session gives this invocation its own group.
                    # Clean descendants even when the leader has already exited;
                    # a completed leader alone is not settled encoder work.
                    try:os.killpg(process.pid,signal.SIGKILL)
                    except ProcessLookupError:pass
                    if process.poll() is None:process.wait(timeout=10)
                    # A crash supervisor may be a child subreaper. In that
                    # arrangement killed descendants become our children.
                    # Reap only this invocation's private process group;
                    # never wait for or consume another worker's children.
                    while True:
                        try:os.waitpid(-process.pid,0)
                        except ChildProcessError:break
                        except InterruptedError:continue
