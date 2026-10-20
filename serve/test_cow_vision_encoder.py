"""Real subprocess lifecycle using a fixture executable; no GPU is mocked/qualified."""
from concurrent.futures import CancelledError
import hashlib
import os
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest

from context_time_travel.pools import SharedCheckpointPool, DiskCheckpointPool, PoolFull
from context_time_travel.store import Store, IntegrityError, InvalidOperation
from serve.cow_vision_encoder import CowNativeVisionEncoder


FIXTURE='''#!{python}
import os,sys,struct,time
from pathlib import Path
model=Path(sys.argv[sys.argv.index('--model')+1])
(model.parent/'started').write_text(str(os.getpid()))
mode=model.read_text()
if mode=='hang':time.sleep(20)
if mode in ('spawn-child','spawn-child-exit'):
    import subprocess
    child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(20)'])
    (model.parent/'descendant').write_text(str(child.pid))
    if mode=='spawn-child':time.sleep(20)
print('READY '+('5' if mode=='wrong-ready' else '4'),flush=True)
for line in sys.stdin:
    if line.strip()=='QUIT':break
    _,source,target=line.split()
    rows=2000 if mode=='oversized' else 6
    values=[float('nan') if mode=='nan' else i/32 for i in range(rows*4)]
    payload=struct.pack('<5i',0x31455653,rows,3,2,4)+struct.pack('<'+'f'*len(values),*values)
    if mode=='truncated':payload=payload[:-1]
    if mode!='missing':Path(target).write_bytes(payload)
    if mode=='stderr-overflow':sys.stderr.write('x'*(1024**2+1));sys.stderr.flush()
    print('ERR refused' if mode=='error' else 'OK 6 3 2 1',flush=True)
    if mode=='duplicate':print('OK 6 3 2 1',flush=True)
if mode=='exit':sys.exit(7)
'''


class CowNativeVisionEncoderTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='cow-encoder-fixture-');self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.exe=self.root/'fixture.py';self.exe.write_text(FIXTURE.format(python=sys.executable));self.exe.chmod(0o700)
        self.projector=self.root/'projector';self.projector.write_bytes(b'projector')
        self.model=self.root/'model';self.model.write_text('ok')
        self.store=Store(self.root/'store');self.pool=SharedCheckpointPool(4*1024**2);self.addCleanup(self.pool.close)
        self.disk=DiskCheckpointPool(self.store,4*1024**2)

    def encoder(self,mode='ok',**options):
        self.model.write_text(mode)
        pins={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in (self.exe,self.projector,self.model)}
        return CowNativeVisionEncoder(executable=self.exe,projector=self.projector,model=self.model,
            pinned_sha256=pins,width=4,max_rows=1024,threads=1,timeout=options.pop('timeout',2),**options)

    def run_encoder(self,encoder,**options):
        return encoder.encode(b'trusted normalized bytes',store=self.store,pool=self.pool,disk=self.disk,gpu_id=5,**options)

    def clean(self):
        self.assertEqual(list((self.store.root/'image-staging').iterdir()),[])
        self.assertEqual(self.pool.metrics()['reserved_bytes'],0)
        self.assertEqual(self.disk.metrics()['reserved_bytes'],0)
        marker=self.root/'started'
        if marker.exists():self.assertFalse(Path('/proc/'+marker.read_text()).exists())

    def test_complete_output_and_process_receipt(self):
        result=self.run_encoder(self.encoder())
        self.assertEqual((result.embedding.rows,result.embedding.width),(6,4))
        self.assertEqual(result.receipt['exit_code'],0);self.assertTrue(result.receipt['process_absent'])
        self.assertEqual(result.receipt['gpu_id'],5);self.assertEqual(result.receipt['output_bytes'],116)
        self.clean()

    def test_malformed_protocol_and_payloads_fail_without_leaking(self):
        for mode in ('wrong-ready','duplicate','error','missing','truncated','nan','oversized','exit','stderr-overflow'):
            with self.subTest(mode=mode):
                with self.assertRaises((InvalidOperation,IntegrityError)):self.run_encoder(self.encoder(mode))
                self.clean()

    def test_changed_binary_or_model_is_rejected_before_spawn(self):
        for changed in (self.exe,self.model):
            encoder=self.encoder();changed.write_bytes(changed.read_bytes()+b'\n')
            with self.subTest(path=changed),self.assertRaises(IntegrityError):self.run_encoder(encoder)
            self.assertFalse((self.root/'started').exists())
        self.assertEqual(self.pool.metrics()['reserved_bytes'],0)

    def test_timeout_kills_child_and_releases_staging(self):
        start=time.monotonic()
        with self.assertRaises(TimeoutError):self.run_encoder(self.encoder('hang',timeout=.15))
        self.assertLess(time.monotonic()-start,3);self.clean()

    def test_cancellation_kills_child_and_releases_staging(self):
        cancelled=threading.Event();timer=threading.Timer(.15,cancelled.set);timer.start()
        try:
            with self.assertRaises(CancelledError):self.run_encoder(self.encoder('hang'),cancelled=cancelled.is_set)
        finally:timer.cancel()
        self.clean()

    def test_completion_timeout_and_cancel_stop_the_owned_encoder_descendant(self):
        import signal
        for mode in ('timeout', 'cancel', 'complete'):
            with self.subTest(mode=mode):
                marker=self.root/'descendant'
                marker.unlink(missing_ok=True)
                cancelled=threading.Event()
                timer=threading.Timer(.4,cancelled.set) if mode=='cancel' else None
                if timer:timer.start()
                try:
                    if mode=='complete':
                        result=self.run_encoder(self.encoder('spawn-child-exit'))
                        self.assertEqual(result.receipt['exit_code'],0)
                    else:
                        expected=TimeoutError if mode=='timeout' else CancelledError
                        with self.assertRaises(expected):
                            self.run_encoder(self.encoder('spawn-child',timeout=.4 if mode=='timeout' else 2),
                                             cancelled=cancelled.is_set)
                    self.assertTrue(marker.exists(),'fixture must actually spawn the descendant')
                    child=int(marker.read_text())
                    deadline=time.monotonic()+2
                    while Path('/proc/'+str(child)).exists() and time.monotonic()<deadline:
                        time.sleep(.01)
                    self.assertFalse(Path('/proc/'+str(child)).exists(),
                                     'encoder cancellation left its descendant alive')
                    self.clean()
                finally:
                    if timer:timer.cancel()
                    if marker.exists():
                        child=int(marker.read_text())
                        if Path('/proc/'+str(child)).exists():
                            owner=int((self.root/'started').read_text())
                            if os.getpgid(child)==owner:
                                try:os.kill(child,signal.SIGKILL)
                                except ProcessLookupError:pass

    def test_ram_or_disk_admission_refusal_never_starts_a_child(self):
        encoder=self.encoder();small=SharedCheckpointPool(64);self.addCleanup(small.close)
        with self.assertRaises(PoolFull):encoder.encode(b'x',store=self.store,pool=small,disk=self.disk,gpu_id=0)
        tiny=DiskCheckpointPool(self.store,64)
        with self.assertRaises(PoolFull):encoder.encode(b'x',store=self.store,pool=self.pool,disk=tiny,gpu_id=0)
        self.assertFalse((self.root/'started').exists());self.clean()

    def test_missing_or_wrong_pins_and_split_gpu_are_rejected(self):
        with self.assertRaises(IntegrityError):CowNativeVisionEncoder(executable=self.exe,projector=self.projector,
            model=self.model,pinned_sha256={},width=4)
        encoder=self.encoder()
        with self.assertRaises(InvalidOperation):encoder.encode(b'x',store=self.store,pool=self.pool,disk=self.disk,gpu_id=(0,1))
        self.assertFalse((self.root/'started').exists())


if __name__=='__main__':unittest.main()
