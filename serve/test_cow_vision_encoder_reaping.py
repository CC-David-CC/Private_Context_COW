import ctypes,os,subprocess,sys,tempfile,unittest
from pathlib import Path

class EncoderOwnedReapingTests(unittest.TestCase):
 def test_existing_lifecycle_assertions_hold_when_caller_is_subreaper(self):
  # Crash supervisors adopt orphaned descendants. Preserve the existing
  # exact /proc-absence assertions under that real Linux caller arrangement.
  program = '''
import ctypes,unittest
libc=ctypes.CDLL(None,use_errno=True)
if libc.prctl(36,1,0,0,0)!=0:raise OSError(ctypes.get_errno(),'PR_SET_CHILD_SUBREAPER')
suite=unittest.defaultTestLoader.loadTestsFromName('serve.test_cow_vision_encoder.CowNativeVisionEncoderTests.test_completion_timeout_and_cancel_stop_the_owned_encoder_descendant')
result=unittest.TextTestRunner(verbosity=2).run(suite)
raise SystemExit(not result.wasSuccessful())
'''
  child=subprocess.run([sys.executable,'-B','-c',program],cwd=Path(__file__).resolve().parents[1],env=os.environ.copy(),capture_output=True,text=True,timeout=30)
  self.assertEqual(child.returncode,0,child.stdout+child.stderr)
