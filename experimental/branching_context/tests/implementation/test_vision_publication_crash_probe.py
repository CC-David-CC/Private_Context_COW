import importlib.util,sys,unittest
from pathlib import Path
P=Path(__file__).resolve().parents[2]/'scripts/run_vision_publication_crash_probe.py'
sys.path.insert(0,str(P.parent));spec=importlib.util.spec_from_file_location('crash_probe',P)
probe=importlib.util.module_from_spec(spec);spec.loader.exec_module(probe)
class VisionPublicationCrashProbeTests(unittest.TestCase):
 def test_stage_matrix_preserves_precommit_and_postcommit_distinction(self):
  self.assertEqual(len(set(probe.POINTS)),11)
  self.assertFalse(probe.committed('payload_during_write'))
  self.assertFalse(probe.committed('after_refs_stage_fsync'))
  self.assertTrue(probe.committed('branch_committed_before_ack'))
  self.assertTrue(probe.committed('after_refs_rename_before_dir_fsync'))
  with self.assertRaises(probe.ProbeError):probe.committed('unknown')
 def test_owned_crash_kills_and_reaps_real_cpu_service_and_child(self):
  import subprocess,sys,os,json,signal
  from types import SimpleNamespace
  # This is process-lifetime plumbing only, not an inference/GPU fixture.
  code="import subprocess,sys,json; p=subprocess.Popen([sys.executable,'-c','import sys;sys.stdin.buffer.read()'],stdin=subprocess.PIPE);print(json.dumps(dict(native_pid=p.pid)),flush=True);sys.stdin.buffer.read()"
  process=subprocess.Popen([sys.executable,'-c',code],stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True,start_new_session=True)
  child=json.loads(process.stdout.readline())['native_pid']
  service=SimpleNamespace(process=process,reader=None,_stdout_stream=None)
  gate=dict(native_pid=child,server_pid=process.pid,native_exe=str(Path('/proc',str(child),'exe').resolve()))
  try:
   receipt=probe.owned_crash(service,gate)
   self.assertEqual(receipt['server_exit_code'],-signal.SIGKILL)
   self.assertEqual(receipt['native_exit_code'],-signal.SIGKILL)
   self.assertFalse(Path('/proc',str(child)).exists())
   self.assertFalse(Path('/proc',str(process.pid)).exists())
  finally:
   if process.poll() is None:os.killpg(process.pid,signal.SIGKILL);process.wait(5)
   if process.stdin is not None:process.stdin.close()
 def test_each_recovery_requires_the_exact_work_and_single_move(self):
  for point in probe.POINTS:
   row=dict(point=point,server_exit=-9,native_exit=-9,server_absent=True,native_absent=True,
    no_ack=True,visible_new=probe.committed(point),source_intact=True,receipt_revision_delta=1,
    move_count=1,tokens_exact=True,controller_exact=True,frontier_exact=True,max_abs_error=0.,
    recovery_forward_tokens=0 if probe.committed(point) else 36,recovery_prefill_tokens=0,
    repeat_native_trace_unchanged=True,changed_request_rejected=True,checkpoint_stages_after=[],reference_stages_after=[])
   probe.validate_case(row,outputs=36)
   for k,bad in dict(server_exit=0,native_exit=None,server_absent=False,native_absent=False,no_ack=False,
    visible_new=not probe.committed(point),source_intact=False,receipt_revision_delta=2,move_count=2,
    tokens_exact=False,controller_exact=False,frontier_exact=False,max_abs_error=1.,recovery_prefill_tokens=1,
    recovery_forward_tokens=1,repeat_native_trace_unchanged=False,changed_request_rejected=False,
    checkpoint_stages_after=['.stage-x'],reference_stages_after=['.stage-ref']).items():
    with self.subTest(point=point,key=k),self.assertRaises(probe.ProbeError):probe.validate_case(dict(row,**{k:bad}),outputs=36)
if __name__=='__main__':unittest.main()
