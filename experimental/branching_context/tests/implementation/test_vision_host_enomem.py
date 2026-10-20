"""Host ENOMEM harness contract tests; not native model evidence."""
import importlib.util,unittest
from pathlib import Path
P=Path(__file__).resolve().parents[2]/'scripts/run_vision_host_enomem.py'
spec=importlib.util.spec_from_file_location('host_enomem',P);probe=importlib.util.module_from_spec(spec);spec.loader.exec_module(probe)
class HostEnomemTests(unittest.TestCase):
 def row(self):return dict(uid=1000,limit_bytes=134217728,encoder_exec_observed=True,ENOMEM_syscalls=1,child_exit=127,
  cli_exit=2,state='failed',durable=False,new_branch_absent=True,parent_unchanged=True,parent_payload_unchanged=True,
  native_forward=0,ram_reserved=0,disk_reserved=0,pins=0,staging_empty=True,child_group_absent=True,
  unbounded_retry_encoder_calls=1,unbounded_retry_saved=True,worker_unchanged=True,tokens_exact=True,controller_exact=True,
  frontier_exact=True,max_abs_error=0.,control_prefix_forward=0)
 def test_requires_real_encoder_exec_and_kernel_enomem(self):
  probe.validate(self.row())
  for key,value in dict(uid=0,limit_bytes=0,encoder_exec_observed=False,ENOMEM_syscalls=0,child_exit=0,cli_exit=0,state='complete',durable=True).items():
   with self.subTest(key=key),self.assertRaises(probe.ProbeError):probe.validate(dict(self.row(),**{key:value}))
 def test_failure_preserves_old_branch_and_releases_all_transient_owners(self):
  for key,value in dict(new_branch_absent=False,parent_unchanged=False,parent_payload_unchanged=False,native_forward=1,
   ram_reserved=1,disk_reserved=1,pins=1,staging_empty=False,child_group_absent=False).items():
   with self.subTest(key=key),self.assertRaises(probe.ProbeError):probe.validate(dict(self.row(),**{key:value}))
 def test_recovery_requires_real_encoding_and_exact_continuation(self):
  for key,value in dict(unbounded_retry_encoder_calls=0,unbounded_retry_saved=False,worker_unchanged=False,tokens_exact=False,
   controller_exact=False,frontier_exact=False,max_abs_error=.001,control_prefix_forward=1).items():
   with self.subTest(key=key),self.assertRaises(probe.ProbeError):probe.validate(dict(self.row(),**{key:value}))
 def test_missing_or_boolean_counters_reject(self):
  for key in ['ENOMEM_syscalls','native_forward','unbounded_retry_encoder_calls']:
   row=self.row();row[key]=True
   with self.subTest(key=key),self.assertRaises(probe.ProbeError):probe.validate(row)
  row=self.row();del row['frontier_exact']
  with self.assertRaises(probe.ProbeError):probe.validate(row)
 def test_trace_parser_distinguishes_successful_exec_and_actual_mmap_failure(self):
  trace='42 execve("/p/encoder", ["/p/encoder"], 0) = 0\n42 mmap(NULL, 1, 3, 34, -1, 0) = -1 ENOMEM (Cannot allocate memory)\n'
  result=probe.trace_witness(trace,'/p/encoder')
  self.assertTrue(result['encoder_exec_observed']);self.assertEqual(result['ENOMEM_syscalls'],1)
  self.assertFalse(probe.trace_witness(trace,'/wrong/encoder')['encoder_exec_observed'])
  self.assertEqual(probe.trace_witness('42 write(2, "ENOMEM", 6) = 6','/p/encoder')['ENOMEM_syscalls'],0)
 def test_observer_preserves_real_entrypoint_and_child_binary(self):
  body="import importlib\nimportlib.import_module('serve.cow_http').main()\n"
  value=probe.install_hook(body,Path('/tmp/evidence'),Path('/native/encoder'))
  compile(value,'host-observer','exec')
  self.assertEqual(value.count("importlib.import_module('serve.cow_http').main()"),1)
  self.assertIn('process=_original_popen(command,*args,**kwargs)',value)
  self.assertIn('result=_original_run(self,model,job)',value)
  with self.assertRaises(probe.ProbeError):probe.install_hook(body+body,Path('/tmp/evidence'),Path('/native/encoder'))
if __name__=='__main__':unittest.main()
