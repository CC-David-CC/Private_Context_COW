"""Contract checks for deterministic real-disconnect gates; not GPU evidence."""
import importlib.util,unittest
from pathlib import Path
P=Path(__file__).resolve().parents[2]/'scripts/run_vision_deep_cancel.py'
spec=importlib.util.spec_from_file_location('deep_cancel',P);probe=importlib.util.module_from_spec(spec);spec.loader.exec_module(probe)
class DeepCancelTests(unittest.TestCase):
 def row(self,stage='four'):
  return dict(stage=stage,cli_exit=-15,disconnect_observed=True,state='cancelled',durable=False,branch_unchanged=True,
   parent_unchanged=True,parent_payload_unchanged=True,forward_tokens=4 if stage=='four' else 360,prefill_tokens=0 if stage=='four' else 360,
   progress_tokens=4 if stage=='four' else 0,reservations=0,pins=0,queued=0,worker_healthy=True,
   tokens_exact=True,controller_exact=True,frontier_exact=True,max_abs_error=0.,control_prefix_forward=0)
 def test_after_prefill_cancellation_is_not_zero_work_or_a_commit(self):
  row=self.row('root');probe.validate_cancel(row)
  for key,value in dict(forward_tokens=0,prefill_tokens=0,progress_tokens=1,state='durable',durable=True,disconnect_observed=False,branch_unchanged=False).items():
   with self.subTest(key=key),self.assertRaises(probe.ProbeError):probe.validate_cancel(dict(row,**{key:value}))
 def test_four_tokens_must_be_consumed_not_merely_selected_or_queued(self):
  row=self.row();probe.validate_cancel(row)
  for key,value in dict(forward_tokens=3,prefill_tokens=1,progress_tokens=3,cli_exit=0,parent_unchanged=False,parent_payload_unchanged=False).items():
   with self.subTest(key=key),self.assertRaises(probe.ProbeError):probe.validate_cancel(dict(row,**{key:value}))
 def test_cancellation_recovery_requires_no_leak_and_exact_controller(self):
  for key,value in dict(reservations=1,pins=1,queued=1,worker_healthy=False,tokens_exact=False,controller_exact=False,
   frontier_exact=False,max_abs_error=.1,control_prefix_forward=1).items():
   with self.subTest(key=key),self.assertRaises(probe.ProbeError):probe.validate_cancel(dict(self.row(),**{key:value}))
 def test_publication_handoff_can_finish_but_retry_must_not_repeat_work(self):
  row=dict(disconnect_observed=True,cli_exit=-15,state='durable',durable=True,revision_delta=1,receipt_exact=True,
   retry_native_unchanged=True,parent_unchanged=True,parent_payload_unchanged=True,reservations=0,pins=0,
   tokens_exact=True,controller_exact=True,frontier_exact=True,max_abs_error=0.)
  probe.validate_publication(row)
  for key,value in dict(disconnect_observed=False,state='cancelled',durable=False,revision_delta=2,receipt_exact=False,
   retry_native_unchanged=False,parent_unchanged=False,reservations=1,tokens_exact=False,max_abs_error=.01).items():
   with self.subTest(key=key),self.assertRaises(probe.ProbeError):probe.validate_publication(dict(row,**{key:value}))
 def test_hook_keeps_real_create_emit_run_and_publication(self):
  body="import importlib\nimportlib.import_module('serve.cow_http').main()\n";hook=probe.install_hook(body,Path('/tmp/cancel'))
  compile(hook,'deep-cancel-observer','exec')
  for call in ['_original_create(self,*args,**kwargs)','_original_emit(self,event,**fields)','_original_publish(self,model,job,prepared,*args,**kwargs)','_original_run(self,model,job)']:self.assertIn(call,hook)
  self.assertEqual(hook.count("importlib.import_module('serve.cow_http').main()"),1)
  with self.assertRaises(probe.ProbeError):probe.install_hook(body+body,Path('/tmp/cancel'))
 def test_head_comparison_checks_fields_across_distinct_client_and_store_types(self):
  from dataclasses import make_dataclass
  A=make_dataclass('WireHead',['name','revision','tip','root']);B=make_dataclass('StoreHead',['name','revision','tip','root'])
  wire=A('publish',0,'a'*64,'b'*64);stored=B('publish',0,'a'*64,'b'*64)
  self.assertNotEqual(wire,stored)
  self.assertEqual(probe.head_identity(wire),probe.head_identity(stored))
  for change in [('other',0,'a'*64,'b'*64),('publish',1,'a'*64,'b'*64),('publish',0,'c'*64,'b'*64),('publish',0,'a'*64,'c'*64)]:
   self.assertNotEqual(probe.head_identity(wire),probe.head_identity(B(*change)))
  with self.assertRaises(probe.ProbeError):probe.head_identity(A('publish',True,'a'*64,'b'*64))
if __name__=='__main__':unittest.main()
