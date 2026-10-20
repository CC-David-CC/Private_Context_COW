"""Contract tests for pressure evidence; these are not real-GPU qualification."""
import importlib.util,sys,unittest
from pathlib import Path
P=Path(__file__).resolve().parents[2]/'scripts/run_vision_pool_pressure.py'
spec=importlib.util.spec_from_file_location('vision_pressure',P);probe=importlib.util.module_from_spec(spec);spec.loader.exec_module(probe)
class PoolPressureTests(unittest.TestCase):
 def capacity(self):
  return dict(ram_bytes=100,disk_bytes=100,byte_limit=1024,ram_reserved=0,disk_reserved=0,
   old_ram_absent=True,old_disk_absent=True,old_native_absent=True,lru_evictions=1,
   newest_forward=0,prefix_forward=366,consumed=366,generated=36,total_forward=402,
   encoder_calls=0,embedding_disk_present=True,recomputation_notice=True,parent_unchanged=True,reference_payload_unchanged=True,
   tokens_exact=True,controller_exact=True,frontier_exact=True,max_abs_error=0.)
 def expiry(self):
  return dict(ttl_seconds=2.,held_seconds=4.1,real_sweeps=2,active_pins=1,
   active_sha_before='a'*64,active_sha_during='a'*64,active_ram_survived=True,
   expired_ram_entries=0,expired_disk_files=0,expiry_error=None,old_processes_exited=True,
   new_native_process=True,prefix_forward=366,consumed=366,generated=36,total_forward=402,
   encoder_calls=1,recomputation_notice=True,parent_unchanged=True,reference_payload_unchanged=True,
   tokens_exact=True,controller_exact=True,frontier_exact=True,max_abs_error=0.)
 def test_capacity_requires_all_three_tiers_and_exact_native_work(self):
  probe.validate_capacity(self.capacity())
  for key,value in dict(old_ram_absent=False,old_disk_absent=False,old_native_absent=False,lru_evictions=0,
   newest_forward=1,prefix_forward=0,total_forward=401,encoder_calls=1,ram_bytes=1025,disk_bytes=1025,
   ram_reserved=1,disk_reserved=1).items():
   with self.subTest(key=key),self.assertRaises(probe.ProbeError):probe.validate_capacity(dict(self.capacity(),**{key:value}))
 def test_evicted_embedding_requires_one_real_reencoding_not_zero(self):
  absent=dict(self.capacity(),embedding_disk_present=False,encoder_calls=1)
  probe.validate_capacity(absent)
  for count in (0,2,True):
   with self.subTest(count=count),self.assertRaises(probe.ProbeError):probe.validate_capacity(dict(absent,encoder_calls=count))
  missing=self.capacity();del missing['embedding_disk_present']
  with self.assertRaises(probe.ProbeError):probe.validate_capacity(missing)
 def test_active_lease_requires_real_elapsed_time_sweeps_and_unchanged_bytes(self):
  probe.validate_expiry(self.expiry())
  for key,value in dict(held_seconds=.1,real_sweeps=0,active_pins=0,active_sha_during='b'*64,
   active_ram_survived=False,expired_ram_entries=1,expired_disk_files=1,expiry_error='failure',
   old_processes_exited=False,new_native_process=False,encoder_calls=0).items():
   with self.subTest(key=key),self.assertRaises(probe.ProbeError):probe.validate_expiry(dict(self.expiry(),**{key:value}))
 def test_both_arms_require_exact_controller_frontier_and_notice(self):
  for factory,validate in [(self.capacity,probe.validate_capacity),(self.expiry,probe.validate_expiry)]:
   for key,value in dict(recomputation_notice=False,parent_unchanged=False,reference_payload_unchanged=False,
    tokens_exact=False,controller_exact=False,frontier_exact=False,max_abs_error=.00001).items():
    with self.subTest(key=key),self.assertRaises(probe.ProbeError):validate(dict(factory(),**{key:value}))
 def test_boolean_or_missing_counters_cannot_qualify(self):
  for factory,validate in [(self.capacity,probe.validate_capacity),(self.expiry,probe.validate_expiry)]:
   row=factory();row['prefix_forward']=True
   with self.assertRaises(probe.ProbeError):validate(row)
   row=factory();del row['frontier_exact']
   with self.assertRaises(probe.ProbeError):validate(row)
 def test_observer_calls_real_consume_and_sweeper(self):
  original="import importlib\nimportlib.import_module('serve.cow_http').main()\n"
  hook=probe.install_observer(original,Path('/tmp/probe'))
  compile(hook,'observer','exec')
  self.assertIn('_original_consume(self,lease,spans,embeddings=embeddings)',hook)
  self.assertIn('_original_collect(self)',hook)
  self.assertEqual(hook.count("importlib.import_module('serve.cow_http').main()"),1)
  with self.assertRaises(probe.ProbeError):probe.install_observer(original+original,Path('/tmp/probe'))
if __name__=='__main__':unittest.main()
