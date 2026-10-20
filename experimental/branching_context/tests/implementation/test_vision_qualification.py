"""Qualification runner invariants; synthetic records are not GPU evidence."""
import copy,importlib.util,unittest
from pathlib import Path
P=Path(__file__).resolve().parents[2]/'scripts/run_vision_qualification.py'
spec=importlib.util.spec_from_file_location('vision_qualification',P);probe=importlib.util.module_from_spec(spec);spec.loader.exec_module(probe)
class QualificationTests(unittest.TestCase):
 def test_sampling_includes_actual_final_and_four_token_restart_boundary(self):
  self.assertEqual(probe.sample_points(128),[1,4,16,64,128])
  self.assertEqual(probe.sample_points(93),[1,4,16,64,93])
  self.assertEqual(probe.sample_points(8),[1,4,8])
  self.assertEqual(probe.sample_points(1),[1])
  for n in [0,True,-1]:
   with self.subTest(n=n),self.assertRaises(probe.QualificationError):probe.sample_points(n)
 def test_fixture_keeps_text_and_image_extents_separate_and_suffixes_distinct(self):
  row=dict(text_control_tokens=8192,image_rows=1073,total_cells=9265,input_tokens=[1]*9265,
   suffixes=dict(A=[1]*512,B=[2]*512),suffix_tokens=512,outputs=128)
  probe.validate_fixture_counts(row)
  for key,value in dict(text_control_tokens=8191,image_rows=1074,total_cells=9264,suffix_tokens=511,outputs=True).items():
   with self.subTest(key=key),self.assertRaises(probe.QualificationError):probe.validate_fixture_counts(dict(row,**{key:value}))
  row['suffixes']['B']=row['suffixes']['A']
  with self.assertRaises(probe.QualificationError):probe.validate_fixture_counts(row)
 def test_strict_restore_requires_new_process_zero_work_and_exact_position(self):
  row=dict(forward_tokens=0,consumed=9265,bytes_loaded=1024,native_pid=20,python_pid=10)
  probe.validate_restore(row,consumed=9265,producer_pids={1,2})
  for key,value in dict(forward_tokens=1,consumed=9264,bytes_loaded=0,native_pid=2,python_pid=1).items():
   with self.subTest(key=key),self.assertRaises(probe.QualificationError):probe.validate_restore(dict(row,**{key:value}),consumed=9265,producer_pids={1,2})
 def test_generation_checks_every_actual_token_work_and_early_stop(self):
  row=dict(tokens=[7,8],forward_tokens=2,prefill_tokens=0,frontiers={'1':'one','2':'final'})
  probe.validate_generation(row,limit=128)
  for key,value in dict(forward_tokens=1,prefill_tokens=1,tokens=[],frontiers={'1':'one'}).items():
   with self.subTest(key=key),self.assertRaises(probe.QualificationError):probe.validate_generation(dict(row,**{key:value}),limit=128)
 def test_matching_batch_receipts_are_required_not_only_final_consumed_count(self):
  a=[[0,512],[512,512],[1024,49]];probe.same_batches(a,copy.deepcopy(a))
  for b in [[[0,1024],[1024,49]],[[0,512],[512,561]],None]:
   with self.subTest(b=b),self.assertRaises(probe.QualificationError):probe.same_batches(a,b)
if __name__=='__main__':unittest.main()
