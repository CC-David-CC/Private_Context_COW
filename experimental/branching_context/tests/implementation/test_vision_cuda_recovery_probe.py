"""CPU harness checks only; actual CUDA OOM is a separate executed campaign."""
import copy,importlib.util,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
P=Path(__file__).resolve().parents[2]/'scripts/run_vision_cuda_recovery_probe.py'
sys.path.insert(0,str(P.parent));spec=importlib.util.spec_from_file_location('vision_oom',P)
probe=importlib.util.module_from_spec(spec);spec.loader.exec_module(probe)
from context_time_travel import Store
from context_time_travel.input_spans import InputHistory
class VisionCudaRecoveryProbeTests(unittest.TestCase):
 def test_binding_uses_canonical_images_not_pad_projection_or_legacy_lineage(self):
  with tempfile.TemporaryDirectory() as tmp:
   store=Store(tmp);image=dict(kind='image',original_sha256='a'*64,normalized_sha256='b'*64,encoder_profile_id='c'*64,embedding_sha256='d'*64,embedding_bytes=20+6*4*4,rows=6,width=4,grid_width=3,grid_height=2,pad_token_id=123456)
   first=store.create_input_root('m',[dict(kind='text',tokens=[1,2]),image],{})
   second=store.create_input_root('m',[dict(kind='text',tokens=[1,2]),dict(image,original_sha256='e'*64,embedding_sha256='f'*64)],{})
   def forbidden(*args):raise AssertionError('image projection is not canonical history')
   with patch.object(Store,'lineage',forbidden),patch.object(InputHistory,'native_tokens',property(forbidden)):
    a=probe.image_source_binding(store,first);b=probe.image_source_binding(store,second)
   self.assertEqual((a['consumed'],a['text_tokens'],a['image_rows'],a['image_count']),(8,2,6,1))
   self.assertNotEqual(a['input_identity'],b['input_identity'])
   plain=store.create_root('m',[1,2],{})
   with self.assertRaises(probe.ProbeError):probe.image_source_binding(store,plain)
 def test_decode_gate_requires_exact_mixed_history_and_actual_shared_pages(self):
  binding=dict(snapshot='s',input_identity='a'*64,consumed=366,text_tokens=66,image_rows=300,image_count=1)
  row=dict(operation_id='fault',branch='trial',native_pid=123,lease_active=True,command='decode 42',input_binding=binding,
   fixture_capture=dict(native_capture=True,hot_restore=True,restores_delta=1,captures_delta=1,forward_delta=0,frontier_before_sha256='c'*64,frontier_after_sha256='c'*64),stats=dict(forward_tokens=0,prefill_tokens=0,consumed=366),pages=[[1,2,2,2097152,'K',0]])
  probe.validate_image_gate(row,binding,operation='fault',branch='trial')
  for key,bad in [('fixture_capture',dict(row['fixture_capture'],hot_restore=False)),('fixture_capture',dict(row['fixture_capture'],restores_delta=0)),('fixture_capture',{}),('operation_id','wrong'),('lease_active',False),('command','prefill 42'),('pages',[]),('stats',dict(forward_tokens=1,prefill_tokens=0,consumed=366)),('input_binding',dict(binding,input_identity='b'*64))]:
   with self.subTest(key=key),self.assertRaises(probe.ProbeError):probe.validate_image_gate(dict(row,**{key:bad}),binding,operation='fault',branch='trial')
  with self.assertRaises(probe.ProbeError):probe.validate_image_gate(dict(row,pages=[[1,2,1,2097152,'K',0]]),binding,operation='fault',branch='trial')
 def test_success_requires_actual_oom_replacement_and_both_exact_continuations(self):
  row=dict(oom_verified=True,helper_released=True,old_native_absent=True,replacement_before_followup=True,branch_unchanged_on_error=True,parent_unchanged=True,no_error_ack=True,
   same_service=dict(tokens_exact=True,controller_exact=True,frontier_exact=True,max_abs_error=0.,prefill_tokens=0,forward_tokens=36,output_tokens=36),restart=dict(tokens_exact=True,controller_exact=True,frontier_exact=True,max_abs_error=0.,prefill_tokens=0,forward_tokens=36,output_tokens=36))
  probe.validate_outcome(row)
  for key in ['oom_verified','helper_released','old_native_absent','replacement_before_followup','branch_unchanged_on_error','parent_unchanged','no_error_ack']:
   with self.subTest(key=key),self.assertRaises(probe.ProbeError):probe.validate_outcome(dict(row,**{key:False}))
  for side in ['same_service','restart']:
   for key,bad in [('tokens_exact',False),('controller_exact',False),('frontier_exact',False),('max_abs_error',.001),('max_abs_error',False),('prefill_tokens',1),('prefill_tokens',False),('forward_tokens',35),('forward_tokens',True),('output_tokens',True)]:
    case=copy.deepcopy(row);case[side][key]=bad
    with self.subTest(side=side,key=key),self.assertRaises(probe.ProbeError):probe.validate_outcome(case)
if __name__=='__main__':unittest.main()
