"""Claims from the real-GPU artifact rejection/rebuild campaign must be witnessed."""
import importlib.util
from pathlib import Path
import sys,unittest
SCRIPTS=Path(__file__).resolve().parents[2]/'scripts'
sys.path.insert(0,str(SCRIPTS))
spec=importlib.util.spec_from_file_location('reconstruction_probe',SCRIPTS/'run_vision_reconstruction_probe.py')
probe=importlib.util.module_from_spec(spec);spec.loader.exec_module(probe)

class ReconstructionProbeTests(unittest.TestCase):
 def rejection(self):
  return dict(case='cached-embedding-corrupt',cli_exit=2,state='failed',durable=False,
   prefix_forward_tokens=0,parent_head_unchanged=True,canonical_record_unchanged=True,
   controller_unchanged=True,encoder_calls=0,integrity_error=True,expected_fault_observed=True,
   model_process_exited=True,server_process_exited=True)
 def test_rejection_requires_zero_native_work_and_intact_history(self):
  probe.validate_rejection(self.rejection(),expected_encoder_calls=0)
  for key,value in dict(cli_exit=0,state='durable',durable=True,prefix_forward_tokens=1,
   parent_head_unchanged=False,canonical_record_unchanged=False,controller_unchanged=False,
   encoder_calls=1,integrity_error=False,expected_fault_observed=False,model_process_exited=False,server_process_exited=False).items():
   for missing in [False,True]:
    row=self.rejection()
    if missing:row.pop(key)
    else:row[key]=value
    with self.subTest(key=key,missing=missing),self.assertRaises(probe.ProbeError):probe.validate_rejection(row,expected_encoder_calls=0)
 def test_regeneration_rejection_requires_real_encoder_work(self):
  row=dict(self.rejection(),case='regenerated-embedding-mismatch',encoder_calls=1)
  probe.validate_rejection(row,expected_encoder_calls=1)
  with self.assertRaises(probe.ProbeError):probe.validate_rejection(dict(row,encoder_calls=0),expected_encoder_calls=1)
 def test_native_prefix_and_generated_work_are_separate(self):
  self.assertEqual(probe.work_accounting(402,366,36),dict(native_forward_tokens_total=402,prefix_forward_tokens=366,generated_forward_tokens=36))
  for args in [(402,402,36),(402,366,35),(True,0,1),(0,-1,1)]:
   with self.subTest(args=args),self.assertRaises(probe.ProbeError):probe.work_accounting(*args)
 def test_mutant_changes_one_finite_float_after_real_encoding(self):
  import hashlib,struct
  raw=struct.pack('<5i8f',0x31455653,2,2,1,4,*[i/32 for i in range(8)])
  changed,receipt=probe.mutate_regenerated_payload(raw)
  self.assertEqual(changed[:-4],raw[:-4]);self.assertNotEqual(changed[-4:],raw[-4:])
  self.assertNotEqual(receipt['before_sha256'],receipt['after_sha256'])
  self.assertEqual(receipt['before_sha256'],hashlib.sha256(raw).hexdigest())
  self.assertTrue(receipt['finite_before'] and receipt['finite_after'])
 def test_position_fault_keeps_envelope_checksum_valid_but_not_position(self):
  import hashlib,json
  value=dict(positions='qwen-image-grid-mrope-v1',rotary_cursor=86,position_sha256='a'*64)
  original=json.dumps(dict(manifest=value,checksum='old')).encode()
  for field in ['position_sha256','rotary_cursor']:
   changed=probe.mutate_position_manifest(original,field);parsed=json.loads(changed)
   self.assertNotEqual(parsed['manifest'][field],value[field])
   canonical=json.dumps(parsed['manifest'],sort_keys=True,separators=(',',':'),ensure_ascii=True).encode('ascii')+b'\n'
   self.assertEqual(parsed['checksum'],hashlib.sha256(canonical).hexdigest())
  with self.assertRaises(probe.ProbeError):probe.mutate_position_manifest(original,'other')

if __name__=='__main__':unittest.main()
