"""Offline auditor negative controls; these are not native qualification."""
import importlib.util, tempfile, unittest, struct
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
class VisionAuthorizationAuditTests(unittest.TestCase):
 def module(self):
  spec=importlib.util.spec_from_file_location('authorization_audit',ROOT/'scripts/audit_vision_authorization.py')
  m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
 def test_exact_finite_vectors_are_required(self):
  m=self.module()
  with tempfile.TemporaryDirectory() as d:
   a=Path(d)/'a';b=Path(d)/'b';a.write_bytes(struct.pack('<ff',1,2));b.write_bytes(a.read_bytes())
   self.assertEqual(m.compare_vectors(a,b,2)['max_abs_error'],0)
   for raw in [struct.pack('<ff',1,3),struct.pack('<ff',1,float('nan')),b'x',b'']:
    b.write_bytes(raw)
    with self.subTest(raw=repr(raw)),self.assertRaises(m.AuditFailure):m.compare_vectors(a,b,2)
   a.write_bytes(struct.pack('<ff',1,float('inf')));b.write_bytes(a.read_bytes())
   with self.assertRaises(m.AuditFailure):m.compare_vectors(a,b,2)
 def test_successful_positioned_receipts_must_cover_every_consumed_token(self):
  m=self.module();r=[{'status':'ok','operator':'batched-prefill','position':0,'count':3,'consumed':3,'chunks':[[0,3]]},{'status':'ok','operator':'normal-verifier-decode','position':3,'count':1,'consumed':4,'chunks':[[3,1]]}]
  self.assertEqual(m.validate_work_receipts(r,4),4)
  for bad in [[r[1]],r+[r[1]],[dict(r[0],status='error'),r[1]],[dict(r[0],consumed=9),r[1]],[dict(r[0],chunks=[[0,2]]),r[1]]]:
   with self.subTest(bad=bad),self.assertRaises(m.AuditFailure):m.validate_work_receipts(bad,4)
 def test_denial_requires_guard_reason_and_no_effect_claim(self):
  m=self.module();r={'id':'x','status':401,'response':{'error':'wrong API key'},'native_jobs_delta':0,'encoder_calls_delta':0,'protected_state_unchanged':True,'body_sent':False,'seconds':.01}
  m.validate_denial(r,401,'API key',False)
  for bad in [dict(r,status=400),dict(r,response={'error':'bad JSON'}),dict(r,native_jobs_delta=1),dict(r,protected_state_unchanged=False),dict(r,body_sent=True),dict(r,seconds=4)]:
   with self.subTest(bad=bad),self.assertRaises(m.AuditFailure):m.validate_denial(bad,401,'API key',False)
if __name__=='__main__':unittest.main()
