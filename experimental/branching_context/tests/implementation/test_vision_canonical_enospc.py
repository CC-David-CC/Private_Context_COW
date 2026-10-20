"""Admission checks for the actual CLI/image full-filesystem experiment."""
import copy
import importlib.util
from pathlib import Path
import unittest
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/"scripts"))
P=Path(__file__).resolve().parents[2]/"scripts/run_vision_canonical_enospc.py"
spec=importlib.util.spec_from_file_location("vision_enospc_probe",P)
probe=importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)

class VisionCanonicalEnospcTests(unittest.TestCase):
 def valid(self):
  return dict(cli_exit=2,operation_id="original",state="failed",durable=False,
   publication_retryable=True,errno=28,old_reference_unchanged=True,
   parent_payload_unchanged=True,parent_controller_unchanged=True,
   prepared_checkpoint_unchanged=True,stage_files=[],available_bytes=0,
   retry_same_receipt=True,retry_revision_delta=1,retry_native_trace_unchanged=True,
   replay_native_trace_unchanged=True,final_text="OK",retained_root_usable=True)
 def test_complete_real_fault_and_retry_claim_requires_all_witnesses(self):
  probe.validate_campaign(self.valid())
 def test_no_pass_for_missing_or_changed_witness(self):
  changes=dict(cli_exit=0,operation_id="",state="durable",durable=True,
   publication_retryable=False,errno=5,old_reference_unchanged=False,
   parent_payload_unchanged=False,parent_controller_unchanged=False,
   prepared_checkpoint_unchanged=False,stage_files=[".stage-x"],available_bytes=4096,
   retry_same_receipt=False,retry_revision_delta=2,retry_native_trace_unchanged=False,
   replay_native_trace_unchanged=False,final_text="",retained_root_usable=False)
  for key,bad in changes.items():
   for missing in [False,True]:
    value=self.valid()
    if missing:value.pop(key)
    else:value[key]=bad
    with self.subTest(key=key,missing=missing),self.assertRaises(probe.CampaignError):probe.validate_campaign(value)
 def test_live_filler_binding_is_explicit_and_rejects_wrong_device(self):
  import tempfile
  from run_q4_canonical_enospc import CanonicalEnospcError
  with tempfile.TemporaryDirectory() as temp:
   root=Path(temp);device=root.stat().st_dev
   raw=dict(fill_path=str(root/'.filler'),filled_bytes=4096,allocated_bytes=4096,
    errno=28,filesystem_write_error='[Errno 28] No space left on device',
    available_bytes_after=0,st_dev=device,inode=123)
   original=dict(raw)
   bound=probe.bind_filler_receipt(raw,root)
   self.assertEqual(raw,original)
   self.assertEqual(bound['canonical_st_dev'],device)
   with self.assertRaises(probe.CampaignError):probe.bind_filler_receipt(dict(raw,st_dev=device+1),root)
   with self.assertRaises(CanonicalEnospcError):probe.bind_filler_receipt(dict(raw,available_bytes_after=4096),root)
 def test_cli_retry_uses_checksummed_journal_not_empty_stdout(self):
  import tempfile,json
  sys.path.insert(0,str(Path(__file__).resolve().parents[4]/'tools'))
  from cow_cli import RequestJournal
  with tempfile.TemporaryDirectory() as temp:
   journal=RequestJournal(temp);op='retry-original';url='http://127.0.0.1:1234'
   body=json.dumps(dict(operation_id=op,branch='fault',revision=0)).encode()
   entry=journal.begin(op,url,'/cow/text/answer',body)
   with self.assertRaises(probe.CampaignError):probe.completed_retry_receipt(journal,op,url)
   receipt=dict(snapshot='a'*64,branch=dict(name='fault',tip='a'*64,root='b'*64,revision=1))
   journal.complete(entry,receipt)
   self.assertEqual(probe.completed_retry_receipt(journal,op,url),receipt)
   with self.assertRaises(probe.CampaignError):probe.completed_retry_receipt(journal,op,url+'/other')
   with self.assertRaises(probe.CampaignError):probe.completed_retry_receipt(journal,'different',url)
 def test_writer_hook_preserves_actual_entrypoint_and_ref_writer(self):
  body="setup\nimportlib.import_module('serve.cow_http').main()\n"
  result=probe.install_writer_hook(body,Path('/outside/control'),"branch","original",Path('/bounded'))
  compile(result.replace('setup\n',''),'<hook>','exec')
  self.assertEqual(result.count("importlib.import_module('serve.cow_http').main()"),1)
  self.assertIn('original_ref(self,path,value,branch=branch)',result)
  self.assertIn('getattr(local,"operation",None)==target_operation',result)
 def test_hook_refuses_missing_or_duplicate_entrypoint(self):
  for body in ["", "importlib.import_module('serve.cow_http').main()\n"*2]:
   with self.assertRaises(probe.CampaignError):probe.install_writer_hook(body,Path('/c'),"b","o",Path('/s'))

if __name__=='__main__':unittest.main()
