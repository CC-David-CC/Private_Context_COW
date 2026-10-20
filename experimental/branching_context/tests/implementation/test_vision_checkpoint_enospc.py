"""CPU validation of the checkpoint-volume fault harness, not GPU evidence."""
import copy,importlib.util,sys,unittest
from pathlib import Path
P=Path(__file__).resolve().parents[2]/'scripts/run_vision_checkpoint_enospc.py';sys.path.insert(0,str(P.parent))
spec=importlib.util.spec_from_file_location('image_cp_full',P);probe=importlib.util.module_from_spec(spec);spec.loader.exec_module(probe)
class VisionCheckpointEnospcTests(unittest.TestCase):
 def row(self):
  return dict(operation_id='op',errno=28,cli_exit=2,state='failed',durable=False,publication_retryable=True,
   old_reference_unchanged=True,parent_payload_unchanged=True,parent_controller_unchanged=True,
   payload_published=False,manifest_published=False,stage_files_after=[],stage_existed_during_error=True,
   stage_device=7,checkpoint_device=7,stage_allocated_bytes=4096,stage_logical_bytes=8192,
   requested_payload_bytes=128*1024**2,retry_same_receipt=True,retry_revision_delta=1,
   retry_native_trace_unchanged=True,replay_native_trace_unchanged=True,retry_payload_matches_prepared=True,
   tokens_exact=True,controller_exact=True,frontier_exact=True,max_abs_error=0.,retained_root_usable=True)
 def test_payload_enospc_requires_partial_real_allocation_and_no_publication(self):
  probe.validate_campaign(self.row())
  bad=dict(errno=27,cli_exit=0,state='complete',durable=True,publication_retryable=False,payload_published=True,
   manifest_published=True,stage_files_after=['.stage-left'],stage_existed_during_error=False,
   stage_device=8,stage_allocated_bytes=-1,stage_logical_bytes=0)
  for key,value in bad.items():
   with self.subTest(key=key),self.assertRaises(probe.CampaignError):probe.validate_campaign(dict(self.row(),**{key:value}))
 def test_retry_requires_exact_consumed_snapshot_and_no_native_work(self):
  for key,value in dict(old_reference_unchanged=False,parent_payload_unchanged=False,parent_controller_unchanged=False,
   retry_same_receipt=False,retry_revision_delta=2,retry_native_trace_unchanged=False,replay_native_trace_unchanged=False,
   retry_payload_matches_prepared=False,tokens_exact=False,controller_exact=False,frontier_exact=False,
   max_abs_error=.001,retained_root_usable=False).items():
   with self.subTest(key=key),self.assertRaises(probe.CampaignError):probe.validate_campaign(dict(self.row(),**{key:value}))
 def test_logical_full_length_does_not_hide_short_physical_allocation(self):
  row=self.row();row['stage_allocated_bytes']=0;probe.validate_campaign(row)
  row['stage_allocated_bytes']=4096;row['stage_logical_bytes']=row['requested_payload_bytes'];probe.validate_campaign(row)
  row['stage_allocated_bytes']=row['requested_payload_bytes']
  with self.assertRaises(probe.CampaignError):probe.validate_campaign(row)
 def test_bootstrap_keeps_the_real_entrypoint_and_compiles(self):
  body="import importlib\nimportlib.import_module('serve.cow_http').main()\n"
  transformed=probe.install_writer_hook(body,Path('/tmp/control'),'fault','operation',Path('/tmp/store'))
  compile(transformed,'observer-bootstrap','exec')
  self.assertEqual(transformed.count("importlib.import_module('serve.cow_http').main()"),1)
  self.assertIn('return _original_parts(self,destination,parts)',transformed)
  self.assertIn('with _original_stage(self) as (stream,path)',transformed)
  with self.assertRaises(probe.CampaignError):probe.install_writer_hook(body+body,Path('/tmp/c'),'b','op',Path('/tmp/s'))
 def test_wrong_mount_namespace_device_or_unbounded_capacity_rejects(self):
  good=dict(fstype='ext4',source='/dev/loop9',mountpoint='/tmp/test/checkpoints',image_bytes=512*1024**2,capacity_bytes=500*1024**2,mount_namespace='mnt:[2]',init_mount_namespace='mnt:[1]')
  probe.validate_layout(good,checkpoint='/tmp/test/checkpoints',canonical_device=1,checkpoint_device=2)
  for row in [dict(good,fstype='tmpfs'),dict(good,source='/dev/nvme0n1'),dict(good,mount_namespace='mnt:[1]'),dict(good,image_bytes=1024**3),dict(good,capacity_bytes=600*1024**2),dict(good,mountpoint='/tmp/not-the-checkpoints')]:
   with self.subTest(row=row),self.assertRaises(probe.CampaignError):probe.validate_layout(row,checkpoint='/tmp/test/checkpoints',canonical_device=1,checkpoint_device=2)
  with self.assertRaises(probe.CampaignError):probe.validate_layout(good,checkpoint='/tmp/test/checkpoints',canonical_device=2,checkpoint_device=2)
if __name__=='__main__':unittest.main()
