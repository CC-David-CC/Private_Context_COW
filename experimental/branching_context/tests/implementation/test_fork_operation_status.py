"""Canonical fork operation receipts: actual Store/control plane,synthetic CPU model."""
import hashlib,json,unittest
from context_time_travel.store import Store,IntegrityError,NotFound
import test_pool_branch_actions as branch_fixture
from serve.cow_adapter import CowAdapter
class ForkOperationStatusTests(unittest.TestCase):
 def setUp(self):
  self.fixture=branch_fixture.CoordinatorBranchActionTests();self.addCleanup(self.fixture.doCleanups);self.fixture.setUp()
  self.ns,self.source=self.fixture.namespace_chat();self.store=self.fixture.coordinator.store
 def fork(self):
  return self.fixture.coordinator.fork('alternative',self.source.tip,source_branch='chat',operation_id='fork-status')
 def test_persisted_fork_status_survives_store_reopen_without_model_work(self):
  head=self.fork();before=dict(self.fixture.models[0].stats)
  status=Store(self.store.root).fork_operation_status('fork-status')
  self.assertTrue(status['durable']);self.assertEqual(status['scope'],'canonical-reference')
  self.assertEqual(status['receipt']['branch'],dict(name=head.name,revision=0,tip=head.tip,root=head.root))
  self.assertEqual(status['receipt']['snapshot'],self.source.tip)
  self.assertEqual(dict(self.fixture.models[0].stats),before)
 def test_fork_receipt_reads_in_new_python_process_after_coordinator_close(self):
  import os,subprocess,sys
  self.fork();self.fixture.coordinator.close()
  code="import json,os,sys;from context_time_travel.store import Store;print(json.dumps(dict(pid=os.getpid(),status=Store(sys.argv[1]).fork_operation_status('fork-status'))))"
  result=subprocess.run([sys.executable,'-B','-c',code,str(self.store.root)],capture_output=True,text=True,timeout=15)
  self.assertEqual(result.returncode,0,result.stderr);value=json.loads(result.stdout)
  self.assertNotEqual(value['pid'],os.getpid());self.assertTrue(value['status']['durable'])
  self.assertEqual(value['status']['receipt']['snapshot'],self.source.tip)
  self.assertEqual(value['status']['receipt']['branch']['revision'],0)
 def test_original_fork_receipt_does_not_become_current_head_after_append(self):
  head=self.fork();self.fixture.append('alternative',head.revision,'fork-resume',(258,299)).durable.result(5)
  current=self.store.head('alternative');self.assertNotEqual(current.tip,head.tip)
  status=self.store.fork_operation_status('fork-status')
  self.assertEqual(status['receipt']['snapshot'],head.tip);self.assertEqual(status['receipt']['branch']['revision'],0)
  self.assertEqual(self.store.head('alternative'),current)
 def test_unknown_and_nonfork_operations_are_not_reported_as_forks(self):
  for op in ['missing','root-chat']:
   with self.subTest(op=op),self.assertRaises(NotFound):self.store.fork_operation_status(op)
 def test_corrupt_canonical_fork_metadata_is_rejected(self):
  self.fork();path=self.store.references/'alternative.json';raw=json.loads(path.read_bytes())
  raw['receipts']['fork-status']['result']['tip']='0'*64;path.write_text(json.dumps(raw))
  with self.assertRaises(IntegrityError):self.store.fork_operation_status('fork-status')
 def test_nonowner_cannot_enumerate_fork_operation(self):
  self.fork()
  with self.assertRaises(PermissionError):self.store.for_principal('reader').fork_operation_status('fork-status')
 def test_status_is_read_only_and_adapter_exposes_it(self):
  self.fork()
  def files():return {str(f.relative_to(self.store.root)):hashlib.sha256(f.read_bytes()).hexdigest() for root in [self.store.references,self.store.objects] for f in root.rglob('*') if f.is_file()}
  before=files();status=CowAdapter(self.fixture.coordinator).operation_status('fork-status')
  self.assertTrue(status['durable']);self.assertEqual(files(),before)
if __name__=='__main__':unittest.main()
