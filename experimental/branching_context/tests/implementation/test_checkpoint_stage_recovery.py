"""Actual filesystem/process staging lifetime, without a model or GPU."""
from pathlib import Path
import os,subprocess,sys,tempfile,threading,unittest
from context_time_travel.store import Store
from context_time_travel.real import BranchingModel
from context_time_travel.pools import DiskCheckpointPool

class CheckpointStageRecoveryTests(unittest.TestCase):
 def setUp(self):
  self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
  self.store=Store(self.temp.name);self.pool=DiskCheckpointPool(self.store,1024**2)
  self.model=object.__new__(BranchingModel);self.model.store=self.store
  self.model.checkpoint_directory=self.pool.directory;self.model._checkpoint_fault=lambda _:None
 def test_collect_reclaims_real_dead_writer_stage(self):
  child="""from pathlib import Path
import os,sys
from context_time_travel.store import Store
from context_time_travel.real import BranchingModel
m=object.__new__(BranchingModel);m.store=Store(sys.argv[1]);m.checkpoint_directory=m.store.root/'checkpoints'
def crash(point):
 if point=='payload_before_publish':os._exit(73)
m._checkpoint_fault=crash
m._publish_checkpoint_parts(m.checkpoint_directory/('a'*64+'.state'),(b'x'*8192,))
"""
  p=subprocess.run([sys.executable,'-B','-c',child,str(self.store.root)],capture_output=True,timeout=10)
  self.assertEqual(p.returncode,73,p.stderr.decode())
  stages=list(self.pool.directory.glob('.stage-*'));self.assertEqual(len(stages),1)
  self.assertEqual(stages[0].stat().st_size,8192)
  self.pool.collect()
  self.assertFalse(list(self.pool.directory.glob('.stage-*')),'dead writer staging bytes escaped LRU reclamation')
  self.assertEqual(self.pool._bytes(),0)
 def test_new_pool_reclaims_dead_writer_stage_before_first_request(self):
  code="""import os,sys
from context_time_travel.store import Store
from context_time_travel.real import BranchingModel
m=object.__new__(BranchingModel);m.store=Store(sys.argv[1]);m.checkpoint_directory=m.store.root/'checkpoints'
def crash(point):
 if point=='payload_before_publish':os._exit(73)
m._checkpoint_fault=crash;m._publish_checkpoint_parts(m.checkpoint_directory/('d'*64+'.state'),(b'x'*8192,))
"""
  child=subprocess.run([sys.executable,'-B','-c',code,str(self.store.root)],capture_output=True,timeout=10)
  self.assertEqual(child.returncode,73,child.stderr.decode())
  self.assertTrue(list(self.pool.directory.glob('.stage-*')))
  reopened=DiskCheckpointPool(self.store,1024**2)
  self.assertFalse(list(reopened.directory.glob('.stage-*')),'startup deferred abandoned-stage recovery until pressure')
  self.assertEqual(reopened._bytes(),0)
 def test_collect_preserves_live_writer_during_write_and_before_publish(self):
  for mode in ['parts','file']:
   for phase in ['during_write','before_publish']:
    with self.subTest(mode=mode,phase=phase):
     entered=threading.Event();release=threading.Event();errors=[]
     path=self.pool.directory/(('b' if mode=='parts' else 'c')*64+'.state')
     def gate(point):
      if point=='payload_'+phase:
       entered.set()
       if not release.wait(5):raise TimeoutError('writer not released')
     self.model._checkpoint_fault=gate
     def write():
      try:
       if mode=='parts':self.model._publish_checkpoint_parts(path,(b'a'*8192,b'b'*8192))
       else:self.model._publish_checkpoint_file(path,b'a'*8192+b'b'*8192)
      except BaseException as e:errors.append(e)
     thread=threading.Thread(target=write);thread.start()
     try:
      self.assertTrue(entered.wait(5));stages=list(self.pool.directory.glob('.stage-*'));self.assertEqual(len(stages),1)
      self.pool.collect();self.assertTrue(stages[0].exists(),'collector unlinked a live writer')
      reopened=DiskCheckpointPool(self.store,1024**2)
      self.assertTrue(stages[0].exists(),'startup collector unlinked a live writer')
     finally:release.set();thread.join(5)
     self.assertFalse(thread.is_alive());self.assertFalse(errors,errors)
     self.assertEqual(path.read_bytes(),b'a'*8192+b'b'*8192);path.unlink()
 def test_stage_cleanup_never_unlinks_a_replacement_inode(self):
  with self.model._checkpoint_stage() as (stream,path):
   stream.write(b'owned');stream.flush();path.unlink();path.write_bytes(b'replacement')
  self.assertEqual(path.read_bytes(),b'replacement')
 def test_foreign_names_and_stage_symlinks_are_not_followed(self):
  keep=self.pool.directory/'user-not-a-stage';keep.write_bytes(b'keep')
  outside=self.store.root/'outside';outside.write_bytes(b'outside')
  link=self.pool.directory/'.stage-link';link.symlink_to(outside)
  self.pool.collect()
  self.assertEqual(keep.read_bytes(),b'keep');self.assertEqual(outside.read_bytes(),b'outside')
  self.assertTrue(link.is_symlink())

if __name__=='__main__':unittest.main()
