"""Actual POSIX reference-writer crashes and constructor/transaction ordering."""
import os,select,signal,subprocess,sys,tempfile,unittest
from pathlib import Path
from context_time_travel.store import Store
WORKER = r"""
import os,signal,sys
from context_time_travel.store import Store
root,target,point,mode=sys.argv[1:]
def fault(step):
 if step==point:
  if mode=='kill':os.kill(os.getpid(),signal.SIGKILL)
  print('STAGED',flush=True)
  assert sys.stdin.readline()=='publish\n'
s=Store(root,fault_hook=fault);s.move('main',0,target,'interrupted')
"""
class ReferenceStageRecoveryTests(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
  self.store=Store(self.tmp.name);self.root=self.store.create_root('reference-stage-test',[1,2],{'draw':0})
  self.child=self.store.append(self.root,[3],{'draw':1});self.old=self.store.fork('main',self.root)
 def worker(self,point,mode='kill'):
  return subprocess.Popen([sys.executable,'-B','-c',WORKER,str(self.store.root),self.child,point,mode],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
 def test_restart_reclaims_sigkilled_reference_stage_and_retries_once(self):
  before=(self.store.references/'main.json').read_bytes();objects={x.name:x.read_bytes() for x in self.store.objects.glob('*.json')}
  p=self.worker('after_refs_stage_fsync');out,err=p.communicate(timeout=10);self.assertEqual(p.returncode,-signal.SIGKILL,err)
  stages=list(self.store.references.glob('.stage-*'));self.assertEqual(len(stages),1)
  self.assertEqual((self.store.references/'main.json').read_bytes(),before)
  reopened=Store(self.store.root)
  self.assertFalse(list(reopened.references.glob('.stage-*')),'restart left an abandoned branch-reference stage')
  self.assertEqual(reopened.head('main'),self.old)
  receipt=reopened.move('main',0,self.child,'interrupted');self.assertEqual(receipt,reopened.move('main',0,self.child,'interrupted'))
  self.assertEqual(receipt.revision,1);self.assertEqual(len(reopened.events()),2)
  self.assertEqual({x.name:x.read_bytes() for x in reopened.objects.glob('*.json')},objects)
 def test_restart_after_rename_preserves_committed_head_and_receipt(self):
  p=self.worker('after_refs_rename_before_dir_fsync');out,err=p.communicate(timeout=10);self.assertEqual(p.returncode,-signal.SIGKILL,err)
  before=(self.store.references/'main.json').read_bytes();reopened=Store(self.store.root)
  self.assertEqual(reopened.head('main').tip,self.child);self.assertEqual(reopened.head('main').revision,1)
  self.assertEqual(reopened.move('main',0,self.child,'interrupted'),reopened.head('main'))
  self.assertEqual((reopened.references/'main.json').read_bytes(),before)
 def test_constructor_waits_for_live_reference_writer_then_preserves_publication(self):
  writer=self.worker('after_refs_stage_fsync','wait');reader=None
  try:
   self.assertTrue(select.select([writer.stdout],[],[],5)[0]);self.assertEqual(writer.stdout.readline(),'STAGED\n')
   stages=list(self.store.references.glob('.stage-*'));self.assertEqual(len(stages),1);inode=stages[0].stat().st_ino
   code="import sys;from context_time_travel.store import Store;print('ATTEMPT',flush=True);s=Store(sys.argv[1]);print(s.head('main').tip,flush=True)"
   reader=subprocess.Popen([sys.executable,'-B','-c',code,str(self.store.root)],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
   self.assertTrue(select.select([reader.stdout],[],[],5)[0]);self.assertEqual(reader.stdout.readline(),'ATTEMPT\n')
   self.assertFalse(select.select([reader.stdout],[],[],.15)[0],'constructor bypassed the active reference transaction')
   self.assertEqual(stages[0].stat().st_ino,inode)
   out,err=writer.communicate('publish\n',timeout=10);self.assertEqual(writer.returncode,0,err)
   out,err=reader.communicate(timeout=10);self.assertEqual(reader.returncode,0,err);self.assertEqual(out.strip(),self.child)
   self.assertFalse(list(self.store.references.glob('.stage-*')))
  finally:
   for child in (writer,reader):
    if child is not None:
     if child.poll() is None:child.kill()
     child.communicate(timeout=5)
 def test_cleanup_preserves_nonstage_files_symlinks_directories_and_fifo(self):
  ref=self.store.references;outside=self.store.root/'outside';outside.write_bytes(b'keep')
  (ref/'user-file').write_bytes(b'keep');(ref/'.stage-link').symlink_to(outside)
  (ref/'.stage-directory').mkdir();os.mkfifo(ref/'.stage-fifo',0o600)
  stale=ref/'.stage-abandoned';stale.write_bytes(b'unpublished')
  reopened=Store(self.store.root)
  self.assertFalse(stale.exists(),'owned regular abandoned stage survived constructor recovery')
  self.assertTrue((ref/'.stage-link').is_symlink());self.assertEqual(outside.read_bytes(),b'keep')
  self.assertTrue((ref/'.stage-directory').is_dir());self.assertTrue((ref/'.stage-fifo').exists());self.assertEqual((ref/'user-file').read_bytes(),b'keep')
  self.assertEqual(reopened.head('main'),self.old)
if __name__=='__main__':unittest.main()
