"""Coordinator eviction recovery with actual parser and synthetic arithmetic."""
import unittest
import serve.test_cow_grammar_canonical_flow as fixture
class GrammarCoordinatorRebuildTests(unittest.TestCase):
 setUp=fixture.CowCanonicalGrammarFlowTests.setUp
 def test_evicted_active_grammar_resumes_without_mutating_saved_progress(self):
  model=self.coordinator._workers[0]['model']
  prepared=self.text.prepare('evicted',[{'role':'user','content':'read'}],'prepare',namespace_id=self.namespace)
  prepared.durable.result(5)
  head=self.coordinator.store.head('evicted')
  first=self.text.answer('evicted',head.revision,'first',1,grammar_state='READ')
  committed=first.durable.result(5)
  snapshot=committed['snapshot'];before=model.snapshot_bytes(snapshot)
  model.cache.clear()
  for key in list(self.coordinator.pool._entries):self.coordinator.pool.remove(key)
  release=getattr(model.backend,'release_checkpoints',None)
  if release:release()
  for path in model.checkpoint_directory.iterdir():
   if path.is_file():path.unlink()
  head=self.coordinator.store.head('evicted')
  request=self.text.resume('evicted',head.revision,'after-eviction',1)
  receipt=request.durable.result(5)
  self.assertEqual(model.snapshot_bytes(snapshot),before)
  tip=self.coordinator.store.head('evicted').tip
  self.assertEqual(model.history(tip)[-2:],(ord('A'),ord('B')))
  self.assertEqual(model.controller(tip)['accepted_in_scope'],2)
  self.assertNotEqual(tip,snapshot)
