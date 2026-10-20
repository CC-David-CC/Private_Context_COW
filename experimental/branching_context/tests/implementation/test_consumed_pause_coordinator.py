"""Real coordinator/store/controller with a synthetic CPU backend, not GPU."""
import threading,unittest
from context_time_travel.coordinator import ContextCoordinator
from context_time_travel.pools import PoolConfig
import test_vision_sampling_request as fixtures

class PauseCoordinatorTests(unittest.TestCase):
 def test_partial_pause_commits_and_resumes_same_random_sequence(self):
  f=fixtures.VisionSamplingRequestTests();f.setUp();self.addCleanup(f.doCleanups)
  model=f.model;model.profile['execution']['GPU']=[0]
  expected=f.request(steps=8)
  model.store.fork('main',f.root);model.checkpoint(f.root)
  coordinator=ContextCoordinator(str(model.store.root),PoolConfig(gpus=(0,),ram_bytes=1024**2,disk_bytes=1024**2,min_available_bytes=0),lambda _:model)
  self.addCleanup(coordinator.close);coordinator.ready(5)
  consumed=threading.Event();release=threading.Event();self.addCleanup(release.set)
  original=f.backend.consume_generated
  def gate(*args):
   original(*args);consumed.set()
   if not release.wait(5):raise TimeoutError('test gate not released')
  f.backend.consume_generated=gate
  job=coordinator.submit('main',(),8,'first',expected_revision=0,sampling=fixtures.POLICY,seed=42)
  self.assertTrue(consumed.wait(5));self.assertTrue(job.pause_supported)
  self.assertTrue(job.request_pause());self.assertFalse(job.durable.done());release.set()
  receipt=job.durable.result(5);answer=job.snapshot.result(5)
  self.assertEqual(len(answer['tokens']),1);self.assertFalse(answer['stopped'])
  self.assertEqual(answer['counter_deltas']['forward_tokens'],1)
  saved=model.controller(receipt['snapshot']);self.assertEqual(saved['next_draw'],1)
  self.assertEqual(coordinator.store.head('main').tip,receipt['snapshot'])
  # Lost-ack replay returns that exact short result, not the original8-token ceiling.
  before=f.backend.stats['forward_tokens']
  replay=coordinator.submit('main',(),8,'first',expected_revision=0,sampling=fixtures.POLICY,seed=42)
  self.assertEqual(replay.durable.result(5),receipt)
  self.assertEqual(replay.snapshot.result(5)['tokens'],answer['tokens'])
  self.assertEqual(f.backend.stats['forward_tokens'],before)
  # Future completion is the normal durable publication, not the pause ack.
  self.assertFalse(job.request_pause())
  tail=coordinator.submit('main',(),7,'tail',expected_revision=1,sampling=fixtures.POLICY,resume=True)
  final=tail.durable.result(5);result=tail.snapshot.result(5)
  self.assertEqual(answer['tokens']+result['tokens'],expected['tokens'])
  self.assertEqual(model.controller(final['snapshot']),model.controller(expected['snapshot']))
  self.assertEqual(model.controller(receipt['snapshot']),saved)
