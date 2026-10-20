"""Pause uses a normal consumed partial snapshot, never an abort acknowledgment."""
from concurrent.futures import CancelledError
import unittest
from context_time_travel.store import InvalidOperation
import test_vision_grammar_request as fixtures

class ConsumedPauseTests(unittest.TestCase):
 def setUp(self):
  self.fixture=fixtures.VisionGrammarRequestTests();self.fixture.setUp();self.addCleanup(self.fixture.doCleanups)
  self.model=self.fixture.model;self.root=self.fixture.root
 def test_early_pause_consumes_one_token_and_resumes_exact_grammar(self):
  expected=self.fixture.request()
  parent=self.model.snapshot_bytes(self.root)
  paused=self.fixture.request(paused=lambda:True)
  self.assertEqual(paused['tokens'],expected['tokens'][:1])
  self.assertFalse(paused['stopped']);self.assertFalse(paused['grammar_complete'])
  self.assertEqual(self.model.consumed(paused['snapshot']),self.model.consumed(self.root)+1)
  resumed=self.fixture.request(paused['snapshot'],resume=True)
  self.assertEqual(paused['tokens']+resumed['tokens'],expected['tokens'])
  self.assertEqual(self.model.controller(resumed['snapshot']),self.model.controller(expected['snapshot']))
  self.assertEqual(self.model.next_logits(resumed['snapshot']),self.model.next_logits(expected['snapshot']))
  self.assertEqual(self.model.snapshot_bytes(self.root),parent)
 def test_pause_after_second_consumed_token_retains_parser_progress(self):
  seen=[]
  paused=self.fixture.request(on_token=seen.append,paused=lambda:len(seen)>=2)
  self.assertEqual(paused['tokens'],seen);self.assertEqual(len(seen),2)
  resumed=self.fixture.request(paused['snapshot'],resume=True)
  self.assertEqual(resumed['tokens'],[6]);self.assertTrue(resumed['stopped'])
 def test_no_pause_preserves_normal_completion(self):
  actual=self.fixture.request(paused=lambda:False);self.assertEqual(actual['tokens'],[0,0,6]);self.assertTrue(actual['stopped'])
 def test_eos_and_pause_at_same_boundary_keep_stopped_controller(self):
  seen=[];answer=self.fixture.request(on_token=seen.append,paused=lambda:len(seen)==3)
  self.assertEqual(answer['tokens'],[0,0,6]);self.assertTrue(answer['stopped']);self.assertTrue(answer['grammar_complete'])
 def test_noncallable_pause_rejected_before_forward_work(self):
  before=self.fixture.backend.stats['forward_tokens']
  with self.assertRaises(InvalidOperation):self.fixture.request(paused=True)
  self.assertEqual(before,self.fixture.backend.stats['forward_tokens'])
 def test_pause_callback_failure_does_not_publish_or_change_parent(self):
  before=set(self.model.store.objects.iterdir());parent=self.model.snapshot_bytes(self.root)
  def bad():raise RuntimeError('pause observer failed')
  with self.assertRaisesRegex(RuntimeError,'pause observer failed'):self.fixture.request(paused=bad)
  self.assertEqual(set(self.model.store.objects.iterdir()),before);self.assertEqual(parent,self.model.snapshot_bytes(self.root))
  self.assertEqual(self.fixture.request()['tokens'],[0,0,6])
 def test_destructive_cancellation_remains_distinct_and_wins(self):
  seen=[];before=set(self.model.store.objects.iterdir())
  with self.assertRaises(CancelledError):
   self.fixture.request(on_token=seen.append,paused=lambda:True,cancelled=lambda:bool(seen))
  self.assertEqual(set(self.model.store.objects.iterdir()),before)

class SampledPauseTests(unittest.TestCase):
 def setUp(self):
  import test_vision_sampling_request as sampling
  self.sampling=sampling;self.fixture=sampling.VisionSamplingRequestTests()
  self.fixture.setUp();self.addCleanup(self.fixture.doCleanups)
 def test_grammar_pause_restores_nonzero_rng_and_parser_progress(self):
  f=self.fixture;expected=f.request(grammar=True,steps=16);seen=[]
  partial=f.request(grammar=True,steps=16,on_token=seen.append,paused=lambda:len(seen)>=3)
  saved=f.model.controller(partial['snapshot']);self.assertEqual(saved['next_draw'],3)
  tail=f.request(partial['snapshot'],grammar=True,resume=True,steps=16)
  self.assertEqual(partial['tokens']+tail['tokens'],expected['tokens'])
  self.assertEqual(f.model.controller(tail['snapshot']),f.model.controller(expected['snapshot']))
  self.assertEqual(f.model.controller(partial['snapshot']),saved)
 def test_plain_sampled_wrapper_preserves_draw_cursor(self):
  f=self.fixture
  from context_time_travel.seed_controls import seed_stream
  options=dict(sampling=self.sampling.POLICY,stream=seed_stream(42))
  expected=f.model.run_sampled_request(f.root,(),8,**options);seen=[]
  partial=f.model.run_sampled_request(f.root,(),8,**options,on_token=seen.append,paused=lambda:len(seen)>=3)
  tail=f.model.run_sampled_request(partial['snapshot'],(),5,**options,resume=True)
  self.assertEqual(partial['tokens']+tail['tokens'],expected['tokens'])
  self.assertEqual(f.model.controller(tail['snapshot']),f.model.controller(expected['snapshot']))
