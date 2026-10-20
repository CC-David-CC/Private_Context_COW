"""Real grammar parser + synthetic numerical backend; no GPU qualification."""
import unittest
import test_checkpoint_rebuild as baseline
from test_grammar_first import TinyByteTokenizer
from context_time_travel.grammar_first import GrammarEngine
from context_time_travel.store import IntegrityError,InvalidOperation
class GrammarCheckpointRebuildTests(unittest.TestCase):
 def setUp(self):
  baseline.CheckpointRebuildTests.setUp(self)
  self.tokenizer=TinyByteTokenizer()
  self.backend.vocab_size=len(self.tokenizer.tokens)
  self.model.profile['execution']['eos_tokens']=[256]
  self.engine=GrammarEngine(self.tokenizer,{'edit':{'format':'gbnf','source':'root ::= "AB"'}},eos_tokens=[256])
 def fixture(self,stopped=False):
  history=list(self.model.history(self.root))
  state=self.engine.start('edit','retained-stream',history,self.backend.vocab_size,sampling=dict(temperature=0.0,min_p=0.0,top_k=0,top_p=1.0))
  session=self.engine.restore(state,history)
  tokens=[65,66,256] if stopped else [65]
  for token in tokens:
   logits=[0.0]*self.backend.vocab_size;logits[token]=10.0
   self.assertEqual(session.choose(logits),token)
   state=session.consume(state,token,self.tokenizer.token_bytes(token),token==256)
  return self.model.store.append(self.root,tokens,state)
 def test_active_and_stopped_grammar_rebuild_preserve_canonical_progress(self):
  for stopped in (False,True):
   with self.subTest(stopped=stopped):
    snapshot=self.fixture(stopped);before=self.model.snapshot_bytes(snapshot)
    state=self.model.controller(snapshot);history=list(self.model.history(snapshot))
    expected=self.engine.restore(state,history).current_controller
    receipt=self.model.rebuild_materialization(snapshot,grammar_engine=self.engine)
    self.assertEqual(receipt['forward_tokens'],len(history))
    self.assertFalse(receipt['prior_computation_preserved'])
    self.assertEqual(self.model.snapshot_bytes(snapshot),before)
    self.assertEqual(self.model.controller(snapshot),state)
    self.assertEqual(self.engine.restore(state,history).current_controller,expected)
    self.assertEqual(tuple(history),self.backend.tokens)
 def test_wrong_grammar_rejected_before_native_work(self):
  snapshot=self.fixture()
  wrong=GrammarEngine(self.tokenizer,{'edit':{'format':'gbnf','source':'root ::= "AC"'}},eos_tokens=[256])
  with self.assertRaises((IntegrityError,InvalidOperation)):
   self.model.rebuild_materialization(snapshot,grammar_engine=wrong)
  self.assertEqual(self.backend.stats['forward_tokens'],0)
  self.assertNotIn(snapshot,self.model.cache)
 def test_missing_grammar_remains_explicit_before_native_work(self):
  snapshot=self.fixture()
  with self.assertRaises((IntegrityError,InvalidOperation)):
   self.model.rebuild_materialization(snapshot)
  self.assertEqual(self.backend.stats['forward_tokens'],0)
