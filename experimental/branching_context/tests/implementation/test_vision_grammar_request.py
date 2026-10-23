"""Mixed-history GBNF ownership; synthetic backend, real llguidance parser."""
from concurrent.futures import CancelledError
import tempfile
import unittest

from context_time_travel.grammar_first import GrammarEngine
from context_time_travel.input_spans import InputHistory
from context_time_travel.real import BranchingModel
from context_time_travel.store import IntegrityError, InvalidOperation
from tools.strata_tokenizer import Tokenizer
from test_vision_api_binding import MixedCPUBackend
from test_vision_execution import receipt, spans
from test_vision_input_delta import DeltaArtifacts


class MixedGrammarBackend(MixedCPUBackend):
    vocab_size=8

    def __init__(self,profile,tokenizer):
        super().__init__(profile);self.tokenizer=tokenizer;self.restores=0;self.decode_calls=0

    def piece(self,token):return self.tokenizer.token_bytes(token)
    def is_eog(self,token):return token==6
    def validate_grammar_engine(self,engine):
        if engine.tokenizer_sha256!=GrammarEngine(self.tokenizer,{
                'check':dict(format='gbnf',source='root ::= "A"')},[6]).tokenizer_sha256:
            raise InvalidOperation('Grammar tokenizer mismatch')

    def restore(self,*args,**options):
        self.restores+=1;return super().restore(*args,**options)

    def consume_generated(self,lease,tokens):
        start=len(lease['inputs']);self.consume(lease,tokens);self.decode_calls+=1
        self.last_execution_receipt=receipt(lease['inputs'],position=start,operator='normal-verifier-decode')


class VisionGrammarRequestTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.tokenizer=Tokenizer(['A','B','C','D','E','F','<eos>','<image>'],[],
            [1,1,1,1,1,1,3,3],special_ids={'eos':6})
        self.engine=GrammarEngine(self.tokenizer,{
            'EDIT':dict(format='gbnf',source='root ::= "AA"'),
            'TEST':dict(format='gbnf',source='root ::= "B"')},[6])
        self.profile=dict(model_contract_id='mixed-model',execution_profile_id='mixed-grammar',
            model_weights_sha256='a'*64,execution=dict(resume_contract='identical-mixed-native-operators-v1',
                prefill_chunk=128,embedding_width=4,capacity=128,eos_tokens=[6],
                generated_operator='normal-verifier-decode',selected_token_protocol='consumed-selected-receipt-v1'))
        self.backend=MixedGrammarBackend(self.profile,self.tokenizer)
        self.model=BranchingModel(self.temp.name,self.backend,self.profile)
        self.artifacts=DeltaArtifacts(self.model.store)
        self.root=self.model.create_inputs(spans(),artifacts=self.artifacts)

    def request(self,snapshot=None,ids=(),steps=8,**options):
        return self.model.run_state_request(snapshot or self.root,ids,steps,
            engine=self.engine,state=options.pop('state','EDIT'),stream='image-grammar',**options)

    def test_grammar_masks_before_greedy_and_counts_text_not_image_rows(self):
        before=self.backend.stats['forward_tokens'];parent=self.model.snapshot_bytes(self.root)
        result=self.request(ids=[4]);self.assertEqual(result['tokens'],[0,0,6])
        state=self.model.controller(result['snapshot'])
        self.assertEqual(state['constraint']['scope_start'],4)
        self.assertEqual(state['seen'],{'1':1,'2':1,'3':1,'4':1,'0':2,'6':1})
        self.assertEqual(state['next_draw'],0);self.assertTrue(result['grammar_complete']);self.assertTrue(result['stopped'])
        self.assertEqual(self.model.history(result['snapshot']),InputHistory(spans())+(4,0,0,6))
        self.assertEqual(self.backend.stats['forward_tokens']-before,4)
        self.assertEqual(self.backend.decode_calls,3);self.assertEqual(self.model.snapshot_bytes(self.root),parent)

    def test_partial_scope_restores_using_text_scope_not_native_cell_count(self):
        partial=self.request(steps=1);saved=self.model.controller(partial['snapshot'])
        self.assertEqual((saved['constraint']['scope_start'],saved['accepted_in_scope']),(3,1))
        resumed=self.request(partial['snapshot'],resume=True)
        self.assertEqual(resumed['tokens'],[0,6]);self.assertTrue(resumed['stopped'])
        self.assertEqual(self.model.controller(partial['snapshot']),saved)
        full=self.request()
        self.assertEqual(self.model.history(resumed['snapshot']),self.model.history(full['snapshot']))
        self.assertEqual(self.model.controller(resumed['snapshot']),self.model.controller(full['snapshot']))

    def test_branch_a_can_enter_test_while_b_retains_edit_progress(self):
        partial=self.request(steps=1);sid=partial['snapshot']
        a=self.model.store.fork('A',sid);b=self.model.store.fork('B',sid)
        retained=self.model.snapshot_bytes(b.tip);state_b=self.model.controller(b.tip)
        done=self.request(a.tip,resume=True)
        testing=self.request(done['snapshot'],state='TEST',steps=1)
        self.model.store.move(a.name,a.revision,testing['snapshot'],'A-test')
        self.assertEqual(self.model.controller(testing['snapshot'])['constraint']['agent_state'],'TEST')
        self.assertEqual(self.model.store.head('B'),b);self.assertEqual(self.model.snapshot_bytes(b.tip),retained)
        self.assertEqual(self.model.controller(b.tip),state_b)
        self.assertEqual(self.request(b.tip,resume=True)['tokens'],[0,6])

    def test_image_suffix_is_leased_and_does_not_enter_grammar_scope(self):
        delta=InputHistory(spans(original_sha256='e'*64,embedding_sha256='f'*64))
        before=self.backend.stats['forward_tokens'];capture=self.model._save
        def checked_capture(*args,**options):
            self.assertEqual(self.artifacts.active,1);return capture(*args,**options)
        self.model._save=checked_capture
        result=self.request(ids=delta,artifacts=self.artifacts)
        self.assertEqual(self.backend.stats['forward_tokens']-before,len(delta)+3)
        state=self.model.controller(result['snapshot'])
        self.assertEqual(state['constraint']['scope_start'],6);self.assertNotIn('7',state['seen'])
        self.assertEqual(self.model.history(result['snapshot']),InputHistory(spans())+delta+(0,0,6))
        self.assertEqual(self.artifacts.active,0)

    def test_corrupted_repetition_progress_rejects_before_restore(self):
        partial=self.request(steps=1);state=self.model.controller(partial['snapshot'])
        state['seen']['7']=6;bad=self.model._controller_transition(partial['snapshot'],state)
        restores=self.backend.restores;work=self.backend.stats['forward_tokens']
        with self.assertRaises(IntegrityError):self.request(bad,resume=True)
        self.assertEqual(self.backend.restores,restores);self.assertEqual(self.backend.stats['forward_tokens'],work)

    def test_resume_cannot_change_state_or_add_an_image(self):
        partial=self.request(steps=1);restores=self.backend.restores
        for options in (dict(state='TEST'),dict(ids=InputHistory(spans()),artifacts=self.artifacts)):
            with self.subTest(options=options),self.assertRaises(InvalidOperation):
                self.request(partial['snapshot'],resume=True,**options)
        self.assertEqual(self.backend.restores,restores)

    def test_missing_embedding_rejects_before_restore(self):
        restores=self.backend.restores;work=self.backend.stats['forward_tokens']
        with self.assertRaises(IntegrityError):
            self.request(ids=InputHistory(spans()),artifacts=DeltaArtifacts(self.model.store,allowed=set()))
        self.assertEqual(self.backend.restores,restores);self.assertEqual(self.backend.stats['forward_tokens'],work)

    def test_cancelled_progress_does_not_publish_and_parent_remains_usable(self):
        before=set(self.model.store.objects.iterdir());cancel=[]
        with self.assertRaises(CancelledError):
            self.request(on_token=lambda token:cancel.append(token),cancelled=lambda:bool(cancel))
        self.assertEqual(cancel,[0]);self.assertEqual(set(self.model.store.objects.iterdir()),before)
        self.assertEqual(self.request()['tokens'],[0,0,6])

    def test_wrong_tokenizer_or_protocol_rejects_before_restore(self):
        restores=self.backend.restores
        self.model.profile['execution'].pop('selected_token_protocol')
        with self.assertRaises(InvalidOperation):self.request()
        self.assertEqual(self.backend.restores,restores)

    def test_unqualified_image_sampling_is_not_enabled_by_grammar_change(self):
        with self.assertRaisesRegex(InvalidOperation,'Mixed-input.*greedy'):
            self.request(sampling=dict(temperature=.7,min_p=.05,top_k=0,top_p=1.0))
        self.assertEqual(self.backend.restores,0)

    def test_saved_sampled_controller_cannot_enter_greedy_image_resume(self):
        state=self.engine.start('EDIT','sampled',self.model.history(self.root).text_tokens,8,
            sampling=dict(temperature=.7,min_p=.05,top_k=0,top_p=1.0))
        sid=self.model._controller_transition(self.root,state)
        with self.assertRaisesRegex(InvalidOperation,'Mixed-input.*greedy'):
            self.request(sid,resume=True)
        self.assertEqual(self.backend.restores,0)

    def test_wrong_tokenizer_rejects_before_native_restore(self):
        tokenizer=Tokenizer(['Z','B','C','D','E','F','<eos>','<image>'],[],
            [1,1,1,1,1,1,3,3],special_ids={'eos':6})
        foreign=GrammarEngine(tokenizer,{'EDIT':dict(format='gbnf',source='root ::= "Z"')},[6])
        with self.assertRaisesRegex(InvalidOperation,'tokenizer mismatch'):
            self.model.run_state_request(self.root,(),8,engine=foreign,state='EDIT',stream='bad')
        self.assertEqual(self.backend.restores,0)


if __name__=='__main__':unittest.main()
