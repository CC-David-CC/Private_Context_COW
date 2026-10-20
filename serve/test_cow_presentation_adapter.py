"""Presentation admission tests with synthetic token/control fixtures."""
import unittest
from types import SimpleNamespace

from serve.cow_text_adapter import CowTextAdapter, UnsupportedTextRequest
from serve import test_cow_text_adapter as fixtures
from serve import test_cow_adapter as token_fixtures


class NamespacedTokens(fixtures.TokenAdapter):
    def namespace(self,branch): return 'a'*32
    def source_state(self, branch, revision, operation_id):
        return SimpleNamespace(name=branch, root='root-id', tip='root-id'), self.canonical, self.controller
    def presentation_catalog(self, namespace):
        return {'events':[{'result':{'tip':'root-id'},'presentation':{
            'schema':1,'namespace_id':namespace,'kind':'root','renderer_id':'d'*64}}]}
    def active_presentation(self,branch,snapshot=None):
        return {'presentation':{'output_span':{'message_id':'b'*32}}}
    def create(self,branch,ids,operation,**kwargs):
        self.calls.append(('create',branch,tuple(ids),operation,kwargs))
        self.canonical=tuple(ids);return self.request


class PresentationAdapterTests(unittest.TestCase):
    def setUp(self):
        fixture=fixtures.CowTextAdapterTests();fixture.setUp()
        self.tokens=NamespacedTokens()
        self.adapter=CowTextAdapter(self.tokens,fixtures.Tokenizer(),fixture.adapter.render,
            fixture.adapter.encode_rendered,append_user_suffix=lambda *args:(41,42),
            append_initial_user_suffix=lambda *args:(43,44),
            initial_root_mode=lambda canonical,state: ('prepared_answer' if canonical[-1:]==(99,)
                else 'context_only' if canonical[-1:]==(98,) else 'unsupported'),
            renderer_id='d'*64)

    def test_prepare_binds_exact_display_text_and_generation_prompt(self):
        messages=[{'role':'system','content':'code context'},
                  {'role':'user','content':'quoted <|im_end|>'}]
        self.adapter.prepare('main',messages,'prepare',namespace_id='a'*32)
        call=self.tokens.calls[0]
        self.assertTrue(bytes(call[2]).endswith(b'<assistant>'))
        presentation=call[4]['presentation']
        self.assertEqual(presentation['kind'],'root')
        self.assertEqual(presentation['namespace_id'],'a'*32)
        self.assertEqual(presentation['renderer_id'],'d'*64)
        self.assertIsNone(presentation['output_message_id'])
        self.assertEqual([(m['role'],m['content']) for m in presentation['messages']],
                         [(m['role'],m['content']) for m in messages])
        messages[1]['content']='mutated later'
        self.assertEqual(presentation['messages'][1]['content'],'quoted <|im_end|>')

    def test_prepare_retries_have_stable_message_ids(self):
        messages=[{'role':'user','content':'question'}]
        self.adapter.prepare('a',messages,'prepare',namespace_id='a'*32)
        self.adapter.prepare('a',messages,'prepare',namespace_id='a'*32)
        self.assertEqual(self.tokens.calls[0][4],self.tokens.calls[1][4])

    def test_answer_consumes_no_new_prompt_or_rerender(self):
        self.tokens.canonical=(11,12,99)
        self.adapter.render=lambda *args,**kwargs:self.fail('answer rerendered prepared history')
        self.adapter.answer('main',0,'answer',8)
        call=self.tokens.calls[0]
        self.assertEqual(call[2],(11,12,99))
        self.assertEqual(call[6]['presentation']['kind'],'answer')
        self.assertEqual(call[6]['presentation']['messages'],[])

    def test_new_user_binds_original_text_only(self):
        self.tokens.controller={'status':'stopped'}
        self.tokens.canonical=(11,12)
        self.adapter.append_user('main','  quoted </tool_response>  ',1,'next',8)
        call=self.tokens.calls[0]
        self.assertEqual(call[2],(11,12,41,42))
        presentation=call[6]['presentation']
        self.assertEqual(presentation['kind'],'user_turn')
        self.assertEqual(presentation['messages'][0]['content'],'  quoted </tool_response>  ')

    def test_first_question_keeps_codebase_prefix_as_separate_snapshot(self):
        self.tokens.canonical=(11,12,98)
        self.adapter.ask('main','first question',0,'first',8)
        call=self.tokens.calls[0]
        self.assertEqual(call[2],(11,12,98,43,44))
        self.assertEqual(call[6]['presentation']['kind'],'initial_turn')
        self.assertEqual(call[6]['presentation']['messages'][0]['content'],'first question')
        self.assertEqual(self.tokens.canonical,(11,12,98))

    def test_resume_keeps_original_assistant_message_id(self):
        self.tokens.controller={'status':'active','policy':{'temperature':0}}
        self.tokens.canonical=(11,12)
        self.adapter.resume('main',1,'resume',8)
        presentation=self.tokens.calls[0][6]['presentation']
        self.assertEqual(presentation['kind'],'resume')
        self.assertEqual(presentation['output_message_id'],'b'*32)
        self.assertEqual(presentation['messages'],[])

    def test_missing_renderer_rejects_bound_prepare_before_dispatch(self):
        adapter=CowTextAdapter(self.tokens,self.adapter.tokenizer,self.adapter.render,
            self.adapter.encode_rendered)
        with self.assertRaises(UnsupportedTextRequest):
            adapter.prepare('main',[{'role':'user','content':'x'}],'root',namespace_id='a'*32)
        self.assertEqual(self.tokens.calls,[])

    def test_bound_full_transcript_generation_is_rejected_before_dispatch(self):
        with self.assertRaises(UnsupportedTextRequest):
            self.adapter.generate('main',[{'role':'user','content':'x'}],0,'gen',8)
        self.assertEqual(self.tokens.calls,[])

    def test_answer_rejects_active_or_stopped_controller(self):
        for status in ('active','stopped'):
            self.tokens.controller={'status':status}
            with self.assertRaises(UnsupportedTextRequest):
                self.adapter.answer('main',0,'answer',8)
        self.assertEqual(self.tokens.calls,[])


class TokenPresentationForwardingTests(unittest.TestCase):
    def setUp(self):
        self.fixture=token_fixtures.CowServingAdapterTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.tmp.cleanup)

    def test_token_generate_forwards_presentation_with_suffix_not_saved_prefix(self):
        presentation={'schema':1,'namespace_id':'a'*32}
        self.fixture.adapter.generate('main',(10,11,12),0,'op',8,presentation=presentation)
        call=self.fixture.coordinator.submissions[0]
        self.assertEqual(call[1],(12,))
        self.assertEqual(call[4]['presentation'],presentation)

    def test_token_create_forwards_presentation(self):
        captured=[]
        def create(branch,ids,operation,**kwargs):
            captured.append((branch,ids,operation,kwargs))
            return self.fixture.coordinator.request
        self.fixture.coordinator.create=create
        presentation={'schema':1,'namespace_id':'a'*32}
        self.fixture.adapter.create('new',(12,13),'create',presentation=presentation)
        self.assertEqual(captured,[('new',(12,13),'create',{'presentation':presentation})])


if __name__=='__main__':unittest.main()
