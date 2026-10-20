"""Mixed client controls over real Store with synthetic CPU inference/encoder."""
from contextlib import contextmanager
import io
from pathlib import Path
import tempfile
import unittest

from PIL import Image
from context_time_travel.coordinator import ContextCoordinator
from context_time_travel.input_spans import InputHistory
from context_time_travel.pools import PoolConfig
from context_time_travel.real import BranchingModel
from context_time_travel.store import Store, InvalidOperation
from serve.cow_adapter import CowAdapter
from serve.cow_prompt_encoder import CowPromptEncoder
from serve.cow_text_adapter import CowTextAdapter, UnsupportedTextRequest
from serve.cow_vision import CowImageNormalizer, CowVisionPromptEncoder
from serve.cow_vision_preparer import CowVisionPreparer
from serve.cow_workspace import CowWorkspace
from serve.frontend import ChatTemplate
from serve.server import ByteTokenizer
from serve.test_cow_vision_preparer import EncoderFixture
from test_vision_api_binding import MixedCPUBackend
from test_vision_execution import receipt

ROOT=Path(__file__).resolve().parents[1]


class TextTokenizer(ByteTokenizer):
    def token_bytes(self,token):
        return bytes([token]) if token<256 else self.SPECIALS[token-256].encode()


class TextCPUBackend(MixedCPUBackend):
    vocab_size=512;next_token=120
    def piece(self,token):return TextTokenizer().token_bytes(token)
    def is_eog(self,token):return token==257
    def logits(self,lease):return tuple(float(token==self.next_token) for token in range(self.vocab_size))
    @contextmanager
    def generate_request(self,lease,count):
        start=len(lease['inputs'])
        def tokens():
            for _ in range(count):
                token=self.next_token;self.consume(lease,[token]);yield token
                if self.is_eog(token):break
        try:yield tokens()
        finally:self.last_execution_receipt=receipt(lease['inputs'],position=start,operator='normal-verifier-decode')


class CowVisionTextFlowTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.store=Store(self.temp.name);self.namespace=self.store.create_namespace('Image chat','namespace').namespace_id
        self.tokenizer=TextTokenizer();self.encoder=EncoderFixture();self.models=[];self.manager=None
        self.prompt=CowPromptEncoder(self.tokenizer,ChatTemplate(ROOT/'serve/chat_template.jinja'),{'enable_thinking':False})
        self.mixed=CowVisionPromptEncoder(self.prompt,renderer_id='c'*64)
        self.preparer=CowVisionPreparer(self.store,CowImageNormalizer(),self.mixed,self.encoder,original_bytes_limit=16*1024**2)
        self.addCleanup(self.close);self.start()
        stream=io.BytesIO();Image.new('RGB',(3,2),'red').save(stream,format='PNG')
        self.upload=self.preparer.upload(self.namespace,stream.getvalue(),pool=self.manager.pool,disk=self.manager.disk)
        self.messages=[dict(role='user',content=[dict(type='image',image_id=self.upload['image_id']),dict(type='text',text='diagram')])]

    def start(self):
        profile=dict(model_contract_id='mixed-model',execution_profile_id='mixed-text',model_weights_sha256='a'*64,
            execution=dict(GPU=[0],resume_contract='identical-mixed-native-operators-v1',generated_operator='normal-verifier-decode',
                native_request_protocol='consumed-greedy-ack-v1',prefill_chunk=128,embedding_width=4,capacity=2048,eos_tokens=[257]))
        def factory(gpu):
            model=BranchingModel(self.temp.name,TextCPUBackend(profile),profile);self.models.append(model);return model
        self.manager=ContextCoordinator(self.temp.name,PoolConfig(gpus=(0,),ram_bytes=1024**3,disk_bytes=16*1024**2,
            min_available_bytes=0),factory,input_preparer=None if getattr(self,'without_encoder',False) else self.preparer);self.manager.ready(5)
        self.tokens=CowAdapter(self.manager)
        self.text=CowTextAdapter(self.tokens,self.tokenizer,self.prompt.render,self.prompt.encode_rendered,
            append_user_suffix=self.prompt.append_user_suffix,append_initial_user_suffix=self.prompt.append_initial_user_suffix,
            initial_root_mode=self.prompt.initial_root_mode,renderer_id='c'*64,
            vision_preparer=None if getattr(self,'without_encoder',False) else self.preparer,mixed_renderer=self.mixed)
        self.workspace=CowWorkspace(self.store,self.tokenizer,root_mode_resolver=self.text.root_mode_for_projection)

    def close(self):
        if self.manager is not None:self.manager.close();self.manager=None

    def prepare(self):return self.text.prepare('main',self.messages,'prepare',namespace_id=self.namespace).durable.result(5)

    def test_plain_text_root_replay_after_restart_preserves_canonical_span_identity(self):
        messages=[dict(role='system',content='A small recorded project context.')]
        expected=self.text.create('text-root',messages,'text-root-op',
                                  namespace_id=self.namespace).durable.result(5)
        self.close();self.start()
        before=dict(self.models[-1].backend.stats)
        replay=self.text.create('text-root',messages,'text-root-op',namespace_id=self.namespace)
        self.assertEqual(replay.durable.result(5),expected)
        self.assertEqual(replay.snapshot.result(5)['tokens'],[])
        self.assertEqual(self.models[-1].backend.stats,before)

    def test_prepare_answer_and_workspace_keep_image_history(self):
        root=self.prepare();before=self.models[-1].backend.stats['forward_tokens']
        request=self.text.answer('main',1,'answer',2);request.durable.result(5)
        self.assertEqual(request.text_result(),'xx')
        self.assertEqual(self.models[-1].backend.stats['forward_tokens']-before,2)
        view=self.workspace.history('main')
        self.assertEqual(view['messages'][-1]['content'],'xx')
        self.assertEqual(view['image_inputs'][0]['original_sha256'],self.upload['image_id'])
        self.assertIsInstance(self.models[-1].history(root['snapshot']),InputHistory)

    def test_context_only_root_asks_without_reencoding_images(self):
        root=self.text.create('main',self.messages,'context',namespace_id=self.namespace).durable.result(5)
        self.assertEqual(self.workspace.history('main')['initial_root_mode'],'context_only')
        request=self.text.ask('main','What is shown?',1,'ask',2);request.durable.result(5)
        self.assertEqual(request.text_result(),'xx');self.assertEqual(len(self.encoder.calls),1)
        history=self.tokens.history('main')
        self.assertEqual(len([span for span in history.spans if span['kind']=='image']),1)

    def test_active_resume_preserves_assistant_identity_and_text(self):
        self.prepare();self.text.answer('main',1,'answer',2).durable.result(5)
        request=self.text.resume('main',2,'resume',2);request.durable.result(5)
        self.assertEqual(request.text_result(),'xx')
        assistants=[row for row in self.workspace.history('main')['messages'] if row['role']=='assistant']
        self.assertEqual(len(assistants),1);self.assertEqual(assistants[0]['content'],'xxxx')

    def test_next_user_after_eos_appends_only_text_and_preserves_parent(self):
        self.prepare();self.models[-1].backend.next_token=257
        self.text.answer('main',1,'answer',2).durable.result(5)
        parent=self.tokens.history('main');self.models[-1].backend.next_token=120
        request=self.text.append_user('main','literal <|image_pad|>',2,'next',2);request.durable.result(5)
        child=self.tokens.history('main');self.assertEqual(child.segment(0,len(parent)),parent)
        self.assertEqual(len([span for span in child.spans if span['kind']=='image']),1)

    def test_same_answer_retry_replays_after_reopen_without_forward(self):
        self.prepare();self.text.answer('main',1,'answer',2).durable.result(5)
        self.close();self.start();before=self.models[-1].backend.stats['forward_tokens']
        value=self.text.retry('answer','main',2,'answer-retry')
        self.assertTrue(value['replayed']);self.assertEqual(value['answer']['text'],'xx')
        self.assertEqual(self.models[-1].backend.stats['forward_tokens'],before)

    def test_workspace_graph_fork_and_rewind_are_image_safe(self):
        root=self.prepare();self.text.answer('main',1,'answer',2).durable.result(5)
        self.tokens.fork('alternative',root['snapshot'],source_branch='main',operation_id='fork')
        graph=self.workspace.graph(self.namespace)
        self.assertEqual(len(graph['branches']),2)
        self.tokens.rewind('main',2,root['snapshot'],'rewind')
        self.assertEqual(self.workspace.history('main')['initial_root_mode'],'prepared_answer')

    def test_sampler_grammar_seed_and_stops_remain_explicitly_unsupported(self):
        self.prepare();before=self.models[-1].backend.stats['forward_tokens']
        for options in (dict(settings=dict(temperature=.7)),dict(grammar_state='READ'),dict(seed=7),dict(text_stops=['END'])):
            with self.subTest(options=options),self.assertRaises(UnsupportedTextRequest):
                self.text.answer('main',1,'unsupported',2,**options)
        self.assertEqual(self.models[-1].backend.stats['forward_tokens'],before)

    def test_text_only_root_on_mixed_profile_has_compatible_projection(self):
        messages=[dict(role='user',content='plain question')]
        self.text.prepare('main',messages,'prepare',namespace_id=self.namespace).durable.result(5)
        self.assertEqual(self.workspace.history('main')['initial_root_mode'],'prepared_answer')
        self.text.answer('main',1,'answer',2).durable.result(5)

    def test_disk_continuation_needs_no_encoder_after_reopen(self):
        self.prepare();self.close();self.without_encoder=True;self.start()
        calls=len(self.encoder.calls)
        request=self.text.answer('main',1,'cold-answer',2);request.durable.result(5)
        self.assertEqual(request.text_result(),'xx');self.assertEqual(len(self.encoder.calls),calls)
        self.assertEqual(self.models[-1].backend.stats['forward_tokens'],2)

    def test_changed_retry_and_unconfigured_renderer_reject_before_forward(self):
        self.prepare();self.text.answer('main',1,'answer',2).durable.result(5)
        before=self.models[-1].backend.stats['forward_tokens']
        with self.assertRaises(UnsupportedTextRequest):self.text.retry('answer','main',2,'changed',settings=dict(temperature=.7))
        self.text.mixed_renderer=None
        with self.assertRaises(UnsupportedTextRequest):self.text.retry('answer','main',2,'wrong-renderer')
        self.assertEqual(self.models[-1].backend.stats['forward_tokens'],before)

    def test_workspace_reads_preserve_history_and_perform_no_forward(self):
        self.prepare();self.text.answer('main',1,'answer',2).durable.result(5)
        before={str(path):path.read_bytes() for path in self.store.objects.glob('*.json')}
        forward=self.models[-1].backend.stats['forward_tokens']
        self.workspace.history('main');self.workspace.graph(self.namespace);self.workspace.list_chats()
        after={str(path):path.read_bytes() for path in self.store.objects.glob('*.json')}
        self.assertEqual(before,after);self.assertEqual(self.models[-1].backend.stats['forward_tokens'],forward)

    def test_backend_violating_eos_cannot_publish_without_presentation(self):
        root=self.prepare();self.manager.fork('raw',root['snapshot']);backend=self.models[-1].backend
        @contextmanager
        def invalid_output(lease,count):
            start=len(lease['inputs'])
            def tokens():
                for _ in range(count):backend.consume(lease,[257]);yield 257
            try:yield tokens()
            finally:backend.last_execution_receipt=receipt(lease['inputs'],position=start,operator='normal-verifier-decode')
        backend.generate_request=invalid_output
        request=self.manager.submit('raw',(),2,'invalid-native')
        with self.assertRaises(InvalidOperation):request.durable.result(5)
        self.assertEqual(self.store.head('raw').tip,root['snapshot'])

    def test_backend_overrunning_output_bound_cannot_publish(self):
        root=self.prepare();self.manager.fork('raw',root['snapshot']);backend=self.models[-1].backend
        @contextmanager
        def invalid_output(lease,count):
            start=len(lease['inputs'])
            def tokens():
                for _ in range(count+1):backend.consume(lease,[120]);yield 120
            try:yield tokens()
            finally:backend.last_execution_receipt=receipt(lease['inputs'],position=start,operator='normal-verifier-decode')
        backend.generate_request=invalid_output
        request=self.manager.submit('raw',(),2,'invalid-native')
        with self.assertRaises(InvalidOperation):request.durable.result(5)
        self.assertEqual(self.store.head('raw').tip,root['snapshot'])


if __name__=='__main__':unittest.main()
