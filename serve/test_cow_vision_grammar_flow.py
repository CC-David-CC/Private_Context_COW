"""Image GBNF client/store integration: real parser and CPU control backend."""
from contextlib import contextmanager
import io
import json
from pathlib import Path
import tempfile
import threading
import unittest

from PIL import Image
from context_time_travel.coordinator import ContextCoordinator
from context_time_travel.grammar_first import GrammarEngine
from context_time_travel.pools import PoolConfig
from context_time_travel.real import BranchingModel
from context_time_travel.store import Store,IntegrityError,_json_bytes
from serve.cow_adapter import CowAdapter
from serve.cow_grammar import GrammarRegistry
from serve.cow_http import make_server
from serve.cow_prompt_encoder import CowPromptEncoder
from serve.cow_text_adapter import CowTextAdapter,UnsupportedTextRequest
from serve.cow_vision import CowImageNormalizer,CowVisionPromptEncoder
from serve.cow_vision_preparer import CowVisionPreparer
from serve.cow_workspace import CowWorkspace
from serve.frontend import ChatTemplate
from serve.server import ByteTokenizer
from serve.test_cow_vision_preparer import EncoderFixture
from serve.test_cow_vision_text_flow import TextCPUBackend
from test_vision_execution import receipt
from tools.cow_cli import CowClient
from tools.strata_tokenizer import BYTE_TO_UNICODE,Tokenizer

ROOT=Path(__file__).resolve().parents[1]


class ImageGrammarTokenizer(Tokenizer):
    def __init__(self):
        super().__init__([BYTE_TO_UNICODE[i] for i in range(256)]+ByteTokenizer.SPECIALS,[],
            [1]*256+[3]*6,special_ids={'eos_token_id':257})


class ImageGrammarCPUBackend(TextCPUBackend):
    def __init__(self,profile,tokenizer):
        super().__init__(profile);self.tokenizer=tokenizer;self.vocab_size=len(tokenizer.tokens)
    def piece(self,token):return self.tokenizer.token_bytes(token)
    def validate_grammar_engine(self,engine):
        if engine.tokenizer_sha256!=GrammarEngine._tokenizer_hash(self.tokenizer):
            raise IntegrityError('Wrong grammar tokenizer')
    def consume_generated(self,lease,ids):
        start=len(lease['inputs']);self.consume(lease,ids)
        self.last_execution_receipt=receipt(lease['inputs'],position=start,operator='normal-verifier-decode')


class CowVisionGrammarFlowTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.store=Store(self.temp.name);self.namespace=self.store.create_namespace('Images','namespace').namespace_id
        self.tokenizer=ImageGrammarTokenizer();self.encoder=EncoderFixture();self.models=[];self.manager=None
        self.registry_path=Path(self.temp.name)/'registry.json'
        self.registry_path.write_text(json.dumps(dict(schema=1,states={
            'READ':dict(format='gbnf',source='root ::= "AB"'),
            'EDIT':dict(format='gbnf',source='root ::= "XY"'),
            'TEST':dict(format='gbnf',source='root ::= "Z"')})))
        self.registry=GrammarRegistry.load(self.registry_path,self.tokenizer,[257])
        self.prompt=CowPromptEncoder(self.tokenizer,ChatTemplate(ROOT/'serve/chat_template.jinja'),{'enable_thinking':False})
        self.mixed=CowVisionPromptEncoder(self.prompt,renderer_id='c'*64)
        self.preparer=CowVisionPreparer(self.store,CowImageNormalizer(),self.mixed,self.encoder,original_bytes_limit=16*1024**2)
        self.addCleanup(self.close);self.start()
        stream=io.BytesIO();Image.new('RGB',(3,2),'red').save(stream,format='PNG')
        self.upload=self.preparer.upload(self.namespace,stream.getvalue(),pool=self.manager.pool,disk=self.manager.disk)
        self.parts=[dict(type='image',image_id=self.upload['image_id']),dict(type='text',text='diagram')]

    def start(self):
        profile=dict(model_contract_id='mixed-model',execution_profile_id='mixed-grammar',model_weights_sha256='a'*64,
            execution=dict(GPU=[0],resume_contract='identical-mixed-native-operators-v1',generated_operator='normal-verifier-decode',
                native_request_protocol='consumed-greedy-ack-v1',selected_token_protocol='consumed-selected-receipt-v1',
                prefill_chunk=1024,embedding_width=4,capacity=4096,eos_tokens=[257]))
        def factory(gpu):
            model=BranchingModel(self.temp.name,ImageGrammarCPUBackend(profile,self.tokenizer),profile)
            self.models.append(model);return model
        self.manager=ContextCoordinator(self.temp.name,PoolConfig(gpus=(0,),ram_bytes=1024**3,disk_bytes=16*1024**2,
            min_available_bytes=0),factory,input_preparer=None if getattr(self,'without_encoder',False) else self.preparer)
        self.manager.ready(5);self.tokens=CowAdapter(self.manager,grammar_registry=self.registry)
        self.text=CowTextAdapter(self.tokens,self.tokenizer,self.prompt.render,self.prompt.encode_rendered,
            append_user_suffix=self.prompt.append_user_suffix,append_initial_user_suffix=self.prompt.append_initial_user_suffix,
            initial_root_mode=self.prompt.initial_root_mode,renderer_id='c'*64,grammar_registry=self.registry,
            vision_preparer=None if getattr(self,'without_encoder',False) else self.preparer,mixed_renderer=self.mixed)
        self.workspace=CowWorkspace(self.store,self.tokenizer,root_mode_resolver=self.text.root_mode_for_projection)

    def close(self):
        if self.manager is not None:self.manager.close();self.manager=None

    def prepare(self):
        return self.text.prepare('main',[dict(role='user',content=self.parts)],'prepare',namespace_id=self.namespace).durable.result(5)

    def test_image_answer_and_resume_bind_grammar_to_typed_generation_receipt(self):
        self.prepare();request=self.text.answer('main',1,'answer',1,grammar_state='READ');request.durable.result(5)
        self.assertEqual(request.text_result(),'A')
        record=self.manager.committed_generation('answer')['generation']
        self.assertEqual(record['schema'],10);self.assertEqual(record['grammar']['agent_state'],'READ')
        self.assertEqual(record['source_input_kind'],'canonical-mixed-spans-v1')
        state=self.tokens.snapshot_controller(self.store.head('main').tip)
        self.assertNotIn('260',state['seen'])
        resumed=self.text.resume('main',2,'resume',8);resumed.durable.result(5)
        self.assertEqual(resumed.text_result(),'B')
        self.assertEqual(self.manager.committed_generation('resume')['generation']['schema'],10)
        self.assertEqual(self.workspace.history('main')['messages'][-1]['content'],'AB')

    def test_forked_parser_is_independent_when_sibling_enters_test(self):
        self.prepare();partial=self.text.answer('main',1,'partial',1,grammar_state='EDIT').durable.result(5)
        self.tokens.fork('sibling',partial['snapshot'],source_branch='main',operation_id='fork')
        b=self.store.head('sibling');saved=self.store.get_snapshot(b.tip)
        self.text.resume('main',2,'finish-edit',8).durable.result(5)
        request=self.text.append_user('main','test it',3,'test',8,grammar_state='TEST');request.durable.result(5)
        self.assertEqual(request.text_result(),'Z');self.assertEqual(self.store.head('sibling'),b)
        self.assertEqual(self.store.get_snapshot(b.tip),saved)
        tail=self.text.resume('sibling',b.revision,'finish-sibling',8);tail.durable.result(5)
        self.assertEqual(tail.text_result(),'Y')

    def test_committed_answer_replays_without_model_work_after_reopen(self):
        self.prepare();self.text.answer('main',1,'answer',8,grammar_state='READ').durable.result(5)
        self.close();self.start();before=self.models[-1].backend.stats['forward_tokens']
        result=self.text.retry('answer','main',2,'replay')
        self.assertTrue(result['replayed']);self.assertEqual(result['answer']['text'],'AB')
        self.assertEqual(self.models[-1].backend.stats['forward_tokens'],before)

    def test_partial_image_grammar_resumes_from_disk_after_reopen(self):
        self.prepare();self.text.answer('main',1,'partial',1,grammar_state='READ').durable.result(5)
        self.close();self.without_encoder=True;self.start();before=self.models[-1].backend.stats['forward_tokens']
        request=self.text.resume('main',2,'resume-cold',8);request.durable.result(5)
        self.assertEqual(request.text_result(),'B');self.assertEqual(self.models[-1].backend.stats['forward_tokens']-before,2)

    def test_first_image_question_can_select_a_grammar(self):
        self.text.create('main',[dict(role='user',content=self.parts)],'context',namespace_id=self.namespace).durable.result(5)
        request=self.text.ask('main',self.parts,1,'first-image',8,grammar_state='READ');request.durable.result(5)
        self.assertEqual(request.text_result(),'AB')
        descriptor=self.manager.committed_generation('first-image')['generation']
        self.assertEqual((descriptor['schema'],descriptor['kind']),(11,'initial_turn'))

    def test_resume_cannot_switch_the_active_image_grammar(self):
        self.prepare();self.text.answer('main',1,'partial',1,grammar_state='READ').durable.result(5)
        before=self.models[-1].backend.stats['forward_tokens'];head=self.store.head('main')
        with self.assertRaises(UnsupportedTextRequest):self.text.resume('main',2,'wrong-state',8,grammar_state='TEST')
        self.assertEqual(self.store.head('main'),head);self.assertEqual(self.models[-1].backend.stats['forward_tokens'],before)

    def test_legacy_descriptor_cannot_be_reinterpreted_as_grammar_capable(self):
        self.prepare();self.text.answer('main',1,'answer',8,grammar_state='READ').durable.result(5)
        path=self.manager._intents/'answer.json';envelope=json.loads(path.read_text())
        envelope['intent']['generation']['schema']=8
        import hashlib
        envelope['checksum']=hashlib.sha256(_json_bytes(envelope['intent'])).hexdigest();path.write_bytes(_json_bytes(envelope))
        with self.assertRaises(IntegrityError):self.manager.committed_generation('answer')

    def test_new_image_question_records_versioned_grammar_delta_and_replays_without_encoder(self):
        self.prepare();self.text.answer('main',1,'answer',8,grammar_state='READ').durable.result(5)
        request=self.text.append_user('main',self.parts,2,'image-next',8,grammar_state='TEST');request.durable.result(5)
        self.assertEqual(request.text_result(),'Z')
        generation=self.manager.committed_generation('image-next')['generation']
        self.assertEqual(generation['schema'],11);self.assertEqual(generation['grammar']['agent_state'],'TEST')
        self.assertTrue(any(span['kind']=='image' for span in generation['ids']))
        self.close();self.without_encoder=True;self.start();before=self.models[-1].backend.stats['forward_tokens']
        replay=self.text.append_user('main',self.parts,2,'image-next',8,grammar_state='TEST');replay.durable.result(5)
        self.assertEqual(replay.text_result(),'Z');self.assertEqual(self.models[-1].backend.stats['forward_tokens'],before)

    def test_corrupt_grammar_descriptor_cannot_be_replayed(self):
        self.prepare();self.text.answer('main',1,'answer',8,grammar_state='READ').durable.result(5)
        path=self.manager._intents/'answer.json';envelope=json.loads(path.read_text())
        envelope['intent']['generation']['grammar']['spec_sha256']='0'*64
        import hashlib
        envelope['checksum']=hashlib.sha256(_json_bytes(envelope['intent'])).hexdigest();path.write_bytes(_json_bytes(envelope))
        with self.assertRaises(IntegrityError):self.manager.committed_generation('answer')

    def test_unqualified_sampling_seed_stops_and_unknown_grammar_reject_without_work(self):
        self.prepare();before=self.models[-1].backend.stats['forward_tokens']
        for options in (dict(settings={'temperature':.7}),dict(seed=3),dict(text_stops=['END']),dict(grammar_state='UNKNOWN')):
            with self.subTest(options=options),self.assertRaises(UnsupportedTextRequest):
                self.text.answer('main',1,'bad',2,**options)
        self.assertEqual(self.models[-1].backend.stats['forward_tokens'],before)

    def test_authenticated_http_uses_same_image_grammar_path(self):
        self.prepare();server=make_server(self.tokens,api_key='cpu-fixture',text_adapter=self.text,workspace=self.workspace)
        thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        try:
            client=CowClient('http://127.0.0.1:'+str(server.server_address[1]),'cpu-fixture')
            result=client.stream('/cow/text/answer',dict(branch='main',revision=1,operation_id='http-answer',
                max_tokens=8,grammar_state='READ'),lambda _:None)
            self.assertEqual(result['answer']['text'],'AB')
            self.assertEqual(self.manager.committed_generation('http-answer')['generation']['schema'],10)
        finally:server.shutdown();server.server_close();thread.join(5)


if __name__=='__main__':unittest.main()
