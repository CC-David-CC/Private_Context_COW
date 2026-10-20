"""Actual preparation/admission path with CPU fixture inference, not GPU proof."""
from pathlib import Path
import unittest

from context_time_travel.input_spans import InputHistory
from serve.cow_prompt_encoder import CowPromptEncoder
from serve.cow_vision import CowImageNormalizer, CowVisionPromptEncoder
from serve.cow_vision_preparer import CowVisionPreparer
from serve.frontend import ChatTemplate
import serve.test_cow_vision_text_flow as fixture

ROOT=Path(__file__).resolve().parents[1]

class VisionThinkingPreparationTests(unittest.TestCase):
    def setUp(self):
        self.flow=fixture.CowVisionTextFlowTests();self.flow.setUp();self.addCleanup(self.flow.doCleanups)
        f=self.flow;f.close()
        f.prompt=CowPromptEncoder(f.tokenizer,ChatTemplate(ROOT/'serve/chat_template.jinja'),{'enable_thinking':True})
        f.mixed=CowVisionPromptEncoder(f.prompt,renderer_id='c'*64)
        f.preparer=CowVisionPreparer(f.store,CowImageNormalizer(),f.mixed,f.encoder,original_bytes_limit=16*1024**2)
        f.start()
        # The synthetic fixture emits each whole request as one completed chunk.
        # Thinking instructions exceed its old128-token limit; declare its actual
        # one-request bound here. Real GPU profiles/batch sizes stay untouched.
        f.models[-1].profile['execution']['prefill_chunk']=2048
        self.content=[dict(type='image',image_id=f.upload['image_id']),dict(type='text',text='Literal <|im_start|> and <|image_pad|>. Inspect the new image.')]

    def context(self):
        f=self.flow
        return f.text.create('main',f.messages,'context',namespace_id=f.namespace).durable.result(5)['snapshot']

    def test_first_preparation_matches_full_render_and_preserves_source(self):
        f=self.flow;root=self.context();source=f.tokens.history('main');before=f.models[-1].snapshot_bytes(root)
        request=f.text.ask('main',self.content,1,'question',2);request.durable.result(5)
        delta=InputHistory(f.manager.committed_generation('question')['generation']['ids'])
        images=[span for span in source.spans+delta.spans if span['kind']=='image']
        expected=f.mixed.encode(f.messages+[dict(role='user',content=self.content)],images,add_generation_prompt=True)
        self.assertEqual(source+delta,expected)
        self.assertEqual(f.models[-1].snapshot_bytes(root),before);self.assertEqual(len(f.encoder.calls),1)

    def test_original_retry_agrees_with_first_preparation_without_model_work(self):
        f=self.flow;self.context()
        first=f.text.ask('main',self.content,1,'question',2);receipt=first.durable.result(5)
        before=f.models[-1].backend.stats['forward_tokens'];calls=len(f.encoder.calls)
        replay=f.text.ask('main',self.content,1,'question',2)
        self.assertEqual(replay.durable.result(5)['branch'],receipt['branch'])
        self.assertEqual(f.models[-1].backend.stats['forward_tokens'],before);self.assertEqual(len(f.encoder.calls),calls)

    def test_completed_turn_preparation_uses_same_suffix_path_as_retry(self):
        f=self.flow;f.prepare();f.models[-1].backend.next_token=257
        f.text.answer('main',1,'finish',2).durable.result(5)
        source=f.tokens.history('main');head=f.store.head('main');state=f.store.get_snapshot(head.tip)['controller']
        f.models[-1].backend.next_token=120
        request=f.text.append_user('main',self.content,head.revision,'next',2);receipt=request.durable.result(5)
        delta=InputHistory(f.manager.committed_generation('next')['generation']['ids'])
        images=[span for span in delta.spans if span['kind']=='image']
        expected=CowVisionPreparer.render_suffix(f.mixed,source,state,[dict(role='user',content=self.content)],images,'user_turn')
        self.assertEqual(delta,expected)
        self.assertEqual(f.tokens.history('main').prefix(len(source)),source)
        self.assertEqual(f.text.append_user('main',self.content,head.revision,'next',2).durable.result(5)['branch'],receipt['branch'])

if __name__=='__main__':unittest.main()
