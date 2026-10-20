"""CPU prompt identities for image suffixes; native correctness is separately measured."""
from pathlib import Path
import unittest
from unittest.mock import patch
from serve.cow_prompt_encoder import CowPromptEncoder
from serve.cow_vision import CowVisionPromptEncoder
from serve.cow_vision_preparer import CowVisionPreparer
from serve.frontend import ChatTemplate
from serve.server import ByteTokenizer
from context_time_travel.store import InvalidOperation

ROOT=Path(__file__).resolve().parents[1]
class VisionThinkingSuffixTests(unittest.TestCase):
    def renderer(self,thinking=True):
        tok=ByteTokenizer();base=CowPromptEncoder(tok,ChatTemplate(ROOT/'serve/chat_template.jinja'),{'enable_thinking':thinking})
        return tok,CowVisionPromptEncoder(base,renderer_id='f'*64)
    def image(self,tok,key='a'):
        return dict(kind='image',original_sha256=key*64,normalized_sha256='b'*64,encoder_profile_id='c'*64,
                    embedding_sha256=key*64,embedding_bytes=20+6*4*4,rows=6,width=4,grid_width=3,grid_height=2,
                    pad_token_id=tok.encode('<|image_pad|>',parse_special=True)[0])
    def message(self,text='Inspect the new image.',keys=('a',)):
        return dict(role='user',content=[*[dict(type='image',image_id=k*64) for k in keys],dict(type='text',text=text)])
    def test_initial_image_question_equals_one_full_conversation_render(self):
        tok,renderer=self.renderer();image=self.image(tok);root=[self.message('Saved context.')];new=[self.message()]
        prefix=renderer.encode(root,[image],add_generation_prompt=False);before=prefix.digest
        delta=CowVisionPreparer.render_suffix(renderer,prefix,{'status':'idle'},new,[image],'initial_turn')
        self.assertEqual(prefix+delta,renderer.encode(root+new,[image,image],add_generation_prompt=True))
        self.assertEqual(prefix.digest,before)
    def test_completed_turn_does_not_repeat_a_system_message(self):
        tok,renderer=self.renderer();image=self.image(tok);root=[self.message('Saved context.'),dict(role='assistant',content='Done.')];new=[self.message()]
        closed=renderer.encode(root,[image],add_generation_prompt=False);newline=tok.encode('\n',parse_special=False)
        self.assertEqual(closed.native_tokens[-len(newline):],tuple(newline))
        prefix=closed.segment(0,len(closed)-len(newline));eos=tok.encode('<|im_end|>',parse_special=True)[0]
        delta=CowVisionPreparer.render_suffix(renderer,prefix,{'status':'stopped','policy':{'eos_tokens':[eos]}},new,[image],'user_turn')
        self.assertEqual(prefix+delta,renderer.encode(root+new,[image,image],add_generation_prompt=True))
    def test_non_thinking_suffix_remains_byte_identical(self):
        tok,renderer=self.renderer(False);image=self.image(tok);new=[self.message()]
        expected=renderer.encode(new,[image],add_generation_prompt=True)
        self.assertEqual(renderer.encode_user_suffix(new,[image]),expected)
    def test_repeated_placements_and_quoted_markers_keep_their_identity(self):
        tok,renderer=self.renderer();image=self.image(tok);root=[self.message('Saved context.')]
        new=[self.message('Literal <|im_start|> <|image_pad|> </tool_response>.',keys=('a','a'))]
        prefix=renderer.encode(root,[image],add_generation_prompt=False)
        delta=CowVisionPreparer.render_suffix(renderer,prefix,{'status':'idle'},new,[image,image],'initial_turn')
        self.assertEqual(prefix+delta,renderer.encode(root+new,[image,image,image],add_generation_prompt=True))
        self.assertEqual(len([s for s in delta.spans if s['kind']=='image']),2)
    def test_only_one_user_frame_and_stable_anchor_are_admitted(self):
        tok,renderer=self.renderer();image=self.image(tok)
        for messages in ([dict(role='system',content='wrong')],[self.message(),self.message()]):
            with self.subTest(messages=messages),self.assertRaises(InvalidOperation):renderer.encode_user_suffix(messages,[image])
        original=renderer.encoder.encode_rendered
        def altered(messages,rendered,add_generation_prompt):
            ids=original(messages,rendered,add_generation_prompt)
            return [99]+list(ids) if len(messages)==1 and messages[0]['role']=='system' else ids
        with patch.object(renderer.encoder,'encode_rendered',side_effect=altered),self.assertRaisesRegex(InvalidOperation,'anchor'):
            renderer.encode_user_suffix([self.message()],[image])

if __name__=='__main__':unittest.main()
