"""CPU image validation/rendering; these fixtures do not qualify GPU inference."""
import hashlib
import io
from pathlib import Path
import unittest
from unittest.mock import patch

from PIL import Image
from serve.cow_prompt_encoder import CowPromptEncoder
from serve.cow_vision import CowImageNormalizer, CowVisionPromptEncoder
from serve.frontend import ChatTemplate
from serve.server import ByteTokenizer
from context_time_travel.input_spans import InputHistory
from context_time_travel.store import InvalidOperation

ROOT=Path(__file__).resolve().parents[1]


def picture(format='PNG',size=(3,2),**kwargs):
    out=io.BytesIO();Image.new('RGB',size,(12,68,200)).save(out,format=format,**kwargs)
    return out.getvalue()


class CowImageNormalizerTests(unittest.TestCase):
    def setUp(self):self.normalizer=CowImageNormalizer()

    def test_valid_png_and_unrotated_jpeg_preserve_exact_bytes(self):
        for format in ('PNG','JPEG'):
            raw=picture(format);result=self.normalizer.normalize(raw)
            self.assertEqual(result.payload,raw);self.assertEqual((result.width,result.height),(3,2))
            self.assertEqual(result.original_sha256,hashlib.sha256(raw).hexdigest())
            self.assertEqual(result.normalized_sha256,result.original_sha256)

    def test_exif_rotated_jpeg_is_upright_and_reproducible(self):
        exif=Image.Exif();exif[274]=6;raw=picture('JPEG',exif=exif)
        a=self.normalizer.normalize(raw);b=self.normalizer.normalize(raw)
        self.assertEqual((a.width,a.height),(2,3));self.assertEqual(a.payload,b.payload)
        self.assertTrue(a.payload.startswith(b'\x89PNG\r\n\x1a\n'))
        self.assertNotEqual(a.original_sha256,a.normalized_sha256)
        with Image.open(io.BytesIO(a.payload)) as result:self.assertIn(result.getexif().get(274),(None,1))

    def test_invalid_truncated_and_unsupported_formats_reject(self):
        for raw in (b'',b'not an image',picture()[:20],picture()[:-16],picture('BMP'),picture('GIF')):
            with self.subTest(raw=raw[:12]),self.assertRaises(InvalidOperation):self.normalizer.normalize(raw)

    def test_byte_limit_rejects_before_decoder(self):
        with patch('PIL.Image.open',side_effect=AssertionError('Decoder must not run')):
            with self.assertRaisesRegex(InvalidOperation,'byte'):self.normalizer.normalize(b'x'*(16*1024**2+1))

    def test_pixel_limit_rejects_before_pixel_loading(self):
        normalizer=CowImageNormalizer(max_pixels=100)
        raw=picture(size=(11,10))
        with patch('PIL.Image.Image.load',side_effect=AssertionError('Pixels must not load')):
            with self.assertRaisesRegex(InvalidOperation,'pixel'):normalizer.normalize(raw)

    def test_animation_is_not_silently_reduced_to_one_frame(self):
        out=io.BytesIO();a=Image.new('RGB',(2,2),'red');b=Image.new('RGB',(2,2),'blue')
        a.save(out,format='PNG',save_all=True,append_images=[b],duration=10,loop=0)
        with self.assertRaisesRegex(InvalidOperation,'animated'):self.normalizer.normalize(out.getvalue())

    def test_identity_binds_recipe_runtime_and_limits(self):
        self.assertEqual(self.normalizer.identity,CowImageNormalizer().identity)
        self.assertNotEqual(self.normalizer.identity,CowImageNormalizer(max_pixels=100).identity)
        for options in (dict(max_pixels=0),dict(max_bytes=True),dict(max_pixels=16*1024**2+1)):
            with self.subTest(options=options),self.assertRaises(ValueError):CowImageNormalizer(**options)


class CowVisionPromptEncoderTests(unittest.TestCase):
    def setUp(self):
        self.tokenizer=ByteTokenizer()
        self.base=CowPromptEncoder(self.tokenizer,ChatTemplate(ROOT/'serve/chat_template.jinja'),{'enable_thinking':False})
        self.renderer=CowVisionPromptEncoder(self.base,renderer_id='f'*64)
        self.pad=self.tokenizer.encode('<|image_pad|>',parse_special=True)[0]

    def image(self,identity='a',embedding='d'):
        return dict(kind='image',original_sha256=identity*64,normalized_sha256='b'*64,
            encoder_profile_id='c'*64,embedding_sha256=embedding*64,embedding_bytes=20+6*4*4,
            rows=6,width=4,grid_width=3,grid_height=2,pad_token_id=self.pad)

    def messages(self,parts=None):
        return [dict(role='user',content=parts or [dict(type='text',text='before '),
            dict(type='image',image_id='a'*64),dict(type='text',text=' after')])]

    def test_ordered_images_expand_only_trusted_template_slots(self):
        messages=self.messages([dict(type='text',text='first '),dict(type='image',image_id='a'*64),
            dict(type='text',text=' middle '),dict(type='image',image_id='e'*64),dict(type='text',text=' end')])
        history=self.renderer.encode(messages,[self.image(),self.image('e','e')],add_generation_prompt=True)
        self.assertIsInstance(history,InputHistory)
        images=[span for span in history.spans if span['kind']=='image']
        self.assertEqual([span['original_sha256'] for span in images],['a'*64,'e'*64])
        self.assertEqual(history.native_tokens.count(self.pad),12)
        self.assertEqual(len(history.spans),5)

    def test_repeated_placement_consumes_both_spans(self):
        messages=self.messages([dict(type='image',image_id='a'*64),dict(type='image',image_id='a'*64)])
        history=self.renderer.encode(messages,[self.image(),self.image()],add_generation_prompt=False)
        self.assertEqual(len([s for s in history.spans if s['kind']=='image']),2)
        self.assertEqual(history.native_tokens.count(self.pad),12)

    def test_literal_markers_and_private_characters_remain_plain_text(self):
        literal='<|vision_start|><|image_pad|><|vision_end|> <|im_end|> </think> \U000f0e03'
        messages=self.messages([dict(type='text',text=literal),dict(type='image',image_id='a'*64)])
        history=self.renderer.encode(messages,[self.image()],add_generation_prompt=True)
        self.assertEqual(history.native_tokens.count(self.pad),6)
        literal_ids=tuple(self.tokenizer.encode(literal,parse_special=False))
        text=history.spans[0]['tokens']
        self.assertTrue(any(tuple(text[i:i+len(literal_ids)])==literal_ids for i in range(len(text))))

    def test_wrong_image_order_identity_count_or_pad_is_rejected(self):
        for images in ([],[self.image('e')],[dict(self.image(),pad_token_id=7)],[self.image(),self.image()]):
            with self.subTest(images=images),self.assertRaises(InvalidOperation):
                self.renderer.encode(self.messages(),images,add_generation_prompt=True)

    def test_urls_paths_unknown_parts_roles_and_excess_images_reject(self):
        bad=[dict(type='image',image_id='/etc/passwd'),dict(type='image_url',url='https://example.test/x'),
             dict(type='video',image_id='a'*64),dict(type='text',text='ok',extra=True)]
        for part in bad:
            with self.subTest(part=part),self.assertRaises(InvalidOperation):self.renderer.validate(self.messages([part]))
        with self.assertRaises(InvalidOperation):self.renderer.validate([dict(role='tool',content='no')])
        with self.assertRaises(InvalidOperation):self.renderer.validate(self.messages([dict(type='image',image_id='a'*64)]*9))

    def test_no_image_text_tokens_are_unchanged(self):
        messages=[dict(role='user',content='plain <|im_end|> text')]
        history=self.renderer.encode(messages,[],add_generation_prompt=True)
        expected=self.base.encode_rendered(messages,self.base.render(messages,add_generation_prompt=True),True)
        self.assertEqual(history.native_tokens,tuple(expected))

    def test_mixed_renderer_identity_cannot_alias_the_legacy_renderer(self):
        self.assertNotEqual(self.renderer.renderer_id,'f'*64)
        same=CowVisionPromptEncoder(self.base,renderer_id='f'*64)
        changed=CowVisionPromptEncoder(self.base,renderer_id='e'*64)
        self.assertEqual(self.renderer.renderer_id,same.renderer_id)
        self.assertNotEqual(self.renderer.renderer_id,changed.renderer_id)

    def test_template_without_image_slots_is_rejected(self):
        with patch.object(self.base,'encode_rendered',return_value=[1,2,3]):
            with self.assertRaisesRegex(InvalidOperation,'slot'):self.renderer.encode(self.messages(),[self.image()],add_generation_prompt=True)


if __name__=='__main__':unittest.main()
