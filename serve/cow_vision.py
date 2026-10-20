"""Bounded image normalization and literal-safe mixed prompt rendering for COW.

No download, caller-supplied path, inference process or second artifact cache
lives here. The worker supplies leased image spans from the canonical store.
"""
from dataclasses import dataclass
import hashlib
import io
from pathlib import Path
import re
import warnings
import zlib

from context_time_travel.input_spans import InputHistory, canonical_inputs
from context_time_travel.store import InvalidOperation, _json_bytes


def _sha(raw):return hashlib.sha256(raw).hexdigest()


@dataclass(frozen=True)
class NormalizedImage:
    payload: bytes
    width: int
    height: int
    original_sha256: str
    normalized_sha256: str
    media_type: str


class CowImageNormalizer:
    """Validate PNG/JPEG fully; preserve ordinary bytes or apply EXIF rotation.

    Pillow is mandatory here even for PNG/JPEG: accepting a magic prefix alone
    cannot enforce decoded dimensions, animation or truncation limits.
    """
    def __init__(self,*,max_bytes=16*1024**2,max_pixels=16_000_000):
        if (type(max_bytes) is not int or not 0<max_bytes<=16*1024**2
                or type(max_pixels) is not int or not 0<max_pixels<=16_000_000):
            raise ValueError('Invalid image byte/pixel limit')
        import PIL
        from PIL import Image
        self.max_bytes,self.max_pixels=max_bytes,max_pixels
        core=Path(Image.core.__file__)
        self.identity=_sha(_json_bytes(dict(recipe='cow-png-jpeg-exif-v1',
            implementation_sha256=_sha(Path(__file__).read_bytes()),pillow=PIL.__version__,
            pillow_core_sha256=_sha(core.read_bytes()),zlib=zlib.ZLIB_RUNTIME_VERSION,
            max_bytes=max_bytes,max_pixels=max_pixels)))

    def normalize(self,raw):
        if type(raw) is not bytes or not 0<len(raw)<=self.max_bytes:
            raise InvalidOperation('Image byte length exceeds the configured bound')
        from PIL import Image, ImageOps
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('error',Image.DecompressionBombWarning)
                with Image.open(io.BytesIO(raw)) as check:
                    if check.format not in ('PNG','JPEG'):
                        raise InvalidOperation('Only PNG/JPEG images are supported')
                    if not 0<check.width*check.height<=self.max_pixels:
                        raise InvalidOperation('Decoded image exceeds the pixel limit')
                    if getattr(check,'is_animated',False):
                        raise InvalidOperation('An animated image is not a single image')
                    check.verify()
                with Image.open(io.BytesIO(raw)) as image:
                    image.load()
                    orientation=image.getexif().get(274,1)
                    if orientation not in (None,0,1,2,3,4,5,6,7,8):
                        raise InvalidOperation('Invalid EXIF orientation')
                    payload=raw;media_type='image/png' if image.format=='PNG' else 'image/jpeg'
                    if orientation not in (None,0,1):
                        image=ImageOps.exif_transpose(image)
                        if image.mode in ('RGBA','LA') or (image.mode=='P' and 'transparency' in image.info):
                            rgba=image.convert('RGBA');background=Image.new('RGB',rgba.size,'white')
                            background.paste(rgba,mask=rgba.getchannel('A'));image=background
                        elif image.mode!='RGB':image=image.convert('RGB')
                        out=io.BytesIO();image.save(out,format='PNG');payload=out.getvalue();media_type='image/png'
                    if len(payload)>64*1024**2:
                        raise InvalidOperation('Normalized image exceeds the staging byte limit')
                    return NormalizedImage(payload,image.width,image.height,_sha(raw),_sha(payload),media_type)
        except InvalidOperation:raise
        except (OSError,ValueError,SyntaxError,Image.DecompressionBombError,Image.DecompressionBombWarning) as error:
            raise InvalidOperation('Invalid, truncated or oversized image') from error


class CowVisionPromptEncoder:
    """Replace only template-produced image slots with explicit image spans.

    Literal special-looking text uses the existing Service prompt protections.
    Upload identities are references, never filesystem paths or fetched URLs.
    """
    def __init__(self,encoder,*,renderer_id):
        if type(renderer_id) is not str or re.fullmatch('[0-9a-f]{64}',renderer_id) is None:
            raise ValueError('Mixed renderer requires a pinned identity')
        self.encoder,self.base_renderer_id=encoder,renderer_id
        self.renderer_id=_sha(_json_bytes(dict(schema='cow-mixed-renderer-v1',
            base_renderer_id=renderer_id,implementation_sha256=_sha(Path(__file__).read_bytes()))))

    @staticmethod
    def validate(messages):
        if not isinstance(messages,(list,tuple)) or not messages:
            raise InvalidOperation('Mixed messages must be a nonempty list')
        clean=[];images=0
        for message in messages:
            if (type(message) is not dict or set(message)!={'role','content'}
                    or message['role'] not in ('system','user','assistant')):
                raise InvalidOperation('Invalid mixed message role/fields')
            content=message['content']
            if isinstance(content,str):
                clean.append(dict(role=message['role'],content=content));continue
            if not isinstance(content,list) or not content:
                raise InvalidOperation('Mixed content must be text or ordered nonempty parts')
            parts=[]
            for part in content:
                if (type(part) is dict and set(part)=={'type','text'}
                        and part['type']=='text' and isinstance(part['text'],str)):
                    parts.append(dict(part))
                elif (type(part) is dict and set(part)=={'type','image_id'} and part['type']=='image'
                        and type(part['image_id']) is str and re.fullmatch('[0-9a-f]{64}',part['image_id'])):
                    if message['role']=='system':raise InvalidOperation('System messages cannot contain images')
                    images+=1;parts.append(dict(part))
                    if images>8:raise InvalidOperation('At most eight image placements are supported')
                else:raise InvalidOperation('Unsupported mixed part or image identity')
            clean.append(dict(role=message['role'],content=parts))
        return clean

    def encode_user_suffix(self,messages,image_spans):
        """Render one new user frame without reinserting reasoning instructions.

        The trusted anchor has no image slots. Strip only its exact encoded
        prefix from the ordinary mixed renderer; retain every checked image
        identity and literal-data protection. This is a derived InputHistory
        view, never a second saved history or a positional conversion.
        """
        clean=self.validate(messages)
        if len(clean)!=1 or clean[0]['role']!='user':
            raise InvalidOperation('An image suffix requires exactly one user frame')
        thinking=self.encoder._user_suffix_thinking()
        if not thinking:
            # Preserve the already tested non-thinking path exactly.
            return self.encode(clean,image_spans,add_generation_prompt=True)
        anchor=[dict(role='system',content='COW suffix boundary reference.')]
        rendered=self.encoder.render(anchor,add_generation_prompt=False)
        ids=tuple(self.encoder.encode_rendered(anchor,rendered,False))
        complete=self.encode(anchor+clean,image_spans,add_generation_prompt=True)
        if not ids or len(complete)<=len(ids):
            raise InvalidOperation('Thinking image suffix anchor is empty or incomplete')
        expected=InputHistory([dict(kind='text',tokens=list(ids))])
        if complete.segment(0,len(ids))!=expected:
            raise InvalidOperation('Thinking image suffix anchor token prefix is not stable')
        return complete.segment(len(ids))

    def encode(self,messages,image_spans,*,add_generation_prompt):
        if type(add_generation_prompt) is not bool:raise InvalidOperation('Invalid generation-prompt mode')
        clean=self.validate(messages)
        refs=[part['image_id'] for message in clean if isinstance(message['content'],list)
              for part in message['content'] if part['type']=='image']
        spans=canonical_inputs(image_spans)
        if len(spans)!=len(refs) or any(span['kind']!='image' for span in spans):
            raise InvalidOperation('Image spans differ from requested placements')
        markers=[]
        for literal in ('<|vision_start|>','<|image_pad|>','<|vision_end|>'):
            tokens=self.encoder.tok.encode(literal,parse_special=True)
            if len(tokens)!=1:raise InvalidOperation('Template requires single vision marker IDs')
            markers.append(tokens[0])
        start,pad,end=markers
        if len(set(markers))!=3:raise InvalidOperation('Vision marker identities overlap')
        for ref,span in zip(refs,spans):
            if span['original_sha256']!=ref or span['pad_token_id']!=pad:
                raise InvalidOperation('Image order, identity or pad token differs')
        template_messages=[]
        for message in clean:
            content=message['content']
            if isinstance(content,list):
                content=[dict(type='image',source=part['image_id']) if part['type']=='image' else dict(part)
                         for part in content]
            template_messages.append(dict(role=message['role'],content=content))
        rendered=self.encoder.render(template_messages,add_generation_prompt=add_generation_prompt)
        ids=tuple(self.encoder.encode_rendered(template_messages,rendered,add_generation_prompt))
        result=[];text=[];image=0
        for index,token in enumerate(ids):
            if token!=pad:
                text.append(token);continue
            if (index==0 or index+1==len(ids) or ids[index-1]!=start or ids[index+1]!=end
                    or image==len(spans)):
                raise InvalidOperation('Unexpected native image slot in rendered prompt')
            result.extend([dict(kind='text',tokens=text),spans[image]])
            text=[];image+=1
        if image!=len(spans):raise InvalidOperation('Rendered prompt is missing an image slot')
        result.append(dict(kind='text',tokens=text))
        return InputHistory(result)
