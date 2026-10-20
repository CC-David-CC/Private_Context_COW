"""Trusted image input preparation for the existing COW worker coordinator."""
from concurrent.futures import CancelledError
from contextlib import contextmanager, ExitStack
from dataclasses import dataclass
import hashlib
from pathlib import Path
import re
from types import SimpleNamespace

from context_time_travel.store import IntegrityError, InvalidOperation, _json_bytes
from context_time_travel.vision_artifacts import ImageArtifacts, RetainedImageArtifacts


@dataclass(frozen=True)
class _AdmittedImages:
    request_sha256: str
    artifacts: object
    grants: tuple


class CowVisionPreparer:
    @staticmethod
    def suffix_separator(renderer,history,controller,kind):
        from context_time_travel.input_spans import InputHistory
        if not isinstance(history,InputHistory) or kind not in ('initial_turn','user_turn'):
            raise InvalidOperation('Image suffix requires a supported mixed source and user boundary')
        spans=history.spans
        tail=tuple(spans[-1]['tokens']) if spans and spans[-1]['kind']=='text' else ()
        encoder=renderer.encoder
        suffix=(encoder.append_initial_user_suffix if kind=='initial_turn' else encoder.append_user_suffix)(tail,controller,'')
        frame=encoder._user_frame('')
        if len(suffix)<len(frame) or tuple(suffix[-len(frame):])!=tuple(frame):
            raise InvalidOperation('Image suffix differs from the qualified user frame')
        return tuple(suffix[:-len(frame)])

    @staticmethod
    def render_suffix(renderer,history,controller,messages,images,kind):
        from context_time_travel.input_spans import InputHistory
        separator=CowVisionPreparer.suffix_separator(renderer,history,controller,kind)
        delta=renderer.encode_user_suffix(messages,images)
        return InputHistory([dict(kind='text',tokens=list(separator))])+delta if separator else delta

    def __init__(self,store,normalizer,renderer,encoder,*,original_bytes_limit,
                 upload_idle_seconds=1800,clock_ns=None):
        if store._principal is not None:raise PermissionError('Image preparation requires the canonical owner')
        self.store,self.normalizer,self.renderer,self.encoder=store,normalizer,renderer,encoder
        self.original_bytes_limit,self.upload_idle_seconds=original_bytes_limit,upload_idle_seconds
        self.clock_ns=clock_ns
        self.encoder_profile_id=hashlib.sha256(_json_bytes(dict(schema='cow-image-producer-v1',
            normalizer=normalizer.identity,encoder=encoder.identity))).hexdigest()
        self.identity=hashlib.sha256(_json_bytes(dict(schema='cow-image-root-preparer-v1',
            implementation_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            producer=self.encoder_profile_id,renderer=renderer.renderer_id))).hexdigest()

    def artifacts(self,pool,disk):
        return ImageArtifacts(self.store,pool,disk,encoder_profile_id=self.encoder_profile_id,
            expected_width=self.encoder.width,max_rows=self.encoder.max_rows,
            original_bytes_limit=self.original_bytes_limit,upload_idle_seconds=self.upload_idle_seconds,
            clock_ns=self.clock_ns)

    def _normalization_budget(self,size):
        # Bound original, decoder workspace/rotation/conversion and normalized
        # output before Pillow runs. Persistent byte leases use actual lengths.
        return size+32*self.normalizer.max_pixels+64*1024**2

    def upload(self,namespace_id,original,*,pool,disk):
        self.store.get_namespace(namespace_id)
        if type(original) is not bytes or not 0<len(original)<=self.normalizer.max_bytes:
            raise InvalidOperation('Upload exceeds the original image byte bound')
        with pool.reserve(self._normalization_budget(len(original))):
            normalized=self.normalizer.normalize(original)
            receipt=self.artifacts(pool,disk).upload(namespace_id,original)
        original_media_type='image/png' if original.startswith(b'\x89PNG\r\n\x1a\n') else 'image/jpeg'
        return dict(receipt,width=normalized.width,height=normalized.height,media_type=original_media_type,
            normalized_media_type=normalized.media_type,
            normalized_sha256=normalized.normalized_sha256,normalizer_id=self.normalizer.identity)

    def validate(self,request):
        suffix=isinstance(request,dict) and request.get('schema')==2
        fields={'schema','namespace_id','messages','add_generation_prompt'}|({'kind'} if suffix else set())
        if (type(request) is not dict or set(request)!=fields
                or type(request['schema']) is not int or request['schema'] not in (1,2)
                or type(request['namespace_id']) is not str or not re.fullmatch('[0-9a-f]{32}',request['namespace_id'])
                or type(request['add_generation_prompt']) is not bool):
            raise InvalidOperation('Invalid mixed root request')
        self.store.get_namespace(request['namespace_id'])
        result=dict(schema=request['schema'],namespace_id=request['namespace_id'],
            messages=self.renderer.validate(request['messages']),add_generation_prompt=request['add_generation_prompt'])
        if suffix:
            if (request['kind'] not in ('initial_turn','user_turn') or not request['add_generation_prompt']
                    or len(result['messages'])!=1 or result['messages'][0]['role']!='user'
                    or not isinstance(result['messages'][0]['content'],list)
                    or not any(part['type']=='image' for part in result['messages'][0]['content'])):
                raise InvalidOperation('Image suffix requires one new user message and a generation frame')
            result['kind']=request['kind']
        return result

    def presentation(self,request,operation_id):
        from serve.cow_text_adapter import CowTextAdapter
        request=self.validate(request);messages=[]
        for index,message in enumerate(request['messages']):
            content=message['content']
            if isinstance(content,list):
                content=''.join(part['text'] if part['type']=='text' else '[image sha256:'+part['image_id']+']'
                                for part in content)
            messages.append(dict(id=CowTextAdapter._message_id(operation_id,f'input-{index}'),role=message['role'],content=content))
        kind=request.get('kind','root')
        return dict(schema=1,namespace_id=request['namespace_id'],kind=kind,messages=messages,
            renderer_id=self.renderer.renderer_id,
            output_message_id=None if kind=='root' else CowTextAdapter._message_id(operation_id,'assistant'))

    @contextmanager
    def admit(self,request,*,pool,disk,presentation,operation_id):
        request=self.validate(request)
        expected=self.presentation(request,operation_id)
        if request['schema']==2 and isinstance(presentation,dict) and 'seed' in presentation:
            from context_time_travel.presentation_request import detach_request
            checked=detach_request(presentation,expected_kind=request['kind'])
            expected.update(schema=2,seed=checked['seed'])
        if presentation!=expected:
            raise InvalidOperation('Image presentation differs from trusted request projection')
        artifacts=self.artifacts(pool,disk)
        refs=[part['image_id'] for message in request['messages'] if isinstance(message['content'],list)
              for part in message['content'] if part['type']=='image']
        with ExitStack() as held:
            grants=tuple(held.enter_context(artifacts.lease_uploaded(request['namespace_id'],identity)) for identity in refs)
            yield _AdmittedImages(hashlib.sha256(_json_bytes(request)).hexdigest(),artifacts,grants)

    @contextmanager
    def prepare(self,model,request,*,pool,disk,cancelled=None,admission=None,source_snapshot=None):
        if (type(admission) is not _AdmittedImages
                or admission.request_sha256!=hashlib.sha256(_json_bytes(request)).hexdigest()
                or admission.artifacts.store.root.resolve()!=model.store.root.resolve()):
            raise InvalidOperation('Image preparation requires its admitted original leases')
        execution=model.profile['execution']
        if (execution.get('resume_contract')!='identical-mixed-native-operators-v1'
                or execution.get('embedding_width')!=self.encoder.width
                or type(getattr(model,'gpu_id',None)) is not int):
            raise InvalidOperation('Worker does not support this image encoder profile')
        if request['schema']==2:
            from context_time_travel.input_spans import InputHistory,history_for_snapshot
            if source_snapshot is None:raise InvalidOperation('Image suffix requires its canonical source')
            source_history=history_for_snapshot(model.store,source_snapshot)
            if not isinstance(source_history,InputHistory):raise InvalidOperation('Image suffix requires mixed source history')
            controller=model.store.get_snapshot(source_snapshot)['controller']
            # Refuse an unsupported source boundary before encoding any image.
            self.suffix_separator(self.renderer,source_history,controller,request['kind'])
        elif source_snapshot is not None:
            raise InvalidOperation('Root preparation cannot consume an existing source')
        artifacts=admission.artifacts;images=[];encodings=[];sources=[]
        pad,=self.renderer.encoder.tok.encode('<|image_pad|>',parse_special=True)
        with ExitStack() as held:
            for grant in admission.grants:
                if cancelled is not None and cancelled():raise CancelledError('Image preparation cancelled')
                with pool.reserve(self._normalization_budget(grant['bytes'])):
                    original=artifacts._original(grant['image_id'])
                    normalized=self.normalizer.normalize(original)
                    held.enter_context(pool.reserve(len(original)+len(normalized.payload)))
                def encode(raw):
                    result=self.encoder.encode(raw,store=model.store,pool=pool,disk=disk,
                        gpu_id=model.gpu_id,cancelled=cancelled)
                    encodings.append(result.receipt)
                    return result.embedding.payload
                item=held.enter_context(artifacts.prepare(original=original,normalized=normalized.payload,
                    encode=encode,pad_token_id=pad))
                images.append(item.span);sources.append(dict(original_sha256=item.span['original_sha256'],
                    embedding_sha256=item.embedding.sha256,rows=item.embedding.rows,source=item.source))
            if request['schema']==2:
                # First admission and committed retry must consume the same frame.
                # Rendering a standalone thinking user message repeats its system
                # instructions; use the shared checked suffix path in both cases.
                history=self.render_suffix(self.renderer,source_history,controller,
                    request['messages'],images,request['kind'])
            else:
                history=self.renderer.encode(request['messages'],images,add_generation_prompt=request['add_generation_prompt'])
            yield SimpleNamespace(history=history,artifacts=artifacts,
                receipt=dict(preparer_id=self.identity,encoder_profile_id=self.encoder_profile_id,
                    renderer_id=self.renderer.renderer_id,images=sources,encodings=encodings,
                    consumed_cells=len(history),image_rows=sum(span['rows'] for span in images)))

    def collect_expired(self,*,pool,disk):
        return self.artifacts(pool,disk).collect()

    @contextmanager
    def reconstruction(self,model,history,*,pool,disk,cancelled=None):
        """Supply retained inputs and verified regeneration on the model worker."""
        execution=model.profile['execution']
        if (execution.get('resume_contract')!='identical-mixed-native-operators-v1'
                or execution.get('embedding_width')!=self.encoder.width or type(model.gpu_id) is not int):
            raise InvalidOperation('Worker does not support this image reconstruction profile')
        expected={}
        for span in history.spans:
            if span['kind']=='image' and span['encoder_profile_id']==self.encoder_profile_id:
                prior=expected.setdefault(span['original_sha256'],span['normalized_sha256'])
                if prior!=span['normalized_sha256']:raise IntegrityError('Retained normalization identity is inconsistent')
        receipt=dict(preparer_id=self.identity,encoder_profile_id=self.encoder_profile_id,encoder_calls=0,encodings=[])
        reader=RetainedImageArtifacts(model.store,pool,disk,expected_width=self.encoder.width,
            max_rows=execution['capacity'],reencode_profile_id=self.encoder_profile_id)
        def check_cancelled():
            if cancelled is not None and cancelled():raise CancelledError('Image reconstruction cancelled')
        with ExitStack() as held:
            def reencode(original):
                check_cancelled();identity=hashlib.sha256(original).hexdigest()
                if identity not in expected:raise IntegrityError('Original image is outside the pinned reconstruction profile')
                # The artifact lease already owns the full original buffer.
                with pool.reserve(self._normalization_budget(0)):
                    normalized=self.normalizer.normalize(original)
                    if hashlib.sha256(normalized.payload).hexdigest()!=expected[identity]:
                        raise IntegrityError('Re-encoding changed normalized image identity')
                    held.enter_context(pool.reserve(len(normalized.payload)+20+4*self.encoder.width*self.encoder.max_rows))
                check_cancelled()
                result=self.encoder.encode(normalized.payload,store=model.store,pool=pool,disk=disk,
                    gpu_id=model.gpu_id,cancelled=cancelled)
                receipt['encoder_calls']+=1;receipt['encodings'].append(result.receipt)
                check_cancelled();return normalized.payload,result.embedding.payload
            check_cancelled()
            yield SimpleNamespace(artifacts=reader,reencode=reencode,receipt=receipt)
