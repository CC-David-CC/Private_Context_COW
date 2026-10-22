"""Pinned Qwen3.5-9B llama.cpp text adapter; whole-state copy fallback.

The native serializer includes attention and recurrent/convolution memory.
Our versioned derived payload also owns next logits. Canonical history and
controllers remain in Store; neither lives in this native state image.
Vision and native GPU sampler proposals are not advertised by this adapter.
"""
from dataclasses import dataclass
import ctypes as C
import math
import os
import threading
from .real import NativeBackend
from .store import IntegrityError, InvalidOperation
from .execution_receipt import tokens_digest
from .pools import normalize_gpu_selection
from .qwen35_lora import validate_adapter

_MODEL_LOAD_LOCK=threading.Lock()

def validate_math_profile(profile,environment=None):
    """Reject process math overrides absent from the execution/cache identity."""
    e=profile['execution'];env=os.environ if environment is None else environment
    precise=e.get('f32_activations',False)
    if type(precise) is not bool:
        raise IntegrityError('Explicit boolean F32 activation profile required')
    if env.get('GGML_CUDA_F32_ACTIVATIONS','0')!=('1' if precise else '0'):
        raise IntegrityError('Native activation precision differs from execution profile')
    if precise and (not profile.get('native_build',{}).get('f32_activations_patch_sha256')
            or e.get('cuda_cublas_compute_type')!='f32' or e.get('nvidia_tf32_override')!='0'
            or env.get('GGML_CUDA_CUBLAS_COMPUTE_TYPE')!='f32'
            or env.get('NVIDIA_TF32_OVERRIDE')!='0'):
        raise IntegrityError('F32 activations require the pinned precision build and disabled TF32')

    attention=e.get('f32_attention',False)
    if type(attention) is not bool:
        raise IntegrityError('Explicit boolean FP32 attention profile required')
    if env.get('COW_QWEN35_F32_ATTENTION','0')!=('1' if attention else '0'):
        raise IntegrityError('Native attention precision differs from execution profile')
    if attention and (not precise or e.get('kv_dtype')!='FP32' or e.get('flash_attention') is not False
            or profile.get('native_build',{}).get('attention_precision_binding')!='fp32-attention-v1'):
        raise IntegrityError('FP32 attention requires the matching native build, activations and cache format')
    if not attention and e.get('kv_dtype','FP16')!='FP16':
        raise IntegrityError('FP32 cache requires explicit FP32 attention')


def device_name_for_worker(profile,gpu_id):
    """Map pinned physical PCI IDs to process-visible CUDA names without rebinding CUDA."""
    e=profile['execution'];workers=normalize_gpu_selection(e.get('GPU',[0]))
    if any(type(gpu) is not int for gpu in workers):
        raise ValueError('Dense grouped workers require multi-device checkpoint support')
    if type(gpu_id) is not int or gpu_id not in workers:
        raise ValueError('Dense worker GPU must be selected in its execution profile')
    if e.get('device_binding') is None and workers==(0,):return None
    if e.get('device_binding')!='cuda-single-device-v1':
        raise ValueError('Dense GPU selection requires a pinned device binding')
    env=e.get('environment',{})
    if (env.get('CUDA_VISIBLE_DEVICES')!=','.join(map(str,workers))
            or env.get('CUDA_DEVICE_ORDER')!='PCI_BUS_ID'):
        raise ValueError('Dense CUDA visibility differs from selected physical GPUs')
    return 'CUDA'+str(workers.index(gpu_id))

_GEOMETRY={'block_count':32,'embedding_length':4096,'full_attention_interval':4,
 'attention.head_count':16,'attention.head_count_kv':4,'attention.key_length':256,
 'attention.value_length':256,'ssm.conv_kernel':4,'ssm.state_size':128,
 'ssm.group_count':16,'ssm.time_step_rank':32,'ssm.inner_size':4096}

def validate_geometry(metadata):
    architecture=metadata.get('general.architecture')
    if architecture not in ('qwen35','wen1') or metadata.get('tokenizer.ggml.model')!='gpt2':
        raise InvalidOperation('Dense adapter requires qwen35 or wen1 GPT-2 tokenizer geometry')
    for key,value in _GEOMETRY.items():
        actual=metadata.get(architecture+'.'+key)
        if type(actual) is not int or actual!=value:
            raise InvalidOperation('Unsupported Qwen3.5-9B geometry: '+key)
    return dict(attention_layers=8,recurrent_layers=24)

@dataclass(frozen=True)
class _QwenHandle:
    address: int
    @property
    def _as_parameter_(self):return C.c_void_p(self.address)

class _QwenLease:
    # A rejected foreign-thread __exit__ must not close a generator and strand
    # the owner's native context. State changes only after a checked release.
    def __init__(self,backend):self.backend=backend;self.pointer=None
    def __enter__(self):
        if self.pointer is not None:raise InvalidOperation('Lease already entered')
        self.pointer=self.backend._acquire();return self.pointer
    def __exit__(self,*exc):
        if self.pointer is None:raise InvalidOperation('Lease not active')
        self.backend._release(self.pointer);self.pointer=None

class Qwen35Backend(NativeBackend):
    STATE_FORMAT="qwen35-hybrid-frontier-v1"
    RESUME_CONTRACT="identical-native-batches-v1"
    def __init__(self,profile,gpu_id=None):
        e=profile['execution']
        workers=normalize_gpu_selection(e.get('GPU',[0]))
        if gpu_id is None:
            if len(workers)!=1:raise ValueError('Select an explicit dense worker GPU')
            gpu_id=workers[0]
        self.gpu_id=gpu_id;self._device_name=device_name_for_worker(profile,gpu_id)
        validate_math_profile(profile)
        if (profile.get('backend')!='qwen35-llama.cpp' or e.get('kv_dtype') not in ('FP16','FP32')
                or e.get('sampling_location')!='native-cpu-frontier'
                or e.get('state_format')!=self.STATE_FORMAT
                or e.get('resume_contract')!=self.RESUME_CONTRACT
                or type(e.get('capacity')) is not int or not 128<=e['capacity']<=262144
                or type(e.get('prefill_chunk')) is not int or not 1<=e['prefill_chunk']<=e['capacity']
                or type(e.get('threads')) is not int or e['threads']<1):
            raise IntegrityError('Unsupported Qwen execution profile')
        self._active=None;self._owner=None;self._lock=threading.RLock();self._close_pending=False
        self._adapter=None;self._adapter_spec=validate_adapter(profile)
        self.native_pid=os.getpid()
        super().__init__(profile)
        md={key:self.metadata(key) for key in ['general.architecture','tokenizer.ggml.model']}
        architecture=md['general.architecture']
        for key in _GEOMETRY:
            value=self.metadata(architecture+'.'+key)
            try:md[architecture+'.'+key]=int(value)
            except (ValueError,TypeError):md[architecture+'.'+key]=None
        try:validate_geometry(md)
        except BaseException:self.close();raise
        fn=self.lib.ctt_new_config;fn.argtypes=[C.c_void_p,C.c_int,C.c_int,C.c_int];fn.restype=C.c_void_p
        fn=self.lib.ctt_tokenize;fn.argtypes=[C.c_void_p,C.c_char_p,C.c_int,C.c_void_p,C.c_int,C.c_bool,C.c_bool];fn.restype=C.c_int
        self.lib.ctt_calls.argtypes=[C.c_void_p];self.lib.ctt_calls.restype=C.c_uint64
        self.stats['forward_calls']=0
        self.lib.ctt_sample_mask_v2.argtypes=[C.c_void_p,C.c_void_p,C.c_size_t,C.c_double,C.c_double,C.c_uint64,C.c_size_t,C.c_double]
        self.lib.ctt_sample_mask_v2.restype=C.c_int
        self.lib.ctt_choose_mask.argtypes=[C.c_void_p,C.c_void_p,C.c_size_t]
        self.lib.ctt_choose_mask.restype=C.c_int
        self.stats['sampling_calls']=0
        if self._adapter_spec is not None:
            try:
                self.lib.ctt_lora_load.argtypes=[C.c_void_p,C.c_char_p];self.lib.ctt_lora_load.restype=C.c_void_p
                self.lib.ctt_lora_free.argtypes=[C.c_void_p];self.lib.ctt_lora_free.restype=None
                self.lib.ctt_lora_attach.argtypes=[C.c_void_p,C.c_void_p,C.c_float];self.lib.ctt_lora_attach.restype=C.c_int
                self._adapter=self.lib.ctt_lora_load(self.model,os.fsencode(self._adapter_spec['path']))
                if not self._adapter:raise IntegrityError(self.lib.ctt_error().decode())
                # Check again after native loading; disk mutation cannot be
                # admitted as the originally pinned version.
                validate_adapter(profile)
            except BaseException:self.close();raise

    def _load_model(self):
        # Backend registration/loading is process-global; device choice is per model.
        with _MODEL_LOAD_LOCK:
            if self._device_name is None:return super()._load_model()
            env=self.profile['execution']['environment']
            if any(os.environ.get(k)!=env[k] for k in ('CUDA_VISIBLE_DEVICES','CUDA_DEVICE_ORDER')):
                raise IntegrityError('Dense CUDA environment differs from pinned device binding')
            try:fn=self.lib.ctt_load_device
            except AttributeError as exc:
                raise IntegrityError('Rebuild the dense adapter for explicit GPU selection') from exc
            fn.argtypes=[C.c_char_p,C.c_char_p];fn.restype=C.c_void_p
            return fn(os.fsencode(self.profile['model_path']),self._device_name.encode())

    def metadata(self,key):
        n=self.lib.ctt_meta(self.model,key.encode(),None,0)
        if n<0:return None
        data=C.create_string_buffer(n+1)
        if self.lib.ctt_meta(self.model,key.encode(),data,len(data))!=n:
            raise IntegrityError('Metadata changed during read')
        return data.value.decode()

    def contract_fields(self):
        fields=dict(attention='qwen35-hybrid-gdn-full-attention',state_kind=self.STATE_FORMAT)
        if self._adapter_spec is not None:
            fields['adapters']=((self._adapter_spec['sha256'],self._adapter_spec['scale']),)
        return fields

    def tokenize(self,text,*,add_special=True,parse_special=False):
        raw=text.encode();n=self.lib.ctt_tokenize(self.model,raw,len(raw),None,0,add_special,parse_special)
        if n==0:return ()
        if n>0:raise IntegrityError('Invalid tokenizer size response')
        data=(C.c_int32*(-n))();actual=self.lib.ctt_tokenize(self.model,raw,len(raw),data,-n,add_special,parse_special)
        if actual!=-n:raise IntegrityError('Tokenizer output size mismatch')
        return tuple(data)

    def lease(self):
        return _QwenLease(self)

    def _acquire(self):
        with self._lock:
            if self._active is not None or self._close_pending or not self.model:raise InvalidOperation('Qwen backend unavailable or already leased')
            e=self.profile['execution'];p=self.lib.ctt_new_config(self.model,e['capacity'],e['prefill_chunk'],e['threads'])
            if not p:raise RuntimeError(self.lib.ctt_error().decode())
            if self._adapter is not None and self.lib.ctt_lora_attach(p,self._adapter,self._adapter_spec['scale']):
                self.lib.ctt_free(p)
                raise IntegrityError(self.lib.ctt_error().decode())
            handle=_QwenHandle(p)
            self._active=handle;self._owner=threading.get_ident();self.stats['contexts_created']+=1
            return handle

    def _release(self,p):
        self._check(p)
        with self._lock:
            self.lib.ctt_free(p);self._active=None;self._owner=None;self.stats['contexts_freed']+=1
            if self._close_pending:self._close_native()

    def _check(self,lease):
        if lease is not self._active or self._owner!=threading.get_ident():
            raise InvalidOperation('Unavailable or foreign-thread Qwen lease')

    def position(self,lease):
        self._check(lease);return super().position(lease)

    def consume(self,lease,ids):
        self._check(lease);ids=tuple(ids);start=self.position(lease)+1;e=self.profile['execution']
        if (not ids or any(type(t) is not int or not 0<=t<self.vocab_size for t in ids)
                or start+len(ids)>e['capacity']):raise InvalidOperation('Invalid tokens or Qwen capacity exceeded')
        chunks=[]
        for offset in range(0,len(ids),e['prefill_chunk']):
            block=ids[offset:offset+e['prefill_chunk']];data=(C.c_int32*len(block))(*block)
            before=self.lib.ctt_work(lease);calls=self.lib.ctt_calls(lease)
            if self.lib.ctt_consume(lease,data,len(block)):raise IntegrityError(self.lib.ctt_error().decode())
            delta=self.lib.ctt_work(lease)-before
            if delta!=len(block) or self.lib.ctt_calls(lease)-calls!=1:raise IntegrityError('Native work accounting mismatch')
            self.stats['consumed_tokens']+=delta;self.stats['forward_calls']+=1
            chunks.append([start+offset,len(block)])
        self.last_execution_receipt=dict(operator='batched-prefill',position=start,count=len(ids),
            tokens_sha256=tokens_digest(ids),chunks=chunks)

    consume_generated=consume

    def logits(self,lease):
        self._check(lease);values=super().logits(lease)
        if len(values)!=self.vocab_size or not all(map(math.isfinite,values)):
            raise IntegrityError('Incomplete Qwen frontier')
        return values

    def export(self,lease):
        self._check(lease);return super().export(lease)

    def restore(self,lease,blob,*,history,frontier):
        self._check(lease)
        if (not isinstance(blob,bytes) or len(blob)<24 or not history
                or len(history)>self.profile['execution']['capacity']
                or len(frontier)!=self.vocab_size or not all(map(math.isfinite,frontier))):
            raise IntegrityError('Incomplete Qwen restore boundary')
        before=self.lib.ctt_work(lease)
        result=super().restore(lease,blob,history=history,frontier=frontier)
        if (self.lib.ctt_work(lease)!=before or self.position(lease)!=len(history)-1
                or self.logits(lease)!=tuple(frontier)):
            raise IntegrityError('Qwen restore boundary or zero-forward guarantee failed')
        return result

    def validate_grammar_engine(self,engine):
        from .grammar_first import GrammarEngine
        from tools.strata_tokenizer import Tokenizer
        if not isinstance(engine,GrammarEngine):raise InvalidOperation('Expected checked grammar engine')
        expected=getattr(self,'_grammar_tokenizer_sha256',None)
        if expected is None:
            expected=GrammarEngine._tokenizer_hash(Tokenizer.from_gguf(self.profile['model_path']))
            self._grammar_tokenizer_sha256=expected
        if engine.tokenizer_sha256!=expected:raise IntegrityError('Grammar tokenizer differs from Qwen model')

    def choose_legal(self,lease,mask):
        self._check(lease);data=C.create_string_buffer(mask,len(mask))
        token=self.lib.ctt_choose_mask(lease,data,len(mask))
        if token<0:raise InvalidOperation(self.lib.ctt_error().decode())
        self.stats['sampling_calls']+=1;return token

    def sample_legal(self,lease,mask,temperature,min_p,draw53,*,top_k=0,top_p=1.0,**filters):
        self._check(lease)
        if filters:raise InvalidOperation('Qwen adapter has not qualified these sampler extensions')
        if (type(top_k) is not int or top_k<0 or type(draw53) is not int or not 0<=draw53<2**53):
            raise InvalidOperation('Invalid native sampler integers')
        data=C.create_string_buffer(mask,len(mask))
        token=self.lib.ctt_sample_mask_v2(lease,data,len(mask),temperature,min_p,draw53,top_k,top_p)
        if token<0:raise InvalidOperation(self.lib.ctt_error().decode())
        self.stats['sampling_calls']+=1;return token

    def work_counters(self):
        return dict(consumed_tokens=self.stats['consumed_tokens'],forward_tokens=self.stats['consumed_tokens'],forward_calls=self.stats['forward_calls'],
            sampling_calls=self.stats['sampling_calls'],checkpoint_export_bytes=self.stats['checkpoint_export_bytes'],
            checkpoint_import_bytes=self.stats['checkpoint_import_bytes'])

    def close(self):
        if self._active is not None:
            self._check(self._active);self._close_pending=True;return
        self._close_native()

    def _close_native(self):
        if self._adapter is not None:
            self.lib.ctt_lora_free(self._adapter);self._adapter=None
        super().close()

class Qwen35FreshOracle:
    """Independent native fresh-context forward; no checkpoint read interface."""
    def __init__(self,profile,gpu_id=None):
        import subprocess,tempfile
        self.profile=profile;self.log=tempfile.TemporaryFile();e=profile['execution']
        command=[profile['oracle_binary'],profile['model_path'],str(e['capacity']),str(e['prefill_chunk']),str(e['threads'])]
        environment=None
        if e.get('device_binding') is not None or gpu_id is not None:
            workers=normalize_gpu_selection(e.get('GPU',[0]))
            if gpu_id is None:
                if len(workers)!=1:raise ValueError('Select an explicit oracle worker GPU')
                gpu_id=workers[0]
            device=device_name_for_worker(profile,gpu_id)
            if device is not None:
                command.append(device);environment={**os.environ,**e['environment']}
        adapter=validate_adapter(profile)
        if adapter is not None:command+=['--lora',adapter['path'],str(adapter['scale'])]
        self.process=subprocess.Popen(command,env=environment,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self.log)
        self.requests=0;self.tokens_computed=0
    def _read(self,n):
        from .oracle import LlamaFreshOracle
        return LlamaFreshOracle._read(self,n)
    def tokenize(self,text):
        from .oracle import LlamaFreshOracle
        return LlamaFreshOracle.tokenize(self,text)
    def forward(self,tokens,*,chunk=None):
        limit=self.profile['execution']['prefill_chunk'];chunk=limit if chunk is None else chunk
        if type(chunk) is not int or not 1<=chunk<=limit:raise ValueError('Unsupported oracle chunk')
        ids=tuple(tokens);sizes=[min(chunk,len(ids)-i) for i in range(0,len(ids),chunk)]
        return self._forward(ids,sizes)
    def forward_schedule(self,tokens,schedule):
        from .execution_receipt import validate_schedule
        validate_schedule(schedule,tokens,self.profile['execution']['prefill_chunk'])
        return self._forward(tuple(tokens),[n for event in schedule for _,n in event['chunks']])
    def _forward(self,ids,sizes):
        import struct,hashlib
        if (not ids or len(ids)>self.profile['execution']['capacity'] or sum(sizes)!=len(ids)
                or any(type(t) is not int or t<0 for t in ids)):raise ValueError('Invalid fresh history')
        self.process.stdin.write(('S '+str(len(sizes))+' '+' '.join(map(str,sizes))+' '+' '.join(map(str,ids))+'\n').encode());self.process.stdin.flush()
        n,work,position=struct.unpack('<iQi',self._read(16))
        if work!=len(ids) or position!=len(ids)-1:raise IntegrityError('Oracle omitted prefix work')
        raw=self._read(4*n);self.requests+=1;self.tokens_computed+=work
        self.last=dict(computed_tokens=work,position_max=position,logits_sha256=hashlib.sha256(raw).hexdigest(),chunks=sizes)
        return struct.unpack('<'+str(n)+'f',raw)
    def close(self):
        from .oracle import LlamaFreshOracle
        return LlamaFreshOracle.close(self)


def load_qwen35_backend(profile,gpu_id=None):
    if profile.get('execution',{}).get('state_format')=='qwen35-hybrid-mrope-frontier-v2':
        from .qwen35_mixed_backend import Qwen35MixedBackend
        return Qwen35MixedBackend(profile,gpu_id=gpu_id)
    return Qwen35Backend(profile,gpu_id=gpu_id)
