"""Explicit MiniCPM5 dense Llama contract over the shared native dense runtime.

Whole-state checkpoint copies preserve semantics; physical prefix sharing is
not advertised. Other Llama geometries and adapters require their own evidence.
"""
import ctypes as C
import math
from .qwen35_backend import Qwen35Backend, Qwen35FreshOracle
from .store import InvalidOperation, IntegrityError

GEOMETRY={'block_count':42,'embedding_length':2048,'feed_forward_length':6144,
 'context_length':131072,'attention.head_count':16,'attention.head_count_kv':2,
 'attention.key_length':128,'attention.value_length':128,'rope.dimension_count':128,
 'rope.freq_base':5000000.0,'attention.layer_norm_rms_epsilon':1e-6}

def validate_geometry(metadata):
 if metadata.get('general.architecture')!='llama' or metadata.get('tokenizer.ggml.model')!='gpt2':
  raise InvalidOperation('Dense Llama profile requires the qualified MiniCPM GPT-2 geometry')
 for key,want in GEOMETRY.items():
  value=metadata.get('llama.'+key)
  valid=(type(value) is int and value==want) if type(want) is int else (
   type(value) in (int,float) and math.isfinite(value) and math.isclose(value,want,rel_tol=1e-6,abs_tol=1e-12))
  if not valid:raise InvalidOperation('Unsupported dense Llama geometry: '+key)
 return dict(attention_layers=42,recurrent_layers=0)

def terminal_tokens(metadata):
 """Pin both the model EOS and the chat-template turn terminator."""
 tokens=metadata.get('tokenizer.ggml.tokens');eos=metadata.get('tokenizer.ggml.eos_token_id')
 if (type(tokens) is not list or type(eos) is not int or not 0<=eos<len(tokens)
     or tokens[eos]!='</s>' or tokens.count('<|im_end|>')!=1):
  raise InvalidOperation('Qualified MiniCPM model and chat terminators required')
 return sorted({eos,tokens.index('<|im_end|>')})

class LlamaDenseBackend(Qwen35Backend):
 BACKEND='llama-dense-llama.cpp'
 STATE_FORMAT='llama-dense-frontier-v1'
 GEOMETRY=GEOMETRY
 GEOMETRY_VALIDATOR=staticmethod(validate_geometry)
 ATTENTION_KIND='llama-causal-full-gqa'
 def __init__(self,profile,gpu_id=None):
  if profile.get('adapter') is not None or profile.get('execution',{}).get('adapter') is not None:
   raise IntegrityError('Dense Llama adapter attachment is not yet qualified')
  super().__init__(profile,gpu_id=gpu_id)
  actual=[i for i in range(self.vocab_size) if self.is_eog(i)]
  if profile['execution'].get('eos_tokens')!=actual:
   self.close();raise IntegrityError('Declared stop tokens differ from the native model terminators')
 def _load_model(self):
  try:fn=self.lib.ctt_architecture_contract
  except AttributeError as error:raise IntegrityError('Rebuild the explicit dense Llama binding') from error
  fn.argtypes=[];fn.restype=C.c_char_p
  if fn()!=b'llama-dense-v1':raise IntegrityError('Native binding has the wrong architecture contract')
  return super()._load_model()

class LlamaDenseFreshOracle(Qwen35FreshOracle):
 """Uses the independent native executable and allocates a new context per query."""
 pass
