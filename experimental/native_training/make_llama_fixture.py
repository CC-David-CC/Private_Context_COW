"""Owned tiny ordinary Llama GGUF, including genuine F16 source tensors.

This fixture is not a language model or a copy of a protected evaluation task.
No GGML, PyTorch or NumPy writer is used.
"""
from pathlib import Path
import random,struct

def make_fixture(destination, *, dtype='f16', layers=2, context_length=512):
 if dtype not in ('f16','f32') or type(layers) is not int or not 1<=layers<=42:
  raise ValueError('Explicit tiny floating-point fixture required')
 if type(context_length) is not int or not 2<=context_length<=131072:raise ValueError('Bounded fixture context required')
 rng=random.Random(20261024);metadata=[];tensors=[]
 def string(text):
  raw=text.encode();return struct.pack('<Q',len(raw))+raw
 def meta(name,value):
  if isinstance(value,str):kind=8;raw=string(value)
  elif isinstance(value,float):kind=6;raw=struct.pack('<f',value)
  else:kind=4;raw=struct.pack('<I',value)
  metadata.append(string(name)+struct.pack('<I',kind)+raw)
 meta('general.architecture','llama');meta('general.name','PCC generated tiny ordinary Llama training fixture')
 values={'embedding_length':4,'feed_forward_length':8,'block_count':layers,'attention.head_count':2,
 'attention.head_count_kv':1,'attention.key_length':2,'attention.value_length':2,'context_length':context_length,
 'rope.dimension_count':2,'rope.freq_base':10000.,'attention.layer_norm_rms_epsilon':.000001}
 for key,value in values.items():meta('llama.'+key,value)
 def add(name,a,b=1):
  values=[rng.gauss(0,.15) for _ in range(a*b)]
  if 'norm.weight' in name:values=[v+1 for v in values]
  raw=struct.pack('<'+('e' if dtype=='f16' else 'f')*len(values),*values)
  tensors.append((name,[a] if b==1 else [a,b],raw))
 add('token_embd.weight',4,7)
 for i in range(layers):
  for name,a,b in [('attn_norm.weight',4,1),('ffn_norm.weight',4,1),('attn_q.weight',4,4),
  ('attn_k.weight',4,2),('attn_v.weight',4,2),('attn_output.weight',4,4),('ffn_up.weight',4,8),
  ('ffn_gate.weight',4,8),('ffn_down.weight',8,4)]:add('blk.'+str(i)+'.'+name,a,b)
 add('output_norm.weight',4);add('output.weight',4,7)
 header=b'GGUF'+struct.pack('<IQQ',3,len(tensors),len(metadata))+b''.join(metadata)
 descriptors=[];data=bytearray()
 for name,shape,raw in tensors:
  data.extend(b'\0'*((-len(data))%32));offset=len(data)
  descriptors.append(string(name)+struct.pack('<I',len(shape))+struct.pack('<'+'Q'*len(shape),*shape)+struct.pack('<IQ',1 if dtype=='f16' else 0,offset));data.extend(raw)
 header+=b''.join(descriptors);header+=b'\0'*((-len(header))%32)
 destination=Path(destination)
 with destination.open('xb') as f:f.write(header);f.write(data)
 destination.chmod(0o600);return destination
