"""Deterministic tiny hybrid GGUF for real CPU wiring tests, not a language model.

Uses the pinned GGUF v3 layout, F32 tensors and a test-only byte vocabulary.
No model download, PyTorch, NumPy, hidden answers or native-writer reuse.
"""
from pathlib import Path
import argparse
import math
import random
import struct

def make_fixture(destination, context_length=256, *, dtype="f32"):
    if dtype not in ("f32","f16"):raise ValueError("Explicit floating fixture dtype required")
    if type(context_length) is not int or not 2<=context_length<=32768:raise ValueError("Bounded fixture context required")
    rng=random.Random(20261023);tensors=[];metadata=[]
    def string(value):
        raw=value.encode();return struct.pack('<Q',len(raw))+raw
    def meta(name,value):
        if isinstance(value,str):kind=8;raw=string(value)
        elif isinstance(value,float):kind=6;raw=struct.pack('<f',value)
        elif isinstance(value,list):kind=9;raw=struct.pack('<IQ',5,len(value))+struct.pack('<'+'i'*len(value),*value)
        else:kind=4;raw=struct.pack('<I',value)
        metadata.append(string(name)+struct.pack('<I',kind)+raw)
    meta('general.architecture','wen1');meta('general.name','PCC generated tiny CPU wiring fixture; not Wen')
    values={'embedding_length':16,'feed_forward_length':32,'block_count':2,'attention.head_count':2,
        'attention.head_count_kv':1,'attention.key_length':8,'ssm.state_size':4,'ssm.time_step_rank':2,
        'ssm.group_count':1,'ssm.conv_kernel':3,'full_attention_interval':2,'context_length':context_length,
        'rope.dimension_count':8,'rope.freq_base':10000.,'attention.layer_norm_rms_epsilon':.000001,
        'rope.dimension_sections':[1,1,2,0]}
    for key,value in values.items():meta('wen1.'+key,value)
    def add(name,a,b=1):
        data=[rng.gauss(0,.1) for _ in range(a*b)]
        if 'norm.weight' in name:data=[v+1 for v in data]
        if name.endswith('ssm_a'):data=[-math.exp(v) for v in data]
        tensors.append((name,[a] if b==1 else [a,b],struct.pack('<'+('e' if dtype=='f16' else 'f')*len(data),*data)))
    add('token_embd.weight',16,256)
    for index in range(2):
        prefix='blk.'+str(index)+'.'
        for name,a,c in [('attn_norm.weight',16,1),('post_attention_norm.weight',16,1),
                         ('ffn_up.weight',16,32),('ffn_gate.weight',16,32),('ffn_down.weight',32,16)]:add(prefix+name,a,c)
        if index==0:
            for name,a,c in [('attn_qkv.weight',16,16),('attn_gate.weight',16,8),('ssm_beta.weight',16,2),
                ('ssm_alpha.weight',16,2),('ssm_dt.bias',2,1),('ssm_a',2,1),('ssm_conv1d.weight',3,16),
                ('ssm_norm.weight',4,1),('ssm_out.weight',8,16)]:add(prefix+name,a,c)
        else:
            for name,a,c in [('attn_q.weight',16,32),('attn_k.weight',16,8),('attn_v.weight',16,8),
                ('attn_output.weight',16,16),('attn_q_norm.weight',8,1),('attn_k_norm.weight',8,1)]:add(prefix+name,a,c)
    add('output_norm.weight',16);add('output.weight',16,256)
    header=b'GGUF'+struct.pack('<IQQ',3,len(tensors),len(metadata))+b''.join(metadata)
    descriptors=[];data=bytearray()
    for name,shape,raw in tensors:
        data.extend(b'\0'*((-len(data))%32));offset=len(data)
        descriptors.append(string(name)+struct.pack('<I',len(shape))+struct.pack('<'+'Q'*len(shape),*shape)+struct.pack('<IQ',1 if dtype=='f16' else 0,offset))
        data.extend(raw)
    header+=b''.join(descriptors);header+=b'\0'*((-len(header))%32)
    destination=Path(destination)
    with destination.open('xb') as out:out.write(header);out.write(data)
    destination.chmod(0o600)
    return destination

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('destination',type=Path);parser.add_argument('--context-length',type=int,default=256)
    args=parser.parse_args();print(make_fixture(args.destination,args.context_length))
