"""Trainable Wen import from its own GGUF, never a replacement upstream model.

The optional pinned Transformers GGUF importer reverses the Qwen3.5 tensor
layout transforms. Wen aliases are registered in this process only. Source
GGUF bytes, vocabulary, merges and chat template are never rewritten.
"""
import argparse
from functools import partial
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'experimental/branching_context/src')]


def _digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=True,
        allow_nan=False).encode()).hexdigest()


def validate_wen_source(source):
    from context_time_travel.qwen35_backend import validate_geometry
    metadata=source.metadata
    if metadata.get('general.architecture')!='wen1' or metadata.get('tokenizer.ggml.pre')!='wen1':
        raise ValueError('This importer requires the explicit Wen architecture/tokenizer alias')
    try:geometry=validate_geometry(metadata)
    except Exception as error:raise ValueError('Unsupported Wen geometry') from error
    if any(t.type_name not in ('BF16','F32') for t in source.tensors):
        raise ValueError('This reconstruction is qualified only for BF16/F32 source tensors')
    tokens=metadata.get('tokenizer.ggml.tokens')
    if type(tokens) is not list or not tokens or any(type(t) is not str for t in tokens):
        raise ValueError('Exact source vocabulary required')
    rows={t.name:t for t in source.tensors}
    for name in ('token_embd.weight','output.weight'):
        if name not in rows or rows[name].shape!=[metadata['wen1.embedding_length'],len(tokens)]:
            raise ValueError('Wen vocabulary rows differ from embedding/output tensors')
    tokenizer={key:value for key,value in metadata.items() if key.startswith('tokenizer.')}
    return dict(architecture='wen1',geometry=geometry,vocab_size=len(tokens),
        tokenizer_sha256=_digest(tokenizer),
        tensor_layout_sha256=_digest([(t.name,t.shape,t.type_name) for t in source.tensors]))


def identify(model,expected_sha256):
    from tools.gguf_reader import GGUFFile
    model=Path(model)
    if model.is_symlink() or not model.is_file():raise ValueError('Source GGUF must be a regular file')
    if type(expected_sha256) is not str or not re.fullmatch('[0-9a-f]{64}',expected_sha256):
        raise ValueError('Explicit source GGUF SHA256 required')
    before=model.stat();source=GGUFFile(model);identity=validate_wen_source(source)
    with model.open('rb') as stream:actual=hashlib.file_digest(stream,'sha256').hexdigest()
    after=model.stat()
    if (before.st_size,before.st_mtime_ns,before.st_ctime_ns)!=(after.st_size,after.st_mtime_ns,after.st_ctime_ns):
        raise ValueError('Source GGUF changed during admission')
    if actual!=expected_sha256:raise ValueError('Source GGUF SHA256 mismatch')
    return dict(identity,source_sha256=actual,source_bytes=after.st_size)


def register_wen_aliases():
    # These internal interfaces were inspected at this exact installed version.
    # A future dependency update requires a new import/forward qualification.
    for package,version in [('transformers','5.20.2'),('gguf','0.19.0')]:
        if importlib.metadata.version(package)!=version:
            raise ValueError('Wen training import requires '+package+'=='+version)
    from transformers.integrations.gguf import gguf_config_mapping as configs
    from transformers.integrations.gguf import gguf_conversion_mapping as weights
    from transformers.integrations.gguf import gguf_tokenizer_mapping as tokenizers
    configs.GGUF_CONFIG_ARCHS['wen1']=partial(configs._qwen35_config,prefix='wen1')
    weights.GGUF_ARCHS['wen1']=weights.GGUF_ARCHS['qwen35']
    tokenizers.GGUF_PRE_TOKENIZER_SPLITS['wen1']=tokenizers.GGUF_PRE_TOKENIZER_SPLITS['qwen35']


def load_wen(model,expected_sha256,*,device='cpu'):
    identity=identify(model,expected_sha256)
    register_wen_aliases()
    import torch
    from transformers import AutoModelForCausalLM,GgufConfig
    model=Path(model).resolve(strict=True)
    loaded,information=AutoModelForCausalLM.from_pretrained(str(model.parent),gguf_file=model.name,
        quantization_config=GgufConfig(dequantize=True),dtype=torch.bfloat16,
        device_map=device,attn_implementation='eager',local_files_only=True,
        output_loading_info=True)
    if information.get('missing_keys') or information.get('unexpected_keys') or information.get('mismatched_keys'):
        raise ValueError('GGUF reconstruction did not bind every model parameter: '+repr(information))
    if loaded.config.vocab_size!=identity['vocab_size']:
        raise ValueError('Reconstructed model vocabulary differs')
    loaded.eval()
    identity.update(training_model_type=loaded.config.model_type,
        config_sha256=_digest(loaded.config.to_dict()),
        packages={name:importlib.metadata.version(name) for name in ('torch','transformers','gguf','safetensors')},
        numerical_qualification='not_yet_compared_to_native')
    return loaded,identity


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',type=Path,required=True)
    parser.add_argument('--expected-sha256',required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--device',default='cpu')
    args=parser.parse_args()
    args.output.mkdir(mode=0o700,parents=True,exist_ok=False)
    model,identity=load_wen(args.model,args.expected_sha256,device=args.device)
    model.save_pretrained(args.output,safe_serialization=True,max_shard_size='4GB')
    from tools.gguf_reader import GGUFFile
    # The exact custom tokenizer remains the authority for training input IDs;
    # no upstream tokenizer or encoder assets are fetched as substitutes.
    metadata=GGUFFile(args.model).metadata
    tokenizer={key:value for key,value in metadata.items() if key.startswith('tokenizer.')}
    (args.output/'wen-tokenizer-metadata.json').write_text(json.dumps(tokenizer,ensure_ascii=True))
    identity['files_sha256']={}
    for path in args.output.iterdir():
        if path.is_file():
            with path.open('rb') as stream:
                identity['files_sha256'][path.name]=hashlib.file_digest(stream,'sha256').hexdigest()
    (args.output/'wen-reconstruction.json').write_text(json.dumps(identity,indent=2)+'\n')
    for path in args.output.iterdir():
        if path.is_file():path.chmod(0o600)
    print(json.dumps(identity,sort_keys=True))


if __name__=='__main__':main()
