"""One immutable native LoRA version bound to model and execution identities."""
import hashlib
import json
import math
from pathlib import Path
import re


def validate_adapter(profile):
    spec=profile.get('adapter')
    execution=profile['execution']
    contract=profile.get('model_contract',{})
    if spec is None:
        if execution.get('adapter') is not None or contract.get('adapters'):
            raise ValueError('Unbound adapter identity')
        return None
    if type(spec) is not dict or set(spec)!={'path','sha256','scale'}:
        raise ValueError('One explicit adapter path, SHA256 and scale required')
    scale=spec['scale']
    if type(scale) not in (int,float) or not math.isfinite(scale) or not 0<scale<=16:
        raise ValueError('Finite positive adapter scale at most 16 required')
    if type(spec['sha256']) is not str or not re.fullmatch('[a-f0-9]{64}',spec['sha256']):
        raise ValueError('Explicit adapter SHA256 required')
    identity=dict(sha256=spec['sha256'],scale=float(scale))
    if (execution.get('adapter')!=identity or contract.get('adapters')!=[identity]
            or hashlib.sha256(json.dumps(contract,sort_keys=True).encode()).hexdigest()!=profile['model_contract_id']):
        raise ValueError('Adapter model/execution identities disagree')
    if profile.get('native_build',{}).get('lora_binding')!='llama-adapter-per-lease-v1':
        raise ValueError('Rebuild native adapter for the checked LoRA binding')
    if type(spec['path']) is not str:raise ValueError('Absolute regular adapter file required')
    path=Path(spec['path'])
    if not path.is_absolute() or path.is_symlink() or not path.is_file() or path.resolve()!=path:
        raise ValueError('Absolute regular adapter file required')
    with path.open('rb') as stream:actual=hashlib.file_digest(stream,'sha256').hexdigest()
    if actual!=identity['sha256']:raise ValueError('Adapter SHA256 mismatch')
    return dict(path=str(path),**identity)
