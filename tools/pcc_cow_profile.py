"""Owner-side identity binding for an existing dense COW execution profile.

No model loading, GPU selection, source qualification, or client authority.
The ordinary COW validator/admission still verifies the resulting profile.
"""
from copy import deepcopy
from pathlib import Path
import hashlib
import json


def execution_identity(*, native_build, execution, hardware):
    """Match NativeBackend's established serialization, not S3 record digests."""
    value=dict(build=native_build,execution=execution,hardware=hardware)
    return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()


def bind_dense_profile(template, *, model_path, model_sha256, gpu_index, gpu_uuid):
    """Bind immutable weights and explicit physical placement to a copied profile.

    Inputs are owner observations, not untrusted model-selected paths/grants.
    This builds identities only; it does not claim the file matches its hash,
    the GPU exists, memory is available, or the build is qualified.
    """
    if (template.get('backend')!='llama-dense-llama.cpp'
            or template['execution'].get('state_format')!='llama-dense-frontier-v1'
            or template['execution'].get('kv_dtype')!='FP32'
            or template['execution'].get('device_binding')!='cuda-single-device-v1'):
        raise ValueError('Existing dense FP32 single-device COW profile required')
    if (not isinstance(model_path,(str,Path)) or not Path(model_path).is_absolute()
            or type(model_sha256) is not str or len(model_sha256)!=64
            or any(c not in '0123456789abcdef' for c in model_sha256)
            or type(gpu_index) is not int or gpu_index<0
            or type(gpu_uuid) is not str or not gpu_uuid.startswith('GPU-')
            or len(gpu_uuid)<=4 or any(c.isspace() for c in gpu_uuid)):
        raise ValueError('Explicit absolute model path, SHA256 and physical GPU observation required')
    result=deepcopy(template)
    result.update(model_path=str(Path(model_path)),model_weights_sha256=model_sha256)
    contract=dict(weights_sha256=model_sha256,architecture='llama',state_kind='llama-dense-frontier-v1',
                  positions='absolute-zero-based',kv_dtype='FP32',input_kind='token_ids')
    result['model_contract_id']=hashlib.sha256(json.dumps(contract,sort_keys=True).encode()).hexdigest()
    result['execution']['GPU']=[gpu_index]
    result['execution']['environment']['CUDA_VISIBLE_DEVICES']=str(gpu_index)
    result['hardware']['gpu_uuid']=gpu_uuid
    result['execution_profile_id']=execution_identity(native_build=result['native_build'],
        execution=result['execution'],hardware=result['hardware'])
    return result
