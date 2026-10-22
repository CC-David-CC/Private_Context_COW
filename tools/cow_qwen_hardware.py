"""Bind a dense-model profile to its compiled GPU backend and observed device.

CUDA keeps its previous profile fields. HIP is initially qualified only for a
single AMD GPU; inventory/admission tests do not substitute for GPU inference.
"""
import json
from pathlib import Path
import platform
import subprocess


def execution_hardware(build):
    names=[Path(path).name for path in build['sha256']]
    cuda=any(name.startswith('libggml-cuda.so') for name in names)
    hip=any(name.startswith('libggml-hip.so') for name in names)
    if cuda==hip:raise ValueError('Dense execution requires exactly one pinned GPU backend')
    if cuda:
        gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name,uuid,driver_version','--format=csv,noheader'],text=True).strip()
        if not gpu:raise ValueError('CUDA GPU inventory is empty')
        return dict(host=platform.node(),gpu=gpu),{'CUDA_VISIBLE_DEVICES':'0'}
    raw=subprocess.check_output(['rocm-smi','--showproductname','--showuniqueid','--showdriverversion','--json'],text=True)
    try:rows=json.loads(raw)
    except (ValueError,TypeError) as error:raise ValueError('Malformed HIP inventory') from error
    if not isinstance(rows,dict):raise ValueError('Malformed HIP inventory')
    cards=[key for key in rows if key.startswith('card')]
    if cards!=['card0']:raise ValueError('Dense HIP preview requires exactly one AMD GPU at card0')
    card=rows['card0'];system=rows.get('system',{})
    if not isinstance(system,dict):raise ValueError('Malformed HIP inventory driver fields')
    driver=system.get('Driver version')
    if (not isinstance(card,dict) or 'AMD' not in str(card.get('Card Vendor',''))
            or not card.get('Unique ID') or not driver):
        raise ValueError('HIP inventory requires AMD device, unique identity and driver')
    runtime=subprocess.check_output(['hipconfig','--version'],text=True).strip()
    if not runtime:raise ValueError('HIP runtime version is unavailable')
    return dict(host=platform.node(),gpu=json.dumps(rows,sort_keys=True),
        gpu_backend='hip',hip_runtime=runtime),{'HIP_VISIBLE_DEVICES':'0','ROCR_VISIBLE_DEVICES':'0'}
