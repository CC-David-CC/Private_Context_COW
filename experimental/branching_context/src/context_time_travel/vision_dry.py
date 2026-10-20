"""Bounded derived DRY input over canonical mixed spans; no stored state."""
from .input_spans import InputHistory
import hashlib
import json
import math
import struct
from .store import IntegrityError, InvalidOperation

PROTOCOL='mixed-text-runs-dry-u53-v1'
SAMPLING_PROTOCOL='mixed-text-runs-sampling-u53-v2'


def supports_sampling(backend,protocol):
    ready=getattr(backend,'_ready',{})
    families=ready.get('vision_sampling_protocols',())
    return (backend.profile['execution'].get('vision_dry_protocol')==PROTOCOL
        and ready.get('vision_dry_protocol')==PROTOCOL
        and ready.get('vision_dry_sampling_protocol')==SAMPLING_PROTOCOL
        and type(families) in (list,tuple) and protocol in families)


def text_window(history,last_n):
    if not isinstance(history,InputHistory) or type(last_n) is not int or not 0<=last_n<=2**32-1:
        raise InvalidOperation('Mixed DRY needs canonical inputs and a uint32 window')
    spans=history.spans
    total=sum(len(span['tokens']) for span in spans if span['kind']=='text')
    if last_n==0:return dict(tokens=(),boundaries=(),text_count=total)
    offset=max(0,total-last_n);position=0;tokens=[];boundaries=[]
    for span in spans:
        if span['kind']=='image':
            if position>=offset and (not boundaries or boundaries[-1]!=position-offset):
                boundaries.append(position-offset)
        else:
            run=span['tokens'];skip=min(len(run),max(0,offset-position))
            tokens.extend(run[skip:]);position+=len(run)
    return dict(tokens=tuple(tokens),boundaries=tuple(boundaries),text_count=total)


def sampling_window(backend,lease,policy,history,breakers,protocol):
    """Validate the actual text tail before creating any transport files."""
    backend._check(lease)
    if (not supports_sampling(backend,protocol) or not isinstance(lease.inputs,InputHistory)
            or len(lease.inputs)!=lease.consumed):
        raise InvalidOperation('Mixed DRY entropy sampling requires the checked native text-run command')
    window=text_window(lease.inputs,policy['dry_penalty_last_n'])
    if (type(history) not in (list,tuple) or tuple(history)!=window['tokens']
            or type(breakers) not in (list,tuple)
            or any(type(t) is not int or not 0<=t<backend.vocab_size for t in (*history,*breakers))
            or list(breakers)!=sorted(set(breakers)) or not set(breakers)<=set(history)):
        raise InvalidOperation('Mixed DRY tail differs from canonical actual text')
    return window


def sampling_request(backend,lease,command,window):
    """Reuse ordinary sampler receipt checks after verifying image bindings."""
    # VC07 deferred: add each proposal-family command only with exact native
    # barrier receipts and its existing RNG/feedback receipt checks. Do not map
    # V5/V6/V7 here merely because a CPU-only handler fixture compiles.
    name,arguments=command.split(' ',1)
    if name!='sample_mask_v4':raise InvalidOperation('Unqualified mixed DRY sampler command')
    response=backend._request('sample_mask_image_v4 '+arguments)
    expected=dict(text_tokens=window['text_count'],dry_boundary_count=len(window['boundaries']))
    boundaries=response.get('dry_boundaries')
    if (response.get('vision_dry_sampling_protocol')!=SAMPLING_PROTOCOL
            or any(type(response.get(k)) is not int or response[k]!=v for k,v in expected.items())
            or type(boundaries) is not list or any(type(p) is not int for p in boundaries)
            or boundaries!=list(window['boundaries'])):
        raise IntegrityError('Native mixed DRY receipt differs from exact image boundaries or text extent')
    from .strata_backend import _trace
    _trace('CTT_NATIVE_TRACE',dict(pid=backend.process.pid,sampling_protocol=SAMPLING_PROTOCOL,
        selector_protocol=response.get('protocol'),consumed=lease.consumed,
        selected_token=response.get('token'),native_receipt=dict(response)))
    return response


def sample(backend,lease,mask,policy,draw53,history,breakers):
    """Checked read-only command; native derives the same window independently."""
    backend._check(lease)
    if (backend.profile['execution'].get('vision_dry_protocol')!=PROTOCOL
            or backend._ready.get('vision_dry_protocol')!=PROTOCOL
            or not isinstance(lease.inputs,InputHistory) or len(lease.inputs)!=lease.consumed):
        raise InvalidOperation('Mixed DRY requires its pinned native text-run protocol')
    window=text_window(lease.inputs,policy['dry_penalty_last_n'])
    if (type(history) not in (tuple,list) or tuple(history)!=window['tokens']
            or type(breakers) not in (tuple,list)
            or any(type(t) is not int or not 0<=t<backend.vocab_size for t in (*history,*breakers))
            or list(breakers)!=sorted(set(breakers)) or not set(breakers)<=set(history)):
        raise InvalidOperation('Mixed DRY tail differs from canonical actual text')
    backend._sequence+=1
    path=backend._directory/('mixed-dry-'+str(backend._sequence)+'.mask')
    history_path=path.with_suffix('.history');breakers_path=path.with_suffix('.breakers')
    try:
        path.write_bytes(mask)
        history_path.write_bytes(struct.pack('<'+'I'*len(history),*history))
        breakers_path.write_bytes(struct.pack('<'+'I'*len(breakers),*breakers))
        fields=[lease.consumed,policy['temperature'],policy['min_p'],policy['top_k'],policy['top_p'],draw53,
            json.dumps(str(path)),policy['dry_multiplier'],policy['dry_base'],policy['dry_allowed_length'],
            policy['dry_penalty_last_n'],json.dumps(str(history_path)),json.dumps(str(breakers_path))]
        response=backend._request('sample_mask_image_dry_v1 '+' '.join(map(str,fields)))
    finally:
        for file in (path,history_path,breakers_path):file.unlink(missing_ok=True)
    legal=int.from_bytes(mask,'little').bit_count()
    expected=dict(consumed=lease.consumed,logits_count=backend.vocab_size,mask_bytes=len(mask),
        legal_count=legal,draw53=draw53,top_k=policy['top_k'],dry_history_count=len(history),
        dry_breaker_count=len(breakers),dry_boundary_count=len(window['boundaries']),text_tokens=window['text_count'],
        dry_allowed_length=policy['dry_allowed_length'],dry_penalty_last_n=policy['dry_penalty_last_n'])
    if (response.get('protocol')!=PROTOCOL or response.get('operator')!='full-vocab-min-p'
            or any(type(response.get(k)) is not int or response[k]!=v for k,v in expected.items())
            or any(type(response.get(k)) not in (int,float) or not math.isfinite(response[k]) or response[k]!=policy[k]
                for k in ('temperature','min_p','top_p','dry_multiplier','dry_base'))):
        raise IntegrityError('Native mixed DRY receipt differs from canonical input or policy')
    token=response.get('token');kept=response.get('kept_count');affected=response.get('dry_penalized_candidates')
    if (type(token) is not int or not 0<=token<backend.vocab_size or not(mask[token>>3]&(1<<(token&7)))
            or type(kept) is not int or not 1<=kept<=legal
            or type(affected) is not int or not 0<=affected<=len(set(history))):
        raise IntegrityError('Native mixed DRY receipt has invalid support')
    from .strata_backend import _trace
    _trace('CTT_NATIVE_TRACE',dict(pid=backend.process.pid,selected_token=token,sampling_protocol=PROTOCOL,
        consumed=lease.consumed,draw53=draw53,text_tokens=window['text_count'],dry_boundaries=list(window['boundaries']),
        dry_history_sha256=hashlib.sha256(struct.pack('<'+'I'*len(history),*history)).hexdigest(),dry_penalized_candidates=affected))
    return token
