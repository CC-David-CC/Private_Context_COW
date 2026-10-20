"""Checked native capabilities for the incremental mixed-controller campaign.

This module owns no controller state and implements no sampling mathematics.
Extended image samplers remain closed until their image-history tests exist.
"""
from .min_p import (supports_policy,dry_enabled,entropy_enabled,xtc_enabled,
                    mirostat_enabled,adaptive_enabled,SAMPLING_PROTOCOL_V4)
from .store import InvalidOperation
from .vision_execution import RESUME_CONTRACT
from .sampling_proposal import (XTC_PROPOSAL_PROTOCOL,MIROSTAT2_PROPOSAL_PROTOCOL,
                                ADAPTIVE_P_PROPOSAL_PROTOCOL,proposal_protocol_supported)


def has_entropy_command(backend):
    """V4 is independent of history when DRY is off; require its native query."""
    ready=getattr(backend,'_ready',{})
    return (ready.get('sampling_protocol')==SAMPLING_PROTOCOL_V4
        or ready.get('sampling_v4_protocol')==SAMPLING_PROTOCOL_V4)


def has_xtc_command(backend):
    ready=getattr(backend,'_ready',{})
    return (ready.get('sampling_protocol')==XTC_PROPOSAL_PROTOCOL
        or ready.get('sampling_v5_protocol')==XTC_PROPOSAL_PROTOCOL)


def has_mirostat_command(backend):
    ready=getattr(backend,'_ready',{})
    return ((ready.get('sampling_protocol')==MIROSTAT2_PROPOSAL_PROTOCOL
            or ready.get('mirostat2_protocol')==MIROSTAT2_PROPOSAL_PROTOCOL)
        and ready.get('mirostat2_feedback_protocol')=='mirostat2-state-v1'
        and type(ready.get('mirostat2_feedback_version')) is int
        and ready.get('mirostat2_feedback_version')==1)


def has_adaptive_command(backend):
    ready=getattr(backend,'_ready',{})
    return (ready.get('adaptive_p_protocol')==ADAPTIVE_P_PROPOSAL_PROTOCOL
        and ready.get('adaptive_p_feedback_protocol')=='adaptive-p-state-v1'
        and type(ready.get('adaptive_p_feedback_version')) is int
        and ready.get('adaptive_p_feedback_version')==1)


def supports_proposal_sampling(execution,backend,policy):
    # With DRY off, V5/V6/V7 selection depends on the frontier and canonical
    # controller, not image pad IDs or an invented repetition history.
    # VC07 deferred: image DRY+XTC/Mirostat/Adaptive needs independently derived
    # text-run barriers, a queried native family, transport/controller tests and
    # GPU restart evidence. The isolated native V5 fixture alone cannot open
    # this admission guard. Keep unsupported combinations explicitly rejected.
    # DRY+XTC can help explore different learned answers: DRY discourages repeats
    # and XTC sometimes removes obvious candidates. This is sampled generation,
    # not direct weight extraction or a guarantee of correct answers. David does
    # not need the mixed-image combination yet. Its individual samplers and
    # text-only pipeline exist; a future PR can connect and qualify this path.
    command=(has_adaptive_command if adaptive_enabled(policy) else
             has_mirostat_command if mirostat_enabled(policy) else has_xtc_command)
    return (execution.get('resume_contract')==RESUME_CONTRACT
        and execution.get('selected_token_protocol')=='consumed-selected-receipt-v1'
        and not dry_enabled(policy)
        and command(backend)
        and callable(getattr(backend,'sample_selection_proposal',None))
        and proposal_protocol_supported(execution.get('sampling_protocol'),policy)
        and supports_policy(execution.get('sampling_protocol'),policy))


def require_proposal_sampling(execution,backend,policy):
    if not supports_proposal_sampling(execution,backend,policy):
        raise InvalidOperation('Mixed-input proposal sampler/controller is unavailable')


def supports_basic_sampling(execution,backend,policy=None):
    policy=(dict(temperature=0.,min_p=0.,top_k=0,top_p=1.,repetition_penalty=1.)
            if policy is None else policy)
    if any(check(policy) for check in (adaptive_enabled,mirostat_enabled,xtc_enabled)):
        return supports_proposal_sampling(execution,backend,policy)
    if entropy_enabled(policy):
        if not has_entropy_command(backend):return False
        if dry_enabled(policy):
            from .vision_dry import supports_sampling
            if not supports_sampling(backend,SAMPLING_PROTOCOL_V4):return False
    if dry_enabled(policy):
        from .vision_dry import PROTOCOL
        if execution.get('vision_dry_protocol')!=PROTOCOL:return False
    return (execution.get('resume_contract')==RESUME_CONTRACT
        and execution.get('selected_token_protocol')=='consumed-selected-receipt-v1'
        and callable(getattr(backend,'sample_legal',None))
        and supports_policy(execution.get('sampling_protocol'),policy))


def require_basic_sampling(execution,backend,policy=None):
    if not supports_basic_sampling(execution,backend,policy):
        raise InvalidOperation('Mixed-input sampler/controller is unavailable; use qualified greedy controls')
