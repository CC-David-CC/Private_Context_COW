"""Private, consume-bound selection proposals for versioned samplers.

This module defines controller plumbing only. It does not implement XTC or
enable a public sampling mode. A proposal is transient request state and must
never be serialized into a checkpoint.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
import math
from types import MappingProxyType
from typing import Any

from .min_p import draw53
from .store import IntegrityError, InvalidOperation

DRAW_LIMIT = 1 << 128
MAX_PROPOSAL_DRAWS = 2
XTC_PROPOSAL_PROTOCOL = "full-vocab-xtc-min-p-u53-v5"
MIROSTAT2_PROPOSAL_PROTOCOL = "full-vocab-mirostat2-u53-v6"
ADAPTIVE_P_PROPOSAL_PROTOCOL = "full-vocab-adaptive-p-u53-v7"


def proposal_protocol_supported(profile_protocol: str | None, policy: dict) -> bool:
    """Whether a pinned native profile has the proposal family this policy needs."""
    from .min_p import (
        SAMPLING_PROTOCOL_V6, SAMPLING_PROTOCOL_V7,
        adaptive_enabled, mirostat_enabled, xtc_enabled,
    )
    if adaptive_enabled(policy):
        return profile_protocol == ADAPTIVE_P_PROPOSAL_PROTOCOL
    if mirostat_enabled(policy):
        return profile_protocol in (MIROSTAT2_PROPOSAL_PROTOCOL, SAMPLING_PROTOCOL_V7)
    if xtc_enabled(policy):
        return profile_protocol in (XTC_PROPOSAL_PROTOCOL, SAMPLING_PROTOCOL_V6, SAMPLING_PROTOCOL_V7)
    return profile_protocol in (XTC_PROPOSAL_PROTOCOL, SAMPLING_PROTOCOL_V6, SAMPLING_PROTOCOL_V7)


@dataclass(frozen=True)
class SelectionProposal:
    """A selected token and its uncommitted draw accounting.

    ``feedback_proposal`` is transient. Mirostat v2 proposals carry a detached,
    read-only versioned state envelope; it is copied into the canonical
    controller only after the matching selected token is consumed.
    """

    token: int
    draw_start: int
    draw_count: int
    feedback_proposal: Any
    source_digest: str


def controller_digest(state: dict) -> str:
    """Hash the complete current controller to bind a proposal to its source."""
    try:
        payload = json.dumps(state, ensure_ascii=False, sort_keys=True,
                             separators=(",", ":"), allow_nan=False).encode("utf-8")
    except (TypeError, ValueError, UnicodeError) as exc:
        raise IntegrityError("Proposal source controller is not canonical JSON") from exc
    return hashlib.sha256(payload).hexdigest()


def proposal_source_digest(base_digest: str, epoch: int, next_draw: int) -> str:
    """Bind one proposal to a session-local epoch without rehashing its state.

    ``base_digest`` is computed once when a detached controller is restored or
    configured. Sessions own the mutable controller privately, so a monotonic
    consumed-token epoch plus the RNG cursor identifies each subsequent
    selection boundary in constant time, even when ``policy.allowed`` is very
    large.
    """
    if (not isinstance(base_digest, str) or len(base_digest) != 64
            or any(ch not in "0123456789abcdef" for ch in base_digest)
            or type(epoch) is not int or epoch < 0
            or type(next_draw) is not int or not 0 <= next_draw < DRAW_LIMIT):
        raise IntegrityError("Invalid request-local proposal binding")
    payload = (b"selection-proposal-binding-v1\0" + bytes.fromhex(base_digest)
               + b"\0" + str(epoch).encode("ascii") + b"\0"
               + str(next_draw).encode("ascii"))
    return hashlib.sha256(payload).hexdigest()


def draw_candidates(state: dict) -> tuple[int, ...]:
    """Return at most two deterministic u53 candidates without advancing state."""
    _require_controller(state)
    start = state.get("next_draw")
    if type(start) is not int or not 0 <= start < DRAW_LIMIT:
        raise IntegrityError("Invalid proposal draw cursor")
    _, maximum = _draw_count_bounds(state)
    if maximum == 0:
        return ()
    available = min(MAX_PROPOSAL_DRAWS, DRAW_LIMIT - 1 - start)
    if available <= 0:
        raise IntegrityError("Proposal draw cursor is exhausted")
    try:
        return tuple(draw53(state["stream_key"], start + offset)
                     for offset in range(available))
    except (KeyError, TypeError, ValueError) as exc:
        raise IntegrityError("Invalid proposal draw source") from exc


def validate_proposal(state: dict, mask: bytes, proposal: SelectionProposal,
                      *, available_draws: int | None = None,
                      expected_source_digest: str | None = None) -> SelectionProposal:
    """Check token, source identity, and bounded draw range without mutation."""
    _require_controller(state)
    allowed = state.get("policy", {}).get("allowed")
    if (type(mask) is not bytes or not mask or not isinstance(allowed, list)
            or len(mask) != (len(allowed) + 7) // 8
            or not any(mask)):
        raise IntegrityError("Proposal requires a nonempty immutable support mask")
    remainder = len(allowed) & 7
    if remainder and mask[-1] & (0xFF ^ ((1 << remainder) - 1)):
        raise IntegrityError("Proposal support mask has nonzero padding bits")
    if type(proposal) is not SelectionProposal:
        raise IntegrityError("Selector did not return a bound selection proposal")
    if (type(proposal.token) is not int or not 0 <= proposal.token < len(mask) * 8
            or not mask[proposal.token >> 3] & (1 << (proposal.token & 7))):
        raise IntegrityError("Proposal token is outside legal support")
    if (type(proposal.draw_start) is not int
            or proposal.draw_start != state.get("next_draw")):
        raise IntegrityError("Proposal draw start differs from controller cursor")
    minimum_draws, maximum_draws = _draw_count_bounds(state)
    if (type(proposal.draw_count) is not int
            or not minimum_draws <= proposal.draw_count <= maximum_draws):
        raise IntegrityError("Proposal draw count is outside the supported range")
    if available_draws is not None and (
            type(available_draws) is not int or proposal.draw_count > available_draws):
        raise IntegrityError("Proposal consumes unavailable draw candidates")
    if proposal.draw_start + proposal.draw_count >= DRAW_LIMIT:
        raise IntegrityError("Proposal would exhaust the controller cursor")
    mirostat = _mirostat_active(state)
    adaptive = _adaptive_active(state)
    feedback = proposal.feedback_proposal
    if mirostat:
        feedback = _validated_mirostat_state(feedback)
        proposal = replace(proposal, feedback_proposal=MappingProxyType(feedback))
    elif adaptive:
        feedback = _validated_adaptive_state(feedback)
        proposal = replace(proposal, feedback_proposal=MappingProxyType(feedback))
    elif feedback is not None or state.get("sampler_state") is not None:
        raise IntegrityError("Unexpected feedback state for a stateless sampler")
    digest = proposal.source_digest
    expected = (controller_digest(state) if expected_source_digest is None
                else expected_source_digest)
    if (not isinstance(digest, str) or len(digest) != 64
            or any(ch not in "0123456789abcdef" for ch in digest)
            or digest != expected):
        raise IntegrityError("Proposal is not bound to this controller state")
    return proposal


def commit_proposal(state: dict, proposal: SelectionProposal, token: int, *,
                    expected_source_digest: str | None = None) -> dict:
    """Return a shallow private candidate with the consumed proposal's draw range.

    Session callers already own a detached controller and pass its O(1)
    request-local binding. Standalone callers may omit it and receive a full
    source-controller binding check.
    """
    _require_controller(state)
    if type(proposal) is not SelectionProposal:
        raise IntegrityError("Commit requires a checked selection proposal")
    if type(token) is not int or token != proposal.token:
        raise IntegrityError("Consumed token differs from selection proposal")
    allowed = state.get("policy", {}).get("allowed")
    if (not isinstance(allowed, list) or type(token) is not int
            or not 0 <= token < len(allowed)):
        raise IntegrityError("Proposal token is outside the controller vocabulary")
    if (type(proposal.draw_start) is not int
            or proposal.draw_start != state.get("next_draw")):
        raise IntegrityError("Proposal is stale for this controller state")
    if (not isinstance(proposal.source_digest, str) or len(proposal.source_digest) != 64
            or any(ch not in "0123456789abcdef" for ch in proposal.source_digest)
            or proposal.source_digest != (controller_digest(state)
                                          if expected_source_digest is None
                                          else expected_source_digest)):
        raise IntegrityError("Proposal source controller changed before commit")
    if _mirostat_active(state):
        candidate_state = _validated_mirostat_state(proposal.feedback_proposal)
    elif _adaptive_active(state):
        candidate_state = _validated_adaptive_state(proposal.feedback_proposal)
    elif proposal.feedback_proposal is not None or state.get("sampler_state") is not None:
        raise IntegrityError("Unexpected feedback state for a stateless sampler")
    else:
        candidate_state = None
    minimum_draws, maximum_draws = _draw_count_bounds(state)
    if (type(proposal.draw_count) is not int
            or not minimum_draws <= proposal.draw_count <= maximum_draws
            or proposal.draw_start + proposal.draw_count >= DRAW_LIMIT):
        raise IntegrityError("Invalid proposal draw range at commit")
    candidate = dict(state)
    candidate["next_draw"] = proposal.draw_start + proposal.draw_count
    if candidate_state is not None:
        candidate["sampler_state"] = candidate_state
    return candidate


def _require_controller(state: dict) -> None:
    if (not isinstance(state, dict) or type(state.get("schema")) is not int
            or state.get("schema") not in (4, 6) or state.get("status") != "active"):
        raise InvalidOperation("Proposal requires an active schema-4 controller")
    if _mirostat_active(state):
        _validated_mirostat_state(state.get("sampler_state"))
    elif _adaptive_active(state):
        _validated_adaptive_state(state.get("sampler_state"))
    elif state.get("sampler_state", object()) is not None:
        raise InvalidOperation("Stateless schema-4 controller must have null sampler state")


def _draw_count_bounds(state: dict) -> tuple[int, int]:
    temperature = state.get("policy", {}).get("temperature")
    if (type(temperature) not in (int, float) or not math.isfinite(temperature)
            or temperature < 0):
        raise IntegrityError("Invalid proposal temperature")
    if _mirostat_active(state) or _adaptive_active(state):
        return (1, 1)
    return (0, 0) if temperature == 0 else (1, MAX_PROPOSAL_DRAWS)


def _mirostat_active(state: dict) -> bool:
    from .min_p import mirostat_enabled
    return mirostat_enabled(state.get("policy", {}))


def _adaptive_active(state: dict) -> bool:
    from .min_p import adaptive_enabled
    return adaptive_enabled(state.get("policy", {}))


def _validated_mirostat_state(value):
    from .mirostat2 import validate_state
    try:
        return validate_state(value)
    except (TypeError, ValueError) as exc:
        raise IntegrityError("Invalid Mirostat v2 feedback state") from exc


def _validated_adaptive_state(value):
    from .adaptive_p import validate_state
    try:
        return validate_state(value)
    except (TypeError, ValueError) as exc:
        raise IntegrityError("Invalid Adaptive-P feedback state") from exc
