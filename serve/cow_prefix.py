"""Token-ID validation for optional native COW prompt reuse.

Callers pass the canonical stored input IDs and IDs from encoding the complete,
currently rendered prompt. Reuse is safe only when the latter begins with the
former exactly. Comparing prompt strings is not enough: encoding a longer text
can merge or otherwise change tokens at the old/new boundary.
"""


class RenderedPrefixMismatch(ValueError):
    """The fully rendered prompt does not preserve the canonical token prefix."""


def _token_ids(values, name):
    try:
        result = tuple(values)
    except TypeError as error:
        raise ValueError(f"{name} must be an iterable of nonnegative integer token IDs") from error
    if any(type(token) is not int or token < 0 for token in result):
        raise ValueError(f"{name} must contain only nonnegative integer token IDs")
    return result


def exact_append_suffix(canonical_token_ids, rendered_token_ids):
    """Return the exact newly rendered token suffix, or reject prompt reuse.

    `rendered_token_ids` must be produced by rendering and tokenizing the whole
    current prompt with the serving tokenizer. This helper intentionally has no
    text-prefix fallback: if the tokenizer changed segmentation across the
    boundary, the request must rebuild from a suitable root or use ordinary
    full-prompt serving.
    """
    canonical = _token_ids(canonical_token_ids, 'canonical history')
    rendered = _token_ids(rendered_token_ids, 'rendered prompt')
    if len(rendered) < len(canonical) or rendered[:len(canonical)] != canonical:
        raise RenderedPrefixMismatch(
            'fully rendered prompt token IDs do not preserve the exact canonical history prefix')
    return rendered[len(canonical):]


def exact_input_append_suffix(canonical, rendered):
    """Retain full mixed identities, admitting only new text continuation cells.

    Import lazily so the ordinary token-only helper remains standalone.
    Image creation/append must use the explicit artifact-aware input path.
    """
    from context_time_travel.input_spans import InputHistory, generated_text
    if (not isinstance(canonical,InputHistory) or not isinstance(rendered,InputHistory)
            or len(rendered)<len(canonical) or rendered.prefix(len(canonical))!=canonical):
        raise RenderedPrefixMismatch('Rendered mixed inputs do not preserve the canonical input prefix')
    return generated_text(rendered.segment(len(canonical)))
