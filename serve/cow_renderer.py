"""Stable identity for the inputs that determine a rendered prompt's token IDs.

Compute this once after loading the tokenizer and presentation configuration. Keep
the resulting value with the loaded renderer; request handling should reuse it.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


_SCHEMA = "strata-cow-renderer-identity-v1"
_IMPLEMENTATION_FILES = {
    "serve.cow_prompt_encoder": "serve/cow_prompt_encoder.py",
    "serve.frontend": "serve/frontend.py",
    "serve.server": "serve/server.py",
    "tools.strata_tokenizer": "tools/strata_tokenizer.py",
}


def _canonical_bytes(value: Any) -> bytes:
    def validate(item: Any) -> None:
        if item is None or isinstance(item, (str, bool, int)):
            return
        if isinstance(item, float):
            if not math.isfinite(item):
                raise ValueError("renderer identity inputs cannot contain non-finite numbers")
            return
        if isinstance(item, Mapping):
            if any(not isinstance(key, str) for key in item):
                raise TypeError("renderer identity mappings require string keys")
            for child in item.values():
                validate(child)
            return
        if isinstance(item, (list, tuple)):
            for child in item:
                validate(child)
            return
        raise TypeError(f"unsupported renderer identity value: {type(item).__name__}")

    validate(value)
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _metadata_fingerprint(value: Any) -> str:
    return _sha256(_canonical_bytes(value))


def _implementation_hashes() -> dict[str, str]:
    root = Path(__file__).resolve().parents[1]
    hashes: dict[str, str] = {}
    for name, relative in _IMPLEMENTATION_FILES.items():
        path = root / relative
        try:
            source = path.read_bytes()
        except OSError as exc:
            raise ValueError(f"cannot fingerprint implementation {name}") from exc
        hashes[name] = _sha256(source)
    return hashes


def _tokenizer_metadata(tokenizer: Any) -> dict[str, Any]:
    required = ("tokens", "pre", "token_types", "special_ids", "ranks")
    for field in required:
        if not hasattr(tokenizer, field):
            raise ValueError(f"tokenizer metadata missing {field}")

    tokens = tokenizer.tokens
    if (not isinstance(tokens, Sequence) or isinstance(tokens, (str, bytes)) or not tokens
            or any(not isinstance(token, str) for token in tokens)
            or len(set(tokens)) != len(tokens)):
        raise ValueError("tokenizer tokens must be a nonempty unique string sequence")
    token_set = set(tokens)
    pre = tokenizer.pre
    if not isinstance(pre, str) or not pre:
        raise ValueError("tokenizer pre must be a nonempty string")

    token_types = tokenizer.token_types
    if (not isinstance(token_types, Sequence) or isinstance(token_types, (str, bytes))
            or len(token_types) != len(tokens)
            or any(not isinstance(kind, int) or isinstance(kind, bool) for kind in token_types)):
        raise ValueError("tokenizer token_types must be an integer for every token")

    special_ids = tokenizer.special_ids
    if not isinstance(special_ids, Mapping) or not special_ids:
        raise ValueError("tokenizer special_ids must be a nonempty mapping")
    normalized_specials: dict[str, int] = {}
    for key, token_id in special_ids.items():
        if (not isinstance(key, str) or not key.startswith("tokenizer.ggml.")
                or not key.endswith("_token_id")
                or not isinstance(token_id, int) or isinstance(token_id, bool)
                or not 0 <= token_id < len(tokens)):
            raise ValueError("tokenizer special_ids contains an invalid entry")
        normalized_specials[key] = token_id

    ranks = tokenizer.ranks
    if not isinstance(ranks, Mapping) or not ranks:
        raise ValueError("tokenizer ranks must be a nonempty mapping")
    normalized_ranks: list[list[Any]] = []
    seen_values: set[int] = set()
    for pair, rank in ranks.items():
        if (not isinstance(pair, tuple) or len(pair) != 2
                or any(not isinstance(piece, str) for piece in pair)
                or pair[0] not in token_set or pair[1] not in token_set
                or not isinstance(rank, int) or isinstance(rank, bool) or rank < 0
                or rank in seen_values):
            raise ValueError("tokenizer ranks contains an invalid merge entry")
        seen_values.add(rank)
        normalized_ranks.append([pair[0], pair[1], rank])
    normalized_ranks.sort(key=lambda item: (item[0], item[1]))

    return {
        "pre": pre,
        "tokens_sha256": _metadata_fingerprint(list(tokens)),
        "token_types_sha256": _metadata_fingerprint(list(token_types)),
        "special_ids_sha256": _metadata_fingerprint(normalized_specials),
        "ranks_sha256": _metadata_fingerprint(normalized_ranks),
        "token_count": len(tokens),
        "merge_count": len(normalized_ranks),
    }


def _dependency_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for distribution, key in (("regex", "regex"), ("Jinja2", "jinja2")):
        try:
            versions[key] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError as exc:
            raise ValueError(f"required renderer dependency missing: {key}") from exc
    return versions


def renderer_identity_receipt(tokenizer: Any, template_source: str,
                               template_kwargs: Mapping[str, Any], *,
                               model_contract_id: str) -> dict[str, Any]:
    """Return canonical, path-independent fingerprints for loaded render inputs."""
    if not isinstance(template_source, str) or not template_source:
        raise ValueError("template_source must be a nonempty exact string")
    if not isinstance(template_kwargs, Mapping):
        raise ValueError("template_kwargs must be a mapping")
    if not isinstance(model_contract_id, str) or not model_contract_id.strip():
        raise ValueError("model_contract_id must be a nonempty qualified contract")

    # Canonical encoding validates JSON compatibility and rejects non-string map keys,
    # NaN, and unsupported objects before a digest can silently omit their meaning.
    kwargs_bytes = _canonical_bytes(template_kwargs)
    return {
        "schema": _SCHEMA,
        "model_contract_id": model_contract_id,
        "tokenizer": _tokenizer_metadata(tokenizer),
        "template_source_sha256": _sha256(template_source.encode("utf-8")),
        "template_kwargs_sha256": _sha256(kwargs_bytes),
        "implementation_sha256": _implementation_hashes(),
        "dependencies": _dependency_versions(),
    }


def renderer_identity(tokenizer: Any, template_source: str,
                      template_kwargs: Mapping[str, Any], *,
                      model_contract_id: str) -> str:
    """Return the SHA-256 identity to compute once when a renderer is loaded."""
    receipt = renderer_identity_receipt(tokenizer, template_source, template_kwargs,
                                        model_contract_id=model_contract_id)
    return _sha256(_canonical_bytes(receipt))
