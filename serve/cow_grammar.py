"""Load immutable server-owned GBNF and JSON Schema definitions.

This module does not accept client paths or own parser progress. Every start
and restore delegates to the existing :class:`GrammarEngine`, whose
controller binds parser identity and accepted-token progress to canonical
history.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import stat
from types import MappingProxyType

from context_time_travel.grammar_first import (
    GrammarEngine, LLGUIDANCE_VERSION, MAX_GRAMMAR_SOURCE_BYTES,
    normalize_grammar_source,
)
from context_time_travel.store import InvalidOperation


MAX_REGISTRY_BYTES = 4 * 1024 * 1024
MAX_REGISTRY_STATES = 64
MAX_TOTAL_GRAMMAR_SOURCE_BYTES = 3 * 1024 * 1024


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON object key")
        result[key] = value
    return result


def _reject_constant(value):
    raise ValueError("non-finite JSON number")


def _read_config(path: Path) -> bytes:
    try:
        if not stat.S_ISREG(path.lstat().st_mode):
            raise InvalidOperation("Grammar registry configuration must be a regular file")
    except InvalidOperation:
        raise
    except OSError as exc:
        raise InvalidOperation("Grammar registry configuration cannot be opened safely") from exc
    flags = (os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
             | getattr(os, "O_NONBLOCK", 0))
    try:
        descriptor = os.open(path, flags)
        with os.fdopen(descriptor, "rb") as stream:
            info = os.fstat(stream.fileno())
            if not stat.S_ISREG(info.st_mode):
                raise InvalidOperation("Grammar registry configuration must be a regular file")
            if info.st_size > MAX_REGISTRY_BYTES:
                raise InvalidOperation("Grammar registry configuration exceeds size limit")
            raw = stream.read(MAX_REGISTRY_BYTES + 1)
    except InvalidOperation:
        raise
    except OSError as exc:
        raise InvalidOperation("Grammar registry configuration must be a regular file and open safely") from exc
    if len(raw) > MAX_REGISTRY_BYTES:
        raise InvalidOperation("Grammar registry configuration exceeds size limit")
    return raw


@dataclass(frozen=True, slots=True)
class GrammarRegistry:
    """An immutable registry bound to one tokenizer and compiled engine."""

    MAX_CONFIG_BYTES = MAX_REGISTRY_BYTES

    _engine: GrammarEngine
    states: tuple[str, ...]
    engine_id: str
    spec_sha256: str
    tokenizer_sha256: str
    llguidance_version: str

    @classmethod
    def load(cls, config_path, tokenizer, eos_tokens):
        """Read and compile the startup registry from a host-supplied path.

        Registry definitions contain inline GBNF text or JSON Schema objects.
        Paths inside the JSON are never opened or resolved.
        """
        try:
            path = Path(config_path)
        except (TypeError, ValueError, OSError) as exc:
            raise InvalidOperation("Invalid grammar registry configuration path") from exc
        raw = _read_config(path)
        try:
            value = json.loads(raw.decode("utf-8"), object_pairs_hook=_unique_object,
                               parse_constant=_reject_constant)
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError, RecursionError) as exc:
            raise InvalidOperation("Grammar registry must be strict UTF-8 JSON without duplicate keys") from exc
        return cls.from_config(value, tokenizer, eos_tokens)

    @classmethod
    def from_config(cls, value, tokenizer, eos_tokens):
        """Compile a detached inline definition; never open client-supplied paths.

        This primitive adds no HTTP admission or mutable registry. Startup-file
        and inline compilation retain exactly the same engine identities.
        """
        try:
            raw = json.dumps(value, ensure_ascii=False, allow_nan=False).encode("utf-8")
        except (TypeError, ValueError, UnicodeError, RecursionError) as exc:
            raise InvalidOperation("Grammar configuration must be finite JSON") from exc
        if len(raw) > MAX_REGISTRY_BYTES:
            raise InvalidOperation("Grammar registry configuration exceeds size limit")
        value = json.loads(raw)
        if not isinstance(value, dict) or set(value) != {"schema", "states"}:
            raise InvalidOperation("Grammar registry must contain exactly schema and states")
        if type(value["schema"]) is not int or value["schema"] != 1:
            raise InvalidOperation("Unsupported grammar registry schema")
        definitions = value["states"]
        if not isinstance(definitions, dict) or not definitions or len(definitions) > MAX_REGISTRY_STATES:
            raise InvalidOperation("Grammar registry must define 1..64 named states")

        engine_states = {}
        total_source_bytes = 0
        for name, definition in definitions.items():
            if not isinstance(definition, dict) or set(definition) != {"format", "source"}:
                raise InvalidOperation("Each grammar state must contain only format and source")
            source, source_bytes = normalize_grammar_source(definition["format"], definition["source"])
            total_source_bytes += len(source_bytes)
            if total_source_bytes > MAX_TOTAL_GRAMMAR_SOURCE_BYTES:
                raise InvalidOperation("Total grammar source exceeds registry limit")
            engine_states[name] = MappingProxyType({
                "format": definition["format"], "source": source
            })

        engine = GrammarEngine(tokenizer, engine_states, eos_tokens=eos_tokens)
        return cls(engine, tuple(sorted(engine_states)), engine.engine_id,
                   engine.spec_sha256, engine.tokenizer_sha256,
                   LLGUIDANCE_VERSION)

    def start(self, agent_state, stream, history, vocab_size, *, sampling=None):
        if agent_state not in self.states:
            raise InvalidOperation("Unknown host-selected grammar state")
        return self._engine.start(agent_state, stream, history, vocab_size, sampling=sampling)

    @property
    def engine(self):
        """The immutable compiled engine used by the canonical controller path."""
        return self._engine

    def descriptor(self, agent_state):
        """Return an intent-safe identity reference, never parser progress/source."""
        if not isinstance(agent_state, str) or agent_state not in self.states:
            raise InvalidOperation("Unknown host-selected grammar state")
        return {
            "schema": 1,
            "agent_state": agent_state,
            "engine_id": self.engine_id,
            "spec_sha256": self.spec_sha256,
            "tokenizer_sha256": self.tokenizer_sha256,
            "llguidance_version": self.llguidance_version,
        }

    def matches_controller(self, controller):
        """Check this registry against the saved canonical grammar identity."""
        if not isinstance(controller, dict):
            return False
        constraint = controller.get("constraint")
        if not isinstance(constraint, dict):
            return False
        state = constraint.get("agent_state")
        try:
            descriptor = self.descriptor(state)
        except InvalidOperation:
            return False
        return all(constraint.get(key) == value for key, value in descriptor.items()
                   if key != "schema")

    def restore(self, controller_state, history):
        return self._engine.restore(controller_state, history)
