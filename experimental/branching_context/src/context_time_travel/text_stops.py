"""Bounded UTF-8 byte matching for canonical text stop controls.

The matcher is deliberately independent of tokenization and presentation.
Callers feed the complete bytes of each consumed token. It returns bytes safe
to display and retains only the longest suffix that could still begin a stop.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


STATE_SCHEMA = 1
MAX_STOP_SEQUENCES = 16
MAX_STOP_BYTES = 256
MAX_TOTAL_STOP_BYTES = 1024


def normalize_stop_sequences(values):
    """Validate and return detached, ordered UTF-8 stop strings."""
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes, bytearray)):
        raise ValueError("Text stops must be a sequence of strings")
    if len(values) > MAX_STOP_SEQUENCES:
        raise ValueError("Too many text stop sequences")
    result = []
    seen = set()
    total = 0
    for value in values:
        if not isinstance(value, str) or not value:
            raise ValueError("Text stop sequences must be nonempty strings")
        try:
            encoded = value.encode("utf-8", errors="strict")
        except UnicodeEncodeError as error:
            raise ValueError("Text stop sequence is not valid UTF-8") from error
        if len(encoded) > MAX_STOP_BYTES:
            raise ValueError("Text stop sequence exceeds the byte limit")
        if encoded in seen:
            raise ValueError("Duplicate text stop sequence")
        seen.add(encoded)
        total += len(encoded)
        if total > MAX_TOTAL_STOP_BYTES:
            raise ValueError("Text stop sequences exceed the total byte limit")
        result.append(value)
    return tuple(result)


def _valid_pending(pending, encoded):
    if not pending:
        return True
    return any(len(pending) < len(stop) and stop.startswith(pending) for stop in encoded)


@dataclass(frozen=True)
class FeedResult:
    """Bytes safe for display and whether a complete marker has stopped output."""

    safe_bytes: bytes
    stopped: bool
    matched_stop: str | None = None


class TextStopMatcher:
    """Incremental matcher with a bounded ambiguous suffix.

    If multiple markers complete on the same byte, the longest wins; ties of
    equal length follow declaration order. A completed marker is selected by
    earliest completion position, so token/chunk boundaries cannot change the
    result. Bytes after the marker in its completing token are consumed by the
    caller but are never returned for display.
    """

    def __init__(self, stop_sequences=(), *, pending=b"", stopped=False,
                 matched_stop=None, finished=False):
        self._stops = normalize_stop_sequences(stop_sequences)
        self._encoded = tuple(value.encode("utf-8") for value in self._stops)
        if type(pending) is not bytes or not _valid_pending(pending, self._encoded):
            raise ValueError("Pending text-stop bytes are not an ambiguous stop prefix")
        if type(stopped) is not bool or type(finished) is not bool:
            raise ValueError("Text-stop status flags must be boolean")
        if stopped:
            if (finished or pending or not isinstance(matched_stop, str)
                    or matched_stop not in self._stops):
                raise ValueError("Stopped text matcher requires its matched marker and no pending bytes")
        elif matched_stop is not None:
            raise ValueError("Active text matcher cannot have a matched marker")
        if finished and pending:
            raise ValueError("Finalized text matcher cannot have pending bytes")
        self._pending = pending
        self._stopped = stopped
        self._matched_stop = matched_stop
        self._finished = finished

    @property
    def stop_sequences(self):
        return self._stops

    @property
    def pending(self):
        return self._pending

    @property
    def stopped(self):
        return self._stopped

    @property
    def matched_stop(self):
        return self._matched_stop

    @property
    def finished(self):
        return self._finished

    def feed(self, piece, *, final=False):
        """Consume one complete tokenizer piece, optionally finalizing at EOS."""
        if type(piece) is not bytes or type(final) is not bool:
            raise ValueError("Text-stop input must be bytes and final must be boolean")
        if self._stopped:
            raise ValueError("Cannot feed a stopped text matcher")
        if self._finished:
            raise ValueError("Cannot feed a finalized text matcher")
        combined = self._pending + piece
        if self._encoded:
            matches = []
            for order, marker in enumerate(self._encoded):
                start = combined.find(marker)
                if start >= 0:
                    end = start + len(marker)
                    matches.append((end, -len(marker), order, start))
            if matches:
                _end, _negative_length, order, start = min(matches)
                self._pending = b""
                self._stopped = True
                self._matched_stop = self._stops[order]
                return FeedResult(combined[:start], True, self._matched_stop)

        if final:
            self._pending = b""
            self._finished = True
            return FeedResult(combined, False, None)

        keep = 0
        for stop in self._encoded:
            limit = min(len(combined), len(stop) - 1)
            for size in range(limit, keep, -1):
                if stop.startswith(combined[-size:]):
                    keep = size
                    break
        if keep:
            self._pending = combined[-keep:]
            return FeedResult(combined[:-keep], False, None)
        self._pending = b""
        return FeedResult(combined, False, None)

    def export_state(self):
        """Return the exact bounded canonical matcher descriptor."""
        return {
            "schema": STATE_SCHEMA,
            "stop_sequences": list(self._stops),
            "pending_hex": self._pending.hex(),
            "stopped": self._stopped,
            "matched_stop": self._matched_stop,
            "finished": self._finished,
        }

    @classmethod
    def restore(cls, state):
        if type(state) is not dict or set(state) != {
                "schema", "stop_sequences", "pending_hex", "stopped", "matched_stop", "finished"}:
            raise ValueError("Invalid text-stop state fields")
        if type(state.get("schema")) is not int or state["schema"] != STATE_SCHEMA:
            raise ValueError("Unsupported text-stop state schema")
        pending_hex = state.get("pending_hex")
        if (not isinstance(pending_hex, str) or len(pending_hex) > 2 * (MAX_STOP_BYTES - 1)
                or len(pending_hex) % 2
                or any(character not in "0123456789abcdef" for character in pending_hex)):
            raise ValueError("Invalid text-stop pending bytes")
        return cls(state["stop_sequences"], pending=bytes.fromhex(pending_hex),
                   stopped=state["stopped"], matched_stop=state["matched_stop"],
                   finished=state["finished"])
