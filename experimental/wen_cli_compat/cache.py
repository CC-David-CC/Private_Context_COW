"""Ephemeral standard-API prefix lookup; COW remains the state authority."""
from collections import OrderedDict
from dataclasses import dataclass
import threading
import uuid


class SessionBudgets:
    """An explicit endpoint session bounds model calls across every CLI."""
    def __init__(self, calls=8, sessions=128):
        self.calls, self.sessions, self.counts = calls, sessions, {}
        self.lock = threading.Lock()

    def admit(self, session):
        if session is None:
            return
        with self.lock:
            if session not in self.counts and len(self.counts) >= self.sessions:
                raise ValueError('Experimental session count exhausted')
            count = self.counts.get(session, 0)
            if count >= self.calls:
                raise ValueError('Session model-request budget exhausted')
            self.counts[session] = count+1


@dataclass(frozen=True)
class Entry:
    identity: tuple
    tokens: tuple
    snapshot: str


class PrefixIndex:
    def __init__(self, limit=128, token_limit=1_000_000):
        self.limit, self.token_limit = limit, token_limit
        self.entries = OrderedDict()
        self.lock = threading.RLock()

    def remember(self, identity, tokens, snapshot):
        tokens = tuple(tokens)
        if not tokens or len(tokens) > self.token_limit:
            return
        key = (tuple(identity), snapshot)
        with self.lock:
            self.entries[key] = Entry(tuple(identity), tokens, snapshot)
            self.entries.move_to_end(key)
            while (len(self.entries) > self.limit or
                   sum(len(e.tokens) for e in self.entries.values()) > self.token_limit):
                self.entries.popitem(last=False)

    def find(self, identity, tokens):
        tokens = tuple(tokens)
        with self.lock:
            choices = [(k, e) for k, e in self.entries.items()
                       if e.identity == tuple(identity) and len(e.tokens) <= len(tokens)
                       and tokens[:len(e.tokens)] == e.tokens]
            if not choices:
                return None
            key, entry = max(choices, key=lambda pair: len(pair[1].tokens))
            self.entries.move_to_end(key)
            return entry


class CachedCow:
    """Every request forks a committed snapshot, including simultaneous siblings.

    This index is a lookup aid only. Canonical inputs, controllers, recurrent/KV
    checkpoints, verification and numerical eviction belong to the COW Store.
    The owning process uses a private TemporaryDirectory, not a persistent chat DB.
    """
    def __init__(self, adapter, identity, index=None, max_requests=256):
        self.adapter, self.identity = adapter, tuple(identity)
        self.index = index or PrefixIndex()
        self.lock = threading.RLock()
        self.requests, self.max_requests = 0, max_requests

    def start(self, full_ids, closed_ids, max_tokens, *, settings=None,
              cancelled=None, seed=None, text_stops=None):
        full_ids, closed_ids = tuple(full_ids), tuple(closed_ids)
        if not full_ids:
            raise ValueError('Empty rendered prompt')
        # Tokenization may merge at a frame boundary. Never use a text match.
        if not closed_ids or full_ids[:len(closed_ids)] != closed_ids:
            closed_ids = full_ids
        operation = 'compat_' + uuid.uuid4().hex
        branch = 'compat_' + uuid.uuid4().hex
        with self.lock:
            if self.requests >= self.max_requests:
                raise ValueError('Experimental service request budget exhausted; restart the service')
            self.requests += 1
            source = self.index.find(self.identity, full_ids)
            # A cached prefix may be longer than the proposed closed-message
            # root. In either case it is used only when its exact IDs match.
            reused = len(source.tokens) if source else 0
            if source is None:
                root = 'compat_root_' + uuid.uuid4().hex
                created = self.adapter.create(root, closed_ids, operation + '_root', cancelled=cancelled)
                receipt = created.durable.result()
                snapshot = receipt['snapshot']
                self.index.remember(self.identity, closed_ids, snapshot)
            else:
                snapshot = source.snapshot
            self.adapter.fork(branch, snapshot)
            head = self.adapter.head(branch)
        request = self.adapter.generate(branch, full_ids, head.revision, operation, max_tokens,
                                        settings=settings, cancelled=cancelled, seed=seed,
                                        text_stops=text_stops)
        return branch, request, reused

    def committed(self, branch, receipt):
        # Called only after durable success. Never index provisional output.
        history = self.adapter.history(branch)
        if not isinstance(history, (tuple, list)):
            raise ValueError('This experimental compatibility service accepts text tokens only')
        self.index.remember(self.identity, history, receipt['snapshot'])
