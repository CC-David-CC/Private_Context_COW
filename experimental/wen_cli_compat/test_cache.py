import unittest
from types import SimpleNamespace
from concurrent.futures import Future
from cache import PrefixIndex, CachedCow


def ready(value):
    f = Future(); f.set_result(value); return f


class Adapter:
    def __init__(self):
        self.snapshots, self.heads, self.created, self.generated = {}, {}, [], []
    def create(self, name, ids, operation, cancelled=None):
        snapshot = 's' + str(len(self.snapshots))
        self.snapshots[snapshot] = tuple(ids)
        self.created.append(tuple(ids))
        return SimpleNamespace(durable=ready(dict(snapshot=snapshot)))
    def fork(self, name, snapshot):
        self.heads[name] = snapshot
    def head(self, name):
        return SimpleNamespace(revision=0)
    def generate(self, branch, ids, revision, operation, max_tokens, **kwargs):
        source = self.snapshots[self.heads[branch]]
        assert tuple(ids[:len(source)]) == source
        self.generated.append((branch, source, tuple(ids[len(source):])))
        return SimpleNamespace(durable=ready(None))


class Tests(unittest.TestCase):
    def test_exact_tokens_and_identity_not_text_or_question_match(self):
        idx = PrefixIndex(); idx.remember(('model', 'adapter', 'profile', 'renderer'), (1, 2), 'a')
        self.assertIsNone(idx.find(('other', 'adapter', 'profile', 'renderer'), (1, 2, 3)))
        self.assertIsNone(idx.find(('model', 'new-adapter', 'profile', 'renderer'), (1, 2, 3)))
        self.assertIsNone(idx.find(('model', 'adapter', 'profile', 'renderer'), (1, 9, 3)))
        self.assertEqual(idx.find(('model', 'adapter', 'profile', 'renderer'), (1, 2, 3)).snapshot, 'a')

    def test_longest_valid_prefix_and_bounded_lru(self):
        idx = PrefixIndex(limit=2, token_limit=6)
        idx.remember(('a',), (1,), 'a'); idx.remember(('a',), (1, 2), 'b')
        self.assertEqual(idx.find(('a',), (1, 2, 3)).snapshot, 'b')
        idx.remember(('a',), (9, 8, 7), 'c')
        self.assertIsNone(idx.find(('a',), (1, 7)))
        self.assertEqual(len(idx.entries), 2)

    def test_continuations_fork_same_immutable_source(self):
        a = Adapter(); c = CachedCow(a, ('model', 'profile', 'renderer'))
        b1, _, reused = c.start((1, 2, 3), (1, 2), 4)
        b2, _, reused2 = c.start((1, 2, 4), (1, 2), 4)
        self.assertNotEqual(b1, b2)
        self.assertEqual((reused, reused2), (0, 2))
        self.assertEqual(a.created, [(1, 2)])
        self.assertEqual([r[1:] for r in a.generated], [((1, 2), (3,)), ((1, 2), (4,))])

    def test_token_boundary_change_rebuilds_and_request_budget_is_bounded(self):
        a = Adapter(); c = CachedCow(a, ('a',), max_requests=1)
        c.start((1, 7), (1, 2), 3)
        self.assertEqual(a.created, [(1, 7)])
        with self.assertRaises(ValueError): c.start((1, 7), (), 3)


if __name__ == '__main__': unittest.main()
