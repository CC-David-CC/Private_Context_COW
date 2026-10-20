"""Controller value semantics and canonical reads on the performance path."""
import copy
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))
from context_time_travel import controller
from context_time_travel.store import Store, IntegrityError


class ControllerCopy(unittest.TestCase):
    def test_greedy_tie_and_returned_policy_are_independent(self):
        original = controller.start({'temperature': 0}, 's', (1, 1), 8)
        before = copy.deepcopy(original)
        token, selected = controller.choose((0., 3., 3., 0., 0., 0., 0., 0.), original)
        self.assertEqual(token, 1)
        selected['policy']['allowed'][0] = 7
        selected['seen']['1'] = 99
        self.assertEqual(original, before)
        consumed = controller.consume(original, 1, b'x', False)
        consumed['policy']['allowed'].clear()
        self.assertEqual(original, before)

    def test_penalty_grammar_and_rng_continue_exactly(self):
        policy = dict(temperature=0, repetition_penalty=2., grammar={
            'start': 'a', 'transitions': {'a': {'1': 'a', '2': 'a'}}})
        state = controller.start(policy, 's', (1,), 4)
        token, selected = controller.choose((99., 3., 2., 99.), state)
        self.assertEqual(token, 2)
        self.assertEqual(selected['next_draw'], 0)
        state = controller.start({'rejection_threshold': .1}, 'rng', (), 4)
        before = copy.deepcopy(state)
        self.assertEqual(controller.choose((1., 2., 3., 4.), state),
                         controller.choose((1., 2., 3., 4.), before))
        self.assertEqual(state, before)

    def test_nonfinite_allowed_logit_still_rejected(self):
        state = controller.start({'temperature': 0}, 's', (), 3)
        with self.assertRaises(IntegrityError):
            controller.choose((1., float('nan'), 2.), state)

    def test_read_detaches_controller_and_detects_tampering(self):
        with tempfile.TemporaryDirectory() as path:
            store = Store(path)
            state = controller.start({'temperature': 0}, 's', (), 32)
            root = store.create_root('model', (1, 2), state)
            row = store.get_snapshot(root)
            row['controller']['policy']['allowed'].clear()
            self.assertEqual(store.get_snapshot(root)['controller'], state)
            source = store.objects / (root + '.json')
            source.write_bytes(source.read_bytes().replace(b'active', b'broken'))
            with self.assertRaises(IntegrityError):
                store.get_snapshot(root)


if __name__ == '__main__':
    unittest.main()
