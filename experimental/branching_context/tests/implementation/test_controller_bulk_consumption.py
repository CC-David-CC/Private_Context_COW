"""Bulk controller updates must equal ordered consumption and detach the parent."""
import copy
import unittest

from context_time_travel import controller


class BulkControllerTests(unittest.TestCase):
    def state(self, **policy):
        state = controller.start(policy, 'fixed-stream', [1,2,1], 8)
        state['next_draw'] = 19
        return state

    def assert_same(self, state, events):
        original = copy.deepcopy(state)
        expected = state
        for token,piece,eog in events:
            expected = controller.consume(expected, token, piece, eog)
        actual = controller.consume_many(state, iter(events))
        self.assertEqual(actual, expected)
        self.assertEqual(state, original)
        actual['policy']['allowed'].clear()
        self.assertEqual(state, original)

    def test_pending_utf8_and_nonzero_rng_cursor_match_ordered_consumption(self):
        self.assert_same(self.state(), [(3,b'\xf0\x9f',False),(4,b'\x90\x84',False),(5,b'\xe2',False)])

    def test_grammar_penalties_and_stop_sequences_match_ordered_consumption(self):
        self.assert_same(self.state(grammar={'start':'s','transitions':{'s':{'3':'t'},'t':{'4':'s'}}},
            repetition_penalty=1.2, stop_sequences=[[3,4]]),
            [(3,b'x',False),(4,b'y',False),(5,b'z',False)])

    def test_exception_during_events_cannot_mutate_parent(self):
        state = self.state()
        original = copy.deepcopy(state)
        def events():
            yield 3,b'x',False
            raise RuntimeError('interrupted')
        with self.assertRaisesRegex(RuntimeError, 'interrupted'):
            controller.consume_many(state, events())
        self.assertEqual(state, original)


if __name__ == '__main__':
    unittest.main()
