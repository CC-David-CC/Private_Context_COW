"""CPU feedback isolation and JSON restart checks, without model inference."""
import copy
import json
import unittest

import run_min_p_probe
from context_time_travel.sampled_controller import PlainSamplingSession, start
from context_time_travel.store import InvalidOperation
from test_sampling_proposal import ProposalTokenizer


POLICY = dict(temperature=.8, min_p=0., top_k=0, top_p=1.,
              repetition_penalty=1., mirostat=2, mirostat_tau=4., mirostat_eta=.05)


class MirostatResumeTests(unittest.TestCase):
    def session(self, state, history):
        tokenizer = ProposalTokenizer()
        return PlainSamplingSession.restore(state, history, 4,
            tokenizer.token_bytes, tokenizer.is_eog)

    def advance(self, session):
        token = run_min_p_probe.reference_choice(session, [1., 2., 0., -8.], b'\x07')
        session._consume_selected(token, ProposalTokenizer().token_bytes(token), False)
        return token

    def test_progressed_feedback_forks_detach_and_json_restart_reproduces_continuation(self):
        history = [0]
        state = start(POLICY, 'mirostat-resume', history, 4, [3])
        initial_mu = state['sampler_state']['mu']
        producer = self.session(state, history)
        for _ in range(3):
            history.append(self.advance(producer))
        parent = producer.current_controller
        self.assertEqual(parent['next_draw'], 3)
        self.assertNotEqual(parent['sampler_state']['mu'], initial_mu)
        frozen = copy.deepcopy(parent)
        left = self.session(parent, history)
        right = self.session(json.loads(json.dumps(parent, allow_nan=False)), history)
        repeat = self.session(parent, history)
        left.configure_generation(dict(POLICY, mirostat_tau=3.5, temperature=.65))
        self.assertEqual(left.current_controller['sampler_state'], parent['sampler_state'])
        self.advance(left)
        left_committed = left.current_controller
        self.assertEqual(right.current_controller, frozen)
        self.assertEqual(parent, frozen)
        right_tokens = [self.advance(right) for _ in range(4)]
        repeated_tokens = [self.advance(repeat) for _ in range(4)]
        self.assertEqual(right_tokens, repeated_tokens)
        self.assertEqual(right.current_controller, repeat.current_controller)
        self.assertEqual(right.current_controller['next_draw'], 7)
        self.assertEqual(left.current_controller, left_committed)
        self.assertEqual(parent, frozen)

    def test_abandoned_pending_choice_cannot_publish_or_advance_saved_feedback(self):
        state = start(POLICY, 'mirostat-abandon', (0,), 4, [3])
        frozen = copy.deepcopy(state)
        pending = self.session(state, (0,))
        token = run_min_p_probe.reference_choice(pending, [1., 2., 0., -8.], b'\x07')
        with self.assertRaises(InvalidOperation):
            _ = pending.current_controller
        self.assertEqual(pending._state, frozen)
        self.assertEqual(state, frozen)
        restarted = self.session(json.loads(json.dumps(state)), (0,))
        self.assertEqual(self.advance(restarted), token)
        self.assertEqual(restarted.current_controller['next_draw'], 1)
        self.assertEqual(state, frozen)


if __name__ == '__main__':
    unittest.main()
