"""CPU contracts for explicit COW generation seeds and in-scope reseeding."""
import copy
import unittest

from context_time_travel.sampled_controller import start
from context_time_travel import controller
from context_time_travel.seed_controls import reseed_controller, seed_stream
from context_time_travel.store import IntegrityError, InvalidOperation
from tools import cow_cli
from context_time_travel import adaptive_p, sampled_controller
from context_time_travel.sampling_proposal import SelectionProposal
from context_time_travel.presentation_request import detach_request


POLICY = {
    "temperature": 0.7,
    "min_p": 0.05,
    "top_k": 0,
    "top_p": 1.0,
    "repetition_penalty": 1.0,
}


class SeedControlTests(unittest.TestCase):
    def test_seeded_presentation_uses_schema_two_and_legacy_stays_schema_one(self):
        request = {
            "schema": 1, "namespace_id": "a" * 32, "kind": "answer",
            "messages": [], "renderer_id": "b" * 64, "output_message_id": "c" * 32,
        }
        self.assertEqual(detach_request(request)["schema"], 1)
        seeded = dict(request, schema=2, seed=0)
        self.assertEqual(detach_request(seeded)["schema"], 2)
        with self.assertRaisesRegex(InvalidOperation, "Invalid presentation request fields"):
            detach_request(dict(seeded, schema=1))
        with self.assertRaisesRegex(InvalidOperation, "Invalid presentation request fields"):
            detach_request(dict(request, schema=2))

    def test_seed_stream_is_stable_and_zero_is_valid(self):
        self.assertEqual(seed_stream(0), "cow-seed-v1:0")
        self.assertEqual(seed_stream(42), seed_stream(42))
        self.assertNotEqual(seed_stream(42), seed_stream(43))

    def test_seed_rejects_bool_negative_float_and_overflow(self):
        for bad in (True, -1, 1.0, "7", 1 << 64):
            with self.subTest(seed=bad), self.assertRaises(InvalidOperation):
                seed_stream(bad)

    def test_reseed_resets_only_rng_key_and_cursor(self):
        state = start(POLICY, seed_stream(7), (1, 1, 2), 8, [7])
        state["next_draw"] = 9
        before = copy.deepcopy(state)
        reseeded = reseed_controller(state, seed=0, history=(1, 1, 2), vocab_size=8)
        self.assertEqual(reseeded["next_draw"], 0)
        self.assertEqual(reseeded["stream_key"], start(
            POLICY, seed_stream(0), (1, 1, 2), 8, [7])["stream_key"])
        for field in set(before) - {"stream_key", "next_draw"}:
            self.assertEqual(reseeded[field], before[field], field)
        self.assertEqual(state, before, "reseed must not mutate the caller's canonical value")

    def test_reseed_preserves_feedback_and_parser_payloads(self):
        adaptive = dict(POLICY, adaptive_target=0.2, adaptive_decay=0.8)
        tokenizer = type("Tokenizer", (), {
            "token_bytes": staticmethod(lambda token: bytes((token,))),
            "is_eog": staticmethod(lambda token: token == 7),
        })()
        state = start(adaptive, seed_stream(1), (1, 2), 8, [7])
        session = sampled_controller.PlainSamplingSession.restore(
            state, (1, 2), 8, tokenizer.token_bytes, tokenizer.is_eog)

        def selector(mask, temperature, min_p, *, sampler_state, adaptive_target,
                     adaptive_decay, draw_start, draw_candidates, source_digest, **filters):
            proposal = adaptive_p.choose([0., 1., -1., 0., 0., 0., 0., 0.], mask,
                sampler_state, adaptive_target=adaptive_target, adaptive_decay=adaptive_decay,
                draw53=draw_candidates[0])
            next_state = adaptive_p.accept(sampler_state, proposal["proposal"], proposal["token"])
            return SelectionProposal(proposal["token"], draw_start, 1, next_state, source_digest)

        mask = bytes((0xff,))
        token = session.choose(mask, selector)
        session._consume_selected(token, tokenizer.token_bytes(token), False)
        state = session.current_controller
        consumed_history = (1, 2, token)
        self.assertEqual(state["next_draw"], 1)
        expected = copy.deepcopy(state)
        reseeded = reseed_controller(state, seed=2, history=consumed_history, vocab_size=8)
        for field in set(expected) - {"stream_key", "next_draw"}:
            self.assertEqual(reseeded[field], expected[field], field)

        grammar_state = controller.start({
            "temperature": 0.0, "eos_tokens": [7],
            "grammar": {"start": "s", "transitions": {"s": {"3": "t"}, "t": {}}},
        }, seed_stream(3), (), 8)
        advanced = controller.consume(grammar_state, 3, b"A", False)
        self.assertEqual(advanced["grammar_state"], "t")
        reseeded_grammar = reseed_controller(advanced, seed=4, history=(3,), vocab_size=8)
        self.assertEqual(reseeded_grammar["grammar_state"], "t")
        self.assertEqual(reseeded_grammar["policy"]["grammar"], advanced["policy"]["grammar"])

        from context_time_travel.grammar_first import GrammarEngine
        from test_grammar_first import TinyByteTokenizer
        tokenizer=TinyByteTokenizer()
        engine=GrammarEngine(tokenizer,{
            'answer':{'format':'gbnf','source':'root ::= "AA" | "AB"'},
        },eos_tokens=[TinyByteTokenizer.EOS])
        grammar_state=engine.start('answer',seed_stream(5),[],len(tokenizer.tokens))
        grammar_session=engine.restore(grammar_state,[])
        logits=[0.0]*len(tokenizer.tokens)
        logits[ord('A')]=3.0
        selected=grammar_session.choose(logits)
        grammar_state=grammar_session.consume(grammar_state,selected,
            tokenizer.token_bytes(selected),False)
        self.assertEqual(grammar_state['constraint']['scope_start'],0)
        grammar_reseeded=reseed_controller(grammar_state,seed=6,
            history=(ord('A'),),vocab_size=len(tokenizer.tokens))
        self.assertEqual(grammar_reseeded['constraint'],grammar_state['constraint'])
        resumed_grammar=engine.restore(grammar_reseeded,[ord('A')])
        next_logits=[0.0]*len(tokenizer.tokens)
        next_logits[ord('B')]=4.0
        next_token=resumed_grammar.choose(next_logits)
        final_grammar=resumed_grammar.consume(grammar_reseeded,next_token,
            tokenizer.token_bytes(next_token),False)
        self.assertTrue(resumed_grammar.is_complete())
        self.assertEqual(final_grammar['accepted_in_scope'],2)

    def test_reseed_preserves_stopped_status_and_checks_canonical_history(self):
        stopped = controller.consume(
            controller.start({"temperature": 0.0, "eos_tokens": [7]}, seed_stream(1), (1,), 8),
            7, b"", True)
        self.assertEqual(stopped["status"], "stopped")
        reseeded = reseed_controller(stopped, seed=2, history=(1, 7), vocab_size=8)
        self.assertEqual(reseeded["status"], "stopped")
        state = start(POLICY, seed_stream(1), (1,), 8, [7])
        with self.assertRaisesRegex(IntegrityError, "repetition state differs"):
            reseed_controller(state, seed=2, history=(1, 2), vocab_size=8)

    def test_cli_accepts_seed_zero_for_new_generation_and_reseed(self):
        answer = cow_cli.parser().parse_args(["answer", "--branch", "main", "--seed", "0"])
        reseed = cow_cli.parser().parse_args(["reseed", "--branch", "main", "--seed", "0"])
        self.assertEqual(answer.seed, 0)
        self.assertEqual(reseed.seed, 0)

    def test_cli_rejects_invalid_seed_syntax(self):
        for value in ("-1", "1.0", "18446744073709551616"):
            with self.subTest(seed=value), self.assertRaises(SystemExit):
                cow_cli.parser().parse_args(["answer", "--branch", "main", "--seed", value])
        with self.assertRaises(SystemExit):
            cow_cli.parser().parse_args(["continue", "--branch", "main", "--seed", "3"])


if __name__ == "__main__":
    unittest.main()
