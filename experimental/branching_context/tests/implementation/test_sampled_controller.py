"""CPU-only schema-3 Min-P controller/session boundary tests."""
import copy
import threading
import unittest

from context_time_travel import controller as legacy_controller
from context_time_travel.min_p import draw53
from context_time_travel.sampled_controller import (
    MIN_P_SAMPLER_ORDER,
    PlainSamplingSession,
    choose_owned,
    start,
    validate_controller,
)
from context_time_travel.store import IntegrityError, InvalidOperation


def policy(**changes):
    value = {
        "temperature": 0.70,
        "min_p": 0.05,
        "top_k": 0,
        "top_p": 1.0,
        "repetition_penalty": 1.0,
    }
    value.update(changes)
    return value


def make_mask(vocab_size, allowed):
    value = bytearray((vocab_size + 7) // 8)
    for token in allowed:
        value[token >> 3] |= 1 << (token & 7)
    return bytes(value)


class ByteTokenizer:
    def __init__(self, pieces):
        self.pieces = tuple(pieces)

    def token_bytes(self, token):
        return self.pieces[token]

    def is_eog(self, token):
        return token == len(self.pieces) - 1


class SampledControllerTests(unittest.TestCase):
    def setUp(self):
        self.tokenizer = ByteTokenizer((b"a", b"b", b"\xe2", b"\x82", b"\xac", b""))
        self.vocab_size = len(self.tokenizer.pieces)
        self.eos = (self.vocab_size - 1,)

    def started(self, history=(0,), **changes):
        return start(policy(**changes), "branch-A", history, self.vocab_size, self.eos)

    def restore(self, state, history=(0,)):
        return PlainSamplingSession.restore(
            state, history, self.vocab_size,
            self.tokenizer.token_bytes, self.tokenizer.is_eog,
        )

    def test_start_uses_complete_schema3_without_changing_legacy_start(self):
        state = self.started()
        self.assertEqual(state["schema"], 3)
        self.assertIsNone(state["constraint"])
        self.assertEqual(state["sampler_order"], MIN_P_SAMPLER_ORDER)
        self.assertEqual(state["policy"]["allowed"], list(range(self.vocab_size)))
        self.assertEqual(state["policy"]["min_p"], 0.05)
        self.assertEqual(state["policy"]["top_k"], 0)
        self.assertEqual(state["policy"]["top_p"], 1.0)
        self.assertEqual(state["policy"]["repetition_penalty"], 1.0)
        old = legacy_controller.start(
            {"temperature": 0, "eos_tokens": list(self.eos)},
            "branch-A", (0,), self.vocab_size,
        )
        self.assertEqual(old["schema"], 1)
        self.assertEqual(old["sampler_order"], [
            "grammar/support", "repetition-penalty", "temperature", "categorical",
        ])

    def test_validate_detaches_and_accepts_nonzero_draw_cursor(self):
        state = self.started()
        state["next_draw"] = 42
        detached = validate_controller(state, history=(0,), vocab_size=self.vocab_size)
        self.assertEqual(detached["next_draw"], 42)
        detached["policy"]["allowed"].clear()
        self.assertEqual(state["policy"]["allowed"], list(range(self.vocab_size)))

    def test_validate_rejects_bad_sampler_policy_and_controller_fields(self):
        state = self.started()
        variants = []
        bad = copy.deepcopy(state); bad["sampler_order"] = ["temperature"]; variants.append(bad)
        bad = copy.deepcopy(state); bad["constraint"] = {}; variants.append(bad)
        bad = copy.deepcopy(state); bad["policy"]["temperature"] = True; variants.append(bad)
        bad = copy.deepcopy(state); bad["policy"]["top_k"] = False; variants.append(bad)
        bad = copy.deepcopy(state); bad["policy"]["min_p"] = float("nan"); variants.append(bad)
        bad = copy.deepcopy(state); bad["policy"]["allowed"][-1] = 1; variants.append(bad)
        bad = copy.deepcopy(state); bad["stream_key"] = "x" * 64; variants.append(bad)
        bad = copy.deepcopy(state); bad["next_draw"] = True; variants.append(bad)
        bad = copy.deepcopy(state); bad["seen"] = {"0": 0}; variants.append(bad)
        bad = copy.deepcopy(state); bad["output_pending_hex"] = "f"; variants.append(bad)
        for candidate in variants:
            with self.subTest(candidate=candidate), self.assertRaises((IntegrityError, InvalidOperation, ValueError)):
                validate_controller(candidate, history=(0,), vocab_size=self.vocab_size)

    def test_restore_checks_full_history_counts_and_reconstructs_utf8_pending(self):
        state = self.started()
        session = self.restore(state)
        before = session.current_controller
        token = session.choose(make_mask(self.vocab_size, (2,)), lambda *args: 2)
        after = session.consume(before, token, b"\xe2", False)
        self.assertEqual(after["accepted_in_scope"], 1)
        self.assertEqual(after["output_pending_hex"], "e2")
        resumed = self.restore(after, history=(0, 2))
        self.assertEqual(resumed.current_controller, after)

        before = resumed.current_controller
        token = resumed.choose(make_mask(self.vocab_size, (3,)), lambda *args: 3)
        after = resumed.consume(before, token, b"\x82", False)
        before = resumed.current_controller
        token = resumed.choose(make_mask(self.vocab_size, (4,)), lambda *args: 4)
        after = resumed.consume(before, token, b"\xac", False)
        self.assertEqual(after["output_pending_hex"], "")
        self.assertEqual(after["accepted_in_scope"], 3)

        bad = copy.deepcopy(after); bad["seen"]["0"] += 1
        with self.assertRaises(IntegrityError):
            self.restore(bad, history=(0, 2, 3, 4))
        bad = copy.deepcopy(after); bad["output_pending_hex"] = "ff"
        with self.assertRaises(IntegrityError):
            self.restore(bad, history=(0, 2, 3, 4))

    def test_selector_failure_or_illegal_token_does_not_advance_draw(self):
        for selector in (
            lambda *args: (_ for _ in ()).throw(RuntimeError("selector failed")),
            lambda *args: 1,
            lambda *args: True,
        ):
            state = self.started()
            session = self.restore(state)
            before = session.current_controller
            with self.assertRaises((RuntimeError, IntegrityError)):
                session.choose(make_mask(self.vocab_size, (2,)), selector)
            self.assertEqual(session.current_controller, before)
            self.assertEqual(before["next_draw"], 0)

    def test_choose_owned_increments_only_after_selector_success_and_checks_mask(self):
        state = self.started()
        mask = make_mask(self.vocab_size, (1, 4))
        observed = []

        def selector(actual_mask, temperature, min_p, draw):
            observed.append((actual_mask, temperature, min_p, draw))
            return 4

        self.assertEqual(choose_owned(state, mask, selector), 4)
        self.assertEqual(state["next_draw"], 1)
        self.assertEqual(observed[0], (
            mask, 0.70, 0.05,
            draw53(state["stream_key"], 0),
        ))
        for bad_mask in (b"", make_mask(self.vocab_size, ())):
            untouched = self.started()
            with self.subTest(mask=bad_mask), self.assertRaises(IntegrityError):
                choose_owned(untouched, bad_mask, selector)
            self.assertEqual(untouched["next_draw"], 0)
        padded = self.started()
        with self.assertRaises(IntegrityError):
            choose_owned(padded, bytes((0x80,)), selector)
        self.assertEqual(padded["next_draw"], 0)

    def test_pending_sample_is_not_an_exportable_controller(self):
        session = self.restore(self.started())
        before = session.current_controller
        token = session.choose(make_mask(self.vocab_size, (1,)), lambda *args: 1)
        with self.assertRaises(InvalidOperation):
            _ = session.current_controller
        after = session.consume(before, token, b"b", False)
        self.assertEqual(after["next_draw"], 1)
        self.assertEqual(after["accepted_in_scope"], 1)

    def test_public_consume_rejects_stale_controller_and_poisons_failed_transition(self):
        session = self.restore(self.started())
        initial = session.current_controller
        token = session.choose(make_mask(self.vocab_size, (2,)), lambda *args: 2)
        current = session.consume(initial, token, b"\xe2", False)
        token = session.choose(make_mask(self.vocab_size, (3,)), lambda *args: 3)
        with self.assertRaises(InvalidOperation):
            session.consume(initial, token, b"\x82", False)
        session.consume(current, token, b"\x82", False)

        session = self.restore(self.started())
        current = session.current_controller
        token = session.choose(make_mask(self.vocab_size, (2,)), lambda *args: 2)
        with self.assertRaises(IntegrityError):
            session.consume(current, token, b"wrong", False)
        with self.assertRaises(IntegrityError):
            _ = session.current_controller

    def test_owner_thread_guard_and_stop_state(self):
        session = self.restore(self.started())
        failures = []

        def foreign_call():
            try:
                session.is_stopped()
            except Exception as exc:
                failures.append(exc)

        thread = threading.Thread(target=foreign_call)
        thread.start(); thread.join()
        self.assertEqual(len(failures), 1)
        self.assertIsInstance(failures[0], InvalidOperation)
        self.assertFalse(session.is_complete())

        current = session.current_controller
        token = session.choose(make_mask(self.vocab_size, self.eos), lambda *args: self.eos[0])
        stopped = session.consume(current, token, b"", True)
        self.assertEqual(stopped["status"], "stopped")
        self.assertTrue(session.is_stopped())
        self.assertFalse(session.is_complete())


if __name__ == "__main__":
    unittest.main()
