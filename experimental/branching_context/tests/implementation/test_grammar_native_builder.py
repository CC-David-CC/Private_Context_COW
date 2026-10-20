"""Fail-closed tests for the isolated grammar probe driver builder."""
import importlib.util
from pathlib import Path
import unittest


BUILDER_PATH = Path(__file__).resolve().parents[2] / "scripts" / "build_grammar_probe.py"
spec = importlib.util.spec_from_file_location("grammar_native_builder", BUILDER_PATH)
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)


DRIVER = r'''    std::cout << "{\"status\":\"ready\",\"vocab\":" << forward.vocab()
              << ",\"capacity\":" << state.max_cells
              << ",\"native_request_protocol\":\"consumed-greedy-ack-v1\"}" << std::endl;
        } else if (command == "decode") {
            int32_t token; std::string extra;
            const int64_t position = session.ids().size();
            const bool prepared = session.begin(position, 1, error);
            const bool finished = session.finish({token}, logits, error);
            if (!finished) { error_reply(error); continue; }
            std::cout << "{\"status\":\"ok\",\"consumed\":" << session.ids().size() << "}" << std::endl;
        } else if (command == "capture") {
            auto image = session.freeze(error);
'''


class GrammarNativeBuilderTests(unittest.TestCase):
    def test_adds_separate_ready_capability_and_actual_decode_boundary(self):
        transformed = builder.instrument_selected_token_receipt(DRIVER)
        self.assertIn('selected_token_protocol', transformed)
        self.assertIn('consumed-selected-receipt-v1', transformed)
        self.assertIn('native_request_protocol', transformed)
        self.assertIn('consumed-greedy-ack-v1', transformed)
        self.assertIn('normal-verifier-decode', transformed)
        self.assertIn('"count\\":1', transformed)
        self.assertIn('"chunks\\":[[', transformed)
        self.assertIn('"consumed\\":', transformed)
        self.assertLess(transformed.index('session.finish'), transformed.index('normal-verifier-decode'))
        self.assertEqual(transformed.count('selected_token_protocol'), 1)

    def test_requires_exactly_one_ready_anchor(self):
        broken = DRIVER.replace(builder.READY_OLD, "", 1)
        with self.assertRaisesRegex(ValueError, "ready"):
            builder.instrument_selected_token_receipt(broken)
        with self.assertRaisesRegex(ValueError, "ready"):
            builder.instrument_selected_token_receipt(DRIVER + DRIVER)

    def test_requires_exactly_one_decode_success_anchor(self):
        old = 'std::cout << "{\\"status\\":\\"ok\\",\\"consumed\\":" << session.ids().size() << "}" << std::endl;'
        broken = DRIVER.replace(old, "")
        with self.assertRaisesRegex(ValueError, "decode"):
            builder.instrument_selected_token_receipt(broken)
        with self.assertRaisesRegex(ValueError, "decode"):
            builder.instrument_selected_token_receipt(DRIVER.replace(old, old + "\n            " + old))


if __name__ == "__main__":
    unittest.main()
