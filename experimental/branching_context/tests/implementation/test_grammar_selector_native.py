"""CPU-only tests for the generated native grammar-selection extension."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import random
import struct
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
BUILDER_PATH = ROOT / "scripts" / "build_grammar_selector_probe.py"
HEADER_PATH = ROOT / "native" / "grammar_select.h"
spec = importlib.util.spec_from_file_location("grammar_selector_builder", BUILDER_PATH)
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)


DRIVER = r'''#include <fstream>
    std::cout << "{\"status\":\"ready\",\"vocab\":" << forward.vocab()
              << ",\"capacity\":" << state.max_cells
              << ",\"native_request_protocol\":\"consumed-greedy-ack-v1\",\"selected_token_protocol\":\"consumed-selected-receipt-v1\"}" << std::endl;
    std::vector<float> logits((size_t)forward.vocab());
        } else if (command == "generate") {
        } else if (command == "stats") {
            std::cout << "{\"status\":\"ok\",\"consumed\":" << session.ids().size()
                      << ",\"forward_tokens\":" << (forward.completed_forward_tokens + prefill_tokens)
                      << ",\"prefill_tokens\":" << prefill_tokens
                      << ",\"prefill_chunks\":" << prefill_chunks << "}" << std::endl;
'''


def python_reference(scores, mask):
    if not scores or len(mask) != (len(scores) + 7) // 8:
        raise ValueError("width")
    if len(scores) % 8 and mask[-1] & ~((1 << (len(scores) % 8)) - 1):
        raise ValueError("padding")
    if any(not (float("-inf") < x < float("inf")) for x in scores):
        raise ValueError("nonfinite")
    allowed = [i for i in range(len(scores)) if mask[i // 8] & (1 << (i % 8))]
    if not allowed:
        raise ValueError("empty")
    return max(allowed, key=lambda i: (scores[i], -i))


class BuilderTests(unittest.TestCase):
    def test_generated_driver_adds_readiness_command_and_stats_without_forward(self):
        out = builder.instrument_grammar_selector(DRIVER)
        self.assertIn('grammar_selection_protocol', out)
        self.assertIn('finite-mask-argmax-v1', out)
        self.assertIn('command == "select_mask"', out)
        self.assertIn('grammar_selection_calls', out)
        self.assertIn('grammar_selection_logit_bytes', out)
        self.assertIn('grammar_selection_mask_bytes', out)
        self.assertIn('mask_file.gcount()', out)
        self.assertIn('mask_file.get(extra_byte)', out)
        self.assertIn('selection_protocol', out)
        self.assertIn('logits_count', out)
        self.assertIn('mask_bytes', out)
        start = out.index('command == "select_mask"')
        end = out.index('command == "generate"', start)
        select = out[start:end]
        self.assertNotIn('forward.run(', select)
        self.assertNotIn('session.begin(', select)
        self.assertNotIn('session.finish(', select)
        self.assertIn('session.logits()', select)
        self.assertLess(select.index('session.ids().size()'), select.index('grammar_select::select'))
        self.assertLess(select.index('++grammar_selection_calls'), select.index('selection_protocol'))

    def test_builder_rejects_missing_or_ambiguous_anchors(self):
        with self.assertRaisesRegex(ValueError, "ready"):
            builder.instrument_grammar_selector(DRIVER.replace('selected_token_protocol', 'removed'))
        with self.assertRaisesRegex(ValueError, "stats"):
            builder.instrument_grammar_selector(DRIVER.replace('prefill_chunks', 'removed_chunks'))

    def test_helper_matches_seeded_python_reference(self):
        rng = random.Random(78112026)
        for vocab in (1, 7, 8, 9, 31, 257):
            scores = [rng.uniform(-100, 100) for _ in range(vocab)]
            mask = bytearray((vocab + 7) // 8)
            for i in range(vocab):
                if rng.randrange(2):
                    mask[i // 8] |= 1 << (i % 8)
            if not any(mask):
                mask[0] = 1
            self.assertEqual(builder.select_reference(scores, bytes(mask)), python_reference(scores, bytes(mask)))

    def test_cpp_helper_edges_and_randomized_reference(self):
        source = r'''#include <iostream>
#include <limits>
#include <vector>
#include "grammar_select.h"
static int check(const std::vector<float>& s, const std::vector<unsigned char>& m,
                 size_t consumed, size_t expected) {
  try { return static_cast<int>(grammar_select::select(s.data(), s.size(), m.data(), m.size(), consumed, expected).token); }
  catch (const std::exception&) { return -1; }
}
int main() {
  std::vector<float> tie{-0.0f, +0.0f, 8.0f};
  std::vector<unsigned char> both{0x03};
  std::cout << check(tie, both, 3, 3) << '\n';
  std::vector<float> hidden_nan{1.0f, std::numeric_limits<float>::quiet_NaN(), 2.0f};
  std::vector<unsigned char> masked{0x05};
  std::cout << check(hidden_nan, masked, 3, 3) << '\n';
  std::vector<float> hidden_inf{1.0f, std::numeric_limits<float>::infinity(), 2.0f};
  std::cout << check(hidden_inf, masked, 3, 3) << '\n';
  std::vector<float> two{1.0f, 2.0f};
  std::vector<unsigned char> empty{0x00};
  std::cout << check(two, empty, 2, 2) << '\n';
  std::vector<float> nine(9, 1.0f);
  std::vector<unsigned char> padding{0x01, 0x80};
  std::cout << check(nine, padding, 9, 9) << '\n';
  std::vector<unsigned char> short_mask{0x01};
  std::cout << check(nine, short_mask, 9, 9) << '\n';
  std::cout << check(two, both, 2, 3) << '\n';
  size_t n, consumed, expected;
  while (std::cin >> n >> consumed >> expected) {
    std::vector<unsigned char> m(n / 8 + (n % 8 != 0));
    std::vector<float> s(n);
    for (auto& x : m) { unsigned int v; std::cin >> v; x = static_cast<unsigned char>(v); }
    for (auto& x : s) std::cin >> x;
    std::cout << check(s, m, consumed, expected) << '\n';
  }
  return 0;
}'''
        rng = random.Random(99202610)
        cases = []
        expected = []
        for vocab in (1, 2, 7, 8, 9, 15, 16, 17, 31, 64, 257):
            for _ in range(4):
                scores = [rng.uniform(-12.0, 12.0) for _ in range(vocab)]
                mask = bytearray((vocab + 7) // 8)
                for i in range(vocab):
                    if rng.getrandbits(1):
                        mask[i // 8] |= 1 << (i % 8)
                if not any(mask):
                    mask[0] = 1
                cases.append(f"{vocab} 5 5 " + " ".join(map(str, mask)) + " " + " ".join(map(repr, scores)))
                scores32 = [struct.unpack("<f", struct.pack("<f", x))[0] for x in scores]
                expected.append(str(python_reference(scores32, bytes(mask))))
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            cpp = td / "test.cpp"
            exe = td / "test"
            cpp.write_text(source, encoding="utf-8")
            subprocess.run(["g++", "-O3", "-DNDEBUG", "-std=c++20", "-Wall", "-Wextra", "-I", str(HEADER_PATH.parent), str(cpp), "-o", str(exe)], check=True)
            got = subprocess.run([str(exe)], input="\n".join(cases) + "\n", check=True, text=True, capture_output=True).stdout.splitlines()
        self.assertEqual(got[:7], ["0", "-1", "-1", "-1", "-1", "-1", "-1"])
        self.assertEqual(got[7:], expected)


if __name__ == "__main__":
    unittest.main()
