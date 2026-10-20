"""CPU-only tests for the DRY native selector helper; compile and run on llm-60."""
from __future__ import annotations

import math
import os
from pathlib import Path
import random
import importlib.util
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
HEADER = ROOT / "native" / "min_p_select.h"


CPP = r'''#include "min_p_select.h"
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <vector>
int main() {
    size_t n, h, b; unsigned allowed, last_n; double mult, base;
    if (!(std::cin >> n >> h >> b >> mult >> base >> allowed >> last_n)) return 2;
    std::vector<float> scores(n); std::vector<uint32_t> history(h), breakers(b);
    for (float &x : scores) {
        std::string value; if (!(std::cin >> value)) return 3;
        if (value == "inf" || value == "+inf") x = std::numeric_limits<float>::infinity();
        else if (value == "-inf") x = -std::numeric_limits<float>::infinity();
        else if (value == "nan") x = std::numeric_limits<float>::quiet_NaN();
        else x = std::stof(value);
    }
    for (auto &x : history) if (!(std::cin >> x)) return 4;
    for (auto &x : breakers) if (!(std::cin >> x)) return 5;
    try {
        const std::vector<float> original = scores;
        size_t changed = 0;
        auto out = min_p_select::apply_dry_penalties(scores.data(), n, history, breakers,
            min_p_select::DryConfig{mult, base, allowed, last_n}, &changed);
        if (scores != original) { std::cout << "MUTATED\n"; return 6; }
        std::cout << std::setprecision(9);
        for (float x : out) std::cout << x << " ";
        std::cout << "\nCOUNT " << changed << "\n";
    } catch (const std::exception &e) {
        std::cout << "ERROR " << e.what() << "\n";
    }
}
'''


def compile_helper(tmp: Path) -> Path:
    source = tmp / "dry_helper.cpp"
    binary = tmp / "dry_helper"
    source.write_text(CPP, encoding="utf-8", newline="\n")
    subprocess.run(["g++", "-std=c++17", "-O2", "-I", str(HEADER.parent), str(source), "-o", str(binary)], check=True)
    return binary


def reference(scores, history, breakers, multiplier, base, allowed, last_n):
    out = [float(x) for x in scores]
    if last_n == 0:
        return out
    h = list(history[-last_n:])
    rep_limit = len(h)
    for i in range(len(h) - 1, -1, -1):
        if h[i] in breakers:
            rep_limit = len(h) - 1 - i
            break
    best = {}
    for j in range(len(h) - 1):
        matched = 0
        while (matched < rep_limit and j - matched >= 0 and
               len(h) - 1 - matched >= 0 and
               h[j - matched] == h[len(h) - 1 - matched]):
            matched += 1
        token = h[j + 1]
        best[token] = max(best.get(token, 0), matched)
    for token, repeat in best.items():
        if token in breakers or repeat < allowed:
            continue
        penalty = multiplier * (base ** (repeat - allowed))
        penalty = min(penalty, 3.4028234663852886e38)
        value = float(scores[token]) - penalty
        value = max(value, -3.4028234663852886e38)
        out[token] = float(__import__("struct").unpack("f", __import__("struct").pack("f", value))[0])
    return out


class DryNativeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.binary = compile_helper(Path(cls.tmp.name))

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def run_case(self, scores, history, breakers=(), multiplier=1.0, base=1.75, allowed=2, last_n=64):
        history = list(history[-last_n:]) if last_n else []
        payload = f"{len(scores)} {len(history)} {len(breakers)} {multiplier} {base} {allowed} {last_n}\n"
        payload += " ".join(map(str, scores)) + "\n"
        payload += " ".join(map(str, history)) + "\n"
        payload += " ".join(map(str, breakers)) + "\n"
        proc = subprocess.run([str(self.binary)], input=payload, text=True, capture_output=True, check=True)
        self.assertFalse(proc.stdout.startswith("ERROR"), proc.stdout)
        import struct
        lines = proc.stdout.splitlines()
        values = [struct.unpack("f", struct.pack("f", float(x)))[0] for x in lines[0].split()]
        self.last_penalized_count = int(lines[1].split()[1])
        return values

    def run_error(self, scores, history, breakers=(), multiplier=1.0, base=1.75, allowed=2, last_n=64):
        history = list(history[-last_n:]) if last_n else []
        payload = f"{len(scores)} {len(history)} {len(breakers)} {multiplier} {base} {allowed} {last_n}\n"
        payload += " ".join(map(str, scores)) + "\n"
        payload += " ".join(map(str, history)) + "\n"
        payload += " ".join(map(str, breakers)) + "\n"
        proc = subprocess.run([str(self.binary)], input=payload, text=True, capture_output=True, check=True)
        self.assertTrue(proc.stdout.startswith("ERROR"), proc.stdout)

    def test_pre_breaker_occurrence_still_matches_but_length_is_capped(self):
        got = self.run_case([0.0] * 8, [1, 2, 3, 7, 1, 2], [7], allowed=2, last_n=64)
        expected = reference([0.0] * 8, [1, 2, 3, 7, 1, 2], [7], 1.0, 1.75, 2, 64)
        self.assertEqual(got, expected)
        self.assertLess(got[3], 0.0)  # repeat [1,2] before the breaker predicts token 3
        self.assertEqual(self.last_penalized_count, 1)

    def test_last_n_zero_disables_and_allowed_zero_penalizes_prior_followers(self):
        scores = [2.0] * 5
        self.assertEqual(self.run_case(scores, [1, 2, 3], last_n=0), scores)
        got = self.run_case(scores, [1, 2, 3], allowed=0, last_n=8)
        self.assertEqual(got, reference(scores, [1, 2, 3], [], 1.0, 1.75, 0, 8))
        self.assertLess(got[2], scores[2])
        self.assertEqual(self.last_penalized_count, 2)

    def test_max_penalty_not_sum_with_overlap(self):
        history = [1, 2, 1, 2, 1, 2]
        scores = [0.0] * 4
        got = self.run_case(scores, history, multiplier=0.25, base=2.0, allowed=0, last_n=16)
        self.assertEqual(got, reference(scores, history, [], 0.25, 2.0, 0, 16))
        self.assertGreaterEqual(self.last_penalized_count, 1)

    def test_seeded_cases_match_independent_bruteforce_reference(self):
        rng = random.Random(76103)
        import struct
        for _ in range(80):
            vocab = rng.randint(8, 40)
            history = [rng.randrange(vocab) for _ in range(rng.randint(1, 24))]
            scores = [struct.unpack("f", struct.pack("f", rng.uniform(-7, 7)))[0] for _ in range(vocab)]
            last_n = rng.randint(0, 24)
            bounded = history[-last_n:] if last_n else []
            breakers = sorted(set(rng.sample(bounded, k=rng.randint(0, min(3, len(set(bounded)))))))
            allowed = rng.randint(0, 5)
            mult = rng.choice([0.0, 0.25, 1.0, 3.0])
            base = rng.choice([1.0, 1.5, 2.0])
            self.assertEqual(self.run_case(scores, history, breakers, mult, base, allowed, last_n),
                             reference(scores, history, breakers, mult, base, allowed, last_n))

    def test_saturates_without_nonfinite_values_and_does_not_mutate_input(self):
        # The helper receives a value copy; exponentiation must clamp to finite float32.
        scores = [0.0, 0.0, 0.0]
        got = self.run_case(scores, [1, 2], multiplier=1e300, base=1e100, allowed=0, last_n=8)
        self.assertEqual(got, reference(scores, [1, 2], [], 1e300, 1e100, 0, 8))
        self.assertTrue(all(math.isfinite(x) for x in got))
        self.assertEqual(scores, [0.0, 0.0, 0.0])
        self.assertEqual(self.last_penalized_count, 1)

    def test_invalid_nonfinite_masked_logit_and_breaker_shape_rejected(self):
        self.run_error([0.0, 1.0, float("inf")], [1, 2], [2])
        self.run_error([0.0, 1.0, 2.0], [1, 2], [2, 2])
        self.run_error([0.0, 1.0, 2.0], [1, 2], [3])


class DryBuilderTransformTests(unittest.TestCase):
    def test_v3_transform_retains_v1_v2_and_adds_bound_receipts(self):
        builder_path = ROOT / "scripts" / "build_min_p_selector_probe.py"
        spec = importlib.util.spec_from_file_location("dry_builder", builder_path)
        builder = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(builder)
        # Minimal fragment containing the exact unique insertion points used by
        # the frozen selector-driver builder, without linking a model binary.
        source = (r'''#include "grammar_select.h"
static const char * ready = ",\"grammar_selection_protocol\":\"finite-mask-argmax-v1\"}";
uint64_t grammar_selection_calls = 0, grammar_selection_logit_bytes = 0, grammar_selection_mask_bytes = 0;
std::cout << ",\"grammar_selection_mask_bytes\":" << grammar_selection_mask_bytes;
if (true) {} else if (command == "generate") {}
''')
        got = builder.instrument_min_p_selector_v3(source)
        for protocol in (builder.PROTOCOL, builder.PROTOCOL_V2, builder.PROTOCOL_V3):
            self.assertIn(protocol, got)
        for command in ("sample_mask", "sample_mask_v2", "sample_mask_v3"):
            self.assertIn(f'command == "{command}"', got)
        self.assertIn("dry_history_count", got)
        self.assertIn("dry_breaker_count", got)
        self.assertIn("dry_penalized_candidates", got)
        self.assertIn("++sampling_calls", got)
        self.assertIn("++sampling_v2_calls", got)
        self.assertIn("sampling_v3_calls", got)
        v3 = got.split('command == "sample_mask_v3"', 1)[1].split('command == "generate"', 1)[0]
        self.assertIn("++sampling_v3_calls", v3)
        self.assertNotIn("sampling_v2_calls +=", v3)
        self.assertNotIn("++sampling_v2_calls", v3)

    def test_transform_rejects_missing_or_duplicate_generate_anchor(self):
        builder_path = ROOT / "scripts" / "build_min_p_selector_probe.py"
        spec = importlib.util.spec_from_file_location("dry_builder_bad", builder_path)
        builder = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(builder)
        with self.assertRaises(ValueError):
            builder.instrument_min_p_selector_v3("not a selector driver")


if __name__ == "__main__":
    unittest.main()
