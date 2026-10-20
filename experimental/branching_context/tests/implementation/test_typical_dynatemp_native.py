"""Standalone v4 C++ selector checks against an independent Python reference."""
from __future__ import annotations

import math
import importlib.util
import random
import shutil
import struct
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DRIVER = r'''#include "min_p_select.h"
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>
int main() {
  size_t n, mask_bytes, top_k, history_count, breaker_count;
  unsigned long long draw53;
  double temperature, min_p, top_p, dry_multiplier, dry_base;
  double typical_p, dyna_range, dyna_exponent;
  unsigned dry_allowed, dry_last_n;
  while (std::cin >> n >> mask_bytes >> temperature >> min_p >> top_k >> top_p >> draw53
                  >> dry_multiplier >> dry_base >> dry_allowed >> dry_last_n
                  >> typical_p >> dyna_range >> dyna_exponent
                  >> history_count >> breaker_count) {
    std::vector<float> logits(n);
    for (auto &score : logits) { std::string raw; std::cin >> raw; score = std::stof(raw); }
    std::vector<uint8_t> mask(mask_bytes);
    for (auto &byte : mask) { unsigned value; std::cin >> value; byte = static_cast<uint8_t>(value); }
    std::vector<uint32_t> history(history_count), breakers(breaker_count);
    for (auto &token : history) std::cin >> token;
    for (auto &token : breakers) std::cin >> token;
    try {
      const auto result = min_p_select::select_v4(
          logits.data(), logits.size(), mask.data(), mask.size(), temperature, min_p,
          top_k, top_p, static_cast<uint64_t>(draw53), history, breakers,
          min_p_select::DryConfig{dry_multiplier, dry_base, dry_allowed, dry_last_n},
          typical_p, dyna_range, dyna_exponent);
      const auto legacy = min_p_select::select_with_dry(
          logits.data(), logits.size(), mask.data(), mask.size(), temperature, min_p,
          static_cast<uint64_t>(draw53), top_k, top_p, history, breakers,
          min_p_select::DryConfig{dry_multiplier, dry_base, dry_allowed, dry_last_n});
      std::cout << std::setprecision(17) << "OK " << result.token << " " << result.legal_count << " "
                << result.min_p_kept_count << " " << result.top_k_kept_count << " "
                << result.typical_kept_count << " "
                << result.kept_count << " " << result.effective_temperature << " "
                << result.total_weight << " " << result.dry_penalized_candidates << " "
                << legacy.token << "\n";
    } catch (const std::exception &error) { std::cout << "ERR " << error.what() << "\n"; }
  }
}
'''


def _f32(value):
    return struct.unpack("<f", struct.pack("<f", float(value)))[0]


def reference(logits, mask, *, temperature, min_p, top_k, top_p, draw53,
              dry_multiplier, dry_base, dry_allowed, dry_last_n,
              history, breakers, typical_p, dyna_range, dyna_exponent):
    scores = [_f32(value) for value in logits]
    vocab = len(scores)
    legal = [i for i in range(vocab) if mask[i >> 3] & (1 << (i & 7))]
    if not legal or any(not math.isfinite(value) for value in scores):
        raise ValueError("invalid scores or support")
    if not (temperature > 0 and 0 <= min_p <= 1 and 0 < top_p <= 1
            and 0 < typical_p <= 1 and dyna_range >= 0 and dyna_exponent > 0):
        raise ValueError("invalid selector parameters")

    dry_changed = 0
    if dry_multiplier > 0 and dry_last_n > 0:
        from context_time_travel.dry import FLOAT32_MAX, penalties_from_ids
        penalties = penalties_from_ids(history, breakers, dry_multiplier=dry_multiplier,
                                       dry_base=dry_base, dry_allowed_length=dry_allowed,
                                       dry_penalty_last_n=dry_last_n)
        for token, penalty in penalties.items():
            updated = _f32(max(-FLOAT32_MAX, float(scores[token]) - penalty))
            dry_changed += updated != scores[token]
            scores[token] = updated
    elif history or breakers:
        raise ValueError("inactive DRY requires empty history")

    maximum = max(scores[token] for token in legal)
    min_support = legal if min_p == 0 else [token for token in legal
                                           if scores[token] >= maximum + math.log(min_p)]
    if not min_support:
        raise ValueError("empty Min-P support")
    min_p_kept = len(min_support)
    if top_k:
        min_support = sorted(sorted(min_support, key=lambda token: (-scores[token], token))[:top_k])

    if typical_p < 1:
        maximum = max(scores[token] for token in min_support)
        weights = {token: math.exp(float(scores[token]) - maximum) for token in min_support}
        total = 0.0
        for token in sorted(min_support):
            total += weights[token]
        log_probs = {token: float(scores[token]) - maximum - math.log(total) for token in min_support}
        probs = {token: weights[token] / total for token in min_support}
        entropy = 0.0
        for token in sorted(min_support):
            if probs[token] > 0:
                entropy -= probs[token] * log_probs[token]
        ranked = sorted(min_support, key=lambda token: (abs(-log_probs[token] - entropy), token))
        typical_support = []
        cumulative = 0.0
        for token in ranked:
            typical_support.append(token)
            cumulative += probs[token]
            if cumulative > typical_p:
                break
    else:
        typical_support = min_support[:]
    typical_kept = len(typical_support)

    low = max(0.0, temperature - dyna_range)
    high = temperature + dyna_range
    if not math.isfinite(high):
        raise ValueError("DynaTemp range overflow")
    if dyna_range == 0:
        effective = temperature
    elif len(typical_support) == 1:
        effective = low
    else:
        maximum = max(scores[token] for token in typical_support)
        weights = {token: math.exp(float(scores[token]) - maximum) for token in typical_support}
        total = 0.0
        for token in sorted(typical_support):
            total += weights[token]
        entropy = 0.0
        for token in sorted(typical_support):
            prob = weights[token] / total
            if prob > 0:
                entropy -= prob * math.log(prob)
        norm = min(1.0, max(0.0, entropy / math.log(len(typical_support))))
        effective = low + (high - low) * math.pow(norm, dyna_exponent)

    if effective == 0:
        token = max(typical_support, key=lambda item: (scores[item], -item))
        return (token, len(legal), min_p_kept, len(min_support), typical_kept,
                1, effective, 1.0, dry_changed)
    maximum = max(scores[token] for token in typical_support)
    weights = {token: math.exp((float(scores[token]) - maximum) / effective)
               for token in typical_support}
    total = 0.0
    for token in sorted(typical_support):
        total += weights[token]
    if top_p < 1:
        ranked = sorted(typical_support, key=lambda token: (-scores[token], token))
        kept, cumulative = [], 0.0
        for token in ranked:
            kept.append(token)
            cumulative += weights[token]
            if cumulative >= top_p * total:
                break
        typical_support = kept
        weights = {token: weights[token] for token in kept}
        total = 0.0
        for token in sorted(kept):
            total += weights[token]
    final_count = len(typical_support)
    target = (draw53 / (1 << 53)) * total
    cumulative = 0.0
    token = None
    for candidate in sorted(typical_support):
        cumulative += weights[candidate]
        if cumulative > target:
            token = candidate
            break
    if token is None:
        token = max(typical_support, key=lambda item: (weights[item], -item))
    return (token, len(legal), min_p_kept, len(min_support), typical_kept,
            final_count, effective, total, dry_changed)


class CombinedNativeSamplerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        compiler = shutil.which("g++") or shutil.which("c++")
        if compiler is None:
            raise RuntimeError("llm-60 CPU test host must provide g++/c++")
        cls.tmp = tempfile.TemporaryDirectory(prefix="sampler-v4-native-")
        root = Path(cls.tmp.name)
        source, cls.exe = root / "driver.cpp", root / "driver"
        source.write_text(DRIVER, encoding="utf-8", newline="\n")
        proc = subprocess.run([compiler, "-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror",
                               "-DMIN_P_SELECT_ENABLE_V4=1",
                               "-I", str(ROOT / "native"), str(source), "-o", str(cls.exe)],
                              capture_output=True, text=True, check=False)
        if proc.returncode:
            raise AssertionError(f"native selector build failed: {proc.stderr}")

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def run_case(self, logits, mask, **kwargs):
        params = dict(temperature=0.8, min_p=0.0, top_k=0, top_p=1.0, draw53=0,
                      dry_multiplier=0.0, dry_base=1.75, dry_allowed=2, dry_last_n=256,
                      history=(), breakers=(), typical_p=1.0, dyna_range=0.0,
                      dyna_exponent=1.0)
        params.update(kwargs)
        mask_bytes = bytes(mask)
        values = [len(logits), len(mask_bytes), params["temperature"], params["min_p"],
                  params["top_k"], params["top_p"], params["draw53"],
                  params["dry_multiplier"], params["dry_base"], params["dry_allowed"],
                  params["dry_last_n"], params["typical_p"], params["dyna_range"],
                  params["dyna_exponent"], len(params["history"]), len(params["breakers"])]
        values.extend(repr(float(value)) for value in logits)
        values.extend(str(byte) for byte in mask_bytes)
        values.extend(str(token) for token in params["history"])
        values.extend(str(token) for token in params["breakers"])
        proc = subprocess.run([str(self.exe)], input=" ".join(map(str, values)) + "\n",
                              text=True, capture_output=True, check=False)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        fields = proc.stdout.strip().split()
        if fields[0] == "ERR":
            return fields
        self.assertEqual(fields[0], "OK", proc.stdout)
        return (int(fields[1]), int(fields[2]), int(fields[3]), int(fields[4]),
                int(fields[5]), int(fields[6]), float(fields[7]), float(fields[8]),
                int(fields[9]), int(fields[10]))

    def test_v4_combines_ordered_filters_and_dry(self):
        logits = [3.0, 2.8, 2.2, 2.0, 1.5, 0.0]
        mask = [0b00111111]
        args = dict(temperature=0.7, min_p=0.15, top_k=5, top_p=0.88, draw53=123456789,
                    dry_multiplier=0.9, dry_base=1.75, dry_allowed=2, dry_last_n=4,
                    history=(1, 2, 1, 2), breakers=(), typical_p=0.72,
                    dyna_range=0.35, dyna_exponent=1.2)
        actual = self.run_case(logits, mask, **args)
        expected = reference(logits, mask, **args)
        self.assertEqual(actual[:6], expected[:6])
        self.assertAlmostEqual(actual[6], expected[6], places=14)
        self.assertAlmostEqual(actual[7], expected[7], places=14)
        self.assertEqual(actual[8], expected[8])

    def test_v4_static_typical_and_dyna_defaults_agree_with_v3(self):
        # Existing v3 remains available. The v4 all-default tail is expected to
        # preserve its no-Typical/no-Dyna selection result exactly.
        logits = [0.1, -0.3, 1.25, 0.9, -2.0, 0.2, 0.8, -0.1]
        actual = self.run_case(logits, [0xFF], min_p=0.04, top_k=4, top_p=0.91,
                               draw53=(1 << 52) + 71, typical_p=1.0,
                               dyna_range=0.0)
        expected = reference(logits, [0xFF], min_p=0.04, top_k=4, top_p=0.91,
                             draw53=(1 << 52) + 71, typical_p=1.0,
                             dyna_range=0.0, dry_multiplier=0.0, dry_base=1.75,
                             dry_allowed=2, dry_last_n=256, history=(), breakers=(),
                             temperature=0.8, dyna_exponent=1.0)
        self.assertEqual(actual[0], expected[0])
        self.assertEqual(actual[0], actual[9])

    def test_ties_underflow_and_temperature_floor(self):
        logits = [4.0, 4.0, 3.0, -1000.0]
        mask = [0b00001111]
        actual = self.run_case(logits, mask, top_k=1, typical_p=0.5,
                               dyna_range=1.0, draw53=999)
        self.assertEqual(actual[0], 0)
        self.assertEqual(actual[2], 4)
        self.assertEqual(actual[3], 1)
        self.assertEqual(actual[4], 1)
        self.assertAlmostEqual(actual[6], 0.0)

    def test_invalid_parameters_and_inactive_dry_history_fail_closed(self):
        logits, mask = [1.0, 0.0], [0b11]
        for kwargs in ({"typical_p": 0.0}, {"dyna_range": -0.1},
                       {"dyna_exponent": 0.0}, {"history": (0,)},
                       {"dry_multiplier": 0.2, "dry_last_n": 0, "history": (0,)}):
            with self.subTest(kwargs=kwargs):
                result = self.run_case(logits, mask, **kwargs)
                self.assertEqual(result[0], "ERR")

    def test_seeded_combined_vectors_match_independent_reference(self):
        rng = random.Random(20261016)
        for case in range(60):
            vocab = rng.randint(4, 32)
            logits = [rng.uniform(-12.0, 12.0) for _ in range(vocab)]
            legal_ids = [token for token in range(vocab) if rng.random() < 0.72]
            if not legal_ids:
                legal_ids = [rng.randrange(vocab)]
            mask = bytearray((vocab + 7) // 8)
            for token in legal_ids:
                mask[token >> 3] |= 1 << (token & 7)
            dry_active = case % 3 == 0
            if dry_active:
                history = tuple(rng.randrange(vocab) for _ in range(rng.randint(2, 10)))
                breaker_count = rng.randrange(0, min(3, len(set(history)) + 1))
                breaker_values = tuple(sorted(set(rng.sample(history, k=breaker_count))))
                dry_multiplier, dry_last_n = 0.6, len(history)
            else:
                history, breaker_values = (), ()
                dry_multiplier, dry_last_n = 0.0, 256
            params = dict(
                temperature=rng.choice((0.25, 0.8, 1.4)),
                min_p=rng.choice((0.0, 0.05, 0.3)),
                top_k=rng.choice((0, 1, 3, vocab + 2)),
                top_p=rng.choice((0.35, 0.8, 1.0)),
                draw53=rng.randrange(1 << 53),
                dry_multiplier=dry_multiplier, dry_base=1.75, dry_allowed=2,
                dry_last_n=dry_last_n, history=history, breakers=breaker_values,
                typical_p=rng.choice((0.15, 0.55, 0.9, 1.0)),
                dyna_range=rng.choice((0.0, 0.2, 0.8)),
                dyna_exponent=rng.choice((0.5, 1.0, 2.0)),
            )
            with self.subTest(case=case, params=params):
                actual = self.run_case(logits, mask, **params)
                expected = reference(logits, mask, **params)
                self.assertEqual(actual[:6], expected[:6])
                self.assertAlmostEqual(actual[6], expected[6], places=13)
                self.assertAlmostEqual(actual[7], expected[7], places=13)
                self.assertEqual(actual[8], expected[8])

    def test_builder_v4_preserves_legacy_commands_and_requires_unique_anchors(self):
        builder_path = ROOT / "scripts" / "build_min_p_selector_probe.py"
        spec = importlib.util.spec_from_file_location("v4_builder_test", builder_path)
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        builder = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(builder)
        source = "\n".join((
            '#include "grammar_select.h"',
            builder.READY_ANCHOR,
            builder.STATS_ANCHOR,
            builder.COUNTERS_ANCHOR,
            '} else if (command == "generate") {',
        ))
        generated = builder.instrument_min_p_selector_v4(source)
        for protocol in (builder.PROTOCOL, builder.PROTOCOL_V2,
                         builder.PROTOCOL_V3, builder.PROTOCOL_V4):
            self.assertIn(protocol, generated)
        for command in ("sample_mask", "sample_mask_v2", "sample_mask_v3", "sample_mask_v4"):
            self.assertIn(f'command == "{command}"', generated)
        for field in ("sampling_v4_calls", "typical_filter_calls",
                      "dynatemp_filter_calls", "dry_penalized_candidates_v4"):
            self.assertIn(field, generated)
        with self.assertRaises(ValueError):
            builder.instrument_min_p_selector_v4(source.replace(builder.READY_ANCHOR, ""))


if __name__ == "__main__":
    unittest.main()
