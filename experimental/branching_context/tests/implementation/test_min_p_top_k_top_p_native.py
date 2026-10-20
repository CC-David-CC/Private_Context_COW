"""CPU-only native tests for Min-P followed by Top-K and Top-P sampling."""
from __future__ import annotations

import math
import importlib.util
from pathlib import Path
import random
import shutil
import struct
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
HEADER = ROOT / "native" / "min_p_select.h"
BUILDER_PATH = ROOT / "scripts" / "build_min_p_selector_probe.py"
_builder_spec = importlib.util.spec_from_file_location("min_p_selector_builder_v2", BUILDER_PATH)
builder = importlib.util.module_from_spec(_builder_spec)
_builder_spec.loader.exec_module(builder)

DRIVER_STUB = r'''#include "grammar_select.h"
#include <fstream>
    std::vector<float> logits((size_t)forward.vocab());
    uint64_t grammar_selection_calls = 0, grammar_selection_logit_bytes = 0, grammar_selection_mask_bytes = 0;
    std::cout << "{\"status\":\"ready\",\"grammar_selection_protocol\":\"finite-mask-argmax-v1\"}" << std::endl;
                      << ",\"grammar_selection_mask_bytes\":" << grammar_selection_mask_bytes
        } else if (command == "generate") {
'''


CPP = r'''#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>
#include "min_p_select.h"

using min_p_select::Result;
using min_p_select::select;

template<class F> bool rejects(F&& f) {
    try { f(); } catch (const std::exception&) { return true; }
    return false;
}

int main() {
    // Top-K chooses the highest raw scores, breaking ties by lower token ID.
    const float scores[] = {5.0f, 5.0f, 4.0f, 3.0f, 2.0f};
    const uint8_t all[] = {0x1f};
    const auto k2 = select(scores, 5, all, 1, 1.0, 0.0, 0, 2, 1.0);
    const auto k2_last_draw = select(scores, 5, all, 1, 1.0, 0.0,
                                    (uint64_t(1) << 53) - 1, 2, 1.0);
    std::cout << "topk " << k2.token << ' ' << k2.legal_count << ' '
              << k2.kept_count << ' ' << k2_last_draw.token << '\n';
    const auto k_clamped = select(scores, 5, all, 1, 1.0, 0.0, 0, 99, 1.0);
    std::cout << "clamped " << k_clamped.token << ' ' << k_clamped.legal_count
              << ' ' << k_clamped.kept_count << '\n';

    // Top-P ranks by descending probability, but categorical accumulation
    // remains token-ID order over the selected set.
    const float p_scores[] = {0.0f, -0.1f, -0.2f};
    const uint8_t p_all[] = {0x07};
    const auto p60_low_draw = select(p_scores, 3, p_all, 1, 1.0, 0.0,
                                     0, 0, 0.60);
    const auto p60_high_draw = select(p_scores, 3, p_all, 1, 1.0, 0.0,
                                      (uint64_t(1) << 53) - 1, 0, 0.60);
    std::cout << "topp " << p60_low_draw.token << ' ' << p60_low_draw.kept_count
              << ' ' << p60_high_draw.token << '\n';

    // Top-P ranks the kept set by weight, then the categorical CDF scans IDs.
    const float reverse_rank[] = {-0.1f, -2.0f, 0.0f};
    const auto token_order = select(reverse_rank, 3, p_all, 1, 1.0, 0.0,
                                    uint64_t(1) << 51, 0, 0.90);
    std::cout << "token_order " << token_order.token << ' '
              << token_order.kept_count << '\n';

    // Top-P denominator sums in token-ID order, independently of rank order.
    float sum_order_scores[102] = {0.0f};
    for (size_t token = 1; token < 101; ++token) sum_order_scores[token] = -36.0f;
    const uint8_t sum_order_mask[13] = {0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
        0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0x3f};
    double id_order_total = 0.0;
    for (size_t token = 0; token < 102; ++token)
        id_order_total += std::exp(static_cast<double>(sum_order_scores[token]));
    const auto sum_order = select(sum_order_scores, 102, sum_order_mask, 13,
        1.0, 0.0, 0, 0, 1.0 / id_order_total);
    std::cout << "top_p_sum_order " << sum_order.kept_count << '\n';

    // Temperature changes Top-P mass and is applied after raw-logit Top-K.
    const float temp_p_scores[] = {0.0f, -1.0f, -2.0f};
    const auto cold_p = select(temp_p_scores, 3, p_all, 1, 0.5, 0.0,
                               0, 0, 0.75);
    const auto warm_p = select(temp_p_scores, 3, p_all, 1, 2.0, 0.0,
                               0, 0, 0.75);
    std::cout << "temperature_top_p " << cold_p.kept_count << ' '
              << warm_p.kept_count << '\n';

    // Min-P precedes Top-K; Top-P is applied after temperature scaling.
    const float combined_scores[] = {0.0f, -0.2f, -0.8f, -4.0f};
    const uint8_t combined_all[] = {0x0f};
    const auto combined = select(combined_scores, 4, combined_all, 1,
                                 0.5, 0.1, uint64_t(1) << 52, 2, 0.75);
    std::cout << "combined " << combined.token << ' ' << combined.legal_count
              << ' ' << combined.kept_count << '\n';
    const float minp_scores[] = {0.0f, -1.0f, -3.0f};
    const auto minp_before = select(minp_scores, 3, p_all, 1, 1.0, 0.1,
                                    0, 0, 0.9999);
    std::cout << "minp " << minp_before.legal_count << ' '
              << minp_before.kept_count << '\n';
    const float zero_temperature_scores[] = {100.0f, 2.0f, 2.0f, 1.0f};
    const uint8_t zero_temperature_mask[] = {0x0e};
    const auto zero_temperature = select(zero_temperature_scores, 4,
        zero_temperature_mask, 1, 0.0, 0.5, 123, 1, 0.1);
    std::cout << "temperature_zero " << zero_temperature.token << ' '
              << zero_temperature.legal_count << ' ' << zero_temperature.kept_count
              << ' ' << zero_temperature.total_weight << '\n';

    // Defaults must remain bit-for-bit routed through the v1 selector.
    const auto legacy = select(scores, 5, all, 1, 0.7, 0.2, 1234567);
    const auto defaults = select(scores, 5, all, 1, 0.7, 0.2, 1234567, 0, 1.0);
    uint64_t legacy_weight_bits = 0, defaults_weight_bits = 0;
    std::memcpy(&legacy_weight_bits, &legacy.total_weight, sizeof(legacy_weight_bits));
    std::memcpy(&defaults_weight_bits, &defaults.total_weight, sizeof(defaults_weight_bits));
    std::cout << "legacy " << legacy.token << ' ' << legacy.legal_count << ' '
              << legacy.kept_count << ' ' << defaults.token << ' '
              << defaults.legal_count << ' ' << defaults.kept_count << ' '
              << legacy_weight_bits << ' ' << defaults_weight_bits << '\n';

    const uint8_t empty[] = {0};
    const uint8_t padding[] = {0x8f};
    const float nonfinite[] = {1.0f, std::numeric_limits<float>::infinity()};
    const uint8_t nonfinite_mask[] = {0x01};
    const float one[] = {0.0f};
    const uint8_t one_mask[] = {0x01};
    int invalid = 0;
    invalid += rejects([&]{ select(one, 1, one_mask, 1, 1.0, 0.0, 0, 0, 0.0); });
    invalid += rejects([&]{ select(one, 1, one_mask, 1, 1.0, 0.0, 0, 0, 1.01); });
    invalid += rejects([&]{ select(one, 1, one_mask, 1, -0.1, 0.0, 0, 1, 0.5); });
    invalid += rejects([&]{ select(one, 1, one_mask, 1, 1.0, 0.0, 0, 0, std::numeric_limits<double>::quiet_NaN()); });
    invalid += rejects([&]{ select(one, 1, one_mask, 1, 1.0, 0.0, 0, 0, 0.5); });
    invalid += rejects([&]{ select(one, 1, empty, 1, 1.0, 0.0, 0, 0, 0.5); });
    invalid += rejects([&]{ select(one, 1, padding, 1, 1.0, 0.0, 0, 0, 0.5); });
    invalid += rejects([&]{ select(nonfinite, 2, nonfinite_mask, 1, 1.0, 0.0, 0, 0, 0.5); });
    std::cout << "invalid " << invalid << '\n';

    size_t vocab = 0, mask_bytes = 0, top_k = 0;
    double temperature = 1.0, min_p = 0.0, top_p = 1.0;
    uint64_t draw53 = 0;
    while (std::cin >> vocab >> temperature >> min_p >> top_k >> top_p >> draw53 >> mask_bytes) {
        std::vector<uint8_t> mask(mask_bytes);
        std::vector<float> scores(vocab);
        for (auto& byte : mask) { unsigned int value; std::cin >> value; byte = static_cast<uint8_t>(value); }
        for (auto& score : scores) std::cin >> score;
        try {
            const auto result = select(scores.data(), scores.size(), mask.data(), mask.size(),
                temperature, min_p, draw53, top_k, top_p);
            std::cout << "R " << result.token << ' ' << result.legal_count << ' '
                      << result.kept_count << '\n';
        } catch (const std::exception&) { std::cout << "R -1 0 0\n"; }
    }
    return 0;
}
'''


class MinPTopKTopPNativeTests(unittest.TestCase):
    @staticmethod
    def _f32(value):
        return struct.unpack("<f", struct.pack("<f", value))[0]

    @classmethod
    def _reference(cls, scores, mask, temperature, min_p, top_k, top_p, draw53):
        legal = [i for i in range(len(scores)) if mask[i // 8] & (1 << (i % 8))]
        maximum = max(scores[i] for i in legal)
        cutoff = -math.inf if min_p == 0 else maximum + math.log(min_p)
        ranked = [i for i in legal if scores[i] >= cutoff]
        ranked.sort(key=lambda i: (-scores[i], i))
        if top_k:
            ranked = ranked[:top_k]
        weights = {i: math.exp((scores[i] - scores[ranked[0]]) / temperature)
                   for i in ranked}
        if top_p < 1:
            ranked_total = 0.0
            for i in ranked:
                ranked_total += weights[i]
            threshold = top_p * ranked_total
            cumulative = 0.0
            prefix = 0
            for i in ranked:
                cumulative += weights[i]
                prefix += 1
                if cumulative >= threshold:
                    break
            ranked = ranked[:max(1, prefix)]
        retained = set(ranked)
        total = 0.0
        for i in sorted(retained):
            total += weights[i]
        target = (draw53 / float(1 << 53)) * total
        cumulative = 0.0
        last = None
        for i in range(len(scores)):
            if i not in retained:
                continue
            weight = weights[i]
            if weight <= 0:
                continue
            last = i
            cumulative += weight
            if cumulative > target:
                return i, len(legal), len(retained)
        return last, len(legal), len(retained)

    @classmethod
    def setUpClass(cls):
        compiler = shutil.which("g++")
        if compiler is None:
            raise AssertionError("g++ is required for native helper tests on llm-60")
        cls.temp = tempfile.TemporaryDirectory(prefix="min-p-top-k-top-p-native-")
        root = Path(cls.temp.name)
        source = root / "test_selector.cpp"
        cls.executable = root / "test_selector"
        source.write_text(CPP, encoding="utf-8")
        result = subprocess.run(
            [compiler, "-O2", "-std=c++20", "-Wall", "-Wextra", "-Werror",
             "-I", str(HEADER.parent), str(source), "-o", str(cls.executable)],
            capture_output=True, text=True, check=False)
        if result.returncode:
            raise AssertionError(f"native selector compile failed: {result.stderr}")
        cls.rows = {line.split()[0]: line.split()[1:]
                    for line in subprocess.run([str(cls.executable)], check=True,
                                               input="", capture_output=True, text=True).stdout.splitlines()}

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_top_k_uses_raw_logit_order_and_token_id_ties(self):
        self.assertEqual(self.rows["topk"], ["0", "5", "2", "1"])

    def test_top_p_keeps_smallest_ranked_prefix_and_samples_token_order(self):
        self.assertEqual(self.rows["topp"], ["0", "2", "1"])
        self.assertEqual(self.rows["token_order"], ["0", "2"])
        self.assertEqual(self.rows["temperature_top_p"], ["1", "2"])

    def test_filter_order_is_min_p_then_top_k_then_temperature_then_top_p(self):
        self.assertEqual(self.rows["combined"], ["0", "4", "2"])
        self.assertEqual(self.rows["minp"], ["3", "2"])
        self.assertEqual(self.rows["temperature_zero"], ["1", "3", "1", "1"])

    def test_default_top_k_and_top_p_match_legacy_v1_result(self):
        row = self.rows["legacy"]
        self.assertEqual(row[:3], row[3:6])
        self.assertEqual(row[6], row[7])

    def test_invalid_limits_empty_support_padding_and_nonfinite_logits_reject(self):
        self.assertEqual(self.rows["invalid"], ["7"])

    def test_top_k_larger_than_support_clamps_to_all_min_p_candidates(self):
        self.assertEqual(self.rows["clamped"], ["0", "5", "5"])

    def test_v2_builder_is_opt_in_and_retains_v1_command(self):
        transformed = builder.instrument_min_p_selector_v2(DRIVER_STUB)
        self.assertIn('\\"sampling_protocol\\":\\"full-vocab-min-p-u53-v2\\"', transformed)
        self.assertIn('\\"sampling_v1_protocol\\":\\"full-vocab-min-p-u53-v1\\"', transformed)
        self.assertIn('command == "sample_mask"', transformed)
        self.assertIn('command == "sample_mask_v2"', transformed)
        self.assertIn('consumed temperature min_p top_k top_p draw53', transformed)
        self.assertIn("std::from_chars", transformed)
        self.assertIn("top_k_text[0] == '-'", transformed)
        self.assertIn("temperature < 0.0", transformed)
        self.assertIn("temperature == 0.0 ? draw53 != 0", transformed)
        self.assertIn('std::from_chars', transformed)
        self.assertIn('top_k_text[0] ==', transformed)
        self.assertIn('\\"top_k\\":', transformed)
        self.assertIn('\\"top_p\\":', transformed)
        self.assertIn("std::setprecision(17)", transformed)
        for counter in ("sampling_v2_calls", "sampling_v2_logit_bytes",
                        "sampling_v2_mask_bytes", "sampling_v2_ns"):
            self.assertIn(counter, transformed)
        for aggregate in ("++sampling_calls", "sampling_logit_bytes +=",
                          "sampling_mask_bytes +=", "sampling_ns += selection_elapsed_ns"):
            self.assertIn(aggregate, transformed)
        for subset in ("++sampling_v2_calls", "sampling_v2_logit_bytes +=",
                       "sampling_v2_mask_bytes +=",
                       "sampling_v2_ns += selection_elapsed_ns"):
            self.assertIn(subset, transformed)

    def test_top_p_denominator_uses_token_id_accumulation_order(self):
        self.assertEqual(self.rows["top_p_sum_order"], ["1"])

    def test_seeded_v2_masks_match_independent_binary64_reference(self):
        rng = random.Random(20261016)
        inputs = []
        expected = []
        for vocab in (1, 7, 8, 9, 31, 257, 1025):
            for _ in range(8):
                scores = [self._f32(rng.uniform(-30.0, 30.0)) for _ in range(vocab)]
                mask = bytearray((vocab + 7) // 8)
                for token in range(vocab):
                    if rng.getrandbits(1):
                        mask[token // 8] |= 1 << (token % 8)
                if not any(mask):
                    mask[0] = 1
                temperature = rng.choice((0.1, 0.5, 1.0, 2.0))
                min_p = rng.choice((0.0, 0.01, 0.1, 0.5))
                support = sum((mask[i // 8] >> (i % 8)) & 1 for i in range(vocab))
                top_k = rng.choice((0, 1, 3, support + 2))
                top_p = rng.choice((0.05, 0.2, 0.6, 0.95, 1.0))
                draw53 = rng.randrange(1 << 53)
                expected.append(self._reference(scores, bytes(mask), temperature,
                                                min_p, top_k, top_p, draw53))
                row = [str(vocab), repr(temperature), repr(min_p), str(top_k),
                       repr(top_p), str(draw53), str(len(mask))]
                row.extend(str(value) for value in mask)
                row.extend(format(value, ".9g") for value in scores)
                inputs.append(" ".join(row))
        result = subprocess.run([str(self.executable)], input="\n".join(inputs) + "\n",
                                check=True, capture_output=True, text=True)
        actual = [tuple(map(int, line.split()[1:])) for line in result.stdout.splitlines()
                  if line.startswith("R ")]
        self.assertEqual(actual, expected)


if __name__ == "__main__":
    unittest.main()
