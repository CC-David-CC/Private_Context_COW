"""CPU-only C++ tests for the full-vocabulary Min-P selector helper."""
import importlib.util
import math
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
spec = importlib.util.spec_from_file_location("min_p_selector_builder", BUILDER_PATH)
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)

DRIVER = r'''#include "grammar_select.h"
#include <fstream>
    std::vector<float> logits((size_t)forward.vocab());
    uint64_t grammar_selection_calls = 0, grammar_selection_logit_bytes = 0, grammar_selection_mask_bytes = 0;
    std::cout << "{\"status\":\"ready\",\"grammar_selection_protocol\":\"finite-mask-argmax-v1\"}" << std::endl;
        } else if (command == "select_mask") {
            std::cout << "{\"status\":\"ok\",\"selection_protocol\":\"finite-mask-argmax-v1\"}" << std::endl;
        } else if (command == "generate") {
            std::cout << "{\"status\":\"ok\",\"prefill_chunks\":" << prefill_chunks
                      << ",\"grammar_selection_mask_bytes\":" << grammar_selection_mask_bytes
                      << "}" << std::endl;
'''


CPP = r'''#include <cmath>
#include <cstdint>
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
    const float tie[] = {2.0f, 2.0f};
    const uint8_t tie_mask[] = {0x03};
    auto tie_first = select(tie, 2, tie_mask, 1, 0.7, 1.0, 0);
    auto tie_last = select(tie, 2, tie_mask, 1, 0.7, 1.0, (uint64_t(1) << 53) - 1);
    std::cout << "ties " << tie_first.token << ' ' << tie_last.token << ' '
              << tie_first.legal_count << ' ' << tie_first.kept_count << '\n';

    const float cutoff_scores[] = {0.0f, -2.0f, -10.0f};
    const uint8_t all3[] = {0x07};
    auto cutoff = select(cutoff_scores, 3, all3, 1, 1.0, 0.1, 0);
    std::cout << "cutoff " << cutoff.token << ' ' << cutoff.legal_count << ' '
              << cutoff.kept_count << '\n';

    const float temp_scores[] = {0.0f, -1.0f};
    const uint8_t both[] = {0x03};
    auto cold = select(temp_scores, 2, both, 1, 0.5, 0.0, uint64_t(8) * (uint64_t(1) << 53) / 10);
    auto warm = select(temp_scores, 2, both, 1, 2.0, 0.0, uint64_t(8) * (uint64_t(1) << 53) / 10);
    std::cout << "temperature " << cold.token << ' ' << warm.token << '\n';

    const float underflow_scores[] = {0.0f, -1000.0f, -1.0f};
    auto underflow = select(underflow_scores, 3, all3, 1, 1.0, 0.0,
                            (uint64_t(1) << 53) - 1);
    std::cout << "underflow " << underflow.token << ' ' << underflow.kept_count << '\n';

    const float masked_scores[] = {1.0f, 100.0f, 2.0f};
    const uint8_t mask02[] = {0x05};
    auto masked = select(masked_scores, 3, mask02, 1, 1.0, 0.0, 0);
    std::cout << "masked " << masked.token << ' ' << masked.legal_count << '\n';

    const float one[] = {0.0f};
    const uint8_t one_mask[] = {0x01};
    const uint8_t empty_mask[] = {0x00};
    const uint8_t bad_padding[] = {0x81};
    const uint8_t* short_mask = nullptr;
    const float bad_logits[] = {0.0f, std::numeric_limits<float>::quiet_NaN()};
    const uint8_t bad_logits_mask[] = {0x01};
    int invalid = 0;
    invalid += rejects([&]{ select(one, 1, one_mask, 1, 0.0, 0.0, 0); });
    invalid += rejects([&]{ select(one, 1, one_mask, 1, std::numeric_limits<double>::infinity(), 0.0, 0); });
    invalid += rejects([&]{ select(one, 1, one_mask, 1, 1.0, -0.1, 0); });
    invalid += rejects([&]{ select(one, 1, one_mask, 1, 1.0, 1.1, 0); });
    invalid += rejects([&]{ select(one, 1, one_mask, 1, 1.0, 0.0, uint64_t(1) << 53); });
    invalid += rejects([&]{ select(one, 1, empty_mask, 1, 1.0, 0.0, 0); });
    invalid += rejects([&]{ select(one, 1, short_mask, 0, 1.0, 0.0, 0); });
    invalid += rejects([&]{ select(one, 1, bad_padding, 1, 1.0, 0.0, 0); });
    invalid += rejects([&]{ select(bad_logits, 2, bad_logits_mask, 1, 1.0, 0.0, 0); });
    std::cout << "invalid " << invalid << '\n';
    size_t vocab, mask_bytes;
    double temperature, min_p;
    uint64_t draw53;
    while (std::cin >> vocab >> temperature >> min_p >> draw53 >> mask_bytes) {
        std::vector<uint8_t> mask(mask_bytes);
        std::vector<float> scores(vocab);
        for (auto& byte : mask) { unsigned int value; std::cin >> value; byte = static_cast<uint8_t>(value); }
        for (auto& score : scores) std::cin >> score;
        try {
            const auto result = select(scores.data(), scores.size(), mask.data(), mask.size(),
                                       temperature, min_p, draw53);
            std::cout << "R " << result.token << ' ' << result.legal_count << ' '
                      << result.kept_count << '\n';
        } catch (const std::exception&) { std::cout << "R -1 0 0\n"; }
    }
    return 0;
}
'''


def _f32(value):
    return struct.unpack("<f", struct.pack("<f", value))[0]


def _reference(scores, mask, temperature, min_p, draw53):
    if not scores or len(mask) != (len(scores) + 7) // 8:
        raise ValueError("width")
    if len(scores) % 8 and mask[-1] & ~((1 << (len(scores) % 8)) - 1):
        raise ValueError("padding")
    if any(not math.isfinite(score) for score in scores):
        raise ValueError("nonfinite")
    legal = [i for i in range(len(scores)) if mask[i // 8] & (1 << (i % 8))]
    if not legal:
        raise ValueError("empty")
    maximum = max(scores[i] for i in legal)
    cutoff = -math.inf if min_p == 0 else maximum + math.log(min_p)
    kept = [i for i in legal if scores[i] >= cutoff]
    weights = {i: math.exp((scores[i] - maximum) / temperature) for i in kept}
    total = 0.0
    for i in kept:
        total += weights[i]
    target = (draw53 / float(1 << 53)) * total
    cumulative = 0.0
    last_positive = None
    for i in kept:
        weight = weights[i]
        if weight <= 0:
            continue
        last_positive = i
        cumulative += weight
        if cumulative > target:
            return i, len(legal), len(kept)
    return last_positive, len(legal), len(kept)


class MinPSelectorNativeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        compiler = shutil.which("g++")
        if compiler is None:
            raise AssertionError("g++ is required for native helper tests on the Linux build host")
        cls.temp = tempfile.TemporaryDirectory(prefix="min-p-selector-native-")
        root = Path(cls.temp.name)
        source = root / "test_min_p.cpp"
        cls.executable = root / "test_min_p"
        source.write_text(CPP, encoding="utf-8")
        compile_result = subprocess.run(
            [compiler, "-O2", "-std=c++20", "-Wall", "-Wextra", "-Werror",
             "-I", str(HEADER.parent), str(source), "-o", str(cls.executable)],
            check=False, capture_output=True, text=True)
        if compile_result.returncode:
            raise AssertionError(f"native helper compile failed: {compile_result.stderr}")
        cls.output = subprocess.run([str(cls.executable)], input="", check=True, capture_output=True,
                                    text=True).stdout.splitlines()

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_full_support_min_p_cutoff_and_draw_order(self):
        rows = {line.split()[0]: line.split()[1:] for line in self.output}
        self.assertEqual(rows["ties"], ["0", "1", "2", "2"])
        self.assertEqual(rows["cutoff"], ["0", "3", "2"])
        self.assertEqual(rows["masked"], ["0", "2"])

    def test_temperature_and_zero_weight_tokens(self):
        rows = {line.split()[0]: line.split()[1:] for line in self.output}
        self.assertEqual(rows["temperature"], ["0", "1"])
        self.assertEqual(rows["underflow"], ["2", "3"])

    def test_rejects_invalid_parameters_logits_and_masks(self):
        rows = {line.split()[0]: line.split()[1:] for line in self.output}
        self.assertEqual(rows["invalid"], ["9"])

    def test_seeded_full_vocabulary_cases_match_binary64_reference(self):
        rng = random.Random(20261016)
        cases, expected = [], []
        for vocab in (1, 7, 8, 9, 31, 257, 4099):
            for _ in range(12):
                scores = [_f32(rng.uniform(-100.0, 100.0)) for _ in range(vocab)]
                mask = bytearray((vocab + 7) // 8)
                for token in range(vocab):
                    if rng.getrandbits(1):
                        mask[token // 8] |= 1 << (token % 8)
                if not any(mask):
                    mask[0] = 1
                temperature = rng.choice((0.07, 0.3, 0.7, 1.0, 2.5))
                min_p = rng.choice((0.0, 0.001, 0.05, 0.2, 0.8, 1.0))
                draw53 = rng.randrange(1 << 53)
                expected.append(_reference(scores, bytes(mask), temperature, min_p, draw53))
                fields = [str(vocab), repr(temperature), repr(min_p), str(draw53), str(len(mask))]
                fields.extend(str(value) for value in mask)
                fields.extend(format(value, ".9g") for value in scores)
                cases.append(" ".join(fields))
        result = subprocess.run([str(self.executable)], input="\n".join(cases) + "\n",
                                check=True, capture_output=True, text=True)
        rows = [line.split() for line in result.stdout.splitlines() if line.startswith("R ")]
        actual = [(int(row[1]), int(row[2]), int(row[3])) for row in rows]
        self.assertEqual(actual, expected)

    def test_generated_driver_adds_opt_in_sampling_command_and_separate_counters(self):
        result = builder.instrument_min_p_selector(DRIVER)
        self.assertIn('sampling_protocol', result)
        self.assertIn('full-vocab-min-p-u53-v1', result)
        self.assertIn('command == "sample_mask"', result)
        self.assertIn('\\"operator\\":\\"full-vocab-min-p\\"', result)
        for field in ('token', 'consumed', 'logits_count', 'mask_bytes',
                      'legal_count', 'kept_count', 'draw53'):
            self.assertIn('\\"' + field + '\\":', result)
        for counter in ('sampling_calls', 'sampling_logit_bytes', 'sampling_mask_bytes', 'sampling_ns'):
            self.assertIn(counter, result)
        start = result.index('command == "sample_mask"')
        end = result.index('command == "generate"', start)
        command = result[start:end]
        self.assertIn('std::quoted(mask_path)', command)
        self.assertIn('(in >> extra)', command)
        self.assertIn('std::isfinite(temperature)', command)
        self.assertIn('std::isfinite(min_p)', command)
        self.assertIn('draw53 >= (uint64_t(1) << 53)', command)
        self.assertNotIn('forward.run(', command)
        self.assertNotIn('session.begin(', command)
        self.assertNotIn('session.finish(', command)
        self.assertNotIn('random', command.lower())

    def test_generated_driver_transform_fails_closed_on_missing_or_duplicate_anchors(self):
        with self.assertRaisesRegex(ValueError, "ready"):
            builder.instrument_min_p_selector(DRIVER.replace('grammar_selection_protocol', 'removed'))
        with self.assertRaisesRegex(ValueError, "stats"):
            builder.instrument_min_p_selector(DRIVER.replace('grammar_selection_mask_bytes', 'removed'))
        with self.assertRaisesRegex(ValueError, "generate"):
            builder.instrument_min_p_selector(DRIVER.replace('command == "generate"',
                                                               'command == "not_generate"'))


if __name__ == "__main__":
    unittest.main()
