"""CPU native checks for the opt-in v5 XTC-after-Top-P selector."""
from __future__ import annotations

import shutil
import random
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
NATIVE = ROOT / "native"
sys.path.insert(0, str(ROOT / "scripts"))
from build_min_p_selector_probe import _append_v5_response_receipt

DRIVER = r'''#include "min_p_select.h"
#include <cstdint>
#include <iostream>
#include <limits>
#include <vector>

using min_p_select::DryConfig;
static constexpr uint64_t absent = uint64_t(1) << 53;

int main() {
  const std::vector<uint32_t> history;
  const std::vector<uint32_t> breakers;
  const DryConfig dry{0.0, 1.75, 2, 256};
  const std::vector<uint8_t> full{0x07};
  const std::vector<float> scores{3.0f, 2.0f, 1.0f};
  auto emit = [](const min_p_select::V5Result& r) {
    std::cout << "OK " << r.base.token << ' ' << r.base.kept_count << ' '
              << r.base.effective_temperature << ' ' << r.draw_count << ' '
              << r.activation_draws << ' ' << r.activated << ' ' << r.removed_count << '\n';
  };
  try {
    const auto legacy = min_p_select::select_v4(scores.data(), scores.size(), full.data(), full.size(),
        1.0, 0.0, 0, 1.0, 0, history, breakers, dry, 1.0, 0.0, 1.0);
    const auto disabled = min_p_select::select_v5(scores.data(), scores.size(), full.data(), full.size(),
        1.0, 0.0, 0, 1.0, 0, absent, history, breakers, dry, 1.0, 0.0, 1.0, 0.0, 0.1);
    std::cout << "PARITY " << legacy.token << ' ' << disabled.base.token << ' '
              << disabled.draw_count << ' ' << disabled.activation_draws << ' '
              << legacy.v5_adjusted_logits.empty() << '\n';

    emit(min_p_select::select_v5(scores.data(), scores.size(), full.data(), full.size(),
        1.0, 0.0, 0, 1.0, 0, (uint64_t(1) << 53) - 1, history, breakers, dry,
        1.0, 0.0, 1.0, 1.0, 0.2));

    const auto unactivated = min_p_select::select_v5(scores.data(), scores.size(), full.data(), full.size(),
        1.0, 0.0, 0, 1.0, (uint64_t(1) << 53) - 1, 0, history, breakers, dry,
        1.0, 0.0, 1.0, 0.5, 0.2);
    const auto second_draw_legacy = min_p_select::select_v4(scores.data(), scores.size(), full.data(), full.size(),
        1.0, 0.0, 0, 1.0, 0, history, breakers, dry, 1.0, 0.0, 1.0);
    std::cout << "SECOND " << second_draw_legacy.token << ' ' << unactivated.base.token << ' '
              << unactivated.draw_count << ' ' << unactivated.activation_draws << ' '
              << unactivated.activated << '\n';

    const auto threshold_disabled = min_p_select::select_v5(scores.data(), scores.size(), full.data(), full.size(),
        1.0, 0.0, 0, 1.0, 0, absent, history, breakers, dry,
        1.0, 0.0, 1.0, 1.0, 0.75);
    std::cout << "DISABLED " << threshold_disabled.base.token << ' '
              << threshold_disabled.draw_count << ' ' << threshold_disabled.activation_draws << '\n';

    const auto topk_topp = min_p_select::select_v5(scores.data(), scores.size(), full.data(), full.size(),
        1.0, 0.0, 2, 0.75, 0, 0, history, breakers, dry,
        1.0, 0.0, 1.0, 1.0, 0.2);
    std::cout << "TOPP " << topk_topp.base.token << ' '
              << topk_topp.pre_xtc_count << ' ' << topk_topp.removed_count << ' '
              << topk_topp.base.kept_count << ' ' << topk_topp.draw_count << '\n';

    const std::vector<float> underflow{0.0f, -1000.0f, -2000.0f};
    emit(min_p_select::select_v5(underflow.data(), underflow.size(), full.data(), full.size(),
        1.0, 0.0, 0, 1.0, 0, 0, history, breakers, dry,
        1.0, 0.0, 1.0, 1.0, 0.0));

    const std::vector<uint8_t> one{0x01};
    emit(min_p_select::select_v5(scores.data(), scores.size(), one.data(), one.size(),
        1.0, 0.0, 0, 1.0, (uint64_t(1) << 53) - 1, absent, history, breakers, dry,
        1.0, 0.0, 1.0, 1.0, 0.0));

    const auto zero_effective = min_p_select::select_v5(scores.data(), scores.size(), full.data(), full.size(),
        0.2, 0.0, 1, 1.0, 0, absent, history, breakers, dry,
        1.0, 0.3, 1.0, 1.0, 0.0);
    std::cout << "ZERO " << zero_effective.base.token << ' '
              << zero_effective.base.effective_temperature << ' '
              << zero_effective.draw_count << ' '
              << zero_effective.activation_draws << '\n';

    int rejected = 0;
    const auto must_reject = [&](uint64_t second, double probability, double threshold) {
      try {
        (void) min_p_select::select_v5(scores.data(), scores.size(), full.data(), full.size(),
            1.0, 0.0, 0, 1.0, 0, second, history, breakers, dry,
            1.0, 0.0, 1.0, probability, threshold);
      } catch (const std::exception&) { ++rejected; }
    };
    must_reject(absent, 1.0, 0.0); // second draw required by active, nonsingleton XTC
    must_reject(0, -0.1, 0.0);
    must_reject(0, 0.1, 1.1);
    std::cout << "REJECT " << rejected << '\n';
  } catch (const std::exception& error) {
    std::cout << "ERROR " << error.what() << '\n';
    return 2;
  }
}
'''

PREPARED_DRIVER = r'''#include "xtc_select.h"
#include <iostream>
#include <vector>
int main() {
  size_t n = 0, count = 0;
  double temperature = 0.0, probability = 0.0, threshold = 0.0;
  uint64_t draw = 0;
  while (std::cin >> n >> count >> temperature >> probability >> threshold >> draw) {
    std::vector<float> scores(n);
    std::vector<double> weights(n);
    std::vector<size_t> support(count);
    for (float& value : scores) std::cin >> value;
    for (double& value : weights) std::cin >> value;
    for (size_t& value : support) std::cin >> value;
    const auto result = xtc_select::filter_prepared_weights(
        scores, support, weights, temperature, probability, threshold, draw);
    std::cout << result.activation_draws << ' ' << result.activated << ' '
              << result.pre_xtc_count << ' ' << result.kept_count << ' '
              << result.removed_count;
    for (size_t token : result.retained_ids) std::cout << ' ' << token;
    std::cout << '\n';
  }
  return std::cin.eof() ? 0 : 2;
}
'''


class XtcPipelineNativeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        compiler = shutil.which("g++") or shutil.which("c++")
        if compiler is None:
            raise RuntimeError("g++/c++ is required on the llm-60 CPU test host")
        cls.tmp = tempfile.TemporaryDirectory(prefix="xtc-v5-native-")
        root = Path(cls.tmp.name)
        source, cls.exe = root / "driver.cpp", root / "driver"
        source.write_text(DRIVER, encoding="utf-8", newline="\n")
        proc = subprocess.run(
            [compiler, "-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror",
             "-DMIN_P_SELECT_ENABLE_V4=1", "-DMIN_P_SELECT_ENABLE_V5=1",
             "-I", str(NATIVE), str(source), "-o", str(cls.exe)],
            capture_output=True, text=True, check=False)
        if proc.returncode:
            raise AssertionError(f"native v5 compile failed: {proc.stderr}")
        prepared_source, cls.prepared_exe = root / "prepared.cpp", root / "prepared"
        prepared_source.write_text(PREPARED_DRIVER, encoding="utf-8", newline="\n")
        proc = subprocess.run(
            [compiler, "-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror",
             "-I", str(NATIVE), str(prepared_source), "-o", str(cls.prepared_exe)],
            capture_output=True, text=True, check=False)
        if proc.returncode:
            raise AssertionError(f"prepared XTC native compile failed: {proc.stderr}")

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_native_v5_matches_disabled_parity_draw_routing_and_underflow_contract(self):
        proc = subprocess.run([str(self.exe)], capture_output=True, text=True, check=False)
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        lines = proc.stdout.splitlines()
        self.assertEqual(lines[0], "PARITY 0 0 1 0 1")
        active = lines[1].split()
        self.assertEqual(active[:2], ["OK", "2"])
        self.assertEqual(active[2:], ["2", "1", "2", "1", "1", "1"])
        second = lines[2].split()
        self.assertEqual(second, ["SECOND", "0", "0", "2", "1", "0"])
        disabled = lines[3].split()
        self.assertEqual(disabled, ["DISABLED", "0", "1", "0"])
        topk_topp = lines[4].split()
        self.assertEqual(topk_topp, ["TOPP", "1", "2", "1", "1", "2"])
        underflow = lines[5].split()
        self.assertEqual(underflow[:2], ["OK", "2"])
        self.assertEqual(underflow[2:], ["1", "1", "2", "1", "1", "2"])
        singleton = lines[6].split()
        self.assertEqual(singleton[:2], ["OK", "0"])
        self.assertEqual(singleton[2:], ["1", "1", "1", "0", "0", "0"])
        zero = lines[7].split()
        self.assertEqual(zero[0], "ZERO")
        self.assertEqual(zero[2:], ["0", "1", "0"])
        self.assertEqual(lines[8], "REJECT 3")

    @staticmethod
    def _reference(scores, weights, support, temperature, probability, threshold, draw):
        total = 0.0
        for token in support:
            total += weights[token]
        if temperature == 0.0 or probability == 0.0 or threshold > 0.5 or len(support) < 2:
            return 0, False, 0, support
        if draw / float(1 << 53) > probability:
            return 1, False, 0, support
        ranked = sorted(support, key=lambda token: (-scores[token], token))
        eligible = []
        for token in ranked:
            if weights[token] / total < threshold:
                break
            eligible.append(token)
        removed = eligible[:-1] if len(eligible) >= 2 else []
        retained = [token for token in support if token not in set(removed)]
        return 1, True, len(removed), retained

    def test_prepared_filter_matches_independent_seeded_reference(self):
        rng = random.Random(0x585443)
        cases = []
        for index in range(40):
            n = 4 + index % 13
            support = sorted(token for token in range(n) if rng.random() < 0.7)
            if len(support) < 2:
                support = [0, 1]
            scores = [float(rng.randrange(-40, 41)) for _ in range(n)]
            peak = max(scores[token] for token in support)
            weights = [0.0] * n
            for token in support:
                weights[token] = rng.choice((0.0, 1.0, 2.0, 4.0)) if index % 5 == 0 else \
                    math.exp((scores[token] - peak) / rng.choice((0.5, 1.0, 1.5)))
            if not any(weights[token] > 0 for token in support):
                weights[support[0]] = 1.0
            temperature = rng.choice((0.0, 0.5, 1.0, 2.0))
            probability = rng.choice((0.0, 0.1, 0.5, 1.0))
            threshold = rng.choice((0.0, 0.01, 0.15, 0.5, 0.75, 1.0))
            draw = rng.choice((0, (1 << 52), (1 << 53) - 1))
            cases.append((scores, weights, support, temperature, probability, threshold, draw))

        stdin_lines = []
        expected = []
        for scores, weights, support, temperature, probability, threshold, draw in cases:
            values = [len(scores), len(support), temperature, probability, threshold, draw]
            stdin_lines.append(" ".join(map(str, values)))
            stdin_lines.append(" ".join(map(repr, scores)))
            stdin_lines.append(" ".join(map(repr, weights)))
            stdin_lines.append(" ".join(map(str, support)))
            expected.append(self._reference(scores, weights, support, temperature,
                                            probability, threshold, draw))
        proc = subprocess.run([str(self.prepared_exe)], input="\n".join(stdin_lines) + "\n",
                              capture_output=True, text=True, check=False)
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        rows = [line.split() for line in proc.stdout.splitlines()]
        self.assertEqual(len(rows), len(cases))
        for index, (row, case, expected_row) in enumerate(zip(rows, cases, expected)):
            scores, weights, support, temperature, probability, threshold, draw = case
            self.assertEqual(int(row[0]), expected_row[0], index)
            self.assertEqual(bool(int(row[1])), expected_row[1], index)
            self.assertEqual(int(row[2]), len(support), index)
            self.assertEqual(int(row[3]), len(expected_row[3]), index)
            self.assertEqual(int(row[4]), expected_row[2], index)
            self.assertEqual([int(token) for token in row[5:]], expected_row[3], index)

    def test_builder_emits_single_finite_mass_receipt_expression(self):
        v4_response = (r'<< ",\"min_p_kept_count\":" << selection.min_p_kept_count'
                       r' << ",\"top_k_kept_count\":" << selection.top_k_kept_count'
                       r' << ",\"draw53\":" << draw53'
                       r' << ",\"effective_temperature\":" << selection.base.effective_temperature << "}";')
        generated = _append_v5_response_receipt(v4_response)
        self.assertEqual(generated.count(r'<< ",\"min_p_kept_count\":" << selection.min_p_kept_count'), 1)
        self.assertEqual(generated.count(r'<< ",\"top_k_kept_count\":" << selection.top_k_kept_count'), 1)
        self.assertEqual(generated.count(r'<< ",\"draw53\":" << draw53'), 1)
        self.assertEqual(generated.count(r'<< ",\"total_weight\":" << selection.base.total_weight'), 1)
        self.assertIn(r'<< ",\"draw53_second\":" << selection.draw53_second', generated)
        duplicate = (r'<< ",\"total_weight\":" << selection.base.total_weight'
                     r' << ",\"total_weight\":" << selection.base.total_weight')
        malformed_v4_response = v4_response.replace(
            r'<< ",\"draw53\":" << draw53',
            r'<< ",\"draw53\":" << draw53 ' + duplicate)
        with self.assertRaisesRegex(ValueError, "total_weight"):
            _append_v5_response_receipt(malformed_v4_response)


if __name__ == "__main__":
    unittest.main()
