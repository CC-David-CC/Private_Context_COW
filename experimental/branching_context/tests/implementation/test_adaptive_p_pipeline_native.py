"""CPU parity and command-shape checks for native Adaptive-P v7."""
from __future__ import annotations

import inspect
import json
import math
import re
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HEADER = ROOT / "native" / "adaptive_p_select.h"
import sys
sys.path.insert(0, str(ROOT / "scripts"))
import build_min_p_selector_probe as builder


CPP = r'''#include "min_p_select.h"
#include "adaptive_p_select.h"
#include <iomanip>
#include <iostream>
#include <vector>
int main() {
  std::cout << std::setprecision(17);
  {
    const std::vector<double> weights{1.0, 0.0, 0.25, 0.125};
    const std::vector<std::uint32_t> ids{0, 2, 3};
    const adaptive_p_select::State state{0.9, 1.0};
    const auto result = adaptive_p_select::select_prepared(
        weights, ids, state, 0.55, 0.90, (std::uint64_t{1} << 52));
    std::cout << result.token << ' ' << result.original_probability << ' '
              << *result.adapted_target << ' ' << result.next_weighted_sum << ' '
              << result.next_total_weight << ' ' << result.total_weight << '\n';
  }
  {
    const std::vector<double> weights{0.0, 0.2, 0.8};
    const std::vector<std::uint32_t> ids{1, 2};
    const adaptive_p_select::State state{0.7, 2.0};
    const auto result = adaptive_p_select::select_prepared(
        weights, ids, state, -1.0, 0.5, (std::uint64_t{1} << 53) - 1);
    std::cout << result.token << ' ' << result.original_probability << ' '
              << result.next_weighted_sum << ' ' << result.next_total_weight << ' '
              << result.feedback_enabled << '\n';
  }
  {
    const std::vector<double> weights{0.0, 0.0, 2.5, 0.0};
    const std::vector<std::uint32_t> ids{2};
    const adaptive_p_select::State state{0.0, 1.0};
    const auto result = adaptive_p_select::select_prepared(
        weights, ids, state, 0.0, 0.0, 0);
    std::cout << result.token << ' ' << result.original_probability << ' '
              << *result.adapted_target << ' ' << result.next_weighted_sum << ' '
              << result.next_total_weight << '\n';
  }
  {
    // This is the same history extent calculation used by the v7 handler.
    // Disabled DRY accepts only an empty tail even when last_n is configured;
    // active DRY requires the entire bounded native-session tail.
    const auto disabled = adaptive_p_select::expected_dry_history_count(1000, 0.0, 256);
    const auto active = adaptive_p_select::expected_dry_history_count(1000, 0.8, 256);
    const bool disabled_empty_ok = disabled == 0 && std::vector<uint32_t>{}.size() == disabled;
    const bool active_truncated_ok = std::vector<uint32_t>(255).size() == active;
    const bool active_exact_ok = std::vector<uint32_t>(256).size() == active;
    std::cout << disabled << ' ' << active << ' ' << disabled_empty_ok << ' '
              << active_truncated_ok << ' ' << active_exact_ok << '\n';
  }
}
'''


def _expected(weights, ids, state, target, decay, draw53):
    total = sum(weights[i] for i in ids)
    original = [weights[i] / total for i in ids]
    feedback = target >= 0.0
    adapted = None
    sampling = original
    if feedback:
        mean = state[0] / state[1]
        adapted = min(1.0, max(0.0, 2.0 * target - mean))
        transformed = []
        for probability in original:
            distance = abs((probability - adapted) / 0.3)
            transformed.append(5.0 - 10.0 * distance * distance / (1.0 + distance))
        maximum = max(transformed)
        raw = [math.exp(value - maximum) for value in transformed]
        transformed_total = sum(raw)
        sampling = [value / transformed_total for value in raw]
    sample_total = sum(sampling)
    threshold = (draw53 / float(1 << 53)) * sample_total
    cumulative = 0.0
    index = len(ids) - 1
    for i, weight in enumerate(sampling):
        if weight > 0.0:
            cumulative += weight
            if cumulative > threshold:
                index = i
                break
    probability = original[index]
    next_weighted = probability + decay * state[0] if feedback else state[0]
    next_total = 1.0 + decay * state[1] if feedback else state[1]
    return ids[index], probability, adapted, next_weighted, next_total, sample_total, feedback


class AdaptivePNativeTests(unittest.TestCase):
    def test_prepared_transform_preserves_reference_left_association_bitwise(self):
        # This distance is reachable from a finite probability and adapted target.
        # It distinguishes (10*d)*d from 10*(d*d) by two binary64 ULPs.
        d = 0.823182772635959
        left = 5.0 - ((10.0 * d) * d) / (1.0 + d)
        regrouped = 5.0 - (10.0 * (d * d)) / (1.0 + d)
        self.assertEqual(left.hex(), "0x1.4883a3d00fb32p+0")
        self.assertEqual(regrouped.hex(), "0x1.4883a3d00fb34p+0")
        self.assertNotEqual(left, regrouped)

        cpp = r'''#include "adaptive_p_select.h"
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
int main() {
  const double d = 0.823182772635959;
  const double actual = adaptive_p_select::transform_distance(d);
  const double expected = 5.0 - ((10.0 * d) * d) / (1.0 + d);
  std::uint64_t actual_bits = 0, expected_bits = 0;
  std::memcpy(&actual_bits, &actual, sizeof(actual));
  std::memcpy(&expected_bits, &expected, sizeof(expected));
  std::cout << std::hex << actual_bits << ' ' << expected_bits << '\n';
}
'''
        compiler = shutil.which("g++") or shutil.which("c++")
        self.assertIsNotNone(compiler)
        with tempfile.TemporaryDirectory(prefix="adaptive-p-rounding-") as temp:
            root = Path(temp)
            source, executable = root / "rounding.cpp", root / "rounding"
            source.write_text(cpp, encoding="utf-8", newline="\n")
            built = subprocess.run([compiler, "-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror",
                                    "-I", str(ROOT / "native"), str(source), "-o", str(executable)],
                                   capture_output=True, text=True)
            self.assertEqual(built.returncode, 0, built.stderr)
            actual_bits, expected_bits = subprocess.check_output([str(executable)], text=True).split()
        self.assertEqual(actual_bits, expected_bits)

    def test_builder_advertises_versioned_v7_pipeline(self):
        self.assertEqual(builder.PROTOCOL_V7, "full-vocab-adaptive-p-u53-v7")
        self.assertTrue(callable(builder.instrument_min_p_selector_v7))
        build_source = inspect.getsource(builder.build)
        copy_block = build_source.split("copied_header.write_bytes(header_path.read_bytes())", 1)[1].split(
            "compile_argv =", 1)[0]
        self.assertIn('if protocol in ("v4", "v5", "v6", "v7"):', copy_block)
        self.assertIn('if protocol in ("v5", "v6", "v7"):', copy_block)

    def test_v7_feedback_ready_literal_is_valid_json(self):
        prior = '{"mirostat2_feedback_protocol":"mirostat2-state-v1",' \
                '"mirostat2_feedback_version":1}'
        cpp_literal = json.dumps(prior)[1:-1]
        patched = builder.insert_adaptive_p_ready_fields(cpp_literal)
        source = '#include <iostream>\nint main(){std::cout << "' + patched + '";}\n'
        compiler = shutil.which("g++") or shutil.which("c++")
        self.assertIsNotNone(compiler)
        with tempfile.TemporaryDirectory(prefix="adaptive-p-ready-") as temp:
            cpp, exe = Path(temp) / "ready.cpp", Path(temp) / "ready"
            cpp.write_text(source, encoding="utf-8", newline="\n")
            compiled = subprocess.run([compiler, "-std=c++20", "-Wall", "-Wextra", "-Werror",
                                       str(cpp), "-o", str(exe)], text=True, capture_output=True)
            self.assertEqual(compiled.returncode, 0, compiled.stderr)
            output = subprocess.check_output([str(exe)], text=True)
        ready = json.loads(output)
        self.assertEqual(ready["adaptive_p_feedback_protocol"], "adaptive-p-state-v1")
        self.assertEqual(ready["adaptive_p_feedback_version"], 1)
        self.assertEqual(ready["mirostat2_feedback_version"], 1)

    def test_prepared_weight_helper_matches_reference_without_mutation(self):
        compiler = shutil.which("g++") or shutil.which("c++")
        self.assertIsNotNone(compiler, "g++/c++ is required for the CPU header test")
        with tempfile.TemporaryDirectory(prefix="adaptive-p-prepared-") as temp:
            root = Path(temp)
            cpp = root / "probe.cpp"
            exe = root / "probe"
            cpp.write_text(CPP, encoding="utf-8", newline="\n")
            compile_result = subprocess.run([
                compiler, "-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror",
                "-DMIN_P_SELECT_ENABLE_V4=1", "-DMIN_P_SELECT_ENABLE_V5=1",
                "-I", str(ROOT / "native"), str(cpp), "-o", str(exe),
            ], capture_output=True, text=True)
            self.assertEqual(compile_result.returncode, 0, compile_result.stderr)
            rows = subprocess.check_output([str(exe)], text=True).splitlines()
        first = _expected([1.0, 0.0, 0.25, 0.125], [0, 2, 3], (0.9, 1.0),
                          0.55, 0.90, 1 << 52)
        row = rows[0].split()
        self.assertEqual(int(row[0]), first[0])
        for actual, expected in zip(map(float, row[1:]), first[1:6]):
            self.assertAlmostEqual(actual, expected, delta=2e-15)
        second = _expected([0.0, 0.2, 0.8], [1, 2], (0.7, 2.0),
                           -1.0, 0.5, (1 << 53) - 1)
        row = rows[1].split()
        self.assertEqual(int(row[0]), second[0])
        self.assertAlmostEqual(float(row[1]), second[1], delta=2e-15)
        self.assertAlmostEqual(float(row[2]), second[3], delta=2e-15)
        self.assertAlmostEqual(float(row[3]), second[4], delta=2e-15)
        self.assertEqual(row[4], "0")
        row = rows[2].split()
        self.assertEqual((int(row[0]), float(row[1])), (2, 1.0))
        self.assertEqual(float(row[2]), 0.0)
        self.assertEqual(float(row[3]), 1.0)
        self.assertEqual(float(row[4]), 1.0)
        row = rows[3].split()
        self.assertEqual(row, ["0", "256", "1", "0", "1"])

    def test_v7_instrumentation_retains_v6_and_has_one_adaptive_handler(self):
        source = inspect.getsource(builder.instrument_min_p_selector_v7)
        ready_fields = inspect.getsource(builder.insert_adaptive_p_ready_fields)
        self.assertIn("adaptive_p_feedback_protocol", ready_fields)
        self.assertIn("sample_mask_adaptive_p", source)
        self.assertIn("PROTOCOL_V7", source)
        self.assertIn("sampling_v6_protocol", source)
        self.assertIn("adaptive-p-state-v1", source)
        self.assertIn("expected_dry_history_count", source)

    def test_extracted_v7_handler_accepts_disabled_and_exact_dry_tails(self):
        """Compile and run the exact generated handler with a tiny fake session."""
        source = inspect.getsource(builder.instrument_min_p_selector_v7)
        match = re.search(r"command = r'''(.*?)'''", source, re.DOTALL)
        self.assertIsNotNone(match, "v7 command handler must remain extractable for CPU boundary tests")
        compiler = shutil.which("g++") or shutil.which("c++")
        self.assertIsNotNone(compiler)
        native = ROOT / "native"
        wrapper = r'''#define MIN_P_SELECT_ENABLE_V4 1
#define MIN_P_SELECT_ENABLE_V5 1
#include "min_p_select.h"
#include "mirostat2_select.h"
#include "adaptive_p_select.h"
#include <charconv>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>
struct FakeSession {
  std::vector<uint32_t> ids_{0, 1, 2, 3};
  std::vector<float> logits_{0.4f, 0.3f, 0.2f, 0.1f};
  const std::vector<uint32_t>& ids() const { return ids_; }
  const std::vector<float>& logits() const { return logits_; }
};
struct FakeForward { size_t vocab() const { return 4; } };
int main() {
  FakeSession session; FakeForward forward;
  uint64_t sampling_ns=0, sampling_adaptive_p_ns=0, sampling_calls=0;
  uint64_t sampling_adaptive_p_calls=0, sampling_logit_bytes=0, sampling_adaptive_p_logit_bytes=0;
  uint64_t sampling_mask_bytes=0, sampling_adaptive_p_mask_bytes=0;
  std::string line; std::getline(std::cin, line); std::istringstream in(line);
  std::string command; in >> command;
  auto error_reply = [&](const std::string& message) {
    command = "done";
    std::cout << "{\"status\":\"error\",\"message\":\"" << message << "\"}" << std::endl;
  };
  while (true) {
    if (command == "noop") {
''' + match.group(1) + r'''
    }
    break;
  }
}
'''
        with tempfile.TemporaryDirectory(prefix="adaptive-p-handler-") as temp:
            root = Path(temp)
            cpp, exe, mask = root / "handler.cpp", root / "handler", root / "mask.bin"
            empty = root / "empty.bin"
            exact = root / "exact.bin"
            short = root / "short.bin"
            mask.write_bytes(b"\x0f")
            empty.write_bytes(b"")
            # FakeSession exposes a four-token vocabulary; use valid token IDs.
            exact.write_bytes(b"\x01\x00\x00\x00\x02\x00\x00\x00\x03\x00\x00\x00")
            short.write_bytes(b"\x02\x00\x00\x00\x03\x00\x00\x00")
            cpp.write_text(wrapper, encoding="utf-8", newline="\n")
            compiled = subprocess.run([
                compiler, "-std=c++20", "-O2", "-Wall", "-Wextra", "-Werror",
                "-I", str(native), str(cpp), "-o", str(exe),
            ], text=True, capture_output=True)
            self.assertEqual(compiled.returncode, 0, compiled.stderr)

            def invoke(multiplier, last_n, history_path):
                cmd = (f'sample_mask_adaptive_p 4 1.0 0.0 0 1.0 0 "{mask}" '
                       f'{multiplier} 1.2 0 {last_n} "{history_path}" "{empty}" '
                       '1.0 0.0 1.0 0.5 0.9 1.0 2.0')
                result = subprocess.run([str(exe)], input=cmd + "\n", text=True, capture_output=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                return json.loads(result.stdout)

            disabled = invoke(0.0, 256, empty)
            self.assertEqual(disabled["status"], "ok", disabled)
            self.assertEqual(disabled["dry_history_count"], 0)
            self.assertEqual(disabled["dry_penalty_last_n"], 256)
            self.assertEqual(disabled["protocol"], builder.PROTOCOL_V7)
            self.assertEqual(disabled["operator"], "full-vocab-adaptive-p")
            self.assertEqual(disabled["dynatemp_range"], 0.0)
            self.assertEqual(disabled["dynatemp_exponent"], 1.0)
            self.assertGreater(disabled["categorical_total_weight"], 0.0)
            self.assertEqual(disabled["state_total_weight"], 2.0)
            active = invoke(0.8, 3, exact)
            self.assertEqual(active["status"], "ok", active)
            self.assertEqual(active["dry_history_count"], 3)
            truncated = invoke(0.8, 3, short)
            self.assertEqual(truncated["status"], "error")
            self.assertIn("history/breaker file is missing or malformed", truncated["message"])


if __name__ == "__main__":
    unittest.main()
