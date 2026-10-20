"""CPU checks for the versioned native Mirostat2 selector protocol."""
from __future__ import annotations

import sys
import json
import inspect
import math
import re
import shlex
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import build_min_p_selector_probe as builder


class Mirostat2NativeTests(unittest.TestCase):
    def test_v6_builder_emits_native_feedback_command(self):
        self.assertEqual(builder.PROTOCOL_V6, "full-vocab-mirostat2-u53-v6")
        self.assertTrue(callable(builder.instrument_min_p_selector_v6))

    @classmethod
    def setUpClass(cls):
        cls.compiler = shutil.which("g++")

    def _compile_helper(self, body: str) -> Path:
        if not self.compiler:
            self.skipTest("g++ is unavailable on this CPU test host")
        temp = tempfile.TemporaryDirectory(prefix="mirostat2-native-")
        self.addCleanup(temp.cleanup)
        root = Path(temp.name)
        source = root / "probe.cpp"
        binary = root / "probe"
        source.write_text(body, encoding="utf-8", newline="\n")
        subprocess.run([
            self.compiler, "-std=c++17", "-O2", "-fno-fast-math",
            "-I", str(ROOT / "native"), str(source), "-o", str(binary),
        ], check=True, capture_output=True, text=True)
        return binary

    def test_native_selector_matches_independent_mirostat_reference(self):
        binary = self._compile_helper(r'''#include "mirostat2_select.h"
#include <iomanip>
#include <iostream>
int main() {
  const std::vector<float> logits{3.0f, 2.0f, 1.0f, -2.0f};
  const std::vector<std::uint32_t> support{3, 1, 0};
  for (std::uint64_t draw : {std::uint64_t{0}, std::uint64_t{1} << 52,
                             (std::uint64_t{1} << 53) - 1}) {
    auto r = mirostat2_select::choose(logits, support, 0.8, draw, 1.25, 2.0, 0.2);
    std::cout << std::setprecision(17) << r.token << " " << r.legal_ids.size() << " "
              << r.kept_ids.size() << " " << r.legal_total_weight << " " << r.total_weight << " "
              << r.selected_log_probability << " " << r.selected_probability << " "
              << r.surprise_bits << " " << r.next_mu << "\n";
  }
}''')
        rows = [line.split() for line in subprocess.check_output([str(binary)], text=True).splitlines()]
        logits = [3.0, 2.0, 1.0, -2.0]
        support = [0, 1, 3]
        temperature, mu, tau, eta = 0.8, 1.25, 2.0, 0.2
        maximum = max(logits[i] for i in support)
        shifted = {i: (logits[i] - maximum) / temperature for i in support}
        legal_weight = sum(math.exp(shifted[i]) for i in support)
        ranked = sorted(support, key=lambda i: (-shifted[i], i))
        keep = [i for i in ranked if -(shifted[i] - math.log(legal_weight)) / math.log(2.0) <= mu]
        if not keep:
            keep = [ranked[0]]
        keep.sort()
        weights = {i: math.exp(shifted[i] - max(shifted[j] for j in keep)) for i in keep}
        total = sum(weights.values())
        for row, draw in zip(rows, [0, 1 << 52, (1 << 53) - 1]):
            target = (draw / float(1 << 53)) * total
            cumulative = 0.0
            selected = None
            for i in keep:
                cumulative += weights[i]
                if cumulative > target:
                    selected = i
                    break
            if selected is None:
                selected = keep[-1]
            logp = shifted[selected] - max(shifted[j] for j in keep) - math.log(total)
            probability = math.exp(logp)
            surprise = -logp / math.log(2.0)
            next_mu = mu - eta * (surprise - tau)
            self.assertEqual(int(row[0]), selected)
            self.assertEqual((int(row[1]), int(row[2])), (len(support), len(keep)))
            for actual, expected in zip(map(float, row[3:]),
                                        [legal_weight, total, logp, probability, surprise, next_mu]):
                self.assertAlmostEqual(actual, expected, delta=3e-15)

    def test_draw_zero_keeps_finite_log_probability_when_probability_underflows(self):
        binary = self._compile_helper(r'''#include "mirostat2_select.h"
#include <iostream>
int main() {
  std::vector<float> logits{-745.0f, 0.0f, 0.0f};
  std::vector<std::uint32_t> support{0, 1, 2};
  auto r = mirostat2_select::choose(logits, support, 1.0, 0, 2000.0, 1.0, 0.1);
  std::cout << r.token << " " << r.selected_log_probability << " "
            << r.selected_probability << " " << r.surprise_bits << "\n";
}''')
        token, logp, probability, surprise = subprocess.check_output([str(binary)], text=True).split()
        self.assertEqual(int(token), 0)
        self.assertTrue(math.isfinite(float(logp)))
        self.assertEqual(float(probability), 0.0)
        self.assertGreater(float(surprise), 1000.0)

    def test_invalid_parameters_support_and_nonfinite_masked_logits_fail_closed(self):
        binary = self._compile_helper(r'''#include "mirostat2_select.h"
#include <iostream>
template<class F> bool fails(F f) { try { f(); return false; } catch (...) { return true; } }
int main() {
  bool a = fails([] { mirostat2_select::choose({1.0f}, {0}, 1.0, 0, 0.0, 0.0, 0.0); });
  bool b = fails([] { mirostat2_select::choose({1.0f}, {0, 0}, 1.0, 0, 0.0, 1.0, 0.0); });
  bool c = fails([] { mirostat2_select::choose({1.0f, INFINITY}, {0}, 1.0, 0, 0.0, 1.0, 0.0); });
  bool d = fails([] { mirostat2_select::choose({1.0f}, {0}, 1.0, (std::uint64_t{1} << 53), 0.0, 1.0, 0.0); });
  std::cout << a << b << c << d << "\n";
}''')
        self.assertEqual(subprocess.check_output([str(binary)], text=True).strip(), "1111")

    def test_dry_precedes_mirostat_and_preserves_the_input_frontier(self):
        binary = self._compile_helper(r'''#include "min_p_select.h"
#include "mirostat2_select.h"
#include <iostream>
int main() {
  const std::vector<float> logits{1.0f, 1.0f, 3.0f};
  const std::vector<std::uint32_t> history{1, 2, 1};
  const std::vector<std::uint32_t> breakers{};
  min_p_select::DryConfig config{4.0, 2.0, 1, 8};
  std::size_t penalized = 0;
  const auto adjusted = min_p_select::apply_dry_penalties(
      logits.data(), logits.size(), history, breakers, config, &penalized);
  const auto draw = (std::uint64_t{7} * (std::uint64_t{1} << 53)) / 10;
  const auto result = mirostat2_select::choose(adjusted, {0, 1, 2}, 1.0, draw, 100.0, 1.0, 0.0);
  std::cout << logits[2] << " " << adjusted[2] << " " << penalized << " "
            << result.token << " " << result.kept_ids.size() << "\n";
}''')
        logits, adjusted, penalized, token, kept = map(
            int, subprocess.check_output([str(binary)], text=True).split())
        self.assertEqual(logits, 3)
        self.assertEqual(adjusted, -1)
        self.assertEqual(penalized, 1)
        self.assertEqual(token, 1)
        self.assertEqual(kept, 3)

    def _compile_command_handler(self) -> Path:
        source = inspect.getsource(builder.instrument_min_p_selector_v6)
        matches = re.findall(r"command = r'''(.*?)'''", source, flags=re.DOTALL)
        self.assertEqual(len(matches), 1, "the v6 handler template must have one source of truth")
        block = matches[0]
        opening = block.find('} else if (command == "sample_mask_mirostat2") {')
        ending = block.rfind('} else if (command == "generate") {')
        self.assertGreaterEqual(opening, 0)
        self.assertGreater(ending, opening)
        body_start = block.find("{", opening) + 1
        body = block[body_start:ending]
        wrapper = r'''#include "min_p_select.h"
#include "mirostat2_select.h"
#include <algorithm>
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
  std::vector<std::uint32_t> consumed{0, 1, 0};
  std::vector<float> scores{1.0f, 2.0f, 0.0f, -1.0f};
  const std::vector<std::uint32_t>& ids() const { return consumed; }
  const std::vector<float>& logits() const { return scores; }
} session;
struct FakeForward { std::size_t vocab() const { return 4; } } forward;
std::uint64_t sampling_ns=0, sampling_mirostat2_ns=0, sampling_calls=0;
std::uint64_t sampling_mirostat2_calls=0, sampling_logit_bytes=0, sampling_mirostat2_logit_bytes=0;
std::uint64_t sampling_mask_bytes=0, sampling_mirostat2_mask_bytes=0;
void error_reply(const std::string& text) { std::cerr << "ERROR " << text << "\n"; }
int main() {
  std::string line;
  while (std::getline(std::cin, line)) {
    std::istringstream in(line);
    std::string command, extra;
    in >> command;
    if (command == "sample_mask_mirostat2") {
''' + body + r'''
    } else {
      error_reply("unexpected test command");
    }
  }
  std::cerr << "COUNTERS " << sampling_calls << " " << sampling_mirostat2_calls << "\n";
}'''
        temp = tempfile.TemporaryDirectory(prefix="mirostat2-command-")
        self.addCleanup(temp.cleanup)
        root = Path(temp.name)
        source_path = root / "handler.cpp"
        binary = root / "handler"
        source_path.write_text(wrapper, encoding="utf-8", newline="\n")
        subprocess.run([
            self.compiler, "-std=c++17", "-O2", "-fno-fast-math",
            "-I", str(ROOT / "native"), str(source_path), "-o", str(binary),
        ], check=True, capture_output=True, text=True)
        return binary

    def test_actual_command_accepts_empty_disabled_dry_and_checks_active_tail(self):
        if not self.compiler:
            self.skipTest("g++ is unavailable on this CPU test host")
        binary = self._compile_command_handler()
        with tempfile.TemporaryDirectory(prefix="mirostat2-files-") as temp:
            root = Path(temp)
            mask = root / "mask.bin"
            history = root / "history.bin"
            breakers = root / "breakers.bin"
            mask.write_bytes(b"\x0f")
            breakers.write_bytes(b"")

            def invoke(multiplier: float, last_n: int, history_ids: list[int]):
                history.write_bytes(b"".join(i.to_bytes(4, "little") for i in history_ids))
                command = (
                    f'sample_mask_mirostat2 3 1.0 0 "{mask}" 2.0 0.1 4.0 '
                    f'{multiplier} 1.0 0 {last_n} "{history}" "{breakers}"'
                )
                return subprocess.run([str(binary)], input=command + "\n", text=True,
                                      capture_output=True, check=False)

            disabled = invoke(0.0, 256, [])
            self.assertEqual(disabled.returncode, 0)
            self.assertEqual(disabled.stderr.strip(), "COUNTERS 1 1")
            receipt = json.loads(disabled.stdout.strip())
            self.assertEqual(receipt["status"], "ok")
            self.assertEqual(receipt["dry_history_count"], 0)
            self.assertEqual(receipt["dry_penalty_last_n"], 256)

            truncated = invoke(1.0, 256, [0, 1])
            self.assertEqual(truncated.returncode, 0)
            self.assertEqual(truncated.stdout, "")
            self.assertIn("history/breaker file is missing or malformed", truncated.stderr)
            self.assertTrue(truncated.stderr.strip().endswith("COUNTERS 0 0"))

            active = invoke(1.0, 256, [0, 1, 0])
            self.assertEqual(active.returncode, 0)
            self.assertEqual(active.stderr.strip(), "COUNTERS 1 1")
            receipt = json.loads(active.stdout.strip())
            self.assertEqual(receipt["status"], "ok")
            self.assertEqual(receipt["dry_history_count"], 3)


if __name__ == "__main__":
    unittest.main()
