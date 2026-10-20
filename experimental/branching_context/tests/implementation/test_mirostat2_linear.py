"""Bounded work and independent probability checks for Mirostat retention."""
from __future__ import annotations

import json
import random
import shutil
import struct
import subprocess
import tempfile
import unittest
from pathlib import Path

from context_time_travel.mirostat2 import choose


ROOT = Path(__file__).resolve().parents[2]


class Mirostat2LinearTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        compiler = shutil.which("g++")
        if compiler is None:
            raise RuntimeError("these native CPU checks require g++")
        cls.temp = tempfile.TemporaryDirectory(prefix="mirostat-linear-")
        cls.addClassCleanup(cls.temp.cleanup)
        source = Path(cls.temp.name) / "probe.cpp"
        cls.binary = Path(cls.temp.name) / "probe"
        source.write_text(r'''#include "mirostat2_select.h"
#include <iomanip>
#include <iostream>
int main() {
  size_t n, k;
  double temperature, mu, tau, eta;
  std::uint64_t draw;
  while (std::cin >> n >> k >> temperature >> draw >> mu >> tau >> eta) {
    std::vector<float> logits(n);
    std::vector<std::uint32_t> support(k);
    for (auto& score : logits) std::cin >> score;
    for (auto& id : support) std::cin >> id;
    const auto r = mirostat2_select::choose(logits, support, temperature, draw, mu, tau, eta);
    std::cout << std::setprecision(17) << "{\"token\":" << r.token
      << ",\"visits\":" << r.retention_candidate_visits
      << ",\"legal_total_weight\":" << r.legal_total_weight
      << ",\"total_weight\":" << r.total_weight
      << ",\"selected_log_probability\":" << r.selected_log_probability
      << ",\"selected_probability\":" << r.selected_probability
      << ",\"surprise_bits\":" << r.surprise_bits
      << ",\"next_mu\":" << r.next_mu << ",\"kept_ids\":[";
    bool first = true;
    for (auto id : r.kept_ids) { if (!first) std::cout << ','; first = false; std::cout << id; }
    std::cout << "]}\n";
  }
}''', encoding="utf-8", newline="\n")
        compiled = subprocess.run([compiler, "-std=c++17", "-O2", "-fno-fast-math",
                                   "-I", str(ROOT / "native"), str(source), "-o", str(cls.binary)],
                                  check=False, capture_output=True, text=True)
        if compiled.returncode:
            raise RuntimeError(compiled.stderr)

    def run_case(self, logits, support, temperature, draw, mu, tau=4.0, eta=0.05):
        values = [len(logits), len(support), temperature, draw, mu, tau, eta, *logits, *support]
        output = subprocess.check_output([str(self.binary)], input=" ".join(map(str, values)) + "\n",
                                         text=True)
        return json.loads(output)

    def test_full_vocabulary_retention_visits_each_candidate_once(self):
        logits = [float((i * 37) % 251) / 16 for i in range(248320)]
        for mu in (-3.0, 8.0, 2000.0):
            with self.subTest(mu=mu):
                result = self.run_case(logits, list(range(len(logits))), 0.8, 0, mu)
                self.assertEqual(result["visits"], len(logits))
                if mu < 0:
                    self.assertEqual(result["kept_ids"], [min(
                        range(len(logits)), key=lambda i: (-logits[i], i))])

    def test_seeded_masked_tied_and_extreme_logits_match_independent_reference(self):
        rng = random.Random(937154)
        cases = [([0.0, 0.0, 0.0], [2, 0, 1], 1.0, -1.0),
                 ([-745.0, 0.0, 0.0], [0, 1, 2], 1.0, 2000.0),
                 ([3.0e38, -3.0e38, 0.0], [2, 1, 0], 1.0e-300, 8.0)]
        for n in (1, 4, 31, 257):
            for _ in range(8):
                logits = [struct.unpack("f", struct.pack("f", rng.uniform(-80, 80)))[0]
                          for _ in range(n)]
                support = rng.sample(range(n), rng.randint(1, n))
                cases.append((logits, support, rng.choice((0.1, 0.8, 2.0)),
                              rng.choice((-4.0, 0.0, 2.0, 8.0, 2000.0))))
        for number, (logits, support, temperature, mu) in enumerate(cases):
            logits = [struct.unpack("f", struct.pack("f", x))[0] for x in logits]
            mask = bytearray((len(logits) + 7) // 8)
            for token in support:
                mask[token >> 3] |= 1 << (token & 7)
            for draw in (0, 1 << 52, (1 << 53) - 1):
                with self.subTest(case=number, draw=draw):
                    result = self.run_case(logits, support, temperature, draw, mu)
                    reference = choose(logits, bytes(mask), temperature=temperature,
                                       draw53=draw, mu=mu, tau=4.0, eta=0.05)
                    self.assertEqual(result["visits"], len(support))
                    for key in ("token", "kept_ids", "legal_total_weight", "total_weight",
                                "selected_log_probability", "selected_probability",
                                "surprise_bits", "next_mu"):
                        self.assertEqual(result[key], reference[key], key)


if __name__ == "__main__":
    unittest.main()
