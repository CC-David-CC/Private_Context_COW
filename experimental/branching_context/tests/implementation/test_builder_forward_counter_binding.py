"""Generated native request protocol must report cumulative forward work."""
import os
from pathlib import Path
import shutil
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import build_fast_prefill_probe


def source_root():
    return Path(os.environ["COW_BUILDER_SOURCE_ROOT"]).resolve(strict=True)


class BuilderForwardCounterBindingTests(unittest.TestCase):
    def test_completion_and_stats_use_same_cumulative_forward_counter(self):
        root = source_root()
        with TemporaryDirectory() as temp:
            temp = Path(temp)
            fixture_root = temp / "source"
            for relative in (
                "include/strata/core/branch_driver.hpp",
                "src/program/generate.cpp",
                "src/core/branch_driver.cpp",
            ):
                target = fixture_root / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(root / relative, target)
            build_dir = fixture_root / "build"
            recipe = build_dir / "CMakeFiles" / "strata-branch.dir"
            recipe.mkdir(parents=True)
            (build_dir / "CMakeCache.txt").write_text(
                "CMAKE_CXX_COMPILER:FILEPATH=/usr/bin/c++\n"
                "CMAKE_BUILD_TYPE:STRING=Release\n", encoding="utf-8")
            (recipe / "flags.make").write_text(
                "CXX_DEFINES = -DSTRATA_BRANCH_DRIVER=1\n"
                "CXX_INCLUDES = -Iinclude\n"
                "CXX_FLAGS = -O3 -std=c++20\n", encoding="utf-8")
            (recipe / "link.txt").write_text(
                "c++ CMakeFiles/strata-branch.dir/src/program/generate.cpp.o "
                "CMakeFiles/strata-branch.dir/src/core/branch_driver.cpp.o "
                "-o strata-branch libstrata_engine.a\n", encoding="utf-8")
            output = temp / "generated"
            successful_compile = SimpleNamespace(returncode=0)
            with patch.object(build_fast_prefill_probe.subprocess, "run",
                              return_value=successful_compile):
                build_fast_prefill_probe.build(
                    fixture_root, output, long_demo=True, restore_counters=False)

            generated = (output / "branch_driver.cpp").read_text(encoding="utf-8")
            cumulative = (
                '<< ",\\"forward_tokens\\":" '
                '<< (forward.completed_forward_tokens + prefill_tokens)')
            self.assertEqual(generated.count(cumulative), 2)
            self.assertNotIn(
                '<< ",\\"forward_tokens\\":" '
                '<< (forward.completed_forward_tokens)', generated)


if __name__ == "__main__":
    unittest.main()
