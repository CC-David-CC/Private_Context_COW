"""CPU-only recipe tests for the isolated ordinary-serving baseline builder."""
import json
import subprocess
import tempfile
import unittest
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import patch

import build_normal_serving_baseline as baseline


class NormalRecipeTests(unittest.TestCase):
    def fixture(self, branch=False):
        temp = tempfile.TemporaryDirectory()
        root = Path(temp.name)
        build = root / "build"
        target = "strata-branch" if branch else "strata"
        target_dir = build / "CMakeFiles" / f"{target}.dir"
        target_dir.mkdir(parents=True)
        (build / "CMakeCache.txt").write_text("CMAKE_BUILD_TYPE:STRING=Release\n")
        (target_dir / "flags.make").write_text(
            "CXX_DEFINES = -DSTRATA_NATIVE_EXPERTS=1 -DSTRATA_PREFILL_FUSED=1\n"
            "CXX_INCLUDES = -I" + str(root / "include") + "\n"
            "CXX_FLAGS = -O3 -DNDEBUG -std=c++20 -Wall -Wextra\n"
        )
        gen = f"CMakeFiles/{target}.dir/src/program/generate.cpp.o"
        objects = [gen]
        if not branch:
            objects.append("CMakeFiles/strata.dir/src/core/branch_pages.cpp.o")
        if branch:
            objects.append(f"CMakeFiles/{target}.dir/src/core/branch_driver.cpp.o")
        archive = str(root / "lib/libstrata_core.a")
        (target_dir / "link.txt").write_text(
            "/usr/bin/c++ " + " ".join(objects) + f" -o {target} {archive} -lcudart"
        )
        for rel in objects:
            path = build / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"old object")
        (root / "lib").mkdir()
        (root / "lib/libstrata_core.a").write_bytes(b"archive")
        return temp, root, build

    def candidate_fixture(self, root):
        candidate = root / "candidate"
        source_root = candidate / "source"
        (source_root / "src/program").mkdir(parents=True)
        (source_root / "include").mkdir()
        generate = source_root / "src/program/generate.cpp"
        generate.write_text("int main() { return 0; }\n")
        (candidate / "native-probe-v3").mkdir()
        sources = {str(generate): baseline.sha256(generate)}
        objects = {}
        commands = []
        for relative, source in baseline.CORE_OBJECTS:
            object_path = candidate / relative
            object_path.parent.mkdir(parents=True, exist_ok=True)
            object_path.write_bytes(b"compiled current object")
            source_path = source_root / source
            source_path.parent.mkdir(parents=True, exist_ok=True)
            source_path.write_text("// pinned input\n")
            sources[str(source_path)] = baseline.sha256(source_path)
            objects[str(object_path)] = baseline.sha256(object_path)
            commands.append({"argv": ["c++", "-c", str(source_path), "-o", str(object_path)],
                             "exit_code": 0})
        receipt = {"status": "BUILT_NOT_EXECUTED", "source_sha256": sources,
                   "generated_sha256": objects, "commands": commands}
        (candidate / "native-probe-v3/build.json").write_text(json.dumps(receipt) + "\n")
        return candidate, source_root

    def test_accepts_standard_strata_recipe_and_preserves_full_link_closure(self):
        temp, root, build = self.fixture()
        self.addCleanup(temp.cleanup)
        recipe = baseline.load_standard_recipe(build)
        self.assertEqual(recipe.target, "strata")
        self.assertIn("CXX_FLAGS", recipe.flags)
        self.assertIn("-lcudart", recipe.link_argv)
        self.assertEqual(recipe.generate_object, "CMakeFiles/strata.dir/src/program/generate.cpp.o")

    def test_rejects_private_branch_target_and_driver_definition(self):
        temp, root, build = self.fixture()
        self.addCleanup(temp.cleanup)
        link = build / "CMakeFiles/strata.dir/link.txt"
        link.write_text(link.read_text().replace("-o strata ", "-o strata-branch "))
        with self.assertRaisesRegex(ValueError, "ordinary strata target output"):
            baseline.load_standard_recipe(build)

        temp2, root2, build2 = self.fixture()
        self.addCleanup(temp2.cleanup)
        flags = build2 / "CMakeFiles/strata.dir/flags.make"
        flags.write_text(flags.read_text().replace("STRATA_NATIVE_EXPERTS=1", "STRATA_BRANCH_DRIVER=1"))
        with self.assertRaisesRegex(ValueError, "STRATA_BRANCH_DRIVER"):
            baseline.load_standard_recipe(build2)

        temp3, root3, build3 = self.fixture()
        self.addCleanup(temp3.cleanup)
        link3 = build3 / "CMakeFiles/strata.dir/link.txt"
        link3.write_text(link3.read_text().replace(" -o strata ", " branch_driver.cpp.o -o strata "))
        with self.assertRaisesRegex(ValueError, "branch_driver"):
            baseline.load_standard_recipe(build3)

    def test_rejects_missing_standard_generate_object_and_missing_archive(self):
        temp, root, build = self.fixture()
        self.addCleanup(temp.cleanup)
        (build / "CMakeFiles/strata.dir/src/program/generate.cpp.o").unlink()
        with self.assertRaisesRegex(FileNotFoundError, "generate.cpp.o"):
            baseline.load_standard_recipe(build)

        temp2, root2, build2 = self.fixture()
        self.addCleanup(temp2.cleanup)
        (root2 / "lib/libstrata_core.a").unlink()
        with self.assertRaisesRegex(FileNotFoundError, "libstrata_core.a"):
            baseline.load_standard_recipe(build2)

    def test_plan_substitutes_only_new_generate_and_output_and_adds_current_overrides(self):
        temp, root, build = self.fixture()
        self.addCleanup(temp.cleanup)
        recipe = baseline.load_standard_recipe(build)
        overrides = []
        for index, (relative, _source_relative) in enumerate(baseline.CORE_OBJECTS):
            override_path = root / f"current-{index}-{Path(relative).name}"
            override_path.write_bytes(b"current override")
            overrides.append(override_path)
        source_root = root / "source"
        (source_root / "src/program").mkdir(parents=True)
        (source_root / "include").mkdir()
        source = source_root / "src/program/generate.cpp"
        source.write_text("int main() { return 0; }\n")
        generated = root / "external" / "generate.o"
        output = root / "external" / "strata-normal-baseline"
        compile_argv, link_argv = baseline.make_commands(
            recipe, source_root, source, generated, output, overrides
        )
        self.assertEqual(compile_argv[-1], str(generated))
        self.assertIn(str(source), compile_argv)
        self.assertEqual(link_argv[link_argv.index("-o") + 1], str(output))
        self.assertIn(str(overrides[1]), link_argv)
        self.assertNotIn("CMakeFiles/strata.dir/src/core/branch_pages.cpp.o", link_argv)
        self.assertNotIn("branch_driver.cpp.o", " ".join(link_argv))
        self.assertNotIn("-DSTRATA_BRANCH_DRIVER=1", compile_argv)
        self.assertIn("-I" + str(source_root / "include"), compile_argv)
        self.assertEqual((build / "CMakeFiles/strata.dir/src/program/generate.cpp.o").read_bytes(),
                         b"old object")
        with self.assertRaisesRegex(ValueError, "exactly 9"):
            baseline.make_commands(recipe, source_root, source, generated, output, overrides[:-1])

    def test_rejects_existing_output_before_any_command(self):
        temp, root, build = self.fixture()
        self.addCleanup(temp.cleanup)
        recipe = baseline.load_standard_recipe(build)
        output = root / "already-there"
        output.mkdir()
        with self.assertRaisesRegex(FileExistsError, "output directory"):
            baseline.prepare_output(output)

    def test_plan_only_is_read_only_and_rejects_existing_output(self):
        temp, root, build = self.fixture()
        self.addCleanup(temp.cleanup)
        candidate, source_root = self.candidate_fixture(root)
        output = root / "planned-output"
        self.assertFalse(output.exists())
        plan = baseline.plan_build(source_root, build, candidate, output)
        self.assertFalse(output.exists())
        self.assertEqual(plan["builder_sha256"], baseline.sha256(baseline.__file__))
        self.assertIn("-o", plan["link_argv"])
        self.assertNotIn("branch_driver.cpp.o", " ".join(plan["link_argv"]))
        self.assertIn(str(source_root / "src/core/branch_pages.cpp"), plan["inputs_sha256"])
        (source_root / "src/core/branch_pages.cpp").write_text("// post-receipt mutation\n")
        with self.assertRaisesRegex(ValueError, "COW receipt hash mismatch"):
            baseline.plan_build(source_root, build, candidate, output)
        self.assertFalse(output.exists())

    def test_mocked_build_records_executable_hash_and_stops_after_compile_failure(self):
        temp, root, build = self.fixture()
        self.addCleanup(temp.cleanup)
        candidate, source_root = self.candidate_fixture(root)
        output = root / "successful-output"

        def successful_run(argv, cwd, text, capture_output):
            output_path = Path(argv[argv.index("-o") + 1])
            output_path.write_bytes(b"mock compiled artifact")
            return SimpleNamespace(returncode=0, stdout="ok\n", stderr="")

        with patch.object(baseline.subprocess, "run", side_effect=successful_run) as runner:
            result = baseline.build(source_root, build, candidate, output)
        self.assertEqual(runner.call_count, 2)
        self.assertEqual(result["status"], "BUILT_UNQUALIFIED")
        self.assertEqual(result["builder_sha256"], baseline.sha256(baseline.__file__))
        self.assertEqual(result["serving_scope"], "ordinary-serving-on-current-patched-engine")
        self.assertEqual(result["executable_sha256"],
                         baseline.sha256(output / "strata-normal-baseline"))
        self.assertEqual([row["exit_code"] for row in result["commands"]], [0, 0])

        temp2, root2, build2 = self.fixture()
        self.addCleanup(temp2.cleanup)
        candidate2, source_root2 = self.candidate_fixture(root2)
        output2 = root2 / "failed-output"

        def failed_run(argv, cwd, text, capture_output):
            return SimpleNamespace(returncode=9, stdout="", stderr="compile fixture failure")

        with patch.object(baseline.subprocess, "run", side_effect=failed_run) as runner2:
            with self.assertRaisesRegex(subprocess.CalledProcessError, "returned non-zero exit status 9"):
                baseline.build(source_root2, build2, candidate2, output2)
        self.assertEqual(runner2.call_count, 1)
        failed = json.loads((output2 / "build.json").read_text())
        self.assertEqual(failed["status"], "BUILD_FAILED")
        self.assertEqual([row["exit_code"] for row in failed["commands"]], [9])
        self.assertFalse((output2 / "strata-normal-baseline").exists())


if __name__ == "__main__":
    unittest.main()
