"""CPU-only guards for the prepared browser smoke launcher."""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest


HERE = Path(__file__).resolve().parent
LAUNCHER = HERE / "run-cow-ui-smoke.sh"


class SmokeLauncherTests(unittest.TestCase):
    def _setup(self, root: Path):
        run = root / "run"
        api = root / "api"
        fake_bin = root / "bin"
        run.mkdir()
        api.mkdir()
        fake_bin.mkdir()
        harness = run / "cow_native_smoke.py"
        harness.write_text("# pinned external driver placeholder for launcher test\n")
        marker = root / "python-called"
        fake_python = fake_bin / "python3"
        fake_python.write_text(
            "#!/bin/sh\n"
            "printf '%s\\n' \"$@\" > \"$LAUNCH_CAPTURE\"\n"
            "touch \"$LAUNCH_MARKER\"\n"
            "exit 0\n"
        )
        fake_python.chmod(0o755)
        env = os.environ.copy()
        env.update({
            "RUN": str(run),
            "COW_RUNTIME_API": str(api),
            "COW_FIXTURE": str(root / "fixture.json"),
            "COW_CONTEXT_FILE": str(root / "context.txt"),
            "PATH": str(fake_bin) + os.pathsep + env.get("PATH", ""),
            "LAUNCH_CAPTURE": str(root / "argv.txt"),
            "LAUNCH_MARKER": str(marker),
            "COW_API_KEY": "test-only-not-a-real-key",
        })
        return run, root, marker, env

    def test_existing_output_directory_rejects_before_python_or_model_start(self):
        with tempfile.TemporaryDirectory(prefix="cow-smoke-output-guard-") as temp:
            run, root, marker, env = self._setup(Path(temp))
            (run / "browser").mkdir()
            result = subprocess.run(
                ["bash", str(LAUNCHER)], env=env, text=True,
                capture_output=True, check=False,
            )
            self.assertEqual(result.returncode, 64, result.stderr)
            self.assertIn("output directory must not exist", result.stderr)
            self.assertFalse(marker.exists())
            self.assertFalse((root / "argv.txt").exists())
            self.assertFalse((run / "browser.exit").exists())

    def test_prepared_run_launches_exact_external_driver_when_output_is_absent(self):
        with tempfile.TemporaryDirectory(prefix="cow-smoke-launch-okay-") as temp:
            run, root, marker, env = self._setup(Path(temp))
            result = subprocess.run(
                ["bash", str(LAUNCHER)], env=env, text=True,
                capture_output=True, check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue(marker.exists())
            args = (root / "argv.txt").read_text().splitlines()
            self.assertEqual(args[0], str(run / "cow_native_smoke.py"))
            self.assertIn("--output-dir", args)
            self.assertEqual(args[args.index("--output-dir") + 1], str(run / "browser"))
            self.assertFalse((run / "browser").exists())
            self.assertEqual((run / "browser.exit").read_text().strip(), "0")

    def test_dangling_output_symlink_rejects_before_driver_invocation(self):
        with tempfile.TemporaryDirectory(prefix="cow-smoke-output-link-") as temp:
            run, root, marker, env = self._setup(Path(temp))
            (run / "browser").symlink_to(root / "missing-browser-target")
            result = subprocess.run(
                ["bash", str(LAUNCHER)], env=env, text=True,
                capture_output=True, check=False,
            )
            self.assertEqual(result.returncode, 64, result.stderr)
            self.assertFalse(marker.exists())


if __name__ == "__main__":
    unittest.main()
