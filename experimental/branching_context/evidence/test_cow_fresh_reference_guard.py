"""CPU-only fail-closed controls for post-shutdown reference preflight."""
import json
import os
import fcntl
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


HERE = Path(__file__).resolve().parent
REPO_ROOT = Path(os.environ.get("COW_TEST_API", HERE.parents[2]))
HELPER = HERE / "cow_fresh_reference.py"
FAKE_DEAD_PID = 2_000_000_000
GPU_LOCK = Path("/home/dflanag3/branching-g3/native-client-speed.gpu0.lock")


class FreshReferenceShutdownGuardTests(unittest.TestCase):
    def make_run(self, parent: Path, *, server_pid: bool) -> Path:
        run = parent / "run"
        run.mkdir(mode=0o700)
        (run / "server.exit").write_text("0\n", encoding="ascii")
        (run / "native-exits.jsonl").write_text(
            json.dumps({"pid": FAKE_DEAD_PID, "returncode": 0}) + "\n",
            encoding="utf-8",
        )
        if server_pid:
            (run / "server.pid").write_text(f"{FAKE_DEAD_PID}\n", encoding="ascii")
        return run

    def invoke(self, run: Path, parent: Path) -> subprocess.CompletedProcess:
        missing_profile = parent / "profile-does-not-exist.json"
        missing_fixture = parent / "fixture-does-not-exist.json"
        env = os.environ.copy()
        env["PYTHONPATH"] = os.pathsep.join((
            str(REPO_ROOT / "scripts"),
            str(REPO_ROOT / "src"),
            str(REPO_ROOT),
            env.get("PYTHONPATH", ""),
        ))
        return subprocess.run(
            [sys.executable, str(HELPER), "--run", str(run), "--api", str(REPO_ROOT),
             "--profile", str(missing_profile), "--fixture", str(missing_fixture)],
            text=True, capture_output=True, env=env, check=False,
        )

    def test_missing_server_pid_fails_before_reference_or_artifact_creation(self):
        with tempfile.TemporaryDirectory(prefix="cow-ref-no-server-pid-") as temp:
            parent = Path(temp)
            run = self.make_run(parent, server_pid=False)
            result = self.invoke(run, parent)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("recorded HTTP server PID is missing", result.stderr)
            self.assertFalse((run / "fresh-reference").exists())
            self.assertFalse((run / "fresh-reference-comparison.json").exists())

    def test_recorded_dead_server_pid_passes_pid_guard_before_later_preflight(self):
        with tempfile.TemporaryDirectory(prefix="cow-ref-dead-server-pid-") as temp:
            parent = Path(temp)
            run = self.make_run(parent, server_pid=True)
            result = self.invoke(run, parent)
            self.assertNotEqual(result.returncode, 0)
            self.assertNotIn("recorded HTTP server PID is missing", result.stderr)
            self.assertIn("profile-does-not-exist.json", result.stderr)
            self.assertFalse((run / "fresh-reference").exists())
            self.assertFalse((run / "fresh-reference-comparison.json").exists())

    def test_held_gpu_lock_rejects_reference_before_profile_access(self):
        with tempfile.TemporaryDirectory(prefix="cow-ref-gpu-lock-") as temp:
            parent = Path(temp)
            run = self.make_run(parent, server_pid=True)
            GPU_LOCK.parent.mkdir(parents=True, exist_ok=True)
            fd = os.open(GPU_LOCK, os.O_RDWR | os.O_CREAT, 0o600)
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                result = self.invoke(run, parent)
            finally:
                fcntl.flock(fd, fcntl.LOCK_UN)
                os.close(fd)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("GPU0 lock is held or unavailable", result.stderr)
            self.assertNotIn("profile-does-not-exist.json", result.stderr)
            self.assertFalse((run / "fresh-reference").exists())
            self.assertFalse((run / "fresh-reference-comparison.json").exists())


if __name__ == "__main__":
    unittest.main()
