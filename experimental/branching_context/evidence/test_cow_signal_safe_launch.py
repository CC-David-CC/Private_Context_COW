"""Exercise key-file service shutdown with inherited SIGINT ignored."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest


HERE = Path(__file__).resolve().parent
KEYFILE_LAUNCHER = HERE / "cow_http_keyfile.py"
PID_WRAPPER = HERE / "run-cow-http-pid.sh"


class SignalSafeLaunchTests(unittest.TestCase):
    def test_shell_launchers_are_lf_only(self):
        for path in (
            HERE / "run-cow-http-pid.sh",
            HERE / "start-cow-service.sh",
            HERE / "run-cow-ui-smoke.sh",
        ):
            self.assertNotIn(bytes((13, 10)), path.read_bytes(), str(path))

    def test_ignored_parent_sigint_is_restored_and_native_child_closes_cleanly(self):
        if os.name != "posix":
            self.skipTest("POSIX signal lifecycle test")
        with tempfile.TemporaryDirectory(prefix="cow-sigint-lifecycle-") as temp:
            root = Path(temp)
            run = root / "run"
            api = root / "api"
            package = api / "src" / "context_time_travel"
            serve = api / "serve"
            run.mkdir()
            package.mkdir(parents=True)
            serve.mkdir()
            (package / "__init__.py").write_text("")
            (serve / "__init__.py").write_text("")
            (package / "strata_backend.py").write_text(
                "class StrataBackend:\n"
                "    def __init__(self): self.process = None\n"
                "    def close(self):\n"
                "        self.process.stdin.write('quit\\n')\n"
                "        self.process.stdin.flush()\n"
                "        self.process.wait(timeout=5)\n"
            )
            (serve / "cow_http.py").write_text(
                "import os, subprocess, sys, time\n"
                "from pathlib import Path\n"
                "from context_time_travel.strata_backend import StrataBackend\n"
                "def main():\n"
                "    b = StrataBackend()\n"
                "    b.process = subprocess.Popen([sys.executable, '-c', "
                "'import sys; sys.stdin.readline()'], stdin=subprocess.PIPE, text=True)\n"
                "    Path(os.environ['READY_FILE']).write_text(str(os.getpid()))\n"
                "    try:\n"
                "        while True: time.sleep(0.05)\n"
                "    finally: b.close()\n"
            )
            key = run / "api-key"
            key.write_text("test-key-only\n")
            key.chmod(0o600)
            ready = root / "ready"
            receipt = run / "native-exits.jsonl"
            env = os.environ.copy()
            env.update({
                "RUN": str(run),
                "COW_RUNTIME_API": str(api),
                "COW_PROFILE": str(root / "unused-profile.json"),
                "COW_PYTHON": sys.executable,
                "READY_FILE": str(ready),
            })
            (run / "cow_http_keyfile.py").write_bytes(KEYFILE_LAUNCHER.read_bytes())

            def ignore_sigint_in_child():
                signal.signal(signal.SIGINT, signal.SIG_IGN)

            child = subprocess.Popen(
                ["bash", str(PID_WRAPPER)],
                env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, preexec_fn=ignore_sigint_in_child,
            )
            try:
                server_pid_file = run / "server.pid"
                deadline = time.monotonic() + 8
                while (not ready.exists() or not server_pid_file.exists()) and time.monotonic() < deadline:
                    if child.poll() is not None:
                        stdout, stderr = child.communicate()
                        self.fail(f"launcher exited before ready: {child.returncode} {stdout} {stderr}")
                    time.sleep(0.02)
                self.assertTrue(ready.exists(), "service fixture did not become ready")
                self.assertTrue(server_pid_file.is_file(), "PID wrapper omitted server.pid")
                service_pid = int(server_pid_file.read_text())
                self.assertEqual(int(ready.read_text()), service_pid)
                os.kill(service_pid, signal.SIGINT)
                stdout, stderr = child.communicate(timeout=8)
                self.assertEqual(child.returncode, 0, stderr or stdout)
                rows = [json.loads(line) for line in receipt.read_text().splitlines()]
                self.assertEqual(len(rows), 1)
                self.assertIsInstance(rows[0]["pid"], int)
                self.assertEqual(rows[0]["returncode"], 0)
            finally:
                if child.poll() is None:
                    child.kill()
                    child.wait(timeout=5)


if __name__ == "__main__":
    unittest.main()
