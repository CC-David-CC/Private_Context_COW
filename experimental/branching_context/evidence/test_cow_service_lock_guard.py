"""CPU-only lock rejection control for the service launcher."""
import fcntl
import os
from pathlib import Path
import subprocess
import tempfile
import unittest


HERE = Path(__file__).resolve().parent
START = HERE / "start-cow-service.sh"
GPU_LOCK = Path("/home/dflanag3/branching-g3/native-client-speed.gpu0.lock")


class ServiceLockGuardTests(unittest.TestCase):
    def test_held_lock_stops_before_service_launch_marker(self):
        GPU_LOCK.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(GPU_LOCK, os.O_RDWR | os.O_CREAT, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with tempfile.TemporaryDirectory(prefix="cow-service-lock-") as temp:
                marker = Path(temp) / "launched"
                result = subprocess.run(
                    ["bash", str(START), "--test-launch-marker", str(marker)],
                    env={**os.environ, "RUN": temp},
                    text=True, capture_output=True, check=False,
                )
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("GPU0 lock is held", result.stderr)
                self.assertFalse(marker.exists())
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)

    def test_existing_browser_output_rejects_before_service_marker(self):
        with tempfile.TemporaryDirectory(prefix="cow-service-output-") as temp:
            run = Path(temp)
            (run / "browser").mkdir()
            marker = run / "launched"
            result = subprocess.run(
                ["bash", str(START), "--test-launch-marker", str(marker)],
                env={**os.environ, "RUN": str(run)},
                text=True, capture_output=True, check=False,
            )
            self.assertEqual(result.returncode, 64)
            self.assertIn("output directory must not exist before service launch", result.stderr)
            self.assertFalse(marker.exists())

    def test_dangling_browser_output_symlink_rejects_before_service_marker(self):
        with tempfile.TemporaryDirectory(prefix="cow-service-output-link-") as temp:
            run = Path(temp)
            (run / "browser").symlink_to(run / "not-created")
            marker = run / "launched"
            result = subprocess.run(
                ["bash", str(START), "--test-launch-marker", str(marker)],
                env={**os.environ, "RUN": str(run)},
                text=True, capture_output=True, check=False,
            )
            self.assertEqual(result.returncode, 64)
            self.assertIn("output directory must not exist before service launch", result.stderr)
            self.assertFalse(marker.exists())


if __name__ == "__main__":
    unittest.main()
