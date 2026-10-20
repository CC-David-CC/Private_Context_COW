#!/usr/bin/env python3
"""Send SIGINT only to the recorded COW Python service PID after validating argv."""
import os
from pathlib import Path
import signal
import sys

run = Path(sys.argv[1]).resolve(strict=True)
pid_path = run / "server.pid"
pid = int(pid_path.read_text(encoding="ascii").strip())
cmdline = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
expected_script = str(run / "cow_http_keyfile.py")
expected_key_path = str(run / "api-key")
if expected_script not in cmdline or expected_key_path not in cmdline:
    raise SystemExit("refusing: recorded PID argv does not match this run")
os.kill(pid, signal.SIGINT)
print(f"SIGINT sent to verified Python server PID {pid}")
