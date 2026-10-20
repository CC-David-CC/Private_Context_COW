#!/usr/bin/env python3
"""Read a private key file and run the existing COW HTTP service."""
import json
import os
from pathlib import Path
import signal
import stat
import sys


def main() -> None:
    # Background shells commonly start jobs with SIGINT ignored. Restore
    # Python's normal interrupt behavior before importing the service so an
    # exact-PID SIGINT runs the service's finally/close path.
    signal.signal(signal.SIGINT, signal.default_int_handler)
    if len(sys.argv) < 2:
        raise SystemExit("usage: cow_http_keyfile.py KEY_FILE [cow_http.py options...]")
    key_path = Path(sys.argv[1])
    arguments = sys.argv[2:]
    info = key_path.stat()
    if not stat.S_ISREG(info.st_mode) or stat.S_IMODE(info.st_mode) != 0o600:
        raise SystemExit("key file must be a regular mode-0600 file")
    key = key_path.read_text(encoding="utf-8").strip()
    if not key:
        raise SystemExit("empty key file")

    sys.argv = ["cow_http.py", *arguments, "--api-key", key]
    from context_time_travel.strata_backend import StrataBackend

    original_close = StrataBackend.close

    def close_with_exit_receipt(self):
        pid = self.process.pid
        try:
            return original_close(self)
        finally:
            record_path = Path(os.environ["NATIVE_EXIT_RECEIPT"])
            with record_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps({"pid": pid,
                                         "returncode": self.process.returncode}) + "\n")
                stream.flush()
                os.fsync(stream.fileno())

    StrataBackend.close = close_with_exit_receipt
    from serve.cow_http import main as http_main

    try:
        http_main()
    except KeyboardInterrupt:
        # The HTTP main function's finally block has run before this returns.
        pass


if __name__ == "__main__":
    main()
