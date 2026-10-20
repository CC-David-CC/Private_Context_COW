"""CPU-only HTTP fixture for exercising the real COW browser contract.

This explicitly synthetic fixture reuses the controller and checkpoint
serialization model in ``test_cow_presentation_flow``. It is for browser and
endpoint integration tests, never GPU or model qualification.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import tempfile
import threading
from pathlib import Path

from context_time_travel.coordinator import ContextCoordinator
from context_time_travel.pools import PoolConfig
from serve.cow_adapter import CowAdapter
from serve.cow_http import make_server
from serve.cow_text_adapter import CowTextAdapter
from serve.cow_workspace import CowWorkspace
from serve.test_cow_presentation_flow import CowPresentationFlowTests, FlowModel


class _BrowserPlans(dict):
    """Use caller-random operation IDs with a small deterministic token plan."""

    def __init__(self):
        super().__init__()
        self._requests = 0

    def __missing__(self, operation_id):
        self._requests += 1
        plan = (257,) if self._requests == 1 else (258, 299)
        self[operation_id] = plan
        return plan


class _BrowserFlowModel(FlowModel):
    def __init__(self, directory, gpu, ids, trace, gate):
        super().__init__(directory, gpu, ids, trace, gate)
        self.plans = _BrowserPlans()


class CowBrowserFixtureServer:
    """Own one synthetic coordinator, real HTTP server and optional temp Store."""

    synthetic = True

    def __init__(self, *, store_path=None, host="127.0.0.1", port=0, api_key=None,
                 ram_bytes=64 * 1024**2, disk_bytes=128 * 1024**2):
        if host != "127.0.0.1":
            raise ValueError("Synthetic browser fixture binds only to 127.0.0.1")
        self._temporary_store = tempfile.TemporaryDirectory(prefix="cow-browser-fixture-") \
            if store_path is None else None
        self.store_path = Path(store_path or self._temporary_store.name).resolve()
        self.store_path.mkdir(parents=True, exist_ok=True)
        self._gate = threading.Event()
        self._gate.set()
        self.api_key = api_key
        self.trace = []
        self.models = {}

        def factory(gpu):
            model = _BrowserFlowModel(self.store_path, gpu, (0,), self.trace, self._gate)
            self.models[gpu] = model
            return model

        self.coordinator = ContextCoordinator(str(self.store_path), PoolConfig(
            gpus=(0,), ram_bytes=ram_bytes, disk_bytes=disk_bytes, min_available_bytes=0), factory)
        self.coordinator.ready(5)
        self.text_adapter = CowPresentationFlowTests.make_text(self.coordinator)
        self.workspace = CowWorkspace(self.coordinator.store, self.text_adapter.tokenizer,
            root_mode_resolver=self.text_adapter.root_mode_for_projection)
        self.httpd = make_server(CowAdapter(self.coordinator), host=host, port=port,
                                 api_key=api_key, text_adapter=self.text_adapter,
                                 workspace=self.workspace)
        self.thread = None
        self._closed = False

    @property
    def base_url(self):
        host, port = self.httpd.server_address[:2]
        return f"http://{host}:{port}"

    def ready_record(self):
        return {
            "event": "ready",
            "address": self.base_url,
            "pid": os.getpid(),
            "synthetic": True,
            "api_key_required": bool(self.api_key),
            "store_path": str(self.store_path),
        }

    def start(self):
        if self.thread is not None:
            raise RuntimeError("Fixture server is already started")
        self.thread = threading.Thread(target=self.httpd.serve_forever,
                                       name="cow-browser-fixture-http", daemon=True)
        self.thread.start()
        return self

    def close(self):
        if self._closed:
            return
        self._closed = True
        if self.thread is not None and self.thread.is_alive():
            self.httpd.shutdown()
        self.httpd.server_close()
        if self.thread is not None:
            self.thread.join(timeout=5)
            if self.thread.is_alive():
                raise RuntimeError("Fixture HTTP server did not stop")
        self.coordinator.close()
        if self._temporary_store is not None:
            self._temporary_store.cleanup()

    def __enter__(self):
        return self.start()

    def __exit__(self, exc_type, exc, tb):
        self.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--store", type=Path,
                        help="Persistent Store path; otherwise a private temporary Store is used")
    parser.add_argument("--api-key", default=None,
                        help="Optional API key; default is no key on loopback only")
    args = parser.parse_args(argv)
    if args.port < 0 or args.port > 65535:
        parser.error("--port must be between 0 and 65535")
    server = CowBrowserFixtureServer(store_path=args.store, host=args.host,
                                     port=args.port, api_key=args.api_key).start()
    stopped = threading.Event()

    def request_stop(_signum, _frame):
        stopped.set()

    previous = {}
    for signum in (signal.SIGINT, signal.SIGTERM):
        previous[signum] = signal.signal(signum, request_stop)
    try:
        print(json.dumps(server.ready_record(), separators=(",", ":")), flush=True)
        stopped.wait()
    finally:
        server.close()
        for signum, handler in previous.items():
            signal.signal(signum, handler)


if __name__ == "__main__":
    main()
