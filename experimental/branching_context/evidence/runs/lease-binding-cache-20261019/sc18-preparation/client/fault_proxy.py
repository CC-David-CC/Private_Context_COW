"""Local qualification proxy: forward exact bytes, deliberately withhold one ACK.

No model or response is mocked. The upstream really executes and commits. This
test driver drops a selected completion only after observing its actual receipt.
Authorization is forwarded but never recorded. Do not deploy as a serving proxy.
"""
import hashlib
import http.client
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import threading
from urllib.parse import urlsplit


class LostAckProxy:
    def __init__(self, upstream_url, operation_id):
        origin = urlsplit(upstream_url)
        if (origin.scheme != 'http' or origin.hostname != '127.0.0.1'
                or not origin.port or origin.path or origin.query or origin.fragment or origin.username):
            raise ValueError('qualification proxy needs an explicit loopback upstream origin')
        self.origin = origin
        # None selects the first actual workflow model operation, not setup.
        self.operation_id = operation_id
        self.selected_automatically = False
        self.requests = []
        self.dropped_completion = None
        self.dropped = threading.Event()
        self.lock = threading.Lock()
        owner = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = 'HTTP/1.0'

            def log_message(self, *_args):
                pass

            def do_GET(self):
                self.forward('GET')

            def do_POST(self):
                self.forward('POST')

            def forward(self, method):
                length = int(self.headers.get('Content-Length', '0'))
                if not 0 <= length <= 16 * 1024**2:
                    self.send_error(413)
                    return
                raw = self.rfile.read(length) if length else None
                op = None
                if self.headers.get('Content-Type', '').startswith('application/json') and raw:
                    body = json.loads(raw)
                    op = body.get('operation_id')
                record = dict(method=method, path=self.path, operation_id=op,
                    body_bytes=0 if raw is None else len(raw),
                    body_sha256=None if raw is None else hashlib.sha256(raw).hexdigest())
                with owner.lock:
                    if (owner.operation_id is None and method == 'POST'
                            and self.path in {'/cow/text/answer', '/cow/text/ask',
                                              '/cow/text/append-user', '/cow/text/resume'}
                            and isinstance(op, str) and op.startswith('bwnative-')):
                        owner.operation_id = op
                        owner.selected_automatically = True
                    owner.requests.append(record)
                connection = http.client.HTTPConnection(owner.origin.hostname, owner.origin.port, timeout=180)
                try:
                    headers = {name: value for name, value in self.headers.items()
                        if name.lower() in ('authorization', 'content-type', 'accept', 'x-cow-content-sha256')}
                    connection.request(method, self.path, raw, headers)
                    response = connection.getresponse()
                    record['status'] = response.status
                    kind = response.getheader('Content-Type', 'application/json')
                    self.send_response(response.status)
                    self.send_header('Content-Type', kind)
                    self.send_header('Connection', 'close')
                    self.end_headers()
                    if kind.startswith('application/x-ndjson'):
                        for line in response:
                            event = json.loads(line)
                            if op == owner.operation_id and event.get('event') == 'complete':
                                with owner.lock:
                                    if owner.dropped_completion is None:
                                        owner.dropped_completion = event
                                        record['completion_forwarded'] = False
                                        owner.dropped.set()
                                        return
                            self.wfile.write(line)
                            self.wfile.flush()
                    else:
                        self.wfile.write(response.read())
                    self.wfile.flush()
                finally:
                    self.close_connection = True
                    connection.close()

        self.server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.url = f'http://127.0.0.1:{self.server.server_port}'
        self.thread.start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(5)
        if self.thread.is_alive():
            raise RuntimeError('qualification proxy failed to stop')
