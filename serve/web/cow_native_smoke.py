#!/usr/bin/env python3
"""Drive the real COW browser UI over HTTP using Chromium's DevTools protocol.

The script attaches to an already-running Context COW server; it never launches
model code. Set COW_API_KEY in the environment when the server requires auth.
Browser profile data is created under a private temporary directory and
removed on exit. Its storage medium depends on --tmp-root; the default suits
the installed Chromium Snap profile policy and does not claim RAM-only storage.
Screenshots and a browser performance trace are written to a private,
caller-provided output directory. Answers are checked for canonical
receipt/history agreement, never against a hard-coded generated string.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import importlib.util
import json
import os
import shutil
import socket
import struct
import subprocess
import tempfile
import time
import urllib.request
import uuid
from pathlib import Path
from urllib.parse import urlsplit


class DevTools:
    """Small synchronous CDP client using only the Python standard library."""

    def __init__(self, websocket_url: str):
        parsed = urlsplit(websocket_url)
        if parsed.scheme != "ws" or parsed.hostname not in ("127.0.0.1", "localhost"):
            raise RuntimeError("Chromium DevTools must bind to loopback")
        self.sock = socket.create_connection((parsed.hostname, parsed.port), timeout=10)
        self.sock.settimeout(30)
        key = base64.b64encode(os.urandom(16)).decode("ascii")
        request = (
            f"GET {parsed.path} HTTP/1.1\r\nHost: {parsed.netloc}\r\n"
            "Upgrade: websocket\r\nConnection: Upgrade\r\n"
            f"Sec-WebSocket-Key: {key}\r\nSec-WebSocket-Version: 13\r\n"
            "Origin: http://localhost\r\n\r\n"
        ).encode("ascii")
        self.sock.sendall(request)
        response = self._read_http_headers()
        if not response.startswith(b"HTTP/1.1 101"):
            raise RuntimeError("Chromium DevTools WebSocket upgrade failed")
        self._next_id = 1
        self.events: list[dict] = []

    def _read_http_headers(self) -> bytes:
        data = bytearray()
        while not data.endswith(b"\r\n\r\n"):
            chunk = self.sock.recv(1)
            if not chunk:
                raise RuntimeError("DevTools closed during WebSocket handshake")
            data.extend(chunk)
        return bytes(data)

    def _exact(self, size: int) -> bytes:
        data = bytearray()
        while len(data) < size:
            chunk = self.sock.recv(size - len(data))
            if not chunk:
                raise RuntimeError("DevTools WebSocket closed")
            data.extend(chunk)
        return bytes(data)

    def _send(self, value: dict) -> None:
        payload = json.dumps(value, separators=(",", ":")).encode("utf-8")
        mask = os.urandom(4)
        size = len(payload)
        if size < 126:
            header = bytes((0x81, 0x80 | size))
        elif size <= 0xFFFF:
            header = bytes((0x81, 0x80 | 126)) + struct.pack("!H", size)
        else:
            header = bytes((0x81, 0x80 | 127)) + struct.pack("!Q", size)
        masked = bytes(byte ^ mask[i % 4] for i, byte in enumerate(payload))
        self.sock.sendall(header + mask + masked)

    def _receive(self) -> dict:
        first, second = self._exact(2)
        opcode, length = first & 0x0F, second & 0x7F
        if length == 126:
            length = struct.unpack("!H", self._exact(2))[0]
        elif length == 127:
            length = struct.unpack("!Q", self._exact(8))[0]
        mask = self._exact(4) if second & 0x80 else None
        payload = self._exact(length)
        if mask:
            payload = bytes(byte ^ mask[i % 4] for i, byte in enumerate(payload))
        if opcode == 8:
            raise RuntimeError("Chromium DevTools WebSocket closed")
        if opcode == 9:
            self._send_raw(bytes((0x8A, 0x00)))
            return self._receive()
        if opcode not in (1,):
            return self._receive()
        return json.loads(payload)

    def _send_raw(self, payload: bytes) -> None:
        # Only used for empty pong frames.
        mask = os.urandom(4)
        self.sock.sendall(bytes((0x8A, 0x80)) + mask)

    def call(self, method: str, params: dict | None = None) -> dict:
        command_id = self._next_id
        self._next_id += 1
        self._send({"id": command_id, "method": method, "params": params or {}})
        while True:
            message = self._receive()
            if "id" not in message:
                self.events.append(message)
                continue
            if message["id"] != command_id:
                raise RuntimeError("Unexpected DevTools response ordering")
            if "error" in message:
                raise RuntimeError(f"DevTools command failed: {method}")
            return message.get("result", {})

    def evaluate(self, expression: str):
        result = self.call("Runtime.evaluate", {
            "expression": expression,
            "awaitPromise": True,
            "returnByValue": True,
            "userGesture": True,
        })
        if "exceptionDetails" in result:
            details = result.get('exceptionDetails', {})
            exception = details.get('exception', {})
            description = exception.get('description') or details.get('text') or 'unknown page exception'
            raise RuntimeError('Browser smoke assertion raised in page: ' + str(description)[:1000])
        return result.get("result", {}).get("value")

    def wait_for(self, expression: str, label: str, timeout: float = 90.0):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.evaluate(expression):
                return
            time.sleep(0.2)
        raise RuntimeError(f"Timed out waiting for {label}")

    def screenshot(self, path: Path) -> None:
        result = self.call("Page.captureScreenshot", {
            "format": "png", "captureBeyondViewport": False,
        })
        path.write_bytes(base64.b64decode(result["data"]))

    def close(self) -> None:
        try:
            self.sock.close()
        except OSError:
            pass


class SmokeBudgetLimit(RuntimeError):
    def __init__(self, stage: str, rounds: int):
        super().__init__(f"{stage} answer remained active at the configured smoke budget")
        self.stage = stage
        self.rounds = rounds

def get_json(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=5) as response:
        return json.load(response)


def check_graph(devtools, output, first, forked):
    """Exercise real graph requests, selection, inspection and previous tips."""
    if devtools.evaluate("document.getElementById('chat').textContent.includes('Invalid Date')"):
        raise RuntimeError('Canonical messages without timestamps displayed Invalid Date')
    devtools.wait_for("document.getElementById('pill-text').textContent==='Ready'", 'authenticated COW readiness indicator', timeout=10)
    if devtools.evaluate("performance.getEntriesByType('resource').some(r=>['/metrics','/mcp'].includes(new URL(r.name).pathname))"):
        raise RuntimeError('Native UI polled ordinary metrics or MCP routes')
    devtools.evaluate("ContextGraph.refresh()")
    state = devtools.evaluate("({stale:ContextGraph.stale,graph:ContextGraph.data,"
        "badges:document.querySelectorAll('[data-graph-branch]').length})")
    graph = state['graph']
    if state['stale'] or state['badges'] != 2 or len(graph['branches']) != 2:
        raise RuntimeError('Graph did not display both canonical branches')
    branches = {branch['id']: branch for branch in graph['branches']}
    if (branches[first['branch']]['saved_messages'] != first['count']
            or branches[forked['branch']]['saved_messages'] != forked['count']
            or graph['shared_prefix_messages'] != first['count']):
        raise RuntimeError('Graph counts/shared core differ from canonical histories')
    nodes = {node['id']: node for node in graph['nodes']}
    for branch in branches.values():
        if (branch['tip'] == branch['snapshot']
                or nodes[branch['tip']]['action_snapshot'] != branch['snapshot']):
            raise RuntimeError('Graph tip and canonical execution boundary were confused')
    devtools.screenshot(output / '04-shared-branches.png')
    source = json.dumps(first['branch']); fork = json.dumps(forked['branch'])
    devtools.evaluate("document.getElementById('input').value='unsent source draft';"
        "[...document.querySelectorAll('[data-graph-branch]')].find(b=>b.dataset.graphBranch==="+fork+").click()")
    devtools.wait_for("ContextWorkspace.current?.id==="+fork+"&&!ContextGraph.stale", 'graph fork selection')
    devtools.evaluate("document.getElementById('input').value='unsent fork draft';"
        "[...document.querySelectorAll('[data-graph-branch]')].find(b=>b.dataset.graphBranch==="+source+").click()")
    devtools.wait_for("ContextWorkspace.current?.id==="+source+"&&!ContextGraph.stale", 'graph source selection')
    if devtools.evaluate("document.getElementById('input').value") != 'unsent source draft':
        raise RuntimeError('Graph branch switch lost the source draft')
    devtools.evaluate("(()=>{const n=ContextGraph.data.nodes.find(n=>n.role==='user'&&!n.boundary_available);"
        "if(!n)throw Error('Fixture lacks a message without a snapshot');"
        "ContextGraph.inspect([n.id],ContextGraph.model(ContextGraph.data));})()")
    if devtools.evaluate("!!document.querySelector('#graph-inspector .workspace-message-actions button')"):
        raise RuntimeError('Graph invented actions for a message without a checkpoint')
    devtools.evaluate("[...document.querySelectorAll('[data-graph-branch]')].find(b=>b.dataset.graphBranch==="+fork+").click()")
    devtools.wait_for("ContextWorkspace.current?.id==="+fork+"&&!ContextGraph.stale", 'graph fork reselection')
    if devtools.evaluate("document.getElementById('input').value") != 'unsent fork draft':
        raise RuntimeError('Graph branch switch lost the fork draft')
    before_revision = devtools.evaluate('ContextWorkspace.current.revision')
    devtools.evaluate("(()=>{const node=ContextGraph.data.nodes.find(n=>n.action_snapshot==="+json.dumps(first['tip'])+");"
        "if(!node)throw Error('Shared assistant boundary is missing');"
        "ContextGraph.inspect([node.id],ContextGraph.model(ContextGraph.data));"
        "document.querySelector('#graph-inspector [data-workspace-rewind]').click();})()")
    devtools.wait_for("ContextWorkspace.current?.tip==="+json.dumps(first['tip'])+
        "&&ContextWorkspace.current.revision>"+str(before_revision)+"&&!ContextGraph.stale", 'graph rewind publication')
    devtools.evaluate("(()=>{const toggle=document.getElementById('graph-previous');toggle.checked=true;"
        "toggle.dispatchEvent(new Event('change'));})()")
    devtools.wait_for('ContextGraph.data?.previous_tips_requested===true&&!ContextGraph.stale', 'previous graph tips')
    previous = devtools.evaluate('({tips:ContextGraph.data.previous_tips,nodes:ContextGraph.data.nodes,branches:ContextGraph.data.branches})')
    previous_tip = next((tip for tip in previous['tips'] if tip['snapshot'] == branches[forked['branch']]['snapshot']), None)
    if previous_tip is None or next(node for node in previous['nodes'] if node['id'] == previous_tip['node_id'])['current_branch_count'] != 0:
        raise RuntimeError('Rewound-away history was lost or counted as an active branch')
    if any(branch['saved_messages'] != first['count'] for branch in previous['branches']):
        raise RuntimeError('Previous history changed current saved-message counts')
    devtools.screenshot(output / '05-previous-tip.png')
    devtools.evaluate('ContextWorkspace.selectNative('+source+')')
    return dict(id='SMOKE-GRAPH',status='PASS',branches=2,shared_prefix_messages=first['count'],
        counts_before_rewind={key:value['saved_messages'] for key,value in branches.items()},
        drafts_preserved=True,unavailable_boundary_has_no_actions=True,rewind_published=True,
        previous_tip_retained=True,previous_tip_not_active=True)


def _network_exchange(devtools, path, cursor, *, require_client_ack=False):
    """Return the actual browser request body and completed NDJSON response."""
    events = devtools.events[cursor:]
    requests = [event for event in events if event.get('method') == 'Network.requestWillBeSent'
                and urlsplit(event.get('params', {}).get('request', {}).get('url', '')).path == path]
    if not requests:
        raise RuntimeError(f'Chromium did not issue the expected {path} request')
    request_event = requests[-1]['params']
    request_id = request_event['requestId']
    response_event = next((event['params'] for event in reversed(events)
        if event.get('method') == 'Network.responseReceived'
        and event.get('params', {}).get('requestId') == request_id), None)
    if response_event is None or response_event.get('response', {}).get('status') != 200:
        raise RuntimeError(f'{path} did not receive HTTP 200')
    loading_failed = next((event.get('params', {}) for event in events
        if event.get('method') == 'Network.loadingFailed'
        and event.get('params', {}).get('requestId') == request_id), None)
    loading_finished = any(event.get('method') == 'Network.loadingFinished'
        and event.get('params', {}).get('requestId') == request_id for event in events)
    post = devtools.call('Network.getRequestPostData', {'requestId': request_id}).get('postData')
    if not isinstance(post, str):
        raise RuntimeError(f'Chromium did not expose the actual {path} request body')
    request_body = json.loads(post)
    client_ack = None
    client_stream = None
    if require_client_ack:
        client_ack, outcome = _client_ack_for_request(devtools, path, request_id,
            request_event.get('request', {}), response_event.get('response', {}).get('status'), loading_failed)
        raw = outcome['raw_ndjson']
        client_stream = raw
    else:
        if loading_failed is not None or not loading_finished:
            raise RuntimeError(f'{path} response body is not complete for the matching request')
        response = devtools.call('Network.getResponseBody', {'requestId': request_id})
        raw = response.get('body', '')
        if response.get('base64Encoded'):
            raw = base64.b64decode(raw).decode('utf-8')
    rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
    if not rows or rows[-1].get('event') != 'complete':
        raise RuntimeError(f'{path} response lacks a durable complete event')
    return {'request': request_body, 'post_data': post,
            'post_data_sha256': hashlib.sha256(post.encode('utf-8')).hexdigest(),
            'events': rows, 'complete': rows[-1], 'request_id': request_id,
            'client_ack': client_ack, 'client_stream_ndjson': client_stream}


def _validate_client_complete_ack(path, request_id, request, response_status, failure,
                                  outcome, stream_outcomes):
    """Validate a UI-consumed durable receipt before qualifying a reusable request.

    A specific Chromium ERR_ABORTED is retained as a transport diagnostic. It is
    accepted only when the actual response reader reached EOF and the UI parser
    returned a complete receipt whose durable NDJSON frame is tied to this POST.
    """
    def reject(reason):
        raise RuntimeError(f'{path} lacks a valid client-consumed durable receipt: {reason}')

    if type(response_status) is not int or response_status != 200:
        reject('matching response was not HTTP 200')
    if failure is not None and not (
            failure.get('errorText') == 'net::ERR_ABORTED' and failure.get('canceled') is True):
        reject('CDP failure is not the known canceled ERR_ABORTED classification')
    if not isinstance(request, dict):
        reject('actual POST body is unavailable')
    operation_id = request.get('operation_id')
    branch_name = request.get('branch')
    if not isinstance(operation_id, str) or not operation_id or not isinstance(branch_name, str) or not branch_name:
        reject('actual POST lacks operation_id or branch')
    if not isinstance(outcome, dict):
        reject('client stream observer is missing')
    if (outcome.get('path') != path or outcome.get('operation_id') != operation_id
            or outcome.get('branch') != branch_name):
        reject('client outcome does not match this request path, operation, and branch')
    if outcome.get('outcome') != 'complete' or outcome.get('complete_event') != 'complete':
        reject('UI parser did not return a complete event')
    if outcome.get('acknowledged') is not True:
        reject('UI observer did not verify the returned receipt')
    if (outcome.get('has_receipt') is not True or outcome.get('has_answer') is not True
            or outcome.get('durable_frame_matches_operation') is not True
            or outcome.get('complete_frame_matches_receipt') is not True):
        reject('UI parser result lacks a matching durable and complete frame')
    if outcome.get('raw_ndjson_truncated') is not False:
        reject('raw NDJSON capture is missing or overflowed')
    raw = outcome.get('raw_ndjson')
    if not isinstance(raw, str) or not raw or len(raw.encode('utf-8')) > 65536:
        reject('bounded raw NDJSON is unavailable or exceeds its limit')
    if outcome.get('raw_ndjson_bytes') != len(raw.encode('utf-8')):
        reject('raw NDJSON byte count is missing or inconsistent')
    try:
        frames = [json.loads(line) for line in raw.splitlines() if line]
    except (json.JSONDecodeError, TypeError):
        reject('raw NDJSON contains a parser error')
    if not frames or any(not isinstance(frame, dict) for frame in frames):
        reject('raw NDJSON frames are missing or not objects')
    if frames[-1].get('event') != 'complete':
        reject('complete frame is missing or followed by another frame')
    if sum(frame.get('event') == 'complete' for frame in frames) != 1:
        reject('stream contains duplicate complete frames')
    durable_frames = [frame for frame in frames if frame.get('event') == 'durable']
    if len(durable_frames) != 1:
        reject('stream must contain one durable frame')
    durable = durable_frames[0]
    complete = frames[-1]
    if durable.get('operation') != operation_id:
        reject('durable frame operation does not match the actual POST')
    if complete.get('operation', operation_id) != operation_id:
        reject('complete frame operation does not match the actual POST')
    durable_receipt, complete_receipt = durable.get('receipt'), complete.get('receipt')
    if not isinstance(durable_receipt, dict) or not isinstance(complete_receipt, dict):
        reject('durable or complete receipt is missing')
    answer = complete.get('answer')
    if not isinstance(answer, dict):
        reject('complete frame answer is missing')
    snapshot = durable_receipt.get('snapshot')
    durable_branch, complete_branch = durable_receipt.get('branch'), complete_receipt.get('branch')
    if not isinstance(durable_branch, dict) or not isinstance(complete_branch, dict):
        reject('durable or complete branch receipt is missing')
    revision = durable_branch.get('revision')
    if type(revision) is not int or revision < 0:
        reject('receipt revision is not an exact nonnegative integer')
    if (not isinstance(snapshot, str) or not snapshot or durable_branch.get('name') != branch_name
            or durable_branch.get('tip') != snapshot):
        reject('durable receipt does not identify the request branch and snapshot tip')
    if (complete_receipt.get('snapshot') != snapshot
            or answer.get('snapshot') != snapshot
            or complete_branch.get('name') != branch_name
            or complete_branch.get('tip') != snapshot
            or type(complete_branch.get('revision')) is not int
            or complete_branch.get('revision') != revision):
        reject('complete receipt does not match the durable snapshot, branch, and revision')
    if (outcome.get('receipt_snapshot') != snapshot or outcome.get('answer_snapshot') != snapshot
            or outcome.get('receipt_branch') != branch_name
            or type(outcome.get('receipt_revision')) is not int
            or outcome.get('receipt_revision') != revision):
        reject('UI-returned receipt does not match the raw durable frames')
    if not isinstance(stream_outcomes, list):
        reject('reader events are missing')
    matching_stream = [row for row in stream_outcomes if isinstance(row, dict)
        and row.get('path') == path and row.get('operation_id') == operation_id]
    if not matching_stream:
        reject('reader events for this request are missing')
    if any(row.get('outcome') == 'error' for row in matching_stream):
        reject('response reader reported an error')
    if matching_stream[-1].get('outcome') != 'done':
        reject('response reader did not reach EOF')
    raw_bytes = raw.encode('utf-8')
    return {'request_id': request_id, 'path': path, 'operation_id': operation_id,
        'branch': branch_name, 'receipt': {'snapshot': snapshot, 'branch': branch_name,
            'revision': revision}, 'response_status': response_status,
        'cdp_failure': dict(failure) if failure is not None else None,
        'eof_consumed': True, 'raw_ndjson_bytes': len(raw_bytes),
        'raw_ndjson_sha256': hashlib.sha256(raw_bytes).hexdigest(),
        'durable_frame_sha256': hashlib.sha256(json.dumps(durable, sort_keys=True,
            separators=(',', ':')).encode('utf-8')).hexdigest(),
        'complete_frame_sha256': hashlib.sha256(json.dumps(complete, sort_keys=True,
            separators=(',', ':')).encode('utf-8')).hexdigest()}


def _network_failure_diagnostic(devtools, path, request_id, request, events):
    """Collect safe request/response details for a failed matching operation.

    Headers are deliberately excluded because they may contain the local test key.
    Bodies are bounded in the artifact; hashes retain a stable reference to larger
    payloads without copying them in full.
    """
    failure = next((row.get('params', {}) for row in events
        if row.get('method') == 'Network.loadingFailed'
        and row.get('params', {}).get('requestId') == request_id), {})
    response_event = next((row.get('params', {}).get('response', {}) for row in reversed(events)
        if row.get('method') == 'Network.responseReceived'
        and row.get('params', {}).get('requestId') == request_id), {})
    try:
        post_data = devtools.call('Network.getRequestPostData', {'requestId': request_id}).get('postData')
    except Exception as exc:
        post_data = None
        post_error = type(exc).__name__
    else:
        post_error = None
    try:
        response = devtools.call('Network.getResponseBody', {'requestId': request_id})
        body = response.get('body', '')
        if response.get('base64Encoded'):
            body = base64.b64decode(body).decode('utf-8', errors='replace')
        response_error = None
    except Exception as exc:
        body = ''
        response_error = f'{type(exc).__name__}: {str(exc)[:500]}'
    body_bytes = body.encode('utf-8')
    request_value = post_data
    if isinstance(post_data, str):
        try:
            request_value = json.loads(post_data)
        except (json.JSONDecodeError, TypeError):
            pass
    event_summary = []
    browser_event_summary = []
    for row in events:
        params = row.get('params', {})
        page_method = row.get('method')
        if page_method == 'Page.frameNavigated':
            frame = params.get('frame', {})
            browser_event_summary.append({'event': page_method, 'frame_id': frame.get('id'),
                'url_path': urlsplit(frame.get('url', '')).path})
        elif page_method == 'Page.frameDetached':
            browser_event_summary.append({'event': page_method, 'frame_id': params.get('frameId'),
                'reason': params.get('reason')})
        elif page_method == 'Page.lifecycleEvent':
            browser_event_summary.append({'event': page_method, 'name': params.get('name'),
                'frame_id': params.get('frameId')})
        if params.get('requestId') != request_id:
            continue
        method_name = row.get('method')
        if method_name == 'Network.requestWillBeSent':
            request_data = params.get('request', {})
            event_summary.append({'event': method_name, 'method': request_data.get('method'),
                                  'path': urlsplit(request_data.get('url', '')).path})
        elif method_name == 'Network.responseReceived':
            response_data = params.get('response', {})
            event_summary.append({'event': method_name, 'status': response_data.get('status'),
                                  'mime_type': response_data.get('mimeType')})
        elif method_name == 'Network.dataReceived':
            event_summary.append({'event': method_name, 'data_length': params.get('dataLength'),
                                  'encoded_data_length': params.get('encodedDataLength')})
        elif method_name == 'Network.loadingFinished':
            event_summary.append({'event': method_name,
                                  'encoded_data_length': params.get('encodedDataLength')})
        elif method_name == 'Network.loadingFailed':
            event_summary.append({'event': method_name,
                **{key: params.get(key) for key in ('errorText', 'canceled', 'blockedReason')}})
    try:
        stream_outcomes = devtools.evaluate('window.__cowStreamOutcomes||[]')
    except Exception:
        stream_outcomes = None
    try:
        consume_outcomes = devtools.evaluate('window.__cowConsumeOutcomes||[]')
    except Exception:
        consume_outcomes = None
    try:
        truncated_control = devtools.evaluate('window.__cowTruncatedControl||null')
    except Exception:
        truncated_control = None
    return {
        'path': path,
        'request_id': request_id,
        'method': request.get('method'),
        'request': request_value,
        'post_data_sha256': hashlib.sha256(post_data.encode('utf-8')).hexdigest()
            if isinstance(post_data, str) else None,
        'post_data_error': post_error,
        'response_status': response_event.get('status'),
        'failure': {key: failure.get(key) for key in ('errorText', 'canceled', 'blockedReason')},
        'response_body': body[:65536],
        'response_body_sha256': hashlib.sha256(body_bytes).hexdigest(),
        'response_body_truncated': len(body) > 65536,
        'response_body_error': response_error,
        'events': event_summary,
        'browser_lifecycle_events': browser_event_summary,
        'ui_stream_outcomes': stream_outcomes,
        'ui_consume_outcomes': consume_outcomes,
        'truncated_stream_control': truncated_control,
    }


def _install_stream_observer(devtools):
    """Opt-in diagnostics: capture bounded raw NDJSON, including generated text."""
    devtools.call('Page.setLifecycleEventsEnabled', {'enabled': True})
    devtools.evaluate("""(()=>{
      if(window.__cowStreamObserverInstalled)return true;
      window.__cowStreamObserverInstalled=true;window.__cowStreamOutcomes=[];
      window.__cowConsumeOutcomes=[];
      const streamMeta=new WeakMap(),readerMeta=new WeakMap(),responseMeta=new WeakMap();
      const originalFetch=window.fetch.bind(window);
      window.fetch=async function(...args){
        const response=await originalFetch(...args);
        const url=typeof args[0]==='string'?args[0]:args[0]?.url||'';
        const path=new URL(url,location.href).pathname;
        if(path.startsWith('/cow/text/')){
          let post={};try{post=JSON.parse(args[1]?.body||'{}')}catch{}
          const meta={path,started:performance.now(),operation_id:post.operation_id||null,
            branch:post.branch||null,decoder:new TextDecoder(),raw_ndjson:'',raw_ndjson_bytes:0,overflow:false};
          responseMeta.set(response,meta);streamMeta.set(response.body,meta);
        }
        return response;
      };
      const originalConsume=window.consumeCowEvents||window.CowProtocol?.consumeCowEvents;
      if(typeof originalConsume==='function'){
        const wrappedConsume=async function(...args){
          const meta=responseMeta.get(args[0]);
          try{const value=await originalConsume.apply(this,args);
            const receipt=value?.receipt||{},branch=receipt.branch||{};
            const frames=(meta?.raw_ndjson||'').split('\\n').filter(Boolean);
            let parsed=[];try{parsed=frames.map(line=>JSON.parse(line))}catch{}
            const durableFrame=parsed.find(frame=>frame.event==='durable');
            const completeFrame=parsed.at(-1);
            const durableReceipt=durableFrame?.receipt||{}, durableBranch=durableReceipt.branch||{};
            const acknowledged=value?.event==='complete'&&typeof receipt.snapshot==='string'
              &&typeof meta?.operation_id==='string'
              &&!meta?.overflow&&meta.raw_ndjson_bytes<=65536
              &&branch.name===meta?.branch&&branch.tip===receipt.snapshot
              &&Number.isInteger(branch.revision)
              &&durableFrame?.operation===meta.operation_id
              &&completeFrame?.event==='complete'
              &&completeFrame?.answer?.snapshot===receipt.snapshot
              &&durableReceipt.snapshot===receipt.snapshot
              &&durableBranch.name===branch.name&&durableBranch.tip===branch.tip
              &&durableBranch.revision===branch.revision
              &&completeFrame?.receipt?.snapshot===receipt.snapshot
              &&completeFrame?.receipt?.branch?.name===branch.name
              &&completeFrame?.receipt?.branch?.tip===branch.tip
              &&completeFrame?.receipt?.branch?.revision===branch.revision;
            window.__cowConsumeOutcomes.push({path:meta?.path||null,operation_id:meta?.operation_id||null,
              branch:meta?.branch||null,outcome:'complete',acknowledged,
              complete_event:value?.event||null,receipt_snapshot:receipt.snapshot||null,
              answer_snapshot:value?.answer?.snapshot||null,
              receipt_branch:branch.name||null,receipt_revision:branch.revision??null,
              has_receipt:!!value?.receipt,has_answer:!!value?.answer,
              durable_frame_matches_operation:durableFrame?.operation===meta?.operation_id,
              complete_frame_matches_receipt:completeFrame?.event==='complete'
                &&completeFrame?.receipt?.snapshot===receipt.snapshot
                &&completeFrame?.receipt?.branch?.name===branch.name
                &&completeFrame?.receipt?.branch?.tip===branch.tip
                &&completeFrame?.receipt?.branch?.revision===branch.revision,
              raw_ndjson:meta?.raw_ndjson?.slice(0,65536)||'',
              raw_ndjson_bytes:meta?.raw_ndjson_bytes||0,raw_ndjson_truncated:!!meta?.overflow,
              elapsed_ms:Math.round(performance.now()-(meta?.started||performance.now()))});
            return value;
          }catch(error){window.__cowConsumeOutcomes.push({path:meta?.path||null,
            operation_id:meta?.operation_id||null,branch:meta?.branch||null,outcome:'error',
            error_name:error?.name||'Error',error_message:String(error?.message||error).slice(0,300),
            raw_ndjson:meta?.raw_ndjson?.slice(0,65536)||'',
            raw_ndjson_bytes:meta?.raw_ndjson_bytes||0,raw_ndjson_truncated:!!meta?.overflow,
            elapsed_ms:Math.round(performance.now()-(meta?.started||performance.now()))});throw error;}
        };
        window.consumeCowEvents=wrappedConsume;
        if(window.CowProtocol)window.CowProtocol.consumeCowEvents=wrappedConsume;
        try{consumeCowEvents=wrappedConsume;}catch{}
        Promise.resolve().then(async()=>{
          try{await originalConsume(new Response('{"event":"text","text":"partial"}\\n'),()=>{});
            window.__cowTruncatedControl={rejected:false,error:null};}
          catch(error){window.__cowTruncatedControl={rejected:true,
            error_name:error?.name||'Error',error_message:String(error?.message||error).slice(0,300)};}
        });
      }else window.__cowConsumeOutcomes.push({outcome:'observer_unavailable'});
      const getReader=ReadableStream.prototype.getReader;
      ReadableStream.prototype.getReader=function(...args){
        const reader=getReader.apply(this,args),meta=streamMeta.get(this);
        if(meta)readerMeta.set(reader,meta);return reader;
      };
      const read=ReadableStreamDefaultReader.prototype.read;
      ReadableStreamDefaultReader.prototype.read=function(...args){
        const meta=readerMeta.get(this),result=read.apply(this,args);
        if(!meta)return result;
        return result.then(item=>{let text='';if(item.value)text+=meta.decoder.decode(item.value,{stream:true});
          if(item.done)text+=meta.decoder.decode();
          const textBytes=new TextEncoder().encode(text).byteLength;meta.raw_ndjson_bytes+=textBytes;
          if(meta.raw_ndjson_bytes<=65536)meta.raw_ndjson+=text;else meta.overflow=true;
          window.__cowStreamOutcomes.push({path:meta.path,operation_id:meta.operation_id,
          outcome:item.done?'done':'chunk',bytes:item.value?.byteLength||0,
          elapsed_ms:Math.round(performance.now()-meta.started)});return item;},error=>{
          window.__cowStreamOutcomes.push({path:meta.path,operation_id:meta.operation_id,outcome:'error',
            error_name:error?.name||'Error',error_message:String(error?.message||error).slice(0,300),
            elapsed_ms:Math.round(performance.now()-meta.started)});throw error;
        });
      };
      return true;
    })()""")


def _client_ack_for_request(devtools, path, request_id, request, response_status, failure):
    if (not isinstance(request, dict) or request.get('method') != 'POST'
            or urlsplit(request.get('url', '')).path != path):
        raise RuntimeError(f'{path} CDP request identity does not match the POST')
    post_data = devtools.call('Network.getRequestPostData', {'requestId': request_id}).get('postData')
    if not isinstance(post_data, str):
        raise RuntimeError(f'{path} actual request body is unavailable for client acknowledgment')
    try:
        body = json.loads(post_data)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f'{path} actual request body is not valid JSON') from exc
    operation_id = body.get('operation_id') if isinstance(body, dict) else None
    if not isinstance(operation_id, str):
        raise RuntimeError(f'{path} request has no operation_id for client acknowledgment')
    outcome_expr = "(()=>{const rows=window.__cowConsumeOutcomes||[];return rows.some(x=>x.path===" + \
        json.dumps(path) + "&&x.operation_id===" + json.dumps(operation_id) + ")})()"
    devtools.wait_for(outcome_expr, f'{path} client stream consumption', timeout=10.0)
    outcome = devtools.evaluate("(()=>{const rows=window.__cowConsumeOutcomes||[];"
        "for(let i=rows.length-1;i>=0;i--)if(rows[i].path===" + json.dumps(path) +
        "&&rows[i].operation_id===" + json.dumps(operation_id) + ")return rows[i];return null})()")
    stream_outcomes = devtools.evaluate('window.__cowStreamOutcomes||[]')
    return _validate_client_complete_ack(path, request_id, body, response_status,
                                         failure, outcome, stream_outcomes), outcome


def _wait_for_network_completion(devtools, path, cursor, timeout=90.0, diagnostic_path=None,
                                 require_client_ack=False):
    """Pump CDP events until this operation's matching response is complete."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        events = devtools.events[cursor:]
        for event in events:
            params = event.get('params', {})
            request = params.get('request', {})
            if (event.get('method') == 'Network.requestWillBeSent'
                    and request.get('method') == 'POST'
                    and urlsplit(request.get('url', '')).path == path):
                request_id = params.get('requestId')
                response = next((row.get('params', {}).get('response', {}) for row in events
                    if row.get('method') == 'Network.responseReceived'
                    and row.get('params', {}).get('requestId') == request_id), None)
                failure_event = next((row.get('params', {}) for row in events
                    if row.get('method') == 'Network.loadingFailed'
                    and row.get('params', {}).get('requestId') == request_id), None)
                if failure_event is not None:
                    diagnostic = _network_failure_diagnostic(devtools, path, request_id, request, events)
                    if require_client_ack:
                        try:
                            diagnostic['client_ack'], _ = _client_ack_for_request(devtools, path,
                                request_id, request, (response or {}).get('status'), failure_event)
                        except Exception as ack_error:
                            diagnostic['client_ack_error'] = str(ack_error)
                        else:
                            diagnostic['ui_stream_outcomes'] = devtools.evaluate('window.__cowStreamOutcomes||[]')
                            diagnostic['ui_consume_outcomes'] = devtools.evaluate('window.__cowConsumeOutcomes||[]')
                            if diagnostic_path is not None:
                                target = Path(diagnostic_path)
                                target.parent.mkdir(parents=True, exist_ok=True)
                                target.write_text(json.dumps(diagnostic, indent=2, sort_keys=True) + '\n',
                                                  encoding='utf-8')
                            return request_id
                    if diagnostic_path is not None:
                        target = Path(diagnostic_path)
                        target.parent.mkdir(parents=True, exist_ok=True)
                        target.write_text(json.dumps(diagnostic, indent=2, sort_keys=True) + '\n',
                                          encoding='utf-8')
                    reason = failure_event.get('errorText') or 'unknown CDP network failure'
                    raise RuntimeError(f'{path} request {request_id} failed before durable completion: {reason}')
                complete = any(row.get('method') == 'Network.loadingFinished'
                               and row.get('params', {}).get('requestId') == request_id
                               for row in events)
                if complete:
                    if not response or response.get('status') != 200:
                        raise RuntimeError(f'{path} did not receive HTTP 200')
                    return request_id
        devtools.events.append(devtools._receive())
    raise RuntimeError(f'Timed out waiting for the matching {path} response to finish')


def _application_page_ready(observation, expected_url):
    return (isinstance(observation, dict) and observation.get('url') == expected_url
        and observation.get('readyState') == 'complete'
        and observation.get('hasWorkspace') is True)


def _wait_for_application_page(devtools, base_url, output, *, timeout=90):
    """A DevTools target may still be about:blank before initial navigation."""
    expected = base_url.rstrip('/') + '/'
    deadline = time.monotonic() + timeout
    observations = []
    while True:
        observed = devtools.evaluate("({url:location.href,readyState:document.readyState,"
            "hasWorkspace:typeof ContextWorkspace!=='undefined'})")
        observations.append(observed)
        ready = _application_page_ready(observed, expected)
        result = {'status': 'PASS_APPLICATION_DOCUMENT_READY' if ready else 'WAITING',
                  'expected_url': expected, 'observations': observations}
        if ready or time.monotonic() >= deadline:
            if not ready: result['status'] = 'FAIL_APPLICATION_DOCUMENT_TIMEOUT'
            (Path(output)/'browser-navigation.json').write_text(
                json.dumps(result, indent=2) + '\n', encoding='utf-8')
            if not ready:
                raise RuntimeError('Timed out waiting for the Context COW application document')
            return result
        time.sleep(.1)


def _validate_exact_forward_work(answer, suffix_count):
    counters = answer.get('counter_deltas', {})
    tokens = answer.get('tokens', [])
    forward = counters.get('forward_tokens')
    if (type(suffix_count) is not int or suffix_count < 0
            or type(forward) is not int or forward < 0):
        raise RuntimeError('Ask response lacks exact integer suffix/forward work')
    if forward != suffix_count + len(tokens):
        raise RuntimeError('Ask forward work is not exact suffix plus generated tokens')
    return counters, tokens


def _validate_server_receipt(receipt, fixture, label):
    if not isinstance(receipt, dict):
        raise RuntimeError(f'{label} durable receipt is missing')
    expected_model = fixture.get('model_contract_id')
    expected_execution = fixture.get('execution_profile_id')
    if (receipt.get('model_contract_id') != expected_model
            or receipt.get('execution_profile_id') != expected_execution):
        raise RuntimeError(f'{label} durable receipt does not match the pinned model/profile identity')
    return {'model_contract_id': receipt['model_contract_id'],
            'execution_profile_id': receipt['execution_profile_id']}


def _save_client_stream(output, label, exchange):
    raw = exchange.get('client_stream_ndjson')
    ack = exchange.get('client_ack')
    if not isinstance(raw, str) or not isinstance(ack, dict):
        raise RuntimeError(f'{label} lacks the complete client-consumed stream evidence')
    payload = raw.encode('utf-8')
    digest = hashlib.sha256(payload).hexdigest()
    if digest != ack.get('raw_ndjson_sha256') or len(payload) != ack.get('raw_ndjson_bytes'):
        raise RuntimeError(f'{label} client stream bytes differ from the validated acknowledgment')
    path = output / f'{label}-client-stream.ndjson'
    path.write_bytes(payload)
    evidence = {'path': path.name, 'bytes': len(payload), 'sha256': digest,
        'client_ack': ack}
    (output / f'{label}-client-ack.json').write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    return evidence


def _persist_completed_exchange(output, label, exchange):
    """Persist actual request, completed answer, counters, and client bytes before assertions."""
    output = Path(output)
    post_data = exchange.get('post_data')
    stream = exchange.get('client_stream_ndjson')
    if isinstance(post_data, str):
        (output / f'{label}-request.json').write_bytes(post_data.encode('utf-8'))
    if isinstance(stream, str):
        (output / f'{label}-client-stream.ndjson').write_bytes(stream.encode('utf-8'))
    receipt = {key: exchange.get(key) for key in (
        'request', 'post_data_sha256', 'request_id', 'events', 'complete', 'client_ack')}
    (output / f'{label}-exchange.json').write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    return {'request': f'{label}-request.json' if isinstance(post_data, str) else None,
            'client_stream': f'{label}-client-stream.ndjson' if isinstance(stream, str) else None,
            'exchange': f'{label}-exchange.json'}


def _publication_advanced(source, current, busy):
    return (not busy and isinstance(source, dict) and isinstance(current, dict)
            and current.get('id') == source.get('id')
            and type(current.get('revision')) is int
            and type(source.get('revision')) is int
            and current['revision'] > source['revision']
            and current.get('tip') != source.get('tip')
            and current.get('controller_status') in ('active', 'stopped'))


def _publication_guard_expression(source):
    # The old idle root is already !busy and may have an idle controller.
    # Completion requires a new branch revision and canonical tip.
    return "(()=>{const c=ContextWorkspace.current;return !busy&&c&&c.id===" + json.dumps(source['id']) + \
        "&&Number.isInteger(c.revision)&&c.revision>" + str(source['revision']) + "&&c.tip!==" + json.dumps(source['tip']) + \
        "&&['active','stopped'].includes(c.controller_status)})()"


def _select_context_root_boundary(graph, branch_id):
    if not isinstance(graph, dict):
        raise RuntimeError('Graph is unavailable for the selected branch root')
    branches = [branch for branch in graph.get('branches', [])
                if isinstance(branch, dict) and branch.get('id') == branch_id]
    if len(branches) != 1 or not isinstance(branches[0].get('root'), str):
        raise RuntimeError('Graph does not identify the selected branch root')
    root_tip = branches[0]['root']
    candidates = [node for node in graph.get('nodes', [])
        if isinstance(node, dict) and node.get('role') == 'root'
        and node.get('action_snapshot') == root_tip]
    if len(candidates) != 1:
        raise RuntimeError('Graph must expose a unique root node for the selected branch root')
    node = candidates[0]
    if not isinstance(node.get('id'), str) or node.get('boundary_available') is not True:
        raise RuntimeError('Selected branch root node lacks its actual action boundary')
    return node


def _unique_root_fork_action_index(action_snapshots, root_snapshot):
    if not isinstance(action_snapshots, list):
        raise RuntimeError('Graph inspector fork actions are unavailable')
    matches = [index for index, snapshot in enumerate(action_snapshots)
               if snapshot == root_snapshot]
    if len(matches) != 1:
        raise RuntimeError('Graph inspector must expose exactly one fork action for the selected root snapshot')
    return matches[0]


def _attach_context_file(devtools, filename, raw):
    text = raw.decode('utf-8')
    devtools.evaluate("(()=>{const input=document.getElementById('file');const dt=new DataTransfer();"
        "dt.items.add(new File([" + json.dumps(text) + "]," + json.dumps(filename)
        + ",{type:'text/plain'}));input.files=dt.files;input.dispatchEvent(new Event('change',{bubbles:true}));return true})()")
    devtools.wait_for("document.getElementById('attachments').textContent.includes(" + json.dumps(filename) + ")",
                      'code context attachment')


def _load_context_set(path, profile_id):
    builder_path = Path(__file__).resolve().parents[2] / 'experimental' / 'branching_context' / 'scripts' / 'build_cow_browser_context_fixture.py'
    spec = importlib.util.spec_from_file_location('cow_context_set_builder', builder_path)
    if spec is None or spec.loader is None:
        raise RuntimeError('Cannot load the pinned context-set fixture validator')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.load_reusable_context_set(path, profile_id)


def _continue_old_branch(devtools, branch, expected_tip, cursor, output, question, suffix_count):
    """Continue the exact selected source branch through the real composer."""
    source = devtools.evaluate("(()=>{const c=ContextWorkspace.current;return c?{id:c.id,revision:c.revision,tip:c.tip,status:c.controller_status}:null})()")
    if (not isinstance(source, dict) or source.get('id') != branch
            or source.get('tip') != expected_tip or source.get('status') not in ('active', 'stopped')):
        raise RuntimeError('Old branch selection did not restore its captured canonical state')
    if source['status'] == 'active':
        path = '/cow/text/resume'
        devtools.evaluate("document.getElementById('cow-resume-btn').click()")
    else:
        path = '/cow/text/append-user'
        devtools.evaluate("document.getElementById('input').value=" + json.dumps(question) + ";"
                          "document.getElementById('send-btn').click()")
    request_id = _wait_for_network_completion(devtools, path, cursor,
        diagnostic_path=output / 'old-branch-network-failure.json', require_client_ack=True)
    exchange = _network_exchange(devtools, path, cursor, require_client_ack=True)
    evidence = _persist_completed_exchange(output, 'old-branch', exchange)
    evidence['client_ack'] = _save_client_stream(output, 'old-branch', exchange)
    if exchange['request_id'] != request_id or exchange['request'].get('branch') != branch:
        raise RuntimeError('Old-branch continuation did not match the selected branch request')
    counters, tokens = _validate_exact_forward_work(exchange['complete'].get('answer', {}),
        suffix_count if path == '/cow/text/append-user' else 0)
    history = devtools.evaluate("ContextWorkspace.nativeRequest('workspace/history?branch='+encodeURIComponent(" + json.dumps(branch) + "))")
    if history.get('tip') != exchange['complete'].get('receipt', {}).get('snapshot'):
        raise RuntimeError('Old-branch continuation did not publish its actual durable tip')
    return {'path': path, 'request_id': request_id, 'request': exchange['request'],
            'source_tip': expected_tip, 'new_tip': history['tip'],
            'generated_tokens': len(tokens), 'counter_deltas': counters,
            'controller_status': history.get('controller_status'),
            'client_ack': exchange['client_ack'], 'evidence': evidence}


def _rewind_context_branch(devtools, branch, target_snapshot, expected_source_tip,
                           expected_question, output):
    """Use the real graph action and prove the rewound-away tip remains saved."""
    quoted_branch = json.dumps(branch)
    before = devtools.evaluate("ContextWorkspace.nativeRequest('workspace/history?branch='+encodeURIComponent(" +
        quoted_branch + "))")
    if before.get('tip') != expected_source_tip or before.get('controller_status') not in ('active','stopped'):
        raise RuntimeError('Context fork changed before graph rewind')
    devtools.evaluate('ContextGraph.refresh()')
    devtools.wait_for('ContextGraph.data&&!ContextGraph.stale', 'current canonical graph before rewind')
    state = devtools.evaluate("(()=>{const g=ContextGraph.data;const nodes=g?.nodes||[];" +
        "const matches=nodes.filter(n=>n.action_snapshot===" + json.dumps(target_snapshot) + ");" +
        "return {chat:g?.chat_id,branch:ContextWorkspace.current?.id,nodes:matches.map(n=>({id:n.id,boundary_available:n.boundary_available}))}})()")
    if state.get('branch') != branch or len(state.get('nodes', [])) != 1 or state['nodes'][0].get('boundary_available') is not True:
        raise RuntimeError('Graph does not expose one actual saved rewind boundary on the selected branch')
    revision = before['revision']
    devtools.evaluate("(()=>{const node=ContextGraph.data.nodes.find(n=>n.action_snapshot===" +
        json.dumps(target_snapshot) + ");ContextGraph.inspect([node.id],ContextGraph.model(ContextGraph.data));" +
        "const actions=[...document.querySelectorAll('#graph-inspector [data-workspace-rewind]')];" +
        "const matches=actions.filter(x=>x.dataset.workspaceRewind===" + json.dumps(target_snapshot) + ");" +
        "if(matches.length!==1)throw new Error('expected one rewind action for exact saved root');matches[0].click();return true})()")
    devtools.wait_for("ContextWorkspace.current?.id===" + quoted_branch +
        "&&ContextWorkspace.current?.tip===" + json.dumps(target_snapshot) +
        "&&ContextWorkspace.current?.revision>" + str(revision) + "&&!ContextGraph.stale",
        'durable rewind to saved context root')
    after = devtools.evaluate("ContextWorkspace.nativeRequest('workspace/history?branch='+encodeURIComponent(" +
        quoted_branch + "))")
    if after.get('tip') != target_snapshot or after.get('revision') <= revision:
        raise RuntimeError('Canonical branch did not move to the selected context root')
    if any(message.get('text') == expected_question
           for message in after.get('messages', []) if message.get('role') == 'user'):
        raise RuntimeError('Rewind retained the fork question in active canonical history')
    graph_observation = _refresh_rewound_previous_graph(devtools, after, branch,
        expected_source_tip, before['revision'], output)
    graph = graph_observation['graph']
    retained = _validate_rewound_previous_graph(graph_observation, after, branch,
        expected_source_tip, before['revision'])
    devtools.screenshot(output / '04-rewound-saved-tip.png')
    return {'branch':branch,'revision_before':revision,'revision_after':after['revision'],
        'tip_before':expected_source_tip,'tip_after':after['tip'],'saved_tip_retained':True,
        'active_messages_after':len(after['messages']),'previous_tip_count':len(graph['previous_tips']),
        'previous_graph_observation':'rewind-previous-graph-observation.json'}


def _previous_graph_refresh_expression():
    """Start the toggle refresh and return its first settled graph observation."""
    return ("(()=>{const key='__cowRewindGraphRefreshObserver';if(window[key])"
        "throw new Error('rewind graph observer already installed');"
        "const state=()=>{const graph=ContextGraph.data;const current=ContextWorkspace.current;"
        "const head=graph?.branches?.find(b=>b.id===current?.id);"
        "return {graph:graph?JSON.parse(JSON.stringify(graph)):null,stale:ContextGraph.stale,"
        "epoch:ContextGraph.epoch,selected:{id:current?.id,chat_id:current?.chat_id,"
        "revision:current?.revision,tip:current?.tip},graph_head:head?{id:head.id,"
        "revision:head.revision,snapshot:head.snapshot,tip:head.tip}:null};};"
        "const original=ContextGraph.refresh;const observer={original,attempts:[],state};window[key]=observer;"
        "ContextGraph.refresh=async function(...args){const started_epoch=this.epoch+1;"
        "const started_ms=performance.now();try{return await original.apply(this,args);}finally{"
        "observer.attempts.push({started_epoch,started_ms,settled:state()});}};"
        "const toggle=document.getElementById('graph-previous');toggle.checked=true;"
        "toggle.dispatchEvent(new Event('change'));return ContextGraph.refresh().then(()=>({"
        "attempts:observer.attempts.slice(),initial:state()}));})()")


def _rewound_graph_observer_cleanup_expression():
    return ("(()=>{const key='__cowRewindGraphRefreshObserver';const o=window[key];"
        "if(!o)return {attempts:[],final:null,observer_missing:true};"
        "ContextGraph.refresh=o.original;const final=o.state();const attempts=o.attempts.slice();"
        "delete window[key];return {attempts,final,observer_missing:false};})()")


def _capture_fresh_rewound_graph_expression(after, branch):
    """Capture graph and selected head in one JS turn after the exact head arrives."""
    chat = json.dumps(after.get('chat_id'))
    branch_id = json.dumps(branch)
    revision = json.dumps(after.get('revision'))
    tip = json.dumps(after.get('tip'))
    return ("(()=>{const key='__cowRewindGraphRefreshObserver';const o=window[key];"
        "const g=ContextGraph.data;const c=ContextWorkspace.current;"
        "const h=g?.branches?.find(b=>b.id===" + branch_id + ");"
        "const fresh=!ContextGraph.stale&&g?.previous_tips_requested===true&&g?.chat_id===" + chat +
        "&&c?.id===" + branch_id + "&&c?.chat_id===" + chat + "&&c?.revision===" + revision +
        "&&c?.tip===" + tip + "&&h?.revision===" + revision + "&&h?.snapshot===" + tip + ";"
        "if(!fresh)return null;const observation=o?o.state():{graph:g?JSON.parse(JSON.stringify(g)):null,"
        "stale:ContextGraph.stale,epoch:ContextGraph.epoch,selected:{id:c?.id,chat_id:c?.chat_id,"
        "revision:c?.revision,tip:c?.tip},graph_head:h?{id:h.id,revision:h.revision,"
        "snapshot:h.snapshot,tip:h.tip}:null};const attempts=o?o.attempts.slice():[];"
        "if(o){ContextGraph.refresh=o.original;delete window[key];}"
        "return {observation,attempts,observer_missing:!o};})()")


def _refresh_rewound_previous_graph(devtools, after, branch, expected_source_tip,
                                    source_revision, output):
    observation_path = output / 'rewind-previous-graph-observation.json'
    initial = None
    initial_error = None
    try:
        initial = devtools.evaluate(_previous_graph_refresh_expression())
    except Exception as exc:
        initial_error = f'{type(exc).__name__}: {exc}'
    receipt = {'schema': 'cow-rewind-graph-observation-v3',
        'captured_monotonic_ns': time.monotonic_ns(), 'request': {
            'chat_id': after.get('chat_id'), 'branch_id': branch,
            'head_revision': after.get('revision'), 'head_tip': after.get('tip'),
            'rewound_snapshot': after.get('tip'), 'expected_previous_snapshot': expected_source_tip,
            'expected_previous_revision': source_revision},
        'initial_refresh_observation': initial, 'initial_refresh_error': initial_error,
        'wait_completed': False, 'wait_error': None, 'refresh_attempts': [],
        'final_observation': None}
    _write_graph_observation(observation_path, receipt)
    wait_error = None
    captured = None
    if initial_error is None:
        try:
            deadline = time.monotonic() + 90.0
            capture_expression = _capture_fresh_rewound_graph_expression(after, branch)
            while time.monotonic() < deadline:
                captured = devtools.evaluate(capture_expression)
                if isinstance(captured, dict):
                    break
                time.sleep(0.2)
            if not isinstance(captured, dict):
                raise TimeoutError('Timed out waiting for a current post-rewind graph observation')
        except Exception as exc:
            wait_error = f'{type(exc).__name__}: {exc}'
    cleanup = None
    cleanup_error = None
    if captured is None:
        try:
            cleanup = devtools.evaluate(_rewound_graph_observer_cleanup_expression())
        except Exception as exc:
            cleanup_error = f'{type(exc).__name__}: {exc}'
    receipt.update({'wait_completed': wait_error is None and initial_error is None,
        'wait_error': wait_error, 'observer_cleanup_error': cleanup_error,
        'refresh_attempts': (captured.get('attempts', []) if isinstance(captured, dict)
                             else cleanup.get('attempts', []) if isinstance(cleanup, dict) else []),
        'final_observation': (captured.get('observation') if isinstance(captured, dict)
                              else cleanup.get('final') if isinstance(cleanup, dict) else None),
        'observation_capture': 'atomic_fresh_head' if isinstance(captured, dict) else 'cleanup_final'})
    _write_graph_observation(observation_path, receipt)
    observation = receipt['final_observation']
    error = initial_error or wait_error or cleanup_error
    if error is not None or not isinstance(observation, dict):
        raise RuntimeError(f'Previous-tip graph refresh did not reach the new head; evidence: {observation_path}')
    return observation


def _write_graph_observation(path, receipt):
    path.write_text(json.dumps(receipt, indent=2, ensure_ascii=False) + '\n',
        encoding='utf-8', newline='\n')


def _validate_rewound_previous_graph(observation, after, branch, expected_source_tip,
                                     source_revision):
    graph = observation.get('graph')
    selected = observation.get('selected')
    graph_head = observation.get('graph_head')
    if (not isinstance(graph, dict) or observation.get('stale') is not False
            or graph.get('previous_tips_requested') is not True):
        raise RuntimeError('Previous-tip graph response is missing, stale, or lacks previous points')
    if (not isinstance(selected, dict) or selected.get('id') != branch
            or selected.get('chat_id') != after.get('chat_id')
            or selected.get('revision') != after.get('revision')
            or selected.get('tip') != after.get('tip')):
        raise RuntimeError('Inspected graph belongs to a different selected branch head')
    if (graph.get('chat_id') != after.get('chat_id') or not isinstance(graph_head, dict)
            or graph_head.get('id') != branch or graph_head.get('revision') != after.get('revision')
            or graph_head.get('snapshot') != after.get('tip')):
        raise RuntimeError('Inspected graph does not represent the durable post-rewind head')
    retained = [tip for tip in graph.get('previous_tips', [])
                if tip.get('branch_id') == branch and tip.get('snapshot') == expected_source_tip]
    if len(retained) != 1 or retained[0].get('revision') != source_revision:
        raise RuntimeError('Rewound-away fork tip was not retained as one previous saved point')
    return retained


def _ask_from_ui(devtools, question, cursor, label, diagnostic_path=None):
    source = devtools.evaluate("(()=>{const c=ContextWorkspace.current;return c?{id:c.id,revision:c.revision,tip:c.tip}:null})()")
    if (not isinstance(source, dict) or not isinstance(source.get('id'), str)
            or type(source.get('revision')) is not int or not isinstance(source.get('tip'), str)):
        raise RuntimeError('Cannot identify the source branch before submitting an ask')
    guard = _publication_guard_expression(source)
    devtools.evaluate("document.getElementById('input').value=" + json.dumps(question) + ";"
                      "document.getElementById('send-btn').click()")
    request_id = _wait_for_network_completion(devtools, '/cow/text/ask', cursor,
        diagnostic_path=diagnostic_path, require_client_ack=True)
    devtools.wait_for(guard, label + ' new durable publication')
    exchange = _network_exchange(devtools, '/cow/text/ask', cursor, require_client_ack=True)
    if exchange['request_id'] != request_id:
        raise RuntimeError('Ask response does not match the completed browser request')
    return exchange


def _browser_trace(devtools, output, trace_events):
    devtools.call('Tracing.end')
    deadline = time.monotonic() + 20
    complete = False
    pending_events, devtools.events = devtools.events, []
    while time.monotonic() < deadline:
        message = pending_events.pop(0) if pending_events else devtools._receive()
        if message.get('method') == 'Tracing.dataCollected':
            trace_events.extend(message.get('params', {}).get('value', []))
        elif message.get('method') == 'Tracing.tracingComplete':
            complete = True
            break
        elif 'id' not in message:
            devtools.events.append(message)
    if not complete:
        raise RuntimeError('Chromium trace did not complete')
    trace_path = output / 'browser-trace.json'
    trace_path.write_text(json.dumps({'traceEvents': trace_events}, separators=(',', ':')), encoding='utf-8')
    return trace_path


def run_reusable_context_flow(devtools, output, checks, fixture, context_name, prepared_file,
                              context_bytes):
    """Click the real context control, ask twice, and fork the actual root."""
    branches = fixture['branches']
    question_a, question_b = branches['A']['question'], branches['B']['question']
    expected_root = fixture['root_messages']
    ui_hashes = devtools.evaluate("(async()=>{const paths=" + json.dumps(list(fixture['ui_source_hashes']))
        + ";const rows={};for(const p of paths){const bytes=await(await fetch('/web/'+p.split('/').pop())).arrayBuffer();"
        "const digest=await crypto.subtle.digest('SHA-256',bytes);rows[p]=[...new Uint8Array(digest)].map(x=>x.toString(16).padStart(2,'0')).join('');}return rows})()")
    if ui_hashes != fixture['ui_source_hashes']:
        raise RuntimeError('Served app/workspace/graph JavaScript differs from the CPU fixture receipts')
    devtools.call('Network.enable')
    _install_stream_observer(devtools)
    devtools.wait_for('window.__cowTruncatedControl?.rejected===true',
                      'truncated stream negative control', timeout=10.0)
    devtools.evaluate("document.getElementById('input').value=" + json.dumps(question_a) + ";")
    context_namespace = devtools.evaluate('ContextWorkspace.currentChat')
    existing_context_branches = devtools.evaluate("ContextWorkspace.nativeRequest('workspace/branches?chat='+encodeURIComponent(ContextWorkspace.currentChat)).then(x=>x.branches.length)")
    if not isinstance(context_namespace, str) or existing_context_branches != 0:
        raise RuntimeError('Reusable-context smoke requires the newly created empty chat namespace')
    context_cursor = len(devtools.events)
    devtools.evaluate("document.getElementById('cow-context-btn').click()")
    context_request_id = _wait_for_network_completion(devtools, '/cow/text/context', context_cursor,
        diagnostic_path=output / 'context-network-failure.json', require_client_ack=True)
    context_exchange = _network_exchange(devtools, '/cow/text/context', context_cursor,
                                         require_client_ack=True)
    if context_exchange['request_id'] != context_request_id:
        raise RuntimeError('Context receipt does not match the completed browser request')
    context_post = context_exchange['request']
    context_final = context_exchange['complete']
    if context_post.get('messages') != expected_root:
        raise RuntimeError('Actual Save-as-context request differs from tokenizer fixture messages')
    context_request_path = output / 'actual-context-request.json'
    context_request_path.write_bytes(context_exchange['post_data'].encode('utf-8'))
    context_stream_evidence = _save_client_stream(output, 'actual-context', context_exchange)
    receipt = context_final.get('receipt', {})
    server_identity = _validate_server_receipt(receipt, fixture, 'Context')
    devtools.wait_for("ContextWorkspace.currentChat===" + json.dumps(context_namespace)
        + "&&ContextWorkspace.current&&ContextWorkspace.current.initial_root_mode==='context_only'"
        + "&&ContextWorkspace.current.controller_status==='idle'",
        'durable code-only context root')
    context_history = devtools.evaluate("ContextWorkspace.nativeRequest('workspace/history?branch='+encodeURIComponent(ContextWorkspace.current.id))")
    if (not isinstance(context_history, dict) or context_history.get('initial_root_mode') != 'context_only'
            or context_history.get('controller_status') != 'idle'
            or context_history.get('messages', [{}])[0].get('text') != expected_root[0]['content']):
        raise RuntimeError('Durable context history differs from the actual user-role UI file framing')
    if devtools.evaluate("document.getElementById('input').value") != question_a:
        raise RuntimeError('Save-as-context did not retain the question draft')
    if devtools.evaluate("document.getElementById('attachments').textContent.trim()"):
        raise RuntimeError('Save-as-context left the uploaded file attached after publication')
    root_snapshot = receipt.get('snapshot')
    payload_sha256 = receipt.get('payload_sha256')
    if context_history.get('tip') != root_snapshot or not isinstance(root_snapshot, str):
        raise RuntimeError('Context completion receipt and selected canonical root differ')
    if not isinstance(payload_sha256, str) or len(payload_sha256) != 64:
        raise RuntimeError('Context completion receipt lacks a durable payload digest')
    catalog = devtools.evaluate("ContextWorkspace.nativeRequest('presentation?namespace='+encodeURIComponent(" + json.dumps(context_history['chat_id']) + "))")
    root_presentation = next((event.get('presentation') for event in catalog.get('events', [])
        if event.get('branch') == context_history['id'] and event.get('result', {}).get('tip') == root_snapshot
        and isinstance(event.get('presentation'), dict) and event['presentation'].get('kind') == 'root'), None)
    if not root_presentation or root_presentation.get('renderer_id') != fixture.get('renderer_id'):
        raise RuntimeError('Canonical root receipt is not bound to the fixture renderer identity')
    checks.append({'id': 'SMOKE-REUSABLE-CONTEXT-ROOT', 'status': 'PASS',
        'branch': context_history['id'], 'snapshot': root_snapshot,
        'payload_sha256': payload_sha256, 'controller_status': 'idle',
        'context_only': True, 'draft_retained': True,
        'actual_context_post_matches_fixture': True,
        'canonical_user_text_matches_actual_post': True,
        'request_post_sha256': context_exchange['post_data_sha256'],
        'client_ack': context_exchange['client_ack'],
        'client_stream': context_stream_evidence})
    devtools.screenshot(output / '02-context-only-root.png')

    ask_a = _ask_from_ui(devtools, question_a, len(devtools.events), 'independent question A',
                         output / 'ask-a-network-failure.json')
    ask_a_path = output / 'actual-ask-a-request.json'
    ask_a_path.write_bytes(ask_a['post_data'].encode('utf-8'))
    ask_a_stream_evidence = _save_client_stream(output, 'actual-ask-a', ask_a)
    final_a = ask_a['complete']
    if ask_a['request'].get('content') != question_a:
        raise RuntimeError('Actual A ask body differs from the tokenizer fixture question')
    answer_a = final_a.get('answer', {})
    identity_a = _validate_server_receipt(final_a.get('receipt'), fixture, 'A ask')
    if identity_a != server_identity:
        raise RuntimeError('A ask durable receipt changed server model/profile identity')
    suffix_a = branches['A']['suffix_token_count']
    counters_a, tokens_a = _validate_exact_forward_work(answer_a, suffix_a)
    history_a = devtools.evaluate("ContextWorkspace.nativeRequest('workspace/history?branch='+encodeURIComponent(ContextWorkspace.current.id))")
    if (history_a.get('tip') != final_a.get('receipt', {}).get('snapshot')
            or history_a.get('messages', [{}])[0].get('text') != expected_root[0]['content']
            or not any(message.get('role') == 'user' and message.get('text') == question_a
                       for message in history_a.get('messages', []))):
        raise RuntimeError('A history is not the canonical code root plus independent question')
    status_a = history_a.get('controller_status')
    if status_a not in ('active', 'stopped'):
        raise RuntimeError('A ask did not retain an active or stopped canonical controller')
    checks.append({'id': 'SMOKE-REUSABLE-ASK-A', 'status': 'PASS', 'branch': history_a['id'],
        'source_root': root_snapshot, 'snapshot': history_a['tip'], 'suffix_tokens': suffix_a,
        'generated_tokens': len(tokens_a), 'forward_tokens': counters_a['forward_tokens'],
        'controller_status': status_a, 'complete': status_a == 'stopped',
        'client_ack': ask_a['client_ack'], 'client_stream': ask_a_stream_evidence})

    source_branch = history_a['id']
    devtools.evaluate('ContextGraph.refresh()')
    devtools.wait_for('ContextGraph.data&&!ContextGraph.stale', 'fresh canonical graph before root fork')
    graph_data = devtools.evaluate("(()=>({branches:ContextGraph.data.branches.map(b=>({id:b.id,root:b.root})),"
        "nodes:ContextGraph.data.nodes.map(n=>({id:n.id,role:n.role,action_snapshot:n.action_snapshot,"
        "boundary_available:n.boundary_available}))}))()")
    graph_root = _select_context_root_boundary(graph_data, source_branch)
    if (not graph_root or graph_root.get('boundary_available') is not True
            or graph_root.get('action_snapshot') != root_snapshot):
        graph_state = devtools.evaluate("(()=>({root_id:ContextGraph.data.root_id,branches:ContextGraph.data.branches.map(b=>({id:b.id,root:b.root,tip:b.tip})),"
            "root_nodes:ContextGraph.data.nodes.filter(n=>n.role==='root').map(n=>({id:n.id,action_snapshot:n.action_snapshot,boundary_available:n.boundary_available}))}))()")
        raise RuntimeError('Graph does not expose the actual whole-context root boundary: '
                           + json.dumps(graph_state, sort_keys=True))
    root_id = json.dumps(graph_root['id'])
    # Linear runs are collapsed into segment buttons. Click the actual rendered
    # segment whose inspector chain includes the selected branch's root node.
    graph_button = devtools.evaluate("(()=>{const model=ContextGraph.model(ContextGraph.data);"
        "const rootId=" + root_id + ";const matches=model.edges.filter(e=>e.chain.includes(rootId));"
        "const endpoint=matches.length===1?matches[0].to:null;"
        "const segments=[...document.querySelectorAll('[data-graph-segment]')];"
        "const segment=segments.find(x=>x.dataset.graphSegment===endpoint);"
        "return {found:!!segment,segment:endpoint,matches:matches.length,"
        "segments:segments.map(x=>x.dataset.graphSegment)}})()")
    if not graph_button.get('found'):
        raise RuntimeError('Actual whole-context root boundary is not reachable through a unique graph segment: '
                           + json.dumps(graph_button, sort_keys=True))
    devtools.evaluate("(()=>{const endpoint=" + json.dumps(graph_button['segment']) + ";"
        "const b=[...document.querySelectorAll('[data-graph-segment]')].find(x=>x.dataset.graphSegment===endpoint);"
        "if(!b)throw new Error('root-boundary segment disappeared');b.click();return true})()")
    devtools.wait_for("!document.getElementById('graph-inspector').hidden&&"
                      "[...document.querySelectorAll('#graph-inspector [data-workspace-fork]')].some(x=>x.dataset.workspaceFork==="
                      + json.dumps(root_snapshot) + ")",
                      'actual context-root fork action')
    fork_snapshots = devtools.evaluate("[...document.querySelectorAll('#graph-inspector [data-workspace-fork]')].map(x=>x.dataset.workspaceFork)")
    fork_action_index = _unique_root_fork_action_index(fork_snapshots, root_snapshot)
    devtools.evaluate("(()=>{const buttons=[...document.querySelectorAll('#graph-inspector [data-workspace-fork]')];"
        "const index=" + str(fork_action_index) + ";if(buttons.length!==" + str(len(fork_snapshots)) +
        "||buttons[index]?.dataset.workspaceFork!==" + json.dumps(root_snapshot) + ")"
        "throw new Error('validated root fork action changed before click');buttons[index].click();return true})()")
    devtools.wait_for("!busy&&ContextWorkspace.current&&ContextWorkspace.current.id!==" + json.dumps(source_branch),
                      'forked code-only branch selection')
    fork_branch = devtools.evaluate('ContextWorkspace.current.id')
    fork_history_before = devtools.evaluate("ContextWorkspace.nativeRequest('workspace/history?branch='+encodeURIComponent(ContextWorkspace.current.id))")
    if (fork_history_before.get('tip') != root_snapshot or fork_history_before.get('controller_status') != 'idle'
            or any(message.get('text') == question_a for message in fork_history_before.get('messages', []))):
        raise RuntimeError('Root fork inherited A work instead of the saved code-only root')

    ask_b = _ask_from_ui(devtools, question_b, len(devtools.events), 'independent question B',
                         output / 'ask-b-network-failure.json')
    ask_b_path = output / 'actual-ask-b-request.json'
    ask_b_path.write_bytes(ask_b['post_data'].encode('utf-8'))
    ask_b_stream_evidence = _save_client_stream(output, 'actual-ask-b', ask_b)
    final_b = ask_b['complete']
    if ask_b['request'].get('content') != question_b:
        raise RuntimeError('Actual B ask body differs from the tokenizer fixture question')
    answer_b = final_b.get('answer', {})
    identity_b = _validate_server_receipt(final_b.get('receipt'), fixture, 'B ask')
    if identity_b != server_identity:
        raise RuntimeError('B ask durable receipt changed server model/profile identity')
    suffix_b = branches['B']['suffix_token_count']
    counters_b, tokens_b = _validate_exact_forward_work(answer_b, suffix_b)
    history_b = devtools.evaluate("ContextWorkspace.nativeRequest('workspace/history?branch='+encodeURIComponent(ContextWorkspace.current.id))")
    if (history_b.get('tip') != final_b.get('receipt', {}).get('snapshot')
            or history_b.get('messages', [{}])[0].get('text') != expected_root[0]['content']
            or any(message.get('text') == question_a for message in history_b.get('messages', []))
            or not any(message.get('role') == 'user' and message.get('text') == question_b
                       for message in history_b.get('messages', []))):
        raise RuntimeError('B history is not isolated to its code root and question')
    status_b = history_b.get('controller_status')
    if status_b not in ('active', 'stopped'):
        raise RuntimeError('B ask did not retain an active or stopped canonical controller')
    source_after = devtools.evaluate("ContextWorkspace.nativeRequest('workspace/history?branch='+encodeURIComponent(" + json.dumps(source_branch) + ")).then(h=>({tip:h.tip,messages:h.messages,controller_status:h.controller_status}))")
    if (source_after['tip'] != history_a['tip']
            or not any(message.get('text') == question_a for message in source_after['messages'])
            or any(message.get('text') == question_b for message in source_after['messages'])):
        raise RuntimeError('B ask changed or contaminated the A branch')
    rewind_report = _rewind_context_branch(devtools, fork_branch, root_snapshot,
        history_b['tip'], question_b, output)
    checks.append({'id': 'SMOKE-REUSABLE-ROOT-FORK', 'status': 'PASS',
        'source_branch': source_branch, 'fork_branch': fork_branch, 'fork_snapshot': root_snapshot,
        'fork_contains_context_only_root': True, 'fork_excludes_A': True})
    checks.append({'id': 'SMOKE-REUSABLE-ASK-B', 'status': 'PASS', 'branch': fork_branch,
        'source_root': root_snapshot, 'snapshot': history_b['tip'], 'suffix_tokens': suffix_b,
        'generated_tokens': len(tokens_b), 'forward_tokens': counters_b['forward_tokens'],
        'controller_status': status_b, 'complete': status_b == 'stopped',
        'A_branch_unchanged': True, 'client_ack': ask_b['client_ack'],
        'client_stream': ask_b_stream_evidence})
    checks.append({'id': 'SMOKE-REUSABLE-REWIND', 'status': 'PASS',
        'branch': fork_branch, **rewind_report})
    devtools.screenshot(output / '03-independent-A-B-branches.png')
    first = {'branch': source_branch, 'revision': history_a['revision'], 'status': status_a,
        'count': len(history_a['messages']), 'tip': history_a['tip'],
        'assistantText': ''.join(message.get('text', '') for message in history_a['messages'] if message.get('role') == 'assistant')}
    forked = {'branch': fork_branch, 'status': status_b, 'count': len(history_b['messages']),
        'hasForkQuestion': any(message.get('text') == question_b for message in history_b['messages']),
        'answerCharacters': sum(len(message.get('text', '')) for message in history_b['messages'] if message.get('role') == 'assistant')}
    return first, forked, {'ui_source_hashes': ui_hashes,
        'namespace_id': context_namespace,
        'context_request': {'path': context_request_path.name,
            'bytes': context_request_path.stat().st_size,
            'sha256': context_exchange['post_data_sha256'],
            'operation_id': context_post.get('operation_id'), 'receipt': receipt},
        'ask_a_request': {'path': ask_a_path.name, 'bytes': ask_a_path.stat().st_size,
            'sha256': ask_a['post_data_sha256'], 'operation_id': ask_a['request'].get('operation_id'),
            'answer': final_a.get('answer'), 'receipt': final_a.get('receipt')},
        'ask_b_request': {'path': ask_b_path.name, 'bytes': ask_b_path.stat().st_size,
            'sha256': ask_b['post_data_sha256'], 'operation_id': ask_b['request'].get('operation_id'),
            'answer': final_b.get('answer'), 'receipt': final_b.get('receipt')},
        'root_snapshot': root_snapshot, 'root_payload_sha256': payload_sha256,
        'root_token_count': fixture['root_token_count'],
        'root_token_ids_sha256': fixture['root_token_ids_sha256'],
        'server_receipt_identity': server_identity,
        'rewind': rewind_report,
        'actual_request_bodies_captured': True,
        'offline_checkpoint_file_audit': 'UNRUN; browser has no checkpoint-file access',
        'context_attachment': {'name': context_name, 'bytes': len(context_bytes),
            'sha256': hashlib.sha256(context_bytes).hexdigest()},
        'branches': {'A': {'id': source_branch, 'snapshot': history_a['tip'], 'status': status_a,
            'counter_deltas': counters_a, 'generated_tokens': len(tokens_a)},
            'B': {'id': fork_branch, 'snapshot': history_b['tip'], 'status': status_b,
            'counter_deltas': counters_b, 'generated_tokens': len(tokens_b)}}}


def run_reusable_context_set_flow(devtools, output, checks, rows):
    """Run existing one-context workflow three times in one page/server session."""
    results = []
    for index, row in enumerate(rows):
        fixture = row['fixture']
        filename = row['attachment_name']
        raw = row['context_bytes']
        text = raw.decode('utf-8')
        if index:
            devtools.evaluate("ContextWorkspace.makeNativeChat(" + json.dumps(
                'COW source context ' + fixture['name']) + ").then(()=>true)")
            devtools.wait_for("ContextWorkspace.currentChat&&ContextWorkspace.current===null&&"
                "document.getElementById('workspace-save-state').textContent.includes('Empty namespace')",
                'next distinct source namespace')
            _attach_context_file(devtools, filename, raw)
        case_output = output / ('context-' + str(index + 1))
        case_output.mkdir(mode=0o700)
        before = len(checks)
        first, forked, report = run_reusable_context_flow(
            devtools, case_output, checks, fixture, filename, text, raw)
        for check in checks[before:]:
            check['context'] = fixture['name']
        results.append({'name': fixture['name'], 'namespace_id': report['namespace_id'],
            'attachment_name': filename, 'attachment_sha256': hashlib.sha256(raw).hexdigest(),
            'fixture_sha256': row.get('fixture_sha256'),
            'source_receipts': fixture['source_receipts'],
            'root_token_ids_sha256': fixture['root_token_ids_sha256'],
            'question_token_ids_sha256': {label: fixture['branches'][label]['suffix_token_ids_sha256']
                                          for label in ('A','B')},
            'question_text': {label: fixture['branches'][label]['question'] for label in ('A','B')},
            'renderer_id': fixture['renderer_id'],
            'execution_profile_id': fixture['execution_profile_id'],
            'append_user_separator_token_count': fixture['append_user_separator_token_count'],
            'root_token_count': fixture['root_token_count'],
            'A_suffix_tokens': fixture['branches']['A']['suffix_token_count'],
            'B_suffix_tokens': fixture['branches']['B']['suffix_token_count'],
            'append_user_suffix_token_counts': {
                label: fixture['branches'][label]['append_user_suffix_token_count']
                for label in ('A', 'B')},
            'first_branch': first, 'fork_branch': forked, 'report': report})
        _write_context_set_progress(output, results, len(rows), checks)
    oldest = results[0]
    namespace = oldest['namespace_id']
    branch = oldest['first_branch']['branch']
    expected_tip = oldest['first_branch']['tip']
    devtools.evaluate("ContextWorkspace.openChatNative(" + json.dumps(namespace) + ").then(()=>true)")
    devtools.wait_for("ContextWorkspace.currentChat===" + json.dumps(namespace)
        + "&&ContextWorkspace.current", 'return to oldest source chat')
    devtools.evaluate("ContextWorkspace.selectNative(" + json.dumps(branch) + ").then(()=>true)")
    devtools.wait_for("ContextWorkspace.current?.id===" + json.dumps(branch)
        + "&&ContextWorkspace.current?.tip===" + json.dumps(expected_tip),
        'reselect oldest saved A branch')
    before_metrics = devtools.evaluate("ContextWorkspace.nativeRequest('status')")
    continuation = _continue_old_branch(devtools, branch, expected_tip,
        len(devtools.events), output, oldest['question_text']['A'],
        oldest['append_user_suffix_token_counts']['A'])
    after_metrics = devtools.evaluate("ContextWorkspace.nativeRequest('status')")
    checks.append({'id': 'SMOKE-REVISIT-OLDEST-BRANCH', 'status': 'PASS',
        'namespace_id': namespace, 'branch': branch, 'source_snapshot': expected_tip,
        'request_path': continuation['path'], 'request_id': continuation['request_id'],
        'generated_tokens': continuation['generated_tokens'],
        'counter_deltas': continuation['counter_deltas'],
        'pool_metrics_before': before_metrics, 'pool_metrics_after': after_metrics,
        'same_snapshot_evicted_and_reloaded': 'REQUIRES_POSTRUN_TRACE_AUDIT'})
    return results[0]['first_branch'], results[0]['fork_branch'], {
        'contexts': results, 'context_count': len(results),
        'same_page_and_server': True, 'revisited_oldest_branch': continuation,
        'scope': 'real browser UI over same live server; distinct context-only roots and whole-root A/B forks'}


def _write_context_set_progress(output, results, expected_contexts, checks):
    path = Path(output) / 'context-set-progress.json'
    payload = {'schema': 'cow-browser-context-set-progress-v1',
        'status': 'IN_PROGRESS' if len(results) < expected_contexts else 'CONTEXTS_COMPLETE',
        'completed_context_count': len(results), 'expected_context_count': expected_contexts,
        'completed_contexts': results, 'checks_so_far': checks}
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    os.replace(temporary, path)


def run(args) -> dict:
    output = args.output_dir.resolve()
    output.mkdir(mode=0o700, parents=True, exist_ok=False)
    os.chmod(output, 0o700)
    profile_root = Path(args.tmp_root).resolve()
    if not profile_root.is_dir():
        raise RuntimeError("Temporary profile root is unavailable")
    key = os.environ.get("COW_API_KEY", "")
    reusable_fixture = None
    reusable_rows = None
    reusable_mode = args.reusable_context or args.reusable_context_set is not None
    if args.reusable_context_set is not None:
        if args.reusable_context or args.reusable_fixture or args.context_file or args.context_name:
            raise RuntimeError('Use either a reusable context set or one reusable fixture, not both')
        reusable_rows = _load_context_set(args.reusable_context_set, args.server_profile_id)
        reusable_fixture = reusable_rows[0]['fixture']
        args.context_file = Path(reusable_rows[0]['context_file'])
        args.context_name = reusable_rows[0]['attachment_name']
    elif args.reusable_context:
        reusable_fixture = json.loads(args.reusable_fixture.read_text(encoding='utf-8'))
        if reusable_fixture.get('status') != 'PASS_CPU_TOKENIZER_TEMPLATE_ONLY':
            raise RuntimeError('Reusable-context fixture lacks passing CPU tokenizer/template parity')
        if args.server_profile_id != reusable_fixture.get('execution_profile_id'):
            raise RuntimeError('Server execution profile ID differs from the fixture profile receipt')
        if args.context_file is None or args.context_name != reusable_fixture.get('attachment_name'):
            raise RuntimeError('Reusable-context mode requires the fixture-matched attachment path and name')
        context_bytes = args.context_file.read_bytes()
        if hashlib.sha256(context_bytes).hexdigest() != reusable_fixture.get('source_body_sha256'):
            raise RuntimeError('Reusable-context file differs from the source fixture body')
        if args.output_limit > 128:
            raise RuntimeError('Reusable-context output limit is capped at 128 tokens')
        reusable_rows = [{'fixture': reusable_fixture, 'context_file': str(args.context_file),
            'attachment_name': args.context_name, 'context_bytes': context_bytes}]
    base_url = args.base_url.rstrip("/")
    if urlsplit(base_url).scheme != "http" or urlsplit(base_url).hostname not in ("127.0.0.1", "localhost"):
        raise RuntimeError("Use the loopback URL of an already-running COW server")
    profile = tempfile.mkdtemp(prefix="cow-browser-profile-", dir=profile_root)
    process = None
    devtools = None
    trace_events: list[dict] = []
    checks: list[dict] = []
    try:
        process = subprocess.Popen([
            args.browser, "--headless=new", "--no-sandbox", "--disable-gpu",
            "--disable-dev-shm-usage", "--no-first-run", "--no-default-browser-check",
            "--disable-background-networking", "--remote-allow-origins=*",
            "--remote-debugging-port=0", f"--user-data-dir={profile}", base_url + "/",
        ], stdout=(output / "chromium.stdout.log").open("wb"),
           stderr=(output / "chromium.stderr.log").open("wb"))
        port_file = Path(profile) / "DevToolsActivePort"
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline and process.poll() is None and not port_file.exists():
            time.sleep(0.1)
        if process.poll() is not None or not port_file.exists():
            raise RuntimeError("Chromium did not start its local DevTools endpoint")
        port = int(port_file.read_text(encoding="ascii").splitlines()[0])
        version = get_json(f"http://127.0.0.1:{port}/json/version")
        deadline = time.monotonic() + 15
        targets = []
        while time.monotonic() < deadline:
            targets = get_json(f"http://127.0.0.1:{port}/json/list")
            page = next((target for target in targets if target.get("type") == "page"), None)
            if page and page.get("webSocketDebuggerUrl"):
                break
            time.sleep(0.1)
        else:
            raise RuntimeError("Chromium page target was not created")
        devtools = DevTools(page["webSocketDebuggerUrl"])
        devtools.call("Page.enable")
        devtools.call("Runtime.enable")
        devtools.call("Emulation.setDeviceMetricsOverride", {
            "width": 1365, "height": 900, "deviceScaleFactor": 1, "mobile": False,
        })
        tracing_started = not getattr(args, 'diagnostic_disable_tracing', False)
        if tracing_started:
            devtools.call("Tracing.start", {"categories": "devtools.timeline,loading,blink.user_timing",
                                              "transferMode": "ReportEvents"})
        _wait_for_application_page(devtools, base_url, output)
        service = devtools.evaluate("fetch('/health').then(r=>r.json())")
        if service.get("service") != "context-cow":
            raise RuntimeError("The connected server did not identify as Context COW")
        checks.append({"id": "SMOKE-HEALTH", "status": "PASS", "service": service["service"]})
        devtools.wait_for("typeof ContextWorkspace!=='undefined'&&ContextWorkspace.nativeMode&&ContextWorkspace.loading===false",
                          "native workspace discovery")
        if key:
            devtools.wait_for("ContextWorkspace.nativeReady===false&&document.getElementById('input').disabled",
                              "unauthenticated workspace rejection")
            stored = devtools.evaluate("(()=>{document.getElementById('api-key').value=" + json.dumps(key) + ";"
                              "document.getElementById('api-key').dispatchEvent(new Event('change'));"
                              "return !!store.get('apikey','')})()")
            if not stored:
                raise RuntimeError("Browser did not retain the provided API key in its temporary profile")
            checks.append({"id": "SMOKE-AUTH", "status": "PASS", "key_value_recorded": False})
        try:
            devtools.wait_for("ContextWorkspace.nativeReady===true", "authenticated workspace initialization")
        except RuntimeError as error:
            if str(error) != "Timed out waiting for authenticated empty namespace":
                raise
            state = devtools.evaluate("({nativeMode:ContextWorkspace.nativeMode,nativeReady:ContextWorkspace.nativeReady,"
                "loading:ContextWorkspace.loading,hasChat:!!ContextWorkspace.currentChat,hasBranch:!!ContextWorkspace.current,"
                "composerDisabled:document.getElementById('input').disabled,keyStored:!!store.get('apikey',''),"
                "workspaceState:document.getElementById('workspace-save-state').textContent,"
                "toast:document.getElementById('toasts').textContent})")
            raise RuntimeError("Authenticated namespace setup did not finish: " + json.dumps(state, ensure_ascii=True)) from None
        # The fixture Store can survive a prior run. Always create a fresh empty
        # namespace so no stale branch or input can satisfy this smoke by reuse.
        devtools.evaluate("ContextWorkspace.makeNativeChat('COW Browser Smoke').then(()=>true)")
        devtools.wait_for("ContextWorkspace.currentChat&&ContextWorkspace.current===null&&document.getElementById('workspace-save-state').textContent.includes('Empty namespace')",
                          "fresh empty namespace")
        namespace_state = devtools.evaluate("({id:ContextWorkspace.currentChat,empty:document.getElementById('workspace-save-state').textContent})")
        empty_branches = devtools.evaluate("ContextWorkspace.nativeRequest('workspace/branches?chat='+encodeURIComponent(ContextWorkspace.currentChat)).then(x=>x.branches.length)")
        if empty_branches != 0 or "Empty namespace" not in namespace_state["empty"]:
            raise RuntimeError("New chat did not remain an empty namespace before first send")
        checks.append({"id": "SMOKE-EMPTY-NAMESPACE", "status": "PASS", "branches": 0})
        devtools.wait_for("![...document.querySelectorAll('.st-toast__title')].some(e=>['API key needed','COW workspace unavailable'].includes(e.textContent))",
                          'initial authentication alerts to expire', timeout=10)
        devtools.screenshot(output / "01-empty-namespace.png")
        devtools.evaluate("document.getElementById('sampling-btn').click();"
                          "document.getElementById('s-max').value=" + json.dumps(str(args.output_limit)) + ";"
                          "document.getElementById('s-apply').click()")
        devtools.wait_for("settings.max===" + json.dumps(str(args.output_limit)), "bounded output setting")

        default_file = "def clamp(value, low, high):\n    return max(low, min(value, high))\n"
        context_bytes = args.context_file.read_bytes() if args.context_file else default_file.encode("utf-8")
        if len(context_bytes) > 1024 * 1024:
            raise RuntimeError("Smoke context file exceeds the 1 MiB bounded attachment limit")
        try:
            prepared_file = context_bytes.decode("utf-8")
        except UnicodeDecodeError as error:
            raise RuntimeError("Smoke context file must be UTF-8 text") from error
        context_name = args.context_name or (args.context_file.name if args.context_file else "smoke_sample.py")
        if not context_name or Path(context_name).name != context_name:
            raise RuntimeError("Context attachment name must be a plain filename")
        context_hash = hashlib.sha256(context_bytes).hexdigest()
        devtools.evaluate("(()=>{const input=document.getElementById('file');const dt=new DataTransfer();"
                          "dt.items.add(new File([" + json.dumps(prepared_file) + "]," + json.dumps(context_name) + ",{type:'text/plain'}));"
                          "input.files=dt.files;input.dispatchEvent(new Event('change',{bubbles:true}));return true})()")
        devtools.wait_for("document.getElementById('attachments').textContent.includes(" + json.dumps(context_name) + ")",
                          "code context attachment")
        if reusable_mode:
            if context_name != reusable_fixture['attachment_name']:
                raise RuntimeError('Browser attachment name differs from the pinned UI fixture')
            if args.reusable_context_set is not None:
                first, forked, reusable_report = run_reusable_context_set_flow(
                    devtools, output, checks, reusable_rows)
                identity_report = reusable_report['contexts'][0]['report']
                attachment_report = [{'name': row['attachment_name'], 'bytes': len(row['context_bytes']),
                    'sha256': hashlib.sha256(row['context_bytes']).hexdigest(),
                    'source_name': row['fixture']['name']} for row in reusable_rows]
            else:
                first, forked, reusable_report = run_reusable_context_flow(
                    devtools, output, checks, reusable_fixture, context_name, prepared_file, context_bytes)
                identity_report = reusable_report
                attachment_report = {'name': context_name, 'bytes': len(context_bytes),
                    'sha256': hashlib.sha256(context_bytes).hexdigest()}
            trace_path = _browser_trace(devtools, output, trace_events) if tracing_started else None
            return {
                'schema': 1,
                'scope': 'real browser UI over live same-origin COW HTTP; opt-in file-context flow; no model started by driver',
                'server_service': service.get('service'),
                'server_model_label': service.get('model'),
                'server_execution_profile_id_caller_supplied': args.server_profile_id,
                'server_execution_profile_id_from_durable_receipts': identity_report['server_receipt_identity']['execution_profile_id'],
                'server_model_contract_id_from_durable_receipts': identity_report['server_receipt_identity']['model_contract_id'],
                'browser': version.get('Browser'),
                'browser_gpu_disabled': True,
                'authenticated_server': bool(service.get('api_key')),
                'api_key_in_report': False,
                'output_limit': args.output_limit,
                'max_resumes_per_answer': 0,
                'answer_forced_complete': False,
                'inference_kind': args.inference_kind,
                'reusable_context': True,
                'context_set_manifest_sha256': (hashlib.sha256(args.reusable_context_set.read_bytes()).hexdigest()
                    if args.reusable_context_set is not None else None),
                'reusable_context_report': reusable_report,
                'context_attachment': attachment_report,
                'tmp_root_storage_claim': 'none; browser profile removed at exit',
                'checks': checks,
                'first_branch': first,
                'fork_branch': forked,
                'artifacts': {str(path.relative_to(output)): {'bytes': path.stat().st_size,
                    'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
                    for path in sorted(output.rglob('*.png')) if path.is_file()}
                    | ({trace_path.name: {'bytes': trace_path.stat().st_size,
                        'sha256': hashlib.sha256(trace_path.read_bytes()).hexdigest()},
                        'trace_event_count': len(trace_events)} if trace_path else
                        {'trace': 'disabled by diagnostic control', 'trace_event_count': None}),
            }
        devtools.evaluate("document.getElementById('input').value='Answer only YES or NO: does clamp bound its result?';"
                          "document.getElementById('send-btn').click()")
        devtools.wait_for("!busy&&ContextWorkspace.current&&ContextWorkspace.current.controller_status",
                          "first answer publication")
        first_rounds = 0
        while True:
            status = devtools.evaluate("ContextWorkspace.current?.controller_status||''")
            if status == "stopped":
                break
            if status != "active":
                raise RuntimeError("First COW answer did not reach a resumable or completed state")
            if first_rounds >= args.max_resumes:
                raise SmokeBudgetLimit("first", first_rounds)
            devtools.evaluate("document.getElementById('cow-resume-btn').click()")
            devtools.wait_for("!busy&&ContextWorkspace.current?.controller_status", "answer resume")
            first_rounds += 1
        first = devtools.evaluate("(async()=>{const h=await ContextWorkspace.nativeRequest('workspace/history?branch='+encodeURIComponent(ContextWorkspace.current.id));"
            "const assistant=h.messages.filter(m=>m.role==='assistant').at(-1);const bubble=[...document.querySelectorAll('.st-msg--assistant .st-bubble')].at(-1);"
            "const ui=messages.filter(m=>m.role==='assistant').at(-1);"
            "const norm=s=>String(s).replace(/\\[([^\\]]+)\\]\\([^)]+\\)/g,'$1').replace(/[`*_~>#]/g,'').replace(/\\s+/g,' ').trim();"
            "return {branch:h.id,revision:h.revision,status:h.controller_status,count:h.messages.length,"
            "hasContext:h.messages.some(m=>m.role==='user'&&m.text.includes(" + json.dumps(context_name) + ")&&m.text.includes(" + json.dumps(prepared_file) + ")),"
            "assistantText:assistant?.text||'',uiText:ui?.text||'',domText:bubble?.innerText||bubble?.textContent||'',"
            "normalizedCanonical:norm(assistant?.text||''),normalizedDom:norm(bubble?.innerText||bubble?.textContent||''),"
            "output_tokens:assistant?.token_count??null,bubblePresent:!!bubble,tip:h.tip,assistantNode:assistant?.node_id||null}})()")
        if first["status"] != "stopped" or not first["hasContext"] or not first["assistantText"]:
            raise RuntimeError("First completed branch lacks canonical code context or assistant output")
        if first["assistantText"] != first["uiText"]:
            raise RuntimeError("UI message model differs from canonical history")
        if not first["bubblePresent"] or not first["domText"].strip() or first["normalizedCanonical"] != first["normalizedDom"]:
            raise RuntimeError("Rendered Markdown bubble does not preserve the canonical text projection")
        checks.append({"id": "SMOKE-PREPARE-ANSWER", "status": "PASS", "branch": first["branch"],
                       "revision": first["revision"], "messages": first["count"],
                       "output_characters": len(first["assistantText"]), "canonical_ui_model_match": True,
                       "normalized_markdown_projection_match": True, "resume_rounds": first_rounds})
        devtools.screenshot(output / "02-first-answer.png")

        fork_button = "document.querySelector('.st-msg--assistant [data-workspace-fork]')"
        if not devtools.evaluate(f"!!{fork_button}"):
            raise RuntimeError("Completed assistant snapshot did not expose a valid Fork here action")
        devtools.evaluate(f"{fork_button}.click()")
        devtools.wait_for("!busy&&ContextWorkspace.current&&ContextWorkspace.current.id!==" + json.dumps(first["branch"]),
                          "fork selection")
        fork_id = devtools.evaluate("ContextWorkspace.current.id")
        devtools.evaluate("document.getElementById('input').value='Answer only YES or NO: are values below low and above high useful clamp tests?';"
                          "document.getElementById('send-btn').click()")
        devtools.wait_for("!busy&&ContextWorkspace.current?.controller_status", "fork answer publication")
        fork_rounds = 0
        while True:
            status = devtools.evaluate("ContextWorkspace.current?.controller_status||''")
            if status == "stopped":
                break
            if status != "active":
                raise RuntimeError("Forked answer did not reach a resumable or completed state")
            if fork_rounds >= args.max_resumes:
                raise SmokeBudgetLimit("fork", fork_rounds)
            devtools.evaluate("document.getElementById('cow-resume-btn').click()")
            devtools.wait_for("!busy&&ContextWorkspace.current?.controller_status", "fork answer resume")
            fork_rounds += 1
        forked = devtools.evaluate("(async()=>{const h=await ContextWorkspace.nativeRequest('workspace/history?branch='+encodeURIComponent(ContextWorkspace.current.id));"
            "const assistant=h.messages.filter(m=>m.role==='assistant').at(-1);const answer=assistant?.text||'';"
            "return {branch:h.id,status:h.controller_status,count:h.messages.length,hasForkQuestion:h.messages.some(m=>m.role==='user'&&m.text.includes('useful clamp tests')),"
            "answerCharacters:answer.length,output_tokens:assistant?.token_count??null,dom:document.querySelectorAll('.st-msg--assistant').length}})()")
        source_after = devtools.evaluate("ContextWorkspace.nativeRequest('workspace/history?branch='+encodeURIComponent(" + json.dumps(first["branch"]) + ")).then(h=>({tip:h.tip,count:h.messages.length,status:h.controller_status}))")
        if forked["branch"] != fork_id or forked["status"] != "stopped" or not forked["hasForkQuestion"] or not forked["answerCharacters"]:
            raise RuntimeError("Fork branch did not retain its independent question and answer")
        if source_after["count"] != first["count"] or source_after["tip"] != first["tip"]:
            raise RuntimeError("Answer on the fork changed the source branch")
        devtools.evaluate("(()=>{const s=document.getElementById('branch-select');s.value=" + json.dumps(first["branch"]) + ";s.dispatchEvent(new Event('change',{bubbles:true}));return true})()")
        devtools.wait_for("ContextWorkspace.current?.id===" + json.dumps(first["branch"]), "source branch reselection")
        source_ui = devtools.evaluate("({count:ContextWorkspace.current.messages.length,branch:document.getElementById('branch-select').value})")
        if source_ui["count"] != first["count"]:
            raise RuntimeError("Branch selector did not restore the original canonical history")
        checks.append({"id": "SMOKE-FORK-ISOLATION", "status": "PASS", "source_branch": first["branch"],
                       "fork_branch": fork_id, "source_unchanged": True, "selection_restored": True,
                       "fork_resume_rounds": fork_rounds, "fork_output_characters": forked["answerCharacters"]})

        if args.check_graph:
            checks.append(check_graph(devtools, output, first, forked))

        # Exercise actual local settings and DOM math treatment without changing canonical history.
        devtools.evaluate("document.getElementById('sampling-btn').click();"
                          "document.getElementById('s-math').setAttribute('aria-checked','true');"
                          "document.getElementById('s-math-engine').value='katex';"
                          "document.getElementById('s-math-engine').dispatchEvent(new Event('change'));"
                          "document.getElementById('s-show').setAttribute('aria-checked','false');"
                          "document.getElementById('s-apply').click();"
                          "messages.push({role:'assistant',text:'Equation fixture: $x^2$.',reasoning:'',time:Date.now()});renderChat()")
        try:
            devtools.wait_for("!![...document.querySelectorAll('.st-msg--assistant')].at(-1)?.querySelector('.chat-math .katex')",
                              "KaTeX typeset output", timeout=30)
        except RuntimeError as error:
            state = devtools.evaluate("({math:settings.math,engine:settings.mathEngine,source:document.querySelector('.chat-math')?.outerHTML||null,"
                "toast:document.getElementById('toasts').textContent,scripts:[...document.scripts].map(s=>s.src).filter(Boolean)})")
            raise RuntimeError("KaTeX did not produce an engine element: " + json.dumps(state)) from error
        math_state = devtools.evaluate("({math:settings.math,engine:settings.mathEngine,show:settings.show,saved:JSON.parse(localStorage.getItem('strata.sampling')||'{}'),"
            "rendered:!![...document.querySelectorAll('.st-msg--assistant')].at(-1)?.querySelector('.chat-math .katex')})")
        if not math_state["math"] or math_state["engine"] != "katex" or math_state["show"] or not math_state["rendered"]:
            raise RuntimeError("Local equation display settings did not apply to the rendered DOM")
        devtools.evaluate("(()=>{document.getElementById('sampling-btn').click();"
                          "document.getElementById('s-math').setAttribute('aria-checked','false');"
                          "document.getElementById('s-apply').click();renderChat();return true})()")
        devtools.wait_for("!document.querySelector('.st-msg--assistant .chat-math')", "plain math rendering")
        plain_math = devtools.evaluate("(()=>{const row=[...document.querySelectorAll('.st-msg--assistant')].at(-1);return {math:settings.math,engine:settings.mathEngine,saved:JSON.parse(localStorage.getItem('strata.sampling')||'{}').math,"
            "hasMath:!!row?.querySelector('.chat-math'),plain:row?.querySelector('.st-bubble')?.textContent.includes('$x^2$')||false}})()")
        if plain_math["math"] or plain_math["saved"] or plain_math["hasMath"] or not plain_math["plain"] or plain_math["engine"] != "katex":
            raise RuntimeError("Disabling math did not persist or restore the source rendering")
        devtools.evaluate("messages.pop();renderChat()")
        checks.append({"id": "SMOKE-MATH", "status": "PASS", "engine": "katex",
                       "enable_and_disable_persisted": True, "typesetter_element_rendered": True})
        devtools.screenshot(output / "03-fork-and-settings.png")

        trace_path = None
        if tracing_started:
            devtools.call("Tracing.end")
            deadline = time.monotonic() + 20
            complete = False
            pending_events, devtools.events = devtools.events, []
            while time.monotonic() < deadline:
                message = pending_events.pop(0) if pending_events else devtools._receive()
                if message.get("method") == "Tracing.dataCollected":
                    trace_events.extend(message.get("params", {}).get("value", []))
                elif message.get("method") == "Tracing.tracingComplete":
                    complete = True
                    break
                elif "id" not in message:
                    devtools.events.append(message)
            if not complete:
                raise RuntimeError("Chromium trace did not complete")
            trace_path = output / "browser-trace.json"
            trace_path.write_text(json.dumps({"traceEvents": trace_events}, separators=(",", ":")), encoding="utf-8")
            checks.append({"id": "SMOKE-TRACE", "status": "PASS", "events": len(trace_events)})
        return {
            "schema": 1,
            "scope": "real browser UI over live same-origin COW HTTP routes; server/model process not started by driver",
            "server_service": service.get("service"),
            "server_model_label": service.get("model"),
            "browser": version.get("Browser"),
            "browser_gpu_disabled": True,
            "authenticated_server": bool(service.get("api_key")),
            "api_key_in_report": False,
            "output_limit": args.output_limit,
            "max_resumes_per_answer": args.max_resumes,
            "max_request_output_budget_upper_bound_per_answer": args.output_limit * (args.max_resumes + 1),
            "inference_kind": args.inference_kind,
            "context_attachment": {"name": context_name, "bytes": len(context_bytes), "sha256": context_hash},
            "tmp_root_storage_claim": "none; profile removed at exit",
            "checks": checks,
            "first_branch": {k: v for k, v in first.items() if k not in (
                "assistantText", "uiText", "domText", "normalizedCanonical", "normalizedDom")},
            "fork_branch": {k: v for k, v in forked.items() if k not in ("assistant",)},
            "artifacts": {
                path.name: {"bytes": path.stat().st_size,
                            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                for path in sorted(output.iterdir()) if path.is_file() and path.suffix == ".png"
            } | ({trace_path.name: {"bytes": trace_path.stat().st_size,
                                  "sha256": hashlib.sha256(trace_path.read_bytes()).hexdigest()},
                "trace_event_count": len(trace_events)} if trace_path else
                {"trace": "disabled by diagnostic control", "trace_event_count": None}),
        }
    finally:
        if devtools:
            try:
                devtools.call("Browser.close")
            except Exception:
                pass
            devtools.close()
        if process and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        shutil.rmtree(profile, ignore_errors=True)


def main() -> int:
    snap_profile_root = Path.home() / "snap/chromium/common"
    default_tmp_root = snap_profile_root if snap_profile_root.is_dir() else Path(tempfile.gettempdir())
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True, help="loopback URL for an already-running COW server")
    parser.add_argument("--output-dir", required=True, type=Path, help="new private directory for screenshots/report/trace")
    parser.add_argument("--browser", default="/snap/bin/chromium")
    parser.add_argument("--tmp-root", default=str(default_tmp_root),
                        help="private temporary browser-profile parent; the exact profile is removed on exit")
    parser.add_argument("--output-limit", type=int, default=8)
    parser.add_argument("--reusable-context", action="store_true",
                        help="opt in to real Save-as-context, whole-root fork, and independent A/B asks")
    parser.add_argument("--reusable-context-set", type=Path,
                        help="run three separately sourced contexts, each with A/B whole-root forks, then revisit the first branch")
    parser.add_argument("--reusable-fixture", type=Path,
                        help="pinned CPU tokenizer/template fixture made by build_cow_browser_context_fixture.py")
    parser.add_argument("--server-profile-id",
                        help="execution profile ID copied from this exact COW server's ready receipt")
    parser.add_argument("--max-resumes", type=int, default=8,
                        help="maximum explicit resume calls per answer; exhaustion is reported as INCOMPLETE, never forced complete")
    parser.add_argument("--context-file", type=Path, help="optional UTF-8 source file to attach instead of the tiny default example")
    parser.add_argument("--context-name", help="attachment filename override; defaults to the context file's basename")
    parser.add_argument("--check-graph", action="store_true", help="also exercise canonical graph selection, drafts, rewind and previous tips")
    parser.add_argument("--diagnostic-disable-tracing", action="store_true",
                        help="disable Chromium tracing as a control for transport diagnostics")
    parser.add_argument("--inference-kind", required=True, choices=("synthetic", "real"),
                        help="caller must label the server qualification; included in the report")
    args = parser.parse_args()
    if args.reusable_context or args.reusable_context_set is not None:
        if not 1 <= args.output_limit <= 128:
            parser.error("reusable-context output limit must be between 1 and 128")
        if not args.server_profile_id:
            parser.error("reusable-context mode requires --server-profile-id")
        if args.reusable_context_set is not None:
            if args.reusable_context or args.reusable_fixture or args.context_file or args.context_name:
                parser.error("--reusable-context-set cannot be combined with single-context inputs")
        else:
            if not args.reusable_fixture:
                parser.error("reusable-context mode requires --reusable-fixture")
            if not args.context_file or not args.context_name:
                parser.error("reusable-context mode requires --context-file and --context-name")
    elif not 1 <= args.output_limit <= 16:
        parser.error("default smoke output limit must be between 1 and 16")
    if not 0 <= args.max_resumes <= 32:
        parser.error("max resumes must be between 0 and 32")
    if args.context_name and not args.context_file:
        parser.error("--context-name requires --context-file")
    report_path = args.output_dir.resolve() / "report.json"
    try:
        result = run(args)
        result["status"] = "PASS"
        result["output_limit"] = args.output_limit
        report_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"event": "complete", "status": "PASS", "checks": len(result["checks"]),
                          "report": str(report_path)}))
        return 0
    except Exception as error:
        args.output_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
        os.chmod(args.output_dir, 0o700)
        status = "INCOMPLETE_BUDGET_LIMIT" if isinstance(error, SmokeBudgetLimit) else "FAIL"
        report_path.write_text(json.dumps({"schema": 1, "status": status, "stage": getattr(error, "stage", None),
                                           "resume_rounds": getattr(error, "rounds", None), "error": str(error),
                                           "inference_kind": args.inference_kind,
                                           "output_limit": args.output_limit,
                                           "max_resumes_per_answer": args.max_resumes,
                                           "max_request_output_budget_upper_bound_per_answer": args.output_limit * (args.max_resumes + 1),
                                           "answer_forced_complete": False,
                                           "api_key_in_report": False}, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"event": "complete", "status": status, "report": str(report_path),
                          "error": str(error)}))
        return 2 if status == "INCOMPLETE_BUDGET_LIMIT" else 1


if __name__ == "__main__":
    raise SystemExit(main())
