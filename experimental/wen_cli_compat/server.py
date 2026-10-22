"""Experimental Chat/Responses frontend over real COW snapshots (text only)."""
import argparse
import hashlib
import hmac
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import select
import signal
import socket
import sys
import tempfile
import threading
import time
import uuid

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT/'experimental/branching_context/src')]
from cache import CachedCow
from serve import responses
from serve.frontend import openai_to_messages, OutputParser
from serve.cow_http import model_text_components
from serve.cow_text_adapter import CowTextRequest
from serve.server import Service


class Frontend:
    def __init__(self, adapter, encoder, profile, model, max_output=2048, trace=None):
        self.adapter, self.encoder, self.profile, self.model = adapter, encoder, profile, model
        renderer = hashlib.sha256(encoder.template.source.encode()).hexdigest()
        self.cache = CachedCow(adapter, (profile['model_contract_id'],
                                       profile['execution_profile_id'], renderer))
        self.max_output, self.trace = max_output, trace
        self.trace_lock = threading.Lock()

    def prepare(self, req, dialect):
        if not isinstance(req, dict): raise ValueError('Request must be a JSON object')
        if req.get('model', self.model) != self.model: raise ValueError('Unknown model')
        if req.get('n', 1) != 1: raise ValueError('Only n=1 is supported')
        if type(req.get('stream', False)) is not bool: raise ValueError('stream must be boolean')
        if dialect == 'responses':
            responses.check_request(req)
            messages = responses.input_messages(req)
            tools, names, skipped = responses.request_tools(req)
            if skipped: raise ValueError('Hosted tools are unsupported: '+', '.join(skipped))
            fmt = responses.text_format(req)
            if fmt and fmt.get('type') != 'text': raise ValueError('Structured final-answer format is not qualified')
        else:
            messages, tools, _ = openai_to_messages(req)
            names = {}
            if req.get('response_format', {}).get('type', 'text') != 'text':
                raise ValueError('Structured final-answer format is not qualified')
            if req.get('tool_choice') == 'none': tools = None
        if not messages: raise ValueError('At least one input message is required')
        for message in messages:
            if not isinstance(message.get('content', ''), str):
                raise ValueError('This experiment accepts text only')
        if req.get('tool_choice') not in (None, 'auto', 'none'):
            raise ValueError('Forced tool choice is not qualified')
        for field in ('logprobs', 'top_logprobs', 'logit_bias', 'response_format'):
            if req.get(field) and field != 'response_format': raise ValueError(field+' is unsupported')
        # A fixed nonthinking renderer is common to all runners in this trial.
        kwargs = dict(enable_thinking=False)
        full = tuple(Service.encode_prompt(self.encoder, messages, tools,
                     dict(kwargs, add_generation_prompt=True)))
        closed = tuple(Service.encode_prompt(self.encoder, messages, tools,
                       dict(kwargs, add_generation_prompt=False)))
        bound = req.get('max_output_tokens' if dialect == 'responses' else 'max_completion_tokens')
        if bound is None: bound = req.get('max_tokens', self.max_output)
        if type(bound) is not int or not 1 <= bound <= self.max_output:
            # Larger client defaults are capped by the declared service budget.
            if type(bound) is int and bound > self.max_output: bound = self.max_output
            else: raise ValueError('Invalid output token limit')
        if len(full)+bound > self.profile['execution']['capacity']:
            raise ValueError('Rendered prompt plus output budget exceeds physical context capacity')
        settings = {key:req[key] for key in ('temperature', 'top_p') if key in req}
        if not settings: settings = {'temperature':0.0}
        seed = req.get('seed')
        stops = req.get('stop')
        if isinstance(stops, str): stops = [stops]
        return dict(req=req, dialect=dialect, messages=messages, tools=tools, names=names,
                    full=full, closed=closed, bound=bound, settings=settings, seed=seed, stops=stops)

    def events(self, plan, cancelled):
        req, dialect = plan['req'], plan['dialect']
        assembler = (responses.Assembler(req, self.model, len(plan['full']), plan['names'], True)
                     if dialect == 'responses' else None)
        identity = 'chatcmpl_'+uuid.uuid4().hex
        created = int(time.time())
        def chat(delta, finish=None, usage=None):
            value = dict(id=identity, object='chat.completion.chunk', created=created,
                         model=self.model, choices=[dict(index=0, delta=delta, finish_reason=finish)])
            if usage is not None: value['usage'] = usage
            return value
        if assembler:
            yield from assembler.start()
        else:
            yield chat(dict(role='assistant', content=''))
        start = time.monotonic()
        branch, request, reused = self.cache.start(plan['full'], plan['closed'], plan['bound'],
            settings=plan['settings'], cancelled=cancelled, seed=plan['seed'], text_stops=plan['stops'])
        view = CowTextRequest(request, self.encoder.tok,
            controller_reader=self.adapter.snapshot_controller, text_stops=plan['stops'] or ())
        parser = OutputParser(thinking=False, tools=plan['tools'], stream_tools=False)
        text, reasoning, calls = [], [], []
        def output(event):
            if event.kind == 'content': text.append(event.text)
            elif event.kind == 'reasoning': reasoning.append(event.text)
            elif event.kind == 'tool_call': calls.append(event.call)
            if assembler: return assembler.feed(event)
            if event.kind == 'content': return [chat(dict(content=event.text))]
            if event.kind == 'reasoning': return [chat(dict(reasoning_content=event.text))]
            if event.kind == 'tool_call':
                call = event.call
                return [chat(dict(tool_calls=[dict(index=len(calls)-1, id=call.id, type='function',
                    function=dict(name=call.name, arguments=json.dumps(call.arguments, ensure_ascii=False)))]))]
            return []
        for event in view.iter_events():
            if event.get('event') == 'text':
                for parsed in parser.feed(event['text']): yield from output(parsed)
        receipt = request.durable.result()
        answer = request.snapshot.result()
        self.cache.committed(branch, receipt)
        controller = self.adapter.snapshot_controller(receipt['snapshot'])
        length = controller.get('status') == 'active'
        for parsed in parser.finish('length' if length else 'stop'): yield from output(parsed)
        n = len(answer['tokens'])
        done = dict(finish='length' if length else 'tool_calls' if calls else 'stop',
                    prompt_tokens=len(plan['full']), completion_tokens=n, reused=reused)
        usage = dict(prompt_tokens=len(plan['full']), completion_tokens=n,
                     total_tokens=len(plan['full'])+n, prompt_tokens_details=dict(cached_tokens=reused))
        if self.trace:
            record = dict(model=self.model, dialect=dialect, prompt_tokens=len(plan['full']),
                          reused_tokens=reused, output_tokens=n, seconds=time.monotonic()-start,
                          finish=done['finish'], snapshot=receipt['snapshot'],
                          request=req, output=dict(content=''.join(text), reasoning=''.join(reasoning),
                          tools=[dict(name=c.name, arguments=c.arguments) for c in calls]))
            with self.trace_lock:
                with self.trace.open('a', encoding='utf-8') as stream: stream.write(json.dumps(record)+'\n')
        if assembler: yield from assembler.finish(done)
        else: yield chat({}, done['finish'], usage)


def make_server(frontend, host, port, key, shutdown):
    if host not in ('127.0.0.1', 'localhost', '::1') and not key:
        raise ValueError('A LAN bind requires an API key')
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args): pass
        def authorized(self):
            return not key or hmac.compare_digest(self.headers.get('Authorization', ''), 'Bearer '+key)
        def send_json(self, code, value):
            raw = json.dumps(value).encode(); self.send_response(code)
            self.send_header('Content-Type', 'application/json'); self.send_header('Content-Length', str(len(raw)))
            self.end_headers(); self.wfile.write(raw)
        def disconnected(self):
            if shutdown.is_set(): return True
            try:
                return bool(select.select([self.connection], [], [], 0)[0]) and not self.connection.recv(1, socket.MSG_PEEK)
            except OSError: return True
        def do_GET(self):
            if not self.authorized(): return self.send_json(401, dict(error='Unauthorized'))
            if self.path == '/v1/models':
                return self.send_json(200, dict(object='list', data=[dict(id=frontend.model, object='model',
                    created=0, owned_by='local', context_length=frontend.profile['execution']['capacity'])]))
            if self.path == '/health': return self.send_json(200, dict(status='ok', experimental=True))
            return self.send_json(404, dict(error='Not found'))
        def do_POST(self):
            if not self.authorized(): return self.send_json(401, dict(error='Unauthorized'))
            if self.path not in ('/v1/chat/completions', '/v1/responses'):
                return self.send_json(404, dict(error='Unsupported endpoint'))
            started = False
            try:
                size = int(self.headers.get('Content-Length', '0'))
                if not 0 < size <= 8*1024**2: raise ValueError('Request body exceeds bounds')
                req = json.loads(self.rfile.read(size))
                dialect = 'responses' if self.path.endswith('/responses') else 'chat'
                plan = frontend.prepare(req, dialect)
                events = frontend.events(plan, self.disconnected)
                if req.get('stream', False):
                    self.send_response(200); self.send_header('Content-Type', 'text/event-stream')
                    self.send_header('Cache-Control', 'no-store'); self.send_header('Connection', 'close')
                    self.end_headers(); self.close_connection = True; started = True
                    for event in events:
                        prefix = ('event: '+event['type']+'\n') if dialect == 'responses' else ''
                        self.wfile.write((prefix+'data: '+json.dumps(event)+'\n\n').encode()); self.wfile.flush()
                    if dialect == 'chat': self.wfile.write(b'data: [DONE]\n\n'); self.wfile.flush()
                elif dialect == 'responses': self.send_json(200, responses.collect(events))
                else:
                    chunks = list(events); message = dict(role='assistant', content=''); tools = []
                    for chunk in chunks:
                        delta = chunk['choices'][0]['delta']
                        message['content'] += delta.get('content', '')
                        if delta.get('reasoning_content'): message['reasoning_content'] = message.get('reasoning_content', '')+delta['reasoning_content']
                        for call in delta.get('tool_calls', []): tools.append({k:v for k,v in call.items() if k!='index'})
                    if tools: message['tool_calls'] = tools
                    result = dict(chunks[-1], object='chat.completion', choices=[dict(index=0, message=message,
                        finish_reason=chunks[-1]['choices'][0]['finish_reason'])])
                    self.send_json(200, result)
            except (BrokenPipeError, ConnectionResetError): pass
            except Exception as error:
                body = responses.error_body(str(error), kind='invalid_request_error' if isinstance(error, ValueError) else 'server_error')
                if not started: self.send_json(400 if isinstance(error, ValueError) else 500, body)
                else:
                    # A stream that fails must never masquerade as a successful completion.
                    event = dict(type='error', error=body['error'])
                    try: self.wfile.write(('event: error\ndata: '+json.dumps(event)+'\n\n').encode()); self.wfile.flush()
                    except OSError: pass
    server = ThreadingHTTPServer((host, port), Handler)
    server.daemon_threads = False
    return server


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--profile', type=Path, required=True); p.add_argument('--model-id', required=True)
    p.add_argument('--host', default='127.0.0.1'); p.add_argument('--port', type=int, default=8100)
    p.add_argument('--api-key-env', default='WEN_COMPAT_API_KEY')
    p.add_argument('--ram-gib', type=int, default=100); p.add_argument('--disk-gib', type=int, default=100)
    p.add_argument('--max-output', type=int, default=2048); p.add_argument('--trace', type=Path)
    p.add_argument('--temporary-parent', type=Path, required=True)
    a = p.parse_args()
    from context_time_travel.coordinator import ContextCoordinator
    from context_time_travel.pools import PoolConfig
    from context_time_travel.real import BranchingModel
    from context_time_travel.qwen35_backend import load_qwen35_backend
    from context_time_travel.strata_backend import verify_runtime_source_closure
    from serve.cow_adapter import CowAdapter
    from serve.cow_prompt_encoder import CowPromptEncoder
    from serve.frontend import ChatTemplate
    from tools.strata_tokenizer import Tokenizer
    from tools.gguf_reader import GGUFFile
    profile = json.loads(a.profile.read_text())
    verify_runtime_source_closure(profile, ROOT)
    os.environ.update(profile['execution']['environment'])
    key = os.environ.get(a.api_key_env)
    if a.host not in ('127.0.0.1', 'localhost', '::1') and not key: raise ValueError('LAN bind requires authentication')
    a.temporary_parent.mkdir(parents=True, exist_ok=True)
    if a.trace:
        a.trace.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(a.trace, os.O_CREAT|os.O_EXCL|os.O_WRONLY, 0o600); os.close(fd)
    shutdown = threading.Event()
    def stop(signum, frame): shutdown.set(); raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, stop)
    with tempfile.TemporaryDirectory(prefix='cow-cli-session-', dir=a.temporary_parent) as tmp:
        directory = Path(tmp); template_path = directory/'template.jinja'
        template_path.write_text(GGUFFile(Path(profile['model_path'])).metadata['tokenizer.chat_template'], encoding='utf-8')
        encoder = CowPromptEncoder(Tokenizer.from_gguf(profile['model_path']), ChatTemplate(template_path))
        store = directory/'store'
        def factory(gpu): return BranchingModel(store, load_qwen35_backend(profile, gpu_id=gpu), profile)
        config = PoolConfig(gpus=profile['execution']['GPU'], ram_bytes=a.ram_gib*1024**3,
                            disk_bytes=a.disk_gib*1024**3, min_available_bytes=8*1024**3)
        coordinator = ContextCoordinator(store, config, factory)
        server = None
        try:
            coordinator.ready()
            frontend = Frontend(CowAdapter(coordinator), encoder, profile, a.model_id, a.max_output, a.trace)
            server = make_server(frontend, a.host, a.port, key, shutdown)
            print(json.dumps(dict(event='ready', model=a.model_id, address=server.server_address,
                                  ephemeral_store=str(store), context=profile['execution']['capacity'])), flush=True)
            server.serve_forever(poll_interval=0.2)
        except KeyboardInterrupt: pass
        finally:
            shutdown.set()
            if server is not None: server.server_close()
            coordinator.close()
    print(json.dumps(dict(event='closed', temporary_cache_removed=True)), flush=True)


if __name__ == '__main__': main()
