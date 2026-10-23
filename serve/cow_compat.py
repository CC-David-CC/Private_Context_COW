"""Opt-in standard CLI APIs over real COW snapshots; text-only compatibility."""
import hashlib
import hmac
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import re
import select
import socket
import threading
import time
import uuid

from .cow_compat_cache import CachedCow, SessionBudgets
from serve import responses
from serve.frontend import openai_to_messages, OutputParser
from serve.cow_text_adapter import CowTextRequest
from serve.server import Service


class Frontend:
    def __init__(self, adapter, encoder, profile, model, max_output=2048, trace=None, *, thinking=False):
        if type(thinking) is not bool:raise ValueError('Explicit thinking mode required')
        self.thinking=thinking
        self.adapter, self.encoder, self.profile, self.model = adapter, encoder, profile, model
        renderer = hashlib.sha256(encoder.template.source.encode()).hexdigest()
        self.cache = CachedCow(adapter, (profile['model_contract_id'],
                                       profile['execution_profile_id'], renderer) if not thinking else (profile['model_contract_id'],profile['execution_profile_id'],renderer,'thinking-on'))
        self.max_output, self.trace = max_output, trace
        self.trace_lock = threading.Lock()
        self.sessions = SessionBudgets()

    def prepare(self, req, dialect):
        if not isinstance(req, dict): raise ValueError('Request must be a JSON object')
        if req.get('model', self.model) != self.model: raise ValueError('Unknown model')
        if type(req.get('n', 1)) is not int or req.get('n', 1) != 1: raise ValueError('Only n=1 is supported')
        if type(req.get('stream', False)) is not bool: raise ValueError('stream must be boolean')
        if dialect == 'responses':
            responses.check_request(req)
            if req.get('store'): raise ValueError('Stored Responses are unsupported; send full history with store=false')
            messages = responses.input_messages(req)
            tools, names, skipped = responses.request_tools(req)
            if skipped: raise ValueError('Hosted tools are unsupported: '+', '.join(skipped))
            fmt = responses.text_format(req)
            if fmt and fmt.get('type') != 'text': raise ValueError('Structured final-answer format is not qualified')
        else:
            messages, tools, _ = openai_to_messages(req)
            names = {}
            fmt = req.get('response_format') or {}
            if not isinstance(fmt,dict) or fmt.get('type', 'text') != 'text':
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
        for field in ('frequency_penalty','presence_penalty'):
            if req.get(field): raise ValueError(field+' is unsupported')
        if 'enable_thinking' in req and (type(req['enable_thinking']) is not bool or req['enable_thinking']!=self.thinking):
            raise ValueError('Request thinking mode differs from the owner-configured profile')
        kwargs = dict(enable_thinking=self.thinking)
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
        parser = OutputParser(thinking=self.thinking, tools=plan['tools'], stream_tools=False)
        text, reasoning, calls, raw = [], [], [], []
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
                if self.trace:raw.append(event['text'])
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
                          thinking=self.thinking,raw_text=''.join(raw),raw_text_sha256=hashlib.sha256(''.join(raw).encode()).hexdigest(),
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
        def route(self):
            match = re.fullmatch(r'/sessions/([a-f0-9]{32})(/v1/[^?]+)', self.path)
            return (match[2], match[1]) if match else (self.path, None)
        def do_GET(self):
            if not self.authorized(): return self.send_json(401, dict(error='Unauthorized'))
            path, _ = self.route()
            if path == '/v1/models':
                return self.send_json(200, dict(object='list', data=[dict(id=frontend.model, object='model',
                    created=0, owned_by='local', context_length=frontend.profile['execution']['capacity'])]))
            if self.path == '/health': return self.send_json(200, dict(status='ok', service='context-cow', compatibility='text-chat-responses-v1'))
            return self.send_json(404, dict(error='Not found'))
        def do_POST(self):
            if not self.authorized(): return self.send_json(401, dict(error='Unauthorized'))
            path, session = self.route()
            if path not in ('/v1/chat/completions', '/v1/responses'):
                return self.send_json(404, dict(error='Unsupported endpoint'))
            started = False
            try:
                size = int(self.headers.get('Content-Length', '0'))
                if not 0 < size <= 8*1024**2: raise ValueError('Request body exceeds bounds')
                req = json.loads(self.rfile.read(size))
                dialect = 'responses' if path.endswith('/responses') else 'chat'
                plan = frontend.prepare(req, dialect)
                frontend.sessions.admit(session)
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

