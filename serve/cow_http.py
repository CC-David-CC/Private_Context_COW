"""Opt-in native COW token-history transport; ordinary chat serving is separate.

Token events are provisional. Only complete acknowledges durable publication.
The caller renders the complete prompt with the pinned tokenizer/template.
POST /cow/cancel requests transient cancellation for an active streamed operation;
inspect GET /cow/operation afterward because publication may win the race.
Cancellation never publishes provisional token output.
"""
import argparse
from concurrent.futures import CancelledError
from dataclasses import asdict, is_dataclass
import hmac
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
import os
from pathlib import Path
import queue
import re
import select
import socket
import threading
import tempfile
import uuid
from urllib.parse import parse_qs, unquote, urlsplit


_SOCKET_ERRORS = (BrokenPipeError, ConnectionResetError, TimeoutError, OSError)


class _ClientDisconnected(Exception):
    """Internal control flow for failures at an HTTP socket I/O site."""


def plain(value):
    return asdict(value) if is_dataclass(value) else value


def model_text_components(model_path,template_kwargs=None,*,model_contract_id=None,mixed_inputs=False):
    """Load only tokenizer/template metadata; computation stays on its worker."""
    from tools.strata_tokenizer import Tokenizer
    from tools.gguf_reader import GGUFFile
    from serve.frontend import ChatTemplate
    from serve.cow_prompt_encoder import CowPromptEncoder
    tokenizer=Tokenizer.from_gguf(model_path)
    source=GGUFFile(Path(model_path)).metadata.get('tokenizer.chat_template')
    if not isinstance(source,str) or not source: raise ValueError('Model has no chat template')
    with tempfile.TemporaryDirectory(prefix='cow-template-') as directory:
        path=Path(directory)/'chat_template.jinja';path.write_text(source,encoding='utf-8')
        template=ChatTemplate(path)
    metadata=GGUFFile(Path(model_path)).metadata
    if metadata.get('general.architecture') in ('qwen35','wen1'):
        from serve.cow_qwen35_prompt import Qwen35PromptEncoder
        encoder=Qwen35PromptEncoder(tokenizer,template,template_kwargs=template_kwargs)
    else:
        encoder=CowPromptEncoder(tokenizer,template,template_kwargs=template_kwargs)
    renderer_id=None
    if model_contract_id is not None:
        from serve.cow_renderer import renderer_identity
        renderer_id=renderer_identity(tokenizer,source,encoder.template_kwargs,
            model_contract_id=model_contract_id)
    mixed_renderer=None
    if mixed_inputs:
        from serve.cow_vision import CowVisionPromptEncoder
        mixed_renderer=CowVisionPromptEncoder(encoder,renderer_id=renderer_id)
    return dict(tokenizer=tokenizer,render=encoder.render,encode_rendered=encoder.encode_rendered,
        append_user_suffix=encoder.append_user_suffix,
        append_initial_user_suffix=encoder.append_initial_user_suffix,
        initial_root_mode=encoder.initial_root_mode,renderer_id=renderer_id,mixed_renderer=mixed_renderer)


def model_text_adapter(token_adapter,model_path,template_kwargs=None,*,model_contract_id=None,
                       grammar_registry=None,mixed_inputs=False,vision_preparer=None):
    from serve.cow_text_adapter import CowTextAdapter
    return CowTextAdapter(token_adapter,**model_text_components(model_path,template_kwargs,
        model_contract_id=model_contract_id,mixed_inputs=mixed_inputs),grammar_registry=grammar_registry,
        vision_preparer=vision_preparer)


def vision_preparer_from_config(store,mixed_renderer,profile,path):
    """Validate an administrator-owned encoder configuration before GPU startup."""
    with Path(path).open('rb') as stream:raw=stream.read(64*1024+1)
    if len(raw)>64*1024:raise ValueError('Image producer configuration exceeds 64 KiB')
    config=json.loads(raw)
    required={'schema','executable','projector','pinned_sha256','embedding_width','original_bytes_limit'}
    optional={'threads','max_rows','flash_attn','timeout'}
    execution=profile['execution']
    if (type(config) is not dict or not required<=set(config) or set(config)-required-optional
            or type(config.get('schema')) is not int or config['schema']!=1
            or type(config.get('original_bytes_limit')) is not int or config['original_bytes_limit']<=0
            or type(config.get('embedding_width')) is not int or config['embedding_width']<=0):
        raise ValueError('Invalid image producer configuration')
    if (mixed_renderer is None or execution.get('resume_contract')!='identical-mixed-native-operators-v1'
            or config['embedding_width']!=execution.get('embedding_width')):
        raise ValueError('Image producer differs from the supported mixed worker profile')
    from serve.cow_vision import CowImageNormalizer
    from serve.cow_vision_encoder import CowNativeVisionEncoder
    from serve.cow_vision_preparer import CowVisionPreparer
    encoder=CowNativeVisionEncoder(executable=config['executable'],projector=config['projector'],
        model=profile['model_path'],pinned_sha256=config['pinned_sha256'],width=config['embedding_width'],
        **{key:config[key] for key in optional if key in config})
    return CowVisionPreparer(store,CowImageNormalizer(),mixed_renderer,encoder,
        original_bytes_limit=config['original_bytes_limit'])


def trace_request_source(trace,identity,adapter,path,data):
    """Record a request's canonical source boundary on the server monotonic clock."""
    if trace is None or path not in ('/cow/text/answer','/cow/text/ask','/cow/text/resume',
                                     '/cow/text/append-user','/cow/text/generate'):
        return None
    branch=data.get('branch')
    if not isinstance(branch,str) or not isinstance(identity,(list,tuple)) or len(identity)!=2:
        return None
    source=plain(adapter.head(branch))
    snapshot=source.get('tip') if isinstance(source,dict) else None
    if not isinstance(snapshot,str):
        return None
    key=(*identity,snapshot)
    fields=dict(branch=branch,snapshot=snapshot,revision=source.get('revision'),
                operation_id=data.get('operation_id'),path=path)
    trace.record('http_request_received',key,**fields)
    return dict(key=key,**fields)


def validate_pool_spill_reload_trace(document,*,model_contract_id,execution_profile_id,
                                    branch,source_snapshot,operation_id):
    """Require one causally ordered, request-bound eviction/read/activation chain."""
    if not isinstance(document,dict) or document.get('overflowed'):
        raise ValueError('Pool trace is missing or overflowed')
    events=document.get('events')
    if not isinstance(events,list) or document.get('event_count')!=len(events):
        raise ValueError('Pool trace event count is incomplete')
    if any(type(row.get('sequence')) is not int or type(row.get('monotonic_ns')) is not int
           for row in events):
        raise ValueError('Pool trace lacks ordered server-clock timestamps')
    if any(row.get('process_id')!=document.get('process_id')
           or row.get('phase_id')!=document.get('phase_id') for row in events):
        raise ValueError('Pool trace mixes process or phase identities')
    if any(a['sequence']>=b['sequence'] or a['monotonic_ns']>b['monotonic_ns']
           for a,b in zip(events,events[1:])):
        raise ValueError('Pool trace ordering is invalid')
    key=[model_contract_id,execution_profile_id,source_snapshot]
    keyed=[row for row in events if row.get('key')==key]
    accepted=[row for row in keyed if row.get('event')=='http_request_accepted'
              and row.get('operation_id')==operation_id and row.get('branch')==branch
              and row.get('snapshot')==source_snapshot and row.get('path') in
              ('/cow/text/ask','/cow/text/append-user','/cow/text/resume')]
    durable=[row for row in keyed if row.get('event')=='http_request_durable'
             and row.get('operation_id')==operation_id and row.get('branch')==branch
             and row.get('snapshot')==source_snapshot]
    acquired=[row for row in keyed if row.get('event')=='request_source_acquired'
              and row.get('operation_id')==operation_id and row.get('branch')==branch
              and row.get('source_snapshot')==source_snapshot and row.get('source_kind')=='shared_pool']
    if len(accepted)!=1 or len(durable)!=1 or len(acquired)!=1:
        raise ValueError('Pool trace lacks a unique accepted, acquired, and durable request')
    accepted,durable,acquired=accepted[0],durable[0],acquired[0]
    received=[row for row in keyed if row.get('event')=='http_request_received'
              and row.get('operation_id')==operation_id and row.get('branch')==branch
              and row.get('snapshot')==source_snapshot and row.get('path')==accepted['path']]
    if len(received)!=1:
        raise ValueError('Pool trace lacks the matching received request')
    received=received[0]
    evictions=[row for row in keyed if row.get('event')=='ram_clean_eviction'
               and type(row.get('freed_bytes')) is int and row['freed_bytes']>0]
    successes=[row for row in keyed if row.get('event')=='disk_preload_succeeded'
               and row.get('success') is True and type(row.get('bytes')) is int and row['bytes']>0]
    for success in successes:
        starts=[row for row in keyed if row.get('event')=='disk_preload_started'
                and row.get('load_id')==success.get('load_id') and row['sequence']<success['sequence']]
        if not starts: continue
        start=starts[-1]
        prior=[row for row in evictions if row['sequence']<start['sequence']]
        if not prior: continue
        eviction=prior[-1]
        if any(row.get('event')=='ram_clean_eviction' and row.get('key')==key
               and success['sequence']<row['sequence']<acquired['sequence'] for row in events):
            continue
        # Read-ahead may complete before HTTP receives the ask; an on-demand
        # load may complete after it. The common causal spine ends at the exact
        # request-bound pool pin and terminal durable acknowledgement.
        if not (eviction['sequence']<start['sequence']<success['sequence']<acquired['sequence']
                and received['sequence']<acquired['sequence']<durable['sequence']
                and received['sequence']<accepted['sequence']<durable['sequence']):
            continue
        return {'key':key,'eviction_sequence':eviction['sequence'],'freed_bytes':eviction['freed_bytes'],
            'load_id':success['load_id'],'load_started_sequence':start['sequence'],
            'load_succeeded_sequence':success['sequence'],'loaded_bytes':success['bytes'],
            'request_id':operation_id,'request_received_sequence':received['sequence'],
            'request_acquired_sequence':acquired['sequence'],'request_durable_sequence':durable['sequence'],
            'result_snapshot':durable.get('result_snapshot'),'source_kind':acquired['source_kind']}
    raise ValueError('No positive freed-eviction → successful disk-read → matching request chain exists')


def persist_pool_trace(path,trace,*,service_ready,coordinator_closed):
    """Publish a bounded local trace only after a ready service closes cleanly."""
    if not service_ready or not coordinator_closed:
        return False
    if trace is None or not callable(getattr(trace,'snapshot',None)):
        raise ValueError('A complete pool trace is required')
    document=trace.snapshot()
    if document.get('overflowed'):
        raise ValueError('Pool trace overflowed; refusing incomplete evidence')
    target=validate_pool_trace_target(path)
    parent=target.parent
    raw=(json.dumps(document,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()
    temporary=parent / ('.'+target.name+'.'+uuid.uuid4().hex+'.tmp')
    descriptor=os.open(temporary,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
    try:
        with os.fdopen(descriptor,'wb') as stream:
            stream.write(raw);stream.flush();os.fsync(stream.fileno())
        os.link(temporary,target)
        directory=os.open(parent,os.O_RDONLY)
        try: os.fsync(directory)
        finally: os.close(directory)
    finally:
        try: temporary.unlink()
        except FileNotFoundError: pass
    return True


def validate_pool_trace_target(path):
    target=Path(path).absolute()
    parent=target.parent
    info=parent.stat()
    if (not parent.is_dir() or parent.is_symlink() or info.st_mode & 0o077
            or (hasattr(os,'getuid') and info.st_uid!=os.getuid())):
        raise PermissionError('Pool trace parent must be an existing private directory')
    if target.exists() or target.is_symlink():
        raise FileExistsError(target)
    return target


def make_server(adapter,host='127.0.0.1',port=0,api_key=None,*,text_adapter=None,workspace=None,
                transition_trace=None,trace_identity=None,grammar_catalog=None,connection_info=None):
    if host not in ('127.0.0.1','localhost','::1') and not api_key:
        raise ValueError('A public bind requires an API key')
    if grammar_catalog is not None:
        if not api_key or text_adapter is None:
            raise ValueError('Grammar registration requires an API key and text adapter')
        from serve.cow_grammar_disk import DurableGrammarCatalog
        from serve.cow_grammar_router import GrammarRouter
        if not isinstance(grammar_catalog,DurableGrammarCatalog):
            raise TypeError('HTTP grammar admission requires a durable catalog')
        router=GrammarRouter(grammar_catalog,startup=adapter.grammar_registry)
        adapter.grammar_registry=router
        text_adapter.grammar_registry=router
    # Startup supplies the already validated active profile. Never publish the
    # profile itself: it also carries host paths, build inputs and model files.
    description = None
    if connection_info is not None:
        keys = {'model_contract_id', 'execution_profile_id', 'eos_tokens'}
        if type(connection_info) is not dict or set(connection_info) != keys:
            raise ValueError('Connection info requires only model/profile identities and EOS IDs')
        for key in keys - {'eos_tokens'}:
            if type(connection_info[key]) is not str or re.fullmatch('[0-9a-f]{64}', connection_info[key]) is None:
                raise ValueError('Connection info requires SHA-256 identities')
        eos = connection_info['eos_tokens']
        if (type(eos) is not list or not 1 <= len(eos) <= 256
                or any(type(token) is not int or not 0 <= token < 2**31 for token in eos)
                or len(set(eos)) != len(eos)):
            raise ValueError('Connection info requires actual distinct EOS token IDs')
        description = dict(schema=1, service='context-cow',
            model_contract_id=connection_info['model_contract_id'],
            execution_profile_id=connection_info['execution_profile_id'], eos_tokens=list(eos),
            capabilities=dict(text=text_adapter is not None, inline_grammars=grammar_catalog is not None))
    active_stops={}
    active_stops_lock=threading.Lock()
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args): pass
        def json(self,code,value):
            raw=json.dumps(plain(value),allow_nan=False).encode()
            try:
                self.send_response(code);self.send_header('Content-Type','application/json')
                self.send_header('Content-Length',str(len(raw)));self.end_headers();self.wfile.write(raw)
            except _SOCKET_ERRORS:
                return False
            return True
        def authorized(self):
            given=self.headers.get('Authorization','')
            if api_key and not hmac.compare_digest(given,'Bearer '+api_key):
                self.json(401,{'error':'Missing or wrong API key'});return False
            origin=self.headers.get('Origin')
            if origin and urlsplit(origin).netloc!=self.headers.get('Host'):
                self.json(403,{'error':'Foreign Origin rejected'});return False
            return True
        def body(self,*,grammar=False):
            size=int(self.headers.get('Content-Length','0'))
            if grammar and size>4*1024*1024:raise ValueError('Grammar body exceeds4MiB')
            if not 0<size<=16*1024*1024: raise ValueError('Body must be 1 byte to 16 MiB')
            try: raw=self.rfile.read(size)
            except _SOCKET_ERRORS as error: raise _ClientDisconnected from error
            if len(raw)!=size: raise ValueError('Incomplete request body')
            if grammar:
                from serve.cow_grammar import _unique_object,_reject_constant
                value=json.loads(raw,object_pairs_hook=_unique_object,parse_constant=_reject_constant)
            else:value=json.loads(raw)
            if not isinstance(value,dict): raise ValueError('Send a JSON object')
            return value
        def client_disconnected(self):
            try:
                ready,_,_=select.select([self.connection],[],[],0)
                return bool(ready) and self.connection.recv(1,socket.MSG_PEEK)==b''
            except _SOCKET_ERRORS:
                return True
        def do_GET(self):
            path=urlsplit(self.path).path
            if workspace is not None:
                if path=='/health':
                    return self.json(200,dict(status='ok',service='context-cow',
                        model='Context COW',images=False,api_key=bool(api_key),
                        capabilities=dict(greedy=True,native_cow=True,tools=False)))
                if path in ('/','/index.html') or path.startswith(('/web/','/fonts/')):
                    web=Path(__file__).resolve().parent/'web'
                    # Shared styles request Outfit under /fonts/, as ordinary serving does.
                    relative=('index.html' if path in ('/','/index.html') else
                              'fonts/'+unquote(path[7:]) if path.startswith('/fonts/') else unquote(path[5:]))
                    try:
                        target=(web/relative).resolve()
                        if not target.is_relative_to(web.resolve()) or not target.is_file():
                            return self.json(404,dict(error='Asset not found'))
                        raw=target.read_bytes()
                    except (OSError,ValueError):
                        return self.json(404,dict(error='Asset not found'))
                    self.send_response(200)
                    self.send_header('Content-Type',mimetypes.guess_type(str(target))[0] or 'application/octet-stream')
                    self.send_header('Content-Length',str(len(raw)));self.end_headers()
                    self.wfile.write(raw);return
            if not self.authorized(): return
            if path.startswith('/cow/cache/checkpoint/'):
                enabled=getattr(adapter,'cache_transfer_enabled',None)
                if not callable(enabled) or not enabled():return self.json(404,dict(error='Cache transfers are disabled'))
                snapshot=path[len('/cow/cache/checkpoint/'):]
                began=False
                try:
                    query=parse_qs(urlsplit(self.path).query,keep_blank_values=True)
                    if (re.fullmatch('[0-9a-f]{64}',snapshot) is None
                            or set(query)!={'model_contract_id','execution_profile_id'}
                            or any(len(v)!=1 or re.fullmatch('[0-9a-f]{64}',v[0]) is None for v in query.values())):
                        raise ValueError('Transfer requires exact snapshot, model and execution identities')
                    from context_time_travel.store import _json_bytes,NotFound
                    from context_time_travel.pools import PoolFull
                    import struct
                    with adapter.export_checkpoint(snapshot,query['model_contract_id'][0],query['execution_profile_id'][0]) as offered:
                        raw=_json_bytes(offered.envelope)
                        self.connection.settimeout(10)
                        began=True
                        self.send_response(200);self.send_header('Content-Type','application/x-cow-checkpoint')
                        self.send_header('Content-Length',str(4+len(raw)+offered.payload_bytes));self.end_headers()
                        self.wfile.write(struct.pack('!I',len(raw)));self.wfile.write(raw)
                        while True:
                            offered.check();chunk=offered.stream.read(offered.chunk_bytes)
                            if not chunk:break
                            self.wfile.write(chunk)
                    return
                except _SOCKET_ERRORS:return
                except Exception as error:
                    if began:
                        self.close_connection=True;return
                    from context_time_travel.store import NotFound
                    from context_time_travel.pools import PoolFull
                    # A vanished optional manifest is a stale inventory hint.
                    absent=isinstance(error,NotFound)
                    return self.json(404 if absent else 429 if isinstance(error,PoolFull) else
                                     400 if isinstance(error,ValueError) else 409,dict(error=str(error)))
            if path == '/cow/info':
                if urlsplit(self.path).query:
                    return self.json(400, dict(error='Connection info takes no query parameters'))
                if description is None:
                    return self.json(503, dict(error='Connection discovery is not configured'))
                return self.json(200, description)
            if path == '/cow/capabilities':
                if urlsplit(self.path).query:
                    return self.json(400, dict(error='Capabilities take no query parameters'))
                if description is None:
                    return self.json(503, dict(error='Connection discovery is not configured'))
                try:
                    from context_time_travel.capabilities import feature
                    report=adapter.capabilities()
                    report['features'].update(text=feature(text_adapter is not None),
                        inline_grammars=feature(grammar_catalog is not None))
                    return self.json(200,dict(schema=1,service='context-cow',
                        model_contract_id=description['model_contract_id'],
                        execution_profile_id=description['execution_profile_id'],**report))
                except Exception as error:
                    return self.json(503,dict(error='Capability discovery unavailable: '+str(error)))
            if path.startswith('/cow/evaluation/'):
                identity = path[len('/cow/evaluation/'):]
                if not identity or '/' in identity or urlsplit(self.path).query:
                    return self.json(404, {'error':'Not found'})
                try:
                    return self.json(200, {'evaluation_id':identity,
                                           'record':adapter.evaluation(identity)})
                except Exception as error:
                    from context_time_travel.store import NotFound
                    return self.json(404 if isinstance(error,NotFound) else 409,{'error':str(error)})
            if path=='/cow/cache/inventory':
                try:
                    from context_time_travel.cache_inventory import RecoveryRequired
                    query=parse_qs(urlsplit(self.path).query,keep_blank_values=True)
                    if (set(query)-{'epoch','since','page_size','cursor'}
                            or any(len(v)!=1 for v in query.values())
                            or ('epoch' in query)!=('since' in query)):
                        raise ValueError('Invalid inventory query')
                    cursor={k:(int(v[0]) if k in ('since','page_size') else v[0]) for k,v in query.items()}
                    return self.json(200,adapter.cache_inventory(**cursor))
                except RecoveryRequired as error:return self.json(409,dict(error=str(error)))
                except ValueError as error:return self.json(400,dict(error=str(error)))
                except Exception as error:return self.json(409,dict(error=str(error)))
            if path=='/cow/status':
                try:
                    query=parse_qs(urlsplit(self.path).query,keep_blank_values=True)
                    if query: raise ValueError('Status takes no query parameters')
                except ValueError as error:return self.json(400,dict(error=str(error)))
                try:return self.json(200,adapter.status())
                except Exception as error:return self.json(409,dict(error=str(error)))
            if path=='/cow/operation':
                try:
                    query=parse_qs(urlsplit(self.path).query,keep_blank_values=True)
                    if set(query)!={'operation'} or len(query['operation'])!=1:
                        raise ValueError('Use exactly one operation query parameter')
                    operation=query['operation'][0]
                    from context_time_travel.store import NotFound
                    try:
                        status=adapter.operation_status(operation)
                    except NotFound:
                        if text_adapter is None: raise
                        status=text_adapter.operation_status(operation)
                    return self.json(200,status)
                except (KeyError,ValueError) as error:return self.json(400,dict(error=str(error)))
                except Exception as error:
                    from context_time_travel.store import NotFound
                    return self.json(404 if isinstance(error,NotFound) else 409,dict(error=str(error)))
            if path=='/cow/text/retry-info':
                if text_adapter is None: return self.json(404,{'error':'Text adapter not configured'})
                try:
                    query=parse_qs(urlsplit(self.path).query,keep_blank_values=True)
                    if 'operation' not in query or set(query)-{'operation','settings','seed','text_stops'}:
                        raise ValueError('Use an operation and optional settings, seed, or text-stop query parameter')
                    if (len(query['operation'])!=1 or ('settings' in query and len(query['settings'])!=1)
                            or ('seed' in query and len(query['seed'])!=1)
                            or ('text_stops' in query and len(query['text_stops'])!=1)):
                        raise ValueError('Use exactly one value for each retry-info query parameter')
                    settings=None
                    if 'settings' in query:
                        settings=json.loads(query['settings'][0])
                    seed=None
                    if 'seed' in query:
                        raw_seed=query['seed'][0]
                        if not raw_seed.isdecimal() or (len(raw_seed)>1 and raw_seed[0]=='0'):
                            raise ValueError('Retry seed must be a canonical unsigned decimal integer')
                        seed=int(raw_seed)
                    text_stops=None
                    if 'text_stops' in query:
                        text_stops=json.loads(query['text_stops'][0])
                    retry_options={} if seed is None else {'seed':seed}
                    return self.json(200,text_adapter.retry_info(query['operation'][0],settings,
                        text_stops=text_stops,**retry_options))
                except (KeyError,ValueError) as error:return self.json(400,dict(error=str(error)))
                except Exception as error:
                    from context_time_travel.store import NotFound
                    return self.json(404 if isinstance(error,NotFound) else 409,dict(error=str(error)))
            if workspace is not None and path.startswith('/cow/workspace/'):
                try:
                    query=parse_qs(urlsplit(self.path).query,keep_blank_values=True)
                    if path=='/cow/workspace/chats':
                        if query: raise ValueError('Chat listing takes no query parameters')
                        return self.json(200,dict(chats=workspace.list_chats()))
                    if path=='/cow/workspace/branches':
                        if set(query)!={'chat'} or len(query['chat'])!=1:
                            raise ValueError('Use exactly one chat query parameter')
                        branches=workspace.list_branches(query['chat'][0])
                        for branch in branches: branch['pending_request']=adapter.pending_request(branch['id'])
                        return self.json(200,dict(branches=branches))
                    if path=='/cow/workspace/history':
                        if set(query)!={'branch'} or len(query['branch'])!=1:
                            raise ValueError('Use exactly one branch query parameter')
                        history=workspace.history(query['branch'][0])
                        history['pending_request']=adapter.pending_request(history['id'])
                        return self.json(200,history)
                    if path=='/cow/workspace/graph':
                        if set(query)-{'chat','previous'} or 'chat' not in query or len(query['chat'])!=1:
                            raise ValueError('Use one chat and optional previous query parameter')
                        previous=query.get('previous',['false'])
                        if len(previous)!=1 or previous[0] not in ('true','false'):
                            raise ValueError('previous must be true or false')
                        graph=workspace.graph(query['chat'][0],
                            include_previous=previous[0]=='true')
                        for branch in graph['branches']:
                            branch['pending_request']=adapter.pending_request(branch['id'])
                        return self.json(200,graph)
                    return self.json(404,dict(error='Not found'))
                except (KeyError,ValueError) as error:return self.json(400,dict(error=str(error)))
                except Exception as error:
                    from context_time_travel.store import NotFound
                    return self.json(404 if isinstance(error,NotFound) else 409,dict(error=str(error)))
            if path in ('/cow/namespaces','/cow/presentation'):
                try:
                    query=parse_qs(urlsplit(self.path).query,keep_blank_values=True)
                    if path=='/cow/namespaces':
                        if query: raise ValueError('Namespace listing takes no query parameters')
                        return self.json(200,{'namespaces':[plain(value) for value in adapter.namespaces()]})
                    if set(query)!={'namespace'} or len(query['namespace'])!=1:
                        raise ValueError('Use exactly one namespace query parameter')
                    return self.json(200,adapter.presentation_catalog(query['namespace'][0]))
                except (KeyError,ValueError) as error:return self.json(400,{'error':str(error)})
                except Exception as error:return self.json(409,{'error':str(error)})
            if urlsplit(self.path).path=='/cow/branches':
                try:
                    query=parse_qs(urlsplit(self.path).query,keep_blank_values=True)
                    if set(query)-{'root'} or ('root' in query and len(query['root'])!=1):
                        raise ValueError('Use at most one root query parameter')
                    root=query['root'][0] if 'root' in query else None
                    return self.json(200,{'branches':[plain(head) for head in adapter.branches(root)]})
                except (KeyError,ValueError) as error: return self.json(400,{'error':str(error)})
                except Exception as error: return self.json(409,{'error':str(error)})
            if urlsplit(self.path).path!='/cow/head': return self.json(404,{'error':'Not found'})
            try:
                branch=parse_qs(urlsplit(self.path).query)['branch'][0]
                head=plain(adapter.head(branch))
                namespace=getattr(adapter,'namespace',None)
                if callable(namespace):head=dict(head,namespace_id=namespace(branch))
                self.json(200,head)
            except (KeyError,ValueError) as error: self.json(400,{'error':str(error)})
        def do_POST(self):
            if not self.authorized(): return
            try: self.connection.settimeout(5)
            except _SOCKET_ERRORS: return
            if urlsplit(self.path).path=='/cow/images':
                return self.image_upload()
            try:
                path=urlsplit(self.path).path
                if path=='/cow/grammars' and grammar_catalog is None:
                    return self.json(404,{'error':'Grammar registration not enabled'})
                data=self.body(grammar=path=='/cow/grammars')
                if path in ('/cow/cancel','/cow/pause'):
                    if set(data)!={'operation_id'}:
                        raise ValueError('Request control requires only operation_id')
                    operation=data['operation_id']
                    if (not isinstance(operation,str)
                            or re.fullmatch('[A-Za-z][A-Za-z0-9_-]{0,127}',operation) is None):
                        raise ValueError('Request-control operation ID is invalid')
                    field = 'pause_requested' if path=='/cow/pause' else 'cancel_requested'
                    with active_stops_lock:
                        active=active_stops.get(operation)
                        if active is not None and not active[1].durable.done():
                            if path=='/cow/pause':
                                handle=getattr(active[1],'request',active[1])
                                pause=getattr(handle,'request_pause',None)
                                if not callable(pause):
                                    return self.json(409,{'error':'Request does not support consumed-token pause'})
                                requested=pause()
                            else:
                                active[0].set();requested=True
                            result=dict(operation_id=operation,active=True,**{field:requested})
                        else:
                            result=dict(operation_id=operation,active=False,**{field:False})
                    return self.json(200,result)
                fields={
                    '/cow/grammars':{'schema','states','operation_id'},
                    '/cow/namespace':{'title','operation_id'},
                    '/cow/publication/retry':{'operation_id'},
                    '/cow/reseed':{'branch','revision','operation_id','seed'},
                    '/cow/text/retry':{'from_operation','operation_id','branch','revision','settings','text_stops','stream','seed'},
                    '/cow/generate':{'branch','rendered_token_ids','revision','operation_id','max_tokens','settings','stream'},
                    '/cow/create':{'branch','token_ids','operation_id','stream'},
                    '/cow/fork':{'name','snapshot','source_branch','operation_id'},
                    '/cow/rewind':{'branch','revision','snapshot','operation_id'},
                    '/cow/evaluate':{'snapshot_id','operation_id','specification','fixtures'},
                    '/cow/evaluation/select':{'evaluation_ids','mandatory'},
                    '/cow/evaluation/adopt':{'branch','expected_revision','evaluation_ids','mandatory','operation_id','presentation_source_operation'},
                    '/cow/text/create':{'branch','messages','operation_id','stream'},
                    '/cow/text/context':{'branch','messages','namespace_id','operation_id','stream'},
                    '/cow/text/prepare':{'branch','messages','namespace_id','operation_id','stream'},
                    '/cow/text/answer':{'branch','revision','operation_id','max_tokens','settings','grammar_state','seed','text_stops','stream'},
                    '/cow/text/ask':{'branch','content','revision','operation_id','max_tokens','settings','grammar_state','seed','text_stops','stream'},
                    '/cow/text/generate':{'branch','messages','revision','operation_id','max_tokens','settings','grammar_state','seed','text_stops','stream'},
                    '/cow/text/resume':{'branch','revision','operation_id','max_tokens','settings','grammar_state','text_stops','stream'},
                    '/cow/text/append-user':{'branch','content','revision','operation_id','max_tokens','settings','grammar_state','seed','text_stops','stream'},
                }
                if path.startswith('/cow/text/') and text_adapter is None:
                    return self.json(404,{'error':'Text adapter not configured'})
                if path in fields and set(data)-fields[path]:
                    raise ValueError('Unsupported request fields: '+', '.join(sorted(set(data)-fields[path])))
                if path=='/cow/grammars':
                    if grammar_catalog is None:
                        return self.json(404,{'error':'Grammar registration not enabled'})
                    if 'operation_id' in data:
                        config={key:value for key,value in data.items() if key!='operation_id'}
                        return self.json(200,grammar_catalog.register_operation(config,data['operation_id']))
                    return self.json(200,grammar_catalog.register(data))
                if path=='/cow/namespace':
                    return self.json(200,adapter.create_namespace(data['title'],data['operation_id']))
                if path=='/cow/publication/retry':
                    receipt=adapter.retry_publication(data['operation_id']).result(timeout=600)
                    return self.json(200,dict(event='complete',receipt=receipt))
                if path=='/cow/fork':
                    options={key:data[key] for key in ('source_branch','operation_id') if key in data}
                    return self.json(200,adapter.fork(data['name'],data['snapshot'],**options))
                if path=='/cow/rewind':
                    return self.json(200,adapter.rewind(data['branch'],data['revision'],data['snapshot'],data['operation_id']))
                if path=='/cow/reseed':
                    if (not isinstance(data.get('branch'),str) or type(data.get('revision')) is not int
                            or not isinstance(data.get('operation_id'),str) or type(data.get('seed')) is not int):
                        raise ValueError('Reseed requires branch, integer revision, operation ID, and unsigned seed')
                    reseed=(text_adapter.reseed(data['branch'],data['seed'],data['revision'],
                               data['operation_id']) if text_adapter is not None else
                            adapter.reseed(data['branch'],data['seed'],data['revision'],
                                data['operation_id']))
                    receipt=reseed.durable.result(timeout=600)
                    return self.json(200,receipt['branch'])
                if path=='/cow/evaluate':
                    fixtures = data['fixtures']
                    if not isinstance(fixtures,dict) or len(fixtures)>64:
                        raise ValueError('Fixtures must be a named hexadecimal byte mapping')
                    specification = data['specification']
                    if (not isinstance(specification,dict) or 'fixtures' in specification
                            or not isinstance(specification.get('evaluator'),dict)
                            or not isinstance(specification['evaluator'].get('source'),str)
                            or len(specification['evaluator']['source'])>1024*1024):
                        raise ValueError('Evaluation specification is invalid or oversized')
                    decoded = {}
                    total = 0
                    for name, encoded in fixtures.items():
                        if (not isinstance(name,str) or not name or len(name)>128
                                or not isinstance(encoded,str) or len(encoded)%2
                                or len(encoded)>8*1024*1024
                                or any(char not in '0123456789abcdefABCDEF' for char in encoded)):
                            raise ValueError('Fixture values must be bounded hexadecimal byte strings')
                        value = bytes.fromhex(encoded)
                        total += len(value)
                        if total > 4*1024*1024:
                            raise ValueError('Total evaluation fixture bytes exceed 4 MiB')
                        decoded[name] = value
                    return self.json(200,adapter.evaluate(data['snapshot_id'],data['operation_id'],
                                                          data['specification'],decoded))
                if path=='/cow/evaluation/select':
                    if (not isinstance(data['evaluation_ids'],list) or not 1<=len(data['evaluation_ids'])<=256
                            or not isinstance(data['mandatory'],list) or not 1<=len(data['mandatory'])<=64):
                        raise ValueError('Evaluation selection lists exceed their bounds')
                    return self.json(200,adapter.select_evaluations(data['evaluation_ids'],data['mandatory']))
                if path=='/cow/evaluation/adopt':
                    if (not isinstance(data['evaluation_ids'],list) or not 1<=len(data['evaluation_ids'])<=256
                            or not isinstance(data['mandatory'],list) or not 1<=len(data['mandatory'])<=64):
                        raise ValueError('Evaluation adoption lists exceed their bounds')
                    source_operation=data.get('presentation_source_operation')
                    if source_operation is not None and not isinstance(source_operation,str):
                        raise ValueError('presentation_source_operation must be a string')
                    return self.json(200,adapter.adopt_evaluations(data['operation_id'],data['branch'],
                        data['expected_revision'],data['evaluation_ids'],data['mandatory'],
                        presentation_source_operation=source_operation))
                if path=='/cow/text/retry':
                    stream=data.get('stream',False)
                    if type(stream) is not bool: raise ValueError('stream must be boolean')
                    if not isinstance(data.get('from_operation'),str) or not isinstance(data.get('operation_id'),str):
                        raise ValueError('from_operation and operation_id are required strings')
                    if not isinstance(data.get('branch'),str) or type(data.get('revision')) is not int:
                        raise ValueError('Retry requires a branch and current integer revision')
                    stop=threading.Event()
                    retry_options={} if 'seed' not in data else {'seed':data['seed']}
                    retry=text_adapter.retry(data['from_operation'],data['branch'],data['revision'],
                        data['operation_id'],settings=data.get('settings'),
                        text_stops=data.get('text_stops'),cancelled=stop.is_set,
                        **retry_options)
                    if isinstance(retry,dict):
                        if stream:
                            raw=(json.dumps(retry,allow_nan=False)+'\n').encode()
                            self.send_response(200);self.send_header('Content-Type','application/x-ndjson')
                            self.send_header('Content-Length',str(len(raw)));self.end_headers();self.wfile.write(raw)
                        else:self.json(200,retry)
                        return
                    request=retry
                    trace_request=None
                else:
                    if path not in ('/cow/generate','/cow/create','/cow/text/generate','/cow/text/create',
                            '/cow/text/context','/cow/text/prepare','/cow/text/answer','/cow/text/ask',
                            '/cow/text/resume','/cow/text/append-user'): return self.json(404,{'error':'Not found'})
                    trace_request=trace_request_source(transition_trace,trace_identity,adapter,path,data)
                    stream=data.get('stream',True)
                    seed_options={'seed':data['seed']} if 'seed' in data else {}
                    text_stop_options={'text_stops':data['text_stops']} if 'text_stops' in data else {}
                    if type(stream) is not bool: raise ValueError('stream must be boolean')
                    stop=threading.Event()
                    if path=='/cow/text/create': request=text_adapter.create(data['branch'],data['messages'],data['operation_id'],
                        cancelled=lambda:stop.is_set() or self.client_disconnected())
                    elif path in ('/cow/text/context','/cow/text/prepare'):
                        prepare=text_adapter.create if path=='/cow/text/context' else text_adapter.prepare
                        request=prepare(data['branch'],data['messages'],data['operation_id'],namespace_id=data['namespace_id'],
                            cancelled=lambda:stop.is_set() or self.client_disconnected())
                    elif path=='/cow/text/answer':
                        request=text_adapter.answer(data['branch'],data['revision'],data['operation_id'],
                            data['max_tokens'],settings=data.get('settings'),cancelled=stop.is_set,
                            **seed_options,
                            **({'grammar_state':data['grammar_state']} if 'grammar_state' in data else {}),
                            **text_stop_options)
                    elif path=='/cow/text/ask':
                        request=text_adapter.ask(data['branch'],data['content'],data['revision'],data['operation_id'],
                            data['max_tokens'],settings=data.get('settings'),
                            cancelled=(lambda:stop.is_set() or self.client_disconnected()) if isinstance(data['content'],list) else stop.is_set,
                            **seed_options,
                            **({'grammar_state':data['grammar_state']} if 'grammar_state' in data else {}),
                            **text_stop_options)
                    elif path=='/cow/text/resume':
                        request=text_adapter.resume(data['branch'],data['revision'],data['operation_id'],
                            data['max_tokens'],settings=data.get('settings'),cancelled=stop.is_set,
                            **({'grammar_state':data['grammar_state']} if 'grammar_state' in data else {}),
                            **text_stop_options)
                    elif path=='/cow/text/append-user':
                        request=text_adapter.append_user(data['branch'],data['content'],data['revision'],
                            data['operation_id'],data['max_tokens'],settings=data.get('settings'),
                            cancelled=(lambda:stop.is_set() or self.client_disconnected()) if isinstance(data['content'],list) else stop.is_set,
                            **seed_options,
                            **({'grammar_state':data['grammar_state']} if 'grammar_state' in data else {}),
                            **text_stop_options)
                    elif path=='/cow/text/generate':
                        request=text_adapter.generate(data['branch'],data['messages'],data['revision'],
                            data['operation_id'],data['max_tokens'],settings=data.get('settings'),cancelled=stop.is_set,
                            **seed_options,
                            **({'grammar_state':data['grammar_state']} if 'grammar_state' in data else {}),
                            **text_stop_options)
                    elif path=='/cow/create': request=adapter.create(data['branch'],data['token_ids'],data['operation_id'])
                    else:
                        request=adapter.generate(data['branch'],data['rendered_token_ids'],data['revision'],
                            data['operation_id'],data['max_tokens'],settings=data.get('settings'),cancelled=stop.is_set)
                if trace_request is not None:
                    transition_trace.record('http_request_accepted',trace_request['key'],
                        **{key:value for key,value in trace_request.items() if key!='key'})
            except _ClientDisconnected: return
            except (KeyError,ValueError) as error: return self.json(400,{'error':str(error)})
            except Exception as error: return self.json(409,{'error':str(error)})
            registered_operation=None
            if stream and isinstance(data.get('operation_id'),str):
                operation=data['operation_id']
                with active_stops_lock:
                    if operation in active_stops:
                        return self.json(409,{'error':'Operation already has an active HTTP request'})
                    active_stops[operation]=(stop,request)
                    registered_operation=operation
            if stream:
                try:
                    self.send_response(200);self.send_header('Content-Type','application/x-ndjson')
                    self.send_header('Cache-Control','no-store');self.end_headers();self.close_connection=True
                except _SOCKET_ERRORS:
                    stop.set()
                    if registered_operation is not None:
                        with active_stops_lock: active_stops.pop(registered_operation,None)
                    return
            def emit(value):
                try:
                    self.wfile.write((json.dumps(value,allow_nan=False)+'\n').encode());self.wfile.flush()
                except _SOCKET_ERRORS as error:
                    stop.set()
                    raise _ClientDisconnected from error
            def disconnected():
                return self.client_disconnected()
            text_view=request if path in ('/cow/text/retry','/cow/text/generate','/cow/text/answer','/cow/text/ask',
                '/cow/text/resume','/cow/text/append-user') else None
            event_queue=request.request.events if text_view is not None else request.events
            def progress(event):
                # The durable future supplies the one terminal response below.
                # Coordinator failure notifications must not duplicate it.
                if event.get('event') in ('failure','cancelled'): return
                values=text_view.decode_event(event) if text_view is not None else [event]
                if stream:
                    for value in values: emit(value)
            try:
                while not request.durable.done():
                    if disconnected(): stop.set();return
                    try: event=event_queue.get(timeout=1)
                    except queue.Empty: continue
                    progress(event)
                while not event_queue.empty(): progress(event_queue.get())
                receipt=request.durable.result()
                if trace_request is not None:
                    transition_trace.record('http_request_durable',trace_request['key'],
                        **{key:value for key,value in trace_request.items() if key!='key'},
                        result_snapshot=receipt.get('snapshot') if isinstance(receipt,dict) else None)
                final=dict(event='complete',answer=request.snapshot.result(),receipt=receipt)
                if text_view is not None:
                    final['answer']=dict(final['answer'],text=text_view.text_result())
                    thinking_reader = getattr(text_adapter, 'thinking_result', None)
                    if callable(thinking_reader):
                        thinking = thinking_reader(receipt['snapshot'])
                        if thinking is not None:
                            final['answer']['thinking'] = thinking
                    preparation=getattr(text_view,'input_preparation',None)
                    if preparation is not None:final['answer']['input_preparation']=preparation
                if stream: emit(final)
                else: self.json(200,final)
            except _ClientDisconnected: stop.set()
            except Exception as error:
                error_fields={'error':str(error)}
                if isinstance(error,OSError) and error.errno is not None:
                    error_fields['errno']=error.errno
                if stream:
                    try:
                        emit(dict(event='error',**error_fields,
                                  cancelled=isinstance(error,CancelledError)))
                    except _ClientDisconnected:
                        stop.set()
                else:
                    self.json(409,error_fields)
            finally:
                if registered_operation is not None:
                    with active_stops_lock:
                        if active_stops.get(registered_operation, (None,None))[0] is stop:
                            active_stops.pop(registered_operation,None)
        def image_upload(self):
            # Raw bounded bytes, not JSON/base64, paths or remote URLs. The
            # shared pool reservation covers incoming bytes until admission.
            if text_adapter is None or getattr(text_adapter,'vision_preparer',None) is None:
                return self.json(404,dict(error='Image preparation is not configured'))
            if not api_key:
                return self.json(403,dict(error='Image upload requires a configured API key'))
            try:
                query=parse_qs(urlsplit(self.path).query,keep_blank_values=True,strict_parsing=True)
                if set(query)!={'namespace'} or len(query['namespace'])!=1:
                    raise ValueError('Use exactly one namespace query parameter')
                if self.headers.get_all('Transfer-Encoding') or self.headers.get_all('Content-Encoding'):
                    raise ValueError('Encoded or chunked image bodies are not supported')
                values={}
                for name in ('Content-Length','Content-Type','X-COW-Content-SHA256'):
                    entries=self.headers.get_all(name,[])
                    if len(entries)!=1:raise ValueError('Use exactly one '+name+' header')
                    values[name]=entries[0]
                length=values['Content-Length']
                if not re.fullmatch('[1-9][0-9]{0,7}',length):
                    raise ValueError('Invalid image Content-Length')
                size=int(length)
                if size>16*1024**2:raise ValueError('Image body exceeds 16 MiB')
                if values['Content-Type'] not in ('application/octet-stream','image/png','image/jpeg'):
                    raise ValueError('Use an image/png, image/jpeg or application/octet-stream body')
                digest=values['X-COW-Content-SHA256']
                if re.fullmatch('[0-9a-f]{64}',digest) is None:
                    raise ValueError('Invalid image content digest')
                namespace=query['namespace'][0]
                with text_adapter.reserve_image_upload(namespace,size):
                    try:raw=self.rfile.read(size)
                    except _SOCKET_ERRORS as error:raise _ClientDisconnected from error
                    if len(raw)!=size:raise ValueError('Incomplete image body')
                    receipt=text_adapter.upload_image(namespace,raw,digest)
                return self.json(200,receipt)
            except _ClientDisconnected:return
            except Exception as error:
                from context_time_travel.store import InvalidOperation,NotFound
                code=404 if isinstance(error,NotFound) else 400 if isinstance(error,(ValueError,InvalidOperation)) else 409
                return self.json(code,dict(error=str(error)))
    server=ThreadingHTTPServer((host,port),Handler)
    if grammar_catalog is not None:
        server.daemon_threads=False
        server.block_on_close=True
    return server


def pool_config_from_args(args,profile=None):
    from context_time_travel.pools import PoolConfig
    if args.min_available_gib<=0:
        raise ValueError('The serving memory-headroom guard must be positive')
    return PoolConfig(gpus=profile['execution']['GPU'] if profile is not None else (0,),ram_bytes=args.ram_gib*1024**3,
        disk_bytes=args.disk_gib*1024**3,min_available_bytes=args.min_available_gib*1024**3,
        checkpoint_idle_seconds=args.checkpoint_idle_seconds,
        scheduling=getattr(args,'scheduling','fifo'),cache_inventory=getattr(args,'cache_inventory',False))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile',type=Path,required=True)
    parser.add_argument('--store',type=Path,required=True)
    parser.add_argument('--host',default='127.0.0.1');parser.add_argument('--port',type=int,default=8099)
    parser.add_argument('--openai-port',type=int,help='Opt-in standard CLI API listener sharing this coordinator')
    parser.add_argument('--openai-model',default='cow')
    parser.add_argument('--openai-max-output',type=int,default=2048)
    parser.add_argument('--api-key');parser.add_argument('--ram-gib',type=int,default=96)
    parser.add_argument('--disk-gib',type=int,default=128)
    parser.add_argument('--cache-inventory',action='store_true',help='Expose recoverable cache-location hints')
    parser.add_argument('--scheduling',choices=('fifo','weighted-prefix','transfer-aware'),default='fifo',
        help='Opt in to shared-prefill admission with backlog weights and bounded-wait aging')
    parser.add_argument('--min-available-gib',type=int,default=16,help='Positive host RAM headroom; guided setup sizes this for the server.')
    parser.add_argument('--checkpoint-idle-seconds',type=float,
        help='Expire unused eligible RAM/disk checkpoint copies; testing: 1800, production example: 10800. Omit to disable.')
    parser.add_argument('--text',action='store_true',help='Enable literal-safe model-template text endpoints')
    parser.add_argument('--vision-config',type=Path,
        help='Pinned image producer JSON; requires --text, --api-key and a compatible mixed-input profile. Omit for restore-only.')
    parser.add_argument('--grammar-registry',type=Path,help='Opt in to a fixed server-owned GBNF/JSON Schema registry (requires --text and llguidance 1.9.1)')
    parser.add_argument('--grammar-store',type=Path,help='Opt in to authenticated durable inline grammar admission; requires --text and --api-key')
    parser.add_argument('--pool-trace-output',type=Path,
        help='Optional private local trace path; written only after ready service closes cleanly')
    args=parser.parse_args()
    if args.openai_port is not None:
        if not args.text:parser.error('--openai-port requires --text')
        if not 1<=args.openai_port<=65535 or args.openai_port==args.port:
            parser.error('Compatibility port must be valid and different from native COW port')
        if (re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._:/-]{0,127}',args.openai_model) is None
                or args.openai_max_output<=0):parser.error('Invalid compatibility model ID or output limit')
    if args.pool_trace_output is not None:
        validate_pool_trace_target(args.pool_trace_output)
    # Import only in opt-in Linux startup: normal serving has no dependency on
    # the experimental store or POSIX locking.
    from context_time_travel.coordinator import ContextCoordinator
    from context_time_travel.pools import PoolConfig,PoolTransitionTrace
    from context_time_travel.real import BranchingModel
    from context_time_travel.strata_backend import StrataBackend,verify_runtime_source_closure
    from serve.cow_adapter import CowAdapter
    profile=json.loads(args.profile.read_text())
    runtime_sources=verify_runtime_source_closure(profile,Path(__file__).resolve().parents[1])
    if args.vision_config is not None and (not args.text or not args.api_key):
        raise ValueError('--vision-config requires --text and --api-key')
    grammar_registry=None
    grammar_catalog=None
    if args.grammar_store is not None and (not args.text or not args.api_key):
        raise ValueError('--grammar-store requires --text and --api-key')
    if args.grammar_registry is not None or args.grammar_store is not None:
        if not args.text:
            raise ValueError('--grammar-registry requires --text')
        try:
            from serve.cow_grammar import GrammarRegistry
            from tools.strata_tokenizer import Tokenizer
            startup_tokenizer=Tokenizer.from_gguf(profile['model_path'])
            if args.grammar_registry is not None:
                grammar_registry=GrammarRegistry.load(args.grammar_registry,startup_tokenizer,
                    profile['execution'].get('eos_tokens',[]))
            if args.grammar_store is not None:
                from serve.cow_grammar_disk import DurableGrammarCatalog
                grammar_catalog=DurableGrammarCatalog(args.grammar_store,startup_tokenizer,
                    profile['execution'].get('eos_tokens',[]))
        except ImportError as error:
            raise RuntimeError('Grammar serving requires the optional pinned llguidance==1.9.1 dependency') from error
    config=pool_config_from_args(args,profile)
    if profile.get('backend')=='qwen35-llama.cpp':
        from context_time_travel.qwen35_backend import device_name_for_worker
        for gpu in config.gpus:device_name_for_worker(profile,gpu)
    components=model_text_components(profile['model_path'],profile['execution'].get('text_template_kwargs'),
        model_contract_id=profile['model_contract_id'],
        mixed_inputs=profile['execution'].get('resume_contract')=='identical-mixed-native-operators-v1') if args.text else None
    preparer=None
    if args.vision_config is not None:
        from context_time_travel.store import Store
        preparer=vision_preparer_from_config(Store(args.store),components['mixed_renderer'],profile,args.vision_config)
    def factory(gpu):
        if profile.get('backend')=='qwen35-llama.cpp':
            from context_time_travel.qwen35_backend import load_qwen35_backend
            backend=load_qwen35_backend(profile,gpu_id=gpu)
        else:
            backend=StrataBackend(profile,gpu_id=gpu)
        return BranchingModel(args.store,backend,profile)
    trace=PoolTransitionTrace('cow-http') if args.pool_trace_output is not None else None
    coordinator=ContextCoordinator(args.store,config,factory,transition_trace=trace,input_preparer=preparer)
    server=None
    compatibility_server=None
    compatibility_thread=None
    compatibility_stop=threading.Event()
    service_ready=False
    coordinator_closed=False
    try:
        coordinator.ready();adapter=CowAdapter(coordinator,grammar_registry=grammar_registry)
        from serve.cow_text_adapter import CowTextAdapter
        text=CowTextAdapter(adapter,**components,grammar_registry=grammar_registry,
            vision_preparer=preparer) if components is not None else None
        from serve.cow_workspace import CowWorkspace
        workspace=CowWorkspace(coordinator.store,text.tokenizer,
            root_mode_resolver=text.root_mode_for_projection) if text is not None else None
        server=make_server(adapter,args.host,args.port,args.api_key,text_adapter=text,workspace=workspace,
            transition_trace=trace,trace_identity=(profile['model_contract_id'],profile['execution_profile_id']),grammar_catalog=grammar_catalog,
            connection_info=(dict(model_contract_id=profile['model_contract_id'],
                execution_profile_id=profile['execution_profile_id'],
                eos_tokens=profile['execution']['eos_tokens']) if profile['execution'].get('eos_tokens') else None))
        if trace is not None:
            # On graceful shutdown, wait for handlers to finish writing their
            # terminal request markers before the trace snapshot is published.
            server.daemon_threads=False
            server.block_on_close=True
        if args.openai_port is not None:
            from serve.cow_compat import Frontend,make_server as make_compatibility_server
            # The callbacks above are bound to the same literal-safe model
            # prompt encoder used by the native text boundary.
            encoder=components['render'].__self__
            frontend=Frontend(adapter,encoder,profile,args.openai_model,args.openai_max_output)
            compatibility_server=make_compatibility_server(frontend,args.host,args.openai_port,
                                                          args.api_key,compatibility_stop)
            compatibility_thread=threading.Thread(target=compatibility_server.serve_forever,
                name='cow-standard-cli',daemon=True)
            compatibility_thread.start()
            server.daemon_threads=False
            server.block_on_close=True
            print(json.dumps(dict(event='compatibility_ready',address=compatibility_server.server_address,
                                  model=args.openai_model)),flush=True)
        print(json.dumps(dict(event='ready',address=server.server_address,
                              execution_profile_id=profile['execution_profile_id'],runtime_sources=runtime_sources)),flush=True)
        service_ready=True
        try: server.serve_forever()
        except KeyboardInterrupt: pass
    finally:
        try:
            compatibility_stop.set()
            if compatibility_server is not None:
                compatibility_server.shutdown();compatibility_server.server_close()
                compatibility_thread.join(timeout=10)
            if server is not None: server.server_close()
        finally:
            try:
                coordinator.close();coordinator_closed=True
            finally:
                try:
                    if trace is not None:
                        persist_pool_trace(args.pool_trace_output,trace,
                            service_ready=service_ready,coordinator_closed=coordinator_closed)
                finally:
                    if grammar_catalog is not None:grammar_catalog.close()


if __name__=='__main__': main()
