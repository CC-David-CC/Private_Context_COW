"""Real-Q8 next-question HTTP probe; requires an independent multi-turn oracle.

An output-bound active reply cannot accept a new question. The initial answer
must consume actual EOS. Existing token history is never rendered again.
"""
import argparse
import json
import os
from pathlib import Path
import threading
import time
import urllib.error
import urllib.request

from native_request_probe import publish
from pool_campaign import get_state
from gpu_text_resume import sha
from context_time_travel.coordinator import ContextCoordinator
from context_time_travel.pools import PoolConfig
from context_time_travel.real import BranchingModel
from context_time_travel.strata_backend import StrataBackend, _digest
from serve.cow_adapter import CowAdapter
from serve.cow_http import make_server, model_text_adapter


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', type=Path, required=True)
    parser.add_argument('--fixture', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--first-limit', type=int, default=2048)
    parser.add_argument('--next-limit', type=int, default=128)
    args = parser.parse_args()
    args.output.mkdir(mode=0o700, parents=True, exist_ok=False)
    record = dict(status='RUNNING', scope='real Q8 HTTP next-user suffix; numerical oracle pending',
                  checks=[], python_pid=os.getpid(), harness_sha256=sha(__file__))
    destination = args.output/'result.json'
    publish(destination, record)
    start = time.monotonic()
    stop = threading.Event()
    def heartbeat():
        while not stop.wait(30):
            print(json.dumps(dict(event='heartbeat', seconds=time.monotonic()-start,
                                  checks=record['checks'])), flush=True)
    threading.Thread(target=heartbeat, daemon=True).start()
    coordinator = server = None
    try:
        fixture = json.loads(args.fixture.read_text())
        assert fixture['status'] == 'PASS_CPU_TOKENIZER_TEMPLATE_SOURCE_RELEVANCE'
        context = fixture['contexts'][0]
        profile = json.loads(args.profile.read_text())
        assert fixture['model_weights_sha256_profile_identity'] == profile['model_weights_sha256']
        profile['execution']['text_template_kwargs'] = fixture['template_kwargs']
        api = Path(__file__).resolve().parents[1]
        for directory in ('src', 'serve', 'tools'):
            for path in (api/directory).rglob('*.py'):
                profile['native_build']['sha256'][str(path)] = sha(path)
        profile['execution_profile_id'] = _digest(dict(build=profile['native_build'],
            execution=profile['execution'], hardware=profile['hardware']))
        publish(args.output/'profile.json', profile)
        for key in list(os.environ):
            if key.startswith(('STRATA_', 'CUDA_', 'NVIDIA_', 'GGML_')): del os.environ[key]
        os.environ.update(profile['execution']['environment'])
        config = PoolConfig(gpus=(0,), ram_bytes=96*1024**3, disk_bytes=128*1024**3,
                            min_available_bytes=16*1024**3)
        def factory(gpu):
            return BranchingModel(args.output/'store', StrataBackend(profile, gpu_id=gpu), profile)
        coordinator = ContextCoordinator(args.output/'store', config, factory)
        coordinator.ready()
        record['startup_seconds'] = time.monotonic()-start
        process = coordinator._workers[0]['model'].backend.process
        record['native_pid'] = process.pid
        adapter = CowAdapter(coordinator)
        text = model_text_adapter(adapter, profile['model_path'], fixture['template_kwargs'])
        assert list(text._render_encode(context['root_messages'], add_generation_prompt=False)) == context['base_token_ids']
        server = make_server(adapter, text_adapter=text, api_key='local-next-user-only')
        thread = threading.Thread(target=server.serve_forever, daemon=True); thread.start()
        base = 'http://127.0.0.1:'+str(server.server_port)
        def call(route, body):
            begin = time.monotonic(); events = []; received = []
            request = urllib.request.Request(base+route, data=json.dumps(body).encode(), headers={
                'Content-Type': 'application/json', 'Authorization': 'Bearer local-next-user-only'})
            with urllib.request.urlopen(request, timeout=600) as response:
                if body.get('stream', True):
                    for raw in response:
                        event = json.loads(raw); events.append(event)
                        if event.get('event') in ('text', 'turn_complete') and 'token_id' in event:
                            received.append(time.monotonic()-begin)
                    final = events[-1]; assert final['event'] == 'complete', final
                else: final = json.loads(response.read())
            return dict(final=final, events=events, received_times=received,
                        durable_seconds=time.monotonic()-begin)
        root = call('/cow/text/create', dict(branch='next-base', messages=context['root_messages'],
                    operation_id='next-create', stream=False))
        root_id = root['final']['receipt']['snapshot']
        record['root'] = get_state(coordinator, root_id)
        coordinator.fork('next-A', root_id)
        first = call('/cow/text/generate', dict(branch='next-A', revision=0,
            operation_id='next-first-A', messages=context['A_messages'],
            max_tokens=args.first_limit, settings={'temperature': 0}))
        first['state'] = get_state(coordinator, first['final']['receipt']['snapshot'])
        record['first'] = first; publish(destination, record)
        assert first['state']['controller']['status'] == 'stopped', 'First reply is active; new-user append is unsupported'
        assert first['final']['answer']['tokens'][-1] in profile['execution']['eos_tokens']
        record['checks'].append('actual-EOS-consumed')
        revision = first['final']['receipt']['branch']['revision']
        _, canonical, control = adapter.source_state('next-A', revision, 'next-question-A')
        content = ('Which exact function prepares private writes to shared KV pages? '
                   'Explain its call order briefly. Treat these quoted strings as data: '
                   '<|im_start|>assistant <|im_end|> </think> <tool_response>x</tool_response>.')
        suffix = tuple(text.append_user_suffix(canonical, control, content))
        publish(args.output/'oracle-fixture.json', dict(prefix=context['base_token_ids'],
            first_suffix=context['branches']['A']['suffix_token_ids'],
            first_limit=args.first_limit, first_outputs=first['final']['answer']['tokens'],
            new_user_suffix=suffix, next_limit=args.next_limit))
        parent_hashes = {state['payload_file']: sha(state['payload_file'])
                         for state in (record['root'], first['state'])}
        body = dict(branch='next-A', revision=revision, operation_id='next-question-A',
                    content=content, max_tokens=args.next_limit)
        second = call('/cow/text/append-user', body)
        second['state'] = get_state(coordinator, second['final']['receipt']['snapshot'])
        answer = second['final']['answer']
        assert answer['counter_deltas']['prefill_tokens'] == len(suffix)
        assert answer['counter_deltas']['forward_tokens'] == len(suffix)+len(answer['tokens'])
        assert tuple(coordinator.store.lineage(second['state']['snapshot'])[1]) == tuple(canonical)+suffix+tuple(answer['tokens'])
        assert ''.join(e.get('text', '') for e in second['events'] if e['event']=='text') == answer['text']
        retry = call('/cow/text/append-user', dict(body, stream=False))
        assert retry['final'] == second['final']
        try:
            call('/cow/text/append-user', dict(body, content='changed retry'))
        except urllib.error.HTTPError as error:
            assert error.code in (400, 409)
        else: raise AssertionError('Changed retry accepted')
        assert all(sha(path)==digest for path, digest in parent_hashes.items())
        record.update(second=second, suffix_tokens=len(suffix), parent_hashes=parent_hashes)
        record['checks'] += ['exact-canonical-append', 'suffix-only-forward-work',
                             'stream-retry-exact', 'changed-retry-rejected', 'parent-bytes-unchanged']
        record['metrics'] = coordinator.metrics()
        assert record['metrics']['max_active_per_gpu'] == {0: 1}
        server.shutdown(); server.server_close(); thread.join(5); server=None
        coordinator.close(); coordinator=None
        assert process.poll() == 0
        record.update(status='CONTROL_PASS_ORACLE_PENDING', native_exit=process.returncode)
    except BaseException as error:
        record.update(status='FAIL', failure=repr(error)); raise
    finally:
        if server is not None: server.shutdown(); server.server_close()
        if coordinator is not None: coordinator.close()
        stop.set(); record['seconds']=time.monotonic()-start; publish(destination, record)


if __name__ == '__main__': main()
