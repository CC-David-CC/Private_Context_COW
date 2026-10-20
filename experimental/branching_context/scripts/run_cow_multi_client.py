"""Exercise concurrent, independent COW branches through the real HTTP API.

Real mode starts one pinned COW service against a disposable copy of a Store.
Synthetic mode is available only to CPU tests using the real HTTP, Store, and
coordinator path with FlowModel; it is not GPU or model evidence.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import http.client
import json
import os
from pathlib import Path
import threading
import time
import uuid
from urllib.parse import quote, urlsplit

from run_cow_client_recovery import (
    ProgressTracker,
    RecoveryFailure,
    _assert_resume_complete,
    _fork,
    _head_history,
    _private_run_dir,
    _ready_url,
    _require_ok,
    _sha256,
    _start_service,
    _trace_worker,
    checkpoint_fingerprint,
)
from context_time_travel.controller import consume_many
from context_time_travel.store import Store


class HttpJsonClient:
    def __init__(self, base_url, *, timeout=1800, api_key=None):
        parsed = urlsplit(base_url)
        if parsed.scheme != 'http' or parsed.hostname not in ('127.0.0.1', 'localhost', '::1'):
            raise ValueError('Harness requires loopback HTTP')
        self.host, self.port = parsed.hostname, parsed.port
        self.prefix = parsed.path.rstrip('/')
        self.timeout, self.api_key = timeout, api_key

    def _request(self, method, path, body=None):
        connection = http.client.HTTPConnection(self.host, self.port, timeout=self.timeout)
        headers = {}
        payload = None
        if body is not None:
            payload = json.dumps(body, separators=(',', ':'), ensure_ascii=False).encode('utf-8')
            headers['Content-Type'] = 'application/json'
        if self.api_key:
            headers['Authorization'] = 'Bearer ' + self.api_key
        try:
            connection.request(method, self.prefix + path, body=payload, headers=headers)
            response = connection.getresponse()
            raw = response.read()
            status = response.status
        finally:
            connection.close()
        try:
            value = json.loads(raw)
        except (TypeError, ValueError):
            value = raw.decode('utf-8', errors='replace')
        return status, value

    def get(self, path):
        return self._request('GET', path)

    def post(self, path, body):
        return self._request('POST', path, body)

    def post_json(self, path, body):
        return self.post(path, body)

    def stream_resume(self, branch, revision, operation_id, max_tokens, event_path,
                      barrier, timeout, progress=None):
        connection = http.client.HTTPConnection(self.host, self.port, timeout=timeout)
        payload = json.dumps(dict(branch=branch, revision=revision,
            operation_id=operation_id, max_tokens=max_tokens), separators=(',', ':')).encode('utf-8')
        headers = {'Content-Type': 'application/json'}
        if self.api_key:
            headers['Authorization'] = 'Bearer ' + self.api_key
        started = None
        events = []
        try:
            barrier.wait(timeout=timeout)
            started = time.monotonic()
            connection.request('POST', self.prefix + '/cow/text/resume', body=payload, headers=headers)
            response = connection.getresponse()
            if response.status != 200:
                detail = response.read().decode('utf-8', errors='replace')
                raise RecoveryFailure(f'resume HTTP {response.status}: {detail}')
            with Path(event_path).open('xb') as stream:
                for raw_line in response:
                    if not raw_line.strip():
                        continue
                    stream.write(raw_line)
                    stream.flush()
                    value = json.loads(raw_line)
                    if not isinstance(value, dict):
                        raise RecoveryFailure('Stream event is not a JSON object')
                    events.append(value)
                    if progress is not None:
                        progress.event(operation_id, value)
        finally:
            connection.close()
        if started is None:
            raise RecoveryFailure('Client start barrier did not release')
        return dict(events=events, started_monotonic=started,
                    completed_monotonic=time.monotonic(), elapsed_seconds=time.monotonic() - started)


def _canonical_sha(value):
    raw = json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False,
                     allow_nan=False).encode('utf-8')
    return hashlib.sha256(raw).hexdigest()


def _head_state(client, branch):
    head, history = _head_history(client, branch)
    if (not isinstance(head.get('tip'), str) or not head.get('tip')
            or not isinstance(head.get('root'), str) or not head.get('root')
            or type(head.get('revision')) is not int):
        raise RecoveryFailure(f'Branch {branch!r} has malformed head metadata')
    if history.get('controller_status') != 'active':
        raise RecoveryFailure(f'Branch {branch!r} has no active controller to resume')
    return dict(head=head, history=history, head_sha256=_canonical_sha(head),
                history_sha256=_canonical_sha(history))


def _status(client):
    status, value = client.get('/cow/status')
    _require_ok(status, value, 'read COW coordinator status')
    if (not isinstance(value, dict) or not isinstance(value.get('active_per_gpu'), dict)
            or not isinstance(value.get('max_active_per_gpu'), dict)
            or type(value.get('queued')) is not int or value['queued'] < 0):
        raise RecoveryFailure('Coordinator status omitted concurrency metrics')
    return value


def _verify_canonical_snapshot(store, source_snapshot, final_snapshot, output_tokens, tokenizer=None):
    source = store.get_snapshot(source_snapshot)
    final = store.get_snapshot(final_snapshot)
    if final.get('parent') != source_snapshot:
        raise RecoveryFailure('Final snapshot parent differs from the acknowledged source tip')
    if final.get('model_contract_id') != source.get('model_contract_id'):
        raise RecoveryFailure('Final snapshot model contract differs from its source')
    if (not isinstance(output_tokens, list) or not output_tokens
            or any(type(token) is not int or token < 0 for token in output_tokens)):
        raise RecoveryFailure('Acknowledged output IDs are malformed')
    if final.get('inputs') != output_tokens:
        raise RecoveryFailure('Final snapshot input suffix differs from acknowledged output IDs')
    source_root, source_tokens = store.lineage(source_snapshot)
    final_root, final_tokens = store.lineage(final_snapshot)
    if source_root != final_root:
        raise RecoveryFailure('Final snapshot ancestry resolves to a different root')
    if final_tokens != source_tokens + output_tokens:
        raise RecoveryFailure('Final canonical token lineage is not source tokens plus acknowledged output')
    controller_status = 'not_performed_no_pinned_tokenizer'
    if tokenizer is not None:
        token_bytes = getattr(tokenizer, 'token_bytes', None)
        if not callable(token_bytes):
            raise RecoveryFailure('Pinned tokenizer does not provide token_bytes')
        source_controller = source.get('controller')
        policy = source_controller.get('policy') if isinstance(source_controller, dict) else None
        eos = policy.get('eos_tokens', []) if isinstance(policy, dict) else []
        try:
            events = [(token, token_bytes(token), token in eos) for token in output_tokens]
            expected_controller = consume_many(source_controller, events)
        except Exception as error:
            raise RecoveryFailure(f'Cannot reconstruct acknowledged controller transition: {error}') from error
        if expected_controller != final.get('controller'):
            raise RecoveryFailure('Final controller state differs from tokenizer-bound output consumption')
        controller_status = 'passed_full_state_reconstruction'
    return dict(parent=final['parent'], source_root=source_root,
        model_contract_id=final['model_contract_id'], source_input_tokens=len(source_tokens),
        final_input_tokens=len(final_tokens), output_tokens=len(output_tokens),
        source_lineage_sha256=_canonical_sha(source_tokens),
        final_lineage_sha256=_canonical_sha(final_tokens),
        controller_validation=controller_status)


def _concurrency_evidence(samples, final_status, *, synthetic_fixture):
    observed_max_active = {}
    for sample in list(samples) + [final_status]:
        if (not isinstance(sample.get('active_per_gpu'), dict)
                or not isinstance(sample.get('max_active_per_gpu'), dict)):
            raise RecoveryFailure('Coordinator status omitted per-GPU active counters')
        if any(type(value) is not int or value < 0 for value in sample['active_per_gpu'].values()):
            raise RecoveryFailure('active_per_gpu contains an invalid counter')
        for gpu, value in sample['max_active_per_gpu'].items():
            if type(value) is not int or value < 0 or value > 1:
                raise RecoveryFailure('max_active_per_gpu is outside the supported one-worker limit')
            observed_max_active[gpu] = max(observed_max_active.get(gpu, 0), value)
    if not observed_max_active or max(observed_max_active.values()) != 1:
        raise RecoveryFailure(f'Observed max_active_per_gpu was not exactly one: {observed_max_active}')
    if not synthetic_fixture and observed_max_active != {'0': 1}:
        raise RecoveryFailure(f'Real serving profile requires exactly GPU 0 single-worker activity: {observed_max_active}')
    overlap = [sample for sample in samples if sample.get('queued', 0) > 0
               and any(type(active) is int and active > 0
                       for active in sample.get('active_per_gpu', {}).values())]
    if not overlap:
        raise RecoveryFailure('No status sample showed queued work while a GPU worker was active')
    if final_status['queued'] != 0 or any(final_status['active_per_gpu'].values()):
        raise RecoveryFailure('Coordinator was not quiescent after all client receipts')
    return observed_max_active, overlap


def _append_jsonl(path, value):
    with Path(path).open('a', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False) + '\n')
        stream.flush()
        os.fsync(stream.fileno())


def _write_json_atomic(path, value):
    path = Path(path)
    temporary = path.with_name('.' + path.name + '.' + uuid.uuid4().hex + '.tmp')
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, 'w', encoding='utf-8', newline='\n') as stream:
            stream.write(json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False) + '\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        if os.name == 'posix':
            directory = os.open(path.parent, os.O_RDONLY | getattr(os, 'O_DIRECTORY', 0))
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
    except BaseException:
        try:
            os.close(descriptor)
        except OSError:
            pass
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise


def run_concurrent_clients(*, base_url, source_branches, store, run_dir, max_tokens=8,
                           timeout=1800, sample_interval=0.025,
                           synthetic_fixture=False, status_observer=None,
                           tokenizer=None, progress=None):
    """Fork each active source and resume the forks concurrently over HTTP.

    ``status_observer`` is a test-only synchronization hook. It receives actual
    HTTP status snapshots and cannot alter request or coordinator state.
    """
    if not isinstance(source_branches, (list, tuple)) or not 3 <= len(source_branches) <= 4:
        raise ValueError('Supply three or four independent active source branches')
    if len(set(source_branches)) != len(source_branches):
        raise ValueError('Source branch names must be distinct')
    if type(max_tokens) is not int or max_tokens <= 0:
        raise ValueError('max_tokens must be a positive integer')
    if sample_interval <= 0:
        raise ValueError('sample_interval must be positive')
    output = Path(run_dir)
    client = HttpJsonClient(base_url, timeout=timeout)
    nonce = uuid.uuid4().hex
    store_input = Path(store).expanduser().absolute()
    if store_input.is_symlink() or not store_input.is_dir():
        raise ValueError('Store must be an existing nonsymlink directory')
    store_path = store_input.resolve(strict=True)
    sources = {}
    if progress is not None:
        progress.phase('preflight_source_branches')
    for branch in source_branches:
        state = _head_state(client, branch)
        state['checkpoint'] = checkpoint_fingerprint(store_path, state['head']['tip'],
            synthetic_fixture=synthetic_fixture)
        sources[branch] = state
    roots = [state['head']['root'] for state in sources.values()]
    tips = [state['head']['tip'] for state in sources.values()]
    if len(set(roots)) != len(roots) or len(set(tips)) != len(tips):
        raise RecoveryFailure('Source branches do not identify distinct context trees and active snapshots')

    if progress is not None:
        progress.phase('fork_disposable_branches', completed='source_preflight_complete')
    children = {}
    for index, branch in enumerate(source_branches):
        child = f'COWMulti{index}{nonce[:18]}'
        operation = f'op-multi-fork-{index}-{nonce}'
        snapshot = sources[branch]['head']['tip']
        receipt = _fork(client, branch, snapshot, child, operation)
        fork_head, fork_history = _head_history(client, child)
        if (fork_head.get('tip') != snapshot or fork_head.get('root') != sources[branch]['head']['root']
                or fork_history.get('controller_status') != 'active'):
            raise RecoveryFailure(f'Fork {child!r} did not preserve its source snapshot/controller: '
                                  f'head={fork_head!r}, history={fork_history!r}')
        children[child] = dict(source_branch=branch, source_snapshot=snapshot,
            source_head_sha256=sources[branch]['head_sha256'], receipt=receipt,
            fork_head=fork_head, fork_history_sha256=_canonical_sha(fork_history),
            source_history_sha256=sources[branch]['history_sha256'],
            operation_id=f'op-multi-resume-{index}-{nonce}')

    sample_path = output / 'status-samples.jsonl'
    sampler_stop = threading.Event()
    samples = []
    sampler_errors = []

    def sample_status():
        while not sampler_stop.is_set():
            try:
                value = _status(client)
                row = dict(sample_monotonic=time.monotonic(), **value)
                samples.append(row)
                _append_jsonl(sample_path, row)
                if status_observer is not None:
                    status_observer(value)
            except BaseException as error:
                sampler_errors.append(f'{type(error).__name__}: {error}')
                return
            sampler_stop.wait(sample_interval)

    sampler = threading.Thread(target=sample_status, name='cow-multi-status-sampler', daemon=True)
    sampler.start()
    if progress is not None:
        progress.phase('concurrent_http_resumes', completed='forks_verified')
    barrier = threading.Barrier(len(children) + 1)
    request_results = {}
    request_errors = {}
    futures = {}
    executor = ThreadPoolExecutor(max_workers=len(children), thread_name_prefix='cow-client')
    try:
        for index, (child, info) in enumerate(children.items()):
            event_path = output / f'request-{index}.ndjson'
            request_client = HttpJsonClient(base_url, timeout=timeout)
            futures[executor.submit(request_client.stream_resume, child, 0,
                info['operation_id'], max_tokens, event_path, barrier, timeout, progress)] = (child, event_path)
        barrier.wait(timeout=timeout)
        for future in as_completed(futures, timeout=timeout):
            child, event_path = futures[future]
            try:
                request_results[child] = future.result()
            except BaseException as error:
                request_errors[child] = f'{type(error).__name__}: {error}'
    except BaseException:
        barrier.abort()
        for future in futures:
            future.cancel()
        raise
    finally:
        executor.shutdown(wait=True, cancel_futures=True)
        sampler_stop.set()
        sampler.join(timeout=max(1.0, timeout))
        if sampler.is_alive():
            raise RecoveryFailure('Coordinator status sampler did not stop')
    if sampler_errors:
        raise RecoveryFailure('Coordinator status sampling failed: ' + '; '.join(sampler_errors))
    if request_errors:
        raise RecoveryFailure('One or more concurrent requests failed: ' + repr(request_errors))

    if progress is not None:
        progress.phase('validate_durable_receipts', completed='all_http_streams_completed')
    canonical_store = Store(store_path)
    per_client = {}
    for child, info in children.items():
        row = request_results[child]
        summary = _assert_resume_complete(row['events'], expected_revision=0,
            require_zero_prefill=not synthetic_fixture)
        receipt_branch = summary['branch']
        if receipt_branch.get('name') != child or receipt_branch.get('revision') != 1:
            raise RecoveryFailure(f'Resume receipt does not identify the expected disposable branch {child!r}')
        status, final_head = client.get('/cow/head?branch=' + quote(child, safe=''))
        _require_ok(status, final_head, f'read final head for {child}')
        status, final_history = client.get('/cow/workspace/history?branch=' + quote(child, safe=''))
        _require_ok(status, final_history, f'read final canonical history for {child}')
        if (final_head.get('tip') != summary['snapshot'] or final_head.get('revision') != 1
                or final_head.get('root') != sources[info['source_branch']]['head']['root']):
            raise RecoveryFailure(f'Final branch head for {child!r} differs from its durable receipt/source root')
        canonical_validation = _verify_canonical_snapshot(canonical_store, info['source_snapshot'],
            summary['snapshot'], summary['output_tokens'], tokenizer=tokenizer)
        per_client[child] = dict(source_branch=info['source_branch'],
            source_snapshot=info['source_snapshot'], disposable_branch=child,
            operation_id=info['operation_id'], latency_seconds=row['elapsed_seconds'],
            client_start_monotonic=row['started_monotonic'],
            client_complete_monotonic=row['completed_monotonic'],
            output_tokens=summary['output_tokens'], counter_deltas=summary['counter_deltas'],
            receipt=summary['branch'], final_head=final_head, final_history=final_history,
            final_history_sha256=_canonical_sha(final_history),
            fork_history_sha256=info['fork_history_sha256'],
            source_history_sha256=info['source_history_sha256'],
            canonical_validation=canonical_validation,
            stream_path=str(output / f'request-{list(children).index(child)}.ndjson'))

    source_after = {}
    for branch, before in sources.items():
        after = _head_state(client, branch)
        if (after['head_sha256'] != before['head_sha256']
                or after['history_sha256'] != before['history_sha256']):
            raise RecoveryFailure(f'Original source branch {branch!r} changed during its fork resume')
        checkpoint_after = checkpoint_fingerprint(store_path, before['head']['tip'],
            synthetic_fixture=synthetic_fixture)
        if checkpoint_after['files'] != before['checkpoint']['files']:
            raise RecoveryFailure(f'Original source checkpoint for {branch!r} changed during resume')
        source_after[branch] = dict(head_sha256=after['head_sha256'],
            history_sha256=after['history_sha256'], checkpoint_files=checkpoint_after['files'])

    final_status = _status(client)
    observed_max_active, overlap = _concurrency_evidence(samples, final_status,
        synthetic_fixture=synthetic_fixture)

    return dict(schema='cow-multi-client-v1', status='PASS',
        mode='synthetic' if synthetic_fixture else 'real',
        scope=('synthetic HTTP/Store/coordinator only; no GPU/model claim' if synthetic_fixture
               else 'one pinned real COW service, independent active source trees'),
        numerical_qualification=dict(status='NOT_PERFORMED',
            reason='Work counters and durable branch receipts are checked; logits are not compared.'),
        client_count=len(per_client), source_branches=list(source_branches),
        source_branch_preflight={branch:dict(head=state['head'],
            history_sha256=state['history_sha256'], checkpoint=state['checkpoint'])
            for branch, state in sources.items()},
        clients=per_client, source_branches_unchanged=source_after,
        status_sample_count=len(samples), status_samples_path=str(sample_path),
        positive_queue_overlap_samples=len(overlap),
        observed_max_active_per_gpu=observed_max_active,
        final_status=final_status,
        arrival_order_claim='none; request timestamps are client-side observations only')


def run_real(*, profile, store, source_branches, run_dir, max_tokens=8, timeout=1800,
             sample_interval=.025, ram_gib=96, disk_gib=128):
    run_dir = _private_run_dir(run_dir)
    progress = ProgressTracker(run_dir, 'real').start()
    process = None
    ready = None
    stop = None
    stop_attempted = False
    try:
        progress.phase('starting_single_cow_http_service')
        process, ready, client = _start_service('real', Path(store), run_dir, 1,
            profile=Path(profile), timeout=timeout, ram_gib=ram_gib, disk_gib=disk_gib,
            progress=progress)
        progress.phase('cow_http_service_ready', completed='service_ready')
        profile_data = json.loads(Path(profile).read_text(encoding='utf-8'))
        from tools.strata_tokenizer import Tokenizer
        tokenizer = Tokenizer.from_gguf(profile_data['model_path'])
        progress.phase('pinned_tokenizer_loaded', completed='tokenizer_ready')
        result = run_concurrent_clients(base_url=_ready_url(ready), source_branches=source_branches,
            store=store, run_dir=run_dir, max_tokens=max_tokens, timeout=timeout,
            sample_interval=sample_interval, synthetic_fixture=False,
            tokenizer=tokenizer, progress=progress)
        progress.phase('stopping_cow_http_service', completed='all_clients_and_store_checks_passed')
        if process is not None:
            stop_attempted = True
            stop = process.stop()
        if not isinstance(stop, dict) or stop.get('exit_code') != 0:
            raise RecoveryFailure(f'COW service exit was not zero: {stop}')
        pool_path = run_dir / 'pool-trace-1.json'
        worker = _trace_worker(pool_path, stop['pid'])
        progress.native_started(worker['native_pid'])
        native_path = run_dir / 'native-trace-1.jsonl'
        result['service'] = dict(ready=ready, stop=stop, worker=worker,
            pool_trace_path=str(pool_path), pool_trace_sha256=_sha256(pool_path),
            native_trace_path=str(native_path),
            native_trace_sha256=_sha256(native_path) if native_path.is_file() else None)
        result['profile_sha256'] = _sha256(Path(profile))
        result['store'] = str(Path(store).resolve())
        _write_json_atomic(run_dir / 'result.json', result)
        progress.finish('PASS')
        return result
    except BaseException as error:
        if process is not None and not stop_attempted:
            stop_attempted = True
            try:
                stop = process.stop()
            except BaseException as stop_error:
                stop = dict(stop_error=f'{type(stop_error).__name__}: {stop_error}')
        pool_path = run_dir / 'pool-trace-1.json'
        native_path = run_dir / 'native-trace-1.jsonl'
        trace = dict(pool_trace_path=str(pool_path) if pool_path.is_file() else None,
            pool_trace_sha256=_sha256(pool_path) if pool_path.is_file() else None,
            native_trace_path=str(native_path) if native_path.is_file() else None,
            native_trace_sha256=_sha256(native_path) if native_path.is_file() else None)
        if pool_path.is_file() and isinstance(stop, dict) and type(stop.get('pid')) is int:
            try:
                trace['worker_close'] = _trace_worker(pool_path, stop['pid'])
            except BaseException as trace_error:
                trace['worker_close_error'] = f'{type(trace_error).__name__}: {trace_error}'
        failure = dict(schema='cow-multi-client-v1', status='FAIL', mode='real',
            numerical_qualification=dict(status='NOT_PERFORMED'),
            error=f'{type(error).__name__}: {error}', service=dict(ready=ready, stop=stop, trace=trace),
            raw_status_samples=str(run_dir / 'status-samples.jsonl'),
            raw_request_streams=[str(path) for path in sorted(run_dir.glob('request-*.ndjson'))])
        _write_json_atomic(run_dir / 'result.json', failure)
        progress.finish('FAIL', failure['error'])
        raise
    finally:
        progress.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', type=Path, required=True)
    parser.add_argument('--store', type=Path, required=True,
                        help='Disposable copy of the completed campaign Store; forks are persisted here')
    parser.add_argument('--branch', action='append', required=True,
                        help='One existing active A branch; repeat 3 or 4 times')
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--max-tokens', type=int, default=8)
    parser.add_argument('--timeout-seconds', type=float, default=1800)
    parser.add_argument('--sample-interval-seconds', type=float, default=.025)
    parser.add_argument('--ram-gib', type=int, default=96)
    parser.add_argument('--disk-gib', type=int, default=128)
    args = parser.parse_args(argv)
    result = run_real(profile=args.profile, store=args.store, source_branches=args.branch,
        run_dir=args.run_dir, max_tokens=args.max_tokens, timeout=args.timeout_seconds,
        sample_interval=args.sample_interval_seconds, ram_gib=args.ram_gib,
        disk_gib=args.disk_gib)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
