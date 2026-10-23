"""Owned, bounded PCC serving sessions for the optional S3 candidate evaluator.

This uses cow_server's real profile, leases and compatibility frontend. It is
an operator adapter, not a workflow runner or a source of training examples.
"""
from contextlib import contextmanager
from pathlib import Path
import copy
import functools
import hashlib
import json
import os
import secrets
import signal
import statistics
import subprocess
import sys
import time
import urllib.request


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def save(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')


def runtime_contract(profile, configuration):
    execution = copy.deepcopy(profile['execution'])
    execution.pop('adapter', None)
    return dict(build=profile['native_build'], execution=execution,
                geometry=profile['geometry'], bridge=profile['bridge_library'],
                thinking=configuration.get('openai_thinking', False),
                output_limit=configuration['openai_max_output'],
                tokenizer=configuration.get('tokenizer_backend', 'python'))


def admit_samples(samples, identity, *, maximum_activity, minimum_free_mib):
    if len(samples) != 5:
        raise ValueError('Exactly five spaced GPU samples are required')
    selected = []
    for sample in samples:
        matches = [row for row in sample if row['uuid'] == identity]
        if len(matches) != 1:
            raise ValueError('Selected physical GPU is missing or ambiguous')
        selected.append(matches[0])
    if (len({r['index'] for r in selected}) != 1
            or statistics.mean(r['activity'] for r in selected) >= maximum_activity
            or min(r['free_mib'] for r in selected) < minimum_free_mib):
        raise RuntimeError('Selected GPU does not meet the declared activity/memory admission')
    return selected[0]['index']


def gpu_samples():
    raw = subprocess.check_output(['nvidia-smi',
        '--query-gpu=index,uuid,memory.free,utilization.gpu',
        '--format=csv,noheader,nounits'], text=True, timeout=15)
    return [dict(index=int(a), uuid=b.strip(), free_mib=int(c), activity=int(d))
            for a, b, c, d in (line.split(',') for line in raw.splitlines())]


def gpu_processes():
    raw = subprocess.check_output(['nvidia-smi',
        '--query-compute-apps=pid,gpu_uuid,used_memory', '--format=csv,noheader,nounits'],
        text=True, timeout=15)
    return [dict(pid=int(a), uuid=b.strip(), memory_mib=int(c))
            for a, b, c in (line.split(',') for line in raw.splitlines())]


def descendants(pid):
    result = {pid}; pending = [pid]
    while pending:
        parent = pending.pop()
        try:
            children = Path('/proc', str(parent), 'task', str(parent), 'children').read_text()
        except FileNotFoundError:
            continue
        for child in map(int, children.split()):
            if child not in result:
                result.add(child); pending.append(child)
    return result


def stop_owned(process):
    if process.poll() is None:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=10)


class DenseCandidateSession:
    """A single-GPU dense PCC session selected by an owner's prepared profile.

    Construct after freezing source and creating the baseline server directory.
    Sessions preserve raw traces, exact prompt token counts, resource samples and
    settlement. A timeout/error raises; it never becomes a failed model answer.
    S3 is imported only when the optional session is used.
    """
    def __init__(self, directory, *, python, gpu_uuid, deadline_seconds=900,
                 maximum_activity=50, minimum_free_mib=32768, reserve_mib=8192):
        self.configuration = json.loads((Path(directory) / 'server.json').read_bytes())
        self.profile = json.loads(Path(self.configuration['profile']).read_bytes())
        if (self.configuration['host'] != '127.0.0.1'
                or self.configuration.get('openai_port') is None
                or not 1 <= self.configuration.get('openai_max_output', 0) <= 32768
                or self.configuration.get('training_config') is not None
                or self.profile['execution'].get('physical_sharing')
                or len(self.profile['execution']['GPU']) != 1
                or self.profile.get('adapter') is not None):
            raise ValueError('Use an unadapted, dedicated, loopback dense evaluation configuration')
        if (type(deadline_seconds) is not int or not 60 <= deadline_seconds <= 3600
                or not 0 < maximum_activity <= 100 or not 0 < reserve_mib < minimum_free_mib
                or not gpu_uuid.startswith('GPU-')):
            raise ValueError('Explicit bounded owner GPU/time limits required')
        # A venv's python is often a symlink to the system binary. Resolving
        # that link discards its environment and changes available dependencies.
        interpreter = Path(python).absolute()
        if not interpreter.is_file() or not os.access(interpreter, os.X_OK):
            raise ValueError('Existing executable Python interpreter required')
        self.python = str(interpreter)
        self.gpu_uuid = gpu_uuid
        self.deadline = deadline_seconds
        self.maximum_activity = maximum_activity
        self.minimum_free_mib, self.reserve_mib = minimum_free_mib, reserve_mib
        self.contract = runtime_contract(self.profile, self.configuration)
        self.runtime_id = digest(self.contract)

    def _profile(self, candidate, output):
        from experimental.branching_context.scripts.prepare_qwen35_profile import make_profile
        p = self.profile; execution = p['execution']
        model = p['model_path']; adapter = None
        if candidate:
            if candidate['mode'] == 'full':
                model = candidate['candidate_path']
            elif candidate['mode'] == 'lora':
                adapter = candidate['candidate_path']
            else:
                raise ValueError('Unknown candidate mode')
        math = ('f32-reference' if execution.get('f32_attention') else
                'f32-activations' if execution.get('f32_activations') else 'native')
        profile = make_profile(model, Path(p['bridge_library']).parent, output,
            execution['capacity'], execution['prefill_chunk'], execution['threads'],
            gpus=execution['GPU'], lora=adapter, math_profile=math)
        if runtime_contract(profile, self.configuration) != self.contract:
            raise ValueError('Candidate serving runtime differs from the frozen baseline')
        expected_base = candidate['candidate_sha256'] if candidate and candidate['mode'] == 'full' else p['model_weights_sha256']
        expected_adapter = candidate['candidate_sha256'] if candidate and candidate['mode'] == 'lora' else None
        if (profile['model_weights_sha256'] != expected_base
                or profile.get('adapter', {}).get('sha256') != expected_adapter):
            raise ValueError('Profile does not select the exact requested candidate')
        return profile, dict(base_sha256=expected_base, adapter_sha256=expected_adapter,
                             runtime_id=self.runtime_id)

    @contextmanager
    def __call__(self, candidate, path, worker_fd):
        from sss.tool_benchmark import OpenAIModel
        from serve.cow_http import model_text_components
        from serve.cow_compat import Frontend
        path = Path(path)
        directory = path / 'server'; directory.mkdir(mode=0o700, exist_ok=False)
        profile, receipt = self._profile(candidate, directory / 'profile.json')
        configuration = dict(self.configuration, profile=str(directory / 'profile.json'),
            store=str(directory / 'store'), api_key_file=str(directory / 'api-key.txt'), ephemeral=True)
        key = 'pcc_' + secrets.token_urlsafe(32)
        key_path = directory / 'api-key.txt'; key_path.write_text(key + '\n'); key_path.chmod(0o600)
        save(directory / 'server.json', configuration)
        samples = []
        for index in range(5):
            samples.append(gpu_samples())
            if index < 4:
                time.sleep(2)
        save(path / 'gpu-admission.json', samples)
        physical = admit_samples(samples, self.gpu_uuid,
            maximum_activity=self.maximum_activity, minimum_free_mib=self.minimum_free_mib)
        if profile['execution']['GPU'] != [physical]:
            raise ValueError('Physical GPU no longer matches the frozen profile')
        components = model_text_components(profile['model_path'], {'enable_thinking': True},
            model_contract_id=profile['model_contract_id'], mixed_inputs=True)
        name = configuration['openai_model']; maximum = configuration['openai_max_output']
        thinking = configuration.get('openai_thinking', False)
        preparer = Frontend(object(), components['render'].__self__, profile, name, maximum, thinking=thinking)
        endpoint = 'http://127.0.0.1:' + str(configuration['openai_port'])
        client = OpenAIModel(endpoint + '/v1', name, key, maximum=maximum, seed=0, timeout=90)
        failed = []

        def measured_model(messages, tools, intent):
            if failed:
                raise RuntimeError('Inference stopped after the first transport/identity failure')
            body = dict(model=name, messages=messages, tools=tools, tool_choice='auto',
                parallel_tool_calls=False, temperature=0., max_tokens=maximum, seed=0, stream=False)
            plan = preparer.prepare(body, 'chat')
            save(intent.with_name(intent.stem + '-tokenization.json'), dict(
                prompt_tokens=len(plan['full']), full_token_sha256=digest(list(plan['full'])),
                closed_token_sha256=digest(list(plan['closed'])), output_reserve=plan['bound'], thinking=thinking))
            try:
                message, usage = client(messages, tools, intent)
                if usage.get('prompt_tokens') != len(plan['full']):
                    raise ValueError('Server prompt tokens differ from the exact local renderer')
                return message, usage
            except Exception as error:
                failed.append(type(error).__name__)
                raise

        root = Path(__file__).resolve().parents[1]
        environment = dict(os.environ, PYTHONPATH=str(root) + os.pathsep + os.environ.get('PYTHONPATH', ''))
        owned = set(); started = time.monotonic()
        with (path / 'server.log').open('x') as log:
            process = subprocess.Popen(['timeout', '--signal=TERM', '--kill-after=15s', str(self.deadline) + 's',
                self.python, '-m', 'tools.pcc_foundation_session', str(directory),
                str(path / 'frontend-trace-private.jsonl')], cwd=root, env=environment,
                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                start_new_session=True, pass_fds=(worker_fd,))
            save(path / 'server-launch.json', dict(pid=process.pid,
                start_ticks=Path('/proc', str(process.pid), 'stat').read_text().split()[21],
                deadline_seconds=self.deadline, gpu_uuid=self.gpu_uuid))
            try:
                ready = False
                for attempt in range(90):
                    owned.update(descendants(process.pid))
                    if process.poll() is not None:
                        raise RuntimeError('Owned candidate server exited before readiness')
                    try:
                        request = urllib.request.Request(endpoint + '/v1/models', headers={'Authorization': 'Bearer ' + key})
                        with urllib.request.urlopen(request, timeout=2) as response:
                            ready = json.load(response)['data'][0]['id'] == name
                    except (OSError, ValueError, KeyError):
                        ready = False
                    if ready:
                        break
                    time.sleep(2)
                if not ready:
                    raise RuntimeError('Candidate startup exceeded its bounded readiness window')
                owned.update(descendants(process.pid))
                allocated = [row for row in gpu_processes() if row['pid'] in owned]
                save(path / 'gpu-owned-ready.json', allocated)
                if not allocated or any(row['uuid'] != self.gpu_uuid for row in allocated):
                    raise RuntimeError('Candidate GPU allocation differs from the selected identity')
                if next(row['free_mib'] for row in gpu_samples() if row['uuid'] == self.gpu_uuid) < self.reserve_mib:
                    raise RuntimeError('Candidate serving memory headroom is below the owner reserve')
                yield measured_model, receipt
            finally:
                owned.update(descendants(process.pid))
                stop_owned(process)
                remaining = [row for row in gpu_processes() if row['pid'] in owned]
                save(path / 'settlement.json', dict(seconds=time.monotonic() - started,
                    process_exit=process.returncode, remaining_gpu_processes=remaining))
                if remaining:
                    raise RuntimeError('Owned candidate GPU allocation did not settle')


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('trace', type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    sys.path[:0] = [str(root), str(root / "experimental/branching_context/src")]
    import serve.cow_compat as compatibility
    from tools import cow_server
    compatibility.Frontend = functools.partial(compatibility.Frontend, trace=args.trace)
    sys.argv = ['cow-server', str(args.directory)]
    cow_server.main()


if __name__ == '__main__':
    main()
