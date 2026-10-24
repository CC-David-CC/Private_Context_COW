"""Exercise actual allocator admission, ordinary updates and immutable reloads.

CPU is the default. CUDA callers must first admit and lease the selected GPUs.
Tiny fixtures establish mechanics, never Wen memory capacity or language quality.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import subprocess
from .make_cpu_fixture import make_fixture


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(executable, output, devices=('CPU',), reference_executable=None):
    os.umask(0o077)
    output = Path(output)
    output.mkdir(mode=0o700, parents=True, exist_ok=False)
    model = make_fixture(output/'fixture.gguf', context_length=2048)
    original = sha(model)
    devices = list(devices)

    def execute(label, request, command='--train', error=None, binary=None):
        path = output/(label+'.json')
        path.write_text(json.dumps(request)+'\n')
        result = subprocess.run([str(binary or executable), command, str(path)],
                                capture_output=True, text=True, timeout=120)
        (output/(label+'.stdout')).write_text(result.stdout)
        (output/(label+'.stderr')).write_text(result.stderr)
        if error:
            assert result.returncode != 0 and error in result.stderr, result.stderr[-3000:]
            assert 'NATIVE_STANDARD_STEP' not in result.stderr
            assert not Path(request['candidate']).exists()
            return None
        assert result.returncode == 0, result.stderr[-3000:]
        return json.loads(result.stdout)

    checks = []
    negatives = 0
    for mode in ('lora', 'full'):
        for length in (64, 375):
            tokens = [i % 128 for i in range(length+1)]
            example = dict(id='owned-sequence', tokens=tokens[:-1], targets=tokens[1:],
                           loss_mask=[i >= length-4 for i in range(length)],
                           verified=True, provenance_sha256='a'*64)
            base = dict(model=str(model), base_sha256=original, mode=mode, rank=2,
                        devices=devices, device_budget_bytes={d: 8 << 30 for d in devices},
                        reserve_bytes=128 << 20, host_reserve_bytes=128 << 20,
                        examples=[example], batch_size=1, steps=1, learning_rate=.0001,
                        sequence_profile='dense-short-v1',
                        block_devices=[i % len(devices) for i in range(4)])
            reports = {}
            candidates = {}
            for profile in ('conservative-v1', 'allocator-v1'):
                label = f'{mode}-{length}-{profile}'
                request = dict(base, candidate=str(output/(label+'.gguf')))
                if profile == 'allocator-v1':
                    request['memory_profile'] = profile
                report = execute(label, request)
                reports[profile] = report
                candidates[profile] = request['candidate']
                if profile == 'allocator-v1':
                    assert all(p['checked_graphs'] > 0 for p in report['allocation_peaks'])
                    assert any(p['planned_graph_buffer_bytes'] > 0 for p in report['allocation_peaks'])
                else:
                    assert 'memory_profile' not in report and 'allocation_peaks' not in report
                reload = dict(request, evaluate_only=True,
                              candidate=str(output/(label+'-unused.gguf')))
                if mode == 'full':
                    reload.update(model=request['candidate'], base_sha256=sha(request['candidate']))
                else:
                    reload.update(initial_adapter=request['candidate'],
                                  initial_adapter_sha256=sha(request['candidate']))
                loaded = execute(label+'-reload', reload)
                assert abs(loaded['losses'][0]-report['steps'][0]['after']) < 1e-5
                assert not Path(reload['candidate']).exists()
            ordinary = reports['conservative-v1']['steps'][0]
            bounded = reports['allocator-v1']['steps'][0]
            for field in ('before', 'after'):
                assert abs(ordinary[field]-bounded[field]) < 1e-5
            # Same operations and allocation order must preserve candidate bytes.
            assert sha(candidates['conservative-v1']) == sha(candidates['allocator-v1'])
            if reference_executable:
                request = dict(base, candidate=str(output/f'{mode}-{length}-previous.gguf'))
                previous = execute(f'{mode}-{length}-previous', request, binary=reference_executable)
                assert sha(request['candidate']) == sha(candidates['conservative-v1'])
                for field in ('before', 'after'):
                    assert previous['steps'][0][field] == ordinary[field]

            # A budget that admits persistent state but cannot fit the first
            # measured graph must refuse before any optimizer update or save.
            lines = (output/f'{mode}-{length}-allocator-v1.stderr').read_text().splitlines()
            plans = [json.loads(s.split(' ', 1)[1]) for s in lines
                     if s.startswith('GRAPH_ALLOCATION_PLAN ')]
            first = next(p for p in plans if p['buffer_bytes'] > 0)
            persistent = next(p for p in reports['allocator-v1']['memory_plan']
                              if p['device'] == first['device'])['admission_estimate']
            budgets = dict(base['device_budget_bytes'])
            budgets[first['device']] = persistent + first['buffer_bytes']//2
            refused = dict(base, memory_profile='allocator-v1', device_budget_bytes=budgets,
                           candidate=str(output/f'{mode}-{length}-refused.gguf'))
            stage1 = execute(f'{mode}-{length}-persistent-only', refused, '--admit-training')
            assert stage1['status'] == 'PERSISTENT_ADMISSION_PASS'
            assert not stage1['weights_loaded'] and not stage1['graph_allocation_qualified']
            assert all(p['checked_graphs'] == 0 for p in stage1['allocation_peaks'])
            execute(f'{mode}-{length}-graph-refused', refused,
                    error='Graph allocation admission refused')
            negatives += 1
            checks.append(dict(mode=mode, length=length, before=bounded['before'], after=bounded['after'],
                               candidate_sha256=sha(candidates['allocator-v1']),
                               candidate_bytes_identical=True, reload_agreement=True,
                               allocation_peaks=reports['allocator-v1']['allocation_peaks']))
            assert sha(model) == original
    for label, changes, error in [
            ('unknown-profile', dict(memory_profile='unknown'), 'Unknown memory admission profile'),
            ('unqualified-sequence', dict(memory_profile='allocator-v1', sequence_profile='short-v1'),
             'Allocator admission is limited'),
            ('persistent-refused', dict(memory_profile='allocator-v1',
                                       device_budget_bytes={d: 1 for d in devices}),
             'Training memory admission refused')]:
        execute(label, dict(base, **changes, candidate=str(output/(label+'.gguf'))), error=error)
        negatives += 1
    result = dict(status='PASS', checks=checks, negative_cases=negatives, devices=devices,
                  previous_binary_compared=bool(reference_executable), base_unchanged=sha(model) == original,
                  scope='tiny native allocator admission, ordinary updates and reload; not Wen qualification')
    (output/'result.json').write_text(json.dumps(result, indent=2)+'\n')
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--executable', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--devices', nargs='+', default=['CPU'])
    p.add_argument('--reference-executable', type=Path)
    a = p.parse_args()
    print(json.dumps(run(a.executable, a.output, a.devices, a.reference_executable)))
