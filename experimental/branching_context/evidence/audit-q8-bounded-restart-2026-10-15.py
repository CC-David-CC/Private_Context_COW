"""Read-only independent audit of retained Q8 bounded restart evidence."""
import hashlib
import json
import math
from pathlib import Path
import struct
import sys

root = Path(sys.argv[1]).resolve(strict=True)
def read(name):
    return json.loads((root / name).read_text(encoding='utf-8'))
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

result, prepare, resume = (read(n) for n in ('result.json', 'prepare.json', 'resume.json'))
assert all(r['status'] == 'PASS' for r in (result, prepare, resume))
assert prepare['python_pid'] != resume['python_pid']
assert set(prepare['native_pids'].values()).isdisjoint(resume['native_pids'].values())
assert all(p['native_exit_success'] and all(v == 0 for v in p['native_exit_codes'].values())
           for p in (prepare, resume))
assert resume['parent_bytes_unchanged'] is True
first = {r['operation'].split('-tree-')[1].split('-')[0]: r
         for r in prepare['requests'] if '-tree-' in r['operation']}
second = {r['operation'].split('-tree-')[1].split('-')[0]: r for r in resume['requests']}
width = 248320 * 4
checks = []
for tree in ('0', '1'):
    a, b = first[tree], second[tree]
    fresh = read('fresh-tree-' + tree + '.stdout')
    outputs = a['answer']['tokens'] + b['answer']['tokens']
    assert outputs == fresh['generated']
    raw_path = root / ('fresh-tree-' + tree + '.f32.decode.f32')
    assert raw_path.stat().st_size == len(outputs) * width
    for row, count in ((a, len(a['answer']['tokens'])), (b, len(outputs))):
        actual = Path(row['state']['frontier_file']).read_bytes()
        assert len(actual) == width
        assert hashlib.sha256(actual).hexdigest() == row['state']['frontier_sha256']
        assert all(math.isfinite(v[0]) for v in struct.iter_unpack('<f', actual))
        with raw_path.open('rb') as f:
            f.seek((count - 1) * width)
            assert f.read(width) == actual
        checks.append({'tree': int(tree), 'output_count': count,
                       'frontier_sha256': hashlib.sha256(actual).hexdigest()})
    delta = b['answer']['counter_deltas']
    if b['answer']['tokens']:
        assert delta['forward_tokens'] == len(b['answer']['tokens'])
        assert delta['prefill_tokens'] == delta['prefill_chunks'] == 0
        assert delta['checkpoint_import_bytes'] > 0
        assert delta['integrity_mismatches'] == 0
    else:
        assert b['mode'] == 'already-stopped/no-inference-request' and delta is None

report = {
    'status': 'PASS_BOUNDED_Q8_RESTART_INDEPENDENT_ARTIFACT_AUDIT',
    'root': str(root), 'profile_id': result['profile_id'],
    'source_sha256': {n: sha(root / n) for n in ('result.json', 'prepare.json', 'resume.json')},
    'outputs': [len(first[t]['answer']['tokens']) + len(second[t]['answer']['tokens']) for t in ('0', '1')],
    'resumed_outputs': [len(second[t]['answer']['tokens']) for t in ('0', '1')],
    'frontier_comparisons': checks, 'unique_frontier_boundaries': 3,
    'numerical_max_error': 0.0, 'parent_bytes_unchanged': True,
    'resumed_forward_tokens': second['0']['answer']['counter_deltas']['forward_tokens'],
    'resumed_prefill_tokens': second['0']['answer']['counter_deltas']['prefill_tokens'],
    'checkpoint_import_bytes': second['0']['answer']['counter_deltas']['checkpoint_import_bytes'],
    'failures': [], 'skips': [],
    'limits': ['Greedy matching-profile execution only; not cumulative G2.',
               'Four comparisons include the same EOS frontier twice; three unique boundaries.',
               'Branch B stopped before restart; no B post-restart inference was exercised.',
               'Counters cover restore plus continuation, not an isolated pre-first-logit restore interval.',
               'No added RNG/grammar/crash-injection or every-token frontier qualification.',
               'This audit verifies recorded artifacts, not a new GPU execution.'],
}
print(json.dumps(report, indent=2))
