"""Read-only audit of the executed Q4 source campaign; not a timing run."""
import hashlib
import json
from pathlib import Path


root = Path('/dev/shm/context-cow-digest-q4-64k-source-20261015-v1')
art = Path('/home/dflanag3/spatial-artifacts/checkpoint-digest-q4-attestation-20261015')
result = json.loads((root / 'result.json').read_text())
prepare = json.loads((root / 'prepare.json').read_text())
resume = json.loads((root / 'resume.json').read_text())
assert result['status'] == prepare['status'] == resume['status'] == 'PASS'
assert result['failures'] == result['skips'] == result['xfails'] == []
assert result['numerical_max_error'] == 0
assert len(result['comparisons']) == 6
assert all(row['outputs'] == 128 and row['resumed_outputs'] == 124
           and row['max_absolute_error'] == 0 for row in result['comparisons'])
for phase in ('prepare', 'resume'):
    assert result['phases'][phase]['exit_code'] == 0
    assert result['phases'][phase]['native_exit_codes'] == {'0': 0}
for name in ('command.exit', 'wrapper.exit'):
    assert (art / 'source64k-command-v1' / name).read_text().strip() == '0'

root_frontiers = []
for index, state in enumerate(prepare['roots']):
    assert len(state['history']) == 65536
    saved = Path(state['frontier_file']).read_bytes()
    assert len(saved) == 993280
    for tree in (2 * index, 2 * index + 1):
        with (root / f'fresh-tree-{tree}.f32.requests.f32').open('rb') as stream:
            fresh = stream.read(len(saved))
        assert saved == fresh, ('root frontier', index, tree)
        root_frontiers.append(dict(root=index, tree=tree, bytes=len(saved),
            sha256=hashlib.sha256(saved).hexdigest(), max_absolute_error=0))

work = []
assert len(resume['requests']) == 6
for request in resume['requests']:
    delta = request['answer']['counter_deltas']
    assert len(request['answer']['tokens']) == 124
    assert delta['forward_tokens'] == 124
    assert delta['prefill_tokens'] == delta['prefill_chunks'] == 0
    assert delta['integrity_mismatches'] == 0
    assert delta['checkpoint_import_bytes'] > 0
    work.append(dict(operation=request['operation'], outputs=124,
        forward_tokens=delta['forward_tokens'], prefill_tokens=delta['prefill_tokens'],
        import_bytes=delta['checkpoint_import_bytes'], integrity_mismatches=0))

audit = dict(status='PASS', scope='greedy matching-schedule source correctness; not timing or durable-disk qualification',
    execution_profile_id=result['profile_id'], model_contract_id=result['model_contract_id'],
    source_output=str(root), command_exit=0, wrapper_exit=0,
    generated_tokens_compared=768, generated_restart_tokens=744,
    sampled_generated_frontier_comparisons=12,
    sampled_generated_boundaries=[4, 128], root_frontier_comparisons=root_frontiers,
    numerical_max_error=0, resume_work=work, native_process_phases=result['phases'],
    failures=[], skips=[], xfails=[],
    raw_sha256={name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                for name in ('result.json', 'prepare.json', 'resume.json', 'profile.json')},
    limitation='RAM-backed source checkpoints; non-greedy controllers and every-output frontier certification excluded')
with (art / 'source64k-independent-audit-v1.json').open('x') as stream:
    json.dump(audit, stream, indent=2)
    stream.write('\n')
print(json.dumps(dict(status=audit['status'], tokens=768, restarted_tokens=744,
    generated_frontier_checks=12, root_frontier_checks=6, numerical_max_error=0)))
