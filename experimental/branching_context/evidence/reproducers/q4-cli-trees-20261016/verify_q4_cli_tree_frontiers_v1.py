"""Independent, process-separated numerical audit of an executed CLI campaign.

No SUT backend is created. Only the separate uncached native oracle evaluates
tokens. Candidate frontiers are read from already-published checkpoint bytes.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys


def read(path):
    return json.loads(Path(path).read_bytes())


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while block := stream.read(8 * 1024**2):
            value.update(block)
    return value.hexdigest()


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('runtime-root', 'profile', 'fixtures', 'campaign', 'run-dir'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    runtime = args.runtime_root.resolve(strict=True)
    profile = read(args.profile)
    fixtures = read(args.fixtures)
    candidate = read(args.campaign)
    require(candidate['status'] == 'PASS_PIPELINE_NUMERICS_PENDING', 'Candidate pipeline did not pass')
    require(sha(args.profile) == fixtures['profile_sha256'] == candidate['profile_sha256'], 'Profile identity differs')
    require(sha(args.fixtures) == candidate['fixture_set_sha256'], 'Fixture set changed')
    require(candidate['execution_profile_id'] == profile['execution_profile_id'], 'Execution identity differs')
    require(candidate['model_contract_id'] == profile['model_contract_id'], 'Model identity differs')
    for service in candidate['services']:
        require(service['stop']['exit_code'] == service['worker']['native_exit_code'] == 0, 'Producer did not exit cleanly')
        for pid in (service['http_pid'], service['worker']['native_pid']):
            require(not Path('/proc', str(pid)).exists(), 'Producer PID still exists')
    for key in list(os.environ):
        if key.startswith(('STRATA_', 'CUDA_', 'NVIDIA_', 'GGML_')):
            del os.environ[key]
    os.environ.update(profile['execution'].get('environment', {}))
    sys.path[:0] = [str(runtime), str(runtime/'tools'), str(runtime/'experimental/branching_context/src'),
                   str(runtime/'experimental/branching_context/scripts')]
    from run_q4_publication_fault import _runtime_closure
    from run_root_reuse_probe import reference, compare_vectors, publish, expected_controller
    from run_cow_client_recovery import ProgressTracker, checkpoint_fingerprint
    from context_time_travel.strata_backend import _validate_execution, _validate_model
    from context_time_travel.store import Store
    from tools.strata_tokenizer import Tokenizer
    closure = _runtime_closure(profile, runtime)
    _validate_execution(profile)
    model = _validate_model(profile)
    sut = Path(profile['execution']['sut_command'][0]).resolve(strict=True)
    oracle = Path(profile['execution']['oracle_command'][0]).resolve(strict=True)
    require(sut != oracle and sha(sut) == candidate['native_sha256'], 'Oracle must be independent')
    command = profile['execution']['oracle_command']
    for flag in ('--prompt-cache', '--conversation-cache-mib'):
        require(command[command.index(flag)+1] == '0', 'Fresh oracle reuse enabled')
    cases = []
    for entry in fixtures['contexts']:
        require(sha(entry['fixture']) == entry['fixture_sha256'], 'Case changed')
        cases.append(read(entry['fixture']))
    output = args.run_dir.absolute()
    output.mkdir(mode=0o700, parents=True, exist_ok=False)
    tracker = ProgressTracker(output, 'independent-cli-tree-frontiers').start()
    record = dict(schema='cow-cli-tree-frontier-audit-v1', status='RUNNING', comparisons=0,
                  numerical_max_error=0, references=[], generations=[], failures=[],
                  campaign_sha256=sha(args.campaign), profile_sha256=sha(args.profile),
                  fixture_set_sha256=sha(args.fixtures), script_sha256=sha(Path(__file__)),
                  execution_profile_id=profile['execution_profile_id'], model_contract_id=profile['model_contract_id'],
                  oracle_sha256=sha(oracle), model_verification=model,
                  checked_boundaries='Candidate saved prefix and final consumed frontier only; fresh repeat every dumped frontier',
                  controller_scope='Deterministic greedy; full canonical controller checked',
                  source_closure=closure, oracle_checkpoint_imports=0)
    result = output/'result.json'
    publish(result, record)
    store = Store(Path(candidate['store']))
    tokenizer = Tokenizer.from_gguf(profile['model_path'])
    groups = {}
    for generation in candidate['generations']:
        groups.setdefault((generation['context_index'], generation['question']), []).append(generation)
    try:
        def exact(left, right, label):
            error = compare_vectors(left, right, label, output)
            record['comparisons'] += 1
            record['numerical_max_error'] = max(record['numerical_max_error'], error)

        for (index, question), generations in sorted(groups.items()):
            case = cases[index]
            prefix = case['root_token_ids']
            suffix = case['branches'][question]['suffix_token_ids']
            ids = prefix + suffix
            count = max(g['output_count'] for g in generations)
            refs = []
            for repeat in range(2):
                name = f'tree{index}-{question}-fresh{repeat}'
                tracker.phase(name)
                fresh = reference(profile, output/name, name, ids, [len(prefix), len(suffix)], count)
                tracker.native_started(fresh['pid'])
                require(not Path('/proc', str(fresh['pid'])).exists(), 'Oracle PID survived wait')
                compact = {key: value for key, value in fresh.items() if key not in ('prompt_frontiers', 'generated_frontiers')}
                compact['prompt_frontier_sha256'] = [hashlib.sha256(v).hexdigest() for v in fresh['prompt_frontiers']]
                compact['generated_frontier_sha256'] = [hashlib.sha256(v).hexdigest() for v in fresh['generated_frontiers']]
                record['references'].append(compact)
                refs.append(fresh)
                publish(result, record)
            require(refs[0]['tokens'] == refs[1]['tokens'], 'Independent greedy repetition differs')
            for field in ('prompt_frontiers', 'generated_frontiers'):
                require(len(refs[0][field]) == len(refs[1][field]), 'Incomplete repeated frontiers')
                for position, (left, right) in enumerate(zip(refs[0][field], refs[1][field])):
                    exact(left, right, f'tree{index}-{question}-{field}-{position}')
            root = candidate['roots'][str(index)]
            with store.read_lease():
                source = root['head']['tip']
                require(vars(store.head(root['branch'])) == root['head'], 'Original root moved')
                require(checkpoint_fingerprint(Path(candidate['store']), source)['files'] == root['checkpoint_before']['files'], 'Original prefix bytes changed')
                manifest = read(Path(candidate['store'])/'checkpoints'/(source+'.json'))['manifest']
                payload = Path(candidate['store'])/'checkpoints'/(manifest['payload_sha256']+'.state')
                with payload.open('rb') as stream:
                    stream.seek(manifest['state_bytes'])
                    exact(stream.read(), refs[0]['prompt_frontiers'][0], f'tree{index}-{question}-saved-prefix')
                for generation in generations:
                    require(generation['output_tokens'] == refs[0]['tokens'][:generation['output_count']], 'CLI output tokens differ from fresh reference')
                    frontier = Path(generation['frontier_file'])
                    require(sha(frontier) == generation['frontier_sha256'], 'Candidate frontier bytes changed')
                    exact(frontier.read_bytes(), refs[0]['generated_frontiers'][generation['output_count']-1], generation['branch']+'-final')
                    snapshot = store.get_snapshot(generation['snapshot'])
                    require(snapshot['parent'] == source, 'Candidate belongs to wrong tree')
                    require(store.lineage(generation['snapshot'])[1] == ids+generation['output_tokens'], 'Canonical history differs')
                    expected = expected_controller(generation['operation'], ids, generation['output_tokens'], tokenizer,
                                                   profile['execution']['eos_tokens'], 248320)
                    require(snapshot['controller'] == expected, 'Canonical controller differs')
                    counters = generation['counter_deltas']
                    require(counters['forward_tokens'] == len(suffix)+generation['output_count'] and counters['prefill_tokens'] == len(suffix), 'Saved prefix evaluated again')
                    record['generations'].append(dict(branch=generation['branch'], output_count=generation['output_count'],
                                                     tokens_exact=True, controller_exact=True, final_frontier_exact=True))
            tracker.phase(name+'-audited', completed=name)
            publish(result, record)
        record['status'] = 'PASS'
        record['all_reference_pids_absent'] = all(not Path('/proc',str(row['pid'])).exists() for row in record['references'])
        require(record['all_reference_pids_absent'], 'Reference PID exists at completion')
    except BaseException as exc:
        record['status'] = 'FAIL'
        if hasattr(exc, 'maximum'):
            record['numerical_max_error'] = exc.maximum
        record['failures'].append(dict(type=type(exc).__name__, message=str(exc)))
        raise
    finally:
        tracker.finish(record['status'], str(record['failures']) if record['failures'] else None)
        tracker.close()
        publish(result, record)
        print(json.dumps({key: record[key] for key in ('status','comparisons','numerical_max_error','failures')}), flush=True)


if __name__ == '__main__':
    main()
