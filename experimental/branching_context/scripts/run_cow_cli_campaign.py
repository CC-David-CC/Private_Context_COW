"""Exercise the actual terminal entrypoint against a real pinned COW service."""
import argparse
import codecs
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from run_cow_client_recovery import (ProgressTracker, _private_run_dir, _ready_url,
    _start_service, _trace_worker, checkpoint_fingerprint)
from context_time_travel.store import Store
from context_time_travel.controller import consume_many
from tools.strata_tokenizer import Tokenizer


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--profile', type=Path, required=True)
    parser.add_argument('--fixture-set', type=Path, required=True)
    parser.add_argument('--run-dir', type=Path, required=True)
    args = parser.parse_args()
    output = _private_run_dir(args.run_dir)
    store_dir = output/'store'
    repo = Path(__file__).resolve().parent
    # The runner explicitly supplies the frozen runtime root, not an inferred
    # location based on this external artifact script.
    repo = Path(os.environ['COW_RUNTIME_REPO']).resolve()
    cli = repo/'tools/cow_cli.py'
    profile = json.loads(args.profile.read_bytes())
    fixture_set = json.loads(args.fixture_set.read_bytes())
    fixture_path = args.fixture_set.parent/fixture_set['contexts'][0]['fixture_file']
    fixture = json.loads(fixture_path.read_bytes())
    assert sha(fixture_path) == fixture_set['contexts'][0]['fixture_sha256']
    context = output/'code-context.txt'
    # Exercise the documented CLI's system-context file representation. This
    # deliberately is not labelled identical to the browser's user attachment.
    context.write_text(fixture['root_messages'][0]['content'], encoding='utf-8', newline='\n')
    tokenizer = Tokenizer.from_gguf(profile['model_path'])
    progress = ProgressTracker(output, 'real').start()
    results, services = [], []
    service = None
    store = None
    index = 0
    ordinal = 1

    def stats():
        trace = output/f'native-trace-{ordinal}.jsonl'
        rows = [json.loads(line) for line in trace.read_text().splitlines() if line]
        states = [r['response'] for r in rows if r.get('command') == 'stats']
        return states[-1] if states else {}

    def invoke(label, *command, expect=0, mutation=True):
        nonlocal index
        index += 1
        progress.phase(label)
        start = time.monotonic()
        before = stats()
        args_list = [sys.executable, str(cli), '--url', url, '--timeout', '300', *command]
        value = subprocess.run(args_list, cwd=repo, capture_output=True, timeout=320)
        elapsed = time.monotonic()-start
        (output/f'{index:02}-{label}.stdout').write_bytes(value.stdout)
        (output/f'{index:02}-{label}.stderr').write_bytes(value.stderr)
        assert value.returncode == expect, (label, value.returncode, value.stderr.decode('utf-8','replace'))
        text = value.stdout.decode('utf-8')
        err = value.stderr.decode('utf-8')
        assert 'operation_id=' not in text and 'durable receipt:' not in text
        row = dict(label=label, command=args_list, exit_code=value.returncode,
            elapsed_seconds=elapsed, stdout_sha256=hashlib.sha256(value.stdout).hexdigest(),
            stderr_sha256=hashlib.sha256(value.stderr).hexdigest())
        if expect == 0 and mutation:
            assert 'durable receipt:\n' in err, (label, err)
            result = json.loads(err.split('durable receipt:\n',1)[1])
            row['receipt'] = result
        elif expect == 0:
            result = json.loads(text)
        else:
            assert 'durable receipt:' not in err
            result = None
        after = stats()
        row['counter_deltas'] = {k:after.get(k,0)-before.get(k,0) for k in
            ('forward_tokens','prefill_tokens','loaded_file_bytes','integrity_mismatches')}
        results.append(row)
        return result, row

    def head(branch):
        return store.head(branch)

    def generated(label, command, branch, *extra, source_idle=False):
        source = head(branch)
        source_record = store.get_snapshot(source.tip)
        before_files = checkpoint_fingerprint(store_dir, source.tip)['files']
        receipt, row = invoke(label, command, '--branch', branch, '--max-tokens','128',
            '--operation-id',f'op-cli-{label}', *extra)
        assert receipt['branch']['name'] == branch
        assert receipt['branch']['revision'] == source.revision+1
        target = head(branch)
        assert target.tip == receipt['snapshot']
        final = store.get_snapshot(target.tip)
        assert final['parent'] == source.tip
        root, old_ids = store.lineage(source.tip)
        new_root, new_ids = store.lineage(target.tip)
        assert root == new_root and new_ids[:len(old_ids)] == old_ids
        suffix = new_ids[len(old_ids):]
        counts = row['counter_deltas']
        assert counts['forward_tokens'] == len(suffix)
        assert counts['integrity_mismatches'] == 0
        prefill = counts['prefill_tokens']
        if command == 'continue':
            assert prefill == 0
        else:
            assert prefill > 0
        tokens = suffix[prefill:]
        assert 0 < len(tokens) <= 128
        if command == 'continue':
            eos = source_record['controller']['policy']['eos_tokens']
            expected = consume_many(source_record['controller'],
                [(token,tokenizer.token_bytes(token),token in eos) for token in tokens])
            assert final['controller'] == expected
        decoder = codecs.getincrementaldecoder('utf-8')(errors='replace')
        pending = '' if source_idle else source_record['controller'].get('output_pending_hex','')
        decoder.setstate((bytes.fromhex(pending),0))
        parts = []
        eos = final['controller']['policy']['eos_tokens']
        for token in tokens:
            if token in eos:
                parts.append(decoder.decode(b'',final=True))
                break
            parts.append(decoder.decode(tokenizer.token_bytes(token),final=False))
        expected_text = ''.join(parts)
        from tools.cow_cli import safe_text
        actual = (output/f'{index:02}-{label}.stdout').read_bytes().decode('utf-8')
        assert actual == safe_text(expected_text)+'\n', (label,'CLI output differs from canonical token decoding')
        assert checkpoint_fingerprint(store_dir,source.tip)['files'] == before_files
        row.update(source=source.tip,snapshot=target.tip,output_tokens=tokens,
            canonical_lineage_exact=True,source_checkpoint_bytes_unchanged=True,
            canonical_stdout_exact=True,controller_status=final['controller']['status'])
        return row

    def stop_service():
        nonlocal service
        closed = service.stop()
        worker = _trace_worker(output/f'pool-trace-{ordinal}.json',closed['pid'])
        assert closed['exit_code'] == 0 and worker['native_exit_code'] == 0
        assert closed['graceful'] is True
        progress.native_started(worker['native_pid'])
        services.append(dict(ordinal=ordinal,closed=closed,worker=worker))
        service = None

    try:
        service, ready, http = _start_service('real',store_dir,output,ordinal,
            profile=args.profile,timeout=300,ram_gib=96,disk_gib=128,progress=progress)
        url = _ready_url(ready)
        store = Store(store_dir)
        invoke('status','status',mutation=False)
        namespace, _ = invoke('namespace','namespace','--title','CLI real source test',
            '--operation-id','op-cli-namespace')
        created, creation = invoke('load','load','--namespace-id',namespace['namespace_id'],
            '--branch','CliBase','--file',str(context),'--operation-id','op-cli-load')
        base = head('CliBase')
        base_record, base_ids = store.lineage(base.tip)
        assert base.tip == base.root == base_record == created['snapshot']
        assert creation['counter_deltas']['prefill_tokens'] == len(base_ids)
        for name in ('CliA','CliB'):
            invoke('fork-'+name,'fork','--source-branch','CliBase','--name',name,
                '--operation-id','op-cli-fork-'+name)
            assert head(name).tip == base.tip
        a = generated('ask-a','ask','CliA','--text',fixture['branches']['A']['question'],source_idle=True)
        a128 = head('CliA').tip
        b = generated('ask-b','ask','CliB','--text',fixture['branches']['B']['question'],source_idle=True)
        assert head('CliA').tip == a128 and head('CliBase') == base
        if store.get_snapshot(a128)['controller']['status'] != 'active':
            raise RuntimeError('A stopped before the planned active-resume fixture; preserve early-stop result')
        first = generated('resume-before-restart','continue','CliA')
        invoke('rewind-a','rewind','--branch','CliA','--snapshot',a128,
            '--operation-id','op-cli-rewind-a')
        assert head('CliA').tip == a128
        invoke('history','history','--branch','CliA',mutation=False)
        invoke('branches','branches',mutation=False)
        invoke('contexts','contexts',mutation=False)
        stopped = stop_service()
        ordinal = 2
        service, ready, http = _start_service('real',store_dir,output,ordinal,
            profile=args.profile,timeout=300,ram_gib=96,disk_gib=128,progress=progress)
        url = _ready_url(ready)
        second = generated('resume-after-restart','continue','CliA')
        assert first['output_tokens'] == second['output_tokens']
        assert first['snapshot'] == second['snapshot']
        assert second['counter_deltas']['prefill_tokens'] == 0
        assert second['counter_deltas']['loaded_file_bytes'] > 0
        assert head('CliBase') == base
        stop_service()
        assert services[0]['worker']['native_pid'] != services[1]['worker']['native_pid']
        report = dict(status='BOUNDED_REAL_CLI_PASS',scope='Actual CLI subprocesses against pinned native COW service',
            profile_sha256=sha(args.profile),execution_profile_id=profile['execution_profile_id'],
            cli_sha256=sha(cli),harness_sha256=sha(__file__),fixture_sha256=sha(fixture_path),
            context_file_sha256=sha(context),root_tokens=len(base_ids),max_outputs_per_request=128,
            calls=results,services=services,restart_tokens_exact=True,
            canonical_resume_snapshot_exact=True,numerical_qualification='NOT_PERFORMED_NO_FRESH_ORACLE',
            limitations=['System-role file root differs from browser attachment root.',
                'Greedy profile only; no new sampler, tool-call, crash-injection or logits qualification.',
                'CLI subprocess times include Python startup and durable service completion; no TTFT/TPOT measured.'])
        (output/'result.json').write_text(json.dumps(report,indent=2)+'\n')
        progress.finish('PASS')
        print(json.dumps(dict(status=report['status'],root_tokens=len(base_ids),calls=len(results),
            outputs=[len(row['output_tokens']) for row in results if 'output_tokens' in row],
            native_pids=[row['worker']['native_pid'] for row in services])))
    except BaseException as error:
        if service is not None:
            try: stop_service()
            except BaseException as cleanup:
                results.append(dict(cleanup_error=str(cleanup)))
        (output/'result.json').write_text(json.dumps(dict(status='FAIL',error=str(error),
            calls=results,services=services),indent=2)+'\n')
        progress.finish('FAIL',str(error))
        raise
    finally:
        progress.close()


if __name__ == '__main__':
    main()
