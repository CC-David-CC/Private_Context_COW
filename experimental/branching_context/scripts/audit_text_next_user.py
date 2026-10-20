"""Compare executed next-user HTTP state with a separate uncached native oracle."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics

from context_time_travel import controller
from native_request_probe import publish
from tools.strata_tokenizer import Tokenizer


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--oracle-frontier', type=Path, required=True,
                        help='Oracle dump stem used by STRATA_DUMP_FIRST_LOGITS')
    parser.add_argument('--oracle-receipt', type=Path, required=True,
                        help='Executed oracle process/build/command receipt')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    case = args.directory
    raw = json.loads((case/'result.json').read_text())
    assert raw['status'] == 'CONTROL_PASS_ORACLE_PENDING'
    assert raw['native_exit'] == 0
    profile = json.loads((case/'profile.json').read_text())
    fixture = json.loads((case/'oracle-fixture.json').read_text())
    receipt = json.loads(args.oracle_receipt.read_text())
    assert receipt['exit_code'] == 0 and receipt['status'] == 'PASS_FRESH_TWO_TURN_ORACLE'
    assert receipt['profile_sha256'] == sha(case/'profile.json')
    assert receipt['fixture_sha256'] == sha(case/'oracle-fixture.json')
    assert receipt['binary_sha256'] == sha(receipt['binary'])
    assert Path(receipt['binary']).resolve() != Path(profile['execution']['sut_command'][0]).resolve()
    stem = str(args.oracle_frontier)
    oracle = json.loads(Path(stem+'.multiturn.json').read_text())
    schedule = json.loads(Path(stem+'.schedule.json').read_text())
    assert schedule['requests'] == [len(fixture['prefix']), len(fixture['first_suffix'])]
    assert oracle['first_stopped_on_actual_eos']
    assert oracle['eos_ids'] == profile['execution']['eos_tokens']
    assert oracle['first_generated'] == fixture['first_outputs'] == raw['first']['final']['answer']['tokens']
    assert oracle['second_generated'] == raw['second']['final']['answer']['tokens']
    canonical = fixture['prefix']+fixture['first_suffix']+oracle['first_generated']
    assert oracle['second_suffix_position'] == len(canonical)
    assert oracle['second_suffix_tokens'] == oracle['second_prefill_tokens'] == len(fixture['new_user_suffix'])
    assert oracle['second_consumed'] == len(canonical)+len(fixture['new_user_suffix'])+len(oracle['second_generated'])
    assert oracle['first_prefill_work_tokens'] == len(fixture['prefix'])+len(fixture['first_suffix'])
    assert oracle['normal_forward_tokens'] == len(oracle['first_generated'])+len(oracle['second_generated'])
    assert oracle['total_prefill_tokens'] == oracle['first_prefill_work_tokens']+len(fixture['new_user_suffix'])
    assert oracle['total_forward_tokens'] == oracle['total_prefill_tokens']+oracle['normal_forward_tokens']
    chunks = json.loads(Path(stem+'.prefill-schedule.json').read_text())
    assert chunks['second'] == [[len(canonical), len(fixture['new_user_suffix'])]]
    assert sum(count for position,count in chunks['first']) == oracle['first_prefill_work_tokens']
    for name, suffix, tokens in (('first', fixture['first_suffix'], oracle['first_generated']),
                                ('second', fixture['new_user_suffix'], oracle['second_generated'])):
        work = raw[name]['final']['answer']['counter_deltas']
        assert work['prefill_tokens'] == len(suffix)
        assert work['forward_tokens'] == len(suffix)+len(tokens)
        assert work['decode_timed_tokens'] == len(tokens)
    assert raw['second']['state']['history'] == canonical+fixture['new_user_suffix']+oracle['second_generated']
    for path, digest in raw['parent_hashes'].items():
        assert sha(path) == digest
    width = 4*profile['execution'].get('vocab_size', raw['root']['manifest']['logits_count'])
    comparisons = []
    with Path(stem+'.requests.f32').open('rb') as stream:
        prefix = stream.read(width)
    assert prefix == Path(raw['root']['frontier_file']).read_bytes()
    comparisons.append('root')
    for name, trace, index in (('first', stem+'.decode.f32', len(oracle['first_generated'])-1),
                               ('second', stem+'.second.decode.f32', len(oracle['second_generated'])-1)):
        assert Path(trace).stat().st_size == (index+1)*width
        with Path(trace).open('rb') as stream:
            stream.seek(index*width); expected = stream.read(width)
        assert expected == Path(raw[name]['state']['frontier_file']).read_bytes()
        comparisons.append(name+'-final')
    tokenizer = Tokenizer.from_gguf(profile['model_path'])
    for name, history, operation, tokens in (
        ('first', fixture['prefix']+fixture['first_suffix'], 'next-first-A', oracle['first_generated']),
        ('second', canonical+fixture['new_user_suffix'], 'next-question-A', oracle['second_generated'])):
        expected = controller.start(dict(temperature=0, eos_tokens=profile['execution']['eos_tokens']),
                                    operation, tuple(history), len(tokenizer.tokens))
        expected = controller.consume_many(expected, ((token, tokenizer.token_bytes(token),
            token in profile['execution']['eos_tokens']) for token in tokens))
        assert expected == raw[name]['state']['controller']
    summary = dict(status='PASS', scope='Q8 exact next-user continuation, greedy non-thinking profile only',
        checks=raw['checks'], numerical_max_error=0., vector_checks=comparisons,
        audit_sha256=sha(__file__), control_harness_sha256=raw['harness_sha256'],
        controller_exact=True, raw_result_sha256=sha(case/'result.json'),
        oracle_receipt_sha256=sha(args.oracle_receipt), oracle_receipt=receipt,
        profile=profile, requests=[], limitations=[
            'root and both final full vectors checked, not every generated frontier',
            'one measured next-user request, no performance medians',
            'active turns require resume; explicit interruption remains separate',
            'ordinary UI/OpenAI and cumulative gates remain unqualified'])
    for name in ('first', 'second'):
        row = raw[name]; times = row['received_times']
        intervals = [b-a for a,b in zip(times,times[1:])]
        summary['requests'].append(dict(name=name, outputs=len(row['final']['answer']['tokens']),
            received_generation_tps=(len(times)-1)/(times[-1]-times[0]), ttft_seconds=times[0],
            tpot_median_ms=1000*statistics.median(intervals), durable_seconds=row['durable_seconds'],
            counter_deltas=row['final']['answer']['counter_deltas'],
            frontier_sha256=row['state']['frontier_sha256']))
    publish(args.output, summary)
    print(json.dumps(dict(status='PASS', vector_checks=comparisons, controller_exact=True)))


if __name__ == '__main__': main()
