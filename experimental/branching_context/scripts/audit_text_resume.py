"""Executed post-run prefix/four-token/frontier checks and compact metrics."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics


def sha(path):
    with Path(path).open('rb') as stream: return hashlib.file_digest(stream,'sha256').hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();directory=args.directory
    raw=json.loads((directory/'result.json').read_text());assert raw['status']=='PASS'
    prepared=json.loads((directory/'prepare.json').read_text());resumed=json.loads((directory/'resume.json').read_text())
    profile=json.loads((directory/'profile.json').read_text())
    assert profile['execution_profile_id']==raw['profile_id']
    assert prepared['status']==resumed['status']=='PASS'
    assert all(v['observed_exit'] and v['exit_code']==v['native_exit']==0 for v in raw['phases'].values())
    root=prepared['root'];width=4*root['manifest']['logits_count']
    first={r['name']:r for r in prepared['requests']}
    output=dict(status='RUNNING',scope=raw['scope'],profile=profile,phases=raw['phases'],
        raw_sha256={n:sha(directory/n) for n in ('result.json','prepare.json','resume.json','fixture.json')},
        harness_sha256=raw['harness_sha256'],requested_output_limit=raw.get('output_limit',128),
        checks=[],requests=[],numerical_max_error=0.,
        limitations=['greedy only, no nonzero RNG/grammar qualification',
                    'prefix, four-output and final full vectors checked; not every generated frontier',
                    'two single measurements, no performance medians',
                    'ordinary UI/OpenAI integration and cumulative gate qualification remain separate'])
    for row in resumed['requests']:
        name=row['name'];base=first[name]
        with (directory/('fresh-resume-'+name+'.f32.requests.f32')).open('rb') as stream: prefix=stream.read(width)
        assert Path(root['frontier_file']).read_bytes()==prefix
        with (directory/('fresh-resume-'+name+'.f32.decode.f32')).open('rb') as stream:
            stream.seek(3*width);four=stream.read(width)
            stream.seek((len(row['all_tokens'])-1)*width);final=stream.read(width)
        assert Path(base['state']['frontier_file']).read_bytes()==four
        assert Path(row['state']['frontier_file']).read_bytes()==final
        assert row['final']['answer']['counter_deltas']['forward_tokens']==len(row['final']['answer']['tokens'])
        assert row['final']['answer']['counter_deltas']['prefill_tokens']==0
        assert sha(base['state']['payload_file'])==resumed['parent_hashes'][base['state']['payload_file']]
        comparison=next(c for c in raw['comparisons'] if c['branch']==name)
        assert comparison['status']=='PASS' and comparison['controller_exact'] and comparison['max_absolute_error']==0
        for point in ('prefix','four-output','final'):
            output['checks'].append(dict(id='TEXT-RESUME-'+name+'-'+point,result='PASS',max_absolute_error=0.))
        times=row['received_times'];intervals=[b-a for a,b in zip(times,times[1:])]
        output['requests'].append(dict(branch=name,first_outputs=len(base['final']['answer']['tokens']),
            resumed_outputs=len(row['final']['answer']['tokens']),total_outputs=len(row['all_tokens']),
            received_generation_tps=(len(times)-1)/(times[-1]-times[0]) if len(times)>1 else None,
            ttft_seconds=times[0],tpot_median_ms=1000*statistics.median(intervals) if intervals else None,
            durable_request_seconds=row['durable_seconds'],native_generation_seconds=row['final']['answer']['generation_seconds'],
            counter_deltas=row['final']['answer']['counter_deltas'],controller_exact=True,
            controller_sha256=hashlib.sha256(json.dumps(row['state']['controller'],sort_keys=True,separators=(',',':')).encode()).hexdigest(),
            controller_draw_cursor=row['state']['controller']['next_draw'],
            all_tokens_sha256=hashlib.sha256(json.dumps(row['all_tokens'],separators=(',',':')).encode()).hexdigest(),
            text_preview=row['all_text'][:500],frontier_sha256=row['state']['frontier_sha256']))
    output.update(status='PASS',prepare_startup_seconds=prepared['startup_seconds'],resume_startup_seconds=resumed['startup_seconds'],
        metrics=resumed['metrics'],parent_hashes=resumed['parent_hashes'])
    args.output.write_text(json.dumps(output,indent=2,allow_nan=False)+'\n')
    print(json.dumps([dict(branch=r['branch'],outputs=r['total_outputs'],resumed_tps=r['received_generation_tps'],ttft=r['ttft_seconds']) for r in output['requests']]))


if __name__=='__main__': main()
