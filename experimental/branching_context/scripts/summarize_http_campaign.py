"""Summarize a terminal real HTTP GPU campaign without discarding fingerprints."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--result',type=Path,required=True)
    parser.add_argument('--profile',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    raw=args.result.read_bytes();result=json.loads(raw)
    profile=json.loads(args.profile.read_bytes())
    if (result.get('execution_profile_id')!=profile['execution_profile_id']
            or result.get('model_contract_id')!=profile['model_contract_id']):
        raise ValueError('Result and profile identities differ')
    if result['status']!='PASS' or result['native_exit']!=0 or len(result['requests'])!=2:
        raise ValueError('Requires completed A/B campaign and native exit')
    if len(result['checks'])!=5 or result['numerical_max_error']!=0:
        raise ValueError('Missing executed correctness checks')
    output=dict(status='PASS',scope=result['scope'],executed_ids=result['checks'],
        numerical_max_error=result['numerical_max_error'],seconds=result['seconds'],
        startup_seconds=result['startup_seconds'],native_exit=result['native_exit'],
        raw_result_sha256=hashlib.sha256(raw).hexdigest(),harness_sha256=result['harness_sha256'],
        profile=profile,metrics=result['metrics'],requests=[],
        prefix_creation_seconds=result['prefix_creation']['seconds'],
        limitations=['two requests, no warmup/repeated medians',
            'explicit COW endpoint, not ordinary OpenAI chat/UI integration',
            'one final generated frontier per branch; not every-token G1 qualification',
            'greedy matching native operators; original cumulative gates remain separate'])
    for row in result['requests']:
        answer=row['final']['answer'];tokens=answer['tokens'];intervals=row['tpot_seconds']
        controller=row['state']['controller']
        controller_digest=hashlib.sha256(json.dumps(controller,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        controller_summary={k:v for k,v in controller.items() if k!='policy'}
        controller_summary['policy']={k:v for k,v in controller.get('policy',{}).items() if k!='allowed'}
        controller_summary['allowed_count']=len(controller.get('policy',{}).get('allowed',[]))
        if len(tokens)!=len(row['received_times']): raise ValueError('Incomplete streamed outputs')
        output['requests'].append(dict(branch=row['name'],outputs=len(tokens),
            output_tokens_sha256=hashlib.sha256(json.dumps(tokens,separators=(',',':')).encode()).hexdigest(),
            received_generation_tps=row['received_generation_tps'],ttft_seconds=row['ttft_seconds'],
            tpot_median_ms=1000*statistics.median(intervals),
            tpot_p95_ms=1000*sorted(intervals)[min(len(intervals)-1,int(.95*len(intervals)))],
            durable_request_seconds=row['seconds'],native_generation_seconds=answer['generation_seconds'],
            capture_seconds=answer['capture_seconds'],counter_deltas=answer['counter_deltas'],
            receipt=row['final']['receipt'],frontier_sha256=row['state']['frontier_sha256'],
            checkpoint_manifest=row['state']['manifest'],
            controller_sha256=controller_digest,controller_summary=controller_summary,
            reference=dict(seconds=row['reference']['seconds'],
                command=row['reference']['command'],work={k:v for k,v in row['reference']['work'].items() if k!='generated'})))
    args.output.write_text(json.dumps(output,indent=2,allow_nan=False)+'\n',newline='\n')
    print(json.dumps([dict(branch=r['branch'],outputs=r['outputs'],tps=r['received_generation_tps'],
        ttft=r['ttft_seconds'],tpot_ms=r['tpot_median_ms'],durable_seconds=r['durable_request_seconds']) for r in output['requests']]))


if __name__=='__main__': main()
