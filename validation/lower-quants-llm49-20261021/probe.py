"""Bounded text/branch smoke probe; run with BranchWrite's Python environment.

This measures streamed API timing, not isolated GPU decode or numerical equality.
No project files or model files are read by this client.
"""
from pathlib import Path
import argparse
import getpass
import hashlib
import json
import statistics
import time
import uuid

def run_probe(server,api_key,output_dir,label='model',max_tokens=128,context_text=''):
    from branchwrite.cow_client import CowClient,OperationJournal
    out=Path(output_dir).expanduser();out.mkdir(mode=0o700,parents=True,exist_ok=False)
    run_id=uuid.uuid4().hex
    result=dict(label=label,run_id=run_id,stages=[],passed=False,
                timing_scope='Streamed API callbacks include COW and HTTP; not isolated GPU decode',
                numerical_max_error=None,oracle_comparisons=0,context_sha256=hashlib.sha256(context_text.encode()).hexdigest())
    times=[]
    client=CowClient.http(server,api_key=api_key,journal=OperationJournal(out/'journal'),
                          timeout=600,stream_handler=lambda text:times.append(time.monotonic()))
    def op(name):return 'probe-'+str(uuid.uuid5(uuid.NAMESPACE_URL,run_id+'/'+name))
    def save():
        text=json.dumps(result,indent=2).replace(api_key,'[REDACTED]')
        (out/'result.json').write_text(text+'\n')
    try:
        result['service_info']=client.transport.request('GET','/cow/info').json_value
        ns=client.create_namespace('Model smoke '+label,op('namespace'))['namespace_id']
        prefix='probe-'+run_id[:12]+'-prefix';started=time.monotonic()
        r=client.load_context(prefix,ns,[
            {'role':'system','content':'You are a helpful assistant. Give accurate, concise answers.'},
            {'role':'user','content':'Background notes: Buddy is a friendly dog. Luna is a curious cat. The website must work without any downloads or external dependencies.'+('\n\n'+context_text if context_text else '')},
            {'role':'assistant','content':'I will use these notes.'}],op('prefix'))
        parent=client.head(prefix)
        result['stages'].append(dict(stage='prefix',seconds=time.monotonic()-started,receipt=r.receipt.raw,answer=r.answer));save()
        for name,prompt in [('A','Write one friendly paragraph about Buddy the dog.'),('B','Write one friendly paragraph about Luna the cat.')]:
            branch='probe-'+run_id[:12]+'-'+name
            fork=client.fork(prefix,parent.tip,branch,op('fork-'+name))
            times.clear();started=time.monotonic()
            r=client.ask(branch,fork.revision,op('ask-'+name),prompt,max_tokens,
                         settings={'temperature':0.0,'min_p':0.0,'top_k':0,'top_p':1.0})
            elapsed=time.monotonic()-started;tokens=r.answer.get('tokens',[])
            if not tokens or not r.answer.get('text') or r.receipt.branch.name!=branch:
                raise RuntimeError('Missing tokens/text or incorrect branch receipt')
            intervals=[b-a for a,b in zip(times,times[1:])]
            row=dict(stage=name,seconds=elapsed,tokens=len(tokens),text=r.answer.get('text'),
                     answer=r.answer,receipt=r.receipt.raw,stream_callbacks=len(times),
                     callback_offsets_seconds=[t-started for t in times],
                     first_callback_seconds=times[0]-started if times else None,
                     inter_callback_tps=(len(times)-1)/(times[-1]-times[0]) if len(times)>1 else None,
                     mean_callback_interval_seconds=statistics.mean(intervals) if intervals else None)
            result['stages'].append(row);save()
        if client.head(prefix)!=parent:raise RuntimeError('Prefix branch reference moved')
        result['parent_tip_unchanged']=True;result['passed']=True
    except Exception as error:
        result['error']=type(error).__name__+': '+str(error)
        raise
    finally:save()
    return result

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--server',required=True)
    parser.add_argument('--output-dir',required=True)
    parser.add_argument('--label',default='model')
    parser.add_argument('--max-tokens',type=int,default=128)
    parser.add_argument('--context-file',type=Path,help='Optional UTF-8 fixture added to the shared starting context')
    args=parser.parse_args()
    if not 1<=args.max_tokens<=4096:parser.error('Choose 1 through 4096 output tokens')
    key=getpass.getpass('COW API key (not saved): ').strip()
    if not key:parser.error('An API key is required')
    result=run_probe(args.server,key,args.output_dir,args.label,args.max_tokens,args.context_file.read_text() if args.context_file else '')
    print(json.dumps(result,indent=2))
    return 0 if result['passed'] else 1

if __name__=='__main__':raise SystemExit(main())
