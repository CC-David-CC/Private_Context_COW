"""Real-native DRY score-copy audit on deliberately repeated consumed input.

Complements the process-separated continuation harness. It does not certify
model quality, arbitrary controller settings or cumulative G1-G5.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import struct
import sys
import time


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for field in ('runtime-root','profile','fixture','run-dir'):
        parser.add_argument('--'+field,type=Path,required=True)
    args=parser.parse_args()
    runtime=args.runtime_root.resolve(strict=True)
    sys.path[:0]=[str(runtime),str(runtime/'tools'),str(runtime/'experimental/branching_context/src'),
                  str(runtime/'experimental/branching_context/scripts')]
    from context_time_travel import min_p
    from context_time_travel.dry import penalties_from_ids,FLOAT32_MAX
    from context_time_travel.sampled_controller import dry_arguments
    from context_time_travel.strata_backend import StrataBackend
    from context_time_travel.real import BranchingModel
    from grammar_long_reference import Heartbeat,_apply_profile_environment
    from run_grammar_first_probe import publish,finish_backend
    from tools.strata_tokenizer import Tokenizer
    os.umask(0o077)
    output=args.run_dir.absolute();output.mkdir(mode=0o700,parents=True,exist_ok=False)
    trace=output/'native-trace.jsonl';os.environ['CTT_NATIVE_TRACE']=str(trace)
    heartbeat=Heartbeat(output/'heartbeat.json').start()
    profile=json.loads(args.profile.read_bytes());_apply_profile_environment(profile)
    result=dict(status='RUNNING',scope=__doc__,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        profile_sha256=hashlib.sha256(args.profile.read_bytes()).hexdigest(),python_pid=os.getpid(),rows=[],failures=[])
    backend=None
    publish(output/'result.json',result)
    try:
        tokenizer=Tokenizer.from_gguf(profile['model_path'])
        fixture=json.loads(args.fixture.read_bytes())
        prefix=tuple(fixture['root_token_ids'][:128])
        picked=[]
        for word in ('alpha',' beta',' gamma',' cat',' dog'):
            for token in tokenizer.encode(word):
                piece=tokenizer.token_bytes(token)
                if (token not in picked and token not in profile['execution']['eos_tokens']
                        and piece and not any(char in piece for char in (b'\n',b':',b'"',b'*'))):
                    picked.append(token)
                if len(picked)==3:break
            if len(picked)==3:break
        assert len(picked)==3
        tail=tuple(picked)*8+tuple(picked[:2])
        assert len(tail)==26
        heartbeat.set('native-start')
        started=time.monotonic();backend=StrataBackend(profile)
        result.update(native_pid=backend.process.pid,startup_seconds=time.monotonic()-started)
        model=BranchingModel(output/'store',backend,profile)
        heartbeat.set('prepare-repeated-history',backend.process.pid)
        root=model.create(prefix);child=model.feed(root,tail)
        result['fixture']=dict(prefix=list(prefix),tail=list(tail),candidate= picked[2],
            source_sha256=hashlib.sha256(args.fixture.read_bytes()).hexdigest())
        history=model.history(child)
        policy=dict(temperature=.7,min_p=.05,top_k=60,top_p=.95,repetition_penalty=1.,
            dry_multiplier=.8,dry_base=1.75,dry_allowed_length=2,dry_penalty_last_n=256)
        context=dry_arguments(policy,history,tokenizer.token_bytes)
        penalties=penalties_from_ids(context['dry_history'],context['dry_breakers'],
            **{key:policy[key] for key in min_p.DRY_DEFAULTS})
        assert penalties.get(picked[2],0)>0,'Fixture must actually trigger DRY'
        parent_blob=model.cache[root].blob
        parent_audit=backend.audit(parent_blob)
        with model._materialize(child) as (lease,_):
            before=backend.work_counters()
            values=backend.logits(lease)
            frontier=struct.pack('<'+'f'*len(values),*values)
            (output/'frontier.f32').write_bytes(frontier)
            expected_count=0
            for token,penalty in penalties.items():
                adjusted=struct.unpack('<f',struct.pack('<f',max(-FLOAT32_MAX,values[token]-penalty)))[0]
                expected_count+=adjusted!=values[token]
            assert expected_count>0
            full=((1<<backend.vocab_size)-1).to_bytes((backend.vocab_size+7)//8,'little')
            single=(1<<picked[2]).to_bytes(len(full),'little')
            for name,mask,temp in (('preset',full,.7),('greedy',full,0.),('single-legal',single,.7)):
                heartbeat.set('select-'+name,backend.process.pid)
                draw=0 if temp==0 else (1<<52)
                expected=min_p.choose(values,mask,temperature=temp,min_p=.05,draw53=draw,
                    top_k=60,top_p=.95,**context)
                old=backend.work_counters()
                token=backend.sample_legal(lease,mask,temp,.05,draw,top_k=60,top_p=.95,**context)
                new=backend.work_counters()
                assert token==expected['token']
                assert new['dry_penalized_candidates']-old['dry_penalized_candidates']==expected_count
                assert new['sampling_v3_calls']-old['sampling_v3_calls']==1
                assert new['sampling_v2_calls']==old['sampling_v2_calls']
                assert struct.pack('<'+'f'*len(values),*backend.logits(lease))==frontier
                result['rows'].append(dict(name=name,temperature=temp,token=token,reference_token=expected['token'],
                    changed_candidates=expected_count,native_counter_before=old,native_counter_after=new))
                publish(output/'result.json',result)
            after=backend.work_counters()
            assert after['forward_tokens']==before['forward_tokens']
            result.update(selection_forward_tokens=0,raw_frontier_unchanged=True,
                          frontier_sha256=hashlib.sha256(frontier).hexdigest())
        assert backend.audit(parent_blob)['sha256']==parent_audit['sha256']
        result.update(parent_unchanged=True,numerical_max_error=0,status='PASS')
    except BaseException as exc:
        result['status']='FAIL';result['failures'].append(dict(type=type(exc).__name__,message=str(exc)))
        raise
    finally:
        if backend is not None:finish_backend(backend,result)
        heartbeat.close();publish(output/'result.json',result)
    if result['status']!='PASS':raise AssertionError('Native cleanup failed')


if __name__=='__main__':main()
