"""Real image DRY selection with nonzero penalties and fresh native forwards.

Controlled token patterns exercise arithmetic, not image-answer quality. The
embedding is retained encoder output; no model checkpoint is imported anywhere.
"""
import argparse
import json
from pathlib import Path
import struct
import time

from context_time_travel.input_spans import InputHistory
from context_time_travel.min_p import choose
from context_time_travel.store import Store
from run_vision_grammar_probe import engine_for
from run_vision_ingest_probe import Native,compare_files,fingerprint,save
from test_vision_dry_native import reference
from vision_fixture import reference_positions


def run(args):
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    profile=json.loads(args.profile.read_text());prior=json.loads(args.fixture.read_text())
    store=Store(Path(prior['targets']['prefix']['payload']['path']).parent.parent)
    from context_time_travel.input_spans import history_for_snapshot
    original=history_for_snapshot(store,prior['targets']['prefix']['snapshot'])
    image=next(span for span in original.spans if span['kind']=='image')
    payload=args.fixture.parent/('oracle-'+image['embedding_sha256']+'.sve')
    assert fingerprint(payload)['sha256']==image['embedding_sha256']
    engine=engine_for(profile)
    letters=[]
    for char in (b'a',b'b',b'c',b'd'):
        matches=[i for i in range(engine.vocab_size) if engine.tokenizer.token_bytes(i)==char]
        assert matches;letters.append(matches[0])
    a,b,c,d=letters;patterns={
        'genuine-repeat':([d]*128,[a,b,c,a,b]),
        'current-bridge':([d]*128+[a,b,c,a],[b]),
        'prior-bridge':([d]*128+[a],[b,c,a,b]),
        'earlier-run':([d]*128+[a,b,c],[a,b])}
    result=dict(status='RUNNING',scope=__doc__,profile=fingerprint(args.profile),fixture=fingerprint(args.fixture),
        embedding=fingerprint(payload),letters=letters,runs=[],comparisons=[],selections=[])
    started=time.monotonic();save(out/'result.json',result)
    try:
        for case,(left,right) in patterns.items():
            history=InputHistory([dict(kind='text',tokens=left),image,dict(kind='text',tokens=right)])
            text=left+right;tail=text[-5:];offset=len(text)-len(tail)
            cuts=[len(left)-offset] if len(left)>=offset else []
            for kind in ('reference','cow'):
                argv=list(profile['execution']['sut_command'])
                if kind=='reference':argv[0]=str(args.reference_binary)
                native=Native(argv,out/(case+'-'+kind));row=dict(case=case,kind=kind,pid=native.proc.pid,argv=argv)
                result['runs'].append(row)
                try:
                    print(json.dumps(dict(phase=case+'-'+kind,pid=native.proc.pid,seconds=time.monotonic()-started)),flush=True)
                    manifest=out/(case+'.manifest')
                    manifest.write_text('COW_VISION_INPUT_V1 0 '+str(len(history))+' 1\n'+
                        ' '.join(map(str,history.native_tokens))+'\n'+
                        ' '.join(map(str,[len(left)]+[image[k] for k in ('rows','grid_width','grid_height','width','pad_token_id','embedding_sha256')]))+
                        ' '+json.dumps(str(payload))+'\n')
                    points=reference_positions(len(history),[dict(start=len(left),rows=image['rows'],
                        grid_width=image['grid_width'],grid_height=image['grid_height'])],profile['execution']['capacity'])
                    positions=out/(case+'.positions.i32');positions.write_bytes(b''.join(struct.pack('<3i',*p) for p in points))
                    command='feed_image '+json.dumps(str(manifest))
                    if kind=='reference':command+=' '+json.dumps(str(positions))
                    native.request(command)
                    frontier,values=native.frontier('prefill');row['frontier']=fingerprint(frontier)
                    row['work']=native.request('stats');assert row['work']['forward_tokens']==len(history)
                    if kind=='reference':
                        reference_frontier=frontier;reference_values=values
                    else:
                        comparison=compare_files(reference_frontier,frontier);assert comparison['exact']
                        result['comparisons'].append(dict(case=case,**comparison))
                        adjusted=reference(list(reference_values),tail,(),cuts,2,.8,1.75)
                        expected_count=sum(x!=y for x,y in zip(adjusted,reference_values))
                        if case in ('genuine-repeat','earlier-run'):assert expected_count==1 and adjusted[c]<reference_values[c]
                        else:assert expected_count==0
                        mask=bytearray((engine.vocab_size+7)//8)
                        for token in (a,b,c):mask[token>>3]|=1<<(token&7)
                        mask=bytes(mask);mask_path=out/(case+'.mask');mask_path.write_bytes(mask)
                        history_path=out/(case+'.history');history_path.write_bytes(struct.pack('<5I',*tail))
                        breakers=out/(case+'.breakers');breakers.write_bytes(b'')
                        settings=([(.7,draw,typical,dynamic) for typical,dynamic in
                            ((.95,0.),(1.,.35),(.95,.35)) for draw in (0,1<<52,(1<<53)-1)]
                            if args.entropy else [(.7,1<<52,1.,0.),(0.,0,1.,0.)])
                        for temperature,draw,typical,dynamic in settings:
                            expected=choose(adjusted,mask,temperature=temperature,min_p=.05,draw53=draw,
                                top_k=60,top_p=.95,typical_p=typical,dynatemp_range=dynamic,
                                dynatemp_exponent=1.)['token']
                            fields=[len(history),temperature,.05,60,.95,draw,json.dumps(str(mask_path)),
                                .8,1.75,2,5,json.dumps(str(history_path)),json.dumps(str(breakers))]
                            command='sample_mask_image_dry_v1'
                            if args.entropy:
                                command='sample_mask_image_v4';fields.extend((typical,dynamic,1.))
                            response=native.request(command+' '+' '.join(map(str,fields)))
                            assert response['token']==expected and response['dry_penalized_candidates']==expected_count,response
                            assert response['text_tokens']==len(text) and response['dry_boundary_count']==len(cuts)
                            if args.entropy:
                                assert response['protocol']=='full-vocab-typical-dyna-min-p-u53-v4'
                                assert response['vision_dry_sampling_protocol']=='mixed-text-runs-sampling-u53-v2'
                                assert response['dry_boundaries']==cuts
                            result['selections'].append(dict(case=case,temperature=temperature,expected_token=expected,
                                typical_p=typical,dynatemp_range=dynamic,
                                expected_penalized_candidates=expected_count,response=response))
                        history_path.write_bytes(struct.pack('<5I',*((tail[0]+1,)+tuple(tail[1:]))))
                        rejected=native.request(command+' '+' '.join(map(str,fields)),allow_error=True)
                        assert rejected['status']=='error' and 'native session tail' in rejected['message'],rejected
                        row['wrong_tail_rejected']=rejected
                        after,_=native.frontier('after-sampling');comparison=compare_files(frontier,after)
                        assert comparison['exact'];result['comparisons'].append(dict(case=case+'-readonly',**comparison))
                        row['after_work']=native.request('stats')
                        assert row['after_work']['forward_tokens']==row['work']['forward_tokens']
                        if args.entropy:
                            assert row['after_work']['sampling_image_dry_combination_calls']==len(settings)
                            assert row['after_work']['sampling_v4_calls']==len(settings)
                        else:assert row['after_work']['sampling_mixed_dry_calls']==len(settings)
                finally:row['exit']=native.close();save(out/'result.json',result)
        result.update(status='PASS',max_abs_error=0.,seconds=time.monotonic()-started)
    except BaseException as error:result.update(status='FAIL',error=repr(error));raise
    finally:save(out/'result.json',result)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('profile','fixture','reference-binary','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--entropy',action='store_true',help='Audit DRY with each V4 entropy filter and both together')
    run(p.parse_args())
