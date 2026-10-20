"""Read-only image-frontier entropy sampling against the unchanged CPU oracle.

No checkpoint is imported. Two fresh native forwards establish the same image
frontier; selector calls must not consume tokens or change its saved logits.
This is a numerical audit, not image-answer quality or throughput measurement.
"""
import argparse
import json
from pathlib import Path
import struct
import time

from context_time_travel.input_spans import history_for_snapshot
from context_time_travel.min_p import choose
from context_time_travel.store import Store
from run_entropy_selection_probe import policy_cases
from run_vision_grammar_probe import engine_for
from run_vision_ingest_probe import Native,compare_files,fingerprint,save
from vision_fixture import reference_positions


def run(args):
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    profile=json.loads(args.profile.read_text());prior=json.loads(args.fixture.read_text())
    store=Store(Path(prior['targets']['prefix']['payload']['path']).parent.parent)
    history=history_for_snapshot(store,prior['targets']['prefix']['snapshot'])
    engine=engine_for(profile);images=[];lines=[];payloads=[];cell=0
    for span in history.spans:
        if span['kind']=='text':cell+=len(span['tokens']);continue
        payload=args.fixture.parent/('oracle-'+span['embedding_sha256']+'.sve')
        assert fingerprint(payload)['sha256']==span['embedding_sha256']
        payloads.append(fingerprint(payload))
        images.append(dict(start=cell,rows=span['rows'],grid_width=span['grid_width'],grid_height=span['grid_height']))
        fields=[cell]+[span[key] for key in ('rows','grid_width','grid_height','width','pad_token_id','embedding_sha256')]
        lines.append(' '.join(map(str,fields))+' '+json.dumps(str(payload)));cell+=span['rows']
    manifest=out/'input.manifest';manifest.write_text(f'COW_VISION_INPUT_V1 0 {len(history)} {len(images)}\n'+
        ' '.join(map(str,history.native_tokens))+'\n'+'\n'.join(lines)+'\n')
    positions=out/'positions.i32';positions.write_bytes(b''.join(struct.pack('<3i',*p)
        for p in reference_positions(len(history),images,profile['execution']['capacity'])))
    result=dict(status='RUNNING',scope=__doc__,profile=fingerprint(args.profile),fixture=fingerprint(args.fixture),
        manifest=fingerprint(manifest),positions=fingerprint(positions),embeddings=payloads,runs=[],rows=[],comparisons=[])
    started=time.monotonic();save(out/'result.json',result)
    try:
        for kind in ('reference','cow'):
            argv=list(profile['execution']['sut_command'])
            if kind=='reference':argv[0]=str(args.reference_binary)
            native=Native(argv,out/kind);row=dict(kind=kind,pid=native.proc.pid,argv=argv);result['runs'].append(row)
            try:
                print(json.dumps(dict(phase=kind,pid=native.proc.pid,seconds=time.monotonic()-started)),flush=True)
                if kind=='cow':
                    assert native.ready['sampling_v4_protocol']=='full-vocab-typical-dyna-min-p-u53-v4'
                    row['vision_info']=native.request('vision_info');assert row['vision_info']['enabled'] is True
                command='feed_image '+json.dumps(str(manifest))
                if kind=='reference':command+=' '+json.dumps(str(positions))
                native.request(command);frontier,values=native.frontier('before-selection')
                row['frontier']=fingerprint(frontier);row['work']=native.request('stats')
                assert row['work']['forward_tokens']==len(history)
                if kind=='reference':reference_frontier=frontier;reference_values=values
                else:
                    comparison=compare_files(reference_frontier,frontier);assert comparison['exact']
                    result['comparisons'].append(comparison)
                    for case in policy_cases(include_dry=False):
                        mask=bytearray(b'\xff'*((engine.vocab_size+7)//8))
                        if engine.vocab_size%8:mask[-1]&=(1<<(engine.vocab_size%8))-1
                        if case['mask_kind']=='single':
                            token=max(range(engine.vocab_size),key=lambda t:reference_values[t])
                            mask=bytearray(len(mask));mask[token>>3]|=1<<(token&7)
                        mask=bytes(mask);mask_path=out/(case['name']+'.mask');mask_path.write_bytes(mask)
                        empty=out/'empty-u32';empty.write_bytes(b'');policy=case['policy']
                        for draw in case['draws']:
                            expected=choose(reference_values,mask,draw53=draw,
                                **{key:value for key,value in policy.items() if key!='repetition_penalty'})
                            fields=[len(history),policy['temperature'],policy['min_p'],policy['top_k'],policy['top_p'],draw,json.dumps(str(mask_path))]
                            if case['expected_protocol']=='v4':
                                fields += [0.,policy['dry_base'],policy['dry_allowed_length'],policy['dry_penalty_last_n'],
                                    json.dumps(str(empty)),json.dumps(str(empty)),policy['typical_p'],policy['dynatemp_range'],policy['dynatemp_exponent']]
                            command='sample_mask_'+case['expected_protocol']+' '+' '.join(map(str,fields))
                            before=native.request('stats');response=native.request(command);after=native.request('stats')
                            assert response['token']==expected['token'] and response['kept_count']==expected['kept_count'],(case,response,expected)
                            assert after['forward_tokens']==before['forward_tokens']
                            counter='sampling_'+case['expected_protocol']+'_calls'
                            assert after[counter]-before[counter]==1
                            if case['expected_protocol']=='v4':
                                assert response['typical_kept_count']==expected['typical_kept_count']
                                assert response['dry_history_count']==response['dry_breaker_count']==response['dry_penalized_candidates']==0
                            result['rows'].append(dict(name=case['name'],policy=policy,draw53=draw,
                                expected=expected,response=response,counters_before=before,counters_after=after))
                    after,_=native.frontier('after-selection');comparison=compare_files(frontier,after)
                    assert comparison['exact'];result['comparisons'].append(comparison)
                    row['after_work']=native.request('stats');assert row['after_work']['forward_tokens']==len(history)
                    assert any(r['name']=='typical-alone' and r['response']['typical_kept_count']<r['response']['legal_count'] for r in result['rows'])
                    assert any(r['name']=='dynatemp-alone' and r['response']['effective_temperature']!=r['policy']['temperature'] for r in result['rows'])
            finally:row['exit']=native.close();save(out/'result.json',result)
        assert all(fingerprint(row['path'])==row for row in payloads)
        result.update(status='PASS',max_abs_error=0.,seconds=time.monotonic()-started)
    except BaseException as error:result.update(status='FAIL',error=repr(error));raise
    finally:save(out/'result.json',result)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('profile','fixture','reference-binary','output'):p.add_argument('--'+name,type=Path,required=True)
    run(p.parse_args())
