"""Image-frontier V5 choices, support removal and exact draw receipts.

Uses two fresh GPU model processes with no checkpoint imports. Read-only native
selection is checked against the unchanged CPU XTC reference; it is not a speed
or image-answer-quality measurement.
"""
import argparse
import json
from pathlib import Path
import struct
import time

from context_time_travel.input_spans import history_for_snapshot
from context_time_travel.store import Store
from context_time_travel.xtc_pipeline import choose_v5
from run_vision_grammar_probe import engine_for
from run_vision_ingest_probe import Native,compare_files,fingerprint,save
from vision_fixture import reference_positions


def run(args):
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    profile=json.loads(args.profile.read_text());fixture=json.loads(args.fixture.read_text())
    store=Store(Path(fixture['targets']['prefix']['payload']['path']).parent.parent)
    history=history_for_snapshot(store,fixture['targets']['prefix']['snapshot']);engine=engine_for(profile)
    images=[];lines=[];payloads=[];cell=0
    for span in history.spans:
        if span['kind']=='text':cell+=len(span['tokens']);continue
        payload=args.fixture.parent/('oracle-'+span['embedding_sha256']+'.sve')
        assert fingerprint(payload)['sha256']==span['embedding_sha256'];payloads.append(fingerprint(payload))
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
    cases=[('inactive','pair',dict(xtc_probability=0.)),('single','single',{}),('xtc','pair',{}),
        ('filtered','pair',dict(top_k=2,top_p=.95)),('combined','pair',dict(typical_p=.95,dynatemp_range=.35)),
        ('threshold-off','pair',dict(xtc_threshold=.6))]
    try:
        for kind in ('reference','cow'):
            argv=list(profile['execution']['sut_command'])
            if kind=='reference':argv[0]=str(args.reference_binary)
            native=Native(argv,out/kind);row=dict(kind=kind,pid=native.proc.pid,argv=argv);result['runs'].append(row)
            try:
                print(json.dumps(dict(phase=kind,pid=native.proc.pid,seconds=time.monotonic()-started)),flush=True)
                if kind=='cow':assert native.ready['sampling_v5_protocol']=='full-vocab-xtc-min-p-u53-v5'
                command='feed_image '+json.dumps(str(manifest))
                if kind=='reference':command+=' '+json.dumps(str(positions))
                native.request(command);frontier,values=native.frontier('before-selection')
                row['frontier']=fingerprint(frontier);row['work']=native.request('stats');assert row['work']['forward_tokens']==len(history)
                if kind=='reference':reference_frontier=frontier;reference_values=values
                else:
                    comparison=compare_files(reference_frontier,frontier);assert comparison['exact'];result['comparisons'].append(comparison)
                    # Select a real close-score pair, rather than changing model
                    # logits. Restrict to ordinary nonempty low-ID text pieces.
                    ordered=sorted((t for t in range(32,min(4096,engine.vocab_size)) if engine.tokenizer.token_bytes(t)),
                        key=lambda t:(reference_values[t],t))
                    pair=min(zip(ordered,ordered[1:]),key=lambda ts:(abs(reference_values[ts[0]]-reference_values[ts[1]]),ts))
                    result['pair']=dict(tokens=list(pair),logits=[reference_values[t] for t in pair])
                    empty=out/'empty-u32';empty.write_bytes(b'')
                    for name,support,updates in cases:
                        policy=dict(temperature=.9,min_p=.02,top_k=0,top_p=1.,repetition_penalty=1.,
                            dry_multiplier=0.,dry_base=1.75,dry_allowed_length=2,dry_penalty_last_n=256,
                            typical_p=1.,dynatemp_range=0.,dynatemp_exponent=1.,xtc_probability=.12,xtc_threshold=.15)
                        policy.update(updates);mask=bytearray((engine.vocab_size+7)//8)
                        for token in pair[:1] if support=='single' else pair:mask[token>>3]|=1<<(token&7)
                        mask=bytes(mask);mask_path=out/(name+'.mask');mask_path.write_bytes(mask)
                        for first,second in ((0,0),(1<<52,1<<52),((1<<53)-1,(1<<53)-1)):
                            expected=choose_v5(reference_values,mask,draw53=first,draw53_second=second,**policy)
                            fields=[len(history),policy['temperature'],policy['min_p'],policy['top_k'],policy['top_p'],first,
                                json.dumps(str(mask_path)),0.,1.75,2,256,json.dumps(str(empty)),json.dumps(str(empty)),
                                policy['typical_p'],policy['dynatemp_range'],policy['dynatemp_exponent'],
                                policy['xtc_probability'],policy['xtc_threshold'],second]
                            before=native.request('stats');response=native.request('sample_mask_v5 '+' '.join(map(str,fields)))
                            after=native.request('stats')
                            for key in ('token','draw53','draw53_second','draw_count','activation_draws','activated',
                                    'removed_count','pre_xtc_count','kept_count','legal_count'):
                                assert type(response[key]) is type(expected[key]) and response[key]==expected[key],(name,key,response,expected)
                            assert response['dry_history_count']==response['dry_breaker_count']==response['dry_penalized_candidates']==0
                            assert after['forward_tokens']==before['forward_tokens']
                            assert after['sampling_v5_calls']-before['sampling_v5_calls']==1
                            assert after['sampling_v4_calls']==before['sampling_v4_calls']
                            result['rows'].append(dict(name=name,policy=policy,expected=expected,response=response,
                                counters_before=before,counters_after=after))
                    after,_=native.frontier('after-selection');comparison=compare_files(frontier,after)
                    assert comparison['exact'];result['comparisons'].append(comparison)
                    row['after_work']=native.request('stats');assert row['after_work']['forward_tokens']==len(history)
                    assert any(r['response']['removed_count']>0 for r in result['rows'])
                    assert {r['response']['draw_count'] for r in result['rows']}=={1,2}
            finally:row['exit']=native.close();save(out/'result.json',result)
        assert all(fingerprint(p['path'])==p for p in payloads)
        result.update(status='PASS',max_abs_error=0.,seconds=time.monotonic()-started)
    except BaseException as error:result.update(status='FAIL',error=repr(error));raise
    finally:save(out/'result.json',result)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('profile','fixture','reference-binary','output'):parser.add_argument('--'+name,type=Path,required=True)
    run(parser.parse_args())
