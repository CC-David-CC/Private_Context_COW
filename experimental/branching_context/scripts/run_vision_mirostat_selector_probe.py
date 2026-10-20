"""Image-frontier Mirostat 2 choices, feedback and read-only state receipts.

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
from context_time_travel import mirostat2
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
    cases=[('single','single',dict(mu=8.)),('negative','pair',dict(mu=-2.5)),('normal','pair',dict(mu=8.)),
        ('wide','pair',dict(mu=2000.)),('no-feedback','pair',dict(mu=8.,eta=0.)),
        ('changed-target','pair',dict(mu=8.,tau=3.5,eta=.2))]
    try:
        for kind in ('reference','cow'):
            argv=list(profile['execution']['sut_command'])
            if kind=='reference':argv[0]=str(args.reference_binary)
            native=Native(argv,out/kind);row=dict(kind=kind,pid=native.proc.pid,argv=argv);result['runs'].append(row)
            try:
                print(json.dumps(dict(phase=kind,pid=native.proc.pid,seconds=time.monotonic()-started)),flush=True)
                if kind=='cow':assert native.ready['mirostat2_protocol']=='full-vocab-mirostat2-u53-v6'
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
                        policy=dict(temperature=.8,mu=8.,tau=4.,eta=.05);policy.update(updates)
                        mask=bytearray((engine.vocab_size+7)//8)
                        for token in pair[:1] if support=='single' else pair:mask[token>>3]|=1<<(token&7)
                        mask=bytes(mask);mask_path=out/(name+'.mask');mask_path.write_bytes(mask)
                        for draw in (0,1<<52,(1<<53)-1):
                            expected=mirostat2.choose(reference_values,mask,draw53=draw,**policy)
                            fields=[len(history),policy['temperature'],draw,json.dumps(str(mask_path)),policy['tau'],
                                policy['eta'],policy['mu'],0.,1.75,2,256,json.dumps(str(empty)),json.dumps(str(empty))]
                            before=native.request('stats');response=native.request('sample_mask_mirostat2 '+' '.join(map(str,fields)))
                            after=native.request('stats')
                            for key in ('token','draw53','legal_count','kept_count','selected_log_probability',
                                    'selected_probability','surprise_bits','next_mu'):
                                # JSON renders an integral double as e.g. 1.
                                # IDs/counts remain strictly integer; numeric
                                # receipts accept int/float, never bool, and
                                # still require exact numerical equality.
                                kinds=(int,) if key in ('token','draw53','legal_count','kept_count') else (int,float)
                                assert type(response[key]) in kinds and response[key]==expected[key],(name,key,response,expected)
                            assert response['mu']==policy['mu'] and response['draw_count']==1
                            assert response['feedback_protocol']=='mirostat2-state-v1' and type(response['feedback_version']) is int and response['feedback_version']==1
                            assert response['dry_history_count']==response['dry_breaker_count']==response['dry_penalized_candidates']==0
                            assert after['forward_tokens']==before['forward_tokens']
                            assert after['sampling_mirostat2_calls']-before['sampling_mirostat2_calls']==1
                            assert after['sampling_v4_calls']==before['sampling_v4_calls']
                            result['rows'].append(dict(name=name,policy=policy,expected=expected,response=response,
                                counters_before=before,counters_after=after))
                    after,_=native.frontier('after-selection');comparison=compare_files(frontier,after)
                    assert comparison['exact'];result['comparisons'].append(comparison)
                    row['after_work']=native.request('stats');assert row['after_work']['forward_tokens']==len(history)
                    assert any(r['response']['next_mu']!=r['response']['mu'] for r in result['rows'])
                    assert all(r['response']['draw_count']==1 for r in result['rows'])
            finally:row['exit']=native.close();save(out/'result.json',result)
        assert all(fingerprint(p['path'])==p for p in payloads)
        result.update(status='PASS',max_abs_error=0.,seconds=time.monotonic()-started)
    except BaseException as error:result.update(status='FAIL',error=repr(error));raise
    finally:save(out/'result.json',result)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('profile','fixture','reference-binary','output'):parser.add_argument('--'+name,type=Path,required=True)
    run(parser.parse_args())
