"""Real native cold-load rejection and missing-position mutation witness.

Use an already executed restart campaign. A test-only mutant must restore the
same state successfully yet disagree after real suffix computation.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path

from context_time_travel.input_spans import InputHistory
from context_time_travel.vision_execution import input_binding, placements
from run_vision_ingest_probe import Native, compare_files, fingerprint, save


def metadata(path, history, *, images=None, ids=None):
    ids = history.native_tokens if ids is None else ids
    images = placements(history) if images is None else images
    lines = [f'COW_VISION_RESTORE_V1 {len(ids)} {len(images)}', ' '.join(map(str,ids))]
    for image in images:
        lines.append(' '.join(str(image[k]) for k in
            ('start','rows','grid_width','grid_height','width','pad_token','embedding_sha256')))
    path.write_text('\n'.join(lines)+'\n')


def campaign(args):
    out = args.output.resolve(); out.mkdir(parents=True, exist_ok=False)
    config = json.loads((args.campaign/'config.json').read_text())
    prepared = json.loads((args.campaign/'prepare/result.json').read_text())
    assert prepared['status']=='PASS'
    profile = json.loads(Path(config['profile']).read_text())
    history = InputHistory(prepared['spans'])
    root = prepared['prefix']; folder = Path(config['store'])/'checkpoints'
    envelope = json.loads((folder/(root+'.json')).read_text()); manifest = envelope['manifest']
    payload = (folder/(manifest['payload_sha256']+'.state')).read_bytes()
    assert hashlib.sha256(payload).hexdigest() == manifest['payload_sha256']
    state = out/'native.state'; state.write_bytes(payload[:manifest['state_bytes']])
    logits = out/'native.f32'; logits.write_bytes(payload[manifest['state_bytes']:])
    inputs = out/'metadata'; metadata(inputs,history)
    argv = list(profile['execution']['sut_command']); argv[0] = str(args.binary.resolve())
    result = dict(status='RUNNING', binary=fingerprint(args.binary), argv=argv,
        expected_position_failure=args.expect_position_failure, rejection_checks=[])
    native = Native(argv, out/'native'); result['native_pid']=native.proc.pid
    try:
        assert logits.stat().st_size == native.ready['vocab']*4
        for kind in ('payload','grid','token'):
            images = copy.deepcopy(placements(history)); ids=list(history.native_tokens)
            if kind=='payload': images[0]['embedding_sha256']='0'*64
            if kind=='grid': images[0].update(grid_width=images[0]['rows'],grid_height=1)
            if kind=='token': ids[0]=native.ready['vocab']
            wrong=out/('wrong-'+kind); metadata(wrong,history,images=images,ids=ids)
            before=native.request('stats')
            response=native.request('load_image '+' '.join(json.dumps(str(p)) for p in (state,wrong,logits)),allow_error=True)
            after=native.request('stats')
            assert response['status']=='error', response
            changes={key:dict(before=value,after=after[key]) for key,value in before.items()
                     if value!=after[key]}
            # Parsing/error reporting may retain small CPU allocator blocks.
            # No execution, position upload, device allocation/copy or active
            # consumed-state counter may change on these metadata refusals.
            assert set(changes)<= {'host_heap_allocated_bytes','host_heap_mmap_bytes'}, changes
            result['rejection_checks'].append(dict(kind=kind,response=response,
                device_and_forward_counters_unchanged=True,host_allocator_changes=changes))
        before=native.request('stats')
        native.request('load_image '+' '.join(json.dumps(str(p)) for p in (state,inputs,logits)))
        after=native.request('stats')
        assert after['forward_tokens']==before['forward_tokens']==0
        assert after['loaded_file_bytes']-before['loaded_file_bytes']==state.stat().st_size
        info=native.request('vision_info')
        assert info['validated_embedding_bytes']==0 and info['consumed']==len(history)
        for key,value in input_binding(history).items():assert info[key]==value
        restored,_=native.frontier('restored')
        result['restored_frontier']=compare_files(prepared['expected']['A']['frontiers'][0],restored)
        assert result['restored_frontier']['exact']
        native.request('feed '+' '.join(map(str,config['suffixes']['A'])))
        continued,_=native.frontier('suffix')
        result['suffix_comparison']=compare_files(prepared['expected']['A']['frontiers'][1],continued)
        result['restore_counters_before']=before;result['restore_counters_after']=after
        if args.expect_position_failure:
            assert not result['suffix_comparison']['exact'], 'Missing-position mutant escaped'
            result['status']='MUTANT_DETECTED'
        else:
            assert result['suffix_comparison']['exact'], result['suffix_comparison']
            result['status']='PASS'
    except BaseException as error:
        result.update(status='FAIL',error=repr(error));raise
    finally:
        result['native_exit']=native.close();save(out/'result.json',result)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('campaign','binary','output'):parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--expect-position-failure',action='store_true')
    campaign(parser.parse_args())
