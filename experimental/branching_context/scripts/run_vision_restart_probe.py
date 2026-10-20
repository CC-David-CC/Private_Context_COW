"""Process-separated real-GPU image checkpoint/restart campaign (greedy).

The retained encoder fixture is used only by prepare and fresh references.
Resume workers receive neither an image payload nor an encoder callback.
"""
import argparse
import copy
from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path
import shlex
import struct
import subprocess
import sys
import threading
import time

from context_time_travel.pools import DiskCheckpointPool, SharedCheckpointPool
from context_time_travel.real import BranchingModel
from context_time_travel.strata_backend import StrataBackend, _digest
from context_time_travel.vision_artifacts import ImageArtifacts
from context_time_travel.vision_execution import RESUME_CONTRACT, VISION_PROTOCOL, POSITION_CONVENTION
from run_vision_ingest_probe import Native, compare_files, fingerprint, save
from vision_fixture import fixture_parts, configure_prefill, boundary_coverage


def frontier(api, snapshot, path):
    values = api.next_logits(snapshot)
    path.write_bytes(struct.pack('<'+'f'*len(values), *values))
    return str(path)


def controller_file(api, snapshot, path):
    save(path, api.controller(snapshot))
    return fingerprint(path)


def audit_unchanged(backend, image, expected, reports):
    actual = backend.audit(image); reports.append(actual)
    # Owners and physical handles describe live allocation ownership. They may
    # change while all retained bytes remain immutable. Audit every saved byte.
    assert (actual['bytes'],actual['consumed'],actual['sha256']) == (
        expected['bytes'],expected['consumed'],expected['sha256']), 'Retained image bytes changed'


def worker(config, stage, branch, out):
    profile = json.loads(Path(config['profile']).read_text())
    result = dict(status='RUNNING', stage=stage, python_pid=os.getpid(), encoder_calls=0)
    backend = ram = None
    try:
        if stage == 'reference':
            fixture = json.loads(Path(config['fixture']).read_text())
            argv = list(profile['execution']['sut_command']); argv[0] = config['reference_binary']
            native = Native(argv, out/'native')
            result['native_pid'] = native.proc.pid
            try:
                manifest = Path(config['fixture']).parent/'input.manifest'
                positions = manifest.parent/'reference-positions.i32'
                native.request('feed_image '+json.dumps(str(manifest))+' '+json.dumps(str(positions)))
                paths = [str(native.frontier('prefix')[0])]
                native.request('feed '+' '.join(map(str, fixture['suffixes'][branch])))
                path, values = native.frontier('suffix'); paths.append(str(path)); tokens = []
                for index in range(config['outputs']):
                    token = max(range(len(values)), key=values.__getitem__); tokens.append(token)
                    native.request('decode '+str(token))
                    path, values = native.frontier('output-'+str(index+1)); paths.append(str(path))
                    if token in profile['execution']['eos_tokens']: break
                result.update(tokens=tokens, frontiers=paths, work=native.request('stats'))
            finally: result['native_exit'] = native.close()
        else:
            os.environ['CTT_NATIVE_TRACE'] = str(out/'native-trace.jsonl')
            backend = StrataBackend(profile, gpu_id=0)
            result['native_pid'] = backend.process.pid
            api = BranchingModel(config['store'], backend, profile)
            if stage == 'prepare':
                fixture = json.loads(Path(config['fixture']).read_text())
                parts = fixture_parts(config['fixture'], config.get('images',[config['image']]))
                ram = SharedCheckpointPool(4*1024**3, checkpoint_idle_seconds=1800)
                disk = DiskCheckpointPool(api.store, 32*1024**3, checkpoint_idle_seconds=1800)
                producer = dict(kind='retained-executed-SVE-fixture', image=parts[0]['original'],
                                embedding=parts[0]['embedding'], fixture=fingerprint(config['fixture']))
                encoder_id = fixture.get('encoder_profile_id',_digest(producer))
                artifacts = ImageArtifacts(api.store, ram, disk, encoder_profile_id=encoder_id,
                                          expected_width=parts[0]['width'], original_bytes_limit=256*1024**2)
                ids = fixture['input_tokens']
                spans = []; cursor = 0
                with ExitStack() as leases:
                    for part in parts:
                        original = Path(part['original']['path']).read_bytes()
                        payload = Path(part['embedding']['path'])
                        held = leases.enter_context(artifacts.prepare(original=original, normalized=original,
                            encode=lambda _,path=payload:path.read_bytes(), pad_token_id=part['pad']))
                        spans.extend([dict(kind='text',tokens=ids[cursor:part['start']]),held.span])
                        cursor = part['start']+part['rows']
                    spans.append(dict(kind='text',tokens=ids[cursor:]))
                    prefix = api.create_inputs(spans, artifacts=artifacts)
                api.store.fork('prefix', prefix)
                result['prefix_save'] = api.save_branch('prefix', 0, prefix, 'save-prefix-v1')
                result['prefix'] = prefix
                result['parent_audit'] = backend.audit(api.cache[prefix].blob)
                result['prefix_chunks'] = api.cache[prefix].execution_schedule[0]['chunks']
                result['boundary_coverage'] = boundary_coverage(parts,result['prefix_chunks'],result['parent_audit'])
                if fixture.get('require_boundaries'):
                    coverage = result['boundary_coverage']
                    assert coverage['batch_boundaries_inside_images'], 'No observed native batch crossing'
                    assert coverage['physical_k_boundaries_inside_images'], 'No physical K-page crossing'
                    assert coverage['partial_qsa_tail_cells'], 'Prefix has no partial QSA tail'
                result['input_counts'] = dict(text_control_tokens=len(ids)-sum(p['rows'] for p in parts),
                    image_rows=sum(p['rows'] for p in parts),total_cells=len(ids),
                    grids=[[p['grid_width'],p['grid_height']] for p in parts])
                result['parent_audits_after_branches'] = []
                prefix_path = frontier(api, prefix, out/'prefix.f32')
                expected = {}
                for name in ('A', 'B'):
                    before_suffix = backend.work_counters()
                    tip = api.feed(prefix, fixture['suffixes'][name])
                    after_suffix = backend.work_counters()
                    suffix_work = {key:after_suffix[key]-before_suffix[key] for key in
                        ('forward_tokens','prefill_tokens','cow_pages','copied_bytes','captures','restores')}
                    assert suffix_work['forward_tokens']==len(fixture['suffixes'][name])
                    if fixture.get('require_boundaries'):
                        assert suffix_work['cow_pages']>0 and suffix_work['copied_bytes']>0
                    paths = [prefix_path, frontier(api, tip, out/(name+'-suffix.f32'))]
                    active = api.start_generation(tip, dict(temperature=0,
                        eos_tokens=profile['execution']['eos_tokens']), 'restart-greedy')
                    tokens = []; four = None
                    for index in range(config['outputs']):
                        generated = api.generate(active, 1)
                        if not generated.accepted_tokens: break
                        active = generated.snapshot_id; tokens.extend(generated.accepted_tokens)
                        paths.append(frontier(api, active, out/(name+'-output-'+str(index+1)+'.f32')))
                        if name == 'A' and len(tokens) == 4:
                            four = active; api.store.fork('generated-four', four)
                            result['generated_save'] = api.save_branch('generated-four', 0, four, 'save-four-v1')
                        if tokens[-1] in profile['execution']['eos_tokens']: break
                    expected[name] = dict(tokens=tokens, frontiers=paths, final=active,
                        controller=controller_file(api, active, out/(name+'-controller.json')),suffix_work=suffix_work)
                    if name == 'A': result['generated_four'] = four
                    audit_unchanged(backend,api.cache[prefix].blob,result['parent_audit'],result['parent_audits_after_branches'])
                result['expected'] = expected
                if len(parts)>1:
                    alternate = api.create_inputs(spans[:2]+[dict(kind='text',tokens=spans[2]['tokens'][:1])],
                                                  artifacts=artifacts)
                    first_info = backend._request('vision_info')
                    assert first_info['images']==1
                    before_switch = backend.work_counters()['forward_tokens']
                    restored = frontier(api,prefix,out/'two-images-after-one.f32')
                    after_switch = backend.work_counters()['forward_tokens']
                    result['grid_switch'] = dict(alternate_snapshot=alternate,alternate_info=first_info,
                        restored_info=backend._request('vision_info'),forward_tokens=after_switch-before_switch,
                        restored_comparison=compare_files(prefix_path,restored))
                    assert result['grid_switch']['forward_tokens']==0
                    assert result['grid_switch']['restored_info']['images']==len(parts)
                    assert result['grid_switch']['restored_comparison']['exact']
                    repeated = api.feed(prefix,fixture['suffixes']['A'])
                    actual = frontier(api,repeated,out/'A-suffix-after-one-image.f32')
                    result['grid_switch']['suffix_comparison']=compare_files(expected['A']['frontiers'][1],actual)
                    assert result['grid_switch']['suffix_comparison']['exact']
                    audit_unchanged(backend,api.cache[prefix].blob,result['parent_audit'],result['parent_audits_after_branches'])
                result['saved_files'] = [fingerprint(p) for p in sorted(api.checkpoint_directory.iterdir())
                                         if p.suffix in ('.state', '.json')]
                result['spans'] = spans
            else:
                # This worker deliberately reads no fixture, image or SVE path.
                forbidden = {str(Path(config['fixture']).resolve()),
                             *(str(Path(path).resolve()) for path in config.get('images',[config['image']]))}
                spawned = []
                def no_encoder_or_embedding(event, arguments):
                    if event == 'open' and isinstance(arguments[0], (str,bytes)):
                        path = os.fsdecode(arguments[0])
                        if path.endswith('.sve') or str(Path(path).resolve()) in forbidden:
                            raise AssertionError('Strict restore attempted an image/embedding read: '+path)
                    if event == 'subprocess.Popen':
                        spawned.append(os.fsdecode(arguments[0]))
                        raise AssertionError('Strict restore attempted an extra subprocess')
                # The one inference worker is already running; no encoder or
                # other helper may be launched for restoration/continuation.
                sys.addaudithook(no_encoder_or_embedding)
                prepared = json.loads(Path(config['prepare_result']).read_text())
                result['embedding_files_present'] = len(list(api.checkpoint_directory.glob('*.sve')))
                assert result['embedding_files_present'] == 0
                parent = prepared['prefix']
                snapshot = parent if stage == 'resume' else prepared['generated_four']
                assert snapshot is not None, 'Fixture stopped before four consumed outputs'
                result['restore'] = api.restore_checkpoint(snapshot)
                assert result['restore']['forward_tokens'] == 0
                info = backend._request('vision_info')
                assert info['validated_embedding_bytes'] == 0
                result['native_vision_after_restore'] = info
                result['restored_frontier'] = frontier(api, snapshot, out/'restored.f32')
                if stage == 'resume':
                    result['parent_audit'] = backend.audit(api.cache[parent].blob)
                    result['parent_audits_after_branches'] = []
                    result['branches'] = {}
                    for name in ('A', 'B'):
                        # The exact suffix IDs were copied into campaign config.
                        before_suffix = backend.work_counters()
                        tip = api.feed(parent, config['suffixes'][name])
                        after_suffix = backend.work_counters()
                        suffix_work = {key:after_suffix[key]-before_suffix[key] for key in
                            ('forward_tokens','prefill_tokens','cow_pages','copied_bytes','captures','restores')}
                        assert suffix_work['forward_tokens']==len(config['suffixes'][name])
                        paths = [result['restored_frontier'], frontier(api, tip, out/(name+'-suffix.f32'))]
                        active = api.start_generation(tip, dict(temperature=0,
                            eos_tokens=profile['execution']['eos_tokens']), 'restart-greedy')
                        tokens = []
                        for index in range(config['outputs']):
                            batch = api.generate(active, 1)
                            if not batch.accepted_tokens: break
                            active = batch.snapshot_id; tokens.extend(batch.accepted_tokens)
                            paths.append(frontier(api, active, out/(name+'-output-'+str(index+1)+'.f32')))
                            if tokens[-1] in profile['execution']['eos_tokens']: break
                        result['branches'][name] = dict(tokens=tokens, frontiers=paths,
                            controller=controller_file(api, active, out/(name+'-controller.json')),suffix_work=suffix_work)
                        audit_unchanged(backend,api.cache[parent].blob,result['parent_audit'],result['parent_audits_after_branches'])
                else:
                    remaining = len(prepared['expected']['A']['tokens'])-4
                    active = snapshot; tokens = []; paths = [result['restored_frontier']]
                    for index in range(remaining):
                        batch = api.generate(active,1)
                        if not batch.accepted_tokens: break
                        active = batch.snapshot_id; tokens.extend(batch.accepted_tokens)
                        paths.append(frontier(api,active,out/('resumed-output-'+str(index+5)+'.f32')))
                    result.update(tokens=tokens,frontiers=paths,final_frontier=paths[-1],
                        controller=controller_file(api,active,out/'controller.json'))
                assert backend._request('vision_info')['validated_embedding_bytes'] == 0
                result['extra_subprocesses'] = spawned
            result['work'] = backend.work_counters()
        result['status'] = 'PASS'
    except BaseException as error:
        result.update(status='FAIL', error=repr(error)); raise
    finally:
        if backend is not None:
            pid = backend.process.pid; backend.close()
            result['native_exit'] = dict(pid=pid, exit_code=backend.process.returncode,
                process_absent=not Path(f'/proc/{pid}').exists())
        if ram is not None: ram.close()
        save(out/'result.json', result)


def campaign(args):
    out = args.output.resolve(); out.mkdir(parents=True, exist_ok=False)
    build = json.loads(args.build.read_text()); fixture = json.loads(args.fixture.read_text())
    profile = copy.deepcopy(json.loads(args.profile.read_text()))
    parts = fixture_parts(args.fixture,args.image)
    chunk = args.prefill_chunk if args.prefill_chunk is not None else fixture.get('prefill_chunk')
    if chunk is not None: profile = configure_prefill(profile,chunk)
    profile['native_build'] = dict(sha256={x['path']: x['sha256'] for x in build['binaries'].values()},
                                  build_receipt=fingerprint(args.build))
    execution = profile['execution']; execution['sut_command'][0] = build['binaries']['cow']['path']
    execution['sut_command'] += ['--vision', '--gpu', '0']
    width = parts[0]['width']
    execution.update(resume_contract=RESUME_CONTRACT, vision_protocol=VISION_PROTOCOL,
        embedding_width=width, GPU=[0], checkpoint_cache=dict(byte_limit=16*1024**3, entry_limit=256))
    execution['environment']['CUDA_VISIBLE_DEVICES'] = '0'
    profile['model_contract'].update(input_kind='canonical-mixed-spans-v1', positions=POSITION_CONVENTION)
    profile['model_contract_id'] = _digest(profile['model_contract'])
    profile['execution_profile_id'] = _digest(dict(build=profile['native_build'], execution=execution, hardware=profile['hardware']))
    save(out/'profile.json', profile)
    config = dict(profile=str(out/'profile.json'), fixture=str(args.fixture.resolve()), image=str(args.image[0].resolve()),
        images=[str(path.resolve()) for path in args.image],
        store=str(out/'store'), outputs=args.outputs, reference_binary=build['binaries']['reference']['path'],
        suffixes=fixture['suffixes'], prepare_result=str(out/'prepare/result.json'))
    config_path = out/'config.json'; save(config_path, config)
    result = dict(status='RUNNING', stages=[], comparisons=[], controller_comparisons=[],
                  scope='Q4 image durable resume, greedy; every generated frontier',
                  fixture=fingerprint(args.fixture),image_count=len(parts),outputs_limit=args.outputs)
    def run(stage, name, branch='A'):
        directory = out/name; directory.mkdir()
        argv = [sys.executable, __file__, '--worker', stage, '--config', str(config_path),
                '--branch', branch, '--output', str(directory)]
        with (directory/'stdout.log').open('w') as stdout, (directory/'stderr.log').open('w') as stderr:
            process = subprocess.Popen(argv, stdout=stdout, stderr=stderr)
            started = time.monotonic()
            while True:
                try: code = process.wait(timeout=30); break
                except subprocess.TimeoutExpired:
                    print(json.dumps(dict(stage=name, python_pid=process.pid,
                        elapsed_seconds=time.monotonic()-started)), flush=True)
            row = dict(stage=name, argv=argv, python_pid=process.pid, exit_code=code,
                process_absent=not Path(f'/proc/{process.pid}').exists(), seconds=time.monotonic()-started)
            result['stages'].append(row); save(out/'result.json', result)
        assert code == 0, row
        value = json.loads((directory/'result.json').read_text())
        assert value['status']=='PASS' and value['native_exit']['exit_code']==0 and value['native_exit']['process_absent']
        assert not Path(f"/proc/{value['native_pid']}").exists()
        return value
    def exact(left, right, label):
        check = dict(label=label, **compare_files(left, right)); result['comparisons'].append(check)
        save(out/'result.json', result); assert check['exact'], check
    def controller_exact(left, right, label):
        a, b = Path(left['path']).read_bytes(), Path(right['path']).read_bytes()
        assert a == b, label
        result['controller_comparisons'].append(dict(label=label, bytes=len(a), sha256=hashlib.sha256(a).hexdigest()))
    try:
        prepared = run('prepare', 'prepare')
        # Remove only derived embeddings inside this isolated campaign store.
        removed = []
        for path in (out/'store/checkpoints').glob('*.sve'):
            removed.append(fingerprint(path)); path.unlink()
        result['removed_embeddings'] = removed; assert removed
        for name in ('A', 'B'):
            first = run('reference', 'reference-'+name+'-0', name)
            second = run('reference', 'reference-'+name+'-1', name)
            expected = prepared['expected'][name]
            assert first['tokens'] == second['tokens'] == expected['tokens']
            assert len(first['frontiers']) == len(second['frontiers']) == len(expected['frontiers'])
            for i, (a,b,c) in enumerate(zip(first['frontiers'], second['frontiers'], expected['frontiers'])):
                exact(a,b,f'{name}-fresh-repeat-{i}'); exact(a,c,f'{name}-uninterrupted-{i}')
        resumed = run('resume', 'resume')
        for name in ('A','B'):
            expected, actual = prepared['expected'][name], resumed['branches'][name]
            assert expected['tokens'] == actual['tokens']
            assert len(expected['frontiers']) == len(actual['frontiers'])
            for i,(a,b) in enumerate(zip(expected['frontiers'],actual['frontiers'])):
                exact(a,b,f'{name}-restart-{i}')
            controller_exact(expected['controller'], actual['controller'], name+'-controller')
        generated = run('resume-generated', 'resume-generated')
        expected = prepared['expected']['A']
        assert generated['tokens'] == expected['tokens'][4:]
        assert len(expected['frontiers'][5:]) == len(generated['frontiers'])
        for index,(a,b) in enumerate(zip(expected['frontiers'][5:],generated['frontiers'])):
            exact(a,b,'generated-four-restart-output-'+str(index+4))
        controller_exact(expected['controller'], generated['controller'], 'generated-controller')
        for item in prepared['saved_files']:
            assert fingerprint(item['path']) == item, 'Saved prefix/generated bytes changed'
        result.update(status='PASS', image_encoder_calls=0, max_abs_error=0.0,
                      qualification='bounded image durable resume; not complete VC05-VC09')
    except BaseException as error:
        result.update(status='FAIL', error=repr(error)); raise
    finally: save(out/'result.json', result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', choices=('prepare','reference','resume','resume-generated'))
    parser.add_argument('--config', type=Path); parser.add_argument('--branch', default='A')
    for name in ('profile','build','fixture','output'): parser.add_argument('--'+name,type=Path)
    parser.add_argument('--image',type=Path,action='append')
    parser.add_argument('--prefill-chunk',type=int)
    parser.add_argument('--outputs',type=int,default=8)
    args = parser.parse_args()
    if args.worker: worker(json.loads(args.config.read_text()), args.worker, args.branch, args.output)
    else: campaign(args)
