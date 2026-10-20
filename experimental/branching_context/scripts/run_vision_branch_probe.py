"""VC010/011: real image-prefix reuse, retained bytes and image/text activation.

This is a private native milestone. It does not claim canonical API publication,
controller persistence, disk restart, or a full vision qualification.
"""
import argparse
import json
from pathlib import Path
import time

from run_vision_ingest_probe import Native, compare_files, fingerprint, save


def campaign(args):
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    profile = json.loads(args.profile.read_text())
    build = json.loads(args.build.read_text())
    fixture = json.loads(args.fixture.read_text())
    assert fixture['status'] == 'PASS' and fixture['fixture'] == 'natural'
    manifest = args.fixture.resolve().parent/'input.manifest'
    positions = args.fixture.resolve().parent/'reference-positions.i32'
    tokens, suffixes = fixture['input_tokens'], fixture['suffixes']
    assert fingerprint(fixture['embedding']['path']) == fixture['embedding']
    result = dict(schema=1, status='RUNNING', started_ns=time.time_ns(),
        scope='VC010/011 native hot image capture and image/text branch isolation',
        build=fingerprint(args.build), profile=fingerprint(args.profile),
        fixture=fingerprint(args.fixture), input_manifest=fingerprint(manifest),
        reference_positions=fingerprint(positions), runs=[], comparisons=[],
        byte_audits=[], restores=[], checkpoints=[], assertions=[],
        disk_restart='UNIMPLEMENTED', canonical_api='UNIMPLEMENTED')
    saved = {}

    def exact(expected, actual, label):
        value = dict(label=label, **compare_files(expected, actual))
        result['comparisons'].append(value)
        save(out/'result.json', result)
        assert value['exact'], value

    def launch(kind, name):
        binary = build['binaries'][kind]
        assert fingerprint(binary['path']) == binary
        argv = list(profile['execution']['sut_command'])
        argv[0] = binary['path']
        argv += ['--vision', '--gpu', '0']
        native = Native(argv, out/name)
        run = dict(name=name, kind=kind, argv=argv, binary=binary,
            pid=native.proc.pid, startup_seconds=native.startup_seconds)
        result['runs'].append(run)
        print(json.dumps(dict(phase='ready', name=name, pid=native.proc.pid)), flush=True)
        return native, run

    def restore(native, handle, expected, label):
        before = native.request('stats')['forward_tokens']
        receipt = native.request('restore '+str(handle))
        path, _ = native.frontier(label)
        after = native.request('stats')['forward_tokens']
        exact(expected, path, label)
        result['restores'].append(dict(label=label, receipt=receipt, forward_delta=after-before,
            position_info=native.request('vision_info')))
        assert before == after, 'restore evaluated saved inputs'

    def audit(native, handle, label):
        path = out/(label+'.audit')
        receipt = native.request('audit '+str(handle)+' '+json.dumps(str(path)))
        value = dict(label=label, receipt=receipt, **fingerprint(path))
        result['byte_audits'].append(value)
        return value

    def generate(native, branch, count, resume=False):
        paths, generated, checkpoint = [], [], None
        _, values = native.frontier(branch+'-generation-start')
        for index in range(count):
            token = max(range(len(values)), key=values.__getitem__)
            native.request('decode '+str(token))
            path, values = native.frontier(f'{branch}-generated-{index+1}')
            generated.append(token)
            paths.append(path)
            if not resume and branch == 'A' and index == 3:
                checkpoint = native.request('capture')['handle']
            if token in profile['execution']['eos_tokens']:
                break
        return paths, generated, checkpoint

    try:
        native, run = launch('cow', 'cow')
        try:
            assert native.request('stats')['forward_tokens'] == 0
            prefix_receipt = native.request('feed_image '+json.dumps(str(manifest)))
            prefix, _ = native.frontier('prefix')
            prefix_info = native.request('vision_info')
            parent = native.request('capture')['handle']  # RED: ingest-only driver refuses.
            parent_bytes = audit(native, parent, 'parent-before')
            for branch in ('A', 'B'):
                if branch == 'B':
                    restore(native, parent, prefix, 'restore-parent-for-B')
                    info = native.request('vision_info')
                    assert info['position_sha256'] == prefix_info['position_sha256']
                suffix_receipt = native.request('feed '+' '.join(map(str, suffixes[branch])))
                after_suffix, _ = native.frontier(branch+'-suffix')
                frontiers, generated, four = generate(native, branch, 8)
                saved[branch] = dict(frontiers=[prefix, after_suffix]+frontiers,
                    generated=generated, prefix_receipt=prefix_receipt, suffix_receipt=suffix_receipt,
                    four=four)
                unchanged = audit(native, parent, 'parent-after-'+branch)
                assert (unchanged['bytes'], unchanged['sha256']) == (parent_bytes['bytes'], parent_bytes['sha256'])
                print(json.dumps(dict(phase='branch-complete', branch=branch, outputs=len(generated))), flush=True)
            assert saved['A']['four'] is not None, 'fixture stopped before generated checkpoint boundary'
            restore(native, saved['A']['four'], saved['A']['frontiers'][5], 'restore-A-after-four')
            replay, replay_tokens, _ = generate(native, 'A-replay', len(saved['A']['generated'])-4, resume=True)
            assert replay_tokens == saved['A']['generated'][4:]
            for index, path in enumerate(replay):
                exact(saved['A']['frontiers'][6+index], path, f'A-replayed-output-{index+5}')
            native.request('reset')
            native.request('feed '+' '.join(map(str, tokens)))
            text_prefix, _ = native.frontier('text-prefix')
            text_info = native.request('vision_info')
            assert text_info['images'] == 0 and text_info['rotary_cursor'] == len(tokens)
            text_handle = native.request('capture')['handle']
            text_suffix = [198, 3710, 30, 198]
            native.request('feed '+' '.join(map(str, text_suffix)))
            text_end, _ = native.frontier('text-suffix')
            restore(native, parent, prefix, 'image-from-text')
            assert native.request('vision_info')['position_sha256'] == prefix_info['position_sha256']
            # Reading saved logits alone cannot expose a missing position-table
            # restore. Actually continue the image branch after text reset.
            native.request('feed '+' '.join(map(str, suffixes['A'])))
            after_text_suffix, _ = native.frontier('image-A-after-text-suffix')
            exact(saved['A']['frontiers'][1], after_text_suffix, 'image-A-after-text-suffix')
            after_text, after_text_tokens, _ = generate(native, 'image-A-after-text',
                len(saved['A']['generated']), resume=True)
            assert after_text_tokens == saved['A']['generated']
            for index, path in enumerate(after_text):
                exact(saved['A']['frontiers'][2+index], path, f'image-A-after-text-output-{index+1}')
            restore(native, text_handle, text_prefix, 'text-from-image')
            assert native.request('vision_info')['position_sha256'] == text_info['position_sha256']
            native.request('feed '+' '.join(map(str, text_suffix)))
            text_replay, _ = native.frontier('text-suffix-repeat')
            exact(text_end, text_replay, 'text-suffix-repeat')
            unchanged = audit(native, parent, 'parent-final')
            assert (unchanged['bytes'], unchanged['sha256']) == (parent_bytes['bytes'], parent_bytes['sha256'])
            work = native.request('stats')['forward_tokens']
            native.request('drop '+str(parent))
            rejected = native.request('restore '+str(parent), allow_error=True)
            assert rejected['status'] == 'error'
            after_rejected, _ = native.frontier('after-stale-handle')
            exact(text_replay, after_rejected, 'stale-handle-leaves-active-state')
            assert native.request('stats')['forward_tokens'] == work
            run['stats'] = native.request('stats')
            expected_work = (len(tokens)*2 + sum(len(suffixes[b])+len(saved[b]['generated']) for b in ('A','B'))
                             + len(replay_tokens) + 2*len(text_suffix)
                             + len(suffixes['A']) + len(after_text_tokens))
            assert run['stats']['forward_tokens'] == expected_work
            run.update(status='PASS', expected_forward_tokens=expected_work,
                generated={b: saved[b]['generated'] for b in ('A', 'B')},
                outputs={b: len(saved[b]['generated']) for b in ('A', 'B')})
            saved['text'] = dict(frontiers=[text_prefix, text_end], generated=[], suffix=text_suffix)
            result['assertions'] += ['parent-audit-unchanged-after-both-siblings-and-switches',
                'restore-forward-work-zero', 'generated-boundary-replay-exact',
                'image-text-image-position-activation', 'stale-handle-rejected-without-active-change']
        finally:
            run['exit'] = native.close()
            save(out/'result.json', result)
        assert run['exit']['exit_code'] == 0 and run['exit']['process_absent'] and not run['exit']['forced_kill']

        for branch in ('A', 'B', 'text'):
            for repetition in (1, 2):
                native, run = launch('reference', f'{branch}-reference-{repetition}')
                try:
                    if branch == 'text':
                        native.request('feed '+' '.join(map(str, tokens)))
                        suffix = saved['text']['suffix']
                    else:
                        native.request('feed_image '+json.dumps(str(manifest))+' '+json.dumps(str(positions)))
                        suffix = suffixes[branch]
                    prefix, _ = native.frontier('prefix')
                    native.request('feed '+' '.join(map(str, suffix)))
                    suffix_path, _ = native.frontier('suffix')
                    frontiers, generated, _ = generate(native, branch+'-reference', len(saved[branch]['generated']), resume=True)
                    assert generated == saved[branch]['generated']
                    for index, (expected, actual) in enumerate(zip(saved[branch]['frontiers'], [prefix, suffix_path]+frontiers)):
                        exact(expected, actual, f'{branch}-reference-{repetition}-frontier-{index}')
                    run.update(status='PASS', generated=generated, stats=native.request('stats'))
                finally:
                    run['exit'] = native.close()
                    save(out/'result.json', result)
                assert run['exit']['exit_code'] == 0 and run['exit']['process_absent'] and not run['exit']['forced_kill']
        result.update(status='PASS', max_logit_abs_error=max(v['max_abs_error'] for v in result['comparisons']))
    except BaseException as error:
        result.update(status='FAIL', error=repr(error))
        raise
    finally:
        result['finished_ns'] = time.time_ns()
        save(out/'result.json', result)
    print(json.dumps(dict(status=result['status'], comparisons=len(result['comparisons']),
        max_error=result['max_logit_abs_error'], restores=len(result['restores']))), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('profile', 'build', 'fixture', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    campaign(parser.parse_args())
