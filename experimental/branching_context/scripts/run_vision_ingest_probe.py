"""Bounded real-GPU VC03 parity with independently supplied reference positions.

This native ingest milestone does not advertise image checkpoint restoration.
Every reference is a new process with no checkpoint/prompt-cache import path.
"""
import argparse
from array import array
import hashlib
import json
import math
import os
from pathlib import Path
import queue
import struct
import subprocess
import sys
import threading
import time


def save(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2)+'\n')
    temp.replace(path)


def fingerprint(path):
    path = Path(path)
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    return dict(path=str(path), bytes=path.stat().st_size, sha256=digest)


class Native:
    def __init__(self, argv, directory):
        self.directory, self.argv = directory, argv
        directory.mkdir()
        self.stderr = (directory/'native.stderr').open('w')
        self.transcript = (directory/'native.stdout').open('w')
        self.events, self.receipts = queue.Queue(), []
        start = time.monotonic()
        self.proc = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=self.stderr, text=True, bufsize=1, env={**os.environ, 'CUDA_VISIBLE_DEVICES': '0'})
        def reader():
            for line in self.proc.stdout:
                self.events.put(line)
            self.events.put(None)
        threading.Thread(target=reader, daemon=True).start()
        try:
            self.ready = self.response('startup')
            assert self.ready['status'] == 'ready', self.ready
        except BaseException:
            self.proc.kill(); self.proc.wait(timeout=30)
            self.stderr.close(); self.transcript.close()
            raise
        self.startup_seconds = time.monotonic()-start

    def response(self, phase):
        start = time.monotonic()
        while True:
            try:
                line = self.events.get(timeout=30)
            except queue.Empty:
                print(json.dumps(dict(phase=phase, pid=self.proc.pid,
                                      elapsed_seconds=time.monotonic()-start)), flush=True)
                if time.monotonic()-start > 180:
                    raise TimeoutError(phase)
                continue
            if line is None:
                raise RuntimeError(f'Native process exited during {phase}; inspect {self.directory}')
            self.transcript.write(line); self.transcript.flush()
            try:
                response = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(response, dict) and 'status' in response:
                return response

    def request(self, command, allow_error=False):
        start = time.monotonic()
        self.proc.stdin.write(command+'\n'); self.proc.stdin.flush()
        value = self.response(command.split()[0])
        self.receipts.append(dict(command=command, response=value, seconds=time.monotonic()-start))
        save(self.directory/'receipts.json', self.receipts)
        if not allow_error:
            assert value['status'] in ('ok', 'closed'), value
        return value

    def frontier(self, name):
        path = self.directory/(name+'.f32')
        assert not any(char.isspace() for char in str(path)), 'Native frontier protocol requires a path without spaces'
        self.request('logits '+str(path))
        # The original COW logits handler accepts whitespace-delimited paths;
        # caller uses names without spaces, while the fresh driver accepts quoting.
        raw = path.read_bytes()
        values = array('f'); values.frombytes(raw)
        assert len(values) == self.ready['vocab'] and all(math.isfinite(x) for x in values)
        return path, values

    def close(self):
        forced = False
        try:
            if self.proc.poll() is None:
                self.request('quit')
                self.proc.wait(timeout=30)
        finally:
            if self.proc.poll() is None:
                self.proc.kill(); self.proc.wait(timeout=30); forced = True
            self.stderr.close(); self.transcript.close()
        return dict(pid=self.proc.pid, exit_code=self.proc.returncode, forced_kill=forced,
                    process_absent=not Path(f'/proc/{self.proc.pid}').exists())


def compare_files(expected, actual):
    left, right = Path(expected).read_bytes(), Path(actual).read_bytes()
    assert len(left) == len(right)
    a, b = array('f'), array('f')
    a.frombytes(left); b.frombytes(right)
    error = max((abs(x-y) for x, y in zip(a, b)), default=0.0)
    return dict(expected=str(expected), actual=str(actual), values=len(a),
                max_abs_error=error, bytes_identical=left == right, exact=error == 0.0 and left == right)


def campaign(args):
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    build = json.loads(args.build.read_text())
    profile = json.loads(args.profile.read_text())
    capacity = profile['execution']['capacity']
    original_tokens = json.loads(args.tokens.read_text())
    raw = args.embedding.read_bytes()
    magic, rows, nx, ny, width = struct.unpack('<5i', raw[:20])
    assert magic == 0x31455653 and len(raw) == 20+rows*width*4 and rows == nx*ny
    pad = 248056
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]/'tools'))
    from strata_tokenizer import Tokenizer
    tokenizer = Tokenizer.from_gguf(profile['model_path'])
    if args.fixture == 'forced':
        # Retain the initial forced four-token suffix case as its own fixture.
        tokens = original_tokens[:3]+[198]*98+original_tokens[3:]
        suffixes = {'A': [198, 3710, 30, 198], 'B': [198, 1829, 30, 198]}
    else:
        # End the shared prefix after the complete image and its closing token.
        # Both continuations then ask a proper question inside the user message.
        end_image = original_tokens.index(pad)+rows
        tokens = original_tokens[:3]+[198]*123+original_tokens[3:end_image+1]
        suffixes = {}
        for branch, side in (('A', 'left'), ('B', 'right')):
            question = f'Describe the {side} half of the picture in two complete sentences.'
            suffixes[branch] = tokenizer.encode(question+
                '<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n', parse_special=True)
    start = tokens.index(pad)
    assert tokens[start:start+rows] == [pad]*rows and pad not in tokens[start+rows:]
    digest = hashlib.sha256(raw).hexdigest()
    manifest = out/'input.manifest'
    header = f'COW_VISION_INPUT_V1 0 {len(tokens)} 1\n'
    header += ' '.join(map(str, tokens))+'\n'
    placement = f'{start} {rows} {nx} {ny} {width} {pad} {digest} {json.dumps(str(args.embedding.resolve()))}\n'
    manifest.write_text(header+placement)
    corrupt = out/'bad-digest.manifest'
    corrupt.write_text(header+placement.replace(digest, '0'*64))
    # Independent CPU coordinates, supplied to the fresh reference as bytes.
    positions = [(p, p, p) for p in range(start)]
    positions += [(start, start+y, start+x) for y in range(ny) for x in range(nx)]
    rotary = start+max(nx, ny)
    positions += [(p, p, p) for p in range(rotary, rotary+capacity-start-rows)]
    flat = [value for point in positions for value in point]
    position_file = out/'reference-positions.i32'
    position_file.write_bytes(struct.pack('<'+'i'*len(flat), *flat))
    def position_digest(consumed):
        cursor = consumed-rows+max(nx, ny)
        payload = b'COW-MROPE-v1\0'+struct.pack('<QQ', consumed, cursor)
        payload += struct.pack('<'+'i'*(consumed*3), *flat[:consumed*3])
        return hashlib.sha256(payload).hexdigest(), cursor
    result = dict(schema=1, status='RUNNING', scope='VC03 native image ingest and matching-schedule parity',
        started_ns=time.time_ns(), source_build=fingerprint(args.build), base_profile=fingerprint(args.profile),
        embedding=fingerprint(args.embedding), input_tokens=tokens,
        text_control_cells=len(tokens)-rows, image_rows=rows, total_prefix_cells=len(tokens),
        image_start=start, grid=[nx, ny], model_width=width, configuration=profile['execution'],
        fixture=args.fixture, suffixes=suffixes, tokenizer_source=fingerprint(
            Path(__file__).resolve().parents[3]/'tools/strata_tokenizer.py'),
        runs=[], comparisons=[], image_checkpoint_restore='UNIMPLEMENTED', cumulative_gates='UNQUALIFIED')
    save(out/'result.json', result)
    prior = {}
    try:
        for branch in ('A', 'B'):
            for kind, repetition in (('reference', 1), ('reference', 2), ('cow', 1)):
                name = f'{branch}-{kind}-{repetition}'
                binary = build['binaries'][kind]
                assert fingerprint(binary['path'])['sha256'] == binary['sha256']
                argv = list(profile['execution']['sut_command'])
                argv[0] = binary['path']
                argv += ['--vision', '--gpu', '0']
                native = Native(argv, out/name)
                run = dict(name=name, branch=branch, kind=kind, argv=argv, binary=binary,
                           pid=native.proc.pid, startup_seconds=native.startup_seconds, frontiers=[], generated=[])
                result['runs'].append(run)
                print(json.dumps(dict(phase='native-ready', run=name, pid=native.proc.pid,
                                      startup_seconds=native.startup_seconds)), flush=True)
                try:
                    initial = native.request('stats')
                    assert initial['forward_tokens'] == 0
                    if kind == 'cow':
                        info = native.request('vision_info')
                        assert info['enabled'] and info['vision_protocol'] == 'explicit-image-rows-mrope-v1'
                        bad = native.request('feed_image '+json.dumps(str(corrupt)), allow_error=True)
                        assert bad['status'] == 'error' and 'digest' in bad['message']
                        assert native.request('stats')['forward_tokens'] == 0
                        run['corrupt_payload_rejected_before_forward'] = True
                    command = 'feed_image '+json.dumps(str(manifest))
                    if kind == 'reference':
                        command += ' '+json.dumps(str(position_file))
                    prefix = native.request(command)
                    assert prefix['consumed'] == len(tokens)
                    run['prefix_receipt'] = prefix
                    if kind == 'cow':
                        info = native.request('vision_info')
                        expected, cursor = position_digest(len(tokens))
                        assert info['position_sha256'] == expected and info['rotary_cursor'] == cursor
                        assert info['validated_embedding_bytes'] == len(raw)
                        run['position_info'] = info
                        if info.get('hot_image_snapshots'):
                            work = native.request('stats')['forward_tokens']
                            captured = native.request('capture')
                            native.request('restore '+str(captured['handle']))
                            assert native.request('stats')['forward_tokens'] == work
                            native.request('drop '+str(captured['handle']))
                        else:
                            rejected = native.request('capture', allow_error=True)
                            assert rejected['status'] == 'error'
                    path, values = native.frontier('prefix')
                    run['frontiers'].append(str(path))
                    suffix = native.request('feed '+' '.join(map(str, suffixes[branch])))
                    run['suffix_receipt'] = suffix
                    path, values = native.frontier('suffix')
                    run['frontiers'].append(str(path))
                    generated_start = time.monotonic()
                    for index in range(8):
                        token = max(range(len(values)), key=values.__getitem__)
                        run['generated'].append(token)
                        receipt = native.request('decode '+str(token))
                        assert receipt['consumed'] == len(tokens)+len(suffixes[branch])+index+1
                        path, values = native.frontier(f'generated-{index+1}')
                        run['frontiers'].append(str(path))
                        if token in profile['execution']['eos_tokens']:
                            break
                    run['generation_audit_wall_seconds'] = time.monotonic()-generated_start
                    run['output_count'] = len(run['generated'])
                    run['output_text'] = tokenizer.decode(run['generated'])
                    run['stopped_on_eos'] = run['generated'][-1] in profile['execution']['eos_tokens']
                    stats = native.request('stats')
                    assert stats['forward_tokens'] == len(tokens)+len(suffixes[branch])+len(run['generated'])
                    assert stats['prefill_tokens'] == len(tokens)+len(suffixes[branch])
                    run['native_stats'] = stats
                    run['native_status'] = Path(f'/proc/{native.proc.pid}/status').read_text()
                    run['gpu_memory'] = subprocess.check_output(['nvidia-smi',
                        '--query-compute-apps=pid,process_name,used_gpu_memory', '--format=csv,noheader'], text=True)
                    if branch not in prior:
                        prior[branch] = run
                    else:
                        reference = prior[branch]
                        assert prefix['chunks'] == reference['prefix_receipt']['chunks']
                        assert suffix['chunks'] == reference['suffix_receipt']['chunks']
                        for expected, actual in zip(reference['frontiers'], run['frontiers']):
                            comparison = compare_files(expected, actual)
                            result['comparisons'].append(comparison)
                            save(out/'result.json', result)
                            assert comparison['exact'], comparison
                        assert len(run['frontiers']) == len(reference['frontiers'])
                        assert run['generated'] == reference['generated'], 'matching-schedule generated tokens differ'
                    run['status'] = 'PASS'
                finally:
                    run['exit'] = native.close()
                    save(out/'result.json', result)
                assert run['exit']['exit_code'] == 0 and run['exit']['process_absent'] and not run['exit']['forced_kill']
                print(json.dumps(dict(phase='case-complete', run=name, outputs=run['output_count'],
                    comparisons=len(result['comparisons']), max_error=max(
                        [c['max_abs_error'] for c in result['comparisons']], default=0))), flush=True)
        result['status'] = 'PASS'
        result['max_logit_abs_error'] = max(c['max_abs_error'] for c in result['comparisons'])
    except BaseException as error:
        result.update(status='FAIL', error=repr(error))
        raise
    finally:
        result['finished_ns'] = time.time_ns()
        save(out/'result.json', result)
    print(json.dumps(dict(status=result['status'], runs=len(result['runs']),
                         comparisons=len(result['comparisons']), max_error=result['max_logit_abs_error'])), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('profile', 'build', 'embedding', 'tokens', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--fixture', choices=('forced', 'natural'), default='natural')
    campaign(parser.parse_args())
