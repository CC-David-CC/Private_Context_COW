"""Offline audit of retained real-GPU namespace HTTP evidence; no inference."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import struct

from context_time_travel import Store, controller
from context_time_travel.store import _json_bytes
from context_time_travel.checkpoint_cache import validate_manifest
from context_time_travel.execution_receipt import validate_operator_schedule
from context_time_travel.strata_backend import _digest
from native_request_probe import publish
from serve.cow_workspace import CowWorkspace
from tools.strata_tokenizer import Tokenizer


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def compare_vectors(actual, expected, count):
    if len(actual) != 4*count or len(expected) != 4*count:
        raise ValueError('Incomplete full-vector evidence')
    left = struct.unpack('<' + str(count) + 'f', actual)
    right = struct.unpack('<' + str(count) + 'f', expected)
    if not all(map(math.isfinite, left + right)):
        raise ValueError('Nonfinite full-vector evidence')
    if actual != expected:
        raise ValueError('Full-vector mismatch; max absolute error=' +
                         str(max(abs(a-b) for a, b in zip(left, right))))
    return 0.0


def checkpoint(store, identity, profile, vocab):
    history = store.lineage(identity)[1]
    record = store.get_snapshot(identity)
    if record['model_contract_id'] != profile['model_contract_id']:
        raise ValueError('Canonical model identity differs')
    directory = store.root/'checkpoints'
    envelope = json.loads((directory/(identity+'.json')).read_bytes())
    required = dict(schema=3, snapshot=identity, consumed=len(history),
        input_identity=hashlib.sha256(_json_bytes(list(history))).hexdigest(),
        model_contract_id=profile['model_contract_id'],
        execution_profile_id=profile['execution_profile_id'],
        positions='absolute-zero-based', logits_count=vocab)
    manifest = validate_manifest(envelope, required)
    if profile['execution']['resume_contract'] != 'identical-native-operators-v1':
        raise ValueError('This auditor requires the qualified native-operator contract')
    digest = validate_operator_schedule(manifest['execution_schedule'], history,
                                        profile['execution']['prefill_chunk'])
    if digest != manifest['schedule_sha256']:
        raise ValueError('Saved execution receipt differs')
    payload = directory/(manifest['payload_sha256']+'.state')
    if payload.stat().st_size != manifest['state_bytes']+4*vocab or sha(payload) != manifest['payload_sha256']:
        raise ValueError('Saved payload integrity differs')
    with payload.open('rb') as stream:
        stream.seek(manifest['state_bytes']); frontier = stream.read(4*vocab)
    return dict(history=list(history), controller=record['controller'],
                frontier=frontier, payload_sha256=manifest['payload_sha256'])


def audit(directory):
    directory = Path(directory).resolve()
    raw = json.loads((directory/'result.json').read_bytes())
    profile = json.loads((directory/'profile.json').read_bytes())
    fixture = json.loads((directory/'fixture.json').read_bytes())
    if raw['status'] != 'PASS' or raw['native_exit'] != 0:
        raise ValueError('GPU run is not terminal PASS')
    if len(raw['requests']) != 2 or [r['name'] for r in raw['requests']] != ['A', 'B']:
        raise ValueError('Incomplete A/B evidence')
    identity = _digest(dict(build=profile['native_build'], execution=profile['execution'],
                            hardware=profile['hardware']))
    if identity != profile['execution_profile_id'] or identity != raw['execution_profile_id']:
        raise ValueError('Profile identity differs')
    for path, expected in profile['native_build']['sha256'].items():
        if sha(path) != expected: raise ValueError('Build/helper fingerprint changed: '+path)
    if not (directory/'store'/'objects').is_dir():
        raise ValueError('Canonical Store is missing')
    store = Store(directory/'store')
    tokenizer = Tokenizer.from_gguf(profile['model_path']); vocab = len(tokenizer.tokens)
    workspace = CowWorkspace(store, tokenizer)
    root = raw['prefix_creation']['final']['receipt']['snapshot']
    parent = checkpoint(store, root, profile, vocab)
    prefix = fixture['prefixes'][0]
    if parent['history'] != prefix or parent['payload_sha256'] != raw['parent_payload_sha256']:
        raise ValueError('Parent history or saved bytes changed')
    output = dict(status='RUNNING', scope='offline audit of real-GPU retained evidence; no new inference',
        raw_result_sha256=sha(directory/'result.json'), profile_sha256=sha(directory/'profile.json'),
        auditor_sha256=sha(__file__), execution_profile_id=identity,
        model_contract_id=profile['model_contract_id'], model_weights_sha256=profile['model_weights_sha256'],
        full_vector_comparisons=0, numerical_max_error=0.0, branches=[])
    for index, row in enumerate(raw['requests']):
        name = row['name']; suffix = fixture['suffixes'][index]
        stem = directory/('fresh-http-'+name+'.f32')
        work = json.loads((directory/('fresh-http-'+name+'.stdout')).read_bytes())
        schedule = json.loads(Path(str(stem)+'.schedule.json').read_bytes())
        if schedule['requests'] != [len(prefix), len(suffix)]:
            raise ValueError('Fresh reference request boundaries differ')
        command = row['reference']['command']
        if Path(command[0]).resolve() == Path(profile['execution']['sut_command'][0]).resolve():
            raise ValueError('Oracle executable is the system under test')
        for flag in ('--prompt-cache', '--conversation-cache-mib'):
            if command[command.index(flag)+1] != '0': raise ValueError('Oracle reuse is enabled')
        tokens = work['generated']; answer = row['final']['answer']
        if not tokens or tokens != answer['tokens'] or tokens != row['reference']['work']['generated']:
            raise ValueError('Output token sequence differs')
        if (work['prefill_tokens'] != len(prefix)+len(suffix)
                or work['normal_forward_tokens'] != len(tokens)
                or work['consumed'] != len(prefix)+len(suffix)+len(tokens)):
            raise ValueError('Fresh native work counters differ')
        counters = answer['counter_deltas']
        if counters['prefill_tokens'] != len(suffix) or counters['forward_tokens'] != len(suffix)+len(tokens):
            raise ValueError('Restored branch repeated prefix work or omitted work')
        target = checkpoint(store, row['final']['receipt']['snapshot'], profile, vocab)
        if target['history'] != prefix+suffix+tokens:
            raise ValueError('Canonical branch history differs from independent reference')
        operations = {e['operation'] for e in row['events'] if 'operation' in e}
        if len(operations) != 1: raise ValueError('Ambiguous request operation')
        expected = controller.start(dict(temperature=0,eos_tokens=profile['execution']['eos_tokens']),
            operations.pop(), tuple(prefix+suffix), vocab)
        expected = controller.consume_many(expected, ((t,tokenizer.token_bytes(t),
            t in profile['execution']['eos_tokens']) for t in tokens))
        if target['controller'] != expected or target['controller'] != row['state']['controller']:
            raise ValueError('Complete greedy controller differs')
        prompts = Path(str(stem)+'.requests.f32'); generated = Path(str(stem)+'.decode.f32')
        if prompts.stat().st_size != 2*4*vocab or generated.stat().st_size != len(tokens)*4*vocab:
            raise ValueError('Incomplete independent vector series')
        with prompts.open('rb') as stream: fresh_root = stream.read(4*vocab)
        with generated.open('rb') as stream:
            stream.seek((len(tokens)-1)*4*vocab); fresh_final = stream.read(4*vocab)
        compare_vectors(parent['frontier'], fresh_root, vocab)
        compare_vectors(target['frontier'], fresh_final, vocab)
        output['full_vector_comparisons'] += 2
        projected = workspace.history('http-'+name)
        if projected['messages'] != row['projected_history']['messages']:
            raise ValueError('Saved presentation differs from canonical token projection')
        output['branches'].append(dict(branch=name, prefix_tokens=len(prefix), suffix_tokens=len(suffix),
            generated_tokens=len(tokens), controller_exact=True, counters=counters,
            checkpoint_payload_sha256=target['payload_sha256'], frontier_sha256=hashlib.sha256(target['frontier']).hexdigest()))
    output['status'] = 'PASS'
    output['limits'] = ['root and final vectors only, not every generated frontier',
                        'greedy matching-schedule contract; not original G1-G5 qualification']
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    try: result = audit(args.directory)
    except Exception as error:
        publish(args.output, dict(status='FAIL', auditor_sha256=sha(__file__), failure=repr(error)))
        raise
    publish(args.output, result)
    print(json.dumps(result))


if __name__ == '__main__': main()
