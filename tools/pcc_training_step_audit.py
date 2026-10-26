"""Read-only first-step full AdamW diagnostics; not a model acceptance gate.

The caller supplies an original completed PCC candidate and its exact base.
This optional owner tool reads artifacts, never evaluates or updates a model.
NumPy is imported only when the audit is requested.
"""
from concurrent.futures import ThreadPoolExecutor
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import struct

from tools.gguf_reader import GGUFFile


def _sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def _cookie(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError('Audit artifacts must be regular files, not links')
    stat = path.stat()
    return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)


def audit_first_adamw_step(base, candidate, *, chunk_elements=1 << 20):
    """Check identities, component bounds and stored-gradient alignment.

    The first fresh zero-decay AdamW step has |delta_i| <= learning_rate.
    The first moment gives the recorded gradient via m/(1-beta1). Its dot
    product with the observed weight delta is a local first-order prediction,
    not independent proof that the derivative or candidate behavior is correct.
    """
    import numpy as np
    if type(chunk_elements) is not int or not 1 <= chunk_elements <= 1 << 24:
        raise ValueError('Bounded positive audit chunk required')
    receipt = candidate['receipt']
    checkpoint = receipt['optimizer_checkpoint']
    config = checkpoint['config']
    if (candidate['mode'] != 'full' or candidate.get('optimizer_state_policy') != 'fresh'
            or config['optimizer'] != 'adamw' or checkpoint['completed_steps'] != 1
            or len(receipt['steps']) != 1 or config['weight_decay'] != 0
            or config['aux_weight_decay'] != 0):
        raise ValueError('Audit requires one fresh full AdamW step with zero decay')
    rate, beta = config['learning_rate'], config['beta1']
    if not math.isfinite(rate) or rate <= 0 or not 0 <= beta < 1:
        raise ValueError('Finite AdamW rate and first-moment coefficient required')
    paths = [Path(base), Path(candidate['candidate_path']), Path(candidate['optimizer_state_path'])]
    expected = [candidate['base_sha256'], candidate['candidate_sha256'], candidate['optimizer_state_sha256']]
    cookies = [_cookie(p) for p in paths]
    with ThreadPoolExecutor(max_workers=3) as pool:
        if list(pool.map(_sha, paths)) != expected:
            raise ValueError('Pinned audit artifact SHA-256 differs')
    left, right = GGUFFile(paths[0]), GGUFFile(paths[1])
    before, after = ({t.name: t for t in model.tensors} for model in (left, right))
    if before.keys() != after.keys() or len(before) != len(left.tensors) or len(after) != len(right.tensors):
        raise ValueError('Full model tensor coverage differs')
    with paths[2].open('rb') as stream:
        if stream.read(8) != b'PCCOPT1\n':
            raise ValueError('Unknown completed optimizer state format')
        size = struct.unpack('<Q', stream.read(8))[0]
        if not 1 <= size <= 4 << 20:
            raise ValueError('Bounded optimizer state header required')
        header = json.loads(stream.read(size))
    if (header['schema'] != 1 or header['kind'] != 'pcc.completed-optimizer-state.v1'
            or header['storage'] != 'le-f32' or header['model_sha256'] != expected[1]
            or header['completed_steps'] != 1 or header['config'] != config
            or header['dataset_sha256'] != checkpoint['dataset_sha256']
            or header['example_cursor'] != receipt['example_cursor']):
        raise ValueError('Original optimizer state binding differs')
    offsets, offset = {}, 16 + size
    for row in header['tensors']:
        elements = math.prod(row['shape'])
        if (row['optimizer'] != 'adamw' or not row['shape'] or any(type(n) is not int or n < 1 for n in row['shape'])
                or row['bytes'] != elements * 4 or row['name'] in offsets):
            raise ValueError('Optimizer tensor layout differs')
        offsets[row['name']] = (offset, elements)
        offset += row['bytes']
    expected_names = {name + suffix for name in before for suffix in ('.opt_m', '.opt_v')}
    if offsets.keys() != expected_names or offset != paths[2].stat().st_size:
        raise ValueError('Complete exact optimizer-state coverage required')
    reported_gradients = {g['name']: g['l2'] for g in receipt['steps'][0]['gradients']}
    if reported_gradients.keys() != before.keys():
        raise ValueError('Original gradient tensor coverage differs')
    rows = []
    for name, tensor in before.items():
        other = after[name]
        if tensor.shape != other.shape or tensor.type_name != 'F32' or other.type_name != 'F32':
            raise ValueError('Audit currently requires matching full F32 tensor shapes')
        n = tensor.elements
        if offsets[name + '.opt_m'][1] != n or offsets[name + '.opt_v'][1] != n:
            raise ValueError('Weight and optimizer-state element counts differ')
        x = np.memmap(paths[0], dtype='<f4', mode='r', offset=left.data_start + tensor.offset, shape=(n,))
        y = np.memmap(paths[1], dtype='<f4', mode='r', offset=right.data_start + other.offset, shape=(n,))
        m = np.memmap(paths[2], dtype='<f4', mode='r', offset=offsets[name + '.opt_m'][0], shape=(n,))
        delta2 = weight2 = gradient2 = dot = maximum = 0.0
        changed = violations = 0
        for start in range(0, n, chunk_elements):
            a = np.asarray(x[start:start + chunk_elements], dtype=np.float64)
            b = np.asarray(y[start:start + chunk_elements], dtype=np.float64)
            gradient = np.asarray(m[start:start + chunk_elements], dtype=np.float64) / (1 - beta)
            if not (np.isfinite(a).all() and np.isfinite(b).all() and np.isfinite(gradient).all()):
                raise ValueError('Nonfinite original weight, candidate or first moment')
            delta = b - a
            absolute = np.abs(delta)
            rounding = 2 * np.maximum(np.abs(np.spacing(a.astype(np.float32))), np.abs(np.spacing(b.astype(np.float32))))
            violations += int(np.count_nonzero(absolute > rate * 1.001 + rounding))
            changed += int(np.count_nonzero(delta))
            delta2 += float(np.dot(delta, delta))
            weight2 += float(np.dot(a, a))
            gradient2 += float(np.dot(gradient, gradient))
            dot += float(np.dot(gradient, delta))
            maximum = max(maximum, float(np.max(absolute, initial=0)))
        magnitude = math.sqrt(gradient2)
        expected_gradient = reported_gradients[name]
        rows.append(dict(name=name, elements=n, changed=changed, delta_l2=math.sqrt(delta2),
            original_l2=math.sqrt(weight2), maximum_absolute_delta=maximum,
            component_bound_violations=violations, recorded_gradient_l2=expected_gradient,
            moment_derived_gradient_l2=magnitude,
            gradient_l2_relative_error=abs(magnitude - expected_gradient) / max(abs(expected_gradient), 1e-30),
            recorded_gradient_dot_delta=dot))
        del x, y, m
    if [_cookie(p) for p in paths] != cookies:
        raise ValueError('Audit artifact changed during observation')
    return dict(schema=1, status='AUDIT_COMPLETE', scope='First fresh full AdamW step only; no acceptance or gradient-check claim',
        base_sha256=expected[0], candidate_sha256=expected[1], optimizer_state_sha256=expected[2],
        learning_rate=rate, tensor_count=len(rows), elements=sum(r['elements'] for r in rows),
        changed_elements=sum(r['changed'] for r in rows),
        component_bound_violations=sum(r['component_bound_violations'] for r in rows),
        recorded_gradient_dot_delta=sum(r['recorded_gradient_dot_delta'] for r in rows),
        maximum_gradient_l2_relative_error=max(r['gradient_l2_relative_error'] for r in rows),
        bound_allowance='0.1 percent plus two F32 spacings per component',
        tensors=rows, model_calls=0, training_updates=0, promoted=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job-record', type=Path, required=True)
    parser.add_argument('--base-model', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    job = json.loads(args.job_record.read_bytes())
    if job['path'] != '/cow/training/jobs' or job['state'] != 'complete':
        raise ValueError('Original completed native job record required')
    original = {k: v for k, v in job.items() if k != 'record_sha256'}
    checksum = hashlib.sha256(json.dumps(original, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode()).hexdigest()
    if job.get('record_sha256') != checksum:
        raise ValueError('Original job record checksum differs')
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError('Audit output already exists; inspect original report')
    result = audit_first_adamw_step(args.base_model, job['candidate'])
    fd = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, 'w') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps({k: v for k, v in result.items() if k != 'tensors'}, allow_nan=False))


if __name__ == '__main__':
    main()
