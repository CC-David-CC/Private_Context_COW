import copy
import hashlib
import json
import math
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest

from tools.pcc_training_step_audit import audit_first_adamw_step


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def gguf(path, values):
    def text(value):
        raw = value.encode()
        return struct.pack('<Q', len(raw)) + raw
    header = struct.pack('<IIQQ', 0x46554747, 3, 2, 0)
    header += text('w') + struct.pack('<IQQIQ', 2, 2, 2, 0, 0)
    header += text('bias') + struct.pack('<IQIQ', 1, 2, 0, 32)
    header += bytes((-len(header)) % 32)
    path.write_bytes(header + struct.pack('<4f', *values[:4]) + bytes(16) + struct.pack('<2f', *values[4:]))


class StepAuditTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def fixture(self, *, flip=False, excessive=False, nonfinite=False):
        base, final, state = (self.root / n for n in ('base.gguf', 'candidate.gguf', 'state.bin'))
        original = [1., 2., 3., 4., .5, -.5]
        gradients = [1., -2., 3., -4., 2., -1.]
        delta = [(1 if g < 0 else -1) * .001 * (-1 if flip else 1) for g in gradients]
        if excessive:
            delta[0] = .1
        gguf(base, original)
        gguf(final, [a + d for a, d in zip(original, delta)])
        config = dict(optimizer='adamw', learning_rate=.001, beta1=.9,
                      weight_decay=0., aux_weight_decay=0.)
        layout, arrays = [], []
        for name, group, shape in [('w', gradients[:4], [2, 2]), ('bias', gradients[4:], [2, 1])]:
            for suffix, values in [('m', [.1 * g for g in group]), ('v', [.05 * g * g for g in group])]:
                if nonfinite and name == 'w' and suffix == 'm':
                    values[0] = float('nan')
                layout.append(dict(name=name + '.opt_' + suffix, shape=shape,
                                   optimizer='adamw', bytes=len(group) * 4))
                arrays.append(struct.pack('<%df' % len(values), *values))
        header = dict(schema=1, kind='pcc.completed-optimizer-state.v1', storage='le-f32',
                      model_sha256=sha(final), completed_steps=1, config=config,
                      dataset_sha256='a' * 64, example_cursor=2, tensors=layout)
        raw = json.dumps(header).encode()
        state.write_bytes(b'PCCOPT1\n' + struct.pack('<Q', len(raw)) + raw + b''.join(arrays))
        reported = [dict(name='w', l2=math.sqrt(sum(g*g for g in gradients[:4]))),
                    dict(name='bias', l2=math.sqrt(sum(g*g for g in gradients[4:])))]
        candidate = dict(mode='full', optimizer_state_policy='fresh', base_sha256=sha(base),
                         candidate_path=str(final), candidate_sha256=sha(final),
                         optimizer_state_path=str(state), optimizer_state_sha256=sha(state),
                         receipt=dict(example_cursor=2, steps=[dict(gradients=reported)],
                         optimizer_checkpoint=dict(config=config, completed_steps=1, dataset_sha256='a'*64)))
        return base, candidate

    def test_mathematical_bound_alignment_and_original_bytes_unchanged(self):
        base, candidate = self.fixture()
        originals = {p: p.read_bytes() for p in self.root.iterdir()}
        result = audit_first_adamw_step(base, candidate, chunk_elements=3)
        self.assertEqual(result['elements'], 6)
        self.assertEqual(result['changed_elements'], 6)
        self.assertEqual(result['component_bound_violations'], 0)
        # Independent scalar reference includes the fixture's F32 serialization.
        f32 = lambda value: struct.unpack('<f', struct.pack('<f', value))[0]
        x = [1., 2., 3., 4., .5, -.5]
        g = [1., -2., 3., -4., 2., -1.]
        expected = sum((f32(a + (.001 if gradient < 0 else -.001)) - a)
                       * f32(.1 * gradient) / (1 - .9) for a, gradient in zip(x, g))
        self.assertAlmostEqual(result['recorded_gradient_dot_delta'], expected, places=12)
        self.assertLess(abs(expected - (-.013)), 2e-6)
        self.assertLess(result['maximum_gradient_l2_relative_error'], 1e-6)
        self.assertEqual(result['training_updates'], 0)
        self.assertEqual(result['model_calls'], 0)
        self.assertFalse(result['promoted'])
        self.assertEqual({p: p.read_bytes() for p in originals}, originals)

    def test_positive_gradient_alignment_is_reported_without_acceptance(self):
        base, candidate = self.fixture(flip=True)
        result = audit_first_adamw_step(base, candidate)
        self.assertEqual(result['component_bound_violations'], 0)
        self.assertGreater(result['recorded_gradient_dot_delta'], 0)
        self.assertFalse(result['promoted'])

    def test_large_component_is_not_hidden_by_total_or_average(self):
        base, candidate = self.fixture(excessive=True)
        result = audit_first_adamw_step(base, candidate, chunk_elements=1)
        self.assertEqual(result['component_bound_violations'], 1)

    def test_artifact_corruption_is_refused(self):
        base, candidate = self.fixture()
        with base.open('ab') as stream:
            stream.write(b'changed')
        with self.assertRaisesRegex(ValueError, 'SHA-256'):
            audit_first_adamw_step(base, candidate)

    def test_nonfinite_first_moment_is_refused_even_with_matching_hash(self):
        base, candidate = self.fixture(nonfinite=True)
        with self.assertRaisesRegex(ValueError, 'Nonfinite'):
            audit_first_adamw_step(base, candidate)

    def test_owner_cli_refuses_changed_job_and_existing_output_before_audit(self):
        base, candidate = self.fixture()
        job = dict(path='/cow/training/jobs', state='complete', candidate=candidate, record_sha256='0'*64)
        record = self.root / 'job.json';record.write_text(json.dumps(job))
        output = self.root / 'report.json'
        command = [sys.executable, '-B', '-m', 'tools.pcc_training_step_audit',
                   '--job-record', str(record), '--base-model', str(base), '--output', str(output)]
        result = subprocess.run(command, capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('Original job record checksum differs', result.stderr)
        self.assertFalse(output.exists())
        job.pop('record_sha256')
        job['record_sha256'] = hashlib.sha256(json.dumps(job, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode()).hexdigest()
        record.write_text(json.dumps(job));output.write_bytes(b'original report')
        result = subprocess.run(command, capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('already exists', result.stderr)
        self.assertEqual(output.read_bytes(), b'original report')

    def test_bound_is_not_applied_to_other_optimizers_or_resumed_steps(self):
        base, candidate = self.fixture()
        for change in ('optimizer', 'step', 'policy', 'decay'):
            value = copy.deepcopy(candidate)
            if change == 'optimizer':
                value['receipt']['optimizer_checkpoint']['config']['optimizer'] = 'muon'
            elif change == 'step':
                value['receipt']['optimizer_checkpoint']['completed_steps'] = 2
            elif change == 'policy':
                value['optimizer_state_policy'] = 'resume'
            else:
                value['receipt']['optimizer_checkpoint']['config']['weight_decay'] = .1
            with self.subTest(change=change), self.assertRaisesRegex(ValueError, 'one fresh full AdamW'):
                audit_first_adamw_step(base, value)


if __name__ == '__main__':
    unittest.main()
