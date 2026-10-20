"""Audit executed real-GPU batches; partition failure is a failing test, not success."""
import array
import hashlib
import json
import os
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
from context_time_travel.numerics import assert_logits_close


class BatchedPreflightTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root=Path(os.environ['CTT_BATCH_RUN'])
        cls.preflight=json.loads((cls.root/'preflight/result.json').read_text())
        cls.layers=json.loads((cls.root/'input-layers/result.json').read_text())

    def test_bp_001_actual_batched_work(self):
        self.assertEqual(len(self.preflight['observations']),3)
        for observation in self.preflight['observations']:
            work=observation['work']
            self.assertEqual(work['continuation_operator'],'batched-prefill')
            self.assertEqual(work['total_prefill_tokens'],work['prompt_consumed']+1)
            self.assertEqual(work['consumed'],work['total_prefill_tokens'])
            self.assertEqual(work['decode_tokens'],1)  # output count, consumed by Prefill
            ids=Path(observation['command'][observation['command'].index('--tokens-file')+1]).read_text().split(',')
            self.assertEqual(work['prefill_tokens'],len(ids))
            frontier=self.root/'preflight'/observation['frontier']
            self.assertEqual(hashlib.sha256(frontier.read_bytes()).hexdigest(),observation['logits_sha256'])

    def test_bp_002_identical_projection_inputs_and_unchanged_outputs(self):
        self.assertTrue(self.layers['instrumentation_preserved_frontiers'])
        images=self.layers['projection_input_images']
        self.assertTrue(images['bit_identical'])
        a,b=[(self.root/'input-layers'/(str(i)+'.trace.input-f16-bf16')).read_bytes() for i in range(2)]
        self.assertEqual(a,b)
        self.assertEqual(len(a),images['bytes_per_trace'])
        for i,data in enumerate((a,b)):
            self.assertEqual(hashlib.sha256(data).hexdigest(),images['sha256'][str(i)+'.trace.input-f16-bf16'])

    def test_bp_003_batched_partition_invariance(self):
        folder=self.root/'preflight'
        references=[(folder/('batch-preflight-oracle-'+str(i)+'.f32')).read_bytes() for i in (1,2)]
        self.assertEqual(references[0],references[1])
        actual=array.array('f');actual.frombytes((folder/'batch-preflight-oracle-0.f32.decode.f32').read_bytes())
        expected=array.array('f');expected.frombytes(references[0])
        if sys.byteorder!='little':actual.byteswap();expected.byteswap()
        self.assertEqual(len(actual),248320);self.assertEqual(len(expected),248320)
        assert_logits_close(tuple(actual),tuple(expected),atol=self.preflight['atol'],rtol=self.preflight['rtol'])


if __name__=='__main__':unittest.main(verbosity=2)
