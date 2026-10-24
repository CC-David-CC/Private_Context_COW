"""Admission and immutable-version tests; these do not qualify GPU gradients."""
from pathlib import Path
import hashlib
import importlib.util
import json
import tempfile
import unittest

MODULE = Path(__file__).with_name('pcc_native_training.py')

def load():
    spec = importlib.util.spec_from_file_location('pcc_native_training', MODULE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

class NativeTrainingContracts(unittest.TestCase):
    def setUp(self):
        self.m = load()

    def example(self):
        return dict(id='owned-repair-1', tokens=[3,4,5], targets=[4,5,6],
                    loss_mask=[False,True,True], verified=True,
                    provenance_sha256='a'*64)

    def test_allocator_admission_is_explicit_and_dense_only(self):
        for sequence in self.m.SEQUENCE_PROFILES:
            self.assertEqual(self.m.validate_memory_profile('conservative-v1',sequence),'conservative-v1')
        self.assertEqual(self.m.validate_memory_profile('allocator-v1','dense-short-v1'),'allocator-v1')
        for memory,sequence in [('unknown','dense-short-v1'),('allocator-v1','short-v1'),('allocator-v1','cpu-long-v1')]:
            with self.assertRaises(ValueError):self.m.validate_memory_profile(memory,sequence)

    def test_response_mask_required(self):
        v=self.example();v['loss_mask']=[False]*3
        with self.assertRaises(ValueError):self.m.validate_examples([v])

    def test_unverified_example_rejected(self):
        v=self.example();v['verified']=False
        with self.assertRaises(ValueError):self.m.validate_examples([v])

    def test_duplicate_examples_rejected(self):
        with self.assertRaises(ValueError):self.m.validate_examples([self.example(),self.example()])

    def test_labels_must_match_next_input_on_context_positions(self):
        v=self.example();v['targets'][0]=9
        with self.assertRaises(ValueError):self.m.validate_examples([v])

    def test_examples_remain_immutable(self):
        v=self.example();before=json.dumps(v,sort_keys=True)
        got=self.m.validate_examples([v]);got[0]['tokens'][0]=999
        self.assertEqual(json.dumps(v,sort_keys=True),before)

    def test_partial_batch_keeps_equal_example_weight(self):
        self.assertEqual(self.m.batch_weights(5,2),[.5,.5,.5,.5,1.])
        with self.assertRaises(ValueError):self.m.batch_weights(0,2)

    def test_ordinary_backprop_is_default_on_every_device_count(self):
        for devices in (1,2,8):
            self.assertEqual(self.m.validate_backprop_mode(None,devices=devices),'standard')
        with self.assertRaises(ValueError):
            self.m.validate_backprop_mode('layer_checkpointed',devices=2)
        self.assertEqual(self.m.validate_backprop_mode('layer_checkpointed',devices=2,experimental=True),'layer_checkpointed')

    def test_promotion_does_not_follow_training_loss(self):
        report=dict(training_loss_decreased=True,repairs=True,held_out=False,preservation=True)
        self.assertFalse(self.m.promotion_eligible(report))
        report['held_out']=True
        self.assertTrue(self.m.promotion_eligible(report))

    def test_device_capacity_cannot_be_counted_twice(self):
        for names in (['CUDA0','CUDA0'],['CUDA0','CUDA00']):
            with self.assertRaises(ValueError):self.m.validate_devices(names)
        self.assertEqual(self.m.validate_devices(['CUDA0','CUDA2']),['CUDA0','CUDA2'])

    def test_mixed_cpu_gpu_placement_is_not_silent_offload(self):
        with self.assertRaises(ValueError):self.m.validate_devices(['CUDA0','CPU'])
        self.assertEqual(self.m.validate_devices(['CPU']),['CPU'])

    def test_candidate_is_content_addressed_and_never_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);model=root/'base.gguf';model.write_bytes(b'original')
            output=root/'candidate.gguf';output.write_bytes(b'new weights')
            result=self.m.finalize_candidate(root/'versions',model,output,'full',{'native':True})
            self.assertEqual(model.read_bytes(),b'original')
            self.assertEqual(result['base_sha256'],hashlib.sha256(b'original').hexdigest())
            self.assertFalse(result['promoted'])
            with self.assertRaises(FileExistsError):
                self.m.finalize_candidate(root/'versions',model,output,'full',{'native':True})

if __name__=='__main__':unittest.main()
