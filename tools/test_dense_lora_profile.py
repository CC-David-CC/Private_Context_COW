"""CPU integrity tests; these do not certify numerical native LoRA behavior."""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from tools import pcc_cow_profile as binding
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'experimental/branching_context/src'))
from context_time_travel import llama_dense_backend as dense
from context_time_travel.real import NativeBackend


class DenseLoraProfileTests(unittest.TestCase):
    def base(self):
        return binding.bind_dense_profile(dict(backend='llama-dense-llama.cpp',
            native_build=dict(sha256={}, lora_binding='llama-adapter-per-lease-v1'),
            hardware=dict(gpu_uuid='GPU-test'),
            execution=dict(state_format='llama-dense-frontier-v1', kv_dtype='FP32',
                device_binding='cuda-single-device-v1', GPU=[0],
                cuda_cublas_compute_type='f32', nvidia_tf32_override='0',
                environment=dict(CUDA_VISIBLE_DEVICES='0',
                    GGML_CUDA_CUBLAS_COMPUTE_TYPE='f32', NVIDIA_TF32_OVERRIDE='0'))),
            model_path='/owned/base.gguf', model_sha256='a'*64, gpu_index=0, gpu_uuid='GPU-test')

    def bind(self, base, path, **kwargs):
        return binding.bind_dense_adapter(base, adapter_path=path,
            adapter_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), **kwargs)

    def test_adapter_versions_and_scale_cannot_share_context_identities(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp).resolve()/'adapter.gguf'; path.write_bytes(b'first hat')
            base=self.base(); original=deepcopy(base)
            first=self.bind(base,path)
            scaled=self.bind(base,path,scale=.5)
            path.write_bytes(b'second hat')
            second=self.bind(base,path)
            self.assertEqual(base,original)
            for key in ('model_contract_id','execution_profile_id'):
                self.assertEqual(len({base[key],first[key],scaled[key],second[key]}),4)
            self.assertEqual({x['model_weights_sha256'] for x in (base,first,scaled,second)},{'a'*64})
            self.assertIsNone(dense.validate_dense_adapter(base))
            self.assertEqual(dense.validate_dense_adapter(second)['sha256'],second['adapter']['sha256'])
            probe=NativeBackend.__new__(NativeBackend);probe.profile=second
            with patch.dict(os.environ,second['execution']['environment']):probe.validate_execution_profile()

    def test_same_adapter_bytes_at_another_path_keep_semantic_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();left=root/'left.gguf';right=root/'right.gguf'
            left.write_bytes(b'same immutable version');right.write_bytes(left.read_bytes())
            a,b=self.bind(self.base(),left),self.bind(self.base(),right)
            self.assertEqual(a['model_contract_id'],b['model_contract_id'])
            self.assertEqual(a['execution_profile_id'],b['execution_profile_id'])
            self.assertNotEqual(a['adapter']['path'],b['adapter']['path'])

    def test_base_rebinding_never_silently_inherits_an_old_hat(self):
        base=self.base();base['adapter']=dict(path='/owned/old.gguf',sha256='b'*64,scale=1.)
        with self.assertRaises(ValueError):
            binding.bind_dense_profile(base,model_path='/owned/other.gguf',model_sha256='c'*64,
                gpu_index=0,gpu_uuid='GPU-test')

    def test_changed_file_or_unbound_execution_is_rejected_before_loading(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp).resolve()/'adapter.gguf';path.write_bytes(b'owned version')
            good=self.bind(self.base(),path)
            for mutate in (lambda p:p['execution']['adapter'].update(scale=.5),
                           lambda p:p.update(model_weights_sha256='b'*64),
                           lambda p:p['native_build'].update(lora_binding='unqualified')):
                bad=deepcopy(good);mutate(bad)
                with self.assertRaises(ValueError):dense.validate_dense_adapter(bad)
            # Even an internally consistent contract for another base is invalid.
            bad=deepcopy(good);bad['model_contract']['weights_sha256']='b'*64
            bad['model_contract_id']=hashlib.sha256(json.dumps(bad['model_contract'],sort_keys=True).encode()).hexdigest()
            with self.assertRaises(ValueError):dense.validate_dense_adapter(bad)
            path.write_bytes(b'mutated version')
            with self.assertRaises(ValueError):dense.validate_dense_adapter(good)

    def test_invalid_scales_or_unqualified_build_cannot_be_bound(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp).resolve()/'adapter.gguf';path.write_bytes(b'version')
            for scale in (True,0,-1,17,float('nan'),float('inf')):
                with self.subTest(scale=scale),self.assertRaises(ValueError):self.bind(self.base(),path,scale=scale)
            base=self.base();base['native_build']['lora_binding']='unqualified'
            with self.assertRaises(ValueError):self.bind(base,path)

    def test_rejects_symlink_and_wrong_file_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp).resolve()/'adapter.gguf';path.write_bytes(b'version')
            link=path.with_name('alias.gguf');link.symlink_to(path)
            with self.assertRaises(ValueError):self.bind(self.base(),link)
            with self.assertRaises(ValueError):
                binding.bind_dense_adapter(self.base(),adapter_path=path,adapter_sha256='0'*64)

    def test_evaluation_close_detects_adapter_mutation_after_loading(self):
        from tools.pcc_cow_evaluation import CowEvaluationSession
        from types import SimpleNamespace
        from unittest.mock import Mock
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();path=root/'adapter.gguf';path.write_bytes(b'first')
            base=root/'base.gguf';base.write_bytes(b'base')
            profile=self.base();profile.update(model_path=str(base),
                model_weights_sha256=hashlib.sha256(base.read_bytes()).hexdigest())
            profile=self.bind(profile,path)
            session=CowEvaluationSession.__new__(CowEvaluationSession)
            session.closed=False;session.profile=profile;session.runtime_root=root
            session.model=SimpleNamespace(evict_all=Mock());session.backend=SimpleNamespace(close=Mock())
            path.write_bytes(b'mutated')
            with patch('tools.pcc_cow_evaluation.verify_sources'):
                with self.assertRaises(ValueError):session.close()
            self.assertTrue(session.closed)
            session.model.evict_all.assert_called_once()
            session.backend.close.assert_called_once()


if __name__=='__main__':unittest.main()
