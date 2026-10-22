"""A precision override must be admitted before allocating native state."""
import unittest
from context_time_travel.qwen35_backend import validate_math_profile
from context_time_travel.store import IntegrityError


class MathProfileTests(unittest.TestCase):
    def precise(self):
        return dict(native_build=dict(f32_activations_patch_sha256='a'*64),
            execution=dict(f32_activations=True,cuda_cublas_compute_type='f32',
                nvidia_tf32_override='0'))

    def environment(self):
        return dict(GGML_CUDA_F32_ACTIVATIONS='1',
            GGML_CUDA_CUBLAS_COMPUTE_TYPE='f32',NVIDIA_TF32_OVERRIDE='0')

    def test_legacy_default_and_matching_precision_are_admitted(self):
        validate_math_profile(dict(execution={}),{})
        validate_math_profile(self.precise(),self.environment())

    def test_unbound_override_and_boolean_alias_reject(self):
        with self.assertRaises(IntegrityError):
            validate_math_profile(dict(execution={}),self.environment())
        p=self.precise();p['execution']['f32_activations']=1
        with self.assertRaises(IntegrityError):validate_math_profile(p,self.environment())

    def test_old_build_cannot_claim_precision_patch(self):
        p=self.precise();p['native_build']={}
        with self.assertRaises(IntegrityError):validate_math_profile(p,self.environment())

    def test_process_environment_cannot_silently_change_math(self):
        for key in self.environment():
            env=self.environment();env[key]='wrong'
            with self.subTest(key=key),self.assertRaises(IntegrityError):
                validate_math_profile(self.precise(),env)

    def test_profile_cannot_claim_different_compute_mode(self):
        for key in ('cuda_cublas_compute_type','nvidia_tf32_override'):
            p=self.precise();p['execution'][key]='wrong'
            with self.subTest(key=key),self.assertRaises(IntegrityError):
                validate_math_profile(p,self.environment())

    def test_fp32_attention_requires_matching_cache_format_and_build(self):
        p=self.precise();env=self.environment();env['COW_QWEN35_F32_ATTENTION']='1'
        p['execution'].update(f32_attention=True,kv_dtype='FP32',flash_attention=False)
        p['native_build']['attention_precision_binding']='fp32-attention-v1'
        validate_math_profile(p,env)
        for key,value in [('kv_dtype','FP16'),('flash_attention',True)]:
            q=self.precise();q['native_build']=p['native_build'];q['execution']=dict(p['execution'],**{key:value})
            with self.subTest(key=key),self.assertRaises(IntegrityError):validate_math_profile(q,env)
        p['native_build'].pop('attention_precision_binding')
        with self.assertRaises(IntegrityError):validate_math_profile(p,env)

    def test_unbound_fp32_attention_rejects_before_native_allocation(self):
        with self.assertRaises(IntegrityError):
            validate_math_profile(dict(execution={}),{'COW_QWEN35_F32_ATTENTION':'1'})

    def test_launcher_fingerprints_fp32_cache_separately(self):
        import importlib.util,json,tempfile
        from pathlib import Path
        from unittest.mock import patch
        from test_qwen35_backend import Qwen35AdmissionTests
        script=Path(__file__).resolve().parents[2]/'scripts/prepare_qwen35_profile.py'
        spec=importlib.util.spec_from_file_location('math_profile_launcher',script)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        metadata=Qwen35AdmissionTests().metadata()
        metadata['tokenizer.ggml.tokens']=['a','<|endoftext|>','<|im_end|>']
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);model=root/'model.gguf';model.write_bytes(b'fixture')
            build=dict(sha256={},f32_activations_patch_sha256='a'*64,
                attention_precision_binding='fp32-attention-v1',device_binding='cuda-single-device-v1')
            (root/'build.json').write_text(json.dumps(build))
            with patch.object(module,'GGUFFile') as reader,patch.object(module.subprocess,'check_output',return_value='fixture'):
                reader.return_value.metadata=metadata
                native=module.make_profile(model,root,root/'native.json',128,8,1,gpus=[0])
                reference=module.make_profile(model,root,root/'reference.json',128,8,1,gpus=[0],math_profile='f32-reference')
                self.assertEqual(reference['execution']['kv_dtype'],'FP32')
                self.assertFalse(reference['execution']['flash_attention'])
                self.assertNotEqual(native['model_contract_id'],reference['model_contract_id'])
                self.assertNotEqual(native['execution_profile_id'],reference['execution_profile_id'])
                validate_math_profile(reference,reference['execution']['environment'])
                build.pop('attention_precision_binding');(root/'build.json').write_text(json.dumps(build))
                with self.assertRaises(ValueError):
                    module.make_profile(model,root,root/'invalid.json',128,8,1,gpus=[0],math_profile='f32-reference')
