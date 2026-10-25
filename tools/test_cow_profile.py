from pathlib import Path
from copy import deepcopy
import os,sys,unittest
from unittest.mock import patch
from tools.pcc_cow_profile import bind_dense_profile

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experimental/branching_context/src'))
from context_time_travel.real import NativeBackend
from context_time_travel.store import IntegrityError


class CowProfileTests(unittest.TestCase):
    def template(self):
        return dict(backend='llama-dense-llama.cpp',native_build={'sha256':{}},hardware={'gpu_uuid':'GPU-original'},
            execution=dict(state_format='llama-dense-frontier-v1',kv_dtype='FP32',device_binding='cuda-single-device-v1',
                environment={'CUDA_VISIBLE_DEVICES':'0','GGML_CUDA_CUBLAS_COMPUTE_TYPE':'f32','NVIDIA_TF32_OVERRIDE':'0'},
                cuda_cublas_compute_type='f32',nvidia_tf32_override='0',GPU=[0],capacity=512))

    def test_original_native_validator_accepts_rebinding_without_loading_a_model(self):
        template=self.template();original=deepcopy(template)
        for gpu,weights in [(0,'a'*64),(7,'b'*64)]:
            value=bind_dense_profile(template,model_path=Path('/owned/model.gguf'),model_sha256=weights,gpu_index=gpu,gpu_uuid='GPU-owned-'+str(gpu))
            probe=NativeBackend.__new__(NativeBackend);probe.profile=value
            with patch.dict(os.environ,value['execution']['environment']):probe.validate_execution_profile()
            self.assertFalse(hasattr(probe,'lib'))
            self.assertFalse(hasattr(probe,'model'))
            value['execution']['capacity']=513
            with patch.dict(os.environ,value['execution']['environment']):
                with self.assertRaises(IntegrityError):probe.validate_execution_profile()
        self.assertEqual(template,original)

    def test_rejects_ambiguous_binding_and_other_execution_contracts(self):
        args=dict(model_path=Path('/owned/model.gguf'),model_sha256='a'*64,gpu_index=0,gpu_uuid='GPU-owned')
        for invalid in [dict(gpu_index=True),dict(gpu_index=-1),dict(model_path='relative.gguf'),
                        dict(model_sha256='z'*64),dict(gpu_uuid='GPU- bad')]:
            with self.assertRaises(ValueError):bind_dense_profile(self.template(),**dict(args,**invalid))
        template=self.template();template['execution']['kv_dtype']='FP16'
        with self.assertRaises(ValueError):bind_dense_profile(template,**args)

if __name__=='__main__':unittest.main()
