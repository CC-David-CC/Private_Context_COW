"""Adapter bytes and scale are model/cache identity, never a mutable label."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from context_time_travel.qwen35_lora import validate_adapter


class LoraProfileTests(unittest.TestCase):
    def profile(self,path,scale=1.):
        identity=dict(sha256=hashlib.sha256(path.read_bytes()).hexdigest(),scale=scale)
        contract=dict(weights_sha256='a'*64,adapters=[identity])
        return dict(adapter=dict(path=str(path),**identity),model_contract=contract,
            model_contract_id=hashlib.sha256(json.dumps(contract,sort_keys=True).encode()).hexdigest(),
            execution=dict(adapter=identity),native_build=dict(lora_binding='llama-adapter-per-lease-v1'))

    def test_adapter_identity_binds_exact_bytes_scale_and_model_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'adapter.gguf';path.write_bytes(b'first-adapter')
            first=self.profile(path)
            self.assertEqual(validate_adapter(first)['path'],str(path))
            self.assertNotEqual(first['model_contract_id'],self.profile(path,.5)['model_contract_id'])
            path.write_bytes(b'new-adapter')
            self.assertNotEqual(first['model_contract_id'],self.profile(path)['model_contract_id'])
            with self.assertRaisesRegex(ValueError,'SHA256'):validate_adapter(first)

    def test_identity_and_execution_cannot_disagree(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'adapter.gguf';path.write_bytes(b'adapter')
            profile=self.profile(path);profile['execution']['adapter']['scale']=2.
            with self.assertRaises(ValueError):validate_adapter(profile)

    def test_unbound_adapter_and_invalid_scale_reject(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'adapter.gguf';path.write_bytes(b'adapter')
            for scale in (0.,float('nan'),float('inf'),True,-1.):
                with self.assertRaises(ValueError):validate_adapter(self.profile(path,scale))
            profile=self.profile(path);profile['model_contract_id']='0'*64
            with self.assertRaises(ValueError):validate_adapter(profile)
            profile=self.profile(path);del profile['adapter']
            with self.assertRaises(ValueError):validate_adapter(profile)

    def test_no_adapter_keeps_the_original_execution_path(self):
        self.assertIsNone(validate_adapter(dict(execution={})))


if __name__=='__main__':unittest.main()
