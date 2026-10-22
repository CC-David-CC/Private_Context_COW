"""GPU routing and process-lock controls; these are not multi-GPU inference evidence."""
import argparse
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from serve import cow_http
from tools import cow_server
from context_time_travel.qwen35_backend import device_name_for_worker
from tools.cow_server_setup import parse_gpus


class GPUSelectionTests(unittest.TestCase):
    def args(self):
        return SimpleNamespace(ram_gib=8,disk_gib=16,min_available_gib=4,
                               checkpoint_idle_seconds=None)

    def test_http_uses_profile_workers_instead_of_gpu_zero(self):
        profile={'execution':{'GPU':[[0,1],4]}}
        self.assertEqual(cow_http.pool_config_from_args(self.args(),profile).gpus,((0,1),4))

    def test_http_preserves_default_for_existing_callers(self):
        self.assertEqual(cow_http.pool_config_from_args(self.args()).gpus,(0,))

    def test_setup_parser_preserves_worker_and_group_meaning(self):
        self.assertEqual(parse_gpus('4'),[4])
        self.assertEqual(parse_gpus('4,6'),[4,6])
        self.assertEqual(parse_gpus('[[0,1],2]'),[(0,1),2])
        for value in ('4,4','[-1]','[true]','[[0,1],1]','[]'):
            with self.subTest(value=value),self.assertRaises(argparse.ArgumentTypeError):parse_gpus(value)

    def test_profile_pins_selected_device_and_retains_legacy_default(self):
        from experimental.branching_context.scripts import prepare_qwen35_profile as prepare
        from context_time_travel.qwen35_backend import _GEOMETRY
        metadata={'general.architecture':'qwen35','tokenizer.ggml.model':'gpt2',
                  'tokenizer.ggml.tokens':['<|endoftext|>','<|im_end|>'],
                  **{'qwen35.'+k:v for k,v in _GEOMETRY.items()}}
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);model=root/'model';model.write_bytes(b'configuration fixture')
            build=root/'build.json';build.write_text(json.dumps({'sha256':{},'device_binding':'cuda-single-device-v1'}))
            def command(args,**kwargs):
                return 'test-source' if args[0]=='git' else 'GPU fixture'
            with patch.object(prepare,'GGUFFile',return_value=SimpleNamespace(metadata=metadata)),patch.object(prepare.subprocess,'check_output',side_effect=command):
                default=prepare.make_profile(model,root,root/'default.json',128,16,2)
                selected=prepare.make_profile(model,root,root/'selected.json',128,16,2,gpus=[4])
                self.assertEqual(default['execution']['environment'],{'CUDA_VISIBLE_DEVICES':'0'})
                self.assertNotIn('device_binding',default['execution'])
                self.assertEqual(selected['execution']['GPU'],[4])
                self.assertEqual(selected['execution']['environment'],{'CUDA_DEVICE_ORDER':'PCI_BUS_ID','CUDA_VISIBLE_DEVICES':'4'})
                self.assertNotEqual(default['execution_profile_id'],selected['execution_profile_id'])
                self.assertEqual(default['model_contract_id'],selected['model_contract_id'])
                build.write_text(json.dumps({'sha256':{}}))
                with self.assertRaisesRegex(ValueError,'Rebuild'):
                    prepare.make_profile(model,root,root/'bad.json',128,16,2,gpus=[4])

    def test_dense_mapping_respects_physical_order(self):
        profile={'execution':{'GPU':[4,2],'environment':{
            'CUDA_DEVICE_ORDER':'PCI_BUS_ID','CUDA_VISIBLE_DEVICES':'4,2'},
            'device_binding':'cuda-single-device-v1'}}
        self.assertEqual(device_name_for_worker(profile,4),'CUDA0')
        self.assertEqual(device_name_for_worker(profile,2),'CUDA1')
        with self.assertRaisesRegex(ValueError,'selected'):device_name_for_worker(profile,0)

    def test_dense_group_and_ambiguous_environment_rejected(self):
        profile={'execution':{'GPU':[[0,1],2],'environment':{'CUDA_VISIBLE_DEVICES':'0,1,2'},
                              'device_binding':'cuda-single-device-v1'}}
        with self.assertRaisesRegex(ValueError,'group'):device_name_for_worker(profile,(0,1))
        profile['execution']['GPU']=[4,2]
        with self.assertRaisesRegex(ValueError,'visibility'):device_name_for_worker(profile,4)

    def test_legacy_gpu_zero_can_use_legacy_native_library(self):
        self.assertIsNone(device_name_for_worker({'execution':{'GPU':[0]}},0))
        with self.assertRaisesRegex(ValueError,'binding'):
            device_name_for_worker({'execution':{'GPU':[4]}},4)

    def test_overlapping_launchers_conflict_but_other_gpu_is_available(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            with cow_server.gpu_locks(root,[[1,4],2]):
                with self.assertRaisesRegex(RuntimeError,'GPU 4'):
                    with cow_server.gpu_locks(root,[4]):self.fail('overlap admitted')
                with cow_server.gpu_locks(root,[3]):pass
            with cow_server.gpu_locks(root,[1,2,4]):pass

    def test_failed_partial_acquisition_releases_earlier_locks(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            with cow_server.gpu_locks(root,[4]):
                with self.assertRaises(RuntimeError):
                    with cow_server.gpu_locks(root,[1,4]):pass
                with cow_server.gpu_locks(root,[1]):pass


if __name__=='__main__':unittest.main()
