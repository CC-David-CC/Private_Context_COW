"""CPU admission tests only; real HIP execution requires the native campaign."""
import json,unittest
from unittest.mock import patch
from tools.cow_qwen_hardware import execution_hardware

def build(*names):return {'sha256':{'/pinned/'+name:'a'*64 for name in names}}

class QwenHardwareTests(unittest.TestCase):
 def test_cuda_keeps_existing_identity_and_environment(self):
  with patch('tools.cow_qwen_hardware.subprocess.check_output',return_value='GPU, uuid, driver\n'),patch('tools.cow_qwen_hardware.platform.node',return_value='host'):
   hw,env=execution_hardware(build('libggml-cuda.so'))
  self.assertEqual(hw,{'host':'host','gpu':'GPU, uuid, driver'})
  self.assertEqual(env,{'CUDA_VISIBLE_DEVICES':'0'})
 def test_hip_records_actual_vendor_device_driver_and_runtime(self):
  rows={'card0':{'Card Vendor':'Advanced Micro Devices, Inc. [AMD/ATI]','Unique ID':'0x123','GFX Version':'gfx1100'},'system':{'Driver version':'kernel'}}
  with patch('tools.cow_qwen_hardware.subprocess.check_output',side_effect=[json.dumps(rows),'7.15\n']),patch('tools.cow_qwen_hardware.platform.node',return_value='amd'):
   hw,env=execution_hardware(build('libggml-hip.so.0.24.0'))
  self.assertEqual(json.loads(hw['gpu']),rows);self.assertEqual(hw['hip_runtime'],'7.15');self.assertEqual(hw['gpu_backend'],'hip')
  self.assertEqual(env,{'HIP_VISIBLE_DEVICES':'0','ROCR_VISIBLE_DEVICES':'0'})
 def test_cpu_only_build_is_not_admitted_as_gpu(self):
  with self.assertRaisesRegex(ValueError,'one pinned GPU backend'):execution_hardware(build('libggml-cpu.so'))
 def test_ambiguous_gpu_build_is_not_silently_selected(self):
  with self.assertRaisesRegex(ValueError,'one pinned GPU backend'):execution_hardware(build('libggml-cuda.so','libggml-hip.so'))
 def test_malformed_hip_inventory_is_rejected(self):
  with patch('tools.cow_qwen_hardware.subprocess.check_output',return_value='not-json'):
   with self.assertRaisesRegex(ValueError,'HIP inventory'):execution_hardware(build('libggml-hip.so'))
 def test_empty_or_foreign_device_is_rejected(self):
  for rows in [{},{'card0':{'Card Vendor':'other','Unique ID':'x'}}]:
   with self.subTest(rows=rows),patch('tools.cow_qwen_hardware.subprocess.check_output',return_value=json.dumps(rows)):
    with self.assertRaises(ValueError):execution_hardware(build('libggml-hip.so'))
 def test_missing_unique_id_or_driver_is_rejected(self):
  for rows in [{'card0':{'Card Vendor':'AMD'}},{'card0':{'Card Vendor':'AMD','Unique ID':'x'}}]:
   with self.subTest(rows=rows),patch('tools.cow_qwen_hardware.subprocess.check_output',return_value=json.dumps(rows)):
    with self.assertRaises(ValueError):execution_hardware(build('libggml-hip.so'))
 def test_unqualified_multi_amd_device_inventory_is_explicit(self):
  rows={'card0':{},'card1':{}}
  with patch('tools.cow_qwen_hardware.subprocess.check_output',return_value=json.dumps(rows)):
   with self.assertRaisesRegex(ValueError,'one AMD GPU'):execution_hardware(build('libggml-hip.so'))

 def test_wrong_inventory_field_types_are_rejected_as_admission_errors(self):
  for rows in [{'card0':[], 'system':{}},{'card0':{},'system':'bad'}]:
   with self.subTest(rows=rows),patch('tools.cow_qwen_hardware.subprocess.check_output',return_value=json.dumps(rows)):
    with self.assertRaises(ValueError):execution_hardware(build('libggml-hip.so'))
