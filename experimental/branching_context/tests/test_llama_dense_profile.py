import copy
import unittest
from context_time_travel.llama_dense_backend import validate_geometry,terminal_tokens
from context_time_travel.qwen35_backend import validate_geometry as validate_hybrid
from context_time_travel.real import supports_consumed_selection
from context_time_travel.capabilities import describe_dense, CapabilityError
from context_time_travel.store import InvalidOperation

GEOMETRY={'general.architecture':'llama','tokenizer.ggml.model':'gpt2',
'llama.block_count':42,'llama.embedding_length':2048,'llama.feed_forward_length':6144,
'llama.context_length':131072,'llama.attention.head_count':16,'llama.attention.head_count_kv':2,
'llama.attention.key_length':128,'llama.attention.value_length':128,'llama.rope.dimension_count':128,
'llama.rope.freq_base':5000000.0,'llama.attention.layer_norm_rms_epsilon':1e-6}

def profile():
 return dict(backend='llama-dense-llama.cpp',model_contract_id='a'*64,execution_profile_id='b'*64,
  execution=dict(capacity=131072,state_format='llama-dense-frontier-v1',resume_contract='identical-native-batches-v1',
   generated_operator='llama-forward-consumed-v1',selected_token_protocol='consumed-selected-receipt-v1',sampling_protocol='full-vocab-min-p-u53-v2'))

class LlamaDenseProfileTests(unittest.TestCase):
 def test_actual_minicpm_geometry_is_all_attention(self):
  self.assertEqual(validate_geometry(GEOMETRY),dict(attention_layers=42,recurrent_layers=0))
 def test_wrong_tensor_geometry_and_tokenizer_are_rejected(self):
  for key,value in [('llama.block_count',41),('llama.attention.head_count_kv',4),('tokenizer.ggml.model','llama'),('llama.rope.freq_base',10000.0),('llama.attention.layer_norm_rms_epsilon',1e-5)]:
   md=dict(GEOMETRY);md[key]=value
   with self.subTest(key=key),self.assertRaises(InvalidOperation):validate_geometry(md)
 def test_hybrid_and_llama_profiles_are_not_aliases(self):
  with self.assertRaises(InvalidOperation):validate_hybrid(GEOMETRY)
  md=dict(GEOMETRY);md['general.architecture']='wen1'
  with self.assertRaises(InvalidOperation):validate_geometry(md)
 def test_consumed_selection_binds_model_state_pair(self):
  p=profile();self.assertTrue(supports_consumed_selection(p))
  p['execution']['state_format']='qwen35-hybrid-frontier-v1';self.assertFalse(supports_consumed_selection(p))
 def test_capabilities_advertise_semantics_without_physical_sharing(self):
  cap=describe_dense(profile());self.assertEqual(cap.inputs,('text',));self.assertFalse(cap.physical_prefix_sharing)
 def test_mislabeled_hybrid_state_is_not_a_llama_capability(self):
  p=profile();p['execution']['state_format']='qwen35-hybrid-frontier-v1'
  with self.assertRaises(CapabilityError):describe_dense(p)

 def test_declared_model_eos_alone_does_not_cover_chat_turns(self):
  md={'tokenizer.ggml.tokens':['<s>','</s>','<|im_start|>','<|im_end|>'],'tokenizer.ggml.eos_token_id':1}
  self.assertEqual(terminal_tokens(md),[1,3])
  md['tokenizer.ggml.tokens']=['<s>','</s>']
  with self.assertRaises(InvalidOperation):terminal_tokens(md)
