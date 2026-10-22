"""Reconstruction admission controls; no GPU/model quality claim."""
import copy
from types import SimpleNamespace
import unittest
from tools.wen_training_import import validate_wen_source


class WenImportTests(unittest.TestCase):
    def source(self):
        geometry={'block_count':32,'embedding_length':4096,'full_attention_interval':4,
            'attention.head_count':16,'attention.head_count_kv':4,'attention.key_length':256,
            'attention.value_length':256,'ssm.conv_kernel':4,'ssm.state_size':128,
            'ssm.group_count':16,'ssm.time_step_rank':32,'ssm.inner_size':4096,
            'feed_forward_length':12288,'context_length':262144,'rope.dimension_count':64,
            'rope.dimension_sections':[11,11,10,0],'rope.freq_base':10000000.,
            'attention.layer_norm_rms_epsilon':1e-6}
        metadata={'general.architecture':'wen1','tokenizer.ggml.model':'gpt2',
            'tokenizer.ggml.pre':'wen1','tokenizer.ggml.tokens':['a','b'],
            'tokenizer.ggml.merges':['a b'],'tokenizer.chat_template':'{{ messages }}'}
        metadata.update({'wen1.'+k:v for k,v in geometry.items()})
        tensors=[SimpleNamespace(name=name,shape=[4096,2],type_name='BF16')
            for name in ('token_embd.weight','output.weight')]
        return SimpleNamespace(metadata=metadata,tensors=tensors)

    def test_custom_vocabulary_and_template_are_bound_to_source_identity(self):
        source=self.source();first=validate_wen_source(source)
        self.assertEqual(first['vocab_size'],2)
        changed=copy.deepcopy(source);changed.metadata['tokenizer.ggml.tokens'][0]='changed'
        self.assertNotEqual(validate_wen_source(changed)['tokenizer_sha256'],first['tokenizer_sha256'])
        changed=copy.deepcopy(source);changed.metadata['tokenizer.chat_template']='other'
        self.assertNotEqual(validate_wen_source(changed)['tokenizer_sha256'],first['tokenizer_sha256'])

    def test_wrong_geometry_and_vocab_rows_reject(self):
        source=self.source();source.metadata['wen1.block_count']=31
        with self.assertRaises(ValueError):validate_wen_source(source)
        source=self.source();source.tensors[0].shape=[4096,3]
        with self.assertRaises(ValueError):validate_wen_source(source)

    def test_reconstruction_rejects_unqualified_quantization(self):
        source=self.source();source.tensors[0].type_name='Q4_K'
        with self.assertRaisesRegex(ValueError,'BF16/F32'):validate_wen_source(source)


if __name__=='__main__':unittest.main()
