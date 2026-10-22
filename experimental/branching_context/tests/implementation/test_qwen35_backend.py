import unittest
from context_time_travel.qwen35_backend import validate_geometry
from context_time_travel.store import InvalidOperation

class Qwen35AdmissionTests(unittest.TestCase):
    def metadata(self):
        return {'general.architecture':'qwen35','tokenizer.ggml.model':'gpt2',
                'qwen35.block_count':32,'qwen35.embedding_length':4096,
                'qwen35.full_attention_interval':4,'qwen35.attention.head_count':16,
                'qwen35.attention.head_count_kv':4,'qwen35.attention.key_length':256,
                'qwen35.attention.value_length':256,'qwen35.ssm.conv_kernel':4,
                'qwen35.ssm.state_size':128,'qwen35.ssm.group_count':16,
                'qwen35.ssm.time_step_rank':32,'qwen35.ssm.inner_size':4096}
    def test_actual_9b_geometry(self):
        self.assertEqual(validate_geometry(self.metadata())['recurrent_layers'],24)
    def test_wen_geometry_uses_its_own_metadata_prefix(self):
        m={key.replace('qwen35.','wen1.'):value for key,value in self.metadata().items()}
        m['general.architecture']='wen1'
        self.assertEqual(validate_geometry(m)['recurrent_layers'],24)
        m['wen1.ssm.state_size']=64
        with self.assertRaises(InvalidOperation):validate_geometry(m)
    def test_foreign_architecture_rejected(self):
        m=self.metadata();m['general.architecture']='qwen4exp'
        with self.assertRaises(InvalidOperation):validate_geometry(m)
    def test_missing_recurrent_state_rejected(self):
        m=self.metadata();del m['qwen35.ssm.conv_kernel']
        with self.assertRaises(InvalidOperation):validate_geometry(m)
    def test_other_geometry_not_silently_admitted(self):
        m=self.metadata();m['qwen35.attention.head_count_kv']=8
        with self.assertRaises(InvalidOperation):validate_geometry(m)

class Qwen35LeaseBoundaryTests(unittest.TestCase):
    def test_same_address_stale_handle_is_rejected(self):
        import threading
        from context_time_travel.qwen35_backend import Qwen35Backend,_QwenHandle
        b=object.__new__(Qwen35Backend);live=_QwenHandle(17);stale=_QwenHandle(17)
        b._active=live;b._owner=threading.get_ident();b._check(live)
        with self.assertRaises(InvalidOperation):b._check(stale)
    def test_foreign_exit_leaves_owner_release_available(self):
        import threading
        from concurrent.futures import ThreadPoolExecutor
        from context_time_travel.qwen35_backend import _QwenLease
        class Owner:
            def __init__(self):self.thread=threading.get_ident();self.releases=0
            def _acquire(self):return object()
            def _release(self,p):
                if threading.get_ident()!=self.thread:raise InvalidOperation('foreign')
                self.releases+=1
        owner=Owner();manager=_QwenLease(owner);handle=manager.__enter__()
        with ThreadPoolExecutor(1) as pool:
            future=pool.submit(manager.__exit__,None,None,None)
            with self.assertRaises(InvalidOperation):future.result()
        self.assertIs(manager.pointer,handle);self.assertEqual(owner.releases,0)
        manager.__exit__(None,None,None);self.assertEqual(owner.releases,1)
    def test_consumed_contract_does_not_relabel_flash(self):
        from context_time_travel.real import supports_consumed_selection
        profile={'backend':'qwen35-llama.cpp','execution':{'resume_contract':'identical-native-batches-v1',
            'generated_operator':'llama-forward-consumed-v1','state_format':'qwen35-hybrid-frontier-v1',
            'selected_token_protocol':'consumed-selected-receipt-v1'}}
        self.assertTrue(supports_consumed_selection(profile))
        profile['backend']='strata';self.assertFalse(supports_consumed_selection(profile))
