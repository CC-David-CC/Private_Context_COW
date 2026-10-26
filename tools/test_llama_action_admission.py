"""Explicit 8k admission; old profile bounds and optimizer routes stay distinct."""
import unittest
from pcc_native_training import (SEQUENCE_PROFILES,validate_examples,validate_memory_profile,
                                validate_optimizer_request,validate_training_budget_request)

PROFILE='dense-llama-action-8k-v1'

class ActionAdmission(unittest.TestCase):
    def example(self,n,selected=1):
        return dict(id='owned',tokens=[0]*n,targets=[0]*n,loss_mask=[i<selected for i in range(n)],
                    verified=True,provenance_sha256='a'*64)

    def test_action_bound_is_explicit_and_old_512_bound_is_preserved(self):
        self.assertEqual(SEQUENCE_PROFILES['dense-llama-short-v1'],512)
        self.assertEqual(len(validate_examples([self.example(8192)],sequence_profile=PROFILE)[0]['tokens']),8192)
        with self.assertRaises(ValueError):validate_examples([self.example(8193)],sequence_profile=PROFILE)
        with self.assertRaises(ValueError):validate_examples([self.example(513)],sequence_profile='dense-llama-short-v1')

    def test_long_context_does_not_expand_selected_target_budget(self):
        validate_examples([self.example(8192,256)],sequence_profile=PROFILE)
        with self.assertRaises(ValueError):validate_examples([self.example(8192,257)],sequence_profile=PROFILE)

    def test_existing_optimizer_modes_work_without_adding_muon_lora(self):
        request=dict(sequence_profile=PROFILE,optimizer_state_mode='fresh')
        for optimizer,mode in [('adamw','full'),('muon','full'),('adamw','lora')]:
            self.assertEqual(validate_optimizer_request(dict(request,optimizer=optimizer,mode=mode)),optimizer)
        with self.assertRaises(ValueError):validate_optimizer_request(dict(request,optimizer='muon',mode='lora'))

    def test_allocator_is_opt_in_and_training_time_budget_is_not_implied(self):
        self.assertEqual(validate_memory_profile('allocator-v1',PROFILE),'allocator-v1')
        with self.assertRaises(ValueError):validate_memory_profile('allocator-v1','short-v1')
        self.assertFalse(validate_training_budget_request(dict(mode='full',sequence_profile=PROFILE)))
        with self.assertRaises(ValueError):
            validate_training_budget_request(dict(mode='full',sequence_profile=PROFILE,training_budget_seconds=1,deadline_seconds=60))


if __name__=='__main__':unittest.main()
