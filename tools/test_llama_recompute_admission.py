"""The selected-head profile preserves context, loss-mask and optimizer bounds."""
import unittest
from tools.pcc_native_training import (validate_examples,validate_memory_profile,
    validate_optimizer_request,validate_training_budget_request,SEQUENCE_PROFILES)

PROFILE='dense-llama-action-recompute-8k-v1'

class SelectedHeadAdmission(unittest.TestCase):
    def example(self,n,selected=2):
        return dict(id='owned',tokens=[0]*n,targets=[0]*n,loss_mask=[i<selected for i in range(n)],
            verified=True,provenance_sha256='a'*64)

    def test_context_and_selected_target_bounds(self):
        validate_examples([self.example(8192,256)],sequence_profile=PROFILE)
        for e in [self.example(8193),self.example(8192,257),self.example(8192,0)]:
            with self.assertRaises(ValueError):validate_examples([e],sequence_profile=PROFILE)

    def test_older_profiles_keep_their_bounds(self):
        self.assertEqual(SEQUENCE_PROFILES['dense-llama-short-v1'],512)
        self.assertEqual(SEQUENCE_PROFILES['dense-llama-action-8k-v1'],8192)
        with self.assertRaises(ValueError):validate_examples([self.example(513)],sequence_profile='dense-llama-short-v1')

    def test_existing_optimizer_modes_and_allocator(self):
        self.assertEqual(validate_memory_profile('allocator-v1',PROFILE),'allocator-v1')
        request=dict(sequence_profile=PROFILE,optimizer_state_mode='fresh')
        for optimizer,mode in [('adamw','full'),('muon','full'),('adamw','lora')]:
            self.assertEqual(validate_optimizer_request(dict(request,optimizer=optimizer,mode=mode)),optimizer)
        with self.assertRaises(ValueError):validate_optimizer_request(dict(request,optimizer='muon',mode='lora'))

    def test_time_budget_is_not_added_by_the_new_profile(self):
        self.assertFalse(validate_training_budget_request(dict(mode='full',sequence_profile=PROFILE)))
        with self.assertRaises(ValueError):
            validate_training_budget_request(dict(mode='full',sequence_profile=PROFILE,training_budget_seconds=1,deadline_seconds=60))

if __name__=='__main__':unittest.main()

