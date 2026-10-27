"""Opt-in optimizer admission; native numerical qualification is separate."""
import unittest
from tools.pcc_native_training import validate_optimizer_request,validate_lora_rank
from tools.pcc_optimizer_profiles import profiles, identity, recipe


class FactorwiseAdmission(unittest.TestCase):
    def test_extended_ranks_need_new_profile(self):
        for rank in (32,128,2048,4096):
            request=dict(mode='lora',rank=rank,lora_rank_profile='extended-4096-v1')
            self.assertEqual(validate_lora_rank(request),rank)
        for request in (dict(mode='lora',rank=2048),dict(mode='full',rank=32,lora_rank_profile='extended-4096-v1'),
                        dict(mode='lora',rank=4097,lora_rank_profile='extended-4096-v1')):
            with self.assertRaises(ValueError):validate_lora_rank(request)
    def request(self, **changes):
        value=dict(optimizer='muon',mode='lora',sequence_profile='dense-llama-short-v1',
                   optimizer_state_mode='fresh',lora_update_space='factorwise-v1')
        value.update(changes)
        return value

    def test_explicit_adaptive_factor_routes(self):
        for optimizer in ('adamw','muon'):
            self.assertEqual(validate_optimizer_request(self.request(optimizer=optimizer)),optimizer)

    def test_legacy_muon_lora_still_refused(self):
        request=self.request();del request['lora_update_space']
        with self.assertRaises(ValueError):validate_optimizer_request(request)

    def test_marker_never_silently_ignored(self):
        for changes in (dict(optimizer='sgd'),dict(mode='full'),
                        dict(lora_update_space=None),dict(lora_update_space='typo'),
                        dict(sequence_profile='short-v1')):
            with self.subTest(changes=changes),self.assertRaises(ValueError):
                validate_optimizer_request(self.request(**changes))

    def test_resume_still_requires_exact_parent_adapter(self):
        with self.assertRaises(ValueError):
            validate_optimizer_request(self.request(optimizer_state_mode='resume'))

    def test_profile_binds_factor_space_without_upgrading_legacy(self):
        profile=dict(schema=3,title='Factorwise Muon',mode='lora',rank=32,
                     optimizer='muon',learning_rate=.001,optimizer_settings=dict(schema=1),
                     state_policy='fresh',training_budget_seconds=None,
                     lora_update_space='factorwise-v1')
        key=identity(profile)
        self.assertEqual(profiles({key:profile},sequence_profile='dense-llama-short-v1',deadline=60)[key],profile)
        self.assertEqual(recipe(profile)['lora_update_space'],'factorwise-v1')
        for changes in (dict(schema=2),dict(lora_update_space='unknown'),dict(mode='full',rank=None)):
            bad=dict(profile,**changes)
            with self.subTest(changes=changes),self.assertRaises(ValueError):
                profiles({identity(bad):bad},sequence_profile='dense-llama-short-v1',deadline=60)
        extended=dict(profile,schema=4,rank=4096,lora_rank_profile='extended-4096-v1')
        self.assertEqual(profiles({identity(extended):extended},sequence_profile='dense-llama-short-v1',deadline=60)[identity(extended)],extended)
        self.assertEqual(recipe(extended)['lora_rank_profile'],'extended-4096-v1')
        for change in (dict(schema=3),dict(rank=4097),dict(lora_rank_profile='unknown')):
            bad=dict(extended,**change)
            with self.subTest(change=change),self.assertRaises(ValueError):
                profiles({identity(bad):bad},sequence_profile='dense-llama-short-v1',deadline=60)


if __name__=='__main__':unittest.main()
