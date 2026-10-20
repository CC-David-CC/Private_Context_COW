"""Ensure real CLI audit actually sends changed DRY settings."""
import unittest
from run_sampler_serving_probe import sampling_arguments


class DryObserverTests(unittest.TestCase):
    def test_changed_dry_fields_are_sent_and_unchanged_fields_are_inherited(self):
        policy=dict(temperature=.7,min_p=.05,top_k=60,top_p=.95,
            dry_multiplier=.8,dry_base=1.75,dry_allowed_length=2,dry_penalty_last_n=256)
        changed=dict(policy,dry_multiplier=1.1,dry_penalty_last_n=128)
        self.assertEqual(sampling_arguments(changed,policy),
            ['--dry-multiplier','1.1','--dry-penalty-last-n','128'])
        self.assertEqual(sampling_arguments(policy,policy),[])

    def test_starting_dry_in_a_later_generation_sends_all_new_dry_fields(self):
        old=dict(temperature=.7,min_p=.05,top_k=60,top_p=.95)
        new=dict(old,dry_multiplier=.8,dry_base=1.75,dry_allowed_length=2,dry_penalty_last_n=256)
        self.assertEqual(sampling_arguments(new,old),
            ['--dry-multiplier','0.8','--dry-base','1.75',
             '--dry-allowed-length','2','--dry-penalty-last-n','256'])


if __name__=='__main__':unittest.main()
