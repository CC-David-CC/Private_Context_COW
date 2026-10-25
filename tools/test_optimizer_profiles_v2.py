"""Versioned policy contracts; native numerical continuation is tested separately."""
import copy
import unittest
from tools.pcc_optimizer_profiles import profiles, select, identity, validate_parent


class VersionedOptimizerProfiles(unittest.TestCase):
    def profile(self, **changes):
        return dict(dict(schema=2, title='Explicit LoRA AdamW', mode='lora', rank=2,
            optimizer='adamw', learning_rate=.001, optimizer_settings=dict(schema=1),
            state_policy='fresh', training_budget_seconds=None), **changes)

    def register(self, *values):
        return profiles({identity(p): p for p in values}, sequence_profile='dense-llama-short-v1', deadline=60)

    def request(self, profile, **changes):
        return dict(dict(mode='lora', rank=2, learning_rate=.001, parent_job_id=None,
            optimizer_profile_id=identity(profile), lesson_ids=['a', 'b']), **changes)

    def test_schema1_remains_full_only_without_new_fields(self):
        old = dict(schema=1, title='Legacy', optimizer='adamw', learning_rate=.001,
            optimizer_settings=dict(schema=1), state_policy='fresh', training_budget_seconds=None)
        key = identity(old); registered = self.register(old)
        self.assertEqual(registered, {key: old})
        self.assertEqual(select(registered, self.request(old, mode='full')), old)
        with self.assertRaises(ValueError): select(registered, self.request(old))
        changed = dict(old, mode='lora', rank=2)
        with self.assertRaises(ValueError): self.register(changed)

    def test_schema2_explicit_lora_rank_rate_and_identity(self):
        p = self.profile(); registered = self.register(p)
        self.assertEqual(select(registered, self.request(p)), p)
        for changes in [dict(mode='full'), dict(rank=4), dict(learning_rate=.002),
                        dict(optimizer_profile_id='f'*64), dict(optimizer_profile_id=None)]:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                select(registered, self.request(p, **changes))
        p['rank'] = 4
        self.assertEqual(next(iter(registered.values()))['rank'], 2)
        next(iter(registered.values()))['rank'] = 4
        with self.assertRaises(ValueError): select(registered, self.request(self.profile()))

    def test_unsupported_policies_refused_at_registration(self):
        for changes in [dict(optimizer='muon'), dict(training_budget_seconds=1.),
                dict(rank=True), dict(rank=0), dict(rank=257), dict(rank=None),
                dict(mode='full'), dict(mode='guess'), dict(schema=True),
                dict(optimizer_settings={'schema': 1, 'unknown': True})]:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.register(self.profile(**changes))
        for optimizer in ('adamw', 'muon'):
            p = self.profile(mode='full', rank=None, optimizer=optimizer)
            self.assertEqual(select(self.register(p), self.request(p, mode='full')), p)

    def test_resume_recipe_and_order_checked_without_state_io(self):
        p = self.profile(); resume = self.profile(state_policy='resume', title='Explicit resume')
        registered = self.register(p, resume)
        parent = dict(request=self.request(p), candidate=dict(receipt=dict(optimizer='adamw',
            optimizer_checkpoint=dict(path='/must-not-open', sha256='0'*64))))
        req = self.request(resume, parent_job_id='prior')
        validate_parent(registered, req, parent)
        for changes in [dict(lesson_ids=['b', 'a']), dict(lesson_ids=['c'])]:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                validate_parent(registered, dict(req, **changes), parent)
        for variant in [self.profile(rank=4), self.profile(learning_rate=.002),
                        self.profile(optimizer_settings=dict(schema=1, beta1=.8))]:
            other = self.register(p, resume, variant)
            changed = copy.deepcopy(parent); changed['request']['optimizer_profile_id'] = identity(variant)
            with self.assertRaises(ValueError): validate_parent(other, req, changed)
        for optimizer, checkpoint in [('sgd', {}), ('adamw', None)]:
            changed = copy.deepcopy(parent)
            changed['candidate']['receipt'].update(optimizer=optimizer, optimizer_checkpoint=checkpoint)
            with self.assertRaises(ValueError): validate_parent(registered, req, changed)
        changed = copy.deepcopy(parent); changed['request'].pop('optimizer_profile_id')
        with self.assertRaises(ValueError): validate_parent(registered, req, changed)
        # Fresh is an explicit new block on inherited weights/hat, not a moments resume.
        validate_parent(registered, self.request(p, parent_job_id='prior', lesson_ids=['new']), parent)
        with self.assertRaises(ValueError): select(registered, self.request(resume))


if __name__ == '__main__': unittest.main()
