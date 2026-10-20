"""DRY policy and branch-owned ordered-history regressions; CPU fixtures."""
import copy
import shlex
import struct
from pathlib import Path
import unittest

from context_time_travel import min_p, sampled_controller
from context_time_travel.store import IntegrityError, InvalidOperation

BASE = dict(temperature=.7, min_p=.05, top_k=60, top_p=.95, repetition_penalty=1.)
DRY = dict(BASE, dry_multiplier=.8, dry_base=1.75,
           dry_allowed_length=2, dry_penalty_last_n=256)
PIECES = (b'a', b'b', b'c', b'\n', b'quote"part', b'')


def session(history, policy=DRY):
    state = sampled_controller.start(policy, 'dry-branch', history, len(PIECES), [5])
    return sampled_controller.PlainSamplingSession.restore(
        state, history, len(PIECES), lambda token: PIECES[token], lambda token: token == 5)


class DryPolicyTests(unittest.TestCase):
    def test_dry_reference_changes_selection_before_filters_without_mutating_logits(self):
        values=[0.,0.,1.,.5,0.,-10.]
        original=list(values)
        plain=min_p.choose(values,b'\x3f',temperature=0,min_p=.05,draw53=0)
        dry=min_p.choose(values,b'\x3f',temperature=0,min_p=.05,draw53=0,
            dry_multiplier=.8,dry_base=1.75,dry_allowed_length=2,dry_penalty_last_n=256,
            dry_history=(0,1,2,3,0,1),dry_breakers=(3,))
        self.assertEqual((plain['token'],dry['token']),(2,3))
        self.assertEqual(values,original)
        self.assertEqual(dry['selection_protocol'],'full-vocab-dry-min-p-u53-v3')

    def test_existing_policy_stays_byte_equivalent_without_dry(self):
        self.assertEqual(min_p.validate_policy(BASE), BASE)

    def test_requested_preset_and_partial_defaults(self):
        self.assertEqual(min_p.validate_policy(DRY), DRY)
        self.assertEqual(min_p.validate_policy(dict(BASE, dry_multiplier=.8)), DRY)

    def test_invalid_dry_fields_are_rejected(self):
        for field, value in (('dry_multiplier', True), ('dry_multiplier', -.1),
                             ('dry_multiplier', float('nan')), ('dry_base', .9),
                             ('dry_base', float('inf')), ('dry_allowed_length', True),
                             ('dry_allowed_length', -1), ('dry_penalty_last_n', 1.5),
                             ('dry_penalty_last_n', 1 << 32)):
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                min_p.validate_policy(dict(DRY, **{field: value}))


class DrySessionTests(unittest.TestCase):
    def test_selector_receives_ordered_tail_and_breakers_from_prompt(self):
        history = (0, 1, 2, 3, 0, 1)
        owned = session(history)
        calls = []
        owned.choose(b'\x3f', lambda *args, **kw: calls.append(kw) or 2)
        self.assertEqual(tuple(calls[0]['dry_history']), history)
        self.assertEqual(tuple(calls[0]['dry_breakers']), (3,))
        self.assertEqual({key:calls[0][key] for key in DRY if key.startswith('dry_')},
                         {key:DRY[key] for key in DRY if key.startswith('dry_')})
        with self.assertRaises(InvalidOperation): _ = owned.current_controller
        owned._consume_selected(2, PIECES[2], False)
        self.assertEqual(owned.current_controller['next_draw'], 1)

    def test_tail_is_bounded_and_reads_only_relevant_token_pieces(self):
        history = (4,) * 100 + (0, 1, 0, 1)
        policy = dict(DRY, dry_penalty_last_n=4)
        state = sampled_controller.start(policy, 'bounded', history, 6, [5])
        reads = []
        owned = sampled_controller.PlainSamplingSession.restore(state, history, 6,
            lambda token: reads.append(token) or PIECES[token], lambda token: token == 5)
        reads.clear()
        calls = []
        owned.choose(b'\x3f', lambda *args, **kw: calls.append(kw) or 0)
        self.assertEqual(tuple(calls[0]['dry_history']), (0, 1, 0, 1))
        self.assertNotIn(4, reads)

    def test_fork_and_restart_derive_independent_history_without_resetting_rng(self):
        history = (0, 1, 0, 1)
        parent = session(history).current_controller
        before = copy.deepcopy(parent)
        left = sampled_controller.PlainSamplingSession.restore(parent, history, 6,
            lambda token: PIECES[token], lambda token: token == 5)
        right = sampled_controller.PlainSamplingSession.restore(parent, history, 6,
            lambda token: PIECES[token], lambda token: token == 5)
        left.choose(b'\x3f', lambda *a, **kw: 2)
        left._consume_selected(2, PIECES[2], False)
        saved = left.current_controller
        restarted = sampled_controller.PlainSamplingSession.restore(saved, history + (2,), 6,
            lambda token: PIECES[token], lambda token: token == 5)
        seen = []
        restarted.choose(b'\x3f', lambda *a, **kw: seen.append(kw['dry_history']) or 0)
        self.assertEqual(tuple(seen[0]), history + (2,))
        self.assertEqual(right.current_controller, before)
        self.assertEqual(parent, before)
        self.assertEqual(saved['next_draw'], 1)

    def test_generation_override_can_disable_dry_and_preserve_other_progress(self):
        owned = session((0, 1, 0, 1))
        initial = owned.current_controller
        owned.configure_generation(BASE)
        current = owned.current_controller
        self.assertNotIn('dry_multiplier', current['policy'])
        self.assertEqual(current['stream_key'], initial['stream_key'])
        self.assertEqual(current['seen'], initial['seen'])
        self.assertEqual(current['next_draw'], initial['next_draw'])
        calls = []
        owned.choose(b'\x3f', lambda *a, **kw: calls.append(kw) or 0)
        self.assertNotIn('dry_history', calls[0])

    def test_zero_temperature_keeps_dry_but_draw_cursor_does_not_advance(self):
        owned = session((0, 1, 0, 1), dict(DRY, temperature=0))
        calls = []
        owned.choose(b'\x3f', lambda *a, **kw: calls.append((a, kw)) or 2)
        owned._consume_selected(2, PIECES[2], False)
        self.assertEqual(owned.current_controller['next_draw'], 0)
        self.assertEqual(calls[0][0][-1], 0)
        self.assertIn('dry', owned.current_controller['sampler_order'])

    def test_corrupt_sampler_order_is_rejected(self):
        state = session((0, 1)).current_controller
        state['sampler_order'] = sampled_controller.FILTERED_SAMPLER_ORDER
        with self.assertRaises(IntegrityError): sampled_controller.validate_controller(state)


class DryBackendTests(unittest.TestCase):
    def setUp(self):
        import test_min_p_integration as fixtures
        self.fixture = fixtures.MinPBackendTests('test_native_sample_is_read_only_and_cleans_transient_mask')
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.backend=self.fixture.backend
        self.backend.profile['execution']['sampling_protocol']='full-vocab-dry-min-p-u53-v3'
        self.calls=[]
        self.response=dict(self.fixture.response,protocol='full-vocab-dry-min-p-u53-v3',
            top_k=60,top_p=.95,dry_multiplier=.8,dry_base=1.75,
            dry_allowed_length=2,dry_penalty_last_n=256,dry_history_count=4,dry_breaker_count=1,
            dry_penalized_candidates=1)
        def request(command):
            fields=shlex.split(command)
            self.calls.append(fields)
            self.assertEqual(Path(fields[7]).read_bytes(), b'\0\1')
            self.assertEqual(Path(fields[12]).read_bytes(),struct.pack('<4I',0,1,3,0))
            self.assertEqual(Path(fields[13]).read_bytes(),struct.pack('<I',3))
            return dict(self.response)
        self.backend._request=request

    def sample(self, **changes):
        args=dict(top_k=60,top_p=.95,dry_multiplier=.8,dry_base=1.75,
            dry_allowed_length=2,dry_penalty_last_n=256,dry_history=(0,1,3,0),dry_breakers=(3,))
        args.update(changes)
        return self.backend.sample_legal(self.fixture.lease,b'\0\1',.7,.05,123,**args)

    def test_v3_binds_policy_tail_and_clean_transient_files(self):
        self.assertEqual(self.sample(),8)
        self.assertEqual(self.calls[0][:7],['sample_mask_v3','4','0.7','0.05','60','0.95','123'])
        self.assertEqual(self.calls[0][8:12],['0.8','1.75','2','256'])
        self.assertEqual(self.fixture.lease.consumed,4)
        self.assertEqual(list(self.backend._directory.iterdir()),[])

    def test_bad_tail_or_missing_capability_fails_before_native(self):
        for changes in (dict(dry_history=(0,1)),dict(dry_breakers=(3,3)),
                        dict(dry_breakers=(4,)),dict(dry_history=(True,1,3,0))):
            with self.subTest(changes=changes),self.assertRaises(InvalidOperation): self.sample(**changes)
        self.backend.profile['execution']['sampling_protocol']=min_p.SAMPLING_PROTOCOL_V2
        with self.assertRaises(InvalidOperation):self.sample()
        self.assertEqual(self.calls,[])
        self.assertEqual(list(self.backend._directory.iterdir()),[])

    def test_receipt_mismatch_rejects_and_cleans_files(self):
        for field,value in (('dry_multiplier',.9),('dry_allowed_length',True),
                            ('dry_history_count',3),('dry_breaker_count',2),
                            ('dry_penalized_candidates',True),('dry_penalized_candidates',10)):
            original=copy.deepcopy(self.response)
            self.response[field]=value
            with self.subTest(field=field),self.assertRaises(IntegrityError):self.sample()
            self.assertEqual(list(self.backend._directory.iterdir()),[])
            self.response=original


class DryGrammarTests(unittest.TestCase):
    def test_gbnf_uses_own_history_and_remains_legal_across_partial_resume(self):
        import test_min_p_integration as fixtures
        fixture=fixtures.MinPRequestTests('test_partial_grammar_fork_resume_keeps_parent_and_nonzero_rng_cursor')
        fixture.setUp();self.addCleanup(fixture.doCleanups)
        fixture.model.profile['execution']['sampling_protocol']='full-vocab-dry-min-p-u53-v3'
        histories=[]
        def sample(lease,mask,t,p,draw,**filters):
            histories.append(filters['dry_history'])
            return next(token for token in range(2) if mask[token//8] & (1<<(token%8)))
        fixture.backend.sample_legal=sample
        first=fixture.request(limit=1,sampling=DRY)
        parent=fixture.model.snapshot_bytes(first['snapshot'])
        before=fixture.model.controller(first['snapshot'])
        result=fixture.request(first['snapshot'],resume=True,sampling=DRY)
        self.assertEqual(tuple(histories[1]),tuple(histories[0])+(first['tokens'][0],))
        self.assertEqual(fixture.model.controller(result['snapshot'])['next_draw'],3)
        self.assertEqual(fixture.model.controller(first['snapshot']),before)
        self.assertEqual(fixture.model.snapshot_bytes(first['snapshot']),parent)


if __name__ == '__main__': unittest.main()
