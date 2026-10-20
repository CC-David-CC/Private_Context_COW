"""Generation-boundary settings; synthetic logits, no GPU claim."""
import copy
import math
import random
import shlex
from pathlib import Path
import unittest

from context_time_travel import min_p, sampled_controller
from context_time_travel.store import InvalidOperation
import test_min_p_integration as fixtures

POLICY = fixtures.POLICY


def ordered_total(values):
    # The contract fixes ordinary binary64 additions in token-ID order;
    # Python 3.12 sum() uses a compensated algorithm instead.
    result = 0.0
    for value in values:
        result += value
    return result


def expected(logits, mask, temperature, min_p_value, top_k, top_p, draw):
    legal = [i for i in range(len(logits)) if mask[i // 8] & (1 << (i % 8))]
    maximum = max(logits[i] for i in legal)
    ranked = sorted((i for i in legal if min_p_value == 0 or logits[i] >= maximum + math.log(min_p_value)),
                    key=lambda i: (-logits[i], i))
    if top_k:
        ranked = ranked[:top_k]
    weights = {i: math.exp((logits[i] - maximum) / temperature) for i in ranked}
    if top_p < 1:
        total = ordered_total(weights[i] for i in sorted(ranked))
        cumulative = 0.0
        for end, i in enumerate(ranked, 1):
            cumulative += weights[i]
            if cumulative >= top_p * total:
                ranked = ranked[:end]
                break
    support = sorted(ranked)
    total = ordered_total(weights[i] for i in support)
    target = draw / (1 << 53) * total
    running = 0.0
    for i in support:
        running += weights[i]
        if weights[i] > 0 and running > target:
            return i, len(support), total
    return next(i for i in reversed(support) if weights[i] > 0), len(support), total


class ConfigurableFilterTests(unittest.TestCase):
    def test_top_p_denominator_uses_token_id_order_before_ranked_cutoff(self):
        logits=[0.]+[-36.]*100+[0.]
        denominator=ordered_total(math.exp(value) for value in logits)
        p=1.0/denominator
        result=min_p.choose(logits,bytes([255]*12+[63]),temperature=1.,min_p=0.,
                            top_k=0,top_p=p,draw53=(1<<53)-1)
        self.assertEqual((result['token'],result['kept_count'],result['total_weight']),(0,1,1.))

    def test_zero_temperature_is_legal_argmax_without_an_rng_draw(self):
        result=min_p.choose([99.,2.,2.],b'\x06',temperature=0,min_p=.05,
                            top_k=2,top_p=.9,draw53=0)
        self.assertEqual((result['token'],result['kept_count'],result['total_weight']),(1,1,1.))
        state=sampled_controller.start(dict(POLICY,temperature=0),'zero',(0,),2,[1])
        state['next_draw']=11
        selected=[]
        sampled_controller.choose_owned(state,b'\x01',lambda *a,**kw:selected.append((a,kw)) or 0)
        self.assertEqual(state['next_draw'],11)
        self.assertEqual(selected[0][0][-1],0)
    def test_filters_compose_and_rank_ties_by_token_id(self):
        for k, p, support in ((2, 1., 2), (0, .5, 2), (3, .5, 2), (100, 1., 4)):
            with self.subTest(k=k, p=p):
                result = min_p.choose([0.] * 4, b'\x0f', temperature=.7, min_p=0,
                                     top_k=k, top_p=p, draw53=(1 << 53) - 1)
                self.assertEqual((result['token'], result['kept_count']), (support - 1, support))

    def test_seeded_filters_match_independent_reference(self):
        rng = random.Random(20261016)
        for case in range(100):
            logits = [rng.choice((-4., -1., 0., .5, 2.)) for _ in range(97)]
            mask = bytes([255] * 12 + [1])
            temperature = rng.choice((.1, .7, 2.))
            mp, k, p = rng.choice((0., .05, 1.)), rng.choice((0, 1, 7, 200)), rng.choice((.01, .8, 1.))
            draw = rng.randrange(1 << 53)
            result = min_p.choose(logits, mask, temperature=temperature, min_p=mp,
                                  top_k=k, top_p=p, draw53=draw)
            self.assertEqual((result['token'], result['kept_count'], result['total_weight']),
                             expected(logits, mask, temperature, mp, k, p, draw), case)

    def test_policy_accepts_all_four_and_rejects_invalid_filters(self):
        self.assertEqual(min_p.validate_policy(dict(POLICY, top_k=42, top_p=.9))['top_k'], 42)
        for field, value in (('top_k', True), ('top_k', -1), ('top_k', 1.5),
                             ('top_p', 0), ('top_p', 1.01), ('top_p', float('nan'))):
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                min_p.validate_policy(dict(POLICY, **{field: value}))


class GenerationBoundaryTests(unittest.TestCase):
    def test_greedy_grammar_partial_can_enable_sampling_without_reset(self):
        fixture=fixtures.MinPRequestTests('test_partial_grammar_fork_resume_keeps_parent_and_nonzero_rng_cursor')
        fixture.setUp();self.addCleanup(fixture.doCleanups)
        first=fixture.request(limit=1,sampling=None)
        previous=fixture.model.controller(first['snapshot'])
        self.assertEqual(previous['schema'],2)
        answer=fixture.request(first['snapshot'],resume=True,sampling=POLICY)
        current=fixture.model.controller(answer['snapshot'])
        self.assertEqual(current['schema'],3)
        self.assertEqual(current['stream_key'],previous['stream_key'])
        self.assertEqual(current['next_draw'],previous['next_draw']+2)
        self.assertEqual(current['accepted_in_scope'],3)
        self.assertEqual(fixture.model.controller(first['snapshot']),previous)

    def test_plain_change_preserves_progress_and_parent(self):
        state = sampled_controller.start(POLICY, 'stream', (0,), 2, [1])
        state['next_draw'] = 7
        before = copy.deepcopy(state)
        session = sampled_controller.PlainSamplingSession.restore(state, (0,), 2, lambda t:b'A', lambda t:False)
        settings = dict(POLICY, temperature=1.2, min_p=.1, top_k=1, top_p=.8)
        session.configure_generation(settings)
        current = session.current_controller
        self.assertEqual({k:v for k,v in current.items() if k not in ('policy','sampler_order')},
                         {k:v for k,v in before.items() if k not in ('policy','sampler_order')})
        self.assertEqual(current['sampler_order'], sampled_controller.FILTERED_SAMPLER_ORDER)
        self.assertEqual(current['policy']['top_k'], 1)
        self.assertEqual(state, before)
        session.choose(b'\x01', lambda *a,**kw:0)
        with self.assertRaises(InvalidOperation):session.configure_generation(POLICY)
        session._consume_selected(0,b'A',False)
        with self.assertRaises(InvalidOperation):session.configure_generation(POLICY)

    def test_grammar_resume_can_change_policy_without_resetting_parser_or_rng(self):
        fixture = fixtures.MinPRequestTests('test_partial_grammar_fork_resume_keeps_parent_and_nonzero_rng_cursor')
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        first = fixture.request(limit=1)
        parent = fixture.model.snapshot_bytes(first['snapshot'])
        result = fixture.request(first['snapshot'], sampling=dict(POLICY, temperature=1.2, min_p=.1), resume=True)
        current = fixture.model.controller(result['snapshot'])
        self.assertEqual(current['next_draw'], 3)
        self.assertEqual(current['policy']['temperature'], 1.2)
        self.assertEqual(fixture.model.snapshot_bytes(first['snapshot']), parent)


class FilteredBackendTests(unittest.TestCase):
    def test_v2_command_and_receipt_bind_filters_before_native_work(self):
        fixture = fixtures.MinPBackendTests('test_native_sample_is_read_only_and_cleans_transient_mask')
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        backend = fixture.backend
        backend.profile['execution']['sampling_protocol'] = min_p.SAMPLING_PROTOCOL_V2
        response = dict(fixture.response, protocol=min_p.SAMPLING_PROTOCOL_V2, top_k=3, top_p=.9)
        commands = []
        def request(command):
            fields = shlex.split(command)
            commands.append(fields[:-1])
            self.assertEqual(Path(fields[-1]).read_bytes(), b'\0\1')
            return dict(response)
        backend._request = request
        self.assertEqual(backend.sample_legal(fixture.lease,b'\0\1',.7,.05,123,top_k=3,top_p=.9),8)
        self.assertEqual(commands,[['sample_mask_v2','4','0.7','0.05','3','0.9','123']])
        response['top_k'] = True
        with self.assertRaises(fixtures.IntegrityError):
            backend.sample_legal(fixture.lease,b'\0\1',.7,.05,123,top_k=3,top_p=.9)
        self.assertEqual(list(backend._directory.iterdir()),[])
        backend.profile['execution']['sampling_protocol'] = min_p.MIN_P_PROTOCOL
        before=len(commands)
        with self.assertRaises(InvalidOperation):
            backend.sample_legal(fixture.lease,b'\0\1',.7,.05,123,top_k=3,top_p=.9)
        self.assertEqual(len(commands),before)


if __name__ == '__main__':unittest.main()
