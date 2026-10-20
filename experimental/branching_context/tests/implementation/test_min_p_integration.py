"""Protocol and branch/controller controls; no GPU execution is claimed here."""
import copy
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import shlex
import unittest
from unittest.mock import patch

from context_time_travel import controller, min_p, sampled_controller
from context_time_travel.store import IntegrityError, InvalidOperation
from context_time_travel import strata_backend as sb
import test_grammar_mask_selection as mask_fixtures
import test_state_request as request_fixtures

POLICY=dict(temperature=.70,min_p=.05,top_k=0,top_p=1.0,repetition_penalty=1.0)
PROTOCOL='full-vocab-min-p-u53-v1'


class MinPBackendTests(unittest.TestCase):
    def setUp(self):
        mask_fixtures.MaskSelectionBackendTests.setUp(self)
        self.backend.profile['execution']['sampling_protocol']=PROTOCOL
        self.response=dict(status='ok',operator='full-vocab-min-p',protocol=PROTOCOL,token=8,
            consumed=4,logits_count=9,mask_bytes=2,legal_count=1,kept_count=1,draw53=123)
        def request(command):
            fields=shlex.split(command)
            self.assertEqual(fields[:5],['sample_mask','4','0.7','0.05','123'])
            self.requests.append(Path(fields[5]).read_bytes())
            return dict(self.response)
        self.backend._request=request

    def sample(self,mask=b'\0\1'):
        return self.backend.sample_legal(self.lease,mask,.7,.05,123)

    def test_native_sample_is_read_only_and_cleans_transient_mask(self):
        self.assertEqual(self.sample(),8)
        self.assertEqual(self.lease.consumed,4)
        self.assertEqual(self.requests,[b'\0\1'])
        self.assertEqual(list(self.backend._directory.iterdir()),[])

    def test_receipt_cannot_alias_bool_int_or_another_draw_frontier_or_support(self):
        good=dict(self.response)
        for field,value in [('token',True),('token',0),('consumed',5),('draw53',124),
            ('draw53',True),('legal_count',True),('legal_count',2),('kept_count',0),
            ('mask_bytes',3),('protocol','wrong'),('operator','greedy')]:
            self.response=dict(good,**{field:value})
            with self.subTest(field=field,value=value),self.assertRaises(IntegrityError):self.sample()
            self.assertEqual(list(self.backend._directory.iterdir()),[])
        self.assertEqual(self.lease.consumed,4)

    def test_invalid_inputs_and_missing_capability_fail_before_native_command(self):
        for temp,p,draw in [(0,.05,123),(.7,-.1,123),(.7,.05,True),(.7,.05,1<<53),
            (float('nan'),.05,123),(.7,float('inf'),123)]:
            with self.assertRaises(InvalidOperation):
                self.backend.sample_legal(self.lease,b'\0\1',temp,p,draw)
        for mask in [b'',b'\0\2',b'\0\0']:
            with self.assertRaises(InvalidOperation):self.sample(mask)
        self.backend.profile['execution'].pop('sampling_protocol')
        with self.assertRaises(InvalidOperation):self.sample()
        self.assertEqual(self.requests,[])

    def test_foreign_thread_fails_before_touching_files_or_native_state(self):
        with ThreadPoolExecutor(max_workers=1) as pool:
            with self.assertRaises(InvalidOperation):pool.submit(self.sample).result()
        self.assertEqual(self.requests,[])
        self.assertEqual(list(self.backend._directory.iterdir()),[])

    def test_ready_capability_is_optional_but_must_match_when_requested(self):
        sb._validate_sampling_capability({'execution':{}},{})
        for ready in [{},{'sampling_protocol':'wrong'}]:
            with self.assertRaises(IntegrityError):sb._validate_sampling_capability(self.backend.profile,ready)
        sb._validate_sampling_capability(self.backend.profile,{'sampling_protocol':PROTOCOL})


class MinPControllerBoundaryTests(unittest.TestCase):
    def state(self):return sampled_controller.start(POLICY,'draws',(0,),2,[1])

    def test_plain_schema3_routes_to_the_checked_validator_and_round_trips(self):
        state=self.state()
        self.assertEqual(controller._validated_state(json.loads(json.dumps(state))),state)
        with self.assertRaises(InvalidOperation):controller.choose((0.,0.),state)

    def test_boolean_seen_count_is_rejected_even_with_equal_history_counts(self):
        state=self.state()
        state['seen']['0']=True
        with self.assertRaises(IntegrityError):sampled_controller.validate_controller(state,history=(0,),vocab_size=2)

    def test_last_supported_draw_can_publish_exhausted_cursor_then_reject_next_draw(self):
        state=self.state()
        state['next_draw']=(1<<128)-2
        session=sampled_controller.PlainSamplingSession.restore(state,(0,),2,lambda t:b'A' if t==0 else b'',lambda t:t==1)
        token=session.choose(b'\1',lambda *args:0)
        session._consume_selected(token,b'A',False)
        self.assertEqual(session.current_controller['next_draw'],(1<<128)-1)
        with self.assertRaises(IntegrityError):session.choose(b'\1',lambda *args:0)


class MinPRequestTests(unittest.TestCase):
    def setUp(self):
        request_fixtures.StateRequestTests.setUp(self)
        self.model.profile['execution']['sampling_protocol']=PROTOCOL
        self.model.profile['execution']['grammar_selection_protocol']='finite-mask-argmax-v1'
        self.samples=[]
        def sample(lease,mask,temp,p,draw):
            self.samples.append(draw)
            return min_p.choose(self.backend.logits(lease),mask,temperature=temp,min_p=p,draw53=draw)['token']
        self.backend.sample_legal=sample
        self.backend.choose_legal=lambda lease,mask:next(t for t in range(2) if mask[t>>3]&(1<<(t&7)))

    def request(self,snapshot=None,**kwargs):
        return self.model.run_state_request(snapshot or self.sid,(),kwargs.pop('limit',8),
            engine=self.engine,state='edit',stream='minp',sampling=kwargs.pop('sampling',POLICY),**kwargs)

    def test_grammar_minp_consumes_then_publishes_complete_draw_progress_once(self):
        result=self.request()
        self.assertEqual(result['tokens'],[0,0,1])
        record=self.model.controller(result['snapshot'])
        self.assertEqual((record['schema'],record['next_draw'],record['accepted_in_scope']),(3,3,3))
        self.assertEqual((self.backend.captures,self.backend.decode_calls),(1,3))
        self.assertEqual(record['policy']['min_p'],.05)
        self.assertEqual(controller._validated_state(record),record)

    def test_partial_grammar_fork_resume_keeps_parent_and_nonzero_rng_cursor(self):
        first=self.request(limit=1)
        original=self.model.snapshot_bytes(first['snapshot'])
        a=self.request(first['snapshot'],sampling=None,resume=True)
        b=self.request(first['snapshot'],sampling=None,resume=True)
        self.assertEqual(a['tokens'],b['tokens'])
        self.assertEqual(self.model.controller(a['snapshot']),self.model.controller(b['snapshot']))
        self.assertEqual(self.model.controller(a['snapshot'])['next_draw'],3)
        self.assertEqual(self.model.snapshot_bytes(first['snapshot']),original)

    def test_resume_explicitly_changes_policy_without_bypassing_grammar(self):
        first=self.request(limit=1)
        result=self.request(first['snapshot'],sampling=dict(POLICY,min_p=.1),resume=True)
        self.assertEqual(self.model.controller(result['snapshot'])['policy']['min_p'],.1)
        self.assertEqual(self.model.controller(first['snapshot'])['policy']['min_p'],.05)
        with self.assertRaises(InvalidOperation):controller.consume(self.model.controller(first['snapshot']),0,b'A',False)

    def test_failed_native_consumption_does_not_publish_rng_or_child_snapshot(self):
        original=self.model.snapshot_bytes(self.sid)
        objects=set(self.model.store.objects.iterdir())
        self.backend.consume_generated=lambda *args:(_ for _ in ()).throw(RuntimeError('failed native consume'))
        with self.assertRaisesRegex(RuntimeError,'failed native consume'):self.request()
        self.assertEqual(self.model.snapshot_bytes(self.sid),original)
        self.assertEqual(set(self.model.store.objects.iterdir()),objects)

    def test_plain_sampling_uses_same_branch_controller_and_saved_resume(self):
        first=self.model.run_sampled_request(self.sid,(),1,sampling=POLICY,stream='plain')
        self.assertEqual(self.model.controller(first['snapshot'])['schema'],3)
        if not first['stopped']:
            second=self.model.run_sampled_request(first['snapshot'],(),2,sampling=None,stream='ignored',resume=True)
            self.assertEqual(self.model.controller(second['snapshot'])['next_draw'],1+len(second['tokens']))

    def test_plain_resume_rejects_stop_tokens_that_differ_from_execution_before_work(self):
        saved=sampled_controller.start(POLICY,'plain',self.model.history(self.sid),2,[0])
        before=(self.backend.restores,self.backend.captures,len(self.samples))
        with patch.object(self.model,'controller',return_value=saved):
            with self.assertRaisesRegex(InvalidOperation,'stop tokens'):
                self.model.run_sampled_request(self.sid,(),1,sampling=None,stream='ignored',resume=True)
        self.assertEqual((self.backend.restores,self.backend.captures,len(self.samples)),before)


if __name__=='__main__':unittest.main()
