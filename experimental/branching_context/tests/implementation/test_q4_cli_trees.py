"""Measurement admission guards; real inference is a separate campaign."""
import base64
import copy
import json
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts'))
from run_q4_cli_trees import validate_observation


class ObservationAdmissionTests(unittest.TestCase):
    def setUp(self):
        self.receipt = {'snapshot':'saved-1','branch':{'name':'A','revision':1,'root':'root','tip':'saved-1'}}
        complete = {'event':'complete','receipt':self.receipt,'answer':{'snapshot':'saved-1','tokens':[11,12]}}
        self.trace = dict(cli_source_sha256='pinned',cli_exit_code=0,cli_exception=None,
            requests=[dict(method='POST',request_path='/cow/text/ask',operation_id='op-a',branch='A',
                           summary=dict(validation=dict(status='valid',errors=[]),token_ids=[11,12]),
                           events=[dict(raw_line_b64=base64.b64encode(json.dumps(complete).encode()).decode())])])

    def check(self, trace):
        return validate_observation(trace,'pinned','A','op-a',self.receipt,[11,12])

    def test_valid_trace_is_admitted(self):
        self.assertEqual(self.check(self.trace)['token_ids'],[11,12])

    def test_no_trace_cannot_be_a_measurement(self):
        with self.assertRaisesRegex(AssertionError,'source differs'):
            self.check({})

    def test_wrong_cli_or_failed_cli_is_rejected(self):
        for key, value in (('cli_source_sha256','other'),('cli_exit_code',1),('cli_exception',{'type':'ValueError'})):
            with self.subTest(key=key):
                trace=copy.deepcopy(self.trace);trace[key]=value
                with self.assertRaises(AssertionError):self.check(trace)

    def test_wrong_request_binding_or_stream_count_is_rejected(self):
        for key,value in (('method','GET'),('request_path','/cow/text/resume'),('operation_id','other'),('branch','B')):
            with self.subTest(key=key):
                trace=copy.deepcopy(self.trace);trace['requests'][0][key]=value
                with self.assertRaises(AssertionError):self.check(trace)
        for requests in ([],self.trace['requests']*2):
            trace=copy.deepcopy(self.trace);trace['requests']=requests
            with self.assertRaisesRegex(AssertionError,'exactly one'):self.check(trace)

    def test_invalid_summary_and_transport_failure_are_rejected(self):
        trace=copy.deepcopy(self.trace)
        trace['requests'][0]['summary']['validation']['status']='truncated'
        with self.assertRaisesRegex(AssertionError,'Invalid observer'):self.check(trace)
        for key in ('open_error','read_error','observer_error'):
            trace=copy.deepcopy(self.trace);trace['requests'][0][key]='failure'
            with self.assertRaisesRegex(AssertionError,'transport failed'):self.check(trace)

    def test_wrong_token_or_receipt_is_rejected(self):
        trace=copy.deepcopy(self.trace);trace['requests'][0]['summary']['token_ids']=[12,11]
        with self.assertRaisesRegex(AssertionError,'stored tokens'):self.check(trace)
        trace=copy.deepcopy(self.trace)
        event={'event':'complete','receipt':{'snapshot':'other'},'answer':{'snapshot':'other','tokens':[11,12]}}
        trace['requests'][0]['events'][0]['raw_line_b64']=base64.b64encode(json.dumps(event).encode()).decode()
        with self.assertRaisesRegex(AssertionError,'receipt differs'):self.check(trace)


if __name__=='__main__':unittest.main()
