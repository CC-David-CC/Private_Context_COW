"""CPU guards for the controlled overlap witness, not GPU emulation."""
import base64,copy,json,unittest
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts'))
from run_q4_cli_overlap import validate_controlled_overlap

def observation(token_time,durable_time):
    rows=[]
    for kind,stamp in (('text',token_time),('durable',durable_time)):
        event={'event':kind}
        rows.append(dict(arrival_monotonic_ns=stamp,raw_line_b64=base64.b64encode(json.dumps(event).encode()).decode()))
    return {'requests':[{'events':rows}]}

class ControlledOverlapGuards(unittest.TestCase):
    def setUp(self):
        self.admission=['op0','op1','op2','op3']
        self.sources=[dict(operation_id=op,monotonic_ns=10+i*10) for i,op in enumerate(self.admission)]
        self.observations={op:observation(15+i*10,100+i) for i,op in enumerate(self.admission)}
        self.holds=[dict(event='publication_enter',operation_id='op0',monotonic_ns=20),
                    dict(event='publication_release',operation_id='op0',monotonic_ns=80)]
        self.pending={op:dict(durable=False,branch_unchanged=True,client_exit=None) for op in self.admission}
    def check(self):
        return validate_controlled_overlap(self.admission,self.sources,self.observations,self.holds,self.pending)
    def test_actual_ordered_overlap_is_admitted(self):
        self.assertTrue(self.check()['later_gpu_token_before_first_publication_release'])
    def test_reordered_native_sources_are_rejected(self):
        self.sources[1],self.sources[2]=self.sources[2],self.sources[1]
        with self.assertRaisesRegex(AssertionError,'FIFO'):self.check()
    def test_later_work_after_release_does_not_prove_overlap(self):
        self.observations['op1']=observation(90,101)
        with self.assertRaisesRegex(AssertionError,'GPU token'):self.check()
    def test_early_ack_is_rejected(self):
        self.observations['op2']=observation(35,79)
        with self.assertRaisesRegex(AssertionError,'durable'):self.check()
    def test_wrong_hold_operation_or_duplicate_witness_is_rejected(self):
        self.holds[0]['operation_id']='other'
        with self.assertRaisesRegex(AssertionError,'hold'):self.check()
        self.setUp();self.holds.append(copy.deepcopy(self.holds[0]))
        with self.assertRaisesRegex(AssertionError,'hold'):self.check()
    def test_pending_branch_or_client_success_is_rejected(self):
        for field,value in (('durable',True),('branch_unchanged',False),('client_exit',0)):
            self.setUp();self.pending['op1'][field]=value
            with self.subTest(field=field):
                with self.assertRaisesRegex(AssertionError,'pending'):self.check()

if __name__=='__main__':unittest.main()
