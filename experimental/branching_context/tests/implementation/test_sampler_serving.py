"""Real queue/store/HTTP-client controls with a synthetic model; not GPU evidence."""
import contextlib
import hashlib
import io
import tempfile
import threading
import unittest
from unittest.mock import patch

from context_time_travel.coordinator import ContextCoordinator
from context_time_travel.min_p import choose
from context_time_travel.pools import PoolConfig
from context_time_travel.sampled_controller import PlainSamplingSession, start
from context_time_travel.store import InvalidOperation, _json_bytes
from serve.cow_adapter import CowAdapter
from serve.cow_text_adapter import CowTextAdapter
from tools import cow_cli
import test_pool_coordinator as pool_fixtures
import serve.test_cow_adapter as adapter_fixtures
import serve.test_cow_text_adapter as text_fixtures


POLICY=dict(temperature=.7,min_p=.05,top_k=0,top_p=1.0,repetition_penalty=1.0)


class SampledModel(pool_fixtures.Model):
    def __init__(self,*args):
        super().__init__(*args)
        self.profile['execution'].update(sampling_protocol='full-vocab-min-p-u53-v1',eos_tokens=[7])
        self.sampled_calls=[]

    def run_sampled_request(self,sid,ids,count,*,sampling,stream,resume=False,on_token,cancelled=None):
        self.gate.wait(5)
        self.sampled_calls.append(dict(sampling=dict(sampling),resume=resume,stream=stream))
        history=self.history(sid)+tuple(ids)
        state=self.store.get_snapshot(sid)['controller'] if resume else start(sampling,stream,history,8,[7])
        session=PlainSamplingSession.restore(state,history,8,lambda t:b'A',lambda t:t==7)
        if resume:session.configure_generation(sampling)
        self.stats['forward_tokens']+=len(ids)
        tokens=[]
        for _ in range(count):
            token=session.choose(b'\xff',lambda mask,temp,p,draw:
                choose([1.,.9,.8,-100.,-100.,-100.,-100.,-100.],mask,
                       temperature=temp,min_p=p,draw53=draw)['token'])
            session._consume_selected(token,b'A',False)
            self.stats['forward_tokens']+=1
            tokens.append(token);on_token(token)
        target=self.store.append(sid,tuple(ids)+tuple(tokens),session.current_controller)
        self.image(target)
        return dict(snapshot=target,tokens=tokens,request_seconds=.001,generation_seconds=.001)


class SampledCoordinatorTests(unittest.TestCase):
    def start(self):
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup)
        models={};gate=threading.Event();gate.set()
        def factory(gpu):
            models[gpu]=SampledModel(temp.name,gpu,(0,),[],gate)
            return models[gpu]
        config=PoolConfig(gpus=(0,),ram_bytes=4096,disk_bytes=16384,min_available_bytes=0)
        coordinator=ContextCoordinator(temp.name,config,factory)
        self.addCleanup(coordinator.close);coordinator.ready(5)
        coordinator.create('base',(1,2),'create').durable.result(5)
        return coordinator,models,gate

    def test_sampler_policy_is_detached_and_bound_to_the_operation(self):
        c,models,gate=self.start();gate.clear()
        policy=dict(POLICY)
        request=c.submit('base',(3,),2,'sample',expected_revision=1,sampling=policy)
        policy['min_p']=.99
        self.assertIs(c.submit('base',(3,),2,'sample',expected_revision=1,sampling=POLICY),request)
        with self.assertRaises(ValueError):
            c.submit('base',(3,),2,'sample',expected_revision=1,sampling=dict(POLICY,min_p=.1))
        gate.set();receipt=request.durable.result(5)
        self.assertEqual(models[0].sampled_calls[0]['sampling'],POLICY)
        self.assertEqual(c.store.get_snapshot(receipt['snapshot'])['controller']['next_draw'],2)

    def test_resume_preserves_nonzero_rng_progress_and_rejects_changed_request_identity(self):
        c,models,_=self.start()
        first=c.submit('base',(3,),2,'first',expected_revision=1,sampling=POLICY).durable.result(5)
        before=models[0].work_counters()['forward_tokens']
        request=c.submit('base',(),3,'continued',expected_revision=2,sampling=POLICY,resume=True)
        receipt=request.durable.result(5)
        state=c.store.get_snapshot(receipt['snapshot'])['controller']
        self.assertEqual(state['next_draw'],5)
        self.assertEqual(state['stream_key'],c.store.get_snapshot(first['snapshot'])['controller']['stream_key'])
        self.assertEqual(models[0].work_counters()['forward_tokens']-before,3)
        with self.assertRaises(ValueError):
            c.submit('base',(),3,'continued',expected_revision=2,sampling=POLICY,resume=False)

    def test_restart_retry_recovers_the_committed_answer_without_new_sampling(self):
        c,models,_=self.start()
        receipt=c.submit('base',(3,),2,'sample',expected_revision=1,sampling=POLICY).durable.result(5)
        c.close()
        resumed=ContextCoordinator(c.store.root,c.config,c._factory);self.addCleanup(resumed.close);resumed.ready(5)
        result=resumed.submit('base',(3,),2,'sample',expected_revision=1,sampling=POLICY).durable.result(5)
        self.assertEqual(result['snapshot'],receipt['snapshot'])
        self.assertEqual(models[0].sampled_calls,[])
        with self.assertRaises(ValueError):
            resumed.submit('base',(3,),2,'sample',expected_revision=1,sampling=dict(POLICY,temperature=.8))

    def test_unsupported_policy_and_profile_fail_before_intent_or_forward(self):
        c,models,_=self.start()
        before=models[0].work_counters()
        for index,policy in enumerate((dict(POLICY,temperature=True),dict(POLICY,top_k=3),dict(POLICY,repetition_penalty=1.1))):
            with self.assertRaises((ValueError,InvalidOperation)):
                c.submit('base',(3,),2,'bad-'+str(index),sampling=policy)
            self.assertFalse((c._intents/('bad-'+str(index)+'.json')).exists())
        models[0].profile['execution'].pop('sampling_protocol')
        with self.assertRaises(InvalidOperation):c.submit('base',(3,),2,'unsupported',sampling=POLICY)
        self.assertFalse((c._intents/'unsupported.json').exists())
        self.assertEqual(models[0].work_counters(),before)

    def test_default_greedy_signature_keeps_the_existing_shape(self):
        c,_,_=self.start()
        request=c.submit('base',(3,),2,'greedy',expected_revision=1)
        expected=hashlib.sha256(_json_bytes(dict(branch='base',ids=[3],max_steps=2,kind='request',
            expected_revision=1,principal=None))).hexdigest()
        self.assertEqual(request.signature,expected)
        request.durable.result(5)

    def test_failed_disk_publication_retries_the_same_sampled_state_without_forward_work(self):
        c,models,_=self.start()
        original=models[0].save_prepared_branch
        def failed(*args,**kwargs):raise OSError('injected disk write failure')
        models[0].save_prepared_branch=failed
        request=c.submit('base',(3,),2,'sample',expected_revision=1,sampling=POLICY)
        with self.assertRaises(OSError):request.durable.result(5)
        self.assertEqual(c.store.head('base').revision,1)
        before=models[0].work_counters()['forward_tokens']
        models[0].save_prepared_branch=original
        c.retry_publication('sample')
        receipt=request.durable.result(5)
        self.assertEqual(models[0].work_counters()['forward_tokens'],before)
        self.assertEqual(c.store.get_snapshot(receipt['snapshot'])['controller']['next_draw'],2)


class SampledAdapterTests(unittest.TestCase):
    def setUp(self):
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup)
        self.coordinator=adapter_fixtures.SyntheticCoordinator(temp.name)
        root=self.coordinator.store.create_root('model',(1,2),{'status':'idle'})
        self.coordinator.store.fork('main',root)
        self.adapter=CowAdapter(self.coordinator)

    def test_positive_sampler_settings_reach_the_queue_with_only_the_new_suffix(self):
        self.adapter.generate('main',(1,2,3),0,'sample',8,settings=POLICY)
        call=self.coordinator.submissions[-1]
        self.assertEqual(call[1],(3,))
        self.assertEqual(call[4]['sampling'],POLICY)

    def test_empty_suffix_resumes_the_saved_sampler_instead_of_resetting_it(self):
        head=self.coordinator.store.head('main')
        state=start(POLICY,'saved',(1,2),8,[7]);state['next_draw']=9
        sid=self.coordinator.store.append(head.tip,(),state)
        self.coordinator.store.move('main',0,sid,'save')
        self.adapter.generate('main',(1,2),1,'resume',8,settings=POLICY)
        self.assertIs(self.coordinator.submissions[-1][4]['resume'],True)
        self.adapter.generate('main',(1,2),1,'changed',8,settings=dict(POLICY,min_p=.1))
        self.assertEqual(self.coordinator.submissions[-1][4]['sampling']['min_p'],.1)
        self.assertEqual(self.coordinator.store.get_snapshot(sid)['controller'],state)

    def test_invalid_sampler_settings_fail_before_submit(self):
        for settings in ({'temperature':-.7},{'min_p':1.05},{'temperature':0,'top_p':0}):
            with self.assertRaises(ValueError):self.adapter.generate('main',(1,2,3),0,'bad',8,settings=settings)
        self.assertEqual(self.coordinator.submissions,[])


class SampledTextTests(unittest.TestCase):
    def test_continue_forwards_the_saved_sampler_and_decoder_boundary(self):
        tokens=text_fixtures.TokenAdapter((1,2))
        tokens.controller=start(POLICY,'saved',(1,2),8,[7])
        tokens.controller['next_draw']=9
        adapter=CowTextAdapter(tokens,text_fixtures.Tokenizer(),lambda *a,**kw:None,lambda *a:None)
        request=adapter.resume('main',0,'resume',8)
        self.assertIs(request.request,tokens.request)
        self.assertEqual(tokens.calls[-1][6]['settings'],POLICY)

    def test_grammar_controllers_cannot_enter_the_plain_text_continue_path(self):
        for schema,temp in ((2,0),(3,.7)):
            tokens=text_fixtures.TokenAdapter((1,2))
            tokens.controller=dict(schema=schema,status='active',constraint={'agent_state':'EDIT'},
                                   policy=dict(POLICY,temperature=temp))
            adapter=CowTextAdapter(tokens,text_fixtures.Tokenizer(),lambda *a,**kw:None,lambda *a:None)
            with self.assertRaises(ValueError):adapter.resume('main',0,'resume',8)
            self.assertEqual(tokens.calls,[])


class SampledCLITests(unittest.TestCase):
    def test_client_journal_body_contains_the_explicit_preset(self):
        with tempfile.TemporaryDirectory() as directory:
            args=cow_cli.parser().parse_args(['--state-dir',directory,'ask','--branch','main','--text','question',
                '--temperature','0.70','--min-p','0.05','--operation-id','sample'])
            with (patch.object(cow_cli,'_head',return_value={'revision':1}),
                  patch.object(cow_cli,'_post_mutation') as post,
                  contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO())):
                post.return_value={'receipt':{'ok':True}}
                cow_cli.run(args)
            body=post.call_args.args[4]
            self.assertEqual(body['settings'],{key:value for key,value in POLICY.items() if key!='repetition_penalty'})

    def test_invalid_negative_temperature_fails_before_network(self):
        with tempfile.TemporaryDirectory() as directory:
            args=cow_cli.parser().parse_args(['--state-dir',directory,'ask','--branch','main','--text','question',
                '--temperature','-0.70'])
            with patch.object(cow_cli.CowClient,'request') as network,self.assertRaises(cow_cli.ClientError):
                cow_cli.run(args)
            network.assert_not_called()


if __name__=='__main__':unittest.main()
