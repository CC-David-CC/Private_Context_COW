"""Controlled nonmodel saved native evidence. No model, GPU, build or native job."""
from pathlib import Path
import copy,fcntl,hashlib,json,os,struct,tempfile,unittest
from contextlib import ExitStack
from unittest import mock
from tools import pcc_native_completion as nc
from tools.pcc_lessons import LessonService,NativeRunner,digest
from tools.pcc_native_training import optimizer_dataset_sha256
from tools.pcc_optimizer_profiles import identity

MODEL='f'*64
GONE=dict(pid=2147480000,start_ticks=1,boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip())


def payload(path,model,dataset,cursor,steps,config):
    header=dict(schema=1,kind='pcc.completed-optimizer-state.v1',storage='le-f32',model_sha256=model,
        dataset_sha256=dataset,example_cursor=cursor,completed_steps=steps,config=config,
        tensors=[dict(name='blk.0.test.opt_m',shape=[1,1],role='hidden_matrix',optimizer='adamw',bytes=4)])
    raw=json.dumps(header).encode();path.write_bytes(b'PCCOPT1\n'+len(raw).to_bytes(8,'little')+raw+struct.pack('<f',.25));path.chmod(0o600)
    return header


class SavedEvidence:
    """Actual service readers with bytes expressly NOT representing a model."""
    def __init__(self,root,*,mode='full',optimizer='adamw',steps=3,parent=None,resume=False,job='original-job',factorwise=False,rank=2):
        self.root=Path(root);self.root.mkdir(mode=0o700,exist_ok=True)
        self.executable=self.root/'nonmodel-executable';self.executable.write_bytes(b'NOT AN EXECUTABLE OR MODEL');self.executable.chmod(0o600)
        self.base=self.root/'nonmodel-base.gguf';self.base.write_bytes(b'CONTROLLED NONMODEL BASE');self.base.chmod(0o600)
        self.config=dict(model=str(self.base),base_sha256=nc.sha(self.base),executable=str(self.executable),devices=['CPU'],
            device_budget_bytes={'CPU':1<<30},reserve_bytes=1<<20,gpu_uuids=[],sequence_profile='dense-llama-short-v1',
            completion_recovery=dict(schema=1,source_closure={str(self.executable):nc.sha(self.executable)}))
        self.profile=None
        if optimizer!='sgd':
            self.profile=dict(schema=2,title='Controlled nonmodel recipe',mode=mode,rank=2 if mode=='lora' else None,
                optimizer=optimizer,learning_rate=.0001,optimizer_settings=dict(schema=1),state_policy='resume' if resume else 'fresh',training_budget_seconds=None)
            if factorwise:
                self.profile.update(schema=3,rank=rank,lora_update_space='factorwise-v1')
                if rank>256:self.profile.update(schema=4,lora_rank_profile='extended-4096-v1')
            self.profile_id=identity(self.profile);self.config['optimizer_profiles']={self.profile_id:self.profile}
            if resume:
                fresh=dict(self.profile,state_policy='fresh');self.config['optimizer_profiles'][identity(fresh)]=fresh
        self.runner=NativeRunner(self.config)
        self.service=LessonService(self.root/'service',MODEL,lambda text:[(ord(c)%7)+1 for c in text],self.runner,
                                    max_tokens=512,max_steps=32,sequence_profile='dense-llama-short-v1')
        self.lessons=[]
        for i in range(3):
            lesson=dict(schema=1,title='Controlled lesson '+str(i),model_contract_id=MODEL,prompt='Input '+str(i),completion='Answer '+str(i),
                rationale='Controlled fixture only',scope='owned-training',evidence=dict(method='host-verified-replay-v1',
                failed_result_sha256='a'*64,corrected_result_sha256='b'*64,verification_sha256=hashlib.sha256(str(i).encode()).hexdigest()))
            result=self.service.submit('/cow/training/lessons',dict(operation_id='lesson-'+str(i),lesson=lesson));self.lessons.append(result['lesson_id'])
        self.job=job
        self.request=dict(operation_id=self.job,lesson_ids=self.lessons,mode=mode,batch_size=2,steps=steps,
                          learning_rate=.0001,rank=rank,parent_job_id=None)
        if self.profile:self.request['optimizer_profile_id']=self.profile_id
        self.parent=None
        if parent is not None:
            # A same-host completed original parent; no training is simulated.
            self.parent=copy.deepcopy(parent)
            parent_dir=self.service.root/'runs/parent-job';parent_dir.mkdir(mode=0o700)
            for path_key,sha_key in [('candidate_path','candidate_sha256'),('optimizer_state_path','optimizer_state_sha256')]:
                if path_key not in self.parent:continue
                old=Path(self.parent[path_key]);new=parent_dir/old.name;new.write_bytes(old.read_bytes());new.chmod(0o600);self.parent[path_key]=str(new)
            self.parent['receipt']['candidate']=self.parent['candidate_path']
            if 'optimizer_checkpoint' in self.parent['receipt']:self.parent['receipt']['optimizer_checkpoint']['path']=self.parent['optimizer_state_path']
            self.request.update(parent_job_id='parent-job',expected_parent_sha256=self.parent['candidate_sha256'])
            prior_request=dict(self.request,operation_id='parent-job',parent_job_id=None)
            if resume:prior_request['optimizer_profile_id']=parent['optimizer_profile_id']
            self.service._write(self.service.root/'operations/parent-job.json',dict(path='/cow/training/jobs',request=prior_request,
                acceptance=dict(schema=1,operation_id='parent-job',request_sha256=digest(['/cow/training/jobs',prior_request]),job_id='parent-job'),
                state='complete',candidate=self.parent))
        self.folder=self.service.root/'runs'/self.job;self.folder.mkdir(mode=0o700)
        self.native=nc.native_request(self.runner.native_request(self.request,[self.service.lesson(i)['example'] for i in self.lessons],self.parent),self.folder)
        self.native['candidate']=str(self.folder/'candidate.gguf');Path(self.native['candidate']).write_bytes(b'CONTROLLED NONMODEL CANDIDATE');Path(self.native['candidate']).chmod(0o600)
        self.rows=[];cursor=self.native.get('example_cursor',0)
        for i in range(steps):
            count=min(2,3-cursor%3);cursor+=count
            self.rows.append(dict(step=i,before=1.0,after=.5,examples=count,example_cursor=cursor,gradients=[dict(name='blk.0.test',l2=1.0)]))
        self.raw=dict(status='MECHANICS_EXECUTED',native=True,backprop_mode='standard',mode=mode,optimizer=optimizer,
            steps=self.rows,candidate=self.native['candidate'],example_cursor=cursor,promoted=False,save_reload_qualified=False)
        if optimizer!='sgd':
            state_path=Path(self.native['optimizer_state_output']);previous_steps=0
            if resume:
                from tools.pcc_native_training import read_optimizer_state
                previous_steps=read_optimizer_state(self.native['initial_optimizer_state'],self.native['initial_optimizer_state_sha256'])['completed_steps']
            header=payload(state_path,nc.sha(self.native['candidate']),optimizer_dataset_sha256(self.native['examples']),cursor,previous_steps+steps,nc.optimizer_configuration(self.native))
            self.raw['optimizer_checkpoint']=dict(path=str(state_path),sha256=nc.sha(state_path),candidate_sha256=header['model_sha256'],dataset_sha256=header['dataset_sha256'],
                completed_steps=header['completed_steps'],example_cursor=cursor,config=header['config'])
        nc.write(self.folder/'request-private.json',self.native)
        ctx=nc.context(self.runner,self.request,self.native['examples'],self.parent)
        self.intent=nc.prepare(self.folder,ctx,self.native)
        launch=dict(executable=str(self.executable),executable_sha256=nc.sha(self.executable),request_sha256=nc.sha(self.folder/'request-private.json'),
            base_sha256=self.native['base_sha256'],parent_adapter_sha256=self.native.get('initial_adapter_sha256'),pid=GONE['pid'],process_identity=GONE,
            runtime_environment=self.intent['runtime_environment'])
        nc.write(self.folder/'launch.json',launch);nc.write(self.folder/'command.exit',0);nc.write(self.folder/'stdout-private.json',self.raw)
        self.reload_folder=self.folder/'completion-reload';self.reload_folder.mkdir(mode=0o700)
        expected=nc.reload_request(self.native,self.raw);nc.write(self.reload_folder/'request-private.json',expected)
        pins=nc.weight_pins(self.native);out=dict(native=True,status='FORWARD_EXECUTED',backprop_mode='standard',losses=[.5]*len(expected['examples']),promoted=False)
        nc.write(self.reload_folder/'stdout-private.json',out);nc.write(self.reload_folder/'command.exit',0)
        nc.write(self.reload_folder/'launch.json',dict(executable=str(self.executable),executable_sha256=nc.sha(self.executable),
            request_sha256=nc.sha(self.reload_folder/'request-private.json'),intent_sha256=self.intent['intent_sha256'],weights_sha256=pins,
            process_identity=GONE,runtime_environment=self.intent['runtime_environment']))
        gate=nc.check_reload(self.native,self.raw,expected,out,pins)
        gate.update(schema=1,kind='pcc.independent-save-reload.v1',intent_sha256=self.intent['intent_sha256'],request_sha256=nc.sha(self.reload_folder/'request-private.json'),
            stdout_sha256=nc.sha(self.reload_folder/'stdout-private.json'),native_stdout_sha256=nc.sha(self.folder/'stdout-private.json'),weights_sha256=pins,promoted=False)
        gate['gate_id']=nc.digest(gate);nc.write(self.folder/'completion-gate.json',gate)
        record=dict(path='/cow/training/jobs',request=self.request,acceptance=dict(schema=1,operation_id=self.job,job_id=self.job,
            request_sha256=digest(['/cow/training/jobs',self.request])),state='interrupted',private_error='Original commit lost: controlled fixture')
        self.service._write(self.service.root/'operations'/(self.job+'.json'),record)
        (self.service.root/'worker.lock').touch(mode=0o600)
        with self.service.locked():pass  # Real admission creates the existing record lock.

    def close(self):self.service.close()
    def inspect(self):
        # Strict internal oracle for saved-evidence negatives. The public driver
        # reports typed HOLD states instead of making clients parse exceptions.
        with nc.idle(self.service):return nc.inspect(self.service,self.job)
    def approval(self):
        value=self.inspect()
        return dict(operation_id='adopt-1',job_id=self.job,approval=nc.APPROVAL,
            expected_job_record_sha256=value['job_record_sha256'],expected_evidence_sha256=value['evidence_sha256'],expected_candidate_sha256=value['candidate_sha256'])
    def adopt(self,request=None):return self.service.submit(nc.ROUTE,self.approval() if request is None else request)


def _hold_inherited(fd,ready,release):
    ready.set();release.wait(10);os.close(fd)


def _crash_adoption(service,request,stage):
    original=service._write
    def write(path,value):
        original(path,value)
        if stage=='intent' and path.name==request['operation_id']+'.json' and value['state']=='recorded':os._exit(31)
        if stage=='job' and path.name==request['job_id']+'.json' and value['state']=='complete':os._exit(33)
        if stage=='ack' and path.name==request['operation_id']+'.json' and value['state']=='complete':os._exit(34)
    service._write=write
    if stage=='version':
        from tools import pcc_native_training
        finalize=pcc_native_training.finalize_candidate
        def publish(*args,**kwargs):
            result=finalize(*args,**kwargs);os._exit(32)
        pcc_native_training.finalize_candidate=publish
    service.submit(nc.ROUTE,request)
    os._exit(99)


class FakePolicy:
    def __init__(self,training):self.training=training
    def describe(self):return dict(schema=1,title='Controlled adoption round only',adapter_id='1'*64,training=self.training,max_rounds=3)
    def verify(self):pass
    def evaluate(self,*args):raise AssertionError('model calls forbidden')
    def validate_evaluation(self,*args):raise AssertionError('evaluation forbidden')
    def validate_lessons(self,*args):pass
    def analyze(self,*args):raise AssertionError('analysis must not be rerun')


class NativeCompletionTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='pcc-completion-nonmodel-');self.addCleanup(self.temp.cleanup)
        self.fixture=SavedEvidence(Path(self.temp.name)/'fixture');self.addCleanup(self.fixture.close)

    def forbidden(self):
        stack=ExitStack()
        for name in ('Popen','run','check_output'):
            stack.enter_context(mock.patch('subprocess.'+name,side_effect=AssertionError('dispatch forbidden')))
        stack.enter_context(mock.patch('tools.pcc_native_training.execute_native',side_effect=AssertionError('native forbidden')))
        stack.enter_context(mock.patch('tools.pcc_lessons.execute_native',side_effect=AssertionError('native forbidden')))
        return stack

    def rewrite(self,path,mutate):
        value=nc.read(path);mutate(value);path.write_bytes(nc.canonical(value)+b'\n')

    def learning_fixture(self,name):
        from tools.pcc_learning_runs import CREATE
        training={k:self.fixture.request[k] for k in ('mode','rank','batch_size','steps','learning_rate','optimizer_profile_id')}
        policy=FakePolicy(training);policy_id=digest(policy.describe())
        job='learning-job-'+digest(dict(run=name,round=1,policy=policy_id))
        f=SavedEvidence(Path(self.temp.name)/name,job=job);self.addCleanup(f.close)
        f.service.configure_learning({policy_id:policy})
        f.service.submit(CREATE,dict(operation_id=name,policy_id=policy_id,parent_job_id=None,expected_parent_sha256=None))
        run=f.service.learning._run(name);run.update(phase='needs_operator',revision=5,training_job_id=job,training_request=f.request,
            operator_reason_code='native_job_unresolved',operator_reason='Controlled original commit lost')
        f.service.learning._save(run)
        f.service.learning._write(f.service.learning._directory(name)/'round-000001.json',dict(schema=1,round=1,parent_job_id=None,
            lesson_ids=f.lessons,training_request=f.request))
        f.adopt()
        return f

    def test_owner_cli_completion_reconcile_lost_ack_reads_original_before_current_phase(self):
        import threading
        from tools import cow_cli
        from serve.cow_http import make_server
        f=self.learning_fixture('cli-recovery');server=make_server(object(),api_key='controlled-fixture-only',training_service=f.service)
        thread=threading.Thread(target=server.serve_forever);thread.start()
        base=cow_cli.CowClient('http://127.0.0.1:'+str(server.server_port),'controlled-fixture-only')
        calls=[];lost=[True];lost_get=[True]
        class Client:
            base_url=base.base_url
            def request(self,method,path,body=None):
                calls.append((method,path))
                if method=='GET' and '/operations/' in path and lost_get[0]:
                    lost_get[0]=False;raise cow_cli.ClientError('controlled lost first metadata status')
                value=base.request(method,path,body)
                if method=='POST' and lost[0]:lost[0]=False;raise cow_cli.ClientError('controlled lost metadata acknowledgment')
                return value
        journal=cow_cli.RequestJournal(Path(self.temp.name)/'learning-cli-journal')
        try:
            with self.forbidden():
                with self.assertRaisesRegex(cow_cli.ClientError,'unresolved'):
                    cow_cli._reconcile_completion(Client(),journal,'cli-recovery','step-original',5)
                self.assertEqual(journal.get('step-original')['state'],'pending')
                self.assertEqual(f.service.learning_run('cli-recovery')['phase'],'candidate_pending')
                first=cow_cli._reconcile_completion(Client(),journal,'cli-recovery','step-original',5)
                count=len(calls);again=cow_cli._reconcile_completion(Client(),journal,'cli-recovery','step-original',5)
                self.assertEqual(first,again);self.assertEqual(len(calls),count)
                with self.assertRaisesRegex(cow_cli.ClientError,'different bytes'):
                    cow_cli._reconcile_completion(Client(),journal,'cli-recovery','step-original',6)
            self.assertEqual([x for x in calls if x[0]=='POST'],[('POST','/cow/training/learning-steps')])
            run=f.service.learning._run('cli-recovery')
            self.assertEqual(run['completed_rounds'],0);self.assertEqual(run['training_job_id'],f.job)
            self.assertEqual(len(run['recovery_history']),1)
        finally:server.shutdown();server.server_close();thread.join(5)

    def test_owner_cli_wrong_phase_reason_revision_or_old_server_holds_without_post(self):
        from tools import cow_cli
        good=dict(schema=1,run_id='owned-run',revision=5,phase='needs_operator',operator_reason_code='native_job_unresolved',
                  available_actions=[dict(action='reconcile',required_fields=[])])
        variants=[dict(good,phase=x) for x in ['training','training_pending','candidate_pending']]
        variants += [dict(good,revision=6),dict(good,revision=True),dict(good,operator_reason_code='assessment_failed'),
                     dict(good,available_actions=None),dict(good,available_actions=['reconcile'])]
        old=dict(good);old.pop('operator_reason_code');variants.append(old)
        for i,status in enumerate(variants):
            class Client:
                base_url='http://controlled-owner-fixture'
                def request(self,method,path,body=None):
                    if method!='GET':raise AssertionError('metadata preflight must not POST')
                    return status
            with self.subTest(status=status),self.forbidden():
                journal=cow_cli.RequestJournal(Path(self.temp.name)/('hold-journal-%d'%i))
                with self.assertRaisesRegex(cow_cli.ClientError,'HOLD'):
                    cow_cli._reconcile_completion(Client(),journal,'owned-run','same-step',5)
                self.assertIsNone(journal.get('same-step'))

    def test_owner_cli_revision_race_cannot_turn_completion_reconcile_into_other_work(self):
        from tools import cow_cli
        f=self.learning_fixture('race-recovery');calls=[]
        class Client:
            base_url='http://controlled-owner-fixture'
            def request(self,method,path,body=None):
                calls.append((method,path))
                if method=='GET' and '/learning-runs/' in path:
                    status=f.service.learning_run('race-recovery')
                    run=f.service.learning._run('race-recovery');run['revision']+=1;f.service.learning._save(run)
                    return status
                if method=='POST':
                    try:return f.service.submit(path,body)
                    except ValueError as error:raise cow_cli.ExplicitRejection(str(error)) from error
                raise AssertionError('unexpected recovery request')
        with self.forbidden():
            journal=cow_cli.RequestJournal(Path(self.temp.name)/'race-journal')
            with self.assertRaisesRegex(cow_cli.ClientError,'revision changed'):
                cow_cli._reconcile_completion(Client(),journal,'race-recovery','race-step',5)
        run=f.service.learning._run('race-recovery');self.assertEqual(run['phase'],'needs_operator')
        self.assertEqual(run['revision'],6);self.assertEqual(run['completed_rounds'],0)

    def test_public_typed_states_and_historical_ack_are_not_current_eligibility(self):
        f=self.fixture
        with self.forbidden():
            value=f.service.completion_recovery(f.job)
            self.assertEqual(value['state'],'eligible');self.assertTrue(value['eligible'])
            self.assertTrue(value['current_evidence_verified'])
            self.assertEqual(value['next_actions'],['adopt_original_completion'])
            for key in ('native_configuration_sha256','source_closure_sha256','runtime_environment_sha256'):
                self.assertRegex(value[key],'^[a-f0-9]{64}$')
            request=f.approval();ack=f.adopt(request)
            value=f.service.completion_recovery(f.job)
            self.assertEqual(value['state'],'adopted');self.assertFalse(value['eligible'])
            self.assertEqual(value['next_actions'],['reconcile_learning_run'])
            (f.folder/'candidate.gguf').write_bytes(b'CHANGED AFTER HISTORICAL ADOPTION')
            self.assertEqual(f.adopt(request),ack)  # Ack is historical metadata.
            value=f.service.completion_recovery(f.job)
        self.assertEqual(value['state'],'ineligible');self.assertFalse(value['current_evidence_verified'])
        self.assertEqual(value['next_actions'],['hold']);self.assertNotIn('candidate_sha256',value)

    def test_public_pending_state_requires_same_original_metadata_operation(self):
        f=self.fixture;request=f.approval();original=f.service._write
        def interrupt(path,value):
            original(path,value)
            if path.name=='adopt-1.json' and value['state']=='recorded':raise OSError('controlled pending metadata intent')
        with self.forbidden():
            with mock.patch.object(f.service,'_write',side_effect=interrupt):
                with self.assertRaises(OSError):f.adopt(request)
            info=f.service.completion_recovery(f.job)
            self.assertEqual(info['state'],'adoption_pending');self.assertFalse(info['eligible'])
            self.assertTrue(info['current_evidence_verified'])
            self.assertEqual(info['pending_operation_id'],'adopt-1');self.assertEqual(info['pending_request'],request)
            self.assertEqual(info['next_actions'],['resume_original_adoption'])
            with self.assertRaisesRegex(ValueError,'original pending adoption'):f.adopt(dict(request,operation_id='different-approval'))
            self.assertFalse((f.service.root/'operations/different-approval.json').exists())
            f.adopt(request)
            self.assertEqual(f.service.completion_recovery(f.job)['state'],'adopted')

    def test_relevant_runtime_environment_drift_refuses_without_inventing_legacy_pins(self):
        f=self.fixture
        for key in ('GGML_CUDA_CUBLAS_COMPUTE_TYPE','GGML_CUDA_DISABLE_FUSION','COW_QWEN35_F32_ATTENTION','GGML_CPU_DISABLE_FUSION','LD_LIBRARY_PATH'):
            with self.subTest(key=key),self.forbidden(),mock.patch.dict(os.environ,{key:'changed-owner-runtime'}):
                with self.assertRaisesRegex(ValueError,'provenance differs'):f.inspect()
                result=f.service.completion_recovery(f.job)
                self.assertEqual(result['state'],'ineligible');self.assertFalse(result['current_evidence_verified'])
        with mock.patch.dict(os.environ,{'AWS_SECRET_ACCESS_KEY':'not-an-allowed-runtime-field'}):
            self.assertNotIn('AWS_SECRET_ACCESS_KEY',nc.runtime_values())
        config=copy.deepcopy(f.runner.config);config.pop('completion_recovery');f.runner.config=config
        result=f.service.completion_recovery(f.job)
        self.assertEqual(result['reason_code'],'legacy_provenance_missing');self.assertEqual(result['next_actions'],['hold'])

    def test_actual_reload_environment_drift_refuses_before_fake_child_dispatch(self):
        f=self.fixture;run=f.service.root/'runs/reload-environment-drift';run.mkdir(mode=0o700)
        environment=dict(os.environ,GGML_CUDA_CUBLAS_COMPUTE_TYPE='changed')
        with self.forbidden(),self.assertRaisesRegex(ValueError,'runtime environment differs'):
            nc.reload_gate(run,f.native,f.raw,f.intent,executable=f.executable,environment=environment,pass_fds=(),timeout=5)

    def test_actual_native_environment_drift_refuses_before_fake_child_dispatch(self):
        from tools.pcc_native_training import execute_native
        f=self.fixture;run=f.service.root/'runs/native-environment-drift'
        environment=dict(os.environ,GGML_CUDA_DISABLE_FUSION='changed')
        with self.forbidden(),self.assertRaisesRegex(ValueError,'before dispatch'):
            execute_native(f.executable,f.native,run,completion_context=f.intent,environment=environment)

    def test_existing_legacy_lock_modes_inside_private_store_are_not_rewritten(self):
        f=self.fixture;paths=[f.service.root/'records.lock',f.service.root/'worker.lock']
        for path in paths:path.chmod(0o644)
        before=[(p.stat().st_ino,p.stat().st_mode) for p in paths]
        with self.forbidden():self.assertEqual(f.service.completion_recovery(f.job)['state'],'eligible')
        self.assertEqual(before,[(p.stat().st_ino,p.stat().st_mode) for p in paths])

    def test_strict_saved_json_refuses_duplicate_nonfinite_and_excess_nesting(self):
        for raw in ('{"x":1,"x":2}','{"x":NaN}','{"x":1e999}'):
            with self.assertRaises(ValueError):nc.strict_json(raw)
        for config in (dict(schema=True,source_closure={str(self.fixture.executable):nc.sha(self.fixture.executable)}),
                       dict(schema=1,source_closure={str(self.fixture.executable):10**63})):
            with self.assertRaises(ValueError):nc.configuration(config)
        bad=self.fixture.folder/'malformed-object.json';nc.write(bad,[1])
        with self.assertRaisesRegex(ValueError,'JSON object'):nc.read(bad)
        with self.assertRaisesRegex(ValueError,'JSON object'):nc.read_version(bad)

    def corrupt_receipt_http_hold(self,raw):
        import threading
        from tools import cow_cli
        from serve.cow_http import make_server
        f=self.fixture;(f.folder/'stdout-private.json').write_bytes(nc.canonical(raw)+b'\n')
        server=make_server(object(),api_key='controlled-fixture-only',training_service=f.service)
        thread=threading.Thread(target=server.serve_forever);thread.start()
        client=cow_cli.CowClient('http://127.0.0.1:'+str(server.server_port),'controlled-fixture-only')
        try:
            with self.forbidden():value=client.request('GET','/cow/training/jobs/'+f.job+'/completion-recovery')
        finally:server.shutdown();server.server_close();thread.join(5)
        self.assertEqual(value['state'],'ineligible');self.assertEqual(value['reason_code'],'original_evidence_invalid')
        self.assertEqual(value['next_actions'],['hold']);self.assertFalse(value['current_evidence_verified'])
        self.assertEqual(value['native_jobs_started'],0);self.assertEqual(value['model_calls_started'],0)
        self.assertNotIn('candidate_sha256',value)
        self.assertEqual(f.service.job(f.job)['state'],'interrupted')
        self.assertFalse((f.service.root/'operations/adopt-1.json').exists())

    def test_malformed_step_and_comparable_nested_receipt_shapes_are_typed_http_hold(self):
        variants=[];raw=copy.deepcopy(self.fixture.raw);raw['steps']=[1];variants.append(raw)
        raw=copy.deepcopy(self.fixture.raw);raw['steps'][0]['timing']=1;variants.append(raw)
        raw=copy.deepcopy(self.fixture.raw);raw['optimizer_checkpoint']=[1];variants.append(raw)
        variants.append([1])
        for raw in variants:
            with self.subTest(shape=raw):self.corrupt_receipt_http_hold(raw)

    def test_malformed_gradient_rows_are_typed_http_hold_without_dispatch(self):
        raw=copy.deepcopy(self.fixture.raw);raw['steps'][0]['gradients']=[1]
        self.corrupt_receipt_http_hold(raw)

    def test_normal_service_committed_job_needs_no_completion_adoption(self):
        f=self.fixture;path=f.service.root/'operations'/(f.job+'.json');record=f.service._read(path)
        record.update(state='complete',candidate=nc.envelope(f.native,f.raw));f.service._write(path,record)
        config=copy.deepcopy(f.runner.config);config.pop('completion_recovery');f.runner.config=config
        with self.forbidden():value=f.service.completion_recovery(f.job)
        self.assertEqual(value['state'],'ineligible');self.assertEqual(value['reason_code'],'original_job_already_complete')
        self.assertEqual(value['next_actions'],['inspect_original_job','learning_status'])
        self.assertFalse(value['eligible']);self.assertFalse(value['current_evidence_verified'])
        self.assertNotIn('candidate_sha256',value);self.assertEqual(value['native_jobs_started'],0)
        self.assertFalse((f.service.root/'operations/adopt-1.json').exists())

    def test_original_complete_evidence_inspection_and_explicit_adoption_never_dispatch(self):
        f=self.fixture;before=nc.sha(f.folder/'candidate.gguf')
        with self.forbidden():
            value=f.inspect();request=f.approval();result=f.adopt(request);again=f.adopt(request)
        self.assertEqual(result,again);self.assertFalse(result['promoted']);self.assertTrue(result['adopted'])
        self.assertEqual(nc.sha(f.folder/'candidate.gguf'),before)
        record=f.service._read(f.service.root/'operations'/f'{f.job}.json')
        self.assertEqual(record['state'],'complete');self.assertIn('private_error',record)
        original=f.service._read(f.service.root/'operations/adopt-1.json')['original_job_record']
        self.assertEqual(original['state'],'interrupted');self.assertEqual(original['private_error'],record['private_error'])
        self.assertFalse(record['candidate']['receipt']['save_reload_qualified'])
        self.assertEqual(value['model_calls_started'],0)

    def test_full_muon_lora_adamw_and_plain_sgd_saved_fixtures(self):
        for mode,optimizer in [('full','muon'),('lora','adamw'),('full','sgd')]:
            with self.subTest(mode=mode,optimizer=optimizer):
                f=SavedEvidence(Path(self.temp.name)/(mode+optimizer),mode=mode,optimizer=optimizer)
                try:
                    with self.forbidden():self.assertTrue(f.adopt()['adopted'])
                finally:f.close()

    def test_final_batch_uses_last_completed_partial_batch_and_wraparound(self):
        f=self.fixture;reload=nc.reload_request(f.native,f.raw)
        self.assertEqual(reload['examples'],f.native['examples'][:2]) # cursor5,count2,start3 modulo3
        for key in ('optimizer','optimizer_settings','optimizer_state_mode','initial_optimizer_state','optimizer_state_output','example_cursor','steps','candidate','training_budget_seconds'):
            self.assertNotIn(key,reload)
        partial=SavedEvidence(Path(self.temp.name)/'partial',steps=2)
        try:self.assertEqual(nc.reload_request(partial.native,partial.raw)['examples'],[partial.native['examples'][2]])
        finally:partial.close()

    def test_factorwise_saved_evidence_recovery_and_forward_reload(self):
        for kind,rank in [('adamw',2),('muon',2),('muon',4096)]:
            with self.subTest(optimizer=kind,rank=rank):
                f=SavedEvidence(Path(self.temp.name)/('factorwise-'+kind+str(rank)),
                                mode='lora',optimizer=kind,factorwise=True,rank=rank)
                try:
                    self.assertEqual(nc.optimizer_configuration(f.native)['lora_update_space'],'factorwise-v1')
                    reload=nc.reload_request(f.native,f.raw)
                    self.assertNotIn('lora_update_space',reload)
                    if rank>256:self.assertEqual(reload['lora_rank_profile'],'extended-4096-v1')
                    with self.forbidden():self.assertTrue(f.adopt()['adopted'])
                finally:f.close()

    def test_missing_legacy_provenance_and_reload_gate_are_refused(self):
        f=self.fixture
        for name in ('completion-intent.json','completion-gate.json'):
            path=f.folder/name;raw=path.read_bytes();path.unlink()
            with self.assertRaises((ValueError,FileNotFoundError)):f.inspect()
            path.write_bytes(raw);path.chmod(0o600)
        self.assertEqual(f.service.job(f.job)['state'],'interrupted')

    def test_live_original_process_blocks_even_valid_output(self):
        f=self.fixture;live=nc.process_identity(os.getpid())
        self.rewrite(f.folder/'launch.json',lambda x:x.update(pid=live['pid'],process_identity=live))
        with self.assertRaisesRegex(ValueError,'still present'):f.inspect()

    def test_busy_worker_lock_and_queued_competing_job_are_refused(self):
        f=self.fixture
        with (f.service.root/'worker.lock').open('r+b') as stream:
            fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
            with self.assertRaisesRegex(ValueError,'worker'):f.inspect()
        req=dict(f.request,operation_id='competing')
        f.service._write(f.service.root/'operations/competing.json',dict(path='/cow/training/jobs',request=req,
            acceptance=dict(schema=1,operation_id='competing',job_id='competing',request_sha256=digest(['/cow/training/jobs',req])),state='queued'))
        with self.assertRaisesRegex(ValueError,'competing work'):f.inspect()

    def test_owner_approval_and_all_cas_digests_are_required(self):
        f=self.fixture;valid=f.approval()
        for key in ('approval','expected_job_record_sha256','expected_evidence_sha256','expected_candidate_sha256'):
            request=dict(valid);request[key]='0'*64
            with self.assertRaises(ValueError):f.adopt(request)
        self.assertFalse((f.service.root/'operations/adopt-1.json').exists())
        self.assertEqual(f.service.job(f.job)['state'],'interrupted')

    def test_changed_duplicate_owner_approval_cannot_retarget_existing_operation(self):
        f=self.fixture;valid=f.approval();f.adopt(valid)
        changed=dict(valid,expected_candidate_sha256='0'*64)
        with self.assertRaisesRegex(ValueError,'different original approval'):f.adopt(changed)

    def test_native_request_source_profile_and_stdout_mutations_refuse_without_commit(self):
        f=self.fixture
        tests=[(f.folder/'request-private.json',lambda x:x.update(learning_rate=.002)),
            (f.folder/'completion-intent.json',lambda x:x['runner_config'].update(reserve_bytes=1)),
            (f.folder/'launch.json',lambda x:x.update(executable_sha256='0'*64)),
            (f.folder/'stdout-private.json',lambda x:x.update(mode='lora')),
            (f.reload_folder/'launch.json',lambda x:x.update(runtime_environment={})),
            (f.folder/'stdout-private.json',lambda x:x['steps'][-1].update(example_cursor=99)),
            (f.folder/'stdout-private.json',lambda x:x['steps'][0]['gradients'][0].update(l2=-1))]
        for path,mutate in tests:
            with self.subTest(path=path.name):
                raw=path.read_bytes();self.rewrite(path,mutate)
                with self.assertRaises(ValueError):f.inspect()
                path.write_bytes(raw)
                self.assertEqual(f.service.job(f.job)['state'],'interrupted')

    def test_failed_unknown_cancelled_and_boolean_exit_are_not_complete(self):
        f=self.fixture;path=f.folder/'command.exit';original=path.read_bytes()
        for value in (1,False,'0',None):
            path.write_bytes(nc.canonical(value)+b'\n')
            with self.assertRaises(ValueError):f.inspect()
        path.write_bytes(original)
        (f.folder/'terminal-status.txt').write_text('TIMEOUT')
        with self.assertRaisesRegex(ValueError,'uncertain'):f.inspect()

    def test_reload_numeric_and_immutability_failures_prevent_adoption(self):
        f=self.fixture
        self.rewrite(f.reload_folder/'stdout-private.json',lambda x:x.update(losses=[.501,.501]))
        with self.assertRaisesRegex(ValueError,'numeric bound'):f.inspect()
        f.reload_folder.joinpath('stdout-private.json').write_bytes(nc.canonical(dict(native=True,status='FORWARD_EXECUTED',backprop_mode='standard',losses=[.5,.5],promoted=False))+b'\n')
        candidate=Path(f.native['candidate']);candidate.write_bytes(b'CHANGED NONMODEL BYTES')
        with self.assertRaises(ValueError):f.inspect()

    def test_conflicting_original_version_is_not_overwritten(self):
        f=self.fixture;path=f.folder/'versions';path.mkdir(mode=0o700)
        nc.write(path/(nc.sha(f.native['candidate'])+'.json'),dict(wrong=True))
        with self.assertRaisesRegex(ValueError,'Conflicting original'):f.inspect()

    def test_exact_resumed_cursor_and_same_mode_sgd_parent_are_supported(self):
        original=nc.inspect(self.fixture.service,self.fixture.job)['candidate']
        resumed=SavedEvidence(Path(self.temp.name)/'resumed',parent=original,resume=True)
        try:
            with self.forbidden():self.assertTrue(resumed.adopt()['adopted'])
            self.assertEqual(resumed.raw['example_cursor'],9)
            self.assertEqual(nc.reload_request(resumed.native,resumed.raw)['examples'],[resumed.native['examples'][2]])
        finally:resumed.close()
        sgd=SavedEvidence(Path(self.temp.name)/'sgd-parent',optimizer='sgd')
        try:
            candidate=nc.inspect(sgd.service,sgd.job)['candidate']
            child=SavedEvidence(Path(self.temp.name)/'after-sgd',parent=candidate)
            try:
                with self.forbidden():self.assertTrue(child.adopt()['adopted'])
            finally:child.close()
        finally:sgd.close()

    def test_original_timed_receipt_uses_actual_completed_updates(self):
        f=self.fixture;request=copy.deepcopy(f.native);request.update(training_budget_seconds=.1,steps=3)
        raw=copy.deepcopy(f.raw);raw['steps']=raw['steps'][:1];raw['example_cursor']=2
        raw['steps'][0]['timing']=dict(seconds=.15,gradient_and_coverage_seconds=.05,optimizer_seconds=.05,post_update_evaluation_seconds=.05)
        raw['training_budget']=dict(schema=1,boundary='whole_update',clock='steady_training_phase',requested_seconds=.1,
            step_cap=3,completed_updates=1,elapsed_seconds=.15,overrun_seconds=.05,remaining_seconds=0,stop_reason='training_budget')
        state_path=Path(request['optimizer_state_output']);header=payload(state_path,nc.sha(request['candidate']),optimizer_dataset_sha256(request['examples']),2,1,nc.optimizer_configuration(request))
        raw['optimizer_checkpoint'].update(completed_steps=1,example_cursor=2,sha256=nc.sha(state_path))
        with self.forbidden():result=nc.reload_request(request,raw)
        self.assertEqual(result['examples'],request['examples'][:2]);self.assertNotIn('training_budget_seconds',result)

    def test_inherited_worker_lock_survives_parent_descriptor_close(self):
        import multiprocessing as mp
        context=mp.get_context('fork');ready=context.Event();release=context.Event();f=self.fixture
        with (f.service.root/'worker.lock').open('r+b') as stream:
            fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
            child=context.Process(target=_hold_inherited,args=(stream.fileno(),ready,release));child.start()
            self.assertTrue(ready.wait(5))
        try:
            with self.assertRaisesRegex(ValueError,'worker'):f.inspect()
        finally:release.set();child.join(5)
        self.assertEqual(child.exitcode,0);self.assertEqual(f.inspect()['job_id'],f.job)

    def test_process_crash_at_each_commit_boundary_reconciles_without_dispatch(self):
        import multiprocessing as mp
        context=mp.get_context('fork')
        for stage,code in [('intent',31),('version',32),('job',33),('ack',34)]:
            with self.subTest(stage=stage):
                f=SavedEvidence(Path(self.temp.name)/('crash-'+stage));request=f.approval()
                try:
                    child=context.Process(target=_crash_adoption,args=(f.service,request,stage));child.start();child.join(10)
                    if child.is_alive():child.terminate();child.join(5);self.fail('crash fixture did not terminate')
                    self.assertEqual(child.exitcode,code)
                    with self.forbidden():
                        info=f.service.completion_recovery(f.job)
                        self.assertTrue(info['current_evidence_verified'])
                        if stage=='ack':self.assertEqual(info['state'],'adopted')
                        else:
                            self.assertEqual(info['state'],'adoption_pending')
                            self.assertEqual(info['pending_request'],request)
                        first=f.adopt(request);again=f.adopt(request)
                    self.assertEqual(first,again);self.assertEqual(f.service.job(f.job)['state'],'complete')
                    self.assertEqual(len(list((f.folder/'versions').glob('*.json'))),1)
                finally:f.close()

    def test_authenticated_http_and_cow_cli_use_the_existing_request_journal(self):
        import threading
        from urllib.error import HTTPError
        from tools import cow_cli
        from serve.cow_http import make_server
        f=self.fixture;server=make_server(object(),api_key='controlled-fixture-only',training_service=f.service)
        thread=threading.Thread(target=server.serve_forever);thread.start()
        try:
            url='http://127.0.0.1:'+str(server.server_port)
            client=cow_cli.CowClient(url,'controlled-fixture-only')
            from urllib.request import Request,urlopen
            with self.assertRaises(HTTPError) as error:
                urlopen(Request(url+'/cow/training/jobs/'+f.job+'/completion-recovery',headers={'Authorization':'Bearer wrong-fixture-key'}))
            self.assertEqual(error.exception.code,401);error.exception.close()
            with self.forbidden():
                info=client.request('GET','/cow/training/jobs/'+f.job+'/completion-recovery')
                request=f.approval();journal=cow_cli.RequestJournal(Path(self.temp.name)/'cli-journal')
                first=cow_cli._training_post(client,journal,nc.ROUTE,request)
                again=cow_cli._training_post(client,journal,nc.ROUTE,request)
            self.assertEqual(first,again);self.assertEqual(info['candidate_sha256'],first['candidate_sha256'])
            args=cow_cli.parser().parse_args(['lesson','adopt-completion',f.job,'--operation-id','adopt-1',
                '--expected-job-record-sha256',request['expected_job_record_sha256'],'--expected-evidence-sha256',request['expected_evidence_sha256'],
                '--expected-candidate-sha256',request['expected_candidate_sha256'],'--approve-original-completion'])
            self.assertEqual(args.lesson_action,'adopt-completion')
        finally:server.shutdown();server.server_close();thread.join(5)

    def test_learning_round_reconciles_same_adopted_job_without_training_or_evaluation(self):
        from tools.pcc_learning_runs import CREATE,STEP
        training={k:self.fixture.request[k] for k in ('mode','rank','batch_size','steps','learning_rate','optimizer_profile_id')}
        policy=FakePolicy(training);policy_id=digest(policy.describe());run_id='learning-adoption'
        job='learning-job-'+digest(dict(run=run_id,round=1,policy=policy_id))
        f=SavedEvidence(Path(self.temp.name)/'round',job=job)
        try:
            f.service.configure_learning({policy_id:policy})
            f.service.submit(CREATE,dict(operation_id=run_id,policy_id=policy_id,parent_job_id=None,expected_parent_sha256=None))
            run=f.service.learning._run(run_id);run.update(phase='needs_operator',revision=5,training_job_id=job,training_request=f.request,
                operator_reason_code='native_job_unresolved',operator_reason='Controlled original commit lost')
            f.service.learning._save(run)
            f.service.learning._write(f.service.learning._directory(run_id)/'round-000001.json',dict(schema=1,round=1,parent_job_id=None,
                lesson_ids=f.lessons,training_request=f.request))
            self.assertEqual(f.service.learning_run(run_id)['available_actions'],[dict(action='stop',required_fields=[])])
            with self.forbidden():
                f.adopt()
                candidate=f.folder/'candidate.gguf';source=f.executable;stdout=f.folder/'stdout-private.json'
                job_path=f.service.root/'operations'/(job+'.json')
                original_candidate=candidate.read_bytes();original_source=source.read_bytes();original_stdout=stdout.read_bytes()
                original_config=copy.deepcopy(f.runner.config);original_job=f.service._read(job_path)
                for kind in ('candidate','source','config','completion','job-envelope','adoption-operation'):
                    if kind=='candidate':candidate.write_bytes(b'CHANGED ADOPTED CANDIDATE')
                    elif kind=='source':source.write_bytes(b'CHANGED NATIVE SOURCE IDENTITY')
                    elif kind=='config':f.runner.config['reserve_bytes']+=1
                    elif kind=='completion':
                        changed=copy.deepcopy(f.raw);changed['steps'][-1]['after']=.51;stdout.write_bytes(nc.canonical(changed))
                    elif kind=='job-envelope':
                        changed=copy.deepcopy(original_job);changed['candidate']['candidate_sha256']='0'*64;f.service._write(job_path,changed)
                    else:
                        changed=copy.deepcopy(original_job);changed['completion_adoption']['operation_id']=1;f.service._write(job_path,changed)
                    with self.subTest(changed=kind):
                        self.assertNotIn('reconcile',[x['action'] for x in f.service.learning_run(run_id)['available_actions']])
                        with self.assertRaises(ValueError):
                            f.service.submit(STEP,dict(operation_id='refuse-'+kind,run_id=run_id,expected_revision=5,action='reconcile'))
                        self.assertEqual(f.service.learning._run(run_id)['phase'],'needs_operator')
                    candidate.write_bytes(original_candidate);source.write_bytes(original_source);stdout.write_bytes(original_stdout)
                    f.runner.config=copy.deepcopy(original_config);f.service._write(job_path,original_job)
                status=f.service.learning_run(run_id)
                self.assertIn('reconcile',[x['action'] for x in status['available_actions']])
                request=dict(operation_id='recover-round',run_id=run_id,expected_revision=5,action='reconcile')
                first=f.service.submit(STEP,request);again=f.service.submit(STEP,request)
            self.assertEqual(first,again);self.assertEqual(f.service.learning_run(run_id)['phase'],'candidate_pending')
            result=f.service.learning._run(run_id)
            self.assertEqual(result['training_job_id'],job);self.assertEqual(result['recovery_history'][0]['original_reason_code'],'native_job_unresolved')
        finally:f.close()

    def test_normal_opt_in_execution_records_separate_reload_gate_with_fake_child_only(self):
        from tools import pcc_native_training
        f=self.fixture;calls=[];last={}
        class NonmodelChild:
            def __init__(self,argv,stdout,stderr,**kwargs):
                calls.append((argv,kwargs));request=nc.read(argv[2]);self.pid=GONE['pid']
                if request.get('evaluate_only'):
                    out=dict(native=True,status='FORWARD_EXECUTED',backprop_mode='standard',losses=[.5]*len(request['examples']),promoted=False)
                else:
                    path=Path(request['candidate']);path.write_bytes(b'NEW CONTROLLED NONMODEL CANDIDATE');path.chmod(0o600)
                    rows=[];cursor=0
                    for i in range(request['steps']):
                        count=min(request['batch_size'],len(request['examples'])-cursor%len(request['examples']));cursor+=count
                        rows.append(dict(step=i,before=1.,after=.5,examples=count,example_cursor=cursor,gradients=[dict(name='test',l2=1.)]))
                    state_path=Path(request['optimizer_state_output']);header=payload(state_path,nc.sha(path),optimizer_dataset_sha256(request['examples']),cursor,request['steps'],nc.optimizer_configuration(request))
                    out=dict(native=True,status='MECHANICS_EXECUTED',backprop_mode='standard',mode=request['mode'],optimizer=request['optimizer'],
                        candidate=str(path),steps=rows,example_cursor=cursor,promoted=False,save_reload_qualified=False,
                        optimizer_checkpoint=dict(path=str(state_path),sha256=nc.sha(state_path),candidate_sha256=nc.sha(path),dataset_sha256=header['dataset_sha256'],
                            completed_steps=request['steps'],example_cursor=cursor,config=header['config']))
                stdout.write(json.dumps(out));stdout.flush()
            def wait(self,timeout):return 0
        run=f.service.root/'runs/fake-normal-path'
        with mock.patch('subprocess.Popen',NonmodelChild),mock.patch.object(nc,'process_identity',return_value=GONE):
            with (f.service.root/'worker.lock').open('r+b') as lease:
                candidate=f.runner(f.request,f.native['examples'],run,None,lease.fileno())
        self.assertEqual(len(calls),2)
        self.assertNotIn('evaluate_only',nc.read(calls[0][0][2]));self.assertTrue(nc.read(calls[1][0][2])['evaluate_only'])
        self.assertEqual(calls[0][1]['env'],calls[1][1]['env']);self.assertEqual(calls[0][1]['pass_fds'],calls[1][1]['pass_fds'])
        self.assertFalse(candidate['receipt']['save_reload_qualified']);self.assertIn('completion_recovery',candidate)
        self.assertTrue(nc.read(run/'completion-gate.json')['passed'])

    def test_changed_evidence_after_owner_inspection_fails_cas_without_commit(self):
        f=self.fixture;request=f.approval();path=f.service.root/'operations'/f'{f.job}.json'
        record=f.service._read(path);record['private_error']='A different original interruption';f.service._write(path,record)
        with self.assertRaisesRegex(ValueError,'changed after owner inspection'):f.adopt(request)
        self.assertFalse((f.service.root/'operations/adopt-1.json').exists())
        self.assertEqual(f.service.job(f.job)['state'],'interrupted')

    def test_changed_registered_profile_lesson_and_source_closure_are_not_rebound(self):
        f=self.fixture;profile=f.runner.registered_optimizer_profiles[f.profile_id]
        original=copy.deepcopy(profile);profile['learning_rate']=.002
        with self.assertRaisesRegex(ValueError,'changed owner optimizer'):f.inspect()
        f.runner.registered_optimizer_profiles[f.profile_id]=original
        path=f.service.root/'lessons'/(f.lessons[0]+'.json');raw=path.read_bytes();record=nc.read(path)
        record['example']['tokens'][0]+=1;f.service._write(path,record)
        with self.assertRaises(ValueError):f.inspect()
        path.write_bytes(raw)
        f.executable.write_bytes(b'CHANGED NONMODEL EXECUTABLE IDENTITY')
        with self.assertRaisesRegex(ValueError,'source closure differs'):f.inspect()

    def test_optimizer_body_and_config_corruption_are_refused_even_with_updated_sha(self):
        f=self.fixture;state_path=Path(f.native['optimizer_state_output']);original=state_path.read_bytes();raw_original=copy.deepcopy(f.raw)
        for kind in ('nonfinite','config','dataset'):
            with self.subTest(kind=kind):
                length=int.from_bytes(original[8:16],'little');header=json.loads(original[16:16+length]);body=original[16+length:]
                raw=copy.deepcopy(raw_original)
                if kind=='nonfinite':body=struct.pack('<f',float('nan'))
                elif kind=='config':header['config']['beta1']=.1;raw['optimizer_checkpoint']['config']=header['config']
                else:header['dataset_sha256']='0'*64;raw['optimizer_checkpoint']['dataset_sha256']='0'*64
                encoded=json.dumps(header).encode();state_path.write_bytes(b'PCCOPT1\n'+len(encoded).to_bytes(8,'little')+encoded+body)
                raw['optimizer_checkpoint']['sha256']=nc.sha(state_path)
                (f.folder/'stdout-private.json').write_bytes(nc.canonical(raw)+b'\n')
                with self.assertRaisesRegex(ValueError,'nonfinite|configuration|dataset'):f.inspect()
        state_path.write_bytes(original);(f.folder/'stdout-private.json').write_bytes(nc.canonical(raw_original)+b'\n')

    def test_looser_reload_gate_cannot_replace_original_bound(self):
        f=self.fixture;path=f.folder/'completion-gate.json';gate=nc.read(path);gate['absolute_bound']=.011
        gate['gate_id']=nc.digest({k:v for k,v in gate.items() if k!='gate_id'});path.write_bytes(nc.canonical(gate)+b'\n')
        with self.assertRaisesRegex(ValueError,'reload gate differs'):f.inspect()

    def test_unrelated_needs_operator_failure_and_stale_round_cannot_be_recovered(self):
        from tools.pcc_learning_runs import CREATE,STEP
        training={k:self.fixture.request[k] for k in ('mode','rank','batch_size','steps','learning_rate','optimizer_profile_id')}
        policy=FakePolicy(training);policy_id=digest(policy.describe());run_id='refused-round'
        job='learning-job-'+digest(dict(run=run_id,round=1,policy=policy_id));f=SavedEvidence(Path(self.temp.name)/'wrong-round',job=job)
        try:
            f.service.configure_learning({policy_id:policy});f.service.submit(CREATE,dict(operation_id=run_id,policy_id=policy_id,parent_job_id=None,expected_parent_sha256=None))
            run=f.service.learning._run(run_id);run.update(phase='needs_operator',revision=7,training_job_id=job,training_request=f.request,
                operator_reason_code='native_job_unresolved',operator_reason='Controlled interruption')
            f.service.learning._save(run);f.service.learning._write(f.service.learning._directory(run_id)/'round-000001.json',
                dict(schema=1,round=1,parent_job_id=None,lesson_ids=f.lessons,training_request=f.request))
            f.adopt()
            with self.forbidden():
                with self.assertRaisesRegex(ValueError,'revision changed'):
                    f.service.submit(STEP,dict(operation_id='stale-reconcile',run_id=run_id,expected_revision=6,action='reconcile'))
                run['operator_reason_code']='assessment_failed';f.service.learning._save(run)
                with self.assertRaisesRegex(ValueError,'native interruption'):
                    f.service.submit(STEP,dict(operation_id='wrong-reconcile',run_id=run_id,expected_revision=7,action='reconcile'))
                run['operator_reason_code']='native_job_unresolved';run['training_request']=dict(run['training_request'],rank=3);f.service.learning._save(run)
                with self.assertRaisesRegex(ValueError,'learning recipe'):
                    f.service.submit(STEP,dict(operation_id='wrong-recipe',run_id=run_id,expected_revision=7,action='reconcile'))
            self.assertEqual(f.service.learning_run(run_id)['phase'],'needs_operator')
        finally:f.close()

    def test_pending_adoption_intent_is_not_a_complete_ack_and_cli_replays_only_its_metadata(self):
        from tools import cow_cli
        f=self.fixture;approval=f.approval();journal=cow_cli.RequestJournal(Path(self.temp.name)/'pending-journal')
        calls=[];original_write=f.service._write;first=[True]
        def interrupt(path,value):
            original_write(path,value)
            if first[0] and path.name=='adopt-1.json' and value['state']=='recorded':
                first[0]=False;raise OSError('Controlled lost commit after approval intent')
        class Client:
            base_url='http://controlled-owner-fixture'
            def request(self,method,path,body=None):
                calls.append((method,path))
                try:
                    if method=='POST':return f.service.submit(path,body)
                    return f.service.operation(path.rsplit('/',1)[1])
                except (OSError,ValueError) as error:raise cow_cli.ClientError(str(error)) from error
        with self.forbidden(),mock.patch.object(f.service,'_write',side_effect=interrupt):
            with self.assertRaises(cow_cli.ClientError):cow_cli._training_post(Client(),journal,nc.ROUTE,approval)
            self.assertEqual(journal.get('adopt-1')['state'],'pending')
            self.assertEqual(f.service.job(f.job)['state'],'interrupted')
            with self.assertRaisesRegex(ValueError,'adoption is pending'):f.service.operation('adopt-1')
            complete=cow_cli._training_post(Client(),journal,nc.ROUTE,approval)
        self.assertTrue(complete['adopted']);self.assertEqual(journal.get('adopt-1')['state'],'complete')
        self.assertEqual([x for x in calls if x[0]=='POST'],[('POST',nc.ROUTE),('POST',nc.ROUTE)])
        self.assertEqual(f.service.job(f.job)['state'],'complete')

    def test_existing_record_lock_is_nonblocking_and_inspection_creates_no_lock_files(self):
        f=self.fixture;path=f.service.root/'records.lock';before=path.read_bytes()
        with path.open('r+b') as stream:
            fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
            with self.assertRaisesRegex(ValueError,'admission is busy'):f.inspect()
        self.assertEqual(path.read_bytes(),before)
        path.unlink()
        with self.assertRaisesRegex(ValueError,'lock is missing'):f.inspect()
        self.assertFalse(path.exists())

    def test_disabled_recovery_keeps_single_legacy_native_call_and_request(self):
        f=self.fixture;config=copy.deepcopy(f.config);config.pop('completion_recovery');runner=NativeRunner(config)
        result=dict(mode='full',candidate_sha256='c'*64,receipt={})
        with mock.patch('tools.pcc_lessons.execute_native',return_value=result) as call:
            with (f.service.root/'worker.lock').open('r+b') as lease:
                runner(f.request,f.native['examples'],f.service.root/'runs/legacy-check',None,lease.fileno())
        self.assertEqual(call.call_count,1);self.assertNotIn('completion_context',call.call_args.kwargs)
        request=call.call_args.args[1];self.assertNotIn('record_process_identity',request)
        self.assertEqual(request['steps'],f.request['steps']);self.assertEqual(request['examples'],f.native['examples'])

    def test_original_optimizer_header_payload_dataset_and_configuration_are_checked(self):
        f=self.fixture;path=Path(f.native['optimizer_state_output']);raw=path.read_bytes()
        for mutation in (raw[:-1],raw[:-4]+struct.pack('<f',float('nan'))):
            path.write_bytes(mutation)
            with self.assertRaises(ValueError):f.inspect()
        path.write_bytes(raw)
        cp=f.raw['optimizer_checkpoint'];state_config=cp['config'];self.assertEqual(state_config,nc.optimizer_configuration(f.native))


if __name__=='__main__':unittest.main()
