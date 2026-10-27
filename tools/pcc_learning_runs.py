"""Opt-in durable rounds over the existing PCC operation and native-job store.

Owner-installed policies own evaluation and analysis. Clients select registered
IDs and admitted lessons; they never submit scores, paths or checker code.
"""
from pathlib import Path
import copy, fcntl, os
from tools.pcc_lessons import canonical, digest, OP, HEX

CREATE='/cow/training/learning-runs'
STEP='/cow/training/learning-steps'


class LearningRuns:
    def __init__(self, service, policies):
        self.service=service;self.root=service.root/'learning';self.policies=dict(policies)
        if self.root.is_symlink():raise ValueError('Learning store link rejected')
        self.root.mkdir(mode=0o700,exist_ok=True)
        for key,policy in self.policies.items():
            descriptor=policy.describe()
            required={'schema','title','adapter_id','training','max_rounds'}
            if (not required<=set(descriptor) or set(descriptor)-required-{'assessment'} or type(descriptor['schema']) is not int or descriptor['schema']!=1 or digest(descriptor)!=key
                    or type(descriptor['max_rounds']) is not int or not 1<=descriptor['max_rounds']<=100000
                    or type(descriptor['title']) is not str or not 1<=len(descriptor['title'])<=160
                    or not HEX.fullmatch(str(descriptor['adapter_id']))):
                raise ValueError('Content-addressed owner learning policy required')
            training=descriptor['training']
            if set(training)!={'mode','rank','batch_size','steps','learning_rate','optimizer_profile_id'}:
                raise ValueError('Explicit registered training recipe required')
            profile=service.runner.validate_optimizer_request(dict(training,parent_job_id=None))
            rank_limit=4096 if profile and profile['schema']==4 else 256
            if any(type(training[k]) is not int or not 1<=training[k]<=maximum
                    for k,maximum in [('rank',rank_limit),('batch_size',128),('steps',service.max_steps)]):
                raise ValueError('Learning policy exceeds owner native step/batch/rank limits')
            profile=service.runner.validate_optimizer_request(dict(training,parent_job_id=None))
            if profile is None or profile['state_policy']!='fresh':
                raise ValueError('Learning rounds currently require explicit fresh-state policies')
            for name in ('evaluate','validate_evaluation','validate_lessons','analyze','verify'):
                if not callable(getattr(policy,name,None)):raise ValueError('Incomplete owner learning adapter')
            policy.verify()
        # Finish only effect-free creation transactions. Completed runs are never
        # reset to round zero if a later run record is missing or damaged.
        for path in (service.root/'operations').glob('*.json'):
            record=service._read(path)
            if record['path']==CREATE and record['state']=='recorded' and 'creation_record' in record:
                self._finish_creation(record,path)

    def _finish_creation(self,record,path):
        identity=record['request']['operation_id'];directory=self._directory(identity)
        directory.mkdir(mode=0o700,exist_ok=True)
        if not (directory/'run.json').exists():
            if record['state']!='recorded' or 'creation_record' not in record:
                raise ValueError('Committed learning run is missing; never recreate its history')
            initial=record['creation_record']
            if initial['creation_request']!=record['request'] or initial['revision']!=0 or initial['phase']!='baseline_pending':
                raise ValueError('Invalid original learning creation intent')
            self._save(initial)
        elif self._run(identity)['creation_request']!=record['request']:
            raise ValueError('Learning creation differs')
        record['state']='complete';self.service._write(path,record)

    def _policy(self,key):
        if key not in self.policies or digest(self.policies[key].describe())!=key:
            raise ValueError('Unknown or changed learning policy')
        return self.policies[key]

    def _directory(self,identity):
        if type(identity) is not str or not OP.fullmatch(identity):raise ValueError('Invalid learning run ID')
        if self.root.is_symlink():raise ValueError('Learning store link rejected')
        path=self.root/identity
        if path.is_symlink():raise ValueError('Learning run link rejected')
        return path

    def _read(self,path):
        value=self.service._read(path)
        if value.get('record_sha256')!=digest({k:v for k,v in value.items() if k!='record_sha256'}):
            raise ValueError('Learning record checksum differs')
        return {k:v for k,v in value.items() if k!='record_sha256'}

    def _write(self,path,value):
        value=copy.deepcopy(value);value['record_sha256']=digest(value)
        self.service._write(path,value)

    def _run(self,identity):return self._read(self._directory(identity)/'run.json')
    def _save(self,run):self._write(self._directory(run['run_id'])/'run.json',run)

    def _parent(self,identity,mode,expected=None):
        if identity is None:return None
        if type(identity) is not str or not OP.fullmatch(identity):raise ValueError('Invalid registered parent job')
        record=self.service._read(self.service.root/'operations'/(identity+'.json'))
        if (record['path']!='/cow/training/jobs' or record['state']!='complete'
                or record['candidate']['mode']!=mode):raise ValueError('Completed same-mode parent required')
        if expected is not None and record['candidate']['candidate_sha256']!=expected:
            raise ValueError('Learning parent SHA differs')
        return record['candidate']

    def capabilities(self):
        return dict(schema=1,model_contract_id=self.service.model,policies=[dict(id=key,**copy.deepcopy(self._policy(key).describe()))
            for key in sorted(self.policies)],promotion=False)

    def status(self,identity):
        with self.service.locked():
            run=self._run(identity)
            result={k:copy.deepcopy(run[k]) for k in ('run_id','policy_id','phase','revision','round','completed_rounds','working_parent_job_id')}
            policy_active=run['policy_id'] in self.policies and digest(self.policies[run['policy_id']].describe())==run['policy_id']
            result.update(schema=1,model_contract_id=self.service.model,promoted=False,policy_available=policy_active)
            phase=run['phase'];actions=[]
            if phase in ('baseline_pending','parent_pending','candidate_pending','report_pending'):
                actions=[dict(action='advance',required_fields=[]),dict(action='stop',required_fields=[])]
            elif phase=='choosing_lessons':actions=[dict(action='advance',required_fields=['lesson_ids']),dict(action='stop',required_fields=[])]
            elif phase in ('training_pending','training','preservation_pending'):actions=[dict(action='reconcile',required_fields=[])]
            elif phase.endswith('_running'):
                pending=self.service._read(self.service.root/'operations'/(run['pending_step_id']+'.json'))
                if pending['state'] not in ('queued','running'):actions=[dict(action='reconcile',required_fields=[])]
            elif phase=='round_complete' and policy_active:
                descriptor=self._policy(run['policy_id']).describe()
                if run['completed_rounds']<descriptor['max_rounds']:actions.append(dict(action='advance',required_fields=[]))
                if descriptor['training']['mode']=='full' and run['working_parent_job_id'] is not None:
                    actions.append(dict(action='preserve',required_fields=['name','notes']))
                actions.append(dict(action='stop',required_fields=[]))
            elif phase=='needs_operator':
                actions=[dict(action='stop',required_fields=[])]
                if run.get('operator_reason_code')=='native_job_unresolved':
                    try:self._adopted_native(run);actions.insert(0,dict(action='reconcile',required_fields=[]))
                    except (ValueError,FileNotFoundError,KeyError):pass
            result['available_actions']=actions if policy_active else []

            for key in ('training_job_id','last_report','last_assessment','operator_reason','operator_reason_code','preserved_export'):
                if key in run:result[key]=copy.deepcopy(run[key])
            job_id=run.get('training_job_id')
        if job_id and (self.service.root/'operations'/(job_id+'.json')).exists():
            result['training_job']=self.service.job(job_id)
        return result

    def round_status(self,identity,number):
        if type(number) is not int or not 1<=number<=100000:raise ValueError('Positive bounded round number required')
        with self.service.locked():
            run=self._run(identity);record=self._read(self._directory(identity)/('round-%06d.json'%number))
            value=dict(schema=1,run_id=identity,round=number,policy_id=run['policy_id'],
                model_contract_id=self.service.model,status=record.get('status','training_intent'),
                parent_job_id=record['parent_job_id'],training_job_id=record['training_request']['operation_id'],
                lesson_ids=record['lesson_ids'],promoted=False)
            for key in ('candidate_job_id','working_parent_job_id','report','assessment'):
                if key in record:value[key]=copy.deepcopy(record[key])
            return value

    def listing(self):
        return dict(schema=1,model_contract_id=self.service.model,runs=[self.status(p.name) for p in sorted(self.root.iterdir()) if p.is_dir() and not p.is_symlink()])

    def _operation(self,path,request,acceptance,state='recorded',**metadata):
        self.service._write(self.service.root/'operations'/(request['operation_id']+'.json'),
            dict(path=path,request=copy.deepcopy(request),acceptance=acceptance,state=state,**copy.deepcopy(metadata)))

    def submit(self,path,request):
        op=request.get('operation_id')
        if type(op) is not str or not OP.fullmatch(op):raise ValueError('Explicit learning operation ID required')
        target=self.service.root/'operations'/(op+'.json');binding=digest([path,request])
        dispatch=None
        with self.service.locked():
            if target.exists():
                prior=self.service._read(target)
                if prior['acceptance']['request_sha256']!=binding:raise ValueError('Learning operation ID binds different request bytes')
                if path==CREATE:self._finish_creation(prior,target)
                return prior['acceptance']
            acceptance=dict(schema=1,operation_id=op,request_sha256=binding)
            if path==CREATE:
                if set(request)!={'operation_id','policy_id','parent_job_id','expected_parent_sha256'}:
                    raise ValueError('Learning creation accepts registered policy and parent identities only')
                policy=self._policy(request['policy_id']);training=policy.describe()['training']
                parent_id=request['parent_job_id'];expected=request['expected_parent_sha256']
                if (parent_id is None)!=(expected is None) or (expected is not None and not HEX.fullmatch(str(expected))):
                    raise ValueError('An initial parent requires its exact artifact SHA')
                parent=self._parent(parent_id,training['mode'],expected)
                if parent and training['mode']=='lora':
                    prior=self.service._read(self.service.root/'operations'/(parent_id+'.json'))
                    if prior['request']['rank']!=training['rank']:raise ValueError('Initial LoRA rank differs')
                directory=self._directory(op)
                if (directory/'run.json').exists():raise ValueError('Learning creation record is missing for an existing run')
                run=dict(schema=1,run_id=op,policy_id=request['policy_id'],creation_request=copy.deepcopy(request),
                    phase='baseline_pending',revision=0,round=1,completed_rounds=0,
                    initial_parent_job_id=parent_id,working_parent_job_id=parent_id)
                acceptance['run_id']=op
                self._operation(path,request,acceptance,creation_record=run)
                directory.mkdir(mode=0o700,exist_ok=True);self._save(run)
                self._operation(path,request,acceptance,'complete',creation_record=run)
                return acceptance
            if path!=STEP:raise ValueError('Unknown learning mutation')
            base={'operation_id','run_id','expected_revision','action'}
            if not base<=set(request) or set(request)-base-{'lesson_ids','name','notes'}:
                raise ValueError('Learning step accepts identities and explicit lesson selection only')
            run=self._run(request['run_id']);policy=self._policy(run['policy_id']);descriptor=policy.describe()
            if type(request['expected_revision']) is not int or request['expected_revision']!=run['revision']:
                raise ValueError('Learning run revision changed; inspect the original run')
            action=request['action'];phase=run['phase'];acceptance.update(run_id=run['run_id'],step_id=op)
            if action=='advance':
                if phase in ('baseline_pending','parent_pending','candidate_pending','report_pending'):
                    if set(request)!=base:raise ValueError('This phase takes no client lessons or scores')
                    dispatch=('phase',run['run_id'],op,phase)
                    run.update(phase=phase.replace('_pending','_running'),pending_step_id=op)
                elif phase=='choosing_lessons':
                    if set(request)!=base|{'lesson_ids'}:raise ValueError('Select explicit verified lessons for this round')
                    ids=request['lesson_ids']
                    if (type(ids) is not list or not 1<=len(ids)<=4096 or any(type(i) is not str for i in ids)
                            or len(set(ids))!=len(ids)):raise ValueError('Distinct admitted lesson IDs required')
                    lessons=[self.service._lesson(identity) for identity in ids]
                    policy.validate_lessons(lessons,run.get('last_report'))
                    training=copy.deepcopy(descriptor['training'])
                    if training['steps']<(len(ids)+training['batch_size']-1)//training['batch_size']:
                        raise ValueError('Registered round must cover every selected lesson')
                    parent_id=run['working_parent_job_id'];parent=self._parent(parent_id,training['mode'])
                    job_id='learning-job-'+digest(dict(run=run['run_id'],round=run['round'],policy=run['policy_id']))
                    native=dict(training,operation_id=job_id,lesson_ids=list(ids),parent_job_id=parent_id)
                    if parent:native['expected_parent_sha256']=parent['candidate_sha256']
                    run.update(phase='training_pending',training_job_id=job_id,training_request=native,pending_step_id=op)
                    self._write(self._directory(run['run_id'])/('round-%06d.json'%run['round']),dict(
                        schema=1,round=run['round'],parent_job_id=parent_id,lesson_ids=list(ids),training_request=native))
                    dispatch=('train',run['run_id'],op,copy.deepcopy(run['training_request']))
                elif phase=='round_complete':
                    if set(request)!=base:raise ValueError('Start the next round before selecting its lessons')
                    if run['completed_rounds']>=descriptor['max_rounds']:raise ValueError('Owner round limit reached')
                    run['round']+=1;run['phase']='choosing_lessons'
                    for key in ('training_job_id','training_request','candidate_evaluation','pending_step_id'):run.pop(key,None)
                else:raise ValueError('Phase is not advanceable; inspect status or reconcile original work')
            elif action=='reconcile':
                if set(request)!=base:raise ValueError('Reconciliation takes no changed training settings')
                if phase=='needs_operator':
                    adoption=self._adopted_native(run)
                    run.setdefault('recovery_history',[]).append(dict(original_reason=run.get('operator_reason'),
                        original_reason_code=run['operator_reason_code'],completion_adoption=copy.deepcopy(adoption),reconciled_by=op))
                    run.update(phase='candidate_pending');run.pop('operator_reason',None);run.pop('operator_reason_code',None)
                elif phase in ('training_pending','training'):
                    dispatch=('train',run['run_id'],op,copy.deepcopy(run['training_request']))
                elif phase=='preservation_pending':
                    dispatch=('preserve',run['run_id'],op,copy.deepcopy(run['preservation_request']))
                elif phase.endswith('_running'):
                    # No second evaluation/update is silently dispatched after owner death.
                    pending=self.service._read(self.service.root/'operations'/(run['pending_step_id']+'.json'))
                    if pending['state'] in ('queued','running'):raise ValueError('Original phase is still owned; read status')
                    interrupted=run['pending_step_id'];run['interrupted_step_id']=interrupted;run['pending_step_id']=op
                    dispatch=('recover',run['run_id'],op,phase.replace('_running','_pending'))
                else:raise ValueError('No unresolved original phase to reconcile')
            elif action=='stop':
                if set(request)!=base or phase not in ('baseline_pending','parent_pending','choosing_lessons','candidate_pending','report_pending','round_complete','needs_operator'):
                    raise ValueError('Stop only at a settled phase boundary')
                run['phase']='stopped'
            elif action=='preserve':
                if set(request)!=base|{'name','notes'} or phase!='round_complete':raise ValueError('Preserve a reported round with an explicit name and notes')
                if descriptor['training']['mode']!='full':raise ValueError('Standalone preservation currently requires full weights; no implicit LoRA merge')
                parent_id=run['working_parent_job_id'];parent=self._parent(parent_id,'full')
                if parent is None:raise ValueError('No completed working candidate to preserve')
                export=dict(operation_id='learning-export-'+digest(dict(run=run['run_id'],step=op)),job_id=parent_id,
                    name=request['name'],notes=request['notes'],expected_sha256=parent['candidate_sha256'],
                    report_sha256=digest(run['last_report']))
                run['preservation_request']=export;run['pending_step_id']=op
                run['phase']='preservation_pending'
                dispatch=('preserve',run['run_id'],op,copy.deepcopy(run['preservation_request']))
            else:raise ValueError('Unknown learning action')
            run['revision']+=1
            self._operation(path,request,acceptance,'queued' if dispatch and dispatch[0] in ('phase','recover') else 'recorded')
            self._save(run)
            if not dispatch:self._operation(path,request,acceptance,'complete')
        if dispatch:
            if dispatch[0] in ('phase','recover'):self.service.executor.submit(self._work,*dispatch[1:],recover=dispatch[0]=='recover')
            else:self._dispatch(*dispatch)
        return acceptance

    def _adopted_native(self,run):
        if run.get('operator_reason_code')!='native_job_unresolved':raise ValueError('Only the original native interruption is recoverable here')
        expected='learning-job-'+digest(dict(run=run['run_id'],round=run['round'],policy=run['policy_id']))
        request=run['training_request'];training=self._policy(run['policy_id']).describe()['training']
        if any(request.get(key)!=value for key,value in training.items()):raise ValueError('Original adopted job differs from registered learning recipe')
        if run['completed_rounds']>=run['round']:raise ValueError('Original native round is already reported')
        if request.get('parent_job_id')!=run['working_parent_job_id']:raise ValueError('Original adopted job parent differs')
        if run['training_job_id']!=expected or request['operation_id']!=expected:raise ValueError('Original derived native job differs')
        round_record=self._read(self._directory(run['run_id'])/('round-%06d.json'%run['round']))
        if round_record['training_request']!=request or round_record['parent_job_id']!=run['working_parent_job_id']:
            raise ValueError('Original learning round/request/parent differs')
        job=self.service._read(self.service.root/'operations'/(expected+'.json'))
        if job.get('path')!='/cow/training/jobs' or job.get('state')!='complete' or job['request']!=request:
            raise ValueError('Original native job has not been adopted as complete')
        adoption=job.get('completion_adoption')
        if type(adoption) is not dict or adoption.get('promoted') is not False:raise ValueError('Original explicit completion adoption is missing')
        if type(adoption.get('operation_id')) is not str or not OP.fullmatch(adoption['operation_id']):
            raise ValueError('Original adoption operation requires a registered ID')
        if type(job.get('candidate')) is not dict:raise ValueError('Original adopted candidate envelope requires an object')
        operation=self.service._read(self.service.root/'operations'/(adoption['operation_id']+'.json'))
        if (operation.get('path')!='/cow/training/job-reconciliations' or operation.get('state')!='complete'
                or operation['request']['job_id']!=expected or operation['acceptance'].get('candidate_sha256')!=job['candidate']['candidate_sha256']
                or operation['validated_evidence_sha256']!=adoption['evidence_sha256']):
            raise ValueError('Original owner completion adoption binding differs')
        # A historical acknowledgment is not evidence that candidate/source/gate
        # bytes are still valid. Admission already holds the service record lock;
        # acquire its existing worker boundary without nesting record flock FDs.
        from tools.pcc_native_completion import worker_idle,adopted_value
        with worker_idle(self.service):
            value=adopted_value(self.service,expected)
        if value['candidate']!=job['candidate'] or value['evidence_sha256']!=adoption['evidence_sha256']:
            raise ValueError('Original adopted completion changed before learning reconciliation')
        return adoption

    def _dispatch(self,kind,identity,operation,request):
        # Original native/export requests were committed before this call. Their
        # ordinary PCC operation IDs make reconnection idempotent across the gap.
        acceptance=self.service.submit('/cow/training/jobs' if kind=='train' else '/cow/training/exports',request)
        with self.service.locked():
            run=self._run(identity)
            expected=run.get('training_request' if kind=='train' else 'preservation_request')
            phases=('training_pending','training') if kind=='train' else ('preservation_pending',)
            if request!=expected or run['phase'] not in phases:
                # Another reconciliation may already have advanced this exact
                # work. Complete only this receipt, never rewind a newer phase.
                target=self.service.root/'operations'/(operation+'.json');record=self.service._read(target)
                record.update(state='complete',already_settled=True);self.service._write(target,record)
                return
            if kind=='train':
                job=self.service._read(self.service.root/'operations'/(request['operation_id']+'.json'))
                if job['state']=='complete':run['phase']='candidate_pending'
                elif job['state'] in ('queued','running'):run['phase']='training'
                else:run.update(phase='needs_operator',operator_reason_code='native_job_unresolved',operator_reason='Original native job '+request['operation_id']+' is '+job['state']+'; no automatic retry')
            else:run.update(preserved_export=acceptance,phase='round_complete')
            original_step=run.pop('pending_step_id',None)
            run['revision']+=1;self._save(run)
            if original_step and original_step!=operation:
                original_path=self.service.root/'operations'/(original_step+'.json')
                original=self.service._read(original_path);original.update(state='complete',reconciled_by=operation);self.service._write(original_path,original)
            target=self.service.root/'operations'/(operation+'.json');record=self.service._read(target);record['state']='complete';self.service._write(target,record)

    def _evaluation(self,run,key):
        reference=run[key];path=self._directory(run['run_id'])/reference['file']
        value=self.service._read(path)
        if digest(value)!=reference['sha256']:raise ValueError('Committed learning evaluation changed')
        return value

    def _work(self,identity,operation,phase,*,recover=False):
        try:
            with (self.service.root/'worker.lock').open('a+') as lease:
                fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
                with self.service.locked():
                    run=self._run(identity);policy=self._policy(run['policy_id'])
                    if run.get('pending_step_id')!=operation:raise ValueError('Phase owner changed')
                    target=self.service.root/'operations'/(operation+'.json');record=self.service._read(target);record['state']='running';self.service._write(target,record)
                policy.verify()
                directory=self._directory(identity)
                if phase=='report_pending':
                    baseline=self._evaluation(run,'baseline_evaluation');parent=self._evaluation(run,'parent_evaluation');candidate=self._evaluation(run,'candidate_evaluation')
                    decision=policy.analyze(baseline,parent,candidate)
                    if (type(decision) is not dict or not {'working_parent','report'}<=set(decision) or set(decision)-{'working_parent','report','assessment'}
                            or decision['working_parent'] not in ('parent','candidate') or decision['report'].get('promoted') is not False):
                        raise ValueError('Owner analysis returned an invalid working-parent decision')
                    report=copy.deepcopy(decision['report'])
                    if 'assessment' in decision:
                        assessment=decision['assessment']
                        if (type(assessment) is not dict or assessment.get('weights_updated') is not False
                                or assessment.get('execution_authorized') is not False
                                or assessment.get('anchor_evaluation_id')!=candidate['evaluation_id']):
                            raise ValueError('Invalid independent assessment-expansion proposal')
                        run['last_assessment']=copy.deepcopy(assessment)
                    report_path=directory/('report-%06d.json'%run['round'])
                    if report_path.exists():
                        if self.service._read(report_path)!=report:raise ValueError('Original report differs; no replacement')
                    else:self.service._write(report_path,report)
                    if decision['working_parent']=='candidate':
                        run['working_parent_job_id']=run['training_job_id'];run['parent_evaluation']=run['candidate_evaluation']
                    run.update(last_report=report,phase='round_complete',completed_rounds=run['round'])
                    round_path=directory/('round-%06d.json'%run['round']);round_record=self._read(round_path)
                    round_record.update(status='complete',candidate_job_id=run['training_job_id'],
                        candidate_evaluation=run['candidate_evaluation'],report_sha256=digest(report),
                        working_parent_job_id=run['working_parent_job_id'],report=report)
                    if 'last_assessment' in run:round_record['assessment']=run['last_assessment']
                    self._write(round_path,round_record)
                else:
                    training=policy.describe()['training'];candidate=None
                    if phase=='parent_pending':candidate=self._parent(run['working_parent_job_id'],training['mode'])
                    elif phase=='candidate_pending':candidate=self._parent(run['training_job_id'],training['mode'])
                    name='baseline' if phase=='baseline_pending' else 'initial-parent' if phase=='parent_pending' else 'candidate-%06d'%run['round']
                    backend=directory/(name+'-backend');filename=name+'-evaluation.json'
                    if recover:
                        if (directory/filename).exists():value=self.service._read(directory/filename)
                        else:
                            restore=getattr(policy,'recover_evaluation',None)
                            value=restore(candidate,backend) if restore is not None else None
                            if value is None:raise ValueError('No complete original evaluation to recover; original child/output receipts require operator review')
                    else:
                        backend.mkdir(mode=0o700,exist_ok=False)
                        value=policy.evaluate(candidate,backend,lease.fileno())
                    policy.validate_evaluation(value,candidate)
                    if not (directory/filename).exists():self.service._write(directory/filename,value)
                    reference=dict(file=filename,sha256=digest(value))
                    if phase=='baseline_pending':
                        run['baseline_evaluation']=reference
                        if run['working_parent_job_id'] is None:run.update(parent_evaluation=reference,phase='choosing_lessons')
                        else:run['phase']='parent_pending'
                    elif phase=='parent_pending':run.update(parent_evaluation=reference,phase='choosing_lessons')
                    else:run.update(candidate_evaluation=reference,phase='report_pending')
                with self.service.locked():
                    current=self._run(identity)
                    if current.get('pending_step_id')!=operation:raise ValueError('Original learning phase lost ownership')
                    run['revision']=current['revision']+1;run.pop('pending_step_id',None)
                    interrupted=run.pop('interrupted_step_id',None);self._save(run)
                    if interrupted:
                        old_path=self.service.root/'operations'/(interrupted+'.json');old=self.service._read(old_path)
                        old.update(state='complete',reconciled_by=operation);self.service._write(old_path,old)
                    record=self.service._read(target);record['state']='complete';self.service._write(target,record)
        except Exception as error:
            with self.service.locked():
                run=self._run(identity);run.update(phase='needs_operator',operator_reason='Original learning phase failed; private diagnostics retained; no automatic retry',revision=run['revision']+1);self._save(run)
                target=self.service.root/'operations'/(operation+'.json');record=self.service._read(target)
                record.update(state='failed',private_error=type(error).__name__+': '+str(error));self.service._write(target,record)
