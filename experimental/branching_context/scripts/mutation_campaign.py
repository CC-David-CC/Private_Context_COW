"""Fail-closed negative testing with real inference and independent fresh oracle.

Mutations exist only in this test harness. Bounded subsets never qualify G5-011.
"""
import argparse
from contextlib import contextmanager
from dataclasses import asdict, replace
import hashlib
import io
import json
import os
from pathlib import Path
import struct
import sys
import unittest
from unittest.mock import patch

REQUIRED={
 'M01':'branch-contamination','M02':'unconsumed-sampled-frontier',
 'M03':'chunked-rng-reset','M04':'same-seed-count-stale-prefix',
 'M05':'shared-partial-page-write','M06':'aba-stale-cas',
 'M07':'double-retry-publication','M08':'corrupt-checkpoint-accepted',
 'M09':'eviction-deletes-history','M10':'evaluator-mutation',
 'M11':'incompatible-state-reuse','M12':'half-consumed-error-publication'}
IMPLEMENTED=('M01','M02','M03','M04','M06','M07','M08','M09','M10','M11','M12')


def detected(control,mutant,expected):
    def clean(row):
        required={'tests','failures','errors','skips','xfail','oracle_requests'}
        return (required<=row.keys() and type(row['tests']) is int and row['tests']==1
            and all(type(row[k]) is list for k in ('failures','errors','skips'))
            and not row['errors'] and not row['skips']
            and type(row['xfail']) is int and row['xfail']==0
            and type(row['oracle_requests']) is int and row['oracle_requests']>0)
    return (clean(control) and not control.get('failures') and clean(mutant)
            and len(mutant.get('failures',[]))==1 and type(mutant.get('mutation_invocations')) is int and mutant['mutation_invocations']>0
            and mutant['failures'][0] in expected)


def campaign_complete(rows):
    return set(rows)==set(REQUIRED) and all(row.get('detected') is True for row in rows.values())


class RecordedResult(unittest.TextTestResult):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs);self.failure_messages=[]
    def addFailure(self,test,error):
        self.failure_messages.append(str(error[1]));super().addFailure(test,error)


class RealProbe(unittest.TestCase):
    def setUp(self):
        from context_time_travel.real import BranchingModel,FreshOracle
        self.backend=BranchingModel.load_backend(self.profile)
        self.addCleanup(self.backend.close)
        self.oracle=FreshOracle(self.profile);self.addCleanup(self.oracle.close)
        self.sut=BranchingModel(self.directory/'store',self.backend,self.profile)
        self.addCleanup(self.sut.evict_all)
        self.P=tuple(self.profile['fixtures']['prefix'][:9])
        self.root=self.sut.create(self.P)
        self.row.update(oracle_requests=0,checks=[],mutation_invocations=0)
        self.addCleanup(self.finish_report)
        self.check(self.root,self.P)
    def finish_report(self):
        self.row['oracle_requests']=self.oracle.requests
        self.row['backend_counters']=dict(self.backend._refresh_stats())
    def check(self,snapshot,history):
        from context_time_travel.numerics import compare_logits
        actual=self.sut.next_logits(snapshot);expected=self.oracle.forward(history)
        value=compare_logits(actual,expected,atol=self.profile['atol'],rtol=self.profile['rtol'])
        self.row['checks'].append(dict(tokens=history,max_absolute_error=value.max_absolute_error,computed_tokens=self.oracle.last['computed_tokens']))
        self.row['oracle_requests']=self.oracle.requests
        if not value.pass_elementwise or not value.top1_stable_within_tolerance:
            for name,values in [('actual',actual),('expected',expected)]:
                (self.directory/(name+'.f32')).write_bytes(struct.pack('<'+str(len(values))+'f',*values))
            self.fail('next-logits')
        if self.sut.history(snapshot)!=history:self.fail('canonical-history')
        if self.sut.consumed(snapshot)!=len(history):self.fail('consumed-frontier')
    def test_M01(self):
        a=tuple(self.profile['fixtures']['suffix_a']);b=tuple(self.profile['fixtures']['suffix_b'])
        first=self.sut.feed(self.root,a);self.check(first,self.P+a)
        second=self.sut.feed(self.root,b);self.check(second,self.P+b)
        self.check(self.root,self.P)
    def test_M02(self):
        token=self.profile['fixtures']['suffix_a'][0]
        start=self.sut.start_generation(self.root,dict(allowed=[token]),'frontier-mutant')
        generated=self.sut.generate(start,1)
        self.assertEqual(generated.accepted_tokens,(token,))
        self.check(generated.snapshot_id,self.P+(token,))

    def test_M03(self):
        token=self.profile['fixtures']['suffix_a'][0]
        start=self.sut.start_generation(self.root,dict(allowed=[token]),'chunked-mutant')
        whole=self.sut.generate(start,4)
        first=self.sut.generate(start,2)
        second=self.sut.generate(first.snapshot_id,2)
        self.check(whole.snapshot_id,self.P+(token,)*4)
        self.check(second.snapshot_id,self.P+(token,)*4)
        if self.sut.controller(whole.snapshot_id)['next_draw']!=4:
            self.fail('chunked-draw-state')
        if self.sut.controller(second.snapshot_id)!=self.sut.controller(whole.snapshot_id):
            self.fail('chunked-draw-state')
        if first.accepted_tokens+second.accepted_tokens!=whole.accepted_tokens:
            self.fail('chunked-continuation')

    def test_M04(self):
        q=(self.profile['fixtures']['suffix_a'][0],)+self.P[1:]
        self.assertNotEqual(q,self.P)
        other=self.sut.create(q)
        policy=dict(allowed=[self.profile['fixtures']['suffix_b'][0]])
        first=self.sut.start_generation(self.root,policy,'same-seed-count')
        second=self.sut.start_generation(other,policy,'same-seed-count')
        left=self.sut.controller(first);right=self.sut.controller(second)
        self.assertEqual(left['stream_key'],right['stream_key'])
        self.assertEqual(left['next_draw'],right['next_draw'])
        self.row['identity_witness']=dict(prefix_p=self.P,prefix_q=q,
            count_p=len(self.P),count_q=len(q),stream_key=left['stream_key'],
            draw_p=left['next_draw'],draw_q=right['next_draw'])
        self.check(second,q)
        self.check(first,self.P)

    def test_M06(self):
        from context_time_travel.store import Conflict
        ta=self.profile['fixtures']['suffix_a'][0]
        tb=self.profile['fixtures']['suffix_b'][0]
        a=self.sut.feed(self.root,(ta,));b=self.sut.feed(self.root,(tb,))
        self.check(a,self.P+(ta,));self.check(b,self.P+(tb,))
        name='m06-branch'
        self.sut.store.fork(name,self.root)
        self.sut.store.move(name,0,a,'m06-to-a')
        self.sut.store.move(name,1,self.root,'m06-back-to-p')
        before=self.sut.store.head(name)
        self.assertEqual((before.tip,before.revision),(self.root,2))
        stale_rejected=False
        try:
            self.sut.store.move(name,0,b,'m06-stale')
        except Conflict:
            stale_rejected=True
        self.row['aba_witness']=dict(root=self.root,tip_a=a,tip_b=b,
            revision_before_stale=before.revision,tip_before_stale=before.tip,
            stale_expected_revision=0,stale_rejected=stale_rejected)
        if not stale_rejected:self.fail('aba-stale-cas')
        after=self.sut.store.head(name)
        self.assertEqual((after.tip,after.revision),(self.root,2))
        self.check(self.root,self.P)

    def test_M07(self):
        ta=self.profile['fixtures']['suffix_a'][0]
        a=self.sut.feed(self.root,(ta,))
        self.check(a,self.P+(ta,))
        name='m07-branch'
        self.sut.store.fork(name,self.root)
        first=self.sut.store.move(name,0,a,'m07-operation')
        retried=self.sut.store.move(name,0,a,'m07-operation')
        head=self.sut.store.head(name)
        self.row['retry_witness']=dict(first_revision=first.revision,
            retry_revision=retried.revision,head_revision=head.revision,
            tip=head.tip,same_receipt=(first==retried))
        if retried!=first or head.revision!=first.revision or head.tip!=a:
            self.fail('double-retry-publication')

    def test_M08(self):
        import json
        import struct
        from context_time_travel.store import IntegrityError
        paths=self.sut.checkpoint(self.root)
        manifest=json.loads(paths['manifest_path'].read_bytes())['manifest']
        payload_path=paths['payload_path']
        payload=bytearray(payload_path.read_bytes())
        offset=manifest['state_bytes']
        payload[offset] ^= 1
        payload_path.write_bytes(payload)
        self.sut.cache.pop(self.root,None)
        rejected=False
        try:
            actual=self.sut.next_logits(self.root)
        except IntegrityError:
            rejected=True
        self.row['corruption_witness']=dict(snapshot=self.root,
            state_bytes=manifest['state_bytes'],corrupted_offset=offset,
            original_payload_sha256=manifest['payload_sha256'],
            rejected=rejected)
        if not rejected:
            from context_time_travel.numerics import compare_logits
            expected=self.oracle.forward(self.P)
            comparison=compare_logits(actual,expected,atol=self.profile['atol'],rtol=self.profile['rtol'])
            self.row['corruption_witness']['accepted_max_absolute_error']=comparison.max_absolute_error
            self.fail('corrupt-checkpoint-accepted')

    def test_M09(self):
        token=self.profile['fixtures']['suffix_a'][0]
        child=self.sut.feed(self.root,(token,))
        history=self.P+(token,)
        self.sut._mutation_m09_root=self.root
        self.sut._mutation_m09_child=child
        self.check(child,history)
        checkpoint=self.sut.checkpoint(child)
        self.assertTrue(checkpoint['manifest_path'].exists())
        self.sut.evict_all()
        try:
            retained=self.sut.history(child)
        except Exception:
            self.row['eviction_witness']=dict(root=self.root,child=child,
                checkpoint_manifest_exists=checkpoint['manifest_path'].exists(),history_retained=False)
            self.fail('eviction-deletes-history')
        self.row['eviction_witness']=dict(root=self.root,child=child,
            checkpoint_manifest_exists=checkpoint['manifest_path'].exists(),
            history_retained=(retained==history))
        if retained!=history:self.fail('eviction-deletes-history')
        self.check(child,history)

    def test_M10(self):
        from context_time_travel import evaluation
        branch='m10-evaluation'
        before_head=self.sut.store.fork(branch,self.root)
        before_bytes=self.sut.snapshot_bytes(self.root)
        before_history=self.sut.history(self.root)
        source="result = {'mandatory': {'unchanged': 'PASS'}, 'score': 1.0, 'unresolved': []}"
        specification=dict(task_root=self.root,
            evaluator=dict(name='immutable-probe',version='1',source=source),
            fixtures={},metric=dict(name='unchanged',order='maximize'),seed=17,
            mandatory=['unchanged'],timeout_seconds=5)
        record=evaluation.evaluate(self.sut.store,self.root,specification)
        after_head=self.sut.store.head(branch)
        unchanged=(after_head==before_head and self.sut.snapshot_bytes(self.root)==before_bytes
                   and self.sut.history(self.root)==before_history)
        self.row['evaluation_witness']=dict(snapshot_id=record['snapshot_id'],
            outcomes=record['outcomes'],before_revision=before_head.revision,
            after_revision=after_head.revision,before_tip=before_head.tip,
            after_tip=after_head.tip,snapshot_bytes_unchanged=(self.sut.snapshot_bytes(self.root)==before_bytes),
            history_unchanged=(self.sut.history(self.root)==before_history))
        if not unchanged:self.fail('evaluator-mutation')
        self.check(self.root,self.P)

    def test_M11(self):
        from context_time_travel.store import IntegrityError
        q=(self.profile['fixtures']['suffix_a'][0],)+self.P[1:]
        self.assertNotEqual(q,self.P)
        other=self.sut.create(q)
        self.check(other,q)
        saved=self.sut.cache[other]
        self.sut.cache[self.root]=replace(saved,model_contract_id='f'*64,
            input_identity=self.sut._input_identity(self.P))
        rejected=False
        try:
            actual=self.sut.next_logits(self.root)
        except IntegrityError:
            rejected=True
        witness=dict(snapshot=self.root,incompatible_snapshot=other,
            prefix_p=self.P,prefix_q=q,count_p=len(self.P),count_q=len(q),
            mismatched_contract_rejected=rejected)
        if not rejected:
            expected=self.oracle.forward(self.P)
            from context_time_travel.numerics import compare_logits
            comparison=compare_logits(actual,expected,atol=self.profile['atol'],rtol=self.profile['rtol'])
            witness['accepted_state_max_absolute_error']=comparison.max_absolute_error
        self.row['incompatible_witness']=witness
        if not rejected:self.fail('incompatible-state-reused')

    def test_M12(self):
        tokens=tuple(self.profile['fixtures']['suffix_a'][:2])
        self.assertEqual(len(tokens),2)
        name='m12-main'
        before_head=self.sut.store.fork(name,self.root)
        before_history=self.sut.history(self.root)
        before_bytes=self.sut.snapshot_bytes(self.root)
        before_objects={path.name for path in self.sut.store.objects.glob('*.json')}
        injected=[]
        published=[]
        def interrupt_after_first(point):
            if point=='after_token_consume' and not injected:
                injected.append(point)
                if self.mutated:
                    self.row['mutation_invocations']+=1
                    partial=self.sut.store.append(self.root,(tokens[0],),self.sut.controller(self.root))
                    published.append(partial)
                raise RuntimeError('injected after first token consume')
        self.sut._lease_fault=interrupt_after_first
        raised=False
        try:
            self.sut.feed(self.root,tokens)
        except RuntimeError as error:
            raised='injected after first token consume' in str(error)
        finally:
            self.sut._lease_fault=lambda point: None
        after_objects={path.name for path in self.sut.store.objects.glob('*.json')}
        after_head=self.sut.store.head(name)
        self.row['publication_witness']=dict(tokens=tokens,injected_after_first=bool(injected),
            raised=raised,new_snapshot_objects=sorted(after_objects-before_objects),
            mutated_partial_snapshots=published,before_revision=before_head.revision,
            after_revision=after_head.revision,root_history_unchanged=(self.sut.history(self.root)==before_history),
            root_bytes_unchanged=(self.sut.snapshot_bytes(self.root)==before_bytes))
        if not injected or not raised or after_objects!=before_objects or after_head!=before_head:
            self.fail('half-consumed-error-publication')
        self.check(self.root,self.P)


def run_attempt(profile,directory,case,*,mutated):
    from context_time_travel.real import BranchingModel
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=False)
    row=dict(tests=0,failures=[],errors=[],skips=[],xfail=0,oracle_requests=0,mutation_invocations=0)
    cls=type('Probe_'+case,(RealProbe,),dict(profile=profile,directory=directory,row=row,mutated=mutated))
    original=BranchingModel._feed
    def replacement(self,snapshot,ids,decoder):
        if case=='M02':
            row['mutation_invocations']+=1;return snapshot
        if hasattr(self,'_mutation_prior_branch'):
            row['mutation_invocations']+=1;snapshot=self._mutation_prior_branch
        result=original(self,snapshot,ids,decoder)
        self._mutation_prior_branch=result
        return result
    original_generate=BranchingModel.generate
    def reset_draw(self,snapshot,max_steps):
        decoder=self.controller(snapshot)
        decoder['next_draw']=0
        row['mutation_invocations']+=1
        snapshot=self._controller_transition(snapshot,decoder)
        return original_generate(self,snapshot,max_steps)
    original_save=BranchingModel._save
    def wrong_prefix(self,lease,history):
        saved=original_save(self,lease,history)
        previous=getattr(self,'_mutation_prior_checkpoint',None)
        if previous is not None and previous.consumed==len(history):
            row['mutation_invocations']+=1
            return replace(previous,input_identity=self._input_identity(history))
        self._mutation_prior_checkpoint=saved
        return saved
    from context_time_travel.store import Store
    original_move=Store.move
    def accept_stale(self,name,expected_revision,target_snapshot_id,operation_id):
        if operation_id=='m06-stale':
            row['mutation_invocations']+=1
            expected_revision=self.head(name).revision
        return original_move(self,name,expected_revision,target_snapshot_id,operation_id)
    retried_operations=set()
    def publish_retry_twice(self,name,expected_revision,target_snapshot_id,operation_id):
        if operation_id=='m07-operation' and operation_id in retried_operations:
            row['mutation_invocations']+=1
            expected_revision=self.head(name).revision
            operation_id=operation_id+'-duplicate'
        retried_operations.add('m07-operation' if operation_id.startswith('m07-operation') else operation_id)
        return original_move(self,name,expected_revision,target_snapshot_id,operation_id)
    original_load=BranchingModel._load_checkpoint
    def accept_corrupt_checkpoint(self,snapshot,history):
        try:
            return original_load(self,snapshot,history)
        except Exception:
            if case!='M08':
                raise
            import json,struct
            from context_time_travel.real import Checkpoint
            row['mutation_invocations']+=1
            manifest=json.loads((self.checkpoint_directory/(snapshot+'.json')).read_bytes())['manifest']
            payload=(self.checkpoint_directory/(manifest['payload_sha256']+'.state')).read_bytes()
            size=manifest['state_bytes']
            frontier=struct.unpack('<'+str(self.backend.vocab_size)+'f',payload[size:])
            blob=payload[:size]
            return Checkpoint(self.profile['model_contract_id'],self.profile['execution_profile_id'],
                len(history),blob,frontier,Checkpoint.digest(blob,frontier),self._input_identity(history))
    original_evict=BranchingModel.evict_all
    def delete_history_on_evict(self):
        result=original_evict(self)
        if not getattr(self,'_m09_mutation_applied',False):
            self._m09_mutation_applied=True
            row['mutation_invocations']+=1
            child=self._mutation_m09_child
            root=self._mutation_m09_root
            self.store.release_history(child)
            self.store.collect_history(root)
        return result
    from context_time_travel import evaluation
    original_evaluate_read_only=evaluation.evaluate_read_only
    mutation_context={}
    def mutate_during_evaluation(candidate,source,**kwargs):
        result=original_evaluate_read_only(candidate,source,**kwargs)
        row['mutation_invocations']+=1
        store=mutation_context['store']
        name='m10-evaluation'
        store.move(name,store.head(name).revision,mutation_context['root'],'m10-mutant-move')
        return result
    original_materialize=BranchingModel._materialize
    @contextmanager
    def reuse_incompatible_state(self,snapshot):
        saved=self.cache.get(snapshot)
        if case=='M11' and saved is not None and saved.model_contract_id!=self.profile['model_contract_id']:
            row['mutation_invocations']+=1
            self.cache[snapshot]=replace(saved,model_contract_id=self.profile['model_contract_id'])
        with original_materialize(self,snapshot) as state:
            yield state
    stream=io.StringIO()
    suite=unittest.TestSuite([cls('test_'+case)])
    if mutated:
        if case=='M06':
            with patch.object(Store,'move',accept_stale):
                result=unittest.TextTestRunner(stream=stream,verbosity=2,resultclass=RecordedResult).run(suite)
        elif case=='M07':
            with patch.object(Store,'move',publish_retry_twice):
                result=unittest.TextTestRunner(stream=stream,verbosity=2,resultclass=RecordedResult).run(suite)
        elif case=='M08':
            with patch.object(BranchingModel,'_load_checkpoint',accept_corrupt_checkpoint):
                result=unittest.TextTestRunner(stream=stream,verbosity=2,resultclass=RecordedResult).run(suite)
        elif case=='M09':
            with patch.object(BranchingModel,'evict_all',delete_history_on_evict):
                result=unittest.TextTestRunner(stream=stream,verbosity=2,resultclass=RecordedResult).run(suite)
        elif case=='M10':
            original_set_up=cls.setUp
            def set_up_with_mutation_store(test):
                original_set_up(test)
                mutation_context['store']=test.sut.store
                mutation_context['root']=test.root
            cls.setUp=set_up_with_mutation_store
            try:
                with patch.object(evaluation,'evaluate_read_only',mutate_during_evaluation):
                    result=unittest.TextTestRunner(stream=stream,verbosity=2,resultclass=RecordedResult).run(suite)
            finally:
                cls.setUp=original_set_up
        elif case=='M11':
            with patch.object(BranchingModel,'_materialize',reuse_incompatible_state):
                result=unittest.TextTestRunner(stream=stream,verbosity=2,resultclass=RecordedResult).run(suite)
        else:
            method,hook={'M03':('generate',reset_draw),'M04':('_save',wrong_prefix)}.get(case,('_feed',replacement))
            with patch.object(BranchingModel,method,hook):
                result=unittest.TextTestRunner(stream=stream,verbosity=2,resultclass=RecordedResult).run(suite)
    else:result=unittest.TextTestRunner(stream=stream,verbosity=2,resultclass=RecordedResult).run(suite)
    row.update(tests=result.testsRun,failures=result.failure_messages,
        errors=[text for test,text in result.errors],skips=[reason for test,reason in result.skipped],
        xfail=len(result.expectedFailures))
    (directory/'unittest.txt').write_text(stream.getvalue())
    (directory/'result.json').write_text(json.dumps(row,indent=2)+'\n')
    return row


def run_campaign(profile,directory,cases):
    cases=list(cases)
    if not cases or len(set(cases))!=len(cases) or any(case not in REQUIRED for case in cases):raise ValueError('Distinct registered mutant IDs required')
    if profile.get('calibration_status')!='FROZEN':raise ValueError('Frozen real profile required')
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=False)
    report=dict(schema=1,status='RUNNING',requested=cases,required=REQUIRED,rows={},
                execution_profile_id=profile['execution_profile_id'],model_weights_sha256=profile['model_weights_sha256'],
                full_campaign_qualified=False,scope='Real negative probes; subset does not qualify G5-011')
    try:
        for case in cases:
            if case not in IMPLEMENTED:
                report.update(status='BLOCKED',reason='Mutant implementation pending: '+case);break
            control=run_attempt(profile,directory/(case+'-control'),case,mutated=False)
            if control['failures'] or control['errors'] or control['skips'] or control['xfail']:
                report['rows'][case]=dict(control=control,detected=False);report['status']='FAIL';break
            mutant=run_attempt(profile,directory/(case+'-mutant'),case,mutated=True)
            expected={'M03':('chunked-draw-state','chunked-continuation'),
                      'M06':('aba-stale-cas',),
                      'M07':('double-retry-publication',),
                      'M08':('corrupt-checkpoint-accepted',),
                      'M09':('eviction-deletes-history',),
                      'M10':('evaluator-mutation',),
                      'M11':('incompatible-state-reused',),
                      'M12':('half-consumed-error-publication',)}.get(case,('next-logits','canonical-history','consumed-frontier'))
            report['rows'][case]=dict(control=control,mutant=mutant,detected=detected(control,mutant,expected))
            print(case,'DETECTED' if report['rows'][case]['detected'] else 'NOT QUALIFIED',flush=True)
            if not report['rows'][case]['detected']:report['status']='FAIL';break
        else:
            report['full_campaign_qualified']=campaign_complete(report['rows'])
            report['status']='PASS' if report['full_campaign_qualified'] else 'BOUNDED_MUTANTS_DETECTED'
        return report
    except BaseException as error:report.update(status='FAIL',error=repr(error));raise
    finally:(directory/'campaign.json').write_text(json.dumps(report,indent=2)+'\n')


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--profile',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--cases',nargs='+',choices=list(REQUIRED),default=list(REQUIRED))
    args=ap.parse_args();sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
    result=run_campaign(json.loads(args.profile.read_bytes()),args.output,args.cases)
    print(json.dumps(dict(status=result['status'],full_campaign_qualified=result['full_campaign_qualified']),indent=2))
    raise SystemExit(0 if result['status'] in ('PASS','BOUNDED_MUTANTS_DETECTED') else 1)

if __name__=='__main__':main()
