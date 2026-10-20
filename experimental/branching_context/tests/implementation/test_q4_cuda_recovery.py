"""CPU evidence-guard tests for same-service Q4 worker recovery.

These controls validate evidence schemas only; they do not emulate CUDA or a
native worker restart.
"""
from __future__ import annotations

import copy
from contextlib import contextmanager
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

SCRIPTS = Path(__file__).resolve().parents[2] / 'scripts'
API_SRC = Path(__file__).resolve().parents[2] / 'src'
sys.path.insert(0, str(SCRIPTS))

import run_q4_cuda_recovery as recovery
from run_q4_cuda_recovery import (
    OOMProbeError,
    base,
    validate_same_service_recovery,
    validate_clean_restart_branch,
    validate_worker_transition_trace,
    validate_clean_worker_exit_trace,
    validate_durable_completion,
    validate_http_source_binding,
    validate_recovery_result,
    _recovery_preflight,
    _service_stop_receipt,
)
import run_q4_publication_fault as publication


def evidence():
    tokens_a = [101, 102, 103, 104, 105, 106, 107, 108]
    tokens_b = [201, 202, 203, 204, 205, 206, 207, 208]
    source = {'tip':'P123', 'payload_sha256':'a' * 64}
    branch_a_before={'revision':0,'tip':'P123'}
    branch_a_after={'revision':1,'tip':'SAME456'}
    branch_b_before={'revision':0,'tip':'P123'}
    branch_b_after={'revision':1,'tip':'RESTART789'}
    def completion(tokens, after, name):
        answer={'tokens':tokens,'snapshot':after['tip']}
        receipt={'snapshot':after['tip'],'branch':dict(after)}
        receipt['branch']['name']=name
        return [{'event':'token','token_id':tokens[0]},
                {'event':'durable','receipt':copy.deepcopy(receipt)},
                {'event':'complete','answer':answer,'receipt':receipt}]
    transition={'process_id':2000,'event_count':3,'overflowed':False,'events':[
        {'sequence':0,'event':'native_worker_closed','pid':2100,'returncode':-9,'close_succeeded':False},
        {'sequence':1,'event':'native_worker_replaced','old_pid':2100,'new_pid':2200,
         'old_returncode':-9,'restart_count':1,'process_id':2000},
        {'sequence':2,'event':'native_worker_closed','pid':2200,'returncode':0,'close_succeeded':True}]}
    clean_trace={'process_id':3000,'event_count':1,'overflowed':False,'events':[
        {'sequence':0,'event':'native_worker_closed','pid':3100,'returncode':0,'close_succeeded':True}]}
    same = {
        'http_pid': 2000,
        'old_native_pid': 2100,
        'replacement': {
            'old_native_pid':2100,
            'old_native_exit':{'pid':2100, 'exit_code':-9, 'pid_absent':True},
            'new_native_process':{'pid':2200, 'parent_pid':2000,'exe':'/usr/bin/native'},
            'factory_after_old_exit':True,
            'restart_count':1,
            'worker_status':{'health':'healthy','restarts':1,'pid':2200},
        },
        'followup':{
            'event':'complete',
            'branch':'FaultBranch',
            'answer':{'tokens':tokens_a,'snapshot':branch_a_after['tip']},
            'events':completion(tokens_a,branch_a_after,'FaultBranch'),
            'branch_before':{'branch':'FaultBranch',**branch_a_before},'branch_after':branch_a_after,
            'counter_deltas':{'forward_tokens':8, 'consumed_tokens':8, 'prefill_tokens':0},
        },
        'source_after':dict(source),
    }
    restart = {
        'branch_before':{'branch':'RecoveryBranch',**branch_b_before},
        'fault_branch':'FaultBranch',
        'source_after':dict(source),
        'followup':{'event':'complete','answer':{'tokens':tokens_b,'snapshot':branch_b_after['tip']},
                    'branch':'RecoveryBranch',
                    'events':completion(tokens_b,branch_b_after,'RecoveryBranch'),
                    'branch_before':{'branch':'RecoveryBranch',**branch_b_before},'branch_after':branch_b_after,
                    'counter_deltas':{'forward_tokens':8,'consumed_tokens':8,'prefill_tokens':0}},
    }
    result = {'status':'PASS','native_binary':'/usr/bin/native',
        'source_before':{**source,'head':{'tip':'P123','revision':4},
            'record_sha256':'b'*64,'checkpoint':{'P123.json':{'sha256':'c'*64}}},
        'native_oom':{'response':{'status':'error','message':'CUDA error 2'},'original_error':'CUDA error 2',
                      'oom_failure_delta':1,
                      'stats_after':{'vmm_last_failed_operation':'cuMemCreate',
                                     'vmm_last_driver_error':2}},
        'first_service':{'http_pid':2000,'old_native':{'pid':2100,'exe':'/usr/bin/native'},'trace_document':transition,
            'stop':{'exit_code':0,'pid_absent_after_stop':True}},
        'same_service':{**same,'worker_transition':{'old_pid':2100,'old_returncode':-9,
            'new_pid':2200,'restart_count':1}}, 'clean_restart':restart,
        'fault_http':{'evidence':{'error':{'event':'error','error':'CUDA error 2'}}},
        'clean_restart_service':{'http_pid':3000,'native_process':{'pid':3100,'parent_pid':3000,'exe':'/usr/bin/native'},
            'stop':{'exit_code':0,'pid_absent_after_stop':True},
            'trace_document':clean_trace},
        'helper':{'released_before_gate_abort':True,'exit_code':0,'pid_absent':True},
        'source_after_all':{'head':{'tip':'P123','revision':4},'payload_sha256':'a'*64,
            'record_sha256':'b'*64,'checkpoint':{'P123.json':{'sha256':'c'*64}}}}
    return result


class MemoryStore:
    def __init__(self, *, name='CliA', revision=0, tip='a'*64, root=None,
                 controller_status='active', tokens=(11,22,33)):
        self.value={'name':name,'revision':revision,'tip':tip,'root':root or tip}
        self.controller={'status':controller_status}
        self.tokens=list(tokens)
        self.leases=[]

    @contextmanager
    def read_lease(self,snapshot):
        if snapshot!=self.value['tip']:
            raise OOMProbeError('lease target is not current snapshot')
        self.leases.append(snapshot)
        yield snapshot

    def head(self,branch):
        if branch!=self.value['name']:
            raise OOMProbeError('unknown branch')
        from types import SimpleNamespace
        return SimpleNamespace(**self.value)

    def get_snapshot(self,snapshot):
        if snapshot!=self.value['tip']:
            raise OOMProbeError('snapshot is not current')
        return {'controller':dict(self.controller)}

    def lineage(self,snapshot):
        if snapshot!=self.value['tip']:
            raise OOMProbeError('lineage is not current')
        return self.value['root'],list(self.tokens)


class Q4CudaRecoveryEvidenceTests(unittest.TestCase):
    def test_valid_same_service_recovery_requires_worker_replacement_and_zero_prefix(self):
        row=evidence()['same_service']
        accepted=validate_same_service_recovery(row,source_snapshot='P123',
            source_payload_sha256='a'*64,http_pid=2000,old_native_pid=2100)
        self.assertEqual(accepted['new_native_pid'],2200)
        self.assertEqual(accepted['followup_tokens'],[101,102,103,104,105,106,107,108])

    def test_same_service_rejects_reused_pid_missing_exit_or_factory_before_exit(self):
        base=evidence()['same_service']
        variants=[]
        same=copy.deepcopy(base); same['replacement']['new_native_process']['pid']=2100; variants.append(same)
        missing=copy.deepcopy(base); missing['replacement']['old_native_exit']['pid_absent']=False; variants.append(missing)
        order=copy.deepcopy(base); order['replacement']['factory_after_old_exit']=False; variants.append(order)
        wrong_parent=copy.deepcopy(base); wrong_parent['replacement']['new_native_process']['parent_pid']=9999; variants.append(wrong_parent)
        for bad in variants:
            with self.subTest(bad=bad), self.assertRaises(OOMProbeError):
                validate_same_service_recovery(bad,source_snapshot='P123',
                    source_payload_sha256='a'*64,http_pid=2000,old_native_pid=2100)

    def test_same_service_requires_one_healthy_restart_on_unchanged_http_pid(self):
        base=evidence()['same_service']
        for key,value in (('restart_count',2),):
            bad=copy.deepcopy(base); bad['replacement'][key]=value
            with self.subTest(key=key), self.assertRaises(OOMProbeError):
                validate_same_service_recovery(bad,source_snapshot='P123',
                    source_payload_sha256='a'*64,http_pid=2000,old_native_pid=2100)
        unhealthy=copy.deepcopy(base); unhealthy['replacement']['worker_status']['health']='restarting'
        with self.assertRaises(OOMProbeError):
            validate_same_service_recovery(unhealthy,source_snapshot='P123',
                source_payload_sha256='a'*64,http_pid=2000,old_native_pid=2100)
        with self.assertRaises(OOMProbeError):
            validate_same_service_recovery(base,source_snapshot='P123',
                source_payload_sha256='a'*64,http_pid=2001,old_native_pid=2100)

    def test_same_service_requires_eight_actual_tokens_and_exact_work_deltas(self):
        base=evidence()['same_service']
        variants=[]
        short=copy.deepcopy(base); short['followup']['answer']['tokens'].pop(); variants.append(short)
        prefix=copy.deepcopy(base); prefix['followup']['counter_deltas']['prefill_tokens']=1; variants.append(prefix)
        wrong=copy.deepcopy(base); wrong['followup']['counter_deltas']['forward_tokens']=7; variants.append(wrong)
        not_consumed=copy.deepcopy(base); not_consumed['followup']['counter_deltas']['consumed_tokens']=7; variants.append(not_consumed)
        bool_token=copy.deepcopy(base); bool_token['followup']['answer']['tokens'][0]=True; variants.append(bool_token)
        for bad in variants:
            with self.subTest(bad=bad), self.assertRaises(OOMProbeError):
                validate_same_service_recovery(bad,source_snapshot='P123',
                    source_payload_sha256='a'*64,http_pid=2000,old_native_pid=2100)

    def test_same_service_rejects_source_head_or_payload_mutation(self):
        base=evidence()['same_service']
        changed=copy.deepcopy(base); changed['source_after']['tip']='Q456'
        payload=copy.deepcopy(base); payload['source_after']['payload_sha256']='b'*64
        for bad in (changed,payload):
            with self.subTest(bad=bad), self.assertRaises(OOMProbeError):
                validate_same_service_recovery(bad,source_snapshot='P123',
                    source_payload_sha256='a'*64,http_pid=2000,old_native_pid=2100)

    def test_restart_branch_must_be_independent_and_still_at_source_p(self):
        base=evidence()['clean_restart']
        self.assertEqual(validate_clean_restart_branch(base,source_snapshot='P123',
            source_payload_sha256='a'*64)['source_tip'],'P123')
        advanced=copy.deepcopy(base); advanced['branch_before']['tip']='AFTER_FAULT'
        same=copy.deepcopy(base); same['branch_before']['branch']=same['fault_branch']
        for bad in (advanced,same):
            with self.subTest(bad=bad), self.assertRaises(OOMProbeError):
                validate_clean_restart_branch(bad,source_snapshot='P123',source_payload_sha256='a'*64)

    def test_clean_restart_requires_eight_tokens_zero_prefix_and_unchanged_source(self):
        base=evidence()['clean_restart']
        variants=[]
        short=copy.deepcopy(base); short['followup']['answer']['tokens']=short['followup']['answer']['tokens'][:7]; variants.append(short)
        prefix=copy.deepcopy(base); prefix['followup']['counter_deltas']['prefill_tokens']=1; variants.append(prefix)
        not_consumed=copy.deepcopy(base); not_consumed['followup']['counter_deltas']['consumed_tokens']=7; variants.append(not_consumed)
        changed=copy.deepcopy(base); changed['source_after']['payload_sha256']='c'*64; variants.append(changed)
        for bad in variants:
            with self.subTest(bad=bad), self.assertRaises(OOMProbeError):
                validate_clean_restart_branch(bad,source_snapshot='P123',source_payload_sha256='a'*64)

    def test_final_result_requires_original_cu_mem_create_error(self):
        result=evidence()
        self.assertEqual(validate_recovery_result(result),'PASS')
        for bad in (
            {**result,'native_oom':{'response':{'status':'error','message':'generic failure'},'oom_failure_delta':1,
                'stats_after':{'vmm_last_failed_operation':'none','vmm_last_driver_error':0}}},
            {**result,'native_oom':{'response':{'status':'error','message':'CUDA error'},'oom_failure_delta':0,
                'stats_after':{'vmm_last_failed_operation':'cuMemCreate','vmm_last_driver_error':2}}},
            {**result,'native_oom':{'response':{'status':'complete','message':'CUDA error'},'oom_failure_delta':1,
                'stats_after':{'vmm_last_failed_operation':'cuMemCreate','vmm_last_driver_error':2}}},
            {**result,'status':'FAIL'},
            {**result,'fault_http':{'evidence':{'error':{'error':'Different native failure'}}}},
            {**result,'helper':{'released_before_gate_abort':False,'exit_code':0,'pid_absent':True}},
            {**result,'helper':{'released_before_gate_abort':True,'exit_code':9,'pid_absent':True}},
            {**result,'source_after_all':{'head':{'tip':'P123','revision':4},'payload_sha256':'a'*64,
                'record_sha256':'b'*64,'checkpoint':{'P123.json':{'sha256':'DIFFERENT'}}}},
        ):
            with self.subTest(bad=bad), self.assertRaises(OOMProbeError):
                validate_recovery_result(bad)

    def test_trace_proves_old_exit_precedes_replacement_and_new_worker_closes_cleanly(self):
        trace=evidence()['first_service']['trace_document']
        self.assertEqual(validate_worker_transition_trace(trace,old_pid=2100,new_pid=2200,
            http_pid=2000)['restart_count'],1)
        replacements=[e for e in trace['events'] if e['event']=='native_worker_replaced']
        closes=[e for e in trace['events'] if e['event']=='native_worker_closed']
        for bad in (
            {'event_count':0,'overflowed':False,'events':[],'process_id':2000},
            dict(trace,events=[dict(closes[0],sequence=2),dict(replacements[0],sequence=1),closes[1]]),
            dict(trace,events=[closes[0],replacements[0],dict(closes[1],returncode=-9)]),
            dict(trace,process_id=3000),
            {'event_count':len(trace['events'])+1,'overflowed':False,'events':trace['events'],'process_id':2000},
            dict(trace,overflowed=True),
        ):
            with self.subTest(bad=bad), self.assertRaises(OOMProbeError):
                validate_worker_transition_trace(bad,old_pid=2100,new_pid=2200,http_pid=2000)

    def test_final_result_requires_distinct_clean_service_and_zero_exit(self):
        result=evidence()
        self.assertEqual(validate_recovery_result(result),'PASS')
        samepid=copy.deepcopy(result); samepid['clean_restart_service']['http_pid']=2000
        nonzero=copy.deepcopy(result); nonzero['clean_restart_service']['stop']['exit_code']=1
        live=copy.deepcopy(result); live['clean_restart_service']['stop']['pid_absent_after_stop']=False
        first_nonzero=copy.deepcopy(result); first_nonzero['first_service']['stop']['exit_code']=-9
        for bad in (samepid,nonzero,live,first_nonzero):
            with self.subTest(bad=bad), self.assertRaises(OOMProbeError):
                validate_recovery_result(bad)

    def test_durable_receipt_must_bind_exact_branch_tip_revision_and_answer(self):
        follow=evidence()['same_service']['followup']
        self.assertEqual(validate_durable_completion(follow['events'],
            branch_before=follow['branch_before'],branch_after=follow['branch_after'],
            branch_name=follow['branch'])['snapshot'],'SAME456')
        mutations=[]
        missing=copy.deepcopy(follow['events']); missing[-1].pop('receipt'); mutations.append(missing)
        wrong_tip=copy.deepcopy(follow['events']); wrong_tip[-1]['receipt']['snapshot']='OTHER'; mutations.append(wrong_tip)
        wrong_revision=copy.deepcopy(follow['events']); wrong_revision[-1]['receipt']['branch']['revision']=8; mutations.append(wrong_revision)
        wrong_branch=copy.deepcopy(follow['events']); wrong_branch[-1]['receipt']['branch']['name']='RecoveryBranch'; mutations.append(wrong_branch)
        missing_durable=copy.deepcopy(follow['events']); missing_durable.pop(1); mutations.append(missing_durable)
        duplicate_durable=copy.deepcopy(follow['events']); duplicate_durable.insert(2,copy.deepcopy(duplicate_durable[1])); mutations.append(duplicate_durable)
        durable_after_complete=copy.deepcopy(follow['events']); durable_after_complete[1],durable_after_complete[2]=durable_after_complete[2],durable_after_complete[1]; mutations.append(durable_after_complete)
        mismatched_durable=copy.deepcopy(follow['events']); mismatched_durable[1]['receipt']['snapshot']='OTHER'; mutations.append(mismatched_durable)
        extra_ack=copy.deepcopy(follow['events']); extra_ack.append({'event':'durable','receipt':extra_ack[-1]['receipt']}); mutations.append(extra_ack)
        duplicate=copy.deepcopy(follow['events']); duplicate.append(copy.deepcopy(duplicate[-1])); mutations.append(duplicate)
        wrong_head=copy.deepcopy(follow['events'])
        bad_after=dict(follow['branch_after'],tip='NOT_THE_RECEIPT')
        for events in mutations:
            with self.subTest(events=events), self.assertRaises(OOMProbeError):
                validate_durable_completion(events,branch_before=follow['branch_before'],
                    branch_after=follow['branch_after'],branch_name=follow['branch'])
        with self.assertRaises(OOMProbeError):
            validate_durable_completion(wrong_head,branch_before=follow['branch_before'],
                branch_after=bad_after,branch_name=follow['branch'])

    def test_profile_pinned_http_source_must_match_independent_pin(self):
        path='/runtime/serve/cow_http.py'; digest='d'*64
        self.assertEqual(validate_http_source_binding(path,digest,{path:digest},digest)['binding'],
                         'same_file_in_selected_profile_closure')
        self.assertEqual(validate_http_source_binding(path,digest,{},digest)['binding'],
                         'separate_explicit_service_source_pin')
        with self.assertRaises(OOMProbeError):
            validate_http_source_binding(path,digest,{path:'e'*64},digest)
        with self.assertRaises(OOMProbeError):
            validate_http_source_binding(path,digest,{},'f'*64)

    def test_terminal_persistence_fails_closed_on_cleanup_or_heartbeat_failure(self):
        class Progress:
            def __init__(self,fail_close=False): self.status=None; self.fail_close=fail_close
            def finish(self,status,error=None): self.status=status
            def close(self):
                if self.fail_close: raise RuntimeError('heartbeat did not stop')
        with tempfile.TemporaryDirectory() as temp:
            out=Path(temp)/'result.json'; progress=Progress()
            result={'status':'PASS'}
            error=base._finish_and_persist(result,out,progress,[{'resource':'service'}],None)
            self.assertIsNotNone(error)
            written=json.loads(out.read_text(encoding='utf-8'))
            self.assertEqual(written['status'],'FAIL')
            self.assertIn('cleanup_failures',written)
        with tempfile.TemporaryDirectory() as temp:
            out=Path(temp)/'result.json'; progress=Progress(fail_close=True)
            result={'status':'PASS'}
            error=base._finish_and_persist(result,out,progress,[],None)
            self.assertIsNotNone(error)
            written=json.loads(out.read_text(encoding='utf-8'))
            self.assertEqual(written['status'],'FAIL')
            self.assertTrue(any(x['resource']=='progress_close' for x in written['cleanup_failures']))

    def test_actual_recovery_preflight_accepts_coherent_pinned_runtime(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); runtime=root/'runtime'
            api=runtime/'experimental/branching_context/src/context_time_travel'
            scripts=runtime/'experimental/branching_context/scripts'; serve=runtime/'serve'
            store=root/'store'; native=root/'native'; helper=root/'pressure-helper'; profile=root/'profile.json'
            for directory in (api,scripts,serve,store/'checkpoints'):
                directory.mkdir(parents=True,exist_ok=True)
            (scripts/'run_cow_client_recovery.py').write_text('# pinned helper\n',encoding='utf-8')
            http=serve/'cow_http.py'; http.write_text('# pinned service\n',encoding='utf-8')
            (store/'refs.json').write_text('{}\n',encoding='utf-8')
            native.write_bytes(b'native'); helper.write_bytes(b'helper')
            modules=('real.py','coordinator.py','strata_backend.py','execution_receipt.py','checkpoint_cache.py')
            module_paths=[]
            for name in modules:
                path=api/name; path.write_text(f'# {name}\n',encoding='utf-8'); module_paths.append(path)
            profile.write_text(json.dumps({'execution':{'GPU':[0],'capacity':98304,
                'prefill_chunk':8192,'kv_dtype':'FP16','sut_command':[str(native)]}}),encoding='utf-8')
            closure={'files':{str(i):{'path':str(path.resolve()),'sha256':base._sha256(path)}
                for i,path in enumerate([*module_paths,http])}}
            args=SimpleNamespace(profile=str(profile),store=str(store),runtime_root=str(runtime),
                api_src=str(runtime/'experimental/branching_context/src'),api_scripts=str(scripts),
                pressure_helper=str(helper),expected_profile_sha256=base._sha256(profile),
                expected_helper_sha256=base._sha256(helper),expected_native_sha256=base._sha256(native),
                expected_http_sha256=base._sha256(http),branch='CliA',expected_source_snapshot='a'*64)
            with patch.object(publication,'_runtime_closure',return_value=closure), \
                 patch.dict(sys.modules,{'context_time_travel':types.ModuleType('context_time_travel'),
                    'context_time_travel.store':types.ModuleType('context_time_travel.store')}):
                sys.modules['context_time_travel.store'].Store=lambda _path:MemoryStore()
                prepared=_recovery_preflight(args)
            self.assertEqual(prepared['service_build_source']['binding'],
                             'same_file_in_selected_profile_closure')
            self.assertEqual(prepared['expected_imports'][str(http.resolve())],base._sha256(http))
            self.assertEqual(len(prepared['profile_api_imports']),5)
            self.assertEqual(prepared['source_head']['tip'],'a'*64)
            self.assertEqual(prepared['source_history']['controller_status'],'active')
            self.assertEqual(prepared['source_history']['token_count'],3)

    def test_token_only_store_history_is_readable_without_presentation_namespace(self):
        store=MemoryStore(tip='a'*64)
        class Client:
            def get(self,path):
                self.asserted_path=path
                return 200,{'name':'CliA','revision':0,'tip':'a'*64,'root':'a'*64}
        client=Client()
        head,history=recovery._head_history(client,store,'CliA')
        self.assertEqual(client.asserted_path,'/cow/head?branch=CliA')
        self.assertEqual(head['tip'],'a'*64)
        self.assertEqual(history['controller_status'],'active')
        self.assertEqual(history['token_count'],3)
        self.assertEqual(store.leases,['a'*64])

    def test_token_only_history_rejects_stale_http_head(self):
        store=MemoryStore(tip='a'*64)
        class Client:
            def get(self,path): return 200,{'name':'CliA','revision':9,'tip':'a'*64,'root':'a'*64}
        with self.assertRaises(OOMProbeError):
            recovery._head_history(Client(),store,'CliA')

    def test_token_only_fork_uses_snapshot_request_and_checks_returned_tip(self):
        class Client:
            def __init__(self): self.calls=[]
            def post_json(self,path,body):
                self.calls.append((path,body))
                return 200,{'branch':{'name':'RecoveryBranch','tip':'a'*64,'revision':0}}
        client=Client()
        head=recovery._fork_token_branch(client,'a'*64,'RecoveryBranch')
        self.assertEqual(client.calls,[('/cow/fork',{'name':'RecoveryBranch','snapshot':'a'*64})])
        self.assertEqual(head,{'name':'RecoveryBranch','tip':'a'*64,'revision':0})

        class WrongTip(Client):
            def post_json(self,path,body):
                self.calls.append((path,body))
                return 200,{'branch':{'name':'RecoveryBranch','tip':'b'*64,'revision':0}}
        with self.assertRaises(OOMProbeError):
            recovery._fork_token_branch(WrongTip(),'a'*64,'RecoveryBranch')

    def test_cleanup_stop_receipt_derives_pid_absence_from_owned_process(self):
        class Process:
            pid=2147483647
        class Service:
            process=Process()
            def stop(self): return {'exit_code':0,'scope':'owned-http-service'}
        receipt=_service_stop_receipt(Service())
        self.assertEqual(receipt['exit_code'],0)
        self.assertEqual(receipt['pid'],2147483647)
        self.assertTrue(receipt['pid_absent_after_stop'])
        with patch.object(Path,'exists',return_value=True):
            still_live=_service_stop_receipt(Service())
        self.assertFalse(still_live['pid_absent_after_stop'])


if __name__=='__main__':
    unittest.main()
