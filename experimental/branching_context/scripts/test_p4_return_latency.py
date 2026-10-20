"""CPU-only tests for the P4 repeated-return harness; no model/GPU imports."""
import hashlib
import importlib
import io
import json
import os
import queue
from pathlib import Path
import stat
import struct
import tempfile
import threading
import time
import unittest
from concurrent.futures import Future
from types import SimpleNamespace

from context_time_travel.store import IntegrityError, Store, _json_bytes

try:
    sut = importlib.import_module('p4_return_latency')
    IMPORT_ERROR = None
except ImportError as error:
    sut = None
    IMPORT_ERROR = error


class ReturnHarnessTests(unittest.TestCase):
    def require_sut(self):
        if sut is None:
            self.fail('P4 return harness is not implemented: '+repr(IMPORT_ERROR))

    def make_source(self, base, *, checkpoint=True, operator_schedule=False):
        source = Path(base)/'source'
        store = Store(source)
        contract = 'contract-test'
        root = store.create_root(contract, [10, 11, 12], {'status':'idle','policy':{'mode':'greedy'}})
        leaf = store.append(root, [13, 14], {'status':'idle','policy':{'mode':'greedy'}})
        store.create_root(contract, [77, 78], {'status':'unrelated'})
        snapshot = leaf
        history = [10, 11, 12, 13, 14]
        state_bytes = b'fixture-native-state'
        frontier = struct.pack('<2f', 0.5, -0.25)
        payload = state_bytes+frontier
        payload_sha = hashlib.sha256(payload).hexdigest()
        if operator_schedule:
            from context_time_travel.execution_receipt import schedule_digest, tokens_digest
            schedule = [dict(operator='batched-prefill',position=0,count=len(history),
                             tokens_sha256=tokens_digest(history),chunks=[[0,len(history)]])]
            schema = 3
        else:
            schedule = None
            schema = 1
        manifest = dict(schema=1,snapshot=snapshot,
            input_identity=hashlib.sha256(_json_bytes(history)).hexdigest(),
            model_contract_id=contract,execution_profile_id='profile-test',
            consumed=len(history),positions='absolute-zero-based',logits_count=2,
            state_bytes=len(state_bytes),payload_sha256=payload_sha)
        manifest['schema'] = schema
        if operator_schedule:
            manifest['execution_schedule'] = schedule
            manifest['schedule_sha256'] = schedule_digest(schedule)
        envelope = dict(manifest=manifest,
            checksum=hashlib.sha256(_json_bytes(manifest)).hexdigest())
        if checkpoint:
            cp = source/'checkpoints'
            cp.mkdir()
            (cp/(snapshot+'.json')).write_bytes(_json_bytes(envelope))
            (cp/(payload_sha+'.state')).write_bytes(payload)
        profile = dict(model_contract_id=contract,execution_profile_id='profile-test',
            execution={'capacity':64,'prefill_chunk':8,'resume_contract':'legacy'})
        if operator_schedule:
            profile['execution']['resume_contract'] = 'identical-native-operators-v1'
        root_state = dict(snapshot=snapshot,history=history,
            controller={'status':'idle','policy':{'mode':'greedy'}},manifest=manifest)
        return source, snapshot, root_state, profile

    def test_import_preserves_canonical_ids_history_and_only_requested_root(self):
        self.require_sut()
        with tempfile.TemporaryDirectory() as temp:
            source, snapshot, state, profile = self.make_source(temp)
            result = sut.import_minimal_store(source, Path(temp)/'target', [state], profile)
            target = Store(Path(temp)/'target')
            self.assertEqual(result['snapshots'], [snapshot])
            self.assertEqual(target.lineage(snapshot)[1], state['history'])
            self.assertEqual(target.get_snapshot(snapshot)['controller'], state['controller'])
            self.assertEqual({p.stem for p in target.objects.glob('*.json')},
                             {snapshot, target.get_snapshot(snapshot)['parent']})
            self.assertEqual(sorted(p.name for p in (Path(temp)/'target'/'checkpoints').iterdir()),
                             sorted([snapshot+'.json',state['manifest']['payload_sha256']+'.state']))

    def test_operator_schedule_is_bound_to_exact_history_and_profile(self):
        self.require_sut()
        with tempfile.TemporaryDirectory() as temp:
            source, snapshot, state, profile = self.make_source(temp, operator_schedule=True)
            target = Path(temp)/'target'
            result = sut.import_minimal_store(source, target, [state], profile)
            self.assertEqual(result['snapshots'], [snapshot])
            state['manifest']['execution_schedule'][0]['chunks'] = [[0, 2], [2, 3]]
            bad_target = Path(temp)/'bad-target'
            with self.assertRaises((IntegrityError, ValueError)):
                sut.import_minimal_store(source, bad_target, [state], profile)
            self.assertFalse(bad_target.exists())

    def test_missing_checkpoint_fails_before_target_becomes_usable(self):
        self.require_sut()
        with tempfile.TemporaryDirectory() as temp:
            source, _, state, profile = self.make_source(temp, checkpoint=False)
            target = Path(temp)/'target'
            with self.assertRaises((FileNotFoundError, ValueError)):
                sut.import_minimal_store(source, target, [state], profile)
            self.assertFalse(target.exists())

    def test_profile_mismatch_fails_before_target_becomes_usable(self):
        self.require_sut()
        with tempfile.TemporaryDirectory() as temp:
            source, _, state, profile = self.make_source(temp)
            profile['execution_profile_id'] = 'wrong-profile'
            target = Path(temp)/'target'
            with self.assertRaises(ValueError):
                sut.import_minimal_store(source, target, [state], profile)
            self.assertFalse(target.exists())

    def test_corrupt_checkpoint_payload_fails_before_target_publish(self):
        self.require_sut()
        with tempfile.TemporaryDirectory() as temp:
            source, _, state, profile = self.make_source(temp)
            payload = source/'checkpoints'/(state['manifest']['payload_sha256']+'.state')
            payload.write_bytes(payload.read_bytes()+b'corruption')
            target = Path(temp)/'target'
            with self.assertRaises(IntegrityError):
                sut.import_minimal_store(source, target, [state], profile)
            self.assertFalse(target.exists())

    def test_target_cannot_be_created_inside_frozen_source_store(self):
        self.require_sut()
        with tempfile.TemporaryDirectory() as temp:
            source, _, state, profile = self.make_source(temp)
            with self.assertRaises(ValueError):
                sut.import_minimal_store(source, source/'derived', [state], profile)
            self.assertFalse((source/'derived').exists())

    def test_readonly_source_bytes_and_modes_remain_unchanged(self):
        self.require_sut()
        with tempfile.TemporaryDirectory() as temp:
            source, _, state, profile = self.make_source(temp)
            paths = [p for p in source.rglob('*') if p.is_file()]
            before = {str(p.relative_to(source)):(hashlib.sha256(p.read_bytes()).hexdigest(),
                                                   stat.S_IMODE(p.stat().st_mode)) for p in paths}
            for path in paths:
                path.chmod(0o444)
            for directory in sorted((p for p in source.rglob('*') if p.is_dir()),
                                    key=lambda p: len(p.parts), reverse=True):
                directory.chmod(0o555)
            source.chmod(0o555)
            try:
                sut.import_minimal_store(source, Path(temp)/'target', [state], profile)
                after = {str(p.relative_to(source)):(hashlib.sha256(p.read_bytes()).hexdigest(),
                                                      stat.S_IMODE(p.stat().st_mode)) for p in paths}
                self.assertEqual(after, {str(Path(k)):(v[0],0o444) for k,v in before.items()})
            finally:
                source.chmod(0o755)
                for directory in source.rglob('*'):
                    if directory.is_dir(): directory.chmod(0o755)
                for path in paths:
                    path.chmod(0o600)

    def test_tier_classification_ignores_native_loaded_file_counter(self):
        self.require_sut()
        key = ('contract','profile','snapshot')
        resident = sut.classify_checkpoint_tier(key, pool_contains=True,
            model_cache_contains=False, events=[], native_counters={'loaded_file_bytes':8192})
        self.assertEqual(resident, 'resident_pool')
        trace = [dict(sequence=4,event='disk_preload_started',key=list(key),load_id=4,
                      process_id=12,phase='test',phase_id='phase-a'),
                 dict(sequence=5,event='disk_preload_succeeded',key=list(key),load_id=4,
                      process_id=12,phase='test',phase_id='phase-a',success=True,bytes=1024)]
        disk = sut.classify_checkpoint_tier(key, pool_contains=False,
            model_cache_contains=False, events=trace, native_counters={'loaded_file_bytes':8192})
        self.assertEqual(disk, 'disk_preload')
        with self.assertRaises(ValueError):
            sut.classify_checkpoint_tier(key, pool_contains=False,
                model_cache_contains=True, events=trace, native_counters={'loaded_file_bytes':0})
        reversed_trace = [dict(trace[0],sequence=6),dict(trace[1],sequence=5)]
        with self.assertRaises(ValueError):
            sut.classify_checkpoint_tier(key,pool_contains=False,
                model_cache_contains=False,events=reversed_trace)
        cross_process = [trace[0],dict(trace[1],process_id=99)]
        with self.assertRaises(ValueError):
            sut.classify_checkpoint_tier(key,pool_contains=False,
                model_cache_contains=False,events=cross_process)

    def test_heartbeat_reports_phase_root_repeat_pid_and_stops(self):
        self.require_sut()
        with tempfile.TemporaryDirectory() as temp:
            beats=[]
            heartbeat=sut.RunHeartbeat(Path(temp)/'heartbeat.json',interval=.01,emit=beats.append).start()
            for repeat in range(8):
                heartbeat.update('measured_request',root_index=2,repeat=repeat)
                time.sleep(.003)
            time.sleep(.04)
            heartbeat.close()
            payload=json.loads((Path(temp)/'heartbeat.json').read_text())
            self.assertEqual(payload['phase'],'measured_request')
            self.assertEqual(payload['root_index'],2)
            self.assertEqual(payload['repeat'],7)
            self.assertEqual(payload['pid'],os.getpid())
            self.assertGreaterEqual(payload['elapsed_seconds'],0)
            self.assertTrue(beats)
            self.assertFalse(heartbeat._thread.is_alive())

    def test_heartbeat_worker_failure_is_surfaced_on_close(self):
        self.require_sut()
        def failed_emit(_message):
            raise OSError('injected heartbeat output failure')
        with tempfile.TemporaryDirectory() as temp:
            heartbeat=sut.RunHeartbeat(Path(temp)/'heartbeat.json',
                                       interval=.01,emit=failed_emit).start()
            try:
                deadline=time.monotonic()+2
                while heartbeat._thread.is_alive() and time.monotonic()<deadline:
                    if heartbeat._thread_error is not None: break
                    time.sleep(.005)
                with self.assertRaisesRegex(RuntimeError,'heartbeat worker failed'):
                    heartbeat.close()
                self.assertIsInstance(heartbeat._thread_error,OSError)
                self.assertFalse(heartbeat._thread.is_alive())
            finally:
                if heartbeat._thread.is_alive(): heartbeat.close()

    def test_cli_returns_nonzero_for_fail_and_preserves_receipt_summary(self):
        self.require_sut()
        args=['--source-store','unused','--source-result','unused','--prepare','unused',
            '--profile','unused','--fixture','unused','--reference-dir','unused',
            '--output','out','--arm','disk','--arm-order','resident,disk',
            '--order-seed','seed','--ram-bytes','1','--disk-bytes','1',
            '--min-available-bytes','0']
        report={'status':'FAIL','measured_requests':9,'samples':[{}],
            'native_exit_observed':False,'error':'native worker exit unobserved'}
        output=io.StringIO()
        code=sut.main(args,runner=lambda _parsed:report,stdout=output)
        self.assertEqual(code,1)
        receipt=json.loads(output.getvalue())
        self.assertEqual(receipt['status'],'FAIL')
        self.assertEqual(receipt['measured_requests'],9)
        self.assertIs(receipt['native_exit_observed'],False)
        self.assertEqual(receipt['error'],report['error'])
        report['status']='PASS'
        self.assertEqual(sut.main(args,runner=lambda _parsed:report,stdout=io.StringIO()),0)

    def test_campaign_inputs_requires_native_prefix_and_explicit_supported_roots(self):
        self.require_sut()
        with tempfile.TemporaryDirectory() as temp:
            source,source_result,prepare,profile,fixture,reference,_ = self.make_fake_campaign(temp)
            with self.assertRaisesRegex(ValueError,'history must contain exactly 65536'):
                sut._campaign_inputs(source,source_result,prepare,profile,fixture,reference)
            roots,*_ = sut._campaign_inputs(source,source_result,prepare,profile,fixture,reference,
                                             expected_prefix_tokens=256)
            self.assertEqual([len(root['history']) for root in roots],[256,256,256])
            with self.assertRaisesRegex(ValueError,'requires an injected CPU fake'):
                sut.run_arm(SimpleNamespace(source_store=source,
                    output=Path(temp)/'production-override',test_prefix_tokens=256))
            fixture_data=json.loads(fixture.read_text())
            fixture_data.pop('tree_roots')
            fixture.write_text(json.dumps(fixture_data))
            with self.assertRaisesRegex(ValueError,'explicitly map all six'):
                sut._campaign_inputs(source,source_result,prepare,profile,fixture,reference,
                                     expected_prefix_tokens=256)
            fixture_data['tree_roots']=[0,0,1,1,2,2]
            fixture_data['suffixes'].pop()
            fixture.write_text(json.dumps(fixture_data))
            with self.assertRaisesRegex(ValueError,'exactly six'):
                sut._campaign_inputs(source,source_result,prepare,profile,fixture,reference,
                                     expected_prefix_tokens=256)

    def test_producer_and_consumer_times_are_separate(self):
        self.require_sut()
        timeline = sut.EventTimeline()
        timeline.accept(dict(event='request_started',time=10.0),consumer_monotonic=10.2)
        timeline.accept(dict(event='token',time=10.4,token=7),consumer_monotonic=10.7)
        timeline.accept(dict(event='token',time=10.5,token=8),consumer_monotonic=10.9)
        timeline.accept(dict(event='snapshot',time=11.0),consumer_monotonic=11.3)
        timeline.accept(dict(event='durable',time=11.2),consumer_monotonic=11.6)
        summary = timeline.summary(submit_monotonic=9.8)
        self.assertAlmostEqual(summary['submit_to_first_token_seconds'],0.9)
        self.assertAlmostEqual(summary['producer_token_gap_seconds'][0],0.1)
        self.assertAlmostEqual(summary['consumer_token_gap_seconds'][0],0.2)
        self.assertAlmostEqual(summary['producer_snapshot_to_durable_seconds'],0.2)

    def test_work_delta_rejects_saved_prefix_forward_work(self):
        self.require_sut()
        self.assertTrue(sut.validate_work_delta(8311, [3,4],
            {'forward_tokens':8313,'prefill_tokens':8311}, saved_prefix_tokens=65536))
        with self.assertRaises(ValueError):
            sut.validate_work_delta(8311, [3,4],
                {'forward_tokens':65536+8313,'prefill_tokens':65536+8311},
                saved_prefix_tokens=65536)

    def test_output_requires_first_reference_prefix_or_actual_eos(self):
        self.require_sut()
        reference = list(range(1,140))
        self.assertTrue(sut.validate_reference_output(reference[:128], reference, [999], 128))
        eos_ref = [21,22,999,23]
        self.assertTrue(sut.validate_reference_output(eos_ref[:3], eos_ref, [999], 128))
        with self.assertRaises(ValueError):
            sut.validate_reference_output([21,22], eos_ref, [999], 128)
        with self.assertRaises(ValueError):
            sut.validate_reference_output([1,3], reference, [999], 128)

    def make_fake_campaign(self, base):
        source = Path(base)/'campaign-store'
        store = Store(source)
        roots = []
        contract = 'fake-contract'
        profile_id = 'fake-profile'
        profile = dict(model_contract_id=contract, execution_profile_id=profile_id,
            model_weights_sha256='0'*64, native_build={'sha256':'1'*64},
            execution=dict(GPU=[0],capacity=65536,prefill_chunk=8,
                resume_contract='legacy',eos_tokens=[],environment={}))
        cp = source/'checkpoints'
        cp.mkdir()
        for index in range(3):
            controller = {'status':'idle','policy':{'eos_tokens':[]}}
            history = [index*256+token for token in range(256)]
            snapshot = store.create_root(contract,history,controller)
            state = b'fake-state-'+bytes([index])
            logits = struct.pack('<2f',.25,-.25)
            payload = state+logits
            payload_sha = hashlib.sha256(payload).hexdigest()
            manifest = dict(schema=1,snapshot=snapshot,
                input_identity=hashlib.sha256(_json_bytes(history)).hexdigest(),
                model_contract_id=contract,execution_profile_id=profile_id,
                consumed=len(history),positions='absolute-zero-based',logits_count=2,
                state_bytes=len(state),payload_sha256=payload_sha)
            envelope = dict(manifest=manifest,
                checksum=hashlib.sha256(_json_bytes(manifest)).hexdigest())
            (cp/(snapshot+'.json')).write_bytes(_json_bytes(envelope))
            (cp/(payload_sha+'.state')).write_bytes(payload)
            roots.append(dict(snapshot=snapshot,history=history,controller=controller,manifest=manifest))
        fixture = dict(suffixes=[[100+i] for i in range(6)],tree_roots=[0,0,1,1,2,2])
        reference_dir = Path(base)/'references'
        reference_dir.mkdir()
        for tree in (0,2,4):
            (reference_dir/f'fresh-tree-{tree}.stdout').write_text(
                json.dumps({'generated':list(range(1000+tree*128,1000+(tree+1)*128))}))
        prepare = Path(base)/'prepare.json'
        prepare.write_text(json.dumps({'status':'PASS','roots':roots}))
        profile_path = Path(base)/'profile.json'
        profile_path.write_text(json.dumps(profile))
        source_result = Path(base)/'source-result.json'
        source_result.write_text(json.dumps({'status':'PASS',
            'profile_id':profile_id,'model_contract_id':contract,
            'build_sha256':profile['native_build']['sha256'],
            'executed_test_ids':['FAKE_FIXTURE_ONLY']}))
        fixture_path = Path(base)/'fixture.json'
        fixture_path.write_text(json.dumps(fixture))
        return source,source_result,prepare,profile_path,fixture_path,reference_dir,profile

    def test_injected_fake_coordinator_runs_both_arm_control_flows(self):
        self.require_sut()
        with tempfile.TemporaryDirectory() as temp:
            source,source_result,prepare,profile_path,fixture_path,reference_dir,profile = self.make_fake_campaign(temp)
            payload_max=max(p.stat().st_size for p in (source/'checkpoints').glob('*.state'))

            class FakePool:
                def __init__(self): self.entries=set()
                def __contains__(self,key): return key in self.entries
                def remove(self,key): self.entries.remove(key)
                def metrics(self): return dict(resident_bytes=len(self.entries)*10,reserved_bytes=0,entries=len(self.entries))

            class FakeBackend:
                def __init__(self):
                    self.process=FakeProcess(FakeCoordinator.native_exit_code,
                        alive=FakeCoordinator.leave_native_alive)
                    self.counts={'forward_tokens':0,'prefill_tokens':0,'loaded_file_bytes':0}
                    self.owner=None
                def claim_worker(self): self.owner=threading.get_ident()
                def check_worker(self):
                    if threading.get_ident()!=self.owner:
                        raise AssertionError('native API called outside its owning worker thread')
                def work_counters(self): self.check_worker();return dict(self.counts)
                def restore(self,lease,blob,**kwargs): self.check_worker();return None
                def consume(self,lease,ids): self.check_worker();return None

            class FakeModel:
                def __init__(self):
                    self.profile=profile
                    self.backend=FakeBackend()
                    self.cache={}
                def _load_checkpoint(self,snapshot,history): return dict(snapshot=snapshot)

            class FakeProcess:
                def __init__(self,exit_code=0,alive=False): self.exit_code=exit_code;self.alive=alive
                def poll(self): return None if self.alive else self.exit_code

            class FakeCoordinator:
                leave_native_alive=False
                native_exit_code=0
                fail_ready=False
                fail_close=False
                def __init__(self,directory,config,trace):
                    self.store=Store(directory)
                    self.config=config
                    self.trace=trace
                    self.pool=FakePool()
                    self._condition=threading.Condition()
                    self._workers={0:{'model':FakeModel(),'busy':False,'active':0}}
                    self._loads={}
                def ready(self):
                    if self.__class__.fail_ready: raise RuntimeError('injected request/startup failure')
                def metrics(self):
                    return dict(ram=self.pool.metrics(),disk={'file_bytes':0},GPU=[0])
                def _run(self,model,job): return None
                def _preload(self,model,snapshot,**kwargs):
                    key=(profile['model_contract_id'],profile['execution_profile_id'],snapshot)
                    if key in self.pool: return None
                    if key in self._loads:
                        pending=self._loads[key]
                        if not pending.done(): return pending
                        self._loads.pop(key)
                    load_id=self.trace.record('disk_preload_started',key)
                    future=Future();self._loads[key]=future
                    def load():
                        time.sleep(.01)
                        model._load_checkpoint(snapshot,self.store.lineage(snapshot)[1])
                        self.pool.entries.add(key)
                        self.trace.record('disk_preload_succeeded',key,success=True,bytes=512,load_id=load_id)
                        future.set_result(True)
                    threading.Thread(target=load,daemon=True).start()
                    return future
                def submit(self,branch,ids,max_steps,operation):
                    handle=SimpleNamespace(operation=operation,events=queue.SimpleQueue(),
                        snapshot=Future(),durable=Future())
                    model=self._workers[0]['model']
                    source=self.store.head(branch).tip
                    with self._condition:
                        self._workers[0]['busy']=True
                        self._workers[0]['active']=1
                    def work():
                        try:
                            model.backend.claim_worker()
                            self._run(model,SimpleNamespace(operation=operation))
                            preload=self._preload(model,source)
                            if preload is not None:
                                self._preload(model,source)  # Same in-flight load, second waiter.
                                preload.result(timeout=5)
                            handle.events.put(dict(event='request_started',time=time.monotonic()))
                            tree=next(i for i in (0,2,4) if [100+i] == list(ids))
                            tokens=list(range(1000+tree*128,1000+(tree+1)*128))[:max_steps]
                            for token in tokens:
                                handle.events.put(dict(event='token',time=time.monotonic(),token=token,turn_complete=False))
                            model.backend.counts['forward_tokens']+=len(ids)+len(tokens)
                            model.backend.counts['prefill_tokens']+=len(ids)
                            model.backend.restore(None,b'fake')
                            model.backend.consume(None,ids)
                            answer_snapshot=self.store.append(source,list(ids)+tokens,
                                self.store.get_snapshot(source)['controller'])
                            answer=dict(snapshot=answer_snapshot,tokens=tokens,generation_seconds=.01,
                                request_seconds=.02,capture_seconds=.003,
                                counter_deltas={'forward_tokens':len(ids)+len(tokens),
                                    'prefill_tokens':len(ids),'loaded_file_bytes':0})
                            receipt=dict(snapshot=answer_snapshot)
                            handle.events.put(dict(event='snapshot',time=time.monotonic(),**answer))
                            handle.snapshot.set_result(answer)
                            handle.events.put(dict(event='durable',time=time.monotonic(),receipt=receipt))
                            handle.durable.set_result(receipt)
                        except BaseException as error:
                            handle.snapshot.set_exception(error)
                            handle.durable.set_exception(error)
                        finally:
                            with self._condition:
                                self._workers[0]['active']=0
                                self._workers[0]['busy']=False
                                self._condition.notify_all()
                    threading.Thread(target=work,daemon=True).start()
                    return handle
                def close(self):
                    if self.__class__.fail_close: raise RuntimeError('injected coordinator close failure')

            free=15*payload_max+10*1024**3
            for arm in ('resident','disk'):
                arguments=SimpleNamespace(source_store=source,source_result=source_result,
                    prepare=prepare,profile=profile_path,
                    fixture=fixture_path,reference_dir=reference_dir,output=Path(temp)/(arm+'-run'),
                    arm=arm,arm_order='resident,disk',order_seed='test-seed',
                    ram_bytes=15*payload_max,disk_bytes=15*payload_max,
                    min_available_bytes=0,preflight_free_bytes_for_test=free,
                    test_prefix_tokens=256)
                result=sut.run_arm(arguments,coordinator_factory=FakeCoordinator)
                self.assertEqual(result['status'],'PASS')
                self.assertEqual(len(result['samples']),9)
                if arm=='disk':
                    for sample in result['samples']:
                        invocations=sample['stages']['pool_preload_invocations']
                        self.assertEqual(len(invocations),2)
                        self.assertEqual(invocations[0]['future_identity'],invocations[1]['future_identity'])
                        self.assertEqual([item['kind'] for item in invocations],
                                         ['first_call_for_future','shared_future_wait'])
                        self.assertTrue(all(item['completed_monotonic'] is not None and
                                            item['elapsed_seconds'] >= 0 for item in invocations))
                expected='resident_pool' if arm=='resident' else 'disk_preload'
                self.assertEqual({sample['tier'] for sample in result['samples']},{expected})
                self.assertTrue(result['native_exit_observed'])
                heartbeat=json.loads((Path(temp)/(arm+'-run')/'heartbeat.json').read_text())
                self.assertEqual(heartbeat['phase'],'cleanup')
                self.assertEqual(heartbeat['pid'],os.getpid())
            FakeCoordinator.leave_native_alive=True
            arguments=SimpleNamespace(source_store=source,source_result=source_result,
                prepare=prepare,profile=profile_path,fixture=fixture_path,
                reference_dir=reference_dir,output=Path(temp)/'unobserved-native-exit',
                arm='resident',arm_order='resident,disk',order_seed='test-seed',
                ram_bytes=15*payload_max,disk_bytes=15*payload_max,
                min_available_bytes=0,preflight_free_bytes_for_test=free,
                test_prefix_tokens=256)
            result=sut.run_arm(arguments,coordinator_factory=FakeCoordinator)
            self.assertEqual(result['status'],'FAIL')
            self.assertIs(result['native_exit_observed'],False)
            unobserved_result=result
            FakeCoordinator.leave_native_alive=False
            FakeCoordinator.native_exit_code=-9
            arguments.output=Path(temp)/'unsuccessful-native-exit'
            result=sut.run_arm(arguments,coordinator_factory=FakeCoordinator)
            self.assertEqual(result['status'],'FAIL')
            self.assertIs(result['native_exit_observed'],True)
            self.assertEqual(result['native_exit_codes'],[-9])
            self.assertIn('successfully',result['error'])
            FakeCoordinator.native_exit_code=0
            FakeCoordinator.leave_native_alive=True
            cli_args=['--source-store',str(source),'--source-result',str(source_result),
                '--prepare',str(prepare),'--profile',str(profile_path),'--fixture',str(fixture_path),
                '--reference-dir',str(reference_dir),'--output','unused','--arm','resident',
                '--arm-order','resident,disk','--order-seed','test-seed',
                '--ram-bytes',str(15*payload_max),'--disk-bytes',str(15*payload_max),
                '--min-available-bytes','0']
            cli_output=io.StringIO()
            self.assertEqual(sut.main(cli_args,runner=lambda _parsed:unobserved_result,stdout=cli_output),1)
            cli_receipt=json.loads(cli_output.getvalue())
            self.assertIs(cli_receipt['native_exit_observed'],False)
            self.assertEqual(cli_receipt['measured_requests'],9)
            FakeCoordinator.leave_native_alive=False
            FakeCoordinator.fail_close=True
            arguments.output=Path(temp)/'close-failure-after-pass'
            with self.assertRaisesRegex(RuntimeError,'coordinator cleanup failed'):
                sut.run_arm(arguments,coordinator_factory=FakeCoordinator)
            close_receipt=json.loads((arguments.output/'result.json').read_text())
            self.assertEqual(close_receipt['status'],'FAIL')
            self.assertEqual(len(close_receipt['samples']),9)
            self.assertIs(close_receipt['native_exit_observed'],True)
            self.assertTrue(any('injected coordinator close failure' in error
                                for error in close_receipt['cleanup_errors']))

    def test_cleanup_failure_is_recorded_without_masking_original_failure(self):
        self.require_sut()
        with tempfile.TemporaryDirectory() as temp:
            source,source_result,prepare,profile_path,fixture_path,reference_dir,profile = self.make_fake_campaign(temp)
            payload_max=max(p.stat().st_size for p in (source/'checkpoints').glob('*.state'))
            output=Path(temp)/'cleanup-failure'
            arguments=SimpleNamespace(source_store=source,source_result=source_result,
                prepare=prepare,profile=profile_path,fixture=fixture_path,
                reference_dir=reference_dir,output=output,arm='disk',
                arm_order='resident,disk',order_seed='test-seed',
                ram_bytes=15*payload_max,disk_bytes=15*payload_max,
                min_available_bytes=0,preflight_free_bytes_for_test=15*payload_max+10*1024**3,
                test_prefix_tokens=256)
            class BrokenCoordinator:
                def __init__(self,directory,config,trace): self._workers={}
                def ready(self): raise RuntimeError('original startup error')
                def close(self): raise RuntimeError('cleanup close error')
            with self.assertRaisesRegex(RuntimeError,'original startup error'):
                sut.run_arm(arguments,coordinator_factory=BrokenCoordinator)
            receipt=json.loads((output/'result.json').read_text())
            self.assertEqual(receipt['status'],'FAIL')
            self.assertEqual(receipt['failed_phase'],'coordinator_startup')
            self.assertIn('original startup error',receipt['error'])
            self.assertTrue(any('cleanup close error' in error for error in receipt['cleanup_errors']))
            self.assertIs(receipt['native_exit_observed'],True)

    def test_cli_exception_uses_persisted_failure_receipt(self):
        self.require_sut()
        with tempfile.TemporaryDirectory() as temp:
            output_path=Path(temp)/'run'
            output_path.mkdir()
            report={'status':'FAIL','measured_requests':7,'samples':[{},{}],
                    'native_exit_observed':False,'error':'native worker process exit was not observed'}
            (output_path/'result.json').write_text(json.dumps(report))
            args=['--source-store','unused','--source-result','unused','--prepare','unused',
                '--profile','unused','--fixture','unused','--reference-dir','unused',
                '--output',str(output_path),'--arm','disk','--arm-order','resident,disk',
                '--order-seed','seed','--ram-bytes','1','--disk-bytes','1',
                '--min-available-bytes','0']
            def failing_runner(_parsed): raise RuntimeError('cleanup raised after report write')
            output=io.StringIO()
            code=sut.main(args,runner=failing_runner,stdout=output)
            self.assertEqual(code,1)
            summary=json.loads(output.getvalue())
            self.assertEqual(summary['measured_requests'],7)
            self.assertIs(summary['native_exit_observed'],False)
            self.assertIn('not observed',summary['error'])

    def test_cli_exception_never_reuses_preexisting_pass_receipt(self):
        self.require_sut()
        with tempfile.TemporaryDirectory() as temp:
            output_path=Path(temp)/'run'
            output_path.mkdir()
            receipt_path=output_path/'result.json'
            original=b'{"status":"PASS","samples":[{"immutable":true}]}\n'
            receipt_path.write_bytes(original)
            args=['--source-store','unused','--source-result','unused','--prepare','unused',
                '--profile','unused','--fixture','unused','--reference-dir','unused',
                '--output',str(output_path),'--arm','disk','--arm-order','resident,disk',
                '--order-seed','seed','--ram-bytes','1','--disk-bytes','1',
                '--min-available-bytes','0']
            def refuse_existing(_parsed): raise FileExistsError('output already exists')
            output=io.StringIO()
            code=sut.main(args,runner=refuse_existing,stdout=output)
            self.assertEqual(code,1)
            summary=json.loads(output.getvalue())
            self.assertEqual(summary['status'],'FAIL')
            self.assertEqual(summary['preexisting_report_status'],'PASS')
            self.assertIn('output already exists',summary['error'])
            self.assertEqual(receipt_path.read_bytes(),original)


if __name__ == '__main__':
    unittest.main(verbosity=2)
