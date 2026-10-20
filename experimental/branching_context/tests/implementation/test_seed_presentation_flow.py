"""Namespaced seed/reseed presentation continuity on the CPU flow fixture."""
import json
import copy
import hashlib
import os
import subprocess
import struct
import sys
import tempfile
import threading
import unittest
from pathlib import Path
import urllib.request
from unittest.mock import patch

import serve.test_cow_presentation_flow as flow
import context_time_travel.store as store_module
from serve.test_cow_answer_retry import CowAnswerRetryTests
from serve.cow_adapter import CowAdapter
from serve.cow_http import make_server
from serve.cow_text_adapter import UnsupportedTextRequest
from context_time_travel.presentation_binding import validate_binding
from context_time_travel.store import InvalidOperation, Store


class SeedPresentationFlowTests(unittest.TestCase):
    def setUp(self):
        self.set_stream_patch=patch.object(flow.FlowModel,'set_random_stream',_set_random_stream,create=True)
        self.set_stream_patch.start()
        self.addCleanup(self.set_stream_patch.stop)
        self.save_patch=patch.object(flow.FlowModel,'save_prepared_branch',
                                     _save_prepared_branch_with_checkpoint)
        self.save_patch.start()
        self.addCleanup(self.save_patch.stop)
        self.fixture = flow.CowPresentationFlowTests("runTest")
        self.fixture.setUp()
        # FlowModel is its own backend in this fixture; mirror the production
        # backend interface consumed by the text retry compatibility check.
        self.fixture.models[0].backend.vocab_size=300
        self.addCleanup(self.fixture.doCleanups)

    def test_answer_reseed_continue_keeps_assistant_message_and_history(self):
        f = self.fixture
        f.models[0].plans["answer-seed-flow"] = (257,)
        server=make_server(CowAdapter(f.coordinator),api_key='seed-flow-key',text_adapter=f.text)
        thread=threading.Thread(target=server.serve_forever,daemon=True)
        thread.start()
        self.addCleanup(self._stop,server,thread)
        f.text.prepare("chat", [{"role": "user", "content": "hello"}],
                       "root-seed-flow", namespace_id=f.namespace1).durable.result(5)
        answer = f.text.answer("chat", 1, "answer-seed-flow", 1)
        original_receipt = answer.durable.result(5)
        answer_head = f.coordinator.store.head("chat")
        original_presentation = f.coordinator.store.active_presentation("chat")
        source_history = tuple(f.coordinator.store.lineage(answer_head.tip)[1])

        body=json.dumps(dict(branch='chat',seed=0,revision=answer_head.revision,
                             operation_id='reseed-seed-flow')).encode()
        request=urllib.request.Request(f'http://127.0.0.1:{server.server_port}/cow/reseed',
            data=body,headers={'Content-Type':'application/json','Authorization':'Bearer seed-flow-key'})
        with urllib.request.urlopen(request,timeout=5) as response:
            reseed_result=json.loads(response.read().decode())
        reseeded_head = f.coordinator.store.head("chat")
        self.assertEqual(reseed_result['tip'],reseeded_head.tip)
        reseed_receipt=f.coordinator.store.presentation_receipt('reseed-seed-flow')
        self.assertEqual(reseed_receipt['presentation']['schema'],2)
        intent=json.loads((f.coordinator.store.root/'pool-operations'/'reseed-seed-flow.json').read_bytes())['intent']
        self.assertEqual(intent['schema'],4)
        self.assertEqual(intent['presentation']['schema'],2)
        for mutate in (
            lambda value: value.update(reseed_seed=1),
            lambda value: value['output_span'].update(count=1),
            lambda value: value['output_span'].update(message_id='f'*32),
        ):
            bad=copy.deepcopy(reseed_receipt['presentation'])
            mutate(bad)
            with self.assertRaises(InvalidOperation):
                validate_binding(f.coordinator.store,answer_head,reseeded_head.tip,bad,
                    namespace_id=f.namespace1,operation_id='reseed-seed-flow')
        reseeded_history = tuple(f.coordinator.store.lineage(reseeded_head.tip)[1])
        self.assertEqual(reseeded_history, source_history)
        self.assertEqual(f.coordinator.store.get_snapshot(reseeded_head.tip)["controller"]["next_draw"], 0)

        second=json.dumps(dict(branch='chat',seed=9,revision=reseeded_head.revision,
                               operation_id='reseed-seed-flow-2')).encode()
        request=urllib.request.Request(f'http://127.0.0.1:{server.server_port}/cow/reseed',
            data=second,headers={'Content-Type':'application/json','Authorization':'Bearer seed-flow-key'})
        with urllib.request.urlopen(request,timeout=5) as response:
            second_result=json.loads(response.read().decode())
        reseeded_head=f.coordinator.store.head('chat')
        self.assertEqual(second_result['tip'],reseeded_head.tip)
        self.assertEqual(tuple(f.coordinator.store.lineage(reseeded_head.tip)[1]),source_history)

        f.models[0].plans["continue-seed-flow"] = (258, 299)
        continued = f.text.resume("chat", reseeded_head.revision, "continue-seed-flow", 2)
        continued.durable.result(5)
        final_head = f.coordinator.store.head("chat")
        final_snapshot = f.coordinator.store.get_snapshot(final_head.tip)
        self.assertEqual(final_snapshot["controller"]["status"], "stopped")
        catalog = f.coordinator.store.presentation_catalog(f.namespace1)
        final_event = next(event for event in reversed(catalog["events"])
                           if event.get("result", {}).get("tip") == final_head.tip
                           and event.get("presentation") is not None)
        self.assertEqual(final_event["presentation"]["output_span"]["message_id"],
                         original_presentation["message_id"])
        self.assertEqual(tuple(f.coordinator.store.lineage(final_head.tip)[1]),
                         source_history + (258, 299))
        self.assertEqual(continued.text_result(), "!")
        from serve.cow_workspace import CowWorkspace
        projected = CowWorkspace(f.coordinator.store, f.text.tokenizer).history("chat")
        self.assertEqual(projected["controller_status"], "stopped")
        self.assertEqual([message["text"] for message in projected["messages"]],
                         ["hello", "hi!"])
        before_stopped_reseed=projected
        stopped_body=json.dumps(dict(branch='chat',seed=10,revision=final_head.revision,
                                     operation_id='reseed-stopped-flow')).encode()
        request=urllib.request.Request(f'http://127.0.0.1:{server.server_port}/cow/reseed',
            data=stopped_body,headers={'Content-Type':'application/json','Authorization':'Bearer seed-flow-key'})
        with urllib.request.urlopen(request,timeout=5) as response:
            json.loads(response.read().decode())
        stopped_head=f.coordinator.store.head('chat')
        stopped_after=f.coordinator.store.get_snapshot(stopped_head.tip)['controller']
        self.assertEqual(stopped_after['status'],'stopped')
        self.assertEqual(tuple(f.coordinator.store.lineage(stopped_head.tip)[1]),
                         tuple(f.coordinator.store.lineage(final_head.tip)[1]))
        stopped_projection=CowWorkspace(f.coordinator.store,f.text.tokenizer).history('chat')
        self.assertEqual([message['id'] for message in stopped_projection['messages']],
                         [message['id'] for message in before_stopped_reseed['messages']])
        self.assertEqual([message['text'] for message in stopped_projection['messages']],
                         [message['text'] for message in before_stopped_reseed['messages']])
        with self.assertRaisesRegex(UnsupportedTextRequest,'active controller'):
            f.text.resume('chat',stopped_head.revision,'continue-after-stopped-reseed',1)
        catalog_before_replay=f.coordinator.store.presentation_catalog(f.namespace1)
        counters_before_replay=dict(f.models[0].stats)
        replay=f.text.retry('answer-seed-flow','chat',stopped_head.revision,
                            'answer-seed-flow-replay')
        self.assertTrue(replay['replayed'])
        self.assertEqual(replay['receipt'],original_receipt)
        self.assertEqual(replay['answer']['snapshot'],answer_head.tip)
        self.assertEqual(replay['answer']['tokens'],[257])
        self.assertEqual(replay['answer']['text'],'hi')
        self.assertEqual(f.models[0].stats,counters_before_replay)
        self.assertEqual(f.coordinator.store.head('chat'),stopped_head)
        self.assertEqual(f.coordinator.store.presentation_catalog(f.namespace1),
                         catalog_before_replay)

    def test_same_seeded_snapshot_keeps_branch_specific_assistant_identity(self):
        f=self.fixture
        f.text.prepare('chat',[{'role':'user','content':'hello'}],
                       'root-seed-forks',namespace_id=f.namespace1).durable.result(5)
        source=f.coordinator.store.head('chat')
        for name in ('seed-a','seed-b'):
            f.coordinator.store.fork(name,source.tip,namespace_id=f.namespace1,
                operation_id='fork-'+name,source_branch='chat',source_revision=source.revision,
                source_tip=source.tip)
        f.models[0].plans['cow-seed-v1:0']=(257,)
        for name in ('seed-a','seed-b'):
            receipt=f.text.answer(name,0,'answer-'+name,1,seed=0).durable.result(5)
            self.assertEqual(receipt['branch']['revision'],1)
            intent=json.loads((f.coordinator.store.root/'pool-operations'/('answer-'+name+'.json')).read_bytes())['intent']
            self.assertEqual(intent['schema'],5)
            self.assertEqual(intent['generation']['schema'],7)
            self.assertEqual(intent['generation']['controls'],
                             {'schema':1,'seed':0,'text_stops':[]})
            self.assertEqual(intent['presentation']['schema'],2)
        self.assertEqual(f.coordinator.store.head('seed-a').tip,
                         f.coordinator.store.head('seed-b').tip)
        a=f.coordinator.store.active_presentation('seed-a')
        b=f.coordinator.store.active_presentation('seed-b')
        self.assertNotEqual(a['message_id'],b['message_id'])
        self.assertEqual(a['presentation']['output_span']['count'],1)
        self.assertEqual(b['presentation']['output_span']['count'],1)

    def test_namespaced_fork_resolves_assistant_source_from_an_ancestor(self):
        f = self.fixture
        store = f.coordinator.store
        f.models[0].plans['ancestor-answer'] = (257,)
        f.models[0].plans['ancestor-continue'] = (258,)
        f.text.prepare('ancestor-main', [{'role': 'user', 'content': 'hello'}],
                       'ancestor-root', namespace_id=f.namespace1).durable.result(5)
        f.text.answer('ancestor-main', 1, 'ancestor-answer', 1).durable.result(5)
        answer_head = store.head('ancestor-main')
        answer_id = store.active_presentation('ancestor-main')['message_id']
        stream_key = store.get_snapshot(answer_head.tip)['controller']['stream_key']
        f.text.resume('ancestor-main', answer_head.revision,
                      'ancestor-continue', 1).durable.result(5)
        later = store.head('ancestor-main')
        store.fork('ancestor-copy', later.tip, namespace_id=f.namespace1,
                   operation_id='ancestor-copy-fork', source_branch='ancestor-main',
                   source_revision=later.revision, source_tip=later.tip)
        resolved = store._published_assistant_message_id(
            f.namespace1, answer_head.tip, stream_key, branch_name='ancestor-copy')
        self.assertEqual(resolved, answer_id)

    def test_adoption_requires_and_persists_source_for_ambiguous_same_snapshot(self):
        """The selected snapshot does not identify which seeded chat owns its text ID."""
        f = self.fixture
        store = f.coordinator.store
        f.models[0].plans['cow-seed-v1:0'] = (257,)
        f.text.prepare('parent', [{'role': 'user', 'content': 'hello'}],
                       'adopt-parent-root', namespace_id=f.namespace1).durable.result(5)
        parent = store.head('parent')
        store.fork('candidate-a', parent.tip, namespace_id=f.namespace1,
                   operation_id='fork-candidate-a', source_branch='parent',
                   source_revision=parent.revision, source_tip=parent.tip)
        store.fork('candidate-b', parent.tip, namespace_id=f.namespace1,
                   operation_id='fork-candidate-b', source_branch='parent',
                   source_revision=parent.revision, source_tip=parent.tip)
        for branch in ('candidate-a', 'candidate-b'):
            f.text.answer(branch, 0, 'answer-' + branch, 1, seed=0).durable.result(5)
        target = store.head('candidate-a').tip
        self.assertEqual(store.head('candidate-b').tip, target)
        a = store.active_presentation('candidate-a')
        b = store.active_presentation('candidate-b')
        self.assertNotEqual(a['message_id'], b['message_id'])

        from context_time_travel.store import InvalidOperation
        from context_time_travel.evaluation import validate_record

        def report(score):
            return validate_record(dict(
                kind='evaluation', schema=1, snapshot_id=target,
                task_root=parent.root,
                evaluator=dict(name='seed-adoption-fixture', version='1',
                               source_sha256=hashlib.sha256(b'eval').hexdigest()),
                fixtures={'expected': hashlib.sha256(b'PASS').hexdigest()},
                metric=dict(name='score', order='maximize'), seed=3,
                environment={'runtime': 'CPU seed provenance fixture'},
                effect_policy={'mode': 'none'},
                outcomes=dict(mandatory={'valid': 'PASS'}, score=score, unresolved=[]),
                resources=dict(wall_seconds=0.0, cpu_seconds=0.0, peak_rss_bytes=0),
                artifacts={}))

        eval_a = store.put_evaluation(report(2))
        adoption_args = ('adopt-seeded-candidate', 'parent', parent.revision,
                         [eval_a], ['valid'])
        before = store.head('parent')
        with self.assertRaisesRegex(InvalidOperation, 'presentation source'):
            store.adopt_evaluated_once(*adoption_args)
        self.assertEqual(store.head('parent'), before)

        server = make_server(CowAdapter(f.coordinator), api_key='seed-adoption-key',
                             text_adapter=f.text)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(self._stop, server, thread)
        cli = Path(__file__).resolve().parents[4] / 'tools' / 'cow_cli.py'
        with tempfile.TemporaryDirectory(prefix='cow-seed-adoption-cli-') as state_dir:
            env = os.environ.copy()
            env.pop('PYTHONPATH', None)
            env['COW_API_KEY'] = 'seed-adoption-key'
            command = [sys.executable, str(cli), '--url',
                f'http://127.0.0.1:{server.server_port}', '--state-dir', state_dir,
                'adopt-evaluation', '--branch', 'parent', '--revision', str(parent.revision),
                '--evaluation-ids', eval_a, '--mandatory', 'valid',
                '--operation-id', 'adopt-seeded-candidate',
                '--presentation-source-operation', 'answer-candidate-a']
            cli_result = subprocess.run(command, capture_output=True, text=True,
                encoding='utf-8', timeout=15, env=env)
            self.assertEqual(cli_result.returncode, 0,
                f'CLI adoption failed stdout={cli_result.stdout!r} stderr={cli_result.stderr!r}')
            adopted = json.loads(cli_result.stdout)
        self.assertEqual(adopted['snapshot_id'], target)
        self.assertEqual(adopted['presentation_source_operation'], 'answer-candidate-a')
        self.assertEqual(store.head('parent').tip, target)
        self.assertEqual(store.active_presentation('parent')['message_id'], a['message_id'])
        branch_record = store._read_branch('parent')
        move_event = branch_record['events'][-1]
        move_receipt = branch_record['receipts']['adopt-seeded-candidate']
        self.assertEqual(move_event['adoption_presentation_source_operation'],
                         'answer-candidate-a')
        self.assertEqual(move_receipt['adoption_presentation_source_operation'],
                         'answer-candidate-a')

        pending = store.evaluation_operation_status('adopt-seeded-candidate')
        self.assertEqual(pending['result']['adoption_presentation_source_operation'],
                         'answer-candidate-a')
        replay = Store(store.root).adopt_evaluated_once(
            *adoption_args, presentation_source_operation='answer-candidate-a')
        self.assertEqual(replay['operation_id'], adopted['operation_id'])
        self.assertEqual(replay['evaluation_id'], adopted['evaluation_id'])
        self.assertEqual(replay['snapshot_id'], adopted['snapshot_id'])
        self.assertEqual((replay['branch'].name, replay['branch'].revision,
                          replay['branch'].tip, replay['branch'].root),
                         (adopted['branch']['name'], adopted['branch']['revision'],
                          adopted['branch']['tip'], adopted['branch']['root']))
        with self.assertRaisesRegex(InvalidOperation, 'different input'):
            Store(store.root).adopt_evaluated_once(
                *adoption_args, presentation_source_operation='answer-candidate-b')

        with self.assertRaisesRegex(InvalidOperation, 'verified publication'):
            Store(store.root).adopt_evaluated_once(
                'adopt-wrong-source', 'parent', store.head('parent').revision,
                [eval_a], ['valid'], presentation_source_operation='fork-candidate-a')
        self.assertFalse((store.evaluation_operations/'adopt-wrong-source.json').exists())

        before_fault = store.head('parent')
        def lose_ack(step):
            if step == 'after_evaluation_adoption_move':
                raise OSError('lost acknowledgement after adoption move')
        fault_store = type(store)(store.root, fault_hook=lose_ack)
        with self.assertRaisesRegex(OSError, 'lost acknowledgement'):
            fault_store.adopt_evaluated_once(
                'adopt-lost-ack', 'parent', before_fault.revision,
                [eval_a], ['valid'], presentation_source_operation='answer-candidate-a')
        moved = fault_store.head('parent')
        self.assertEqual(moved.revision, before_fault.revision+1)
        pending_fault = fault_store.evaluation_operation_status('adopt-lost-ack')
        self.assertEqual(pending_fault['state'], 'pending')
        pending_record = fault_store._read_evaluation_operation('adopt-lost-ack')
        self.assertEqual(pending_record['adoption_presentation_source_operation'],
                         'answer-candidate-a')
        replay_fault = Store(store.root).adopt_evaluated_once(
            'adopt-lost-ack', 'parent', before_fault.revision,
            [eval_a], ['valid'], presentation_source_operation='answer-candidate-a')
        self.assertEqual(replay_fault['branch'], moved)
        self.assertEqual(Store(store.root).active_presentation('parent')['message_id'],
                         a['message_id'])

    def test_legacy_unprovenanced_fork_uses_only_a_unique_verified_source(self):
        f = self.fixture
        store = f.coordinator.store
        f.models[0].plans['legacy-fork-answer'] = (257,)
        f.text.prepare('unique-parent', [{'role': 'user', 'content': 'hello'}],
                       'unique-parent-root', namespace_id=f.namespace1).durable.result(5)
        f.text.answer('unique-parent', 1, 'legacy-fork-answer', 1).durable.result(5)
        head = store.head('unique-parent')
        store.fork('legacy-unique', head.tip, namespace_id=f.namespace1,
                   operation_id='legacy-unique-fork')
        self.assertEqual(store.active_presentation('legacy-unique')['message_id'],
                         store.active_presentation('unique-parent')['message_id'])

        f.text.prepare('ambiguous-parent', [{'role': 'user', 'content': 'hello'}],
                       'ambiguous-parent-root', namespace_id=f.namespace1).durable.result(5)
        ambiguous_head=store.head('ambiguous-parent')
        for branch in ('ambiguous-a','ambiguous-b'):
            store.fork(branch, ambiguous_head.tip, namespace_id=f.namespace1,
                       operation_id='fork-'+branch)
            f.text.answer(branch,0,'answer-'+branch,1,seed=0).durable.result(5)
        store.fork('legacy-ambiguous',store.head('ambiguous-a').tip,
                   namespace_id=f.namespace1,operation_id='legacy-ambiguous-fork')
        from context_time_travel.store import IntegrityError
        with self.assertRaisesRegex(IntegrityError, 'ambiguous'):
            store.active_presentation('legacy-ambiguous')

    def test_legacy_pending_adoption_replays_exact_committed_move_without_new_provenance(self):
        """A pre-upgrade lost ACK keeps its original source-less move digest."""
        f = self.fixture
        store = f.coordinator.store
        f.text.prepare('legacy-adopt-parent', [{'role': 'user', 'content': 'hello'}],
                       'legacy-adopt-root', namespace_id=f.namespace1).durable.result(5)
        parent = store.head('legacy-adopt-parent')
        store.fork('legacy-adopt-candidate', parent.tip, namespace_id=f.namespace1,
                   operation_id='legacy-adopt-fork', source_branch='legacy-adopt-parent',
                   source_revision=parent.revision, source_tip=parent.tip)
        f.models[0].plans['legacy-adopt-answer'] = (257,)
        candidate_head = store.head('legacy-adopt-candidate')
        f.text.answer('legacy-adopt-candidate', candidate_head.revision,
                      'legacy-adopt-answer', 1).durable.result(5)
        target = store.head('legacy-adopt-candidate').tip
        record = store.put_evaluation(dict(
            kind='evaluation', schema=1, snapshot_id=target,
            task_root=parent.root,
            evaluator=dict(name='legacy-adoption-fixture', version='1',
                           source_sha256=hashlib.sha256(b'eval').hexdigest()),
            fixtures={'expected': hashlib.sha256(b'PASS').hexdigest()},
            metric=dict(name='score', order='maximize'), seed=3,
            environment={'runtime': 'CPU legacy pending-adoption fixture'},
            effect_policy={'mode': 'none'},
            outcomes=dict(mandatory={'valid': 'PASS'}, score=1, unresolved=[]),
            resources=dict(wall_seconds=0.0, cpu_seconds=0.0, peak_rss_bytes=0),
            artifacts={}))
        operation = 'legacy-adopt-lost-ack'
        arguments = (operation, 'legacy-adopt-parent', parent.revision, [record], ['valid'])
        request = dict(kind='adoption', branch='legacy-adopt-parent',
                       expected_revision=parent.revision, evaluation_ids=[record],
                       mandatory=['valid'])
        selected = store.get_evaluation(record)['snapshot_id']
        store._claim_evaluation_operation(operation, 'adoption',
            store_module._hash(store_module._json_bytes(request)), dict(
                branch='legacy-adopt-parent', expected_revision=parent.revision,
                evaluation_ids=[record], mandatory=['valid'],
                selected_evaluation_id=record, selected_snapshot_id=selected))
        moved = store.move('legacy-adopt-parent', parent.revision, selected, operation,
                           _evaluation_operation=True)
        original_receipt = store._read_branch('legacy-adopt-parent')['receipts'][operation]
        self.assertNotIn('adoption_presentation_source_operation', original_receipt)

        recovered = Store(store.root).adopt_evaluated_once(*arguments)
        self.assertEqual(recovered['branch'], moved)
        self.assertEqual(recovered['snapshot_id'], selected)
        self.assertEqual(Store(store.root)._read_branch('legacy-adopt-parent')['receipts'][operation],
                         original_receipt)
        committed = Store(store.root)._read_evaluation_operation(operation)
        self.assertEqual(committed['state'], 'committed')
        self.assertNotIn('adoption_presentation_source_operation', committed)

    def test_reseed_from_cold_disk_source_performs_no_forward_work(self):
        f=self.fixture
        f.models[0].plans['cold-reseed-answer']=(257,)
        f.text.prepare('chat',[{'role':'user','content':'hello'}],
            'cold-reseed-root',namespace_id=f.namespace1).durable.result(5)
        answer=f.text.answer('chat',1,'cold-reseed-answer',1)
        answer.durable.result(5)
        model=f.models[0]
        source=f.coordinator.store.head('chat').tip
        key=f.coordinator._key(model,source)
        self.assertIn(key,f.coordinator.pool)
        model.cache.clear()
        f.coordinator.pool.remove(key)
        before=model.stats['forward_tokens']
        before_head=f.coordinator.store.head('chat')
        reseed=f.text.reseed('chat',0,before_head.revision,
                             'cold-reseed-operation')
        receipt=reseed.durable.result(5)
        self.assertEqual(model.stats['forward_tokens'],before)
        self.assertEqual(receipt['branch']['revision'],before_head.revision+1)
        self.assertIn(key,f.coordinator.pool)
        head=f.coordinator.store.head('chat')
        controller=f.coordinator.store.get_snapshot(head.tip)['controller']
        self.assertEqual(controller['stream_key'],hashlib.sha256(
            b'cow-seed-v1:0').hexdigest())


    def test_actual_cli_seed_reseed_continue_and_history(self):
        f=self.fixture
        f.models[0].plans['cow-seed-v1:0']=(257,)
        f.models[0].plans['cow-seed-v1:9']=(258,299)
        f.models[0].plans['cli-continue']=(258,299)
        from serve.cow_workspace import CowWorkspace
        workspace=CowWorkspace(f.coordinator.store,f.text.tokenizer)
        server=make_server(CowAdapter(f.coordinator),api_key='seed-flow-key',
                           text_adapter=f.text,workspace=workspace)
        thread=threading.Thread(target=server.serve_forever,daemon=True)
        thread.start()
        self.addCleanup(self._stop,server,thread)
        cli=Path(__file__).resolve().parents[4]/'tools'/'cow_cli.py'
        with tempfile.TemporaryDirectory(prefix='cow-seed-cli-') as state_dir:
            env=os.environ.copy()
            env.pop('PYTHONPATH',None)
            env['COW_API_KEY']='seed-flow-key'
            env['COW_CLI_STATE_DIR']=state_dir

            def run(*args):
                result=subprocess.run([sys.executable,str(cli),'--url',
                    f'http://127.0.0.1:{server.server_port}','--state-dir',state_dir,*args],
                    capture_output=True,text=True,encoding='utf-8',timeout=15,env=env)
                self.assertEqual(result.returncode,0,
                    f'CLI failed ({result.returncode}) stdout={result.stdout!r} stderr={result.stderr!r}')
                return result

            run('prepare','--branch','cli-chat','--namespace-id',f.namespace1,
                '--text','hello','--operation-id','cli-root')
            answer=run('answer','--branch','cli-chat','--max-tokens','1',
                       '--operation-id','cli-seeded-answer','--seed','0')
            self.assertEqual(answer.stdout,'hi\n')
            self.assertIn('durable receipt:',answer.stderr)
            answer_intent=json.loads((Path(state_dir)/'operations'/'cli-seeded-answer.json').read_bytes())
            self.assertEqual(answer_intent['state'],'complete')
            answer_head=f.coordinator.store.head('cli-chat')
            answer_state=f.coordinator.store.get_snapshot(answer_head.tip)['controller']
            self.assertEqual(answer_state['stream_key'],hashlib.sha256(
                b'cow-seed-v1:0').hexdigest())
            self.assertEqual(answer_state['next_draw'],0)

            reseed=run('reseed','--branch','cli-chat','--seed','9',
                       '--operation-id','cli-reseed')
            self.assertIn('durable receipt:',reseed.stderr)
            reseed_head=f.coordinator.store.head('cli-chat')
            reseeded=f.coordinator.store.get_snapshot(reseed_head.tip)['controller']
            self.assertEqual(reseeded['stream_key'],hashlib.sha256(
                b'cow-seed-v1:9').hexdigest())
            self.assertEqual(reseeded['next_draw'],0)

            f.models[0].plans['cow-seed-v1:9']=(258,299)
            continued=run('continue','--branch','cli-chat','--max-tokens','2',
                          '--operation-id','cli-continue')
            self.assertEqual(continued.stdout,'!\n')
            history=run('history','--branch','cli-chat')
            projection=json.loads(history.stdout)
            self.assertEqual([item['content'] for item in projection['messages']],
                             ['hello','hi!'])
            self.assertEqual(projection['controller_status'],'stopped')
            self.assertEqual(tuple(f.coordinator.store.lineage(
                f.coordinator.store.head('cli-chat').tip)[1]),
                tuple(f.coordinator.store.lineage(reseed_head.tip)[1])+(258,299))

    @staticmethod
    def _stop(server,thread):
        server.shutdown()
        thread.join(5)
        server.server_close()


class SeedColdChangedRetryTests(unittest.TestCase):
    def test_seeded_changed_retry_materializes_active_source_before_reseed(self):
        set_stream_patch=patch.object(flow.FlowModel,'set_random_stream',_set_random_stream,create=True)
        set_stream_patch.start()
        from serve.test_cow_answer_retry import RetryFlowModel
        sampled_patch=patch.object(RetryFlowModel,'run_sampled_request',_run_sampled_with_reseed)
        sampled_patch.start()
        fixture=CowAnswerRetryTests('runTest')
        try:
            fixture.setUp()
            with tempfile.TemporaryDirectory() as state_dir:
                fixture._completed_original(state_dir,temperature=.7)
                continued=fixture._cli(state_dir,'continue','--branch','chat',
                    '--max-tokens','1','--operation-id','seeded-continue','--temperature','.7')
                self.assertEqual(continued.returncode,0,continued.stderr.decode('utf-8','replace'))
                original=fixture.coordinator.committed_generation('seeded-continue')
                source=original['source_head'].tip
                source_controller=fixture.coordinator.store.get_snapshot(source)['controller']
                self.assertGreater(source_controller['next_draw'],0)
                self.assertEqual(source_controller['status'],'active')
                old_model=fixture.models[0]
                old_forward=old_model.stats['forward_tokens']
                fixture._restart_server()
                model=fixture.models[0]
                self.assertNotIn(source,model.cache)
                before_forward=model.stats['forward_tokens']
                retried=fixture._cli(state_dir,'retry','--from-operation','seeded-continue',
                    '--operation-id','seeded-continue-new-stream','--temperature','.7',
                    '--top-k','2','--seed','9')
                self.assertEqual(retried.returncode,0,retried.stderr.decode('utf-8','replace'))
                record=model.sample_records[-1]
                self.assertTrue(record['resume'])
                self.assertEqual(record['source_accepted'],source_controller['accepted_in_scope'])
                self.assertEqual(record['source_stream_key'],hashlib.sha256(
                    b'cow-seed-v1:9').hexdigest())
                self.assertEqual(record['final_controller']['accepted_in_scope'],
                                 source_controller['accepted_in_scope']+1)
                self.assertEqual(record['final_controller']['stream_key'],record['source_stream_key'])
                self.assertEqual(model.stats['forward_tokens']-before_forward,1)
                self.assertEqual(old_model.stats['forward_tokens'],old_forward)
                stored=fixture.coordinator.store.presentation_receipt(
                    'seeded-continue-new-stream')['presentation']
                source_head=fixture.coordinator.committed_generation(
                    'seeded-continue-new-stream')['source_head']
                for mutate in (
                    lambda value: value.pop('generation_seed'),
                    lambda value: value.update(generation_seed=10),
                ):
                    bad=copy.deepcopy(stored)
                    mutate(bad)
                    with self.assertRaises(InvalidOperation):
                        validate_binding(fixture.coordinator.store,
                            source_head,record['final_snapshot'],bad,
                            namespace_id=fixture.namespace,
                            operation_id='seeded-continue-new-stream')
        finally:
            try: fixture.doCleanups()
            finally:
                sampled_patch.stop()
                set_stream_patch.stop()


def _run_sampled_with_reseed(self,sid,ids,count,*,sampling,stream,resume=False,
                             on_token=None,cancelled=None,reseed_seed=None):
    """CPU fixture path for an in-memory seeded continuation transition."""
    from serve.test_cow_answer_retry import RetryTokenizer
    from context_time_travel.sampled_controller import PlainSamplingSession, start
    from context_time_travel.seed_controls import reseed_controller
    history=self.history(sid)
    prompt_history=history+tuple(ids)
    if resume:
        state=copy.deepcopy(self.store.get_snapshot(sid)['controller'])
        if state.get('schema') in (1,2):
            from context_time_travel.sampled_controller import sampler_order
            state=dict(state,schema=3,constraint=state.get('constraint'))
            state['policy']=dict(state['policy'],**sampling)
            state['sampler_order']=sampler_order(sampling)
        session=PlainSamplingSession.restore(state,prompt_history,300,
            RetryTokenizer().token_bytes,lambda token:token==299)
        session.configure_generation(sampling)
        if reseed_seed is not None:
            state=reseed_controller(session.current_controller,seed=reseed_seed,
                history=history,vocab_size=300)
            session=PlainSamplingSession.restore(state,prompt_history,300,
                RetryTokenizer().token_bytes,lambda token:token==299)
    else:
        state=start(sampling,stream,prompt_history,300,[299])
        session=PlainSamplingSession.restore(state,prompt_history,300,
            RetryTokenizer().token_bytes,lambda token:token==299)
    source_controller=session.current_controller
    mask=bytearray(b'\xff'*((300+7)//8))
    mask[-1]&=(1<<(300%8))-1
    outputs=[]
    for _ in range(count):
        if cancelled is not None and cancelled():
            raise RuntimeError('unexpected synthetic cancellation')
        before=session.current_controller
        selector=lambda legal,temp,min_p_value,draw,**filters:self.sample_legal(
            None,legal,temp,min_p_value,draw,**filters)
        token=session.choose(bytes(mask),selector)
        session.consume(before,token,RetryTokenizer().token_bytes(token),token==299)
        outputs.append(token)
        if on_token is not None: on_token(token)
        if session.is_stopped(): break
    target=self.store.append(sid,tuple(ids)+tuple(outputs),session.current_controller)
    consumed=len(ids)+len(outputs)
    self.stats['forward_tokens']+=consumed
    self.stats['consumed_tokens']+=consumed
    self.image(target)
    self.trace.append((stream,tuple(ids),tuple(outputs),'sampled',dict(sampling)))
    self.sample_records.append(dict(resume=resume,
        source_accepted=source_controller.get('accepted_in_scope',0),
        source_stream_key=source_controller.get('stream_key'),
        final_controller=session.current_controller,final_snapshot=target))
    return dict(snapshot=target,tokens=outputs,request_seconds=0.,generation_seconds=0.)


def _set_random_stream(model,snapshot,stream):
    state=copy.deepcopy(model.store.get_snapshot(snapshot)['controller'])
    if state.get('status')=='idle':
        from context_time_travel.store import InvalidOperation
        raise InvalidOperation('No generation scope')
    state['stream_key']=hashlib.sha256(stream.encode()).hexdigest()
    state['next_draw']=0
    target=model.store.append(snapshot,(),state)
    if snapshot in model.cache:
        model.cache[target]=model.cache[snapshot]
    return target


def _save_prepared_branch_with_checkpoint(model, name, revision, prepared, operation,
                                         *, presentation=None,
                                         presentation_source_operation=None):
    """Mirror the real backend's atomic checkpoint receipt in this CPU fixture."""
    cp=prepared.checkpoint
    frontier=struct.pack('<'+str(len(cp.frontier))+'f',*cp.frontier)
    (model.checkpoint_directory/(cp.checksum+'.state')).write_bytes(cp.blob+frontier)
    (model.checkpoint_directory/(prepared.snapshot+'.json')).write_bytes(prepared.envelope)
    model.validate_checkpoint(prepared.snapshot)
    checkpoint_receipt=dict(schema=1,snapshot=prepared.snapshot,
        payload_sha256=cp.checksum,model_contract_id=cp.model_contract_id,
        execution_profile_id=cp.execution_profile_id)
    options={'checkpoint_receipt':checkpoint_receipt}
    if presentation is not None:
        options['presentation']=presentation
    if presentation_source_operation is not None:
        options['presentation_source_operation']=presentation_source_operation
    head=model.store.move(name,revision,prepared.snapshot,operation,**options)
    return dict(snapshot=prepared.snapshot,branch=dict(name=head.name,revision=head.revision,
        tip=head.tip,root=head.root),payload_sha256=cp.checksum,
        model_contract_id=cp.model_contract_id,execution_profile_id=cp.execution_profile_id)


if __name__ == "__main__":
    unittest.main()
