"""CPU service/CLI contract for G4 evaluation and server-side adoption.

Real evaluator execution in this module requires bubblewrap and is intentionally
run on the designated Linux CPU host. It does not claim model/GPU qualification.
"""
from __future__ import annotations

import json
import os
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from unittest.mock import patch

from context_time_travel.store import Store
from serve.cow_adapter import CowAdapter
from serve.cow_http import make_server
from tools import cow_cli
from serve import test_cow_presentation_flow as flow
from serve.cow_workspace import CowWorkspace


EVALUATOR = "result = {'mandatory': {'fixture_ok': 'PASS' if fixtures['expected'] == b'PASS' else 'FAIL'}, 'score': len(candidate['tokens']), 'unresolved': []}"


class _Coordinator:
    def __init__(self, store):
        self.store = store

    def adopt_evaluated_once(self, *args):
        return self.store.adopt_evaluated_once(*args)


class EvaluationServingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.store = Store(cls.temporary.name)
        cls.root = cls.store.create_root('g4-contract', [1], {'status': 'idle'})
        cls.a = cls.store.append(cls.root, [2], {'status': 'idle'})
        cls.b = cls.store.append(cls.root, [3, 4], {'status': 'idle'})
        cls.store.fork('main', cls.root)
        cls.adapter = CowAdapter(_Coordinator(cls.store))
        cls.server = make_server(cls.adapter, api_key='g4-secret')
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.url = f'http://127.0.0.1:{cls.server.server_port}'

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=5)
        cls.temporary.cleanup()

    def request(self, method, path, body=None, key='g4-secret'):
        headers = {'Authorization': 'Bearer ' + key}
        data = None if body is None else json.dumps(body).encode('utf-8')
        if data is not None:
            headers['Content-Type'] = 'application/json'
        request = urllib.request.Request(self.url + path, data=data, headers=headers, method=method)
        try:
            with urllib.request.urlopen(request, timeout=15) as response:
                return response.status, json.loads(response.read())
        except urllib.error.HTTPError as exc:
            with exc:
                return exc.code, json.loads(exc.read())

    def spec(self):
        return {
            'task_root': self.root,
            'evaluator': {'name': 'fixture-check', 'version': '1', 'source': EVALUATOR},
            'metric': {'name': 'consumed_tokens', 'order': 'maximize'},
            'seed': 17,
            'mandatory': ['fixture_ok'],
            'timeout_seconds': 5,
        }

    def test_evaluate_inspect_select_and_adopt_use_server_validation(self):
        before = self.store.head('main')
        created = []
        for suffix, name in ((self.a, 'eval-a'), (self.b, 'eval-b')):
            code, value = self.request('POST', '/cow/evaluate', {
                'snapshot_id': suffix,
                'operation_id': name,
                'specification': self.spec(),
                'fixtures': {'expected': b'PASS'.hex()},
            })
            self.assertEqual(code, 200)
            self.assertEqual(value['record']['snapshot_id'], suffix)
            self.assertEqual(value['record']['effect_policy']['mode'], 'isolated')
            self.assertEqual(value['record']['outcomes']['mandatory'], {'fixture_ok': 'PASS'},
                             value['record']['outcomes'])
            created.append(value['evaluation_id'])

        eval_id = created[0]
        code, inspected = self.request('GET', '/cow/evaluation/' + eval_id)
        self.assertEqual(code, 200)
        self.assertEqual(inspected['evaluation_id'], eval_id)
        self.assertEqual(inspected['record']['snapshot_id'], self.a)

        code, selected = self.request('POST', '/cow/evaluation/select', {
            'evaluation_ids': created, 'mandatory': ['fixture_ok'],
            # The server must reject or ignore client-supplied result authority.
            'winner': self.b, 'score': 10**9,
        })
        self.assertEqual(code, 400)
        self.assertEqual(self.store.head('main'), before)

        code, selected = self.request('POST', '/cow/evaluation/select', {
            'evaluation_ids': created, 'mandatory': ['fixture_ok'],
        })
        self.assertEqual(code, 200)
        self.assertEqual(selected['snapshot_id'], self.b)
        self.assertEqual(selected['record']['snapshot_id'], self.b)
        self.assertEqual(selected['evaluation_id'], created[1])

        code, adopted = self.request('POST', '/cow/evaluation/adopt', {
            'branch': 'main', 'expected_revision': before.revision,
            'evaluation_ids': created, 'mandatory': ['fixture_ok'],
            'operation_id': 'adopt-g4-a',
        })
        self.assertEqual(code, 200)
        self.assertEqual(adopted['branch']['tip'], self.b)
        self.assertEqual(adopted['evaluation_id'], created[1])
        self.assertEqual(self.store.head('main').tip, self.b)

        # A lost acknowledgement can replay the same immutable selection and
        # Store.move receipt; changing the full request under that ID conflicts.
        self.assertEqual(self.request('POST', '/cow/evaluation/adopt', {
            'branch': 'main', 'expected_revision': before.revision,
            'evaluation_ids': created, 'mandatory': ['fixture_ok'],
            'operation_id': 'adopt-g4-a',
        }), (200, adopted))
        code, _ = self.request('POST', '/cow/evaluation/adopt', {
            'branch': 'main', 'expected_revision': before.revision,
            'evaluation_ids': [created[1]], 'mandatory': ['fixture_ok'],
            'operation_id': 'adopt-g4-a',
        })
        self.assertEqual(code, 409)

    def test_fixture_bytes_are_hex_and_auth_and_limits_precede_evaluation(self):
        head = self.store.head('main')
        body = {'snapshot_id': self.a, 'operation_id': 'bad-fixture-path',
                'specification': self.spec(), 'fixtures': {'expected': '/etc/passwd'}}
        code, _ = self.request('POST', '/cow/evaluate', body)
        self.assertEqual(code, 400)
        self.assertEqual(self.store.head('main'), head)

    def test_operation_ids_cannot_cross_namespace_evaluation_or_coordinator_routes(self):
        code, _namespace = self.request('POST','/cow/namespace',
            {'title':'Cross endpoint','operation_id':'CrossEvalOp'})
        self.assertEqual(code,200)
        before = self.store.head('main')
        body = {'snapshot_id':self.a,'operation_id':'CrossEvalOp',
                'specification':self.spec(),'fixtures':{'expected':b'PASS'.hex()}}
        code,_ = self.request('POST','/cow/evaluate',body)
        self.assertEqual(code,409)
        self.assertFalse((self.store.evaluation_operations/'CrossEvalOp.json').exists())
        self.assertEqual(self.store.head('main'),before)
        # A coordinator publication intent is also unavailable to evaluation.
        (self.store.root/'pool-operations').mkdir(mode=0o700,exist_ok=True)
        (self.store.root/'pool-operations'/'PoolOwned.json').write_text('{}',encoding='ascii')
        body['operation_id']='PoolOwned'
        code,_ = self.request('POST','/cow/evaluate',body)
        self.assertEqual(code,409)
        self.assertFalse((self.store.evaluation_operations/'PoolOwned.json').exists())
        code, _ = self.request('POST', '/cow/evaluate', {
            **body, 'operation_id': 'unauth-evaluation', 'fixtures': {'expected': b'PASS'.hex()}},
            key='wrong')
        self.assertEqual(code, 401)
        self.assertEqual(self.store.head('main'), before)

    def test_http_fork_cannot_reuse_a_committed_evaluation_operation_id(self):
        code, result = self.request('POST', '/cow/evaluate', {
            'snapshot_id': self.a, 'operation_id': 'shared-evaluation-fork',
            'specification': self.spec(), 'fixtures': {'expected': b'PASS'.hex()}})
        self.assertEqual(code, 200)
        self.assertEqual(result['operation_id'], 'shared-evaluation-fork')
        code, _ = self.request('POST', '/cow/fork', {
            'name': 'must-not-exist', 'snapshot': self.a,
            'operation_id': 'shared-evaluation-fork'})
        self.assertEqual(code, 409)
        self.assertFalse((self.store.references / 'must-not-exist.json').exists())
        self.assertTrue((self.store.evaluation_operations / 'shared-evaluation-fork.json').exists())

    def test_cli_reads_fixture_bytes_locally_and_sends_no_client_winner(self):
        with tempfile.TemporaryDirectory() as directory:
            spec_path = Path(directory) / 'spec.json'
            fixture_path = Path(directory) / 'expected.bin'
            spec_path.write_text(json.dumps(self.spec()), encoding='utf-8')
            fixture_path.write_bytes(b'PASS')
            state_dir = Path(directory) / 'journal'
            with patch.dict(os.environ, {'COW_API_KEY': 'g4-secret'}):
                result = cow_cli.main([
                    '--url', self.url, '--state-dir', str(state_dir), 'evaluate', '--snapshot-id', self.a,
                    '--spec', str(spec_path), '--fixture', f'expected={fixture_path}',
                    '--operation-id', 'cli-evaluate-a',
                ])
            self.assertEqual(result, 0)
            journal = cow_cli.RequestJournal(state_dir)
            pending = journal.get('cli-evaluate-a')
            sent = json.loads(pending['body_json'])
            self.assertEqual(sent['fixtures'], {'expected': b'PASS'.hex()})
            self.assertNotIn(str(fixture_path), pending['body_json'])
            evaluation_id = pending['receipt']['evaluation_id']
            with patch.dict(os.environ, {'COW_API_KEY': 'g4-secret'}):
                self.assertEqual(cow_cli.main(['--url', self.url, 'inspect-evaluation',
                                               '--evaluation-id', evaluation_id]), 0)
                self.assertEqual(cow_cli.main(['--url', self.url, 'select-evaluations',
                    '--evaluation-ids', evaluation_id, '--mandatory', 'fixture_ok']), 0)
            head = self.store.head('main')
            with patch.dict(os.environ, {'COW_API_KEY': 'g4-secret'}):
                self.assertEqual(cow_cli.main(['--url', self.url, '--state-dir', str(state_dir),
                    'adopt-evaluation', '--branch', 'main', '--revision', str(head.revision),
                    '--evaluation-ids', evaluation_id, '--mandatory', 'fixture_ok',
                    '--operation-id', 'cli-adopt-a']), 0)

    def test_cli_served_evaluate_select_adopt_keeps_real_text_history(self):
        fixture = flow.CowPresentationFlowTests('runTest')
        fixture.setUp()
        server = None
        thread = None
        try:
            coordinator, text = fixture.coordinator, fixture.text
            namespace = coordinator.store.namespaces()[0].namespace_id
            server = make_server(CowAdapter(coordinator), api_key='flow-secret',
                text_adapter=text,
                workspace=CowWorkspace(coordinator.store, text.tokenizer,
                                       root_mode_resolver=text.root_mode_for_projection))
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            url = f'http://127.0.0.1:{server.server_port}'
            with tempfile.TemporaryDirectory() as directory:
                state_dir = Path(directory)/'journal'
                spec_path = Path(directory)/'spec.json'
                fixture_path = Path(directory)/'expected.bin'
                spec = {
                    'task_root': None,
                    'evaluator': {'name':'choose-bang','version':'1',
                        'source': "result = {'mandatory': {'fixture_ok': 'PASS'}, 'score': int(candidate['tokens'][-1] == 258), 'unresolved': []}"},
                    'metric': {'name':'candidate_is_bang','order':'maximize'},
                    'seed': 3, 'mandatory':['fixture_ok'], 'timeout_seconds':5,
                }
                spec_path.write_text(json.dumps(spec),encoding='utf-8')
                fixture_path.write_bytes(b'PASS')
                with patch.dict(os.environ, {'COW_API_KEY':'flow-secret'}):
                    self.assertEqual(cow_cli.main(['--url',url,'--state-dir',str(state_dir),
                        'prepare','--branch','main','--namespace-id',namespace,
                        '--text','hello','--operation-id','g4-prepare']),0)
                    prepared = coordinator.store.head('main')
                    self.assertEqual(cow_cli.main(['--url',url,'--state-dir',str(state_dir),
                        'fork','--source-branch','main','--snapshot',prepared.tip,
                        '--name','candidate-a','--operation-id','g4-fork-a']),0)
                    self.assertEqual(cow_cli.main(['--url',url,'--state-dir',str(state_dir),
                        'fork','--source-branch','main','--snapshot',prepared.tip,
                        '--name','candidate-b','--operation-id','g4-fork-b']),0)
                    self.assertEqual(cow_cli.main(['--url',url,'--state-dir',str(state_dir),
                        'answer','--branch','candidate-a','--max-tokens','1',
                        '--operation-id','alt-answer']),0)
                    self.assertEqual(cow_cli.main(['--url',url,'--state-dir',str(state_dir),
                        'answer','--branch','candidate-b','--max-tokens','1',
                        '--operation-id','resume-op']),0)
                    client = cow_cli.CowClient(url,'flow-secret',15)
                    candidate_a = client.request('GET','/cow/head?branch=candidate-a')['tip']
                    candidate_b = client.request('GET','/cow/head?branch=candidate-b')['tip']
                    spec['task_root'] = prepared.tip
                    spec_path.write_text(json.dumps(spec),encoding='utf-8')
                    self.assertEqual(cow_cli.main(['--url',url,'--state-dir',str(state_dir),
                        'evaluate','--snapshot-id',candidate_a,'--spec',str(spec_path),
                        '--fixture',f'expected={fixture_path}','--operation-id','g4-eval-a']),0)
                    self.assertEqual(cow_cli.main(['--url',url,'--state-dir',str(state_dir),
                        'evaluate','--snapshot-id',candidate_b,'--spec',str(spec_path),
                        '--fixture',f'expected={fixture_path}','--operation-id','g4-eval-b']),0)
                    journal = cow_cli.RequestJournal(state_dir)
                    eval_a = journal.get('g4-eval-a')['receipt']['evaluation_id']
                    eval_b = journal.get('g4-eval-b')['receipt']['evaluation_id']
                    self.assertEqual(cow_cli.main(['--url',url,'--state-dir',str(state_dir),
                        'inspect-evaluation','--evaluation-id',eval_b]),0)
                    self.assertEqual(cow_cli.main(['--url',url,'--state-dir',str(state_dir),
                        'select-evaluations','--evaluation-ids',eval_a,eval_b,
                        '--mandatory','fixture_ok']),0)
                    before = coordinator.store.head('main')
                    self.assertEqual(cow_cli.main(['--url',url,'--state-dir',str(state_dir),
                        'adopt-evaluation','--branch','main','--revision',str(before.revision),
                        '--evaluation-ids',eval_a,eval_b,'--mandatory','fixture_ok',
                        '--operation-id','g4-adopt']),0)

                adopted_head = coordinator.store.head('main')
                self.assertEqual(adopted_head.tip,candidate_b)
                self.assertEqual(adopted_head.revision,before.revision+1)
                self.assertEqual(coordinator.store.head('candidate-a').tip,candidate_a)
                workspace = CowWorkspace(coordinator.store,text.tokenizer,
                    root_mode_resolver=text.root_mode_for_projection)
                adopted_history = workspace.history('main')
                loser_history = workspace.history('candidate-a')
                self.assertEqual(adopted_history['tip'],candidate_b)
                self.assertTrue(any(message.get('text')=='!' for message in adopted_history['messages']))
                self.assertTrue(any(message.get('text')=='hi' for message in loser_history['messages']))
                # Adoption leaves the winning answer's active controller resumable.
                with patch.dict(os.environ, {'COW_API_KEY':'flow-secret'}):
                    self.assertEqual(cow_cli.main(['--url',url,'--state-dir',str(state_dir),
                        'continue','--branch','main','--max-tokens','1',
                        '--operation-id','copy-resume']),0)
                continued = workspace.history('main')
                self.assertEqual(continued['controller_status'],'active')
                self.assertEqual(coordinator.store.head('main').revision,adopted_head.revision+1)
        finally:
            if server is not None:
                server.shutdown()
                server.server_close()
            if thread is not None:
                thread.join(timeout=5)
            fixture.doCleanups()


if __name__ == '__main__':
    unittest.main(verbosity=2)
