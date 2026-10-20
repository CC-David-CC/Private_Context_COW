"""CPU CLI/HTTP text-stop integration tests using the synthetic FlowModel."""
from __future__ import annotations

import json
import contextlib
import copy
import hashlib
import io
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

from context_time_travel.coordinator import ContextCoordinator
from context_time_travel.pools import PoolConfig
from serve.cow_adapter import CowAdapter
from serve.cow_http import make_server
from serve.cow_text_adapter import CowTextAdapter
from serve.cow_workspace import CowWorkspace
from serve.test_cow_answer_retry import RetryFlowModel, RetryTokenizer
from serve.test_cow_presentation_flow import (
    append_initial_user_suffix,
    append_user_suffix,
    encode,
    initial_root_mode,
    render,
)
from context_time_travel.sampled_controller import PlainSamplingSession, start
from context_time_travel.store import NotFound
from tools import cow_cli


REPO = Path(__file__).resolve().parents[1]
CLI = REPO / 'tools' / 'cow_cli.py'


class StopRetryFlowModel(RetryFlowModel):
    plans = {**RetryFlowModel.plans,
             'stop-answer': tuple(ord(char) for char in 'A<STOP>tail') + (299,),
             'changed-stop-retry': tuple(ord(char) for char in 'B<OTHER>tail') + (299,),
             'append-after-stop': tuple(ord(char) for char in 'C<OTHER>tail') + (299,),
             'stop-partial': (ord('A'), 261, ord('<'), ord('S'), ord('T')),
             'stop-direct-mid': (ord('O'),),
             'stop-continue': tuple(map(ord, 'P>tail')) + (299,)}

    def set_random_stream(self, snapshot, stream):
        state = copy.deepcopy(self.store.get_snapshot(snapshot)['controller'])
        if state.get('status') == 'idle':
            raise ValueError('No generation scope')
        state['stream_key'] = hashlib.sha256(stream.encode('utf-8')).hexdigest()
        state['next_draw'] = 0
        target = self.store.append(snapshot, (), state)
        if snapshot in self.cache:
            self.cache[target] = self.cache[snapshot]
        return target

    def run_sampled_request(self, sid, ids, count, *, sampling, stream, resume=False,
                            on_token=None, cancelled=None, text_stops=None):
        """Use the real consumed-token controller with a deterministic CPU plan."""
        if cancelled is not None and cancelled():
            raise RuntimeError('unexpected synthetic cancellation')
        history = self.history(sid)
        prompt_history = history + tuple(ids)
        if resume:
            state = self.store.get_snapshot(sid)['controller']
        else:
            state = start(sampling, stream, prompt_history, 300, [299],
                          text_stops=text_stops or ())
        session = PlainSamplingSession.restore(state, prompt_history, 300,
            RetryTokenizer().token_bytes, lambda token: token == 299)
        if resume:
            session.configure_generation(sampling)
        if stream == 'append-after-stop':
            plan = tuple(self.plans[stream])
        elif tuple(text_stops or ()) == ('<OTHER>',):
            plan = tuple(map(ord, 'B<OTHER>tail')) + (299,)
        else:
            plan = tuple(self.plans.get(stream, (299,)))
        outputs = []
        mask = bytearray(b'\xff' * 38)
        mask[-1] &= 0x0f
        for index in range(count):
            if cancelled is not None and cancelled():
                raise RuntimeError('unexpected synthetic cancellation')
            before = session.current_controller
            token = plan[index] if index < len(plan) else 299
            session.choose(bytes(mask), lambda *_args, **_kwargs: token)
            session.consume(before, token, RetryTokenizer().token_bytes(token), token == 299)
            outputs.append(token)
            if on_token is not None:
                on_token(token)
            if session.is_stopped():
                break
        target = self.store.append(sid, tuple(ids) + tuple(outputs), session.current_controller)
        self.stats['forward_tokens'] += len(ids) + len(outputs)
        self.stats['consumed_tokens'] += len(ids) + len(outputs)
        self.image(target)
        self.trace.append((stream, tuple(ids), tuple(outputs), 'sampled', dict(sampling)))
        return dict(snapshot=target, tokens=outputs, request_seconds=0.0,
                    generation_seconds=0.0)


class CowTextStopsCLITests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.gate = threading.Event()
        self.gate.set()
        self.trace = []
        self.models = {}
        self.config = PoolConfig(gpus=(0,), ram_bytes=1024 * 1024,
                                 disk_bytes=8 * 1024 * 1024,
                                 min_available_bytes=0)

        def factory(gpu):
            model = StopRetryFlowModel(self.root, gpu, (0,), self.trace, self.gate)
            self.models[gpu] = model
            return model

        self.coordinator = ContextCoordinator(self.root, self.config, factory)
        self.coordinator.ready(5)
        self.addCleanup(self._close)
        self.adapter = CowAdapter(self.coordinator)
        self.text = CowTextAdapter(self.adapter, RetryTokenizer(), render, encode,
            append_user_suffix=append_user_suffix,
            append_initial_user_suffix=append_initial_user_suffix,
            initial_root_mode=initial_root_mode, renderer_id='b' * 64)
        self.server = make_server(self.adapter, api_key='stop-test-key', text_adapter=self.text)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base = f'http://127.0.0.1:{self.server.server_port}'

    def _close(self):
        server = getattr(self, 'server', None)
        if server is not None:
            server.shutdown()
            server.server_close()
            self.thread.join(5)
            self.server = None
        coordinator = getattr(self, 'coordinator', None)
        if coordinator is not None:
            coordinator.close()
            self.coordinator = None

    def _cli(self, state_dir, *args):
        env = os.environ.copy()
        env['COW_API_KEY'] = 'stop-test-key'
        env['COW_CLI_STATE_DIR'] = str(state_dir)
        env.pop('PYTHONPATH', None)
        return subprocess.run(
            [sys.executable, str(CLI), '--url', self.base, '--state-dir', str(state_dir), *args],
            cwd=REPO, capture_output=True, env=env, timeout=30, check=False)

    def _create_loaded_branch(self, state_dir, branch):
        context = self.root / f'{branch}.txt'
        context.write_text('context\n', encoding='utf-8')
        created = self._cli(state_dir, 'namespace', '--title', 'Revision CAS',
                            '--operation-id', f'{branch}-namespace')
        self.assertEqual(created.returncode, 0, created.stderr.decode('utf-8', 'replace'))
        namespace = json.loads(created.stderr.decode('utf-8').split(
            'durable receipt:\n', 1)[1])['namespace_id']
        loaded = self._cli(state_dir, 'load', '--file', str(context),
                           '--namespace-id', namespace, '--branch', branch,
                           '--operation-id', f'{branch}-load')
        self.assertEqual(loaded.returncode, 0, loaded.stderr.decode('utf-8', 'replace'))

    def test_cli_ask_stop_hides_marker_but_commits_consumed_tokens(self):
        with tempfile.TemporaryDirectory() as client_state:
            context = self.root / 'context.txt'
            context.write_text('context\n', encoding='utf-8')
            created = self._cli(client_state, 'namespace', '--title', 'Stops',
                                '--operation-id', 'stop-namespace')
            self.assertEqual(created.returncode, 0, created.stderr.decode('utf-8', 'replace'))
            namespace = json.loads(created.stderr.decode('utf-8').split('durable receipt:\n', 1)[1])['namespace_id']
            loaded = self._cli(client_state, 'load', '--file', str(context),
                               '--namespace-id', namespace, '--branch', 'chat',
                               '--operation-id', 'stop-load')
            self.assertEqual(loaded.returncode, 0, loaded.stderr.decode('utf-8', 'replace'))

            answer = self._cli(client_state, 'ask', '--branch', 'chat', '--text', 'say hello',
                               '--max-tokens', '16', '--operation-id', 'stop-answer',
                               '--stop', '<STOP>')
            self.assertEqual(answer.returncode, 0, answer.stderr.decode('utf-8', 'replace'))
            self.assertEqual(answer.stdout.decode('utf-8'), 'A\n')
            head = self.coordinator.store.head('chat')
            controller = self.coordinator.store.get_snapshot(head.tip)['controller']
            self.assertEqual(controller['status'], 'stopped')
            self.assertEqual(controller['text_stop']['matched_stop'], '<STOP>')
            self.assertEqual(controller['accepted_in_scope'], 7)
            self.assertEqual(controller['text_stop']['pending_hex'], '')
            self.assertNotIn('<STOP>', answer.stdout.decode('utf-8'))
            self.assertGreaterEqual(self.models[0].stats['forward_tokens'], 8)
            original_head = self.coordinator.store.head('chat')
            original_receipt = self.coordinator.store.presentation_receipt('stop-answer')
            original_stats = dict(self.models[0].stats)
            with tempfile.TemporaryDirectory() as retry_state:
                replay = self._cli(retry_state, 'retry', '--from-operation', 'stop-answer')
            self.assertEqual(replay.returncode, 0, replay.stderr.decode('utf-8', 'replace'))
            self.assertEqual(replay.stdout, answer.stdout)
            self.assertEqual(self.coordinator.store.head('chat'), original_head)
            self.assertEqual(self.coordinator.store.presentation_receipt('stop-answer'), original_receipt)
            self.assertEqual(self.models[0].stats, original_stats)

            changed = self._cli(client_state, 'retry', '--from-operation', 'stop-answer',
                                '--operation-id', 'changed-stop-retry', '--stop', '<OTHER>')
            self.assertEqual(changed.returncode, 0, changed.stderr.decode('utf-8', 'replace'))
            self.assertEqual(changed.stdout.decode('utf-8'), 'B\n')
            changed_head = self.coordinator.store.head('chat')
            self.assertEqual(changed_head.revision, original_head.revision + 1)
            self.assertEqual(self.coordinator.store.presentation_receipt('stop-answer'), original_receipt)
            changed_controller = self.coordinator.store.get_snapshot(changed_head.tip)['controller']
            self.assertEqual(changed_controller['text_stop']['matched_stop'], '<OTHER>')

            # A new assistant turn starts a fresh matcher even when it uses
            # the same configured marker as the stopped turn before it.
            appended = self._cli(client_state, 'append-user', '--branch', 'chat',
                                 '--text', 'follow up', '--max-tokens', '16',
                                 '--operation-id', 'append-after-stop', '--stop', '<OTHER>')
            self.assertEqual(appended.returncode, 0, appended.stderr.decode('utf-8', 'replace'))
            self.assertEqual(appended.stdout.decode('utf-8'), 'C\n')
            appended_head = self.coordinator.store.head('chat')
            appended_receipt = self.coordinator.store.presentation_receipt('append-after-stop')
            appended_controller = self.coordinator.store.get_snapshot(appended_head.tip)['controller']
            self.assertEqual(appended_controller['text_stop']['matched_stop'], '<OTHER>')
            with tempfile.TemporaryDirectory() as append_retry_state:
                append_replay = self._cli(append_retry_state, 'retry', '--from-operation',
                                          'append-after-stop')
            self.assertEqual(append_replay.returncode, 0,
                             append_replay.stderr.decode('utf-8', 'replace'))
            self.assertEqual(append_replay.stdout, appended.stdout)
            self.assertEqual(self.coordinator.store.head('chat'), appended_head)
            self.assertEqual(self.coordinator.store.presentation_receipt('append-after-stop'),
                             appended_receipt)

    def test_partial_stop_survives_cli_continue_and_history_projection(self):
        with tempfile.TemporaryDirectory() as client_state:
            context = self.root / 'context-partial.txt'
            context.write_text('context\n', encoding='utf-8')
            created = self._cli(client_state, 'namespace', '--title', 'Partial stops',
                                '--operation-id', 'partial-stop-ns')
            self.assertEqual(created.returncode, 0, created.stderr.decode('utf-8', 'replace'))
            namespace = json.loads(created.stderr.decode('utf-8').split('durable receipt:\n', 1)[1])['namespace_id']
            loaded = self._cli(client_state, 'load', '--file', str(context),
                               '--namespace-id', namespace, '--branch', 'partial',
                               '--operation-id', 'partial-stop-load')
            self.assertEqual(loaded.returncode, 0, loaded.stderr.decode('utf-8', 'replace'))
            first = self._cli(client_state, 'ask', '--branch', 'partial', '--text', 'say hello',
                              '--max-tokens', '5', '--operation-id', 'stop-partial', '--stop', '<STOP>')
            self.assertEqual(first.returncode, 0, first.stderr.decode('utf-8', 'replace'))
            self.assertEqual(first.stdout.decode('utf-8'), 'A\n')
            partial_head = self.coordinator.store.head('partial')
            partial_controller = self.coordinator.store.get_snapshot(partial_head.tip)['controller']
            self.assertEqual(partial_controller['status'], 'active')
            self.assertEqual(partial_controller['text_stop']['pending_hex'], '3c5354')
            self.assertEqual(partial_controller['output_pending_hex'], 'e2')
            partial_replay = self.text.retry('stop-partial', 'partial', partial_head.revision,
                                             'stop-partial')
            self.assertTrue(partial_replay['replayed'])
            self.assertEqual(partial_replay['answer']['text'], 'A')

            # Request-level display must inherit the saved matcher and remain
            # partial. A fresh matcher or unconditional flush leaks the pending
            # UTF-8 replacement and ambiguous stop prefix.
            direct = self.text.resume('partial', partial_head.revision,
                                      'stop-direct-mid', 1)
            direct.durable.result(timeout=5)
            self.assertEqual(direct.text_result(), '')
            direct_head = self.coordinator.store.head('partial')
            direct_controller = self.coordinator.store.get_snapshot(direct_head.tip)['controller']
            self.assertEqual(direct_controller['status'], 'active')
            self.assertEqual(direct_controller['text_stop']['pending_hex'], '3c53544f')
            self.assertEqual(direct_controller['output_pending_hex'], 'e2')
            forwards_before_bad_continue = self.models[0].stats['forward_tokens']

            rejected = self._cli(client_state, 'continue', '--branch', 'partial', '--max-tokens', '4',
                                 '--operation-id', 'bad-stop-change', '--stop', '<OTHER>')
            self.assertNotEqual(rejected.returncode, 0)
            self.assertEqual(self.models[0].stats['forward_tokens'], forwards_before_bad_continue)
            self.assertEqual(self.coordinator.store.head('partial'), direct_head)

            second = self._cli(client_state, 'continue', '--branch', 'partial', '--max-tokens', '8',
                               '--operation-id', 'stop-continue')
            self.assertEqual(second.returncode, 0, second.stderr.decode('utf-8', 'replace'))
            self.assertEqual(second.stdout.decode('utf-8'), '\ufffd\n')
            head = self.coordinator.store.head('partial')
            controller = self.coordinator.store.get_snapshot(head.tip)['controller']
            self.assertEqual(controller['status'], 'stopped')
            self.assertEqual(controller['text_stop']['matched_stop'], '<STOP>')
            self.assertEqual(controller['output_pending_hex'], 'e2')
            self.assertEqual(controller['accepted_in_scope'], 8)
            self.assertNotIn('<STOP>', first.stdout.decode('utf-8') + second.stdout.decode('utf-8'))
            history = CowWorkspace(self.coordinator.store, self.text.tokenizer).history('partial')
            assistant = [message for message in history['messages'] if message['role'] == 'assistant']
            self.assertEqual(len(assistant), 1)
            self.assertEqual(assistant[0]['text'], 'A\ufffd')

    def test_cli_revision_rejects_pre_read_stale_head_without_mutation(self):
        with tempfile.TemporaryDirectory() as client_state:
            self._create_loaded_branch(client_state, 'stale-before-read')
            old_revision = self.coordinator.store.head('stale-before-read').revision
            self.models[0].plans['advance-before-read'] = (ord('A'), 299)
            advanced = self._cli(client_state, 'ask', '--branch', 'stale-before-read',
                '--text', 'first answer', '--max-tokens', '2', '--operation-id',
                'advance-before-read')
            self.assertEqual(advanced.returncode, 0,
                advanced.stderr.decode('utf-8', 'replace'))
            head = self.coordinator.store.head('stale-before-read')
            counters = dict(self.models[0].stats)
            stale = self._cli(client_state, 'ask', '--branch', 'stale-before-read',
                '--text', 'must not run', '--max-tokens', '2', '--operation-id',
                'stale-before-read', '--revision', str(old_revision))
            self.assertEqual(stale.returncode, 2)
            self.assertIn('branch revision changed', stale.stderr.decode('utf-8', 'replace'))
            self.assertEqual(self.coordinator.store.head('stale-before-read'), head)
            self.assertEqual(self.models[0].stats, counters)
            with self.assertRaises(NotFound):
                self.coordinator.store.presentation_receipt('stale-before-read')

    def test_cli_revision_is_sent_unchanged_and_server_cas_closes_post_read_race(self):
        with tempfile.TemporaryDirectory() as client_state:
            self._create_loaded_branch(client_state, 'stale-after-read')
            starting = self.coordinator.store.head('stale-after-read')
            counters = dict(self.models[0].stats)
            original_head = cow_cli._head
            original_stream = cow_cli.CowClient.stream
            fired = False
            posted = []
            cli_stderr = io.StringIO()
            cli_stdout = io.StringIO()

            def move_after_head_read(client, branch):
                nonlocal fired
                result = original_head(client, branch)
                if not fired and branch == 'stale-after-read':
                    fired = True
                    current = self.coordinator.store.head('stale-after-read')
                    self.coordinator.store.move(current.name, current.revision, current.tip,
                                                'external-cas-race')
                return result

            def capture_mutation_body(client, path, body, on_provisional):
                if path == '/cow/text/ask':
                    posted.append(json.loads(body.decode('utf-8')))
                return original_stream(client, path, body, on_provisional)

            with patch.object(cow_cli, '_head', move_after_head_read), \
                    patch.object(cow_cli.CowClient, 'stream', capture_mutation_body), \
                    patch.dict(os.environ, {'COW_API_KEY': 'stop-test-key',
                                            'COW_CLI_STATE_DIR': str(client_state)}):
                with contextlib.redirect_stderr(cli_stderr), contextlib.redirect_stdout(cli_stdout):
                    code = cow_cli.main([
                        '--url', self.base, '--state-dir', client_state, 'ask',
                        '--branch', 'stale-after-read', '--text', 'must not run',
                        '--max-tokens', '2', '--operation-id', 'stale-after-read',
                        '--revision', str(starting.revision)])
            self.assertTrue(fired, f'the CLI did not read the branch head; code={code}; '
                                   f'stderr={cli_stderr.getvalue()!r}; stdout={cli_stdout.getvalue()!r}')
            self.assertEqual(posted, [dict(branch='stale-after-read', operation_id='stale-after-read',
                stream=True, max_tokens=2, revision=starting.revision, content='must not run')])
            self.assertEqual(code, 2)
            final = self.coordinator.store.head('stale-after-read')
            self.assertEqual(final.revision, starting.revision + 1)
            self.assertEqual(final.tip, starting.tip)
            self.assertEqual(self.models[0].stats, counters)
            with self.assertRaises(NotFound):
                self.coordinator.store.presentation_receipt('stale-after-read')

    def test_seeded_text_stop_survives_reseed_continue_and_exact_replay(self):
        with tempfile.TemporaryDirectory() as client_state:
            self._create_loaded_branch(client_state, 'seed-stop')
            self.models[0].plans['cow-seed-v1:0'] = tuple(map(ord, 'A<'))
            self.models[0].plans['seed-stop-continue'] = tuple(map(ord, 'STOP>tail')) + (299,)
            answer_revision = self.coordinator.store.head('seed-stop').revision
            first = self._cli(client_state, '--machine-result', 'ask', '--branch', 'seed-stop', '--text', 'question',
                '--max-tokens', '2', '--operation-id', 'seed-stop-answer', '--seed', '0',
                '--stop', '<STOP>', '--temperature', '0.7', '--revision', str(answer_revision))
            self.assertEqual(first.returncode, 0, first.stderr.decode('utf-8', 'replace'))
            machine_result = json.loads(first.stdout.decode('utf-8'))
            self.assertEqual(machine_result['schema'], 1)
            self.assertEqual(machine_result['answer']['text'], 'A')
            self.assertNotIn('<STOP>', machine_result['answer']['text'])
            self.assertEqual(machine_result['receipt']['branch']['name'], 'seed-stop')
            self.assertEqual(machine_result['receipt']['branch']['revision'], answer_revision + 1)
            initial = self.coordinator.store.head('seed-stop')
            initial_controller = self.coordinator.store.get_snapshot(initial.tip)['controller']
            self.assertEqual(initial_controller['status'], 'active')
            self.assertEqual(initial_controller['text_stop']['pending_hex'], '3c')
            self.assertGreater(initial_controller['next_draw'], 0)
            answer_receipt = self.coordinator.store.presentation_receipt('seed-stop-answer')

            reseeded = self._cli(client_state, 'reseed', '--branch', 'seed-stop',
                '--seed', '73',
                '--operation-id', 'seed-stop-reseed')
            self.assertEqual(reseeded.returncode, 0, reseeded.stderr.decode('utf-8', 'replace'))
            reseed_head = self.coordinator.store.head('seed-stop')
            reseed_controller = self.coordinator.store.get_snapshot(reseed_head.tip)['controller']
            self.assertEqual(reseed_controller['text_stop'], initial_controller['text_stop'])
            self.assertEqual(reseed_controller['accepted_in_scope'], initial_controller['accepted_in_scope'])
            self.assertEqual(reseed_controller['status'], 'active')
            self.assertEqual(reseed_controller['next_draw'], 0)
            self.assertNotEqual(reseed_controller['stream_key'], initial_controller['stream_key'])

            continued = self._cli(client_state, 'continue', '--branch', 'seed-stop',
                '--revision', str(reseed_head.revision), '--max-tokens', '16',
                '--operation-id', 'seed-stop-continue')
            self.assertEqual(continued.returncode, 0, continued.stderr.decode('utf-8', 'replace'))
            stopped_head = self.coordinator.store.head('seed-stop')
            stopped_controller = self.coordinator.store.get_snapshot(stopped_head.tip)['controller']
            self.assertEqual(stopped_controller['status'], 'stopped')
            self.assertEqual(stopped_controller['text_stop']['matched_stop'], '<STOP>')
            self.assertEqual(stopped_controller['accepted_in_scope'],
                             initial_controller['accepted_in_scope'] + len('STOP>'))
            self.assertEqual(self.coordinator.store.presentation_receipt('seed-stop-answer'),
                             answer_receipt)
            history = CowWorkspace(self.coordinator.store, self.text.tokenizer).history('seed-stop')
            assistant = [message for message in history['messages'] if message['role'] == 'assistant']
            self.assertEqual(len(assistant), 1)
            self.assertEqual(assistant[0]['text'], 'A')

            with tempfile.TemporaryDirectory() as replay_state:
                replay = self._cli(replay_state, 'retry', '--from-operation', 'seed-stop-continue')
            self.assertEqual(replay.returncode, 0, replay.stderr.decode('utf-8', 'replace'))
            self.assertEqual(replay.stdout, continued.stdout)
            self.assertEqual(self.coordinator.store.head('seed-stop'), stopped_head)


if __name__ == '__main__':
    unittest.main()
