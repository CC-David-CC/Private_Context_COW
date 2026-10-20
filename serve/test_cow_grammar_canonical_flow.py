"""Canonical branch tests for grammar progress through existing COW storage.

These are CPU integration controls: the coordinator, Store, CowTextAdapter,
BranchingModel, GrammarRegistry, GrammarEngine, and controller are real; the
backend is the small deterministic control backend used by the existing root
reuse tests. They exercise canonical grammar routing and do not qualify native
or GPU inference.
"""
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from context_time_travel.coordinator import ContextCoordinator
from context_time_travel.pools import PoolConfig
from context_time_travel.real import BranchingModel
from context_time_travel.store import InvalidOperation
from serve.cow_adapter import CowAdapter
from serve.cow_grammar import GrammarRegistry
from serve.cow_text_adapter import CowTextAdapter, UnsupportedTextRequest
from test_grammar_first import TinyByteTokenizer
from test_root_computation_reuse import CPUBackend


def _event(operator, position, tokens):
    from context_time_travel.execution_receipt import tokens_digest
    tokens = tuple(tokens)
    chunks = ([[position + start, min(16, len(tokens) - start)]
               for start in range(0, len(tokens), 16)]
              if operator == 'batched-prefill'
              else [[position + index, 1] for index in range(len(tokens))])
    return dict(operator=operator, position=position, count=len(tokens),
                tokens_sha256=tokens_digest(tokens), chunks=chunks)


class GrammarCPUBackend(CPUBackend):
    """The existing CPU backend with exact receipts for BranchingModel's path."""
    def __init__(self, profile, tokenizer):
        super().__init__(profile)
        self.vocab_size = len(tokenizer.tokens)
        self.tokenizer = tokenizer
        self.stats.update(forward_tokens=0, consumed_tokens=0, loaded_file_bytes=0)
        self.selected_tokens = []

    def piece(self, token):
        return self.tokenizer.token_bytes(token)

    def is_eog(self, token):
        return token == self.profile['execution']['eos_tokens'][0]

    def consume(self, lease, ids):
        ids = tuple(ids)
        position = len(lease['ids'])
        super().consume(lease, ids)
        self.last_execution_receipt = _event('batched-prefill', position, ids)

    def consume_generated(self, lease, ids):
        ids = tuple(ids)
        position = len(lease['ids'])
        super().consume(lease, ids)
        self.selected_tokens.extend(ids)
        self.last_execution_receipt = _event('normal-verifier-decode', position, ids)

    def validate_grammar_engine(self, engine):
        if engine.vocab_size != self.vocab_size:
            raise InvalidOperation('Grammar tokenizer vocabulary differs from CPU model fixture')
        if engine.tokenizer_sha256 != self.tokenizer_hash:
            raise InvalidOperation('Grammar tokenizer identity differs from CPU model fixture')

    def sample_legal(self, lease, mask, temperature, min_p, draw53, **filters):
        """CPU-only deterministic selector for sampled grammar progress tests."""
        self.stats['sampling_calls'] = self.stats.get('sampling_calls', 0) + 1
        for token in range(self.vocab_size):
            if mask[token >> 3] & (1 << (token & 7)):
                return token
        raise InvalidOperation('Grammar mask unexpectedly has no legal token')

    @property
    def tokenizer_hash(self):
        from context_time_travel.grammar_first import GrammarEngine
        return GrammarEngine._tokenizer_hash(self.tokenizer)


def _render(messages, *, add_generation_prompt):
    rendered = ''.join(f"{message['role']}:{message['content']}\n" for message in messages)
    return rendered + ('assistant:' if add_generation_prompt else '')


def _encode(tokenizer, messages, rendered, add_generation_prompt):
    if rendered != _render(messages, add_generation_prompt=add_generation_prompt):
        raise AssertionError('Unexpected rendered prompt')
    return tuple(tokenizer.encode(rendered, parse_special=False))


def _initial_root_mode(canonical, controller):
    if controller == {'status': 'idle'} and canonical:
        return 'prepared_answer'
    return 'unsupported'


class CowCanonicalGrammarFlowTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='cow-canonical-grammar-')
        self.addCleanup(self.tmp.cleanup)
        self.tokenizer = TinyByteTokenizer()
        self.registry_path = Path(self.tmp.name) / 'registry.json'
        self.registry_path.write_text(json.dumps({'schema': 1, 'states': {
            'READ': {'format': 'gbnf', 'source': 'root ::= "A" "B"'},
            'EDIT': {'format': 'gbnf', 'source': 'root ::= "X" "Y"'},
            'TEST': {'format': 'gbnf', 'source': 'root ::= "Z"'},
        }}), encoding='utf-8')
        self.registry = GrammarRegistry.load(self.registry_path, self.tokenizer,
                                             [TinyByteTokenizer.EOS])

        self.profile = dict(model_contract_id='grammar-cpu-model',
            execution_profile_id='grammar-cpu-execution', model_weights_sha256='a' * 64,
            execution=dict(GPU=[0], resume_contract='identical-native-operators-v1',
                generated_operator='normal-verifier-decode',
                native_request_protocol='consumed-greedy-ack-v1',
                selected_token_protocol='consumed-selected-receipt-v1',
                sampling_protocol='full-vocab-adaptive-p-u53-v7',
                prefill_chunk=16, capacity=4096, eos_tokens=[TinyByteTokenizer.EOS]))
        self.backends = {}

        def factory(gpu):
            backend = GrammarCPUBackend(self.profile, self.tokenizer)
            self.backends[gpu] = backend
            return BranchingModel(self.tmp.name, backend, self.profile)

        self.coordinator = ContextCoordinator(self.tmp.name,
            PoolConfig(gpus=(0,), ram_bytes=64 * 1024**2, disk_bytes=256 * 1024**2,
                       min_available_bytes=0), factory)
        self.addCleanup(self.coordinator.close)
        self.coordinator.ready(timeout=5)
        self.namespace = self.coordinator.store.create_namespace('Grammar fixture', 'ns-op').namespace_id
        self.token_adapter = CowAdapter(self.coordinator, grammar_registry=self.registry)
        self.text = CowTextAdapter(self.token_adapter, self.tokenizer, _render,
            lambda messages, rendered, add_generation_prompt:
                _encode(self.tokenizer, messages, rendered, add_generation_prompt),
            initial_root_mode=_initial_root_mode, renderer_id='b' * 64, grammar_registry=self.registry)

    def _canonical_forked_root(self):
        root_request = self.text.prepare('base', [{'role': 'user', 'content': 'code'}],
            'root-op', namespace_id=self.namespace)
        root_receipt = root_request.durable.result(5)
        root_head = self.coordinator.store.head('base')
        self.assertEqual(root_receipt['snapshot'], root_head.tip)
        a = self.coordinator.fork('BranchA', root_head.tip, source_branch='base',
                                  operation_id='fork-a')
        b = self.coordinator.fork('BranchB', root_head.tip, source_branch='base',
                                  operation_id='fork-b')
        self.assertEqual(a.tip, root_head.tip)
        self.assertEqual(b.tip, root_head.tip)
        self.assertEqual(self.coordinator.store.branch_namespace('BranchA'), self.namespace)
        self.assertEqual(self.coordinator.store.branch_namespace('BranchB'), self.namespace)
        return root_head, a, b

    def _run_state(self, branch, revision, snapshot, state, operation, *, resume=False, limit=1):
        model = self.coordinator._workers[0]['model']
        result = model.run_state_request(snapshot, (), limit, engine=self.registry._engine,
            state=state, stream=operation, resume=resume)
        durable = model.save_branch(branch, revision, result['snapshot'], operation)
        self.assertEqual(durable['snapshot'], result['snapshot'])
        return result, durable

    def test_coordinator_forks_preserve_independent_real_grammar_controllers(self):
        root, a, b = self._canonical_forked_root()
        model = self.coordinator._workers[0]['model']
        parent_bytes = model.snapshot_bytes(root.tip)

        first_a, saved_a = self._run_state('BranchA', a.revision, a.tip, 'READ', 'read-start')
        first_b, saved_b = self._run_state('BranchB', b.revision, b.tip, 'EDIT', 'edit-start')
        self.assertEqual(first_a['tokens'], [ord('A')])
        self.assertEqual(first_b['tokens'], [ord('X')])
        self.assertEqual(self.coordinator.store.head('BranchA').tip, saved_a['snapshot'])
        self.assertEqual(self.coordinator.store.head('BranchB').tip, saved_b['snapshot'])

        state_a = model.controller(saved_a['snapshot'])
        state_b = model.controller(saved_b['snapshot'])
        self.assertEqual(state_a['constraint']['agent_state'], 'READ')
        self.assertEqual(state_b['constraint']['agent_state'], 'EDIT')
        self.assertEqual(state_a['accepted_in_scope'], 1)
        self.assertEqual(state_b['accepted_in_scope'], 1)
        self.assertEqual(state_a['constraint']['engine_id'], self.registry.engine_id)
        self.assertEqual(state_b['constraint']['engine_id'], self.registry.engine_id)
        self.assertEqual(model.snapshot_bytes(root.tip), parent_bytes)
        self.assertEqual(model.history(saved_a['snapshot'])[:-1], model.history(root.tip))
        self.assertEqual(model.history(saved_b['snapshot'])[:-1], model.history(root.tip))

        next_a, durable_a = self._run_state('BranchA', saved_a['branch']['revision'],
            saved_a['snapshot'], 'READ', 'read-resume', resume=True)
        next_b, durable_b = self._run_state('BranchB', saved_b['branch']['revision'],
            saved_b['snapshot'], 'EDIT', 'edit-resume', resume=True)
        self.assertEqual(next_a['tokens'], [ord('B')])
        self.assertEqual(next_b['tokens'], [ord('Y')])
        self.assertEqual(model.history(durable_a['snapshot'])[-2:], (ord('A'), ord('B')))
        self.assertEqual(model.history(durable_b['snapshot'])[-2:], (ord('X'), ord('Y')))
        self.assertEqual(model.controller(durable_a['snapshot'])['accepted_in_scope'], 2)
        self.assertEqual(model.controller(durable_b['snapshot'])['accepted_in_scope'], 2)
        self.assertEqual(self.coordinator.store.head('BranchA').tip, durable_a['snapshot'])
        self.assertEqual(self.coordinator.store.head('BranchB').tip, durable_b['snapshot'])

    def test_active_parser_rejects_host_state_change_before_backend_work(self):
        root, a, _ = self._canonical_forked_root()
        first, saved = self._run_state('BranchA', a.revision, a.tip, 'READ', 'read-partial')
        self.assertEqual(first['tokens'], [ord('A')])
        backend = self.backends[0]
        before = dict(backend.stats)
        with self.assertRaises(InvalidOperation):
            self.coordinator._workers[0]['model'].run_state_request(saved['snapshot'], (), 1,
                engine=self.registry._engine, state='EDIT', stream='wrong-state', resume=True)
        self.assertEqual(backend.stats, before)
        self.assertEqual(self.coordinator.store.head('BranchA').tip, saved['snapshot'])
        self.assertEqual(self.coordinator.store.head('base').tip, root.tip)

    def test_completed_grammar_can_start_a_host_selected_new_generation_scope(self):
        root, a, _ = self._canonical_forked_root()
        complete, saved = self._run_state('BranchA', a.revision, a.tip, 'READ', 'read-complete', limit=2)
        self.assertEqual(complete['tokens'], [ord('A'), ord('B')])
        controller = self.coordinator._workers[0]['model'].controller(saved['snapshot'])
        self.assertEqual(controller['accepted_in_scope'], 2)
        self.assertTrue(self.registry.restore(controller, self.coordinator._workers[0]['model'].history(saved['snapshot'])).is_complete())

        # A different host state is a new grammar scope, never an implicit
        # mutation of the completed READ parser or of the sibling/root.
        test_turn, durable = self._run_state('BranchA', saved['branch']['revision'],
            saved['snapshot'], 'TEST', 'test-new-scope', resume=False, limit=1)
        self.assertEqual(test_turn['tokens'], [ord('Z')])
        transitioned = self.coordinator._workers[0]['model'].controller(durable['snapshot'])
        self.assertEqual(transitioned['constraint']['agent_state'], 'TEST')
        self.assertEqual(transitioned['accepted_in_scope'], 1)
        self.assertEqual(self.coordinator.store.head('base').tip, root.tip)

    def test_model_profile_and_registry_identity_are_checked_by_grammar_engine(self):
        self.assertEqual(self.registry.states, ('EDIT', 'READ', 'TEST'))
        changed_data = json.loads(self.registry_path.read_text(encoding='utf-8'))
        changed_data['states']['READ']['source'] = 'root ::= "A" "Z"'
        changed = json.dumps(changed_data)
        other_path = Path(self.tmp.name) / 'other-registry.json'
        other_path.write_text(changed, encoding='utf-8')
        other = GrammarRegistry.load(other_path, self.tokenizer, [TinyByteTokenizer.EOS])
        self.assertNotEqual(self.registry.engine_id, other.engine_id)
        with self.assertRaises(InvalidOperation):
            self.registry.start('NOT_CONFIGURED', 'bad-state', (), len(self.tokenizer.tokens))

    def test_missing_server_registry_fails_closed_at_load(self):
        with self.assertRaises(InvalidOperation):
            GrammarRegistry.load(Path(self.tmp.name) / 'missing-registry.json',
                                 self.tokenizer, [TinyByteTokenizer.EOS])

    def test_text_answer_selects_registry_state_and_publishes_generation_binding(self):
        self.text.append_user_suffix = lambda _history, _controller, content: tuple(
            self.tokenizer.encode(content, parse_special=False))
        prepared = self.text.prepare('Prepared', [{'role': 'user', 'content': 'read'}],
            'prepared-op', namespace_id=self.namespace)
        prepared.durable.result(5)
        head = self.coordinator.store.head('Prepared')

        answer = self.text.answer('Prepared', head.revision, 'grammar-answer', 3,
                                  grammar_state='READ')
        receipt = answer.durable.result(5)
        self.assertEqual(answer.snapshot.result()['tokens'],
                         [ord('A'), ord('B'), TinyByteTokenizer.EOS])
        saved = self.coordinator.store.get_snapshot(receipt['snapshot'])['controller']
        self.assertEqual(saved['constraint']['agent_state'], 'READ')
        self.assertEqual(saved['constraint']['engine_id'], self.registry.engine_id)
        self.assertEqual(saved['status'], 'stopped')
        intent = json.loads((self.coordinator._intents / 'grammar-answer.json').read_bytes())['intent']
        generation = intent['generation']
        self.assertEqual(generation['schema'], 2)
        self.assertEqual(generation['grammar'], {
            'schema': 1, 'agent_state': 'READ', 'engine_id': self.registry.engine_id,
            'spec_sha256': self.registry.spec_sha256,
            'tokenizer_sha256': self.registry.tokenizer_sha256,
            'llguidance_version': self.registry.llguidance_version,
        })
        self.assertNotIn('accepted_in_scope', generation['grammar'])

    def test_generated_long_grammar_state_name_survives_committed_generation_validation(self):
        state = 'g' + 'a' * 64 + '_s0'
        self.assertGreater(len(state), 64)
        registry_path = Path(self.tmp.name) / 'long-state-registry.json'
        registry_path.write_text(json.dumps({'schema': 1, 'states': {
            state: {'format': 'gbnf', 'source': 'root ::= "A" "B"'},
        }}), encoding='utf-8')
        registry = GrammarRegistry.load(registry_path, self.tokenizer,
                                        [TinyByteTokenizer.EOS])
        self.text.grammar_registry = registry
        self.token_adapter.grammar_registry = registry
        prepared = self.text.prepare('LongGrammar',
            [{'role': 'user', 'content': 'read'}], 'long-state-root',
            namespace_id=self.namespace)
        prepared.durable.result(5)
        head = self.coordinator.store.head('LongGrammar')
        answer = self.text.answer('LongGrammar', head.revision,
            'long-state-answer', 3, grammar_state=state)
        answer.durable.result(5)
        committed = self.coordinator.committed_generation('long-state-answer')
        self.assertEqual(committed['generation']['grammar']['agent_state'], state)

    def test_changed_retry_requires_the_original_grammar_registry_identity(self):
        self.text.append_user_suffix = lambda _history, _controller, content: tuple(
            self.tokenizer.encode(content, parse_special=False))
        prepared = self.text.prepare('RetryGrammar', [{'role': 'user', 'content': 'read'}],
            'retry-grammar-root', namespace_id=self.namespace)
        prepared.durable.result(5)
        head = self.coordinator.store.head('RetryGrammar')
        answer = self.text.answer('RetryGrammar', head.revision, 'retry-grammar-answer', 3,
            grammar_state='READ')
        answer.durable.result(5)
        source = self.coordinator.committed_generation('retry-grammar-answer')
        self.assertEqual(source['generation']['schema'], 2)
        self.assertEqual(source['generation']['grammar'],
            self.registry.descriptor('READ'))

        changed_data = json.loads(self.registry_path.read_text(encoding='utf-8'))
        changed_data['states']['READ']['source'] = 'root ::= "A" "Z"'
        other_path = Path(self.tmp.name) / 'retry-other-registry.json'
        other_path.write_text(json.dumps(changed_data), encoding='utf-8')
        other = GrammarRegistry.load(other_path, self.tokenizer,
                                     [TinyByteTokenizer.EOS])
        before_stats = dict(self.backends[0].stats)
        current = self.coordinator.store.head('RetryGrammar')
        with self.assertRaises(InvalidOperation):
            self.coordinator.submit_retry_generation('retry-grammar-answer',
                'RetryGrammar', current.revision, 'retry-grammar-changed', None,
                source['presentation'], grammar_registry=other)
        self.assertEqual(self.backends[0].stats, before_stats)
        self.assertEqual(self.coordinator.store.head('RetryGrammar').tip,
                         current.tip)


    def test_actual_coordinator_dispatch_restores_parser_and_allows_host_transition_after_completion(self):
        self.text.append_user_suffix = lambda _history, _controller, content: tuple(
            self.tokenizer.encode(content, parse_special=False))
        prepared = self.text.prepare('Routed', [{'role': 'user', 'content': 'read'}],
            'route-root', namespace_id=self.namespace)
        prepared.durable.result(5)
        head = self.coordinator.store.head('Routed')

        first = self.text.answer('Routed', head.revision, 'route-read-first', 1,
                                 grammar_state='READ')
        first_receipt = first.durable.result(5)
        self.assertEqual(first.snapshot.result()['tokens'], [ord('A')])
        active = self.coordinator.store.get_snapshot(first_receipt['snapshot'])['controller']
        self.assertEqual(active['status'], 'active')
        self.assertEqual(active['accepted_in_scope'], 1)
        self.assertEqual(active['constraint']['agent_state'], 'READ')
        head = self.coordinator.store.head('Routed')
        before = dict(self.backends[0].stats)
        with self.assertRaises(UnsupportedTextRequest):
            self.text.resume('Routed', head.revision, 'route-invalid-switch', 1,
                grammar_state='EDIT')
        self.assertEqual(self.backends[0].stats, before)
        self.assertEqual(self.coordinator.store.head('Routed').tip, first_receipt['snapshot'])

        head = self.coordinator.store.head('Routed')
        continuation = self.text.resume('Routed', head.revision, 'route-read-resume', 2)
        continuation_receipt = continuation.durable.result(5)
        self.assertEqual(continuation.snapshot.result()['tokens'], [ord('B'), TinyByteTokenizer.EOS])
        completed = self.coordinator.store.get_snapshot(continuation_receipt['snapshot'])['controller']
        self.assertEqual(completed['status'], 'stopped')
        self.assertEqual(completed['accepted_in_scope'], 3)
        self.assertEqual(completed['constraint']['agent_state'], 'READ')

        head = self.coordinator.store.head('Routed')
        transitioned = self.text.append_user('Routed', 'test', head.revision,
            'route-next-test', 2, grammar_state='TEST')
        transition_receipt = transitioned.durable.result(5)
        self.assertEqual(transitioned.snapshot.result()['tokens'], [ord('Z'), TinyByteTokenizer.EOS])
        next_controller = self.coordinator.store.get_snapshot(transition_receipt['snapshot'])['controller']
        self.assertEqual(next_controller['constraint']['agent_state'], 'TEST')
        self.assertEqual(next_controller['accepted_in_scope'], 2)
        self.assertEqual(self.coordinator.store.head('Routed').tip, transition_receipt['snapshot'])
        self.assertEqual(self.coordinator.store.head('Routed').root,
                         prepared.durable.result(5)['snapshot'])

    def test_explicit_sampler_override_resumes_saved_grammar_progress(self):
        self.text.append_user_suffix = lambda _history, _controller, content: tuple(
            self.tokenizer.encode(content, parse_special=False))
        prepared = self.text.prepare('SampledGrammar', [{'role':'user','content':'read'}],
            'sampled-grammar-root', namespace_id=self.namespace)
        prepared.durable.result(5)
        head = self.coordinator.store.head('SampledGrammar')
        first = self.text.answer('SampledGrammar',head.revision,
            'sampled-grammar-first',1,grammar_state='READ')
        first_receipt = first.durable.result(5)
        before = self.coordinator.store.get_snapshot(first_receipt['snapshot'])['controller']
        self.assertEqual(before['constraint']['agent_state'],'READ')
        self.assertEqual(before['accepted_in_scope'],1)
        self.assertEqual(before['next_draw'],0)

        head = self.coordinator.store.head('SampledGrammar')
        continued = self.text.resume('SampledGrammar',head.revision,
            'sampled-grammar-continue',1,settings={'temperature':0.7,'min_p':0.0})
        receipt = continued.durable.result(5)
        self.assertEqual(continued.snapshot.result()['tokens'],[ord('B')])
        after = self.coordinator.store.get_snapshot(receipt['snapshot'])['controller']
        self.assertEqual(after['schema'],3)
        self.assertEqual(after['policy']['temperature'],0.7)
        self.assertEqual(after['policy']['min_p'],0.0)
        self.assertEqual(after['constraint']['agent_state'],'READ')
        self.assertEqual(after['accepted_in_scope'],2)
        self.assertEqual(after['next_draw'],1)
        self.assertEqual(self.coordinator.store.head('SampledGrammar').tip,receipt['snapshot'])
        self.assertEqual(self.backends[0].stats['sampling_calls'],1)


if __name__ == '__main__':
    unittest.main()
