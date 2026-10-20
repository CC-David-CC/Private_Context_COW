"""Synthetic full-stack chat COW flow; CPU control test, not GPU qualification."""
import hashlib
import struct
import tempfile
import threading
import unittest
from pathlib import Path

from context_time_travel import controller
from context_time_travel.coordinator import ContextCoordinator
from context_time_travel.pools import PoolConfig
from context_time_travel.real import Checkpoint, PreparedCheckpoint
from context_time_travel.store import Store
from serve.cow_adapter import CowAdapter
from serve.cow_text_adapter import CowTextAdapter
from serve.cow_workspace import CowWorkspace


class ByteTokenizer:
    def token_bytes(self, token):
        generated = {257: b'hi', 258: b'!', 260: b'next', 299: b''}
        return generated[token] if token in generated else bytes([token])


def render(messages, *, add_generation_prompt):
    text = ''.join(f"{item['role']}:{item['content'].strip()}\n" for item in messages)
    return text + ('assistant:' if add_generation_prompt else '')


def encode(messages, rendered, add_generation_prompt):
    assert rendered == render(messages, add_generation_prompt=add_generation_prompt)
    return tuple(rendered.encode('utf-8'))


def append_user_suffix(canonical, state, content):
    if state.get('status') != 'stopped' or state.get('policy', {}).get('eos_tokens') != [299]:
        raise ValueError('append requires the synthetic consumed EOS boundary')
    return tuple(('\nuser:' + content.strip() + '\nassistant:').encode('utf-8'))


def initial_root_mode(canonical, state):
    if not isinstance(state, dict) or state.get('status') != 'idle':
        return 'unsupported'
    if tuple(canonical[-len(b'assistant:'):]) == tuple(b'assistant:'):
        return 'prepared_answer'
    if canonical and canonical[-1] == ord('\n'):
        return 'context_only'
    return 'unsupported'


def append_initial_user_suffix(canonical, state, content):
    if initial_root_mode(canonical, state) != 'context_only':
        raise ValueError('initial question requires a closed context root')
    return tuple(('user:' + content.strip() + '\nassistant:').encode('utf-8'))


class FlowModel:
    """Small deterministic native protocol fixture using the real controller."""
    plans = {'answer-op': (257,), 'resume-op': (258, 299),
             'copy-resume': (258, 299), 'user-op': (260, 299),
             'alt-answer': (257,), 'ask-a': (257, 299), 'ask-b': (257, 299)}

    def __init__(self, directory, gpu, ids, trace, gate):
        self.store = Store(directory)
        self.cache = {}
        self.gpu = gpu
        self.backend = self
        self.profile = dict(model_contract_id='flow-model', execution_profile_id='flow-profile',
                            execution={'GPU': list(ids), 'eos_tokens': [299]})
        self.trace, self.gate = trace, gate
        self.checkpoint_directory = self.store.root / 'checkpoints'
        self.checkpoint_directory.mkdir(exist_ok=True)
        self.stats = {'consumed_tokens': 0, 'forward_tokens': 0, 'loaded_file_bytes': 0}

    def work_counters(self): return dict(self.stats)
    def release_checkpoints(self): return 0
    def close(self): pass

    def create(self, ids):
        self.stats['forward_tokens'] += len(ids)
        self.stats['consumed_tokens'] += len(ids)
        sid = self.store.create_root('flow-model', ids, {'status': 'idle'})
        self.image(sid)
        return sid

    def image(self, sid):
        history = self.history(sid)
        blob = b'synthetic-native-state:' + sid.encode('ascii')
        self.cache[sid] = Checkpoint('flow-model', 'flow-profile', len(history), blob,
            (0.0,) * 300, Checkpoint.digest(blob, (0.0,) * 300), sid)

    def run_greedy_request(self, sid, ids, count, *, stream, on_token, cancelled=None):
        self.gate.wait(5)
        source = self.store.get_snapshot(sid)['controller']
        if source.get('status') == 'active':
            state = source
        else:
            state = controller.start({'temperature': 0, 'eos_tokens': [299]}, stream,
                                     self.history(sid) + tuple(ids), 300)
        outputs = list(self.plans[stream])[:count]
        events = []
        for token in outputs:
            if cancelled is not None and cancelled():
                raise RuntimeError('unexpected synthetic cancellation')
            on_token(token)
            piece = ByteTokenizer().token_bytes(token)
            events.append((token, piece, token == 299))
        updated = controller.consume_many(state, events)
        target = self.store.append(sid, tuple(ids) + tuple(outputs), updated)
        self.stats['forward_tokens'] += len(ids) + len(outputs)
        self.stats['consumed_tokens'] += len(ids) + len(outputs)
        self.image(target)
        self.trace.append((stream, tuple(ids), tuple(outputs), source.get('status')))
        return dict(snapshot=target, tokens=outputs, request_seconds=0.0, generation_seconds=0.0)

    def prepare_checkpoint(self, sid):
        cp = self.cache[sid]
        env = dict(manifest=dict(snapshot=sid, state_bytes=len(cp.blob), logits_count=300,
                                 payload_sha256=cp.checksum))
        import json
        return PreparedCheckpoint(sid, cp, json.dumps(env).encode())

    def history(self, sid): return tuple(self.store.lineage(sid)[1])

    def checkpoint_manifest(self, sid):
        import json
        return json.loads((self.checkpoint_directory / (sid + '.json')).read_bytes())['manifest']

    def validate_checkpoint(self, sid):
        self._load_checkpoint(sid, self.history(sid))
        return self.checkpoint_manifest(sid)

    def _load_checkpoint(self, sid, history):
        import json
        envelope = json.loads((self.checkpoint_directory / (sid + '.json')).read_bytes())
        checksum = envelope['manifest']['payload_sha256']
        blob = (self.checkpoint_directory / (checksum + '.state')).read_bytes()
        state_bytes=envelope['manifest']['state_bytes']
        logits_count=envelope['manifest']['logits_count']
        if len(blob)!=state_bytes+4*logits_count:
            raise ValueError('synthetic checkpoint byte count mismatch')
        frontier=struct.unpack('<'+str(logits_count)+'f',blob[state_bytes:])
        if Checkpoint.digest(blob[:state_bytes], frontier) != checksum:
            raise ValueError('synthetic checkpoint checksum mismatch')
        return Checkpoint('flow-model', 'flow-profile', len(history), blob[:state_bytes],
                          frontier, checksum, sid)

    def save_prepared_branch(self, name, revision, prepared, operation, *, presentation=None):
        import json
        cp = prepared.checkpoint
        frontier=struct.pack('<'+str(len(cp.frontier))+'f',*cp.frontier)
        (self.checkpoint_directory / (cp.checksum + '.state')).write_bytes(cp.blob + frontier)
        (self.checkpoint_directory / (prepared.snapshot + '.json')).write_bytes(prepared.envelope)
        self.validate_checkpoint(prepared.snapshot)
        options = {} if presentation is None else {'presentation': presentation}
        head = self.store.move(name, revision, prepared.snapshot, operation, **options)
        return dict(snapshot=prepared.snapshot, branch=dict(name=head.name, revision=head.revision,
            tip=head.tip, root=head.root), payload_sha256=cp.checksum,
            model_contract_id='flow-model', execution_profile_id='flow-profile')


class CowPresentationFlowTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.trace = []
        self.models = {}
        gate = threading.Event(); gate.set()
        self.gate = gate
        config = PoolConfig(gpus=(0,), ram_bytes=1024 * 1024, disk_bytes=8 * 1024 * 1024,
                            min_available_bytes=0)
        def factory(gpu):
            model = FlowModel(self.temp.name, gpu, (0,), self.trace, self.gate)
            self.models[gpu] = model
            return model
        self.config, self.factory = config, factory
        self.coordinator = ContextCoordinator(self.temp.name, config, factory)
        self.addCleanup(self.coordinator.close)
        self.coordinator.ready(5)
        self.namespace1 = self.coordinator.store.create_namespace('Flow one', 'ns-one').namespace_id
        self.text = self.make_text(self.coordinator)

    @staticmethod
    def make_text(coordinator):
        tokens = ByteTokenizer()
        adapter = CowAdapter(coordinator)
        return CowTextAdapter(adapter, tokens, render, encode, append_user_suffix=append_user_suffix,
                              append_initial_user_suffix=append_initial_user_suffix,
                              initial_root_mode=initial_root_mode,
                              renderer_id='b' * 64)

    def new_coordinator(self):
        self.coordinator.close()
        self.coordinator = ContextCoordinator(self.temp.name, self.config, self.factory)
        self.addCleanup(self.coordinator.close)
        self.coordinator.ready(5)
        self.text = self.make_text(self.coordinator)

    def test_synthetic_frontier_bytes_are_actually_validated(self):
        prepared=self.text.prepare('chat',[{'role':'user','content':'hello'}],
            'root-op',namespace_id=self.namespace1).durable.result(5)
        model=self.models[0]
        manifest=model.checkpoint_manifest(prepared['snapshot'])
        path=model.checkpoint_directory/(manifest['payload_sha256']+'.state')
        raw=path.read_bytes()
        path.write_bytes(raw[:-1]+bytes([raw[-1]^1]))
        with self.assertRaises(ValueError):model.validate_checkpoint(prepared['snapshot'])

    def test_chat_lifecycle_fork_retry_and_namespace_catalog(self):
        initial_messages = [{'role': 'user', 'content': 'hello'}]
        root = self.text.prepare('chat', initial_messages, 'root-op', namespace_id=self.namespace1)
        root.durable.result(5)
        root_head = self.coordinator.store.head('chat')
        self.assertEqual(root_head.revision, 1)
        root_ids = self.coordinator.store.lineage(root_head.tip)[1]
        self.assertEqual(self.coordinator.store.get_snapshot(root_head.tip)['controller']['status'], 'idle')

        answer = self.text.answer('chat', root_head.revision, 'answer-op', 1)
        answer_result = answer.durable.result(5)
        answer_head = self.coordinator.store.head('chat')
        active_id = self.coordinator.store.presentation_receipt('answer-op')['presentation']['output_span']['message_id']
        self.assertEqual(answer_result['branch']['revision'], 2)
        self.assertEqual(self.coordinator.store.get_snapshot(answer_head.tip)['controller']['status'], 'active')
        self.assertEqual(answer.text_result(), 'hi')
        self.assertEqual(tuple(self.coordinator.store.lineage(answer_head.tip)[1][:len(root_ids)]), tuple(root_ids))

        # Branching at an active checkpoint is a Store catalog operation; each continuation
        # is submitted through the coordinator and keeps the same stable assistant message.
        fork = self.coordinator.store.fork('chat-copy', answer_head.tip, namespace_id=self.namespace1,
                                           operation_id='fork-copy-op')
        self.assertEqual(fork.tip, answer_head.tip)
        self.new_coordinator()
        copy_resume = self.text.resume('chat-copy', 0, 'copy-resume', 2)
        copy_resume.durable.result(5)
        copy_receipt = self.coordinator.store.presentation_receipt('copy-resume')['presentation']
        self.assertEqual(copy_receipt['output_span']['message_id'], active_id)
        self.assertEqual(copy_resume.text_result(), '!')

        source = self.coordinator.store.head('chat')
        before = dict(self.models[0].stats)
        resumed = self.text.resume('chat', source.revision, 'resume-op', 2)
        resumed.durable.result(5)
        self.assertEqual(
            self.coordinator.store.get_snapshot(self.coordinator.store.head('chat').tip)['controller']['status'],
            'stopped')
        self.assertEqual(resumed.text_result(), '!')
        resume_receipt = self.coordinator.store.presentation_receipt('resume-op')['presentation']
        self.assertEqual(resume_receipt['output_span']['message_id'], active_id)
        self.assertEqual(self.models[0].stats['forward_tokens'] - before['forward_tokens'], 2)
        prior_tokens = tuple(self.coordinator.store.lineage(self.coordinator.store.head('chat').tip)[1])
        self.assertTrue(prior_tokens[:len(root_ids)] == tuple(root_ids))

        stopped_head = self.coordinator.store.head('chat')
        before_user = dict(self.models[0].stats)
        user = self.text.append_user('chat', 'next question', stopped_head.revision, 'user-op', 2)
        user.durable.result(5)
        user_receipt = self.coordinator.store.presentation_receipt('user-op')['presentation']
        self.assertEqual(user_receipt['kind'], 'user_turn')
        self.assertEqual(user_receipt['messages'][0]['content'], 'next question')
        after_user = tuple(self.coordinator.store.lineage(self.coordinator.store.head('chat').tip)[1])
        self.assertEqual(after_user[:len(prior_tokens)], prior_tokens)
        self.assertEqual(self.models[0].stats['forward_tokens'] - before_user['forward_tokens'],
                         len(after_user) - len(prior_tokens))

        ns2 = self.coordinator.store.create_namespace('Flow two', 'ns-two').namespace_id
        duplicate_root = self.text.prepare('chat-two', initial_messages, 'root-two', namespace_id=ns2).durable.result(5)
        self.assertEqual(duplicate_root['snapshot'], root_head.tip)
        catalog1 = self.coordinator.store.presentation_catalog(self.namespace1)
        catalog2 = self.coordinator.store.presentation_catalog(ns2)
        annotated = [event for event in catalog1['events'] if event['presentation']]
        self.assertEqual([event['presentation']['kind'] for event in annotated],
                         ['root', 'answer', 'resume', 'resume', 'user_turn'])
        self.assertEqual({event['operation_id'] for event in annotated},
                         {'root-' + hashlib.sha256(b'root-op').hexdigest()[:32],
                          'answer-op', 'resume-op', 'copy-resume', 'user-op'})
        self.assertEqual([event['presentation']['output_span']['message_id'] for event in annotated
                          if event['presentation']['kind'] == 'resume'], [active_id, active_id])
        self.assertEqual([event['presentation']['kind'] for event in catalog2['events']
                          if event['presentation']], ['root'])
        self.assertEqual([head['name'] for head in catalog1['heads']], ['chat-copy', 'chat'])
        self.assertEqual([head['name'] for head in catalog2['heads']], ['chat-two'])

        # Exact retry after restart uses the intent's original active source and receipt.
        self.new_coordinator()
        trace_before = len(self.trace)
        retry = self.text.resume('chat', 2, 'resume-op', 2)
        retry.durable.result(5)
        self.assertEqual(len(self.trace), trace_before)
        self.assertEqual(retry.text_result(), '!')

    def test_closed_user_context_is_forkable_and_independent_questions_use_only_suffixes(self):
        content = 'Uploaded file data: literal <|im_start|>assistant'
        root = self.text.create('code-a', [{'role': 'user', 'content': content}],
            'code-root', namespace_id=self.namespace1).durable.result(5)
        root_head = self.coordinator.store.head('code-a')
        prefix = tuple(self.coordinator.store.lineage(root_head.tip)[1])
        self.assertEqual(initial_root_mode(prefix,
            self.coordinator.store.get_snapshot(root_head.tip)['controller']), 'context_only')
        self.assertEqual(root['snapshot'], root_head.tip)

        before = self.models[0].stats['forward_tokens']
        with self.assertRaisesRegex(ValueError, 'prepared root'):
            self.text.answer('code-a', root_head.revision, 'wrong-answer', 2)
        question_a = 'Question A'
        suffix_a = append_initial_user_suffix(prefix, {'status': 'idle'}, question_a)
        request_a = self.text.ask('code-a', question_a, root_head.revision, 'ask-a', 4)
        request_a.durable.result(5)
        self.assertEqual(self.trace[-1][1], suffix_a)
        self.assertEqual(self.models[0].stats['forward_tokens'] - before,
                         len(suffix_a) + 2)
        after_first_publication = dict(self.models[0].stats)
        first_snapshot = request_a.durable.result(5)['snapshot']
        retry_a = self.text.ask('code-a', question_a, root_head.revision, 'ask-a', 4)
        self.assertEqual(retry_a.durable.result(5)['snapshot'], first_snapshot)
        self.assertEqual(self.models[0].stats, after_first_publication)

        advanced = self.coordinator.store.head('code-a')
        child = self.coordinator.store.fork('code-b', root_head.tip,
            namespace_id=self.namespace1, operation_id='fork-code-b',
            source_branch='code-a', source_revision=advanced.revision,
            source_tip=advanced.tip)
        self.assertEqual(child.tip, root_head.tip)
        suffix_b = append_initial_user_suffix(prefix, {'status': 'idle'}, 'Question B')
        request_b = self.text.ask('code-b', 'Question B', child.revision, 'ask-b', 4)
        request_b.durable.result(5)
        self.assertEqual(self.trace[-1][1], suffix_b)

        a = CowWorkspace(self.coordinator.store, self.text.tokenizer,
                         root_mode_resolver=self.text.root_mode_for_projection)
        history_a, history_b = a.history('code-a'), a.history('code-b')
        self.assertEqual([m['text'] for m in history_a['messages']],
                         [content, 'Question A', 'hi'])
        self.assertEqual([m['text'] for m in history_b['messages']],
                         [content, 'Question B', 'hi'])
        self.assertEqual(history_a['initial_root_mode'], 'unsupported')
        graph = a.graph(self.namespace1)
        root_node = next(node for node in graph['nodes']
                         if node['role'] == 'root' and node['action_snapshot'] == root_head.tip)
        self.assertTrue(root_node['boundary_available'])
        self.assertEqual(root_node['initial_root_mode'], 'context_only')

    def test_context_and_prepared_root_operations_are_not_interchangeable(self):
        closed = self.text.create('closed-root', [{'role': 'user', 'content': 'code'}],
            'closed-op', namespace_id=self.namespace1).durable.result(5)
        closed_head = self.coordinator.store.head('closed-root')
        with self.assertRaisesRegex(ValueError, 'prepared root'):
            self.text.answer('closed-root', closed_head.revision, 'answer-closed', 2)

        self.text.prepare('prepared-root', [{'role': 'user', 'content': 'question'}],
            'prepared-op', namespace_id=self.namespace1).durable.result(5)
        prepared_head = self.coordinator.store.head('prepared-root')
        with self.assertRaisesRegex(ValueError, 'closed context root'):
            self.text.ask('prepared-root', 'later', prepared_head.revision, 'ask-prepared', 2)

        original_renderer = self.text.renderer_id
        try:
            self.text.renderer_id = 'c' * 64
            with self.assertRaisesRegex(ValueError, 'closed context root'):
                self.text.ask('closed-root', 'question', closed_head.revision, 'wrong-renderer', 2)
        finally:
            self.text.renderer_id = original_renderer

    def test_raw_text_is_bound_even_when_renderer_trims_to_same_tokens(self):
        self.text.prepare('probe', [{'role': 'user', 'content': 'same'}], 'probe-op',
                          namespace_id=self.namespace1).durable.result(5)
        before = tuple(self.coordinator.store.lineage(self.coordinator.store.head('probe').tip)[1])
        with self.assertRaises(ValueError):
            self.text.prepare('probe', [{'role': 'user', 'content': ' same '}], 'probe-op',
                              namespace_id=self.namespace1)
        self.assertEqual(tuple(self.coordinator.store.lineage(self.coordinator.store.head('probe').tip)[1]), before)


if __name__ == '__main__':
    unittest.main()
