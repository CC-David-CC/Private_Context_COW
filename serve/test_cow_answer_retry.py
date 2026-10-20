"""CPU CLI/HTTP retry contract tests using the canonical synthetic FlowModel.

These tests intentionally drive the installed command-line entrypoint through
the real local HTTP server.  They do not use a GPU or a pretrained model.
"""
from __future__ import annotations

import json
import hashlib
import copy
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import unittest
import urllib.error
import urllib.request

from context_time_travel import min_p
from context_time_travel.coordinator import ContextCoordinator
from context_time_travel.pools import PoolConfig
from context_time_travel.sampled_controller import PlainSamplingSession, start
from context_time_travel.store import BranchHead, InvalidOperation, _json_bytes
from context_time_travel.presentation_binding import validate_binding
from serve.cow_adapter import CowAdapter
from serve.cow_http import make_server
from serve.cow_text_adapter import CowTextAdapter
from serve.test_cow_presentation_flow import (
    ByteTokenizer,
    FlowModel,
    append_initial_user_suffix,
    append_user_suffix,
    encode,
    initial_root_mode,
    render,
)


REPO = Path(__file__).resolve().parents[1]
CLI = REPO / "tools" / "cow_cli.py"


class RetryTokenizer(ByteTokenizer):
    def token_bytes(self, token):
        if token==261: return b'\xe2'
        return super().token_bytes(token)


class RetryBackendView:
    def __init__(self, model):
        self.model=model
        self.vocab_size=300

    def __getattr__(self, name):
        return getattr(self.model,name)


class RetryFlowModel(FlowModel):
    """Deterministic CPU model that exercises the real controller boundaries."""
    plans={**FlowModel.plans,'original-answer':(257,),
           'greedy-continue':(258,)}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Match the real BranchingModel interface: vocabulary lives on the
        # backend, not as a synthetic top-level model attribute.
        self.backend = RetryBackendView(self)
        self.profile["execution"]["sampling_protocol"] = min_p.SAMPLING_PROTOCOL_V7
        self.selection_calls = []
        self.sample_records=[]

    def sample_legal(self, _lease, mask, temperature, min_p_value, draw53, **filters):
        self.selection_calls.append((temperature, min_p_value, draw53, dict(filters)))
        # A changed top_k selects a different deterministic token.  This is a
        # test selector, not a claim about model sampling quality.
        if filters.get("top_k",0)==2: return 261
        return 258 if filters.get("top_k", 0) else 257

    def run_sampled_request(self, sid, ids, count, *, sampling, stream, resume=False,
                            on_token=None, cancelled=None):
        if cancelled is not None and cancelled():
            raise RuntimeError("unexpected synthetic cancellation")
        history = self.history(sid)
        prompt_history = history + tuple(ids)
        if resume:
            state=copy.deepcopy(self.store.get_snapshot(sid)['controller'])
            if state.get('schema') in (1,2):
                from context_time_travel.sampled_controller import sampler_order
                state=dict(state,schema=3,constraint=state.get('constraint'))
                state['policy']=dict(state['policy'],**sampling)
                state['sampler_order']=sampler_order(sampling)
            session=PlainSamplingSession.restore(state,prompt_history,300,
                RetryTokenizer().token_bytes,lambda token:token==299)
            source_cursor=state['next_draw']
            source_sampler_state=copy.deepcopy(state.get('sampler_state'))
            session.configure_generation(sampling)
        else:
            state = start(sampling, stream, prompt_history, 300, [299])
            session = PlainSamplingSession.restore(
                state, prompt_history, 300, RetryTokenizer().token_bytes,
                lambda token: token == 299)
            source_cursor=state['next_draw']
            source_sampler_state=copy.deepcopy(state.get('sampler_state'))
        mask = bytearray(b"\xff" * ((300 + 7) // 8))
        if 300 % 8:
            mask[-1] &= (1 << (300 % 8)) - 1
        outputs = []
        for _ in range(count):
            if cancelled is not None and cancelled():
                raise RuntimeError("unexpected synthetic cancellation")
            before = session.current_controller
            selector = lambda legal, temp, min_p_value, draw, **filters: self.sample_legal(
                None, legal, temp, min_p_value, draw, **filters)
            if session.current_controller['schema']==4:
                from context_time_travel.sampling_proposal import SelectionProposal
                def proposal_selector(_mask,_temperature,_min_p_value,*,draw_start,
                                      draw_candidates,source_digest,**filters):
                    from context_time_travel.adaptive_p import choose
                    current=session.current_controller
                    result=choose([0.0]*300,bytes(mask),current['sampler_state'],
                        adaptive_target=filters['adaptive_target'],
                        adaptive_decay=filters['adaptive_decay'],draw53=draw_candidates[0])
                    from context_time_travel.adaptive_p import accept
                    next_state=accept(current['sampler_state'],result['proposal'],result['token'])
                    return SelectionProposal(result['token'],draw_start,1,next_state,source_digest)
                token=session.choose(bytes(mask),proposal_selector)
            else:
                token = session.choose(bytes(mask), selector)
            session.consume(before, token, RetryTokenizer().token_bytes(token), token == 299)
            outputs.append(token)
            if on_token is not None:
                on_token(token)
            if session.is_stopped():
                break
        target = self.store.append(sid, tuple(ids) + tuple(outputs), session.current_controller)
        self.stats["forward_tokens"] += len(ids) + len(outputs)
        self.stats["consumed_tokens"] += len(ids) + len(outputs)
        self.image(target)
        self.trace.append((stream, tuple(ids), tuple(outputs), "sampled", dict(sampling)))
        self.sample_records.append(dict(stream=stream,resume=resume,source_cursor=source_cursor,
            source_sampler_state=source_sampler_state,
            source_accepted=(state.get('accepted_in_scope',0) if resume else 0),
            source_stream_key=(state.get('stream_key') if resume else stream),
            final_controller=session.current_controller))
        return dict(snapshot=target, tokens=outputs, request_seconds=0.0,
                    generation_seconds=0.0)

    def sample_selection_proposal(self,mask,temperature,min_p_value,*,draw_start,
                                  draw_candidates,source_digest,**filters):
        raise AssertionError('The request fixture owns proposal selection per active session')

    def save_prepared_branch(self,name,revision,prepared,operation,*,presentation=None,
                             presentation_source_operation=None):
        import struct
        cp=prepared.checkpoint
        frontier=struct.pack('<'+str(len(cp.frontier))+'f',*cp.frontier)
        (self.checkpoint_directory/(cp.checksum+'.state')).write_bytes(cp.blob+frontier)
        (self.checkpoint_directory/(prepared.snapshot+'.json')).write_bytes(prepared.envelope)
        self.validate_checkpoint(prepared.snapshot)
        metadata=dict(schema=1,snapshot=prepared.snapshot,payload_sha256=cp.checksum,
            model_contract_id='flow-model',execution_profile_id='flow-profile')
        options={} if presentation is None else {'presentation':presentation}
        if presentation_source_operation is not None:
            options['presentation_source_operation']=presentation_source_operation
        head=self.store.move(name,revision,prepared.snapshot,operation,
            checkpoint_receipt=metadata,**options)
        return dict(snapshot=prepared.snapshot,branch=dict(name=head.name,revision=head.revision,
            tip=head.tip,root=head.root),payload_sha256=cp.checksum,
            model_contract_id='flow-model',execution_profile_id='flow-profile')


class CowAnswerRetryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.trace = []
        self.gate = threading.Event()
        self.gate.set()
        self.config = PoolConfig(gpus=(0,), ram_bytes=1024 * 1024,
                                 disk_bytes=8 * 1024 * 1024,
                                 min_available_bytes=0)
        self.models = {}
        self.coordinator = self._new_coordinator()
        self.coordinator.ready(5)
        self.namespace = self.coordinator.store.create_namespace("Retry", "retry-ns").namespace_id
        self._start_server()
        self.addCleanup(self._stop_all)

    def _factory(self, gpu):
        model = RetryFlowModel(self.root, gpu, (0,), self.trace, self.gate)
        self.models[gpu] = model
        return model

    def _new_coordinator(self):
        return ContextCoordinator(self.root, self.config, self._factory)

    @staticmethod
    def _new_text(coordinator):
        return CowTextAdapter(CowAdapter(coordinator), RetryTokenizer(), render, encode,
            append_user_suffix=append_user_suffix,
            append_initial_user_suffix=append_initial_user_suffix,
            initial_root_mode=initial_root_mode, renderer_id="b" * 64)

    def _start_server(self):
        self.adapter = CowAdapter(self.coordinator)
        self.text = self._new_text(self.coordinator)
        self.server = make_server(self.adapter, api_key="retry-test-key",
                                  text_adapter=self.text)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base = f"http://127.0.0.1:{self.server.server_port}"

    def _stop_server(self):
        server = getattr(self, "server", None)
        if server is not None:
            server.shutdown()
            server.server_close()
            self.thread.join(5)
            self.server = None

    def _stop_all(self):
        self._stop_server()
        coordinator = getattr(self, "coordinator", None)
        if coordinator is not None:
            coordinator.close()
            self.coordinator = None

    def _restart_server(self):
        self._stop_server()
        self.coordinator.close()
        self.coordinator = self._new_coordinator()
        self.coordinator.ready(5)
        self._start_server()

    def _http(self, method, path, body=None):
        raw = None if body is None else json.dumps(body).encode("utf-8")
        request = urllib.request.Request(self.base + path, data=raw, method=method,
            headers={"Authorization": "Bearer retry-test-key",
                     "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=5) as response:
                return response.status, response.read().decode("utf-8")
        except urllib.error.HTTPError as error:
            with error:
                return error.code, error.read().decode("utf-8")

    def _cli(self, state_dir, *args):
        env = os.environ.copy()
        env["COW_API_KEY"] = "retry-test-key"
        env["COW_CLI_STATE_DIR"] = str(state_dir)
        env.pop("PYTHONPATH", None)
        command = [sys.executable, str(CLI), "--url", self.base,
                   "--state-dir", str(state_dir), *args]
        return subprocess.run(command, cwd=REPO, capture_output=True, env=env,
                              timeout=20, check=False)

    def _completed_original(self, state_dir, *, temperature=0.7):
        source_file = self.root / "context.txt"
        source_file.write_text("code context\n", encoding="utf-8")
        namespace = self._cli(state_dir, "namespace", "--title", "Retry",
                              "--operation-id", "namespace-op")
        self.assertEqual(namespace.returncode, 0, namespace.stderr.decode("utf-8", "replace"))
        namespace_id = json.loads(namespace.stderr.decode("utf-8").split(
            "durable receipt:\n", 1)[1])["namespace_id"]
        loaded = self._cli(state_dir, "load", "--file", str(source_file),
            "--namespace-id", namespace_id, "--branch", "chat",
            "--operation-id", "context-op")
        self.assertEqual(loaded.returncode, 0, loaded.stderr.decode("utf-8", "replace"))
        args=["ask", "--branch", "chat", "--text", "first question",
            "--max-tokens", "2", "--operation-id", "original-answer"]
        if temperature is not None:
            args.extend(["--temperature", str(temperature)])
        asked = self._cli(state_dir, *args)
        self.assertEqual(asked.returncode, 0, asked.stderr.decode("utf-8", "replace"))
        return asked

    def test_retry_command_replays_exact_answer_after_service_restart_without_checkpoint(self):
        with tempfile.TemporaryDirectory() as first_state, tempfile.TemporaryDirectory() as new_state:
            original = self._completed_original(first_state)
            original_head = self.coordinator.store.head("chat")
            original_receipt = self.coordinator.store.presentation_receipt("original-answer")
            original_snapshot = original_receipt["result"]["tip"]
            old_model = self.models[0]
            old_process_stats = dict(old_model.stats)
            # A completed answer is canonical Store data, not a requirement to
            # keep a numeric checkpoint or the first CLI's local journal.
            for path in self.models[0].checkpoint_directory.glob("*.state"):
                path.unlink()
            self._restart_server()
            # The restarted process starts with a fresh counter baseline.  A
            # replay must leave that process unused; old-process counters are
            # recorded separately as lifecycle evidence.
            restarted_model = self.models[0]
            restarted_before = dict(restarted_model.stats)
            replay = self._cli(new_state, "retry", "--from-operation", "original-answer",
                               "--temperature", "0.7")
            self.assertEqual(replay.returncode, 0, replay.stderr.decode("utf-8", "replace"))
            self.assertEqual(replay.stdout, original.stdout)
            replay_receipt = json.loads(replay.stderr.decode("utf-8").split(
                "durable receipt:\n", 1)[1])
            original_cli_receipt = json.loads(original.stderr.decode("utf-8").split(
                "durable receipt:\n", 1)[1])
            self.assertEqual(replay_receipt, original_cli_receipt)
            self.assertEqual(self.coordinator.store.head("chat"), original_head)
            self.assertEqual(self.coordinator.store.presentation_receipt("original-answer"),
                             original_receipt)
            self.assertEqual(restarted_model.stats, restarted_before)
            self.assertEqual(old_model.stats, old_process_stats)
            status, body = self._http("POST", "/cow/text/retry", {
                "from_operation": "original-answer", "branch": "chat",
                "operation_id": "original-answer", "revision": original_head.revision,
                "stream": False})
            self.assertEqual(status, 200, body)
            saved = json.loads(body)
            self.assertEqual(saved["answer"]["snapshot"], original_snapshot)
            self.assertTrue(saved["replayed"])
            self.assertEqual(saved["receipt"], original_cli_receipt)
            self.assertEqual(saved["receipt"]["snapshot"], original_snapshot)

    def test_explicit_greedy_temperature_replays_and_other_branch_is_rejected(self):
        with tempfile.TemporaryDirectory() as state_dir:
            original=self._completed_original(state_dir,temperature=None)
            head=self.coordinator.store.head("chat")
            before=dict(self.models[0].stats)
            replay=self._cli(state_dir,"retry","--from-operation","original-answer",
                "--temperature","0")
            self.assertEqual(replay.returncode,0,replay.stderr.decode("utf-8","replace"))
            self.assertEqual(replay.stdout,original.stdout)
            self.assertEqual(self.coordinator.store.head("chat"),head)
            self.assertEqual(self.models[0].stats,before)
            wrong_branch=self._cli(state_dir,"retry","--from-operation","original-answer",
                "--branch","another-branch","--temperature","0")
            self.assertNotEqual(wrong_branch.returncode,0)
            self.assertIn(b"exact answer replay is bound to its original branch",wrong_branch.stderr)
            self.assertEqual(self.coordinator.store.head("chat"),head)
            self.assertEqual(self.models[0].stats,before)

    def test_changed_settings_rerun_from_saved_pre_answer_source_with_new_id(self):
        with tempfile.TemporaryDirectory() as first_state, tempfile.TemporaryDirectory() as new_state:
            self._completed_original(first_state)
            before_calls = len(self.trace)
            original = self.coordinator.store.presentation_receipt("original-answer")
            source_snapshot = original["presentation"]["source"]["snapshot"]
            original_answer = original["result"]["tip"]
            current = self.coordinator.store.head("chat")
            self.assertEqual(current.tip, original_answer)
            retried = self._cli(new_state, "retry", "--from-operation", "original-answer",
                "--branch", "chat", "--operation-id", "retry-top-k-one", "--top-k", "1")
            self.assertEqual(retried.returncode, 0, retried.stderr.decode("utf-8", "replace"))
            self.assertEqual(len(self.trace), before_calls + 1)
            stream, ids, outputs, kind, sampling = self.trace[-1]
            self.assertEqual(stream, "original-answer")
            retry_prompt = tuple(self.coordinator.store.lineage(original_answer)[1])
            input_span = original["presentation"]["input_span"]
            expected_ids = retry_prompt[input_span["start"]:
                                        input_span["start"] + input_span["count"]]
            self.assertEqual(ids, expected_ids)
            self.assertEqual(sampling["top_k"], 1)
            self.assertEqual(len(outputs), 2)
            self.assertNotEqual(outputs[0], 257)
            new_receipt = self.coordinator.store.presentation_receipt("retry-top-k-one")
            self.assertEqual(new_receipt["presentation"]["source"]["snapshot"], source_snapshot)
            self.assertEqual(self.coordinator.store.get_snapshot(original_answer)["parent"], source_snapshot)
            self.assertEqual(self.coordinator.store.presentation_receipt("original-answer"), original)
            self.assertEqual(new_receipt["result"]["tip"], self.coordinator.store.head("chat").tip)

    def test_changed_retry_of_retry_keeps_original_generation_stream(self):
        with tempfile.TemporaryDirectory() as state_dir:
            self._completed_original(state_dir)
            original=self.coordinator.store.presentation_receipt("original-answer")
            source_snapshot=original["presentation"]["source"]["snapshot"]
            first=self._cli(state_dir,"retry","--from-operation","original-answer",
                "--operation-id","retry-first","--top-k","1")
            self.assertEqual(first.returncode,0,first.stderr.decode("utf-8","replace"))
            second=self._cli(state_dir,"retry","--from-operation","retry-first",
                "--operation-id","retry-second","--top-p","0.5")
            self.assertEqual(second.returncode,0,second.stderr.decode("utf-8","replace"))
            self.assertEqual(self.trace[-1][0],"original-answer")
            self.assertEqual(self.trace[-1][1],self.trace[1][1])
            first_receipt=self.coordinator.store.presentation_receipt("retry-first")
            second_receipt=self.coordinator.store.presentation_receipt("retry-second")
            self.assertEqual(first_receipt["presentation"]["source"]["snapshot"],source_snapshot)
            self.assertEqual(second_receipt["presentation"]["source"]["snapshot"],source_snapshot)
            self.assertEqual(second_receipt["presentation_source_operation"],"retry-first")
            self.assertEqual(self.coordinator.store.head("chat").tip,second_receipt["result"]["tip"])
            self.assertEqual(self.coordinator.store.presentation_receipt("original-answer"),original)

            # The chain walker must share one canonical refs snapshot across
            # all three receipts. Ordinary annotations must avoid this catalog
            # read entirely once their receipt and branch are already loaded.
            store=self.coordinator.store
            original_read_refs=store._read_refs
            reads=[]
            def counted_read_refs():
                reads.append(1)
                return original_read_refs()
            store._read_refs=counted_read_refs
            try:
                self.assertEqual(store._presentation_stream_operation('retry-second'),
                                 'original-answer')
                self.assertEqual(len(reads),1)
                refs=original_read_refs()
                receipt=refs['receipts']['original-answer']
                branch=store._read_branch('chat')
                reads.clear()
                store._validate_stored_presentation('chat',branch,'original-answer',receipt)
                self.assertEqual(reads,[])
                reads.clear()
                store.presentation_receipt('retry-second')
                self.assertEqual(len(reads),1)
            finally:
                store._read_refs=original_read_refs

    def test_greedy_continue_exact_and_changed_retry_preserve_active_source(self):
        with tempfile.TemporaryDirectory() as state_dir:
            self._completed_original(state_dir,temperature=0)
            active_before=self.coordinator.store.head('chat')
            active_controller=self.coordinator.store.get_snapshot(active_before.tip)['controller']
            self.assertEqual(active_controller['status'],'active')
            self.assertEqual(active_controller['schema'],1)
            continued=self._cli(state_dir,'continue','--branch','chat','--max-tokens','1',
                '--operation-id','greedy-continue','--temperature','0')
            self.assertEqual(continued.returncode,0,continued.stderr.decode('utf-8','replace'))
            descriptor=self.coordinator.committed_generation('greedy-continue')['generation']
            self.assertEqual(descriptor['kind'],'resume')
            self.assertTrue(descriptor['resume'])
            self.assertIsNone(descriptor['sampling'])
            committed_head=self.coordinator.store.head('chat')
            before=dict(self.models[0].stats)
            replay=self._cli(state_dir,'retry','--from-operation','greedy-continue',
                '--temperature','0')
            self.assertEqual(replay.returncode,0,replay.stderr.decode('utf-8','replace'))
            self.assertEqual(replay.stdout,continued.stdout)
            self.assertEqual(self.coordinator.store.head('chat'),committed_head)
            self.assertEqual(self.models[0].stats,before)

            retry=self._cli(state_dir,'retry','--from-operation','greedy-continue',
                '--operation-id','greedy-continue-sampled','--temperature','0.7')
            self.assertEqual(retry.returncode,0,retry.stderr.decode('utf-8','replace'))
            record=self.models[0].sample_records[-1]
            self.assertTrue(record['resume'])
            self.assertEqual(record['source_accepted'],1)
            self.assertEqual(record['source_stream_key'],active_controller['stream_key'])
            self.assertEqual(record['final_controller']['accepted_in_scope'],2)
            self.assertEqual(record['final_controller']['stream_key'],active_controller['stream_key'])

    def test_legacy_schema_two_answer_remains_exactly_replayable(self):
        with tempfile.TemporaryDirectory() as first_state, tempfile.TemporaryDirectory() as new_state:
            original=self._completed_original(first_state)
            intent_path=self.coordinator._intents/"original-answer.json"
            envelope=json.loads(intent_path.read_bytes())
            intent=envelope["intent"]
            intent.pop("generation")
            intent["schema"]=2
            envelope["checksum"]=hashlib.sha256(_json_bytes(intent)).hexdigest()
            intent_path.write_bytes(_json_bytes(envelope))
            head=self.coordinator.store.head("chat")
            before=dict(self.models[0].stats)
            replay=self._cli(new_state,"retry","--from-operation","original-answer")
            self.assertEqual(replay.returncode,0,replay.stderr.decode("utf-8","replace"))
            self.assertEqual(replay.stdout,original.stdout)
            self.assertEqual(self.coordinator.store.head("chat"),head)
            self.assertEqual(self.models[0].stats,before)

    def test_replay_preserves_unflushed_utf8_prefix_at_max_token_boundary(self):
        with tempfile.TemporaryDirectory() as state_dir:
            self._completed_original(state_dir)
            root=self.coordinator.store.presentation_receipt("original-answer")[
                "presentation"]["source"]["snapshot"]
            fork=self._cli(state_dir,"fork","--source-branch","chat","--snapshot",root,
                "--name","chat-pending","--operation-id","pending-fork")
            self.assertEqual(fork.returncode,0,fork.stderr.decode("utf-8","replace"))
            original=self._cli(state_dir,"ask","--branch","chat-pending","--text","partial bytes",
                "--max-tokens","2","--operation-id","pending-answer","--temperature","0.7",
                "--top-k","2")
            self.assertEqual(original.returncode,0,original.stderr.decode("utf-8","replace"))
            original_receipt=self.coordinator.store.presentation_receipt("pending-answer")
            answer_snapshot=original_receipt["result"]["tip"]
            controller=self.coordinator.store.get_snapshot(answer_snapshot)["controller"]
            self.assertEqual(controller["output_pending_hex"],"e2")
            head=self.coordinator.store.head("chat-pending")
            before=dict(self.models[0].stats)
            replay=self._cli(state_dir,"retry","--from-operation","pending-answer","--top-k","2")
            self.assertEqual(replay.returncode,0,replay.stderr.decode("utf-8","replace"))
            self.assertEqual(replay.stdout,original.stdout)
            self.assertEqual(self.coordinator.store.head("chat-pending"),head)
            self.assertEqual(self.models[0].stats,before)

    def test_changed_retry_of_continue_preserves_active_cursor_and_adaptive_feedback(self):
        with tempfile.TemporaryDirectory() as state_dir:
            self._completed_original(state_dir)
            # Start a separate adaptive scope so the active source has both a
            # nonzero draw cursor and mutable EMA feedback.
            head=self.coordinator.store.head("chat")
            prepared=self._cli(state_dir,"rewind","--branch","chat","--snapshot",
                self.coordinator.store.presentation_receipt("original-answer")["presentation"]["source"]["snapshot"],
                "--operation-id","rewind-for-adaptive")
            self.assertEqual(prepared.returncode,0,prepared.stderr.decode("utf-8","replace"))
            current=self.coordinator.store.head("chat")
            first=self._cli(state_dir,"ask","--branch","chat","--text","adaptive question",
                "--max-tokens","2","--operation-id","adaptive-answer",
                "--temperature","0.7","--adaptive-target","0.5","--adaptive-decay","0.9")
            self.assertEqual(first.returncode,0,first.stderr.decode("utf-8","replace"))
            active=self.coordinator.store.head("chat")
            source_controller=self.coordinator.store.get_snapshot(active.tip)["controller"]
            self.assertEqual(source_controller["status"],"active")
            self.assertGreater(source_controller["next_draw"],0)
            self.assertIsInstance(source_controller.get("sampler_state"),dict)
            continued=self._cli(state_dir,"continue","--branch","chat","--max-tokens","1",
                "--operation-id","adaptive-resume","--temperature","0.7",
                "--adaptive-target","0.5","--adaptive-decay","0.9")
            self.assertEqual(continued.returncode,0,continued.stderr.decode("utf-8","replace"))
            original_resume=self.coordinator.store.presentation_receipt("adaptive-resume")
            resume_binding=copy.deepcopy(original_resume['presentation'])
            resume_binding['generation_settings']={'temperature':True}
            source_record=resume_binding['source']
            source_head=BranchHead('chat',source_record['revision'],source_record['snapshot'],
                original_resume['result']['root'])
            with self.assertRaises(InvalidOperation):
                validate_binding(self.coordinator.store,source_head,
                    original_resume['result']['tip'],resume_binding,
                    namespace_id=self.namespace,operation_id='adaptive-resume')
            source=self.coordinator.store.get_snapshot(active.tip)["controller"]
            before_records=len(self.models[0].sample_records)
            retry=self._cli(state_dir,"retry","--from-operation","adaptive-resume",
                "--operation-id","adaptive-resume-changed","--adaptive-target","0.6")
            self.assertEqual(retry.returncode,0,retry.stderr.decode("utf-8","replace"))
            record=self.models[0].sample_records[-1]
            self.assertTrue(record["resume"])
            self.assertEqual(record["source_cursor"],source["next_draw"])
            self.assertEqual(record["source_sampler_state"],source["sampler_state"])
            self.assertGreater(record["final_controller"]["next_draw"],source["next_draw"])
            self.assertEqual(record["final_controller"]["stream_key"],source["stream_key"])
            self.assertEqual(self.coordinator.store.presentation_receipt("adaptive-resume"),original_resume)
            self.assertEqual(self.coordinator.store.presentation_receipt(
                "adaptive-resume-changed")["presentation_source_operation"],"adaptive-resume")
            second_retry=self._cli(state_dir,"retry","--from-operation","adaptive-resume-changed",
                "--operation-id","adaptive-resume-retry-2","--adaptive-decay","0.8")
            self.assertEqual(second_retry.returncode,0,second_retry.stderr.decode("utf-8","replace"))
            chained=self.coordinator.committed_generation("adaptive-resume-retry-2")
            self.assertEqual(chained['source_head'].tip,active.tip)
            self.assertEqual(chained['generation']['stream_operation_id'],
                self.coordinator.committed_generation("adaptive-resume")['generation']['stream_operation_id'])
            self.assertEqual(self.coordinator.store.presentation_receipt(
                "adaptive-resume-retry-2")["presentation_source_operation"],"adaptive-resume-changed")

    def test_stale_retry_cannot_move_branch_but_explicit_source_fork_can(self):
        with tempfile.TemporaryDirectory() as state_dir:
            self._completed_original(state_dir)
            original = self.coordinator.store.presentation_receipt("original-answer")
            source_snapshot = original["presentation"]["source"]["snapshot"]
            answer_head = self.coordinator.store.head("chat")
            moved = self._cli(state_dir, "rewind", "--branch", "chat", "--snapshot",
                source_snapshot, "--operation-id", "other-client-rewind")
            self.assertEqual(moved.returncode, 0, moved.stderr.decode("utf-8", "replace"))
            stale_head = self.coordinator.store.head("chat")
            before = dict(self.models[0].stats)
            status,body=self._http("POST","/cow/text/retry",{
                "from_operation":"original-answer","branch":"chat",
                "operation_id":"stale-retry","revision":answer_head.revision,
                "settings":{"top_k":1},"stream":False})
            self.assertEqual(status,409,body)
            self.assertIn("Requested branch revision changed",body)
            self.assertEqual(self.coordinator.store.head("chat"), stale_head)
            self.assertEqual(self.models[0].stats, before)
            forked = self._cli(state_dir, "fork", "--source-branch", "chat",
                "--snapshot", source_snapshot, "--name", "chat-alt",
                "--operation-id", "explicit-source-fork")
            self.assertEqual(forked.returncode, 0, forked.stderr.decode("utf-8", "replace"))
            retry = self._cli(state_dir, "retry", "--from-operation", "original-answer",
                "--branch", "chat-alt", "--operation-id", "alt-retry", "--top-k", "1")
            self.assertEqual(retry.returncode, 0, retry.stderr.decode("utf-8", "replace"))
            self.assertEqual(self.coordinator.store.head("chat"), stale_head)
            self.assertEqual(self.coordinator.store.head("chat-alt").revision, 1)
            self.assertEqual(self.coordinator.store.presentation_receipt("original-answer"), original)

    def test_retry_route_rejects_uncommitted_or_corrupt_original_before_work(self):
        with tempfile.TemporaryDirectory() as state_dir:
            self._completed_original(state_dir)
            before = dict(self.models[0].stats)
            status, body = self._http("POST", "/cow/text/retry", {
                "from_operation": "unknown-operation", "branch": "chat",
                "operation_id": "retry-missing", "revision": 2,
                "settings": {"top_k": 1}, "stream": False})
            self.assertIn(status, (404, 409), body)
            self.assertNotIn("event\":\"complete", body)
            self.assertEqual(self.models[0].stats, before)

    def test_retry_rejects_corrupt_original_presentation_blob_before_work(self):
        with tempfile.TemporaryDirectory() as state_dir:
            self._completed_original(state_dir)
            receipt = self.coordinator.store.presentation_receipt("original-answer")
            digest = receipt["digest"]
            blob = self.coordinator.store.presentation_bindings / (digest + ".json")
            original_bytes = blob.read_bytes()
            self.addCleanup(lambda: blob.write_bytes(original_bytes) if blob.exists() else None)
            blob.write_bytes(original_bytes + b"corruption")
            before = dict(self.models[0].stats)
            status, body = self._http("POST", "/cow/text/retry", {
                "from_operation": "original-answer", "branch": "chat",
                "operation_id": "retry-corrupt", "revision": 2,
                "settings": {"top_k": 1}, "stream": False})
            self.assertIn(status, (409, 500), body)
            self.assertNotIn("event\":\"complete", body)
            self.assertEqual(self.models[0].stats, before)


    def _commit_changed_retry_without_client_ack(self, state_dir):
        from tools.cow_cli import RequestJournal

        self._completed_original(state_dir)
        old_head = self.coordinator.store.head("chat")
        operation = "changed-retry-lost-ack"
        body = dict(from_operation="original-answer", branch="chat",
                    operation_id=operation, revision=old_head.revision,
                    settings={"top_k": 1}, stream=True)
        journal = RequestJournal(state_dir)
        journal.begin(operation, self.base, "/cow/text/retry",
                      json.dumps(body).encode("utf-8"))
        status, raw = self._http("POST", "/cow/text/retry", body)
        self.assertEqual(status, 200, raw)
        completions = [value for value in map(json.loads, raw.splitlines())
                       if value.get("event") == "complete"]
        self.assertEqual(len(completions), 1, raw)
        self.assertEqual(journal.get(operation)["state"], "pending")
        head = self.coordinator.store.head("chat")
        self.assertEqual(head.revision, old_head.revision + 1)
        return body, completions[0], head

    def test_changed_retry_transport_replay_returns_committed_receipt_without_work(self):
        with tempfile.TemporaryDirectory() as state_dir:
            body, first, head = self._commit_changed_retry_without_client_ack(state_dir)
            before = dict(self.models[0].stats)
            status, raw = self._http("POST", "/cow/text/retry", body)
            self.assertEqual(status, 200, raw)
            completions = [value for value in map(json.loads, raw.splitlines())
                           if value.get("event") == "complete"]
            self.assertEqual(len(completions), 1, raw)
            self.assertEqual(completions[0]["receipt"], first["receipt"])
            self.assertEqual(completions[0]["answer"], first["answer"])
            self.assertEqual(self.coordinator.store.head("chat"), head)
            self.assertEqual(self.models[0].stats, before)

    def test_changed_retry_lost_ack_cli_recovers_original_operation_after_restart(self):
        with tempfile.TemporaryDirectory() as state_dir:
            body, first, head = self._commit_changed_retry_without_client_ack(state_dir)
            self._restart_server()
            before = dict(self.models[0].stats)
            # The service's ephemeral URL changes on restart. Retain the exact
            # original operation/body while rebinding only this test's client
            # endpoint through the public journal constructor.
            with tempfile.TemporaryDirectory() as resumed_state:
                from tools.cow_cli import RequestJournal
                journal = RequestJournal(resumed_state)
                journal.begin(body["operation_id"], self.base, "/cow/text/retry",
                              json.dumps(body).encode("utf-8"))
                recovered = self._cli(resumed_state, "retry-request",
                                      "--operation-id", body["operation_id"])
                self.assertEqual(recovered.returncode, 0,
                                 recovered.stderr.decode("utf-8", "replace"))
                receipt = json.loads(recovered.stderr.decode("utf-8").split(
                    "durable receipt:\n", 1)[1])
                self.assertEqual(receipt, first["receipt"])
                self.assertEqual(journal.get(body["operation_id"])["state"], "complete")
            self.assertEqual(self.coordinator.store.head("chat"), head)
            self.assertEqual(self.models[0].stats, before)


if __name__ == "__main__":
    unittest.main()
