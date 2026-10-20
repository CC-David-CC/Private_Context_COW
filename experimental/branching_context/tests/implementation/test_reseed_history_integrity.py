"""Reseed must validate controller progress against the canonical token history."""
import copy
import json
import struct
import unittest
from unittest.mock import patch

import serve.test_cow_presentation_flow as flow
from context_time_travel.real import BranchingModel
from context_time_travel.store import IntegrityError, InvalidOperation


class ReseedHistoryIntegrityTests(unittest.TestCase):
    def test_coordinator_reseed_rejects_checksummed_controller_with_wrong_seen_counts(self):
        fixture = flow.CowPresentationFlowTests("runTest")
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        model = fixture.models[0]
        model.backend.vocab_size = 300
        model.piece = flow.ByteTokenizer().token_bytes
        model.is_eog = lambda token: token == 299
        model.backend.piece = model.piece
        model.backend.is_eog = model.is_eog
        model.controller = lambda snapshot: model.store.get_snapshot(snapshot)["controller"]
        model._controller_transition = BranchingModel._controller_transition.__get__(
            model, type(model))

        # Use the production transition implementation while keeping the
        # coordinator, Store, branch CAS, and publication path in this real
        # CPU control-plane fixture.
        set_stream_patch = patch.object(
            flow.FlowModel, "set_random_stream", BranchingModel.set_random_stream, create=True)
        set_stream_patch.start()
        self.addCleanup(set_stream_patch.stop)

        fixture.text.prepare("chat", [{"role": "user", "content": "hello"}],
                             "integrity-root", namespace_id=fixture.namespace1).durable.result(5)
        fixture.models[0].plans["integrity-answer"] = (257,)
        fixture.text.answer("chat", 1, "integrity-answer", 1).durable.result(5)

        store = fixture.coordinator.store
        source_head = store.head("chat")
        source = source_head.tip
        original_controller = store.get_snapshot(source)["controller"]
        original_history = tuple(store.lineage(source)[1])
        store.fork("valid-controller", source)
        valid_request = fixture.coordinator.reseed(
            "valid-controller", 5, "valid-reseed-control", expected_revision=0)
        valid_receipt = valid_request.durable.result(5)
        valid_head = store.head("valid-controller")
        valid_controller = store.get_snapshot(valid_head.tip)["controller"]
        expected_controller = copy.deepcopy(original_controller)
        from context_time_travel.seed_controls import seed_stream
        import hashlib
        expected_controller["stream_key"] = hashlib.sha256(seed_stream(5).encode()).hexdigest()
        expected_controller["next_draw"] = 0
        self.assertEqual(valid_controller, expected_controller)
        self.assertEqual(tuple(store.lineage(valid_head.tip)[1]), original_history)
        self.assertEqual(valid_receipt["branch"]["revision"], 1)

        bad_controller = copy.deepcopy(store.get_snapshot(source)["controller"])
        self.assertEqual(bad_controller["status"], "active")
        self.assertEqual(bad_controller["seen"].get("257"), 1)
        bad_controller["seen"]["257"] = 2

        # Store.append writes a fresh canonical JSON object and digest. The
        # corruption is semantic (controller/history disagreement), not a
        # damaged object hash or malformed controller shape.
        bad_source = store.append(source, (), bad_controller)
        self.assertEqual(store.get_snapshot(bad_source)["controller"], bad_controller)
        model.image(bad_source)
        prepared = model.prepare_checkpoint(bad_source)
        checkpoint = prepared.checkpoint
        frontier = struct.pack("<" + str(len(checkpoint.frontier)) + "f", *checkpoint.frontier)
        (model.checkpoint_directory / (checkpoint.checksum + ".state")).write_bytes(
            checkpoint.blob + frontier)
        (model.checkpoint_directory / (bad_source + ".json")).write_bytes(prepared.envelope)
        model.validate_checkpoint(bad_source)
        store.fork("bad-controller", bad_source)

        before_head = store.head("bad-controller")
        before_work = dict(model.stats)
        operation = "reject-bad-controller-reseed"
        request = fixture.coordinator.reseed("bad-controller", 7, operation,
                                             expected_revision=before_head.revision)

        with self.assertRaisesRegex(
                IntegrityError,
                "Reseed controller repetition state differs from canonical history"):
            request.durable.result(5)

        self.assertEqual(store.head("bad-controller"), before_head)
        self.assertNotIn(operation, store._read_refs()["receipts"])
        self.assertEqual(model.stats, before_work)

    def test_grammar_reseed_replays_canonical_parser_with_server_registry(self):
        import serve.test_cow_grammar_canonical_flow as grammar_flow

        fixture = grammar_flow.CowCanonicalGrammarFlowTests("runTest")
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        root_result = fixture.coordinator.create(
            "GrammarReseed", tuple(fixture.tokenizer.encode("root", parse_special=False)),
            "grammar-reseed-root").durable.result(5)
        root_snapshot = root_result["snapshot"]
        model = fixture.coordinator._workers[0]["model"]
        answer = model.run_state_request(
            root_snapshot, (), 1, engine=fixture.registry.engine, state="READ",
            stream="grammar-reseed-answer", text_stops=("AB",))
        self.assertEqual(answer["tokens"], [ord("A")])
        root_head = fixture.coordinator.store.head("GrammarReseed")
        answer_receipt = model.save_branch(
            "GrammarReseed", root_head.revision, answer["snapshot"], "grammar-reseed-answer")
        source = answer_receipt["snapshot"]
        source_state = model.controller(source)
        source_history = model.history(source)
        self.assertEqual(source_state["constraint"]["agent_state"], "READ")
        self.assertEqual(source_state["accepted_in_scope"], 1)
        self.assertEqual(source_state["schema"], 5)
        self.assertEqual(source_state["text_stop"]["pending_hex"], "41")

        # The CowAdapter carries the startup-owned registry into the queued
        # reseed operation; parser progress must remain exactly at the A prefix.
        valid_work = {key: fixture.backends[0].stats[key]
                      for key in ("forward_tokens", "consumed_tokens", "sampling_calls")
                      if key in fixture.backends[0].stats}
        valid = fixture.token_adapter.reseed(
            "GrammarReseed", 17, answer_receipt["branch"]["revision"],
            "grammar-reseed-control").durable.result(5)
        valid_state = model.controller(valid["snapshot"])
        self.assertEqual(valid_state["constraint"], source_state["constraint"])
        self.assertEqual(valid_state["grammar_state"], source_state["grammar_state"])
        self.assertEqual(valid_state["accepted_in_scope"], source_state["accepted_in_scope"])
        self.assertEqual(valid_state["text_stop"], source_state["text_stop"])
        self.assertEqual(valid_state["next_draw"], 0)
        self.assertEqual(model.history(valid["snapshot"]), source_history)
        self.assertEqual({key: fixture.backends[0].stats[key] for key in valid_work}, valid_work)

        fixture.coordinator.fork("MissingGrammarRegistry", source)
        missing_head = fixture.coordinator.store.head("MissingGrammarRegistry")
        missing_operation = "reject-missing-grammar-registry-reseed"
        missing_work = {key: fixture.backends[0].stats[key]
                        for key in ("forward_tokens", "consumed_tokens", "sampling_calls")
                        if key in fixture.backends[0].stats}
        missing = fixture.coordinator.reseed(
            "MissingGrammarRegistry", 18, missing_operation,
            expected_revision=missing_head.revision)
        with self.assertRaisesRegex(InvalidOperation, "Grammar reseed requires its server-owned engine"):
            missing.durable.result(5)
        self.assertEqual(fixture.coordinator.store.head("MissingGrammarRegistry"), missing_head)
        self.assertNotIn(missing_operation, fixture.coordinator.store._read_refs()["receipts"])
        self.assertEqual({key: fixture.backends[0].stats[key] for key in missing_work}, missing_work)

        # READ has consumed A; changing the valid host-selected state to EDIT
        # remains structurally valid and is written as a correctly hashed COW
        # snapshot, but canonical replay must reject it (EDIT expects X).
        def rejected_branch(branch, operation, mutate, error):
            corrupt = copy.deepcopy(source_state)
            mutate(corrupt)
            bad_snapshot = model._controller_transition(source, corrupt)
            self.assertEqual(fixture.coordinator.store.get_snapshot(bad_snapshot)["controller"], corrupt)
            fixture.coordinator.fork(branch, bad_snapshot)
            setup = model.save_branch(branch, 0, bad_snapshot, "setup-" + operation)
            head = fixture.coordinator.store.head(branch)
            self.assertEqual(setup["branch"]["revision"], head.revision)
            before_work = {key: fixture.backends[0].stats[key]
                           for key in ("forward_tokens", "consumed_tokens", "sampling_calls")
                           if key in fixture.backends[0].stats}
            request = fixture.coordinator.reseed(
                branch, 19, operation, expected_revision=head.revision,
                grammar_registry=fixture.registry)
            with self.assertRaisesRegex(IntegrityError, error):
                request.durable.result(5)
            self.assertEqual(fixture.coordinator.store.head(branch), head)
            self.assertNotIn(operation, fixture.coordinator.store._read_refs()["receipts"])
            self.assertEqual({key: fixture.backends[0].stats[key] for key in before_work}, before_work)

        rejected_branch("BadGrammarReseed", "reject-mismatched-grammar-reseed",
            lambda state: state["constraint"].__setitem__("agent_state", "EDIT"),
            "Canonical history violates the selected grammar")
        rejected_branch("BadStopReseed", "reject-mismatched-stop-reseed",
            lambda state: state["text_stop"].__setitem__("pending_hex", ""),
            "Text-stop matcher state does not match accepted grammar history")


if __name__ == "__main__":
    unittest.main()
