"""Atomic COW branch publication of optional presentation annotations."""

import hashlib
import tempfile
import unittest

from context_time_travel import Conflict, IntegrityError, InvalidOperation, NotFound, Store
from context_time_travel.store import _hash, _json_bytes


RENDERER = hashlib.sha256(b"renderer-v1").hexdigest()


def root_binding(namespace, tokens, *, user_id="1" * 32):
    return {
        "schema": 1,
        "namespace_id": namespace,
        "kind": "root",
        "messages": [{"id": user_id, "role": "user", "content": "literal {{ prompt }}"}],
        "source": {"snapshot": None, "revision": 0, "token_count": 0},
        "input_span": {"start": 0, "count": len(tokens),
                       "token_ids_sha256": _hash(_json_bytes(tokens))},
        "output_span": None,
        "renderer_id": RENDERER,
        "turn_complete": False,
    }


def turn_binding(namespace, source, revision, source_tokens, prompt, generated,
                 operation, *, assistant_id="a" * 32, content="next turn"):
    return {
        "schema": 1,
        "namespace_id": namespace,
        "kind": "user_turn",
        "messages": [{"id": "2" * 32, "role": "user", "content": content}],
        "source": {"snapshot": source, "revision": revision,
                   "token_count": len(source_tokens)},
        "input_span": {"start": len(source_tokens), "count": len(prompt),
                       "token_ids_sha256": _hash(_json_bytes(prompt))},
        "output_span": {"start": len(source_tokens) + len(prompt),
                        "count": len(generated), "message_id": assistant_id},
        "renderer_id": RENDERER,
        "turn_complete": False,
    }


class StorePresentationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="store-presentation-")
        self.addCleanup(self.tmp.cleanup)
        self.store = Store(self.tmp.name)
        self.ns = self.store.create_namespace("Chat", "create-chat-1").namespace_id

    def make_namespace_branch(self, name="chat", *, model="same", tokens=(1, 2), controller=None,
                              operation="fork-1", presentation=None):
        root = self.store.create_root(model, list(tokens), controller or {"status": "idle"})
        head = self.store.fork(name, root, namespace_id=self.ns, operation_id=operation,
                               presentation=presentation)
        return root, head

    def test_identical_roots_keep_two_namespace_groups(self):
        second_ns = self.store.create_namespace("Chat", "create-chat-2").namespace_id
        tokens = [4, 5, 6]
        root = self.store.create_root("same-model", tokens, {"status": "idle"})
        same_root = self.store.create_root("same-model", tokens, {"status": "idle"})
        first = self.store.fork("one", root, namespace_id=self.ns, operation_id="fork-one",
                                presentation=root_binding(self.ns, tokens))
        second = self.store.fork("two", same_root, namespace_id=second_ns, operation_id="fork-two",
                                 presentation=root_binding(second_ns, tokens, user_id="3" * 32))
        self.assertEqual(root, same_root)
        self.assertEqual(self.store.branch_namespace(first.name), self.ns)
        self.assertEqual(self.store.branch_namespace(second.name), second_ns)
        self.assertEqual(self.store.presentation_receipt("fork-one")["presentation"]["namespace_id"], self.ns)
        self.assertEqual(self.store.presentation_receipt("fork-two")["presentation"]["namespace_id"], second_ns)

    def test_namespace_fork_retry_returns_original_receipt_after_later_move(self):
        tokens = [1, 2]
        root = self.store.create_root("model", tokens, {"status": "idle"})
        binding = root_binding(self.ns, tokens)
        original = self.store.fork("branch", root, namespace_id=self.ns,
                                   operation_id="fork-retry", presentation=binding)
        target = self.store.append(root, [3], {"status": "stopped"})
        self.store.move("branch", 0, target, "advance")
        retry = self.store.fork("branch", root, namespace_id=self.ns,
                                operation_id="fork-retry", presentation=binding)
        self.assertEqual(retry, original)
        self.assertEqual(self.store.head("branch").tip, target)
        with self.assertRaises(InvalidOperation):
            self.store.fork("branch", root, namespace_id=self.ns,
                            operation_id="fork-retry",
                            presentation=root_binding(self.ns, tokens, user_id="4" * 32))

    def test_fork_operation_id_is_store_global_and_namespace_is_explicit(self):
        root = self.store.create_root("model", [1], {"status": "idle"})
        self.store.fork("first", root, namespace_id=self.ns, operation_id="global-fork")
        with self.assertRaises(InvalidOperation):
            self.store.fork("second", root, namespace_id=self.ns, operation_id="global-fork")
        with self.assertRaises(InvalidOperation):
            self.store.fork("legacy", root, operation_id="missing-namespace")

    def test_namespace_fork_rename_faults_are_retryable(self):
        tokens = [6, 7]
        root = self.store.create_root("fork-fault", tokens, {"status": "idle"})
        binding = root_binding(self.ns, tokens)

        def fail_before(point):
            if point == "after_refs_stage_fsync":
                raise RuntimeError(point)

        with self.assertRaisesRegex(RuntimeError, "after_refs_stage_fsync"):
            Store(self.tmp.name, fault_hook=fail_before).fork(
                "before-fork", root, namespace_id=self.ns,
                operation_id="before-fork-op", presentation=binding)
        with self.assertRaises(NotFound):
            self.store.head("before-fork")
        before_retry = Store(self.tmp.name).fork(
            "before-fork", root, namespace_id=self.ns,
            operation_id="before-fork-op", presentation=binding)
        self.assertEqual(before_retry.revision, 0)

        def fail_after(point):
            if point == "after_refs_rename_before_dir_fsync":
                raise RuntimeError(point)

        with self.assertRaisesRegex(RuntimeError, "after_refs_rename_before_dir_fsync"):
            Store(self.tmp.name, fault_hook=fail_after).fork(
                "after-fork", root, namespace_id=self.ns,
                operation_id="after-fork-op", presentation=binding)
        restarted = Store(self.tmp.name)
        recovered = restarted.fork("after-fork", root, namespace_id=self.ns,
                                   operation_id="after-fork-op", presentation=binding)
        self.assertEqual(recovered.revision, 0)
        self.assertEqual(restarted.presentation_receipt("after-fork-op")["presentation"], binding)

    def make_user_turn(self):
        old_controller = {"status": "stopped", "stream_key": "old-stream",
                          "accepted_in_scope": 2, "policy": {"allowed": [0, 1, 2]}}
        root, _ = self.make_namespace_branch(controller=old_controller)
        prompt = [9, 10]
        generated = [21, 22]
        operation = "turn-one"
        after = {"status": "active", "stream_key": hashlib.sha256(operation.encode()).hexdigest(),
                 "accepted_in_scope": len(generated), "policy": {"allowed": [0, 1, 2]}}
        target = self.store.append(root, prompt + generated, after)
        source_tokens = [1, 2]
        annotation = turn_binding(self.ns, root, 0, source_tokens, prompt, generated, operation)
        head = self.store.move("chat", 0, target, operation, presentation=annotation)
        return root, target, head, annotation, after

    def test_move_receipt_binds_text_spans_and_presentation_digest_atomically(self):
        _root, target, head, annotation, _controller = self.make_user_turn()
        self.assertEqual(head.revision, 1)
        self.assertEqual(self.store.branch_namespace("chat"), self.ns)
        receipt = self.store.presentation_receipt("turn-one")
        self.assertEqual(receipt["presentation"], annotation)
        self.assertEqual(receipt["result"]["tip"], target)
        branch = self.store._read_branch("chat")
        stored = branch["receipts"]["turn-one"]
        self.assertEqual(stored["presentation_digest"], receipt["digest"])
        blob = self.store.root / "presentation-bindings" / (receipt["digest"] + ".json")
        self.assertEqual(blob.read_bytes(), _json_bytes(annotation))
        self.assertEqual(blob.stat().st_mode & 0o777, 0o600)
        self.assertEqual(blob.parent.stat().st_mode & 0o777, 0o700)

    def test_root_binding_move_receipt_revalidates_after_restart(self):
        tokens = [4, 5, 6]
        root = self.store.create_root("root-move", tokens, {"status": "idle"})
        self.store.fork("root-chat", root, namespace_id=self.ns, operation_id="root-fork")
        binding = root_binding(self.ns, tokens)
        result = self.store.move("root-chat", 0, root, "root-publication", presentation=binding)
        self.assertEqual(result.revision, 1)
        restarted = Store(self.tmp.name)
        receipt = restarted.presentation_receipt("root-publication")
        self.assertEqual(receipt["presentation"], binding)
        self.assertEqual(receipt["result"]["revision"], 1)

    def test_move_retry_is_exact_and_changed_text_or_span_conflicts(self):
        root, target, head, annotation, _controller = self.make_user_turn()
        retry = self.store.move("chat", 0, target, "turn-one", presentation=annotation)
        self.assertEqual(retry, head)
        changed_text = {**annotation, "messages": [dict(annotation["messages"][0], content="altered")]}
        with self.assertRaises(InvalidOperation):
            self.store.move("chat", 0, target, "turn-one", presentation=changed_text)
        other_ns = self.store.create_namespace("Other", "other-chat").namespace_id
        changed_namespace = {**annotation, "namespace_id": other_ns}
        with self.assertRaises(InvalidOperation):
            self.store.move("chat", 0, target, "turn-one", presentation=changed_namespace)
        changed_span = {**annotation, "input_span": {**annotation["input_span"], "count": 1}}
        with self.assertRaises(InvalidOperation):
            self.store.move("chat", 0, target, "turn-one", presentation=changed_span)
        self.assertEqual(self.store.head("chat").tip, target)

    def test_descriptor_rejects_wrong_source_revision_namespace_and_target(self):
        root, _target, _head, annotation, _controller = self.make_user_turn()
        malformed = [
            {**annotation, "namespace_id": "f" * 32},
            {**annotation, "source": {**annotation["source"], "revision": 9}},
            {**annotation, "input_span": {**annotation["input_span"], "token_ids_sha256": "0" * 64}},
            {**annotation, "turn_complete": True},
        ]
        for index, descriptor in enumerate(malformed):
            newroot = self.store.create_root("other", [31], {"status": "stopped"})
            self.store.fork("bad" + str(index), newroot, namespace_id=self.ns,
                            operation_id="badfork" + str(index))
            target = self.store.append(root if index == 0 else newroot, [8], {"status": "stopped"})
            with self.subTest(index=index), self.assertRaises(InvalidOperation):
                self.store.move("bad" + str(index), 0, target, "badmove" + str(index),
                                presentation=descriptor)

    def test_presentation_blob_corruption_or_deletion_fails_read(self):
        _root, _target, _head, _annotation, _controller = self.make_user_turn()
        receipt = self.store.presentation_receipt("turn-one")
        path = self.store.root / "presentation-bindings" / (receipt["digest"] + ".json")
        original = path.read_bytes()
        path.write_bytes(original + b" ")
        with self.assertRaises(IntegrityError):
            self.store.presentation_receipt("turn-one")
        path.write_bytes(original)
        path.unlink()
        with self.assertRaises(IntegrityError):
            self.store.presentation_receipt("turn-one")

    def test_fault_before_rename_has_no_receipt_after_rename_retry_recovers_it(self):
        root, _head = self.make_namespace_branch(controller={"status": "stopped"})
        target = self.store.append(root, [7, 8], {"status": "active", "stream_key": hashlib.sha256(b"move-op").hexdigest(),
                                                   "accepted_in_scope": 1, "policy": {"allowed": [0, 1]}})
        descriptor = turn_binding(self.ns, root, 0, [1, 2], [7], [8], "move-op")
        def fail_before(point):
            if point == "after_refs_stage_fsync":
                raise RuntimeError(point)
        before = Store(self.tmp.name, fault_hook=fail_before)
        with self.assertRaisesRegex(RuntimeError, "after_refs_stage_fsync"):
            before.move("chat", 0, target, "move-op", presentation=descriptor)
        with self.assertRaises(NotFound):
            self.store.presentation_receipt("move-op")
        def fail_after(point):
            if point == "after_refs_rename_before_dir_fsync":
                raise RuntimeError(point)
        after = Store(self.tmp.name, fault_hook=fail_after)
        with self.assertRaisesRegex(RuntimeError, "after_refs_rename_before_dir_fsync"):
            after.move("chat", 0, target, "move-op", presentation=descriptor)
        restarted = Store(self.tmp.name)
        result = restarted.move("chat", 0, target, "move-op", presentation=descriptor)
        self.assertEqual(result.revision, 1)
        self.assertEqual(restarted.presentation_receipt("move-op")["presentation"], descriptor)

    def test_blob_link_before_directory_fsync_is_completed_by_retry(self):
        root, _head = self.make_namespace_branch(controller={"status": "stopped"})
        target = self.store.append(root, [7, 8], {"status": "active", "stream_key": hashlib.sha256(b"fsync-op").hexdigest(),
                                                   "accepted_in_scope": 1, "policy": {"allowed": [0, 1]}})
        descriptor = turn_binding(self.ns, root, 0, [1, 2], [7], [8], "fsync-op")
        armed = {"value": True}
        def fail_before_dir_sync(point):
            if point == "before_presentation_blob_dir_fsync" and armed["value"]:
                armed["value"] = False
                raise RuntimeError(point)
        failing = Store(self.tmp.name, fault_hook=fail_before_dir_sync)
        with self.assertRaisesRegex(RuntimeError, "before_presentation_blob_dir_fsync"):
            failing.move("chat", 0, target, "fsync-op", presentation=descriptor)
        self.assertEqual(self.store.head("chat").revision, 0)
        recovered = Store(self.tmp.name).move("chat", 0, target, "fsync-op", presentation=descriptor)
        self.assertEqual(recovered.revision, 1)
        self.assertEqual(Store(self.tmp.name).presentation_receipt("fsync-op")["presentation"], descriptor)

    def test_resume_reuses_prior_assistant_message_id(self):
        _root, first_tip, first_head, first, controller = self.make_user_turn()
        operation = "resume-one"
        stream = controller["stream_key"]
        after = {"status": "active", "stream_key": stream, "accepted_in_scope": 3,
                 "policy": controller["policy"]}
        second_tip = self.store.append(first_tip, [23], after)
        descriptor = {
            "schema": 1, "namespace_id": self.ns, "kind": "resume", "messages": [],
            "source": {"snapshot": first_tip, "revision": first_head.revision,
                       "token_count": len([1, 2, 9, 10, 21, 22])},
            "input_span": {"start": 6, "count": 0, "token_ids_sha256": _hash(_json_bytes([]))},
            "output_span": {"start": 6, "count": 1, "message_id": first["output_span"]["message_id"]},
            "renderer_id": RENDERER, "turn_complete": False,
        }
        result = self.store.move("chat", first_head.revision, second_tip, operation,
                                 presentation=descriptor)
        self.assertEqual(result.revision, 2)
        changed = {**descriptor, "output_span": {**descriptor["output_span"], "message_id": "b" * 32}}
        third = self.store.append(second_tip, [24], {**after, "accepted_in_scope": 4})
        with self.assertRaises(InvalidOperation):
            self.store.move("chat", result.revision, third, "resume-two", presentation=changed)

    def test_active_answer_fork_resolves_and_retains_published_message_id(self):
        _root, active_tip, _head, first, controller = self.make_user_turn()
        fork = self.store.fork("active-fork", active_tip, namespace_id=self.ns,
                               operation_id="active-fork-op")
        resolved = self.store.active_presentation("active-fork")
        assistant_id = first["output_span"]["message_id"]
        self.assertEqual(resolved["message_id"], assistant_id)
        self.assertEqual(resolved["snapshot"], active_tip)
        self.assertEqual(resolved["namespace_id"], self.ns)
        self.assertEqual(resolved["presentation"]["output_span"]["message_id"], assistant_id)
        restricted = self.store.for_principal("reader")
        with self.assertRaises(PermissionError):
            restricted.branch_namespace("active-fork")
        with self.assertRaises(PermissionError):
            restricted.active_presentation("active-fork")
        with self.assertRaises(PermissionError):
            restricted.presentation_receipt("turn-one")

        stream = controller["stream_key"]
        after = {"status": "active", "stream_key": stream, "accepted_in_scope": 3,
                 "policy": controller["policy"]}
        target = self.store.append(active_tip, [23], after)
        descriptor = {
            "schema": 1, "namespace_id": self.ns, "kind": "resume", "messages": [],
            "source": {"snapshot": active_tip, "revision": fork.revision,
                       "token_count": 6},
            "input_span": {"start": 6, "count": 0, "token_ids_sha256": _hash(_json_bytes([]))},
            "output_span": {"start": 6, "count": 1, "message_id": assistant_id},
            "renderer_id": RENDERER, "turn_complete": False,
        }
        moved = self.store.move("active-fork", fork.revision, target, "resume-fork",
                                presentation=descriptor)
        self.assertEqual(moved.tip, target)
        # Request-bound resume resolution remains stable for the source snapshot
        # after the branch head has advanced.
        historical = self.store.active_presentation("active-fork", active_tip)
        self.assertEqual(historical["snapshot"], active_tip)
        self.assertEqual(historical["presentation"]["output_span"]["message_id"], assistant_id)

    def test_initial_question_active_fork_keeps_assistant_identity(self):
        root,head=self.make_namespace_branch()
        state=dict(status='active',stream_key=hashlib.sha256(b'first-question').hexdigest(),
                   policy={'kind':'greedy'},accepted_in_scope=1)
        target=self.store.append(root,[3,4],state)
        binding=turn_binding(self.ns,root,0,[1,2],[3],[4],'first-question')
        binding['kind']='initial_turn'
        self.store.move('chat',0,target,'first-question',presentation=binding)
        fork=self.store.fork('first-fork',target,namespace_id=self.ns,operation_id='first-fork')
        self.assertEqual(self.store.active_presentation(fork.name)['message_id'],'a'*32)
        self.assertEqual(self.store.get_snapshot(root)['inputs'],[1,2])

    def test_legacy_branch_bytes_and_receipts_remain_schema_one(self):
        root = self.store.create_root("legacy", [1, 2], {"status": "idle"})
        head = self.store.fork("legacy", root)
        fork_bytes = (self.store.references / "legacy.json").read_bytes()
        self.assertEqual(self.store._read_branch("legacy")["schema"], 1)
        target = self.store.append(root, [3], {"status": "stopped"})
        self.store.move("legacy", head.revision, target, "legacy-op")
        branch = self.store._read_branch("legacy")
        self.assertEqual(branch["schema"], 1)
        self.assertNotIn("presentation_digest", branch["receipts"]["legacy-op"])
        self.assertNotIn("presentation-bindings", [p.name for p in self.store.root.iterdir()])
        expected_fork = {"schema": 1, "name": "legacy",
            "head": {"revision": 0, "tip": root, "root": root}, "receipts": {},
            "events": [{"type": "fork", "branch": "legacy", "tip": root}], "orders": [0]}
        self.assertEqual(fork_bytes, _json_bytes(expected_fork))
        expected_receipt = {"digest": _hash(_json_bytes({"type": "move", "name": "legacy",
                                                           "expected_revision": 0, "target": target})),
                            "result": {"name": "legacy", "revision": 1,
                                       "tip": target, "root": root}}
        expected_fork["head"] = {"revision": 1, "tip": target, "root": root}
        expected_fork["receipts"] = {"legacy-op": expected_receipt}
        expected_fork["events"].append({"type": "move", "branch": "legacy",
                                         "operation_id": "legacy-op", "from": root,
                                         "to": target, "from_revision": 0, "to_revision": 1})
        expected_fork["orders"].append(1)
        self.assertEqual((self.store.references / "legacy.json").read_bytes(),
                         _json_bytes(expected_fork))


if __name__ == "__main__":
    unittest.main()
