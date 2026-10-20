"""Read-only COW workspace projection tests; no SQLite heads or model work."""

import hashlib
import tempfile
import unittest

from context_time_travel import IntegrityError, NotFound, Store
from context_time_travel.store import _hash, _json_bytes
from serve.cow_workspace import CowWorkspace

RENDERER = hashlib.sha256(b"workspace-renderer-v1").hexdigest()
ASSISTANT = "a" * 32


class ByteTokenizer:
    def __init__(self, mapping):
        self.mapping = mapping
    def token_bytes(self, token):
        return self.mapping[token]


def root_presentation(namespace, tokens, messages, complete=False):
    return {"schema": 1, "namespace_id": namespace, "kind": "root",
            "messages": messages,
            "source": {"snapshot": None, "revision": 0, "token_count": 0},
            "input_span": {"start": 0, "count": len(tokens),
                           "token_ids_sha256": _hash(_json_bytes(tokens))},
            "output_span": None, "renderer_id": RENDERER, "turn_complete": complete}


def turn_presentation(namespace, kind, source, revision, history, prompt, outputs,
                      operation, assistant_id=ASSISTANT, assistant=None,
                      messages=None, complete=False):
    if assistant is not None:
        assistant_id = assistant
    return {"schema": 1, "namespace_id": namespace, "kind": kind,
            "messages": messages or [],
            "source": {"snapshot": source, "revision": revision,
                       "token_count": len(history)},
            "input_span": {"start": len(history), "count": len(prompt),
                           "token_ids_sha256": _hash(_json_bytes(prompt))},
            "output_span": {"start": len(history) + len(prompt),
                            "count": len(outputs), "message_id": assistant_id},
            "renderer_id": RENDERER, "turn_complete": complete}


class CowWorkspaceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="cow-workspace-")
        self.addCleanup(self.tmp.cleanup)
        self.store = Store(self.tmp.name)
        self.ns = self.store.create_namespace("Code chat", "workspace-ns-1").namespace_id
        self.tokenizer = ByteTokenizer({1: b"Q", 2: b"C", 3: bytes.fromhex("e2"), 4: bytes.fromhex("82ac"),
                                        5: b"A", 6: b"B", 7: b"!", 99: b""})
        self.workspace = CowWorkspace(self.store, self.tokenizer)

    def _root(self, *, name="chat", tokens=(1,), messages=None, ns=None,
              controller=None, fork_op="fork-chat", complete=False):
        ns = ns or self.ns
        messages = messages or [{"id": "1" * 32, "role": "user", "content": "question"}]
        root = self.store.create_root("model", list(tokens), controller or {"status": "idle"})
        self.store.fork(name, root, namespace_id=ns, operation_id=fork_op,
                        presentation=root_presentation(ns, list(tokens), messages, complete=complete))
        return root

    def _answer(self, root, *, name="chat", revision=0, operation="answer-one",
                output=(3,), status="active", accepted=None, pending="", assistant=ASSISTANT,
                controller=None):
        generated = list(output)
        active_controller = controller or {
            "status": status, "stream_key": hashlib.sha256(operation.encode()).hexdigest(),
            "accepted_in_scope": len(generated) if accepted is None else accepted,
            "policy": {"temperature": 0, "eos_tokens": []},
            "output_pending_hex": pending,
        }
        target = self.store.append(root, generated, active_controller)
        binding = turn_presentation(self.ns, "answer", root, revision, list(self.store.lineage(root)[1]),
                                    [], generated, operation, assistant,
                                    complete=status == "stopped")
        head = self.store.move(name, revision, target, operation, presentation=binding)
        return target, head, binding

    def test_empty_and_nonempty_chat_and_branch_catalogs(self):
        empty_ns = self.store.create_namespace("Empty", "workspace-ns-empty").namespace_id
        chats = self.workspace.list_chats()
        self.assertEqual({chat["id"] for chat in chats}, {self.ns, empty_ns})
        empty = next(chat for chat in chats if chat["id"] == empty_ns)
        self.assertEqual(empty["branch_count"], 0)
        root = self._root()
        branch = self.workspace.list_branches(self.ns)
        self.assertEqual(len(branch), 1)
        self.assertEqual(branch[0]["id"], "chat")
        self.assertEqual(branch[0]["tip"], root)
        chat = next(row for row in self.workspace.list_chats() if row["id"] == self.ns)
        self.assertEqual(chat["branch_count"], 1)
        self.assertEqual(chat["max_saved_messages"], 1)

    def test_prepared_root_messages_have_no_invented_per_message_checkpoint(self):
        messages = [{"id": "2" * 32, "role": "system", "content": "system text"},
                    {"id": "3" * 32, "role": "user", "content": "question"}]
        root = self._root(tokens=(1, 2), messages=messages)
        history = self.workspace.history("chat")
        self.assertEqual([m["role"] for m in history["messages"]], ["system", "user"])
        self.assertEqual([m["id"] for m in history["messages"]], [m["id"] for m in messages])
        for message in history["messages"]:
            self.assertFalse(message["boundary_available"])
            self.assertIsNone(message["node_id"])
        self.assertEqual(history["saved_points"][0]["snapshot_id"], root)
        self.assertTrue(history["saved_points"][0]["boundary_available"])

    def test_user_and_assistant_count_and_exact_output_boundary(self):
        root = self._root()
        target, _head, _binding = self._answer(root, output=(5,), status="stopped",
                                               operation="complete-answer", assistant="4" * 32)
        history = self.workspace.history("chat")
        self.assertEqual(len(history["messages"]), 2)
        self.assertEqual(history["messages"][0]["text"], "question")
        answer = history["messages"][1]
        self.assertEqual(answer["text"], "A")
        self.assertEqual(answer["id"], "4" * 32)
        self.assertEqual(answer["node_id"], target)
        self.assertTrue(answer["boundary_available"])
        self.assertFalse(answer["incomplete"])

    def test_pending_assistant_is_durable_incomplete_not_provisional(self):
        root = self._root()
        target, _head, _binding = self._answer(root, output=(3,), status="active",
                                               pending="e2", operation="active-answer")
        history = self.workspace.history("chat")
        self.assertTrue(history["pending_controller"]["active"])
        assistant = history["messages"][-1]
        self.assertTrue(assistant["incomplete"])
        self.assertTrue(assistant["pending_utf8"])
        self.assertEqual(assistant["text"], "")
        self.assertEqual(assistant["node_id"], target)
        self.assertNotIn("provisional", assistant)

    def test_resume_extends_same_assistant_id_across_pending_utf8_boundary(self):
        root = self._root()
        active_controller = {"status": "active",
            "stream_key": hashlib.sha256(b"answer-utf8").hexdigest(),
            "accepted_in_scope": 1, "policy": {"temperature": 0, "eos_tokens": [99]},
            "output_pending_hex": "e2"}
        first_tip, first_head, _ = self._answer(root, output=(3,), status="active", pending="e2",
                                                operation="answer-utf8", controller=active_controller)
        stream = hashlib.sha256(b"answer-utf8").hexdigest()
        after = {"status": "stopped", "stream_key": stream, "accepted_in_scope": 3,
                 "policy": {"temperature": 0, "eos_tokens": [99]}, "output_pending_hex": ""}
        second_tokens = [4, 99]
        second_tip = self.store.append(first_tip, second_tokens, after)
        descriptor = turn_presentation(self.ns, "resume", first_tip, first_head.revision,
                                       [1, 3], [], second_tokens, "resume-utf8", complete=True)
        self.store.move("chat", first_head.revision, second_tip, "resume-utf8", presentation=descriptor)
        history = self.workspace.history("chat")
        self.assertEqual(len(history["messages"]), 2)
        answer = history["messages"][-1]
        self.assertEqual(answer["id"], ASSISTANT)
        self.assertEqual(answer["text"], "€")
        self.assertFalse(answer["pending_utf8"])
        self.assertFalse(answer["incomplete"])
        self.assertEqual(answer["node_id"], second_tip)

    def test_initial_turn_input_boundary_is_not_faked(self):
        # An initial question appended to a code-only prepared root is visible,
        # but its prompt and generated output share one persisted snapshot.
        root = self._root(tokens=(2,), messages=[{"id": "5" * 32, "role": "system", "content": "code"}])
        operation = "initial-question"
        after = {"status": "stopped", "stream_key": hashlib.sha256(operation.encode()).hexdigest(),
                 "accepted_in_scope": 1, "policy": {"temperature": 0, "eos_tokens": []},
                 "output_pending_hex": ""}
        ids = [6, 5]
        target = self.store.append(root, ids, after)
        descriptor = turn_presentation(self.ns, "initial_turn", root, 0, [2], [6], [5], operation,
            assistant="6" * 32, messages=[{"id": "7" * 32, "role": "user", "content": "ask now"}], complete=True)
        self.store.move("chat", 0, target, operation, presentation=descriptor)
        history = self.workspace.history("chat")
        user = next(message for message in history["messages"] if message["id"] == "7" * 32)
        self.assertFalse(user["boundary_available"])
        self.assertIsNone(user["node_id"])
        self.assertTrue(history["messages"][-1]["boundary_available"])


    def test_user_turn_input_boundary_is_not_faked(self):
        root = self._root()
        stopped_tip, stopped_head, _ = self._answer(root, output=(5,), status="stopped",
                                                     operation="first-complete-answer")
        operation = "next-question"
        after = {"status": "stopped", "stream_key": hashlib.sha256(operation.encode()).hexdigest(),
                 "accepted_in_scope": 1, "policy": {"temperature": 0, "eos_tokens": []},
                 "output_pending_hex": ""}
        target = self.store.append(stopped_tip, [7, 5], after)
        descriptor = turn_presentation(self.ns, "user_turn", stopped_tip, stopped_head.revision,
            [1, 5], [7], [5], operation,
            assistant="8" * 32, messages=[{"id": "9" * 32, "role": "user", "content": "second"}], complete=True)
        self.store.move("chat", stopped_head.revision, target, operation, presentation=descriptor)
        history = self.workspace.history("chat")
        user = next(message for message in history["messages"] if message["id"] == "9" * 32)
        self.assertFalse(user["boundary_available"])
        self.assertEqual(user["text"], "second")

    def test_forked_active_branches_get_distinct_assistant_bodies(self):
        root = self._root()
        tip, _head, _binding = self._answer(root, output=(5,), status="active", operation="shared-answer")
        self.store.fork("left", tip, namespace_id=self.ns, operation_id="fork-left")
        self.store.fork("right", tip, namespace_id=self.ns, operation_id="fork-right")
        stream = hashlib.sha256(b"shared-answer").hexdigest()
        for branch, token in (("left", 6), ("right", 7)):
            before = self.store.get_snapshot(tip)["controller"]
            after = {**before, "accepted_in_scope": 2, "output_pending_hex": ""}
            target = self.store.append(tip, [token], after)
            operation = "resume-" + branch
            descriptor = turn_presentation(self.ns, "resume", tip, 0, [1, 5], [], [token], operation,
                                           assistant=ASSISTANT)
            self.store.move(branch, 0, target, operation, presentation=descriptor)
        left = self.workspace.history("left")["messages"][-1]
        right = self.workspace.history("right")["messages"][-1]
        self.assertEqual(left["id"], right["id"])
        self.assertNotEqual(left["text"], right["text"])
        self.assertEqual((left["text"], right["text"]), ("AB", "A!"))

    def test_rewind_hides_abandoned_messages_but_retains_saved_points(self):
        root = self._root()
        first_tip, first_head, _ = self._answer(root, output=(5,), status="stopped",
                                               operation="answer-before-rewind")
        operation = "rewind-root"
        descriptor = {"schema": 1, "namespace_id": self.ns, "kind": "rewind", "messages": [],
            "source": {"snapshot": first_tip, "revision": first_head.revision, "token_count": 2},
            "input_span": None, "output_span": None, "renderer_id": RENDERER, "turn_complete": False}
        self.store.move("chat", first_head.revision, root, operation, presentation=descriptor)
        history = self.workspace.history("chat")
        self.assertEqual([message["text"] for message in history["messages"]], ["question"])
        old_tip = next(point for point in history["saved_points"] if point["snapshot_id"] == first_tip)
        self.assertFalse(old_tip["on_selected_ancestry"])
        self.assertTrue(old_tip["boundary_available"])
        self.assertTrue(any(point["operation_id"] == operation for point in history["saved_points"]))

    def test_identical_text_with_different_message_ids_is_not_deduplicated(self):
        root = self._root()
        first_tip, first_head, _ = self._answer(root, output=(5,), status="stopped", operation="answer-complete")
        for branch, user_id, op in (("left", "a1" * 16, "same-left"), ("right", "b2" * 16, "same-right")):
            self.store.fork(branch, first_tip, namespace_id=self.ns, operation_id="fork-" + branch)
            after = {"status": "stopped", "stream_key": hashlib.sha256(op.encode()).hexdigest(),
                     "accepted_in_scope": 1, "policy": {"temperature": 0, "eos_tokens": []},
                     "output_pending_hex": ""}
            target = self.store.append(first_tip, [6, 5], after)
            desc = turn_presentation(self.ns, "user_turn", first_tip, 0, [1, 5], [6], [5], op,
                assistant=("c" if branch == "left" else "d") * 32,
                messages=[{"id": user_id, "role": "user", "content": "same text"}], complete=True)
            self.store.move(branch, 0, target, op, presentation=desc)
        left = self.workspace.history("left")["messages"]
        right = self.workspace.history("right")["messages"]
        left_user = next(message for message in left if message["role"] == "user" and message["text"] == "same text")
        right_user = next(message for message in right if message["role"] == "user" and message["text"] == "same text")
        self.assertNotEqual(left_user["id"], right_user["id"])
        self.assertEqual(sum(message["text"] == "same text" for message in left), 1)
        self.assertEqual(sum(message["text"] == "same text" for message in right), 1)

    def test_same_computational_root_does_not_merge_distinct_chat_namespaces(self):
        other_ns = self.store.create_namespace("Other", "workspace-ns-2").namespace_id
        tokens = [1]
        first_root = self.store.create_root("model", tokens, {"status": "idle"})
        second_root = self.store.create_root("model", tokens, {"status": "idle"})
        self.assertEqual(first_root, second_root)
        first = [{"id": "e" * 32, "role": "user", "content": "same"}]
        second = [{"id": "f" * 32, "role": "user", "content": "same"}]
        self.store.fork("one", first_root, namespace_id=self.ns, operation_id="one", presentation=root_presentation(self.ns, tokens, first))
        self.store.fork("two", second_root, namespace_id=other_ns, operation_id="two", presentation=root_presentation(other_ns, tokens, second))
        one = CowWorkspace(self.store, self.tokenizer).history("one")
        two = CowWorkspace(self.store, self.tokenizer).history("two")
        self.assertNotEqual(one["chat_id"], two["chat_id"])
        self.assertEqual(one["messages"][0]["text"], two["messages"][0]["text"])
        self.assertNotEqual(one["messages"][0]["id"], two["messages"][0]["id"])

    def test_same_namespace_independent_branches_do_not_merge_root_annotations(self):
        first_id, second_id = "1" * 32, "2" * 32
        first = self.store.create_root("model", [1], {"status": "idle"})
        second = self.store.create_root("model", [1], {"status": "idle"})
        self.assertEqual(first, second)
        self.store.fork("independent-a", first, namespace_id=self.ns,
                        operation_id="independent-fork-a",
                        presentation=root_presentation(self.ns, [1], [
                            {"id": first_id, "role": "user", "content": "same root bytes"}]))
        self.store.fork("independent-b", second, namespace_id=self.ns,
                        operation_id="independent-fork-b",
                        presentation=root_presentation(self.ns, [1], [
                            {"id": second_id, "role": "user", "content": "same root bytes"}]))

        selected = self.workspace.history("independent-a")
        self.assertEqual([message["id"] for message in selected["messages"]], [first_id])
        self.assertEqual([message["text"] for message in selected["messages"]], ["same root bytes"])

    def test_explicit_fork_source_inherits_captured_branch_revision_only(self):
        first_id, second_id = "3" * 32, "4" * 32
        first = self.store.create_root("model", [1], {"status": "idle"})
        second = self.store.create_root("model", [1], {"status": "idle"})
        self.assertEqual(first, second)
        self.store.fork("source-a", first, namespace_id=self.ns, operation_id="source-a-root",
            presentation=root_presentation(self.ns, [1], [
                {"id": first_id, "role": "user", "content": "source A"}]))
        self.store.fork("source-b", second, namespace_id=self.ns, operation_id="source-b-root",
            presentation=root_presentation(self.ns, [1], [
                {"id": second_id, "role": "user", "content": "source B"}]))
        answer_tip, answer_head, _ = self._answer(first, name="source-a", output=(5,),
            status="stopped", operation="source-a-answer")
        self.store.fork("captured-child", answer_tip, namespace_id=self.ns,
            operation_id="child-fork", source_branch="source-a",
            source_revision=answer_head.revision, source_tip=answer_tip)
        # Advance the source after the fork. The child must keep the captured
        # revision and must not inherit the competing source-b annotation.
        self.store.move("source-a", answer_head.revision, first, "source-a-rewind",
            presentation={"schema": 1, "namespace_id": self.ns, "kind": "rewind",
                "messages": [], "source": {"snapshot": answer_tip,
                    "revision": answer_head.revision, "token_count": 2},
                "input_span": None, "output_span": None, "renderer_id": RENDERER,
                "turn_complete": False})
        history = self.workspace.history("captured-child")
        self.assertEqual([message["id"] for message in history["messages"]], [first_id, ASSISTANT])
        self.assertEqual([message["text"] for message in history["messages"]], ["source A", "A"])

    def test_corrupt_annotation_missing_canonical_history_and_restricted_owner_fail(self):
        root = self._root()
        receipt = self.store.presentation_receipt("fork-chat")
        blob = self.store.presentation_bindings / (receipt["digest"] + ".json")
        saved = blob.read_bytes()
        blob.write_bytes(saved[:-1] + b" ")
        with self.assertRaises(IntegrityError):
            self.workspace.history("chat")
        blob.write_bytes(saved)
        restricted = self.store.for_principal("reader")
        with self.assertRaises(PermissionError):
            CowWorkspace(restricted, self.tokenizer).history("chat")
        path = self.store.objects / (root + ".json")
        path.unlink()
        with self.assertRaises((IntegrityError, NotFound)):
            CowWorkspace(Store(self.tmp.name), self.tokenizer).history("chat")

    def test_projection_is_read_only_and_has_no_live_pending_request_claim(self):
        root = self._root()
        self._answer(root, output=(5,), status="active", operation="read-only-active")
        before = dict(self.store._work)
        self.workspace.list_chats()
        self.workspace.list_branches(self.ns)
        history = self.workspace.history("chat")
        after = dict(self.store._work)
        for counter in ("reference_written_bytes", "reference_publications", "snapshot_written_bytes"):
            self.assertEqual(after[counter], before[counter])
        self.assertIsNone(history["pending_request"])
        self.assertTrue(history["pending_controller"]["active"])


if __name__ == "__main__":
    unittest.main()
