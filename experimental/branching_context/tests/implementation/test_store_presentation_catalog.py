"""Read-only namespace-scoped presentation catalog tests."""

import hashlib
import threading
import tempfile
import unittest

from context_time_travel import IntegrityError, Store
from context_time_travel.store import _hash, _json_bytes

RENDERER = hashlib.sha256(b"catalog-renderer-v1").hexdigest()


def root_binding(namespace, tokens, user_id):
    return {"schema": 1, "namespace_id": namespace, "kind": "root",
            "messages": [{"id": user_id, "role": "user", "content": "root text"}],
            "source": {"snapshot": None, "revision": 0, "token_count": 0},
            "input_span": {"start": 0, "count": len(tokens),
                           "token_ids_sha256": _hash(_json_bytes(tokens))},
            "output_span": None, "renderer_id": RENDERER, "turn_complete": False}


def turn_binding(namespace, kind, source, revision, source_tokens, prompt,
                 generated, operation, message_id, *, messages=None, done=False):
    return {"schema": 1, "namespace_id": namespace, "kind": kind,
            "messages": messages or [],
            "source": {"snapshot": source, "revision": revision,
                       "token_count": len(source_tokens)},
            "input_span": {"start": len(source_tokens), "count": len(prompt),
                           "token_ids_sha256": _hash(_json_bytes(prompt))},
            "output_span": {"start": len(source_tokens) + len(prompt),
                            "count": len(generated), "message_id": message_id},
            "renderer_id": RENDERER, "turn_complete": done}


class PresentationCatalogTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="presentation-catalog-")
        self.addCleanup(self.tmp.cleanup)
        self.store = Store(self.tmp.name)
        self.ns = self.store.create_namespace("Catalog", "catalog-ns-1").namespace_id

    def test_empty_namespace_has_no_heads_or_events(self):
        catalog = self.store.presentation_catalog(self.ns)
        self.assertEqual(catalog["namespace"]["namespace_id"], self.ns)
        self.assertEqual(catalog["heads"], [])
        self.assertEqual(catalog["events"], [])

    def test_restricted_principal_is_rejected_before_catalog_disclosure(self):
        restricted = self.store.for_principal("reader")
        with self.assertRaises(PermissionError):
            restricted.presentation_catalog(self.ns)

    def test_identical_roots_are_grouped_only_by_exact_namespace(self):
        other_ns = self.store.create_namespace("Other", "catalog-ns-2").namespace_id
        tokens = [5, 6]
        root = self.store.create_root("model", tokens, {"status": "idle"})
        same = self.store.create_root("model", tokens, {"status": "idle"})
        self.assertEqual(root, same)
        first = root_binding(self.ns, tokens, "1" * 32)
        second = root_binding(other_ns, tokens, "2" * 32)
        self.store.fork("one", root, namespace_id=self.ns, operation_id="fork-one", presentation=first)
        self.store.fork("two", same, namespace_id=other_ns, operation_id="fork-two", presentation=second)
        catalog = self.store.presentation_catalog(self.ns)
        self.assertEqual([head["name"] for head in catalog["heads"]], ["one"])
        self.assertEqual([event["operation_id"] for event in catalog["events"]], ["fork-one"])
        self.assertEqual(catalog["events"][0]["presentation"], first)

    def test_full_typed_receipt_projection_order_and_rewind_retains_events(self):
        tokens = [1, 2, 3]
        old = {"status": "idle"}
        root = self.store.create_root("model", tokens, old)
        root_desc = root_binding(self.ns, tokens, "3" * 32)
        self.store.fork("chat", root, namespace_id=self.ns, operation_id="fork-chat",
                        presentation=root_desc)
        assistant = "a" * 32
        answer_op = "answer-one"
        active = {"status": "active", "stream_key": hashlib.sha256(answer_op.encode()).hexdigest(),
                  "accepted_in_scope": 2, "policy": {"limit": 8}}
        answer_tip = self.store.append(root, [10, 11], active)
        answer = turn_binding(self.ns, "answer", root, 0, tokens, [], [10, 11],
                              answer_op, assistant)
        self.store.move("chat", 0, answer_tip, answer_op, presentation=answer)

        resume_op = "resume-one"
        stopped = {**active, "status": "stopped", "accepted_in_scope": 4}
        stopped_tip = self.store.append(answer_tip, [12, 13], stopped)
        resume = turn_binding(self.ns, "resume", answer_tip, 1, tokens + [10, 11], [], [12, 13],
                              resume_op, assistant, done=True)
        self.store.move("chat", 1, stopped_tip, resume_op, presentation=resume)

        user_op = "user-turn-two"
        user_active = {"status": "active", "stream_key": hashlib.sha256(user_op.encode()).hexdigest(),
                       "accepted_in_scope": 1, "policy": {"limit": 8}}
        user_tip = self.store.append(stopped_tip, [20, 21], user_active)
        user = turn_binding(self.ns, "user_turn", stopped_tip, 2,
                            tokens + [10, 11, 12, 13], [20], [21], user_op, "b" * 32,
                            messages=[{"id": "4" * 32, "role": "user", "content": "next"}])
        self.store.move("chat", 2, user_tip, user_op, presentation=user)

        rewind_op = "rewind-answer"
        rewind = {"schema": 1, "namespace_id": self.ns, "kind": "rewind", "messages": [],
                  "source": {"snapshot": user_tip, "revision": 3, "token_count": len(tokens) + 6},
                  "input_span": None, "output_span": None, "renderer_id": RENDERER,
                  "turn_complete": True}
        self.store.move("chat", 3, stopped_tip, rewind_op, presentation=rewind)

        catalog = self.store.presentation_catalog(self.ns)
        self.assertEqual([event["operation_id"] for event in catalog["events"]],
                         ["fork-chat", answer_op, resume_op, user_op, rewind_op])
        self.assertEqual([event["ordinal"] for event in catalog["events"]],
                         sorted(event["ordinal"] for event in catalog["events"]))
        self.assertEqual([event["presentation"]["kind"] for event in catalog["events"]],
                         ["root", "answer", "resume", "user_turn", "rewind"])
        self.assertEqual(catalog["heads"][0]["tip"], stopped_tip)
        self.assertEqual(catalog["events"][2]["presentation"]["output_span"]["message_id"], assistant)
        self.assertEqual(catalog["events"][0]["receipt"]["result"]["revision"], 0)

    def test_catalog_is_detached_and_never_writes_store_state(self):
        tokens = [7]
        root = self.store.create_root("model", tokens, {"status": "idle"})
        descriptor = root_binding(self.ns, tokens, "5" * 32)
        self.store.fork("chat", root, namespace_id=self.ns, operation_id="fork", presentation=descriptor)
        before = dict(self.store._work)
        catalog = self.store.presentation_catalog(self.ns)
        catalog["namespace"]["title"] = "changed"
        catalog["heads"][0]["name"] = "changed"
        catalog["events"][0]["presentation"]["messages"][0]["content"] = "changed"
        after = dict(self.store._work)
        for counter in ("reference_written_bytes", "reference_publications", "snapshot_written_bytes"):
            self.assertEqual(after[counter], before[counter])
        fresh = self.store.presentation_catalog(self.ns)
        self.assertEqual(fresh["namespace"]["title"], "Catalog")
        self.assertEqual(fresh["heads"][0]["name"], "chat")
        self.assertEqual(fresh["events"][0]["presentation"]["messages"][0]["content"], "root text")

    def test_corrupt_namespace_or_referenced_blob_fails_closed(self):
        tokens = [8]
        root = self.store.create_root("model", tokens, {"status": "idle"})
        self.store.fork("chat", root, namespace_id=self.ns, operation_id="fork",
                        presentation=root_binding(self.ns, tokens, "6" * 32))
        receipt = self.store.presentation_receipt("fork")
        blob = self.store.presentation_bindings / (receipt["digest"] + ".json")
        original = blob.read_bytes()
        blob.write_bytes(original[:-1] + (b" " if original[-1:] != b" " else b"\n"))
        with self.assertRaises(IntegrityError):
            self.store.presentation_catalog(self.ns)

    def test_corrupt_namespace_record_fails_closed(self):
        path = self.store.presentation_namespaces / (self.ns + ".json")
        raw = path.read_bytes()
        path.write_bytes(raw[:-2] + b"x\n")
        with self.assertRaises(IntegrityError):
            self.store.presentation_catalog(self.ns)

    def test_unrelated_namespace_descriptors_are_not_loaded(self):
        other_ns = self.store.create_namespace("Other", "catalog-ns-2").namespace_id
        tokens = [9]
        first = self.store.create_root("model", tokens, {"status": "idle"})
        second = self.store.create_root("model", tokens, {"status": "idle"})
        self.store.fork("mine", first, namespace_id=self.ns, operation_id="mine-op",
                        presentation=root_binding(self.ns, tokens, "7" * 32))
        self.store.fork("other", second, namespace_id=other_ns, operation_id="other-op",
                        presentation=root_binding(other_ns, tokens, "8" * 32))
        reads = []
        branch_reads = []
        expected = self.store._read_branch("mine")["receipts"]["mine-op"]["presentation_digest"]
        original = self.store._read_presentation_blob
        original_branch = self.store._read_branch
        def track(digest):
            reads.append(digest)
            return original(digest)
        def track_branch(name):
            branch_reads.append(name)
            return original_branch(name)
        self.store._read_presentation_blob = track
        self.store._read_branch = track_branch
        self.store._read_refs = lambda: (_ for _ in ()).throw(AssertionError("global receipt scan"))
        catalog = self.store.presentation_catalog(self.ns)
        self.assertEqual(reads, [expected])
        self.assertEqual(branch_reads, ["mine", "other"])
        self.assertEqual([head["name"] for head in catalog["heads"]], ["mine"])

    def test_shared_descriptor_blob_is_read_once_per_catalog_query(self):
        tokens = [10]
        root = self.store.create_root("model", tokens, {"status": "idle"})
        descriptor = root_binding(self.ns, tokens, "9" * 32)
        self.store.fork("alpha", root, namespace_id=self.ns, operation_id="alpha-op", presentation=descriptor)
        self.store.fork("beta", root, namespace_id=self.ns, operation_id="beta-op", presentation=descriptor)
        reads = []
        original = self.store._read_presentation_blob
        def track(digest):
            reads.append(digest)
            return original(digest)
        self.store._read_presentation_blob = track
        catalog = self.store.presentation_catalog(self.ns)
        self.assertEqual(len(catalog["events"]), 2)
        self.assertEqual(reads, [self.store._read_branch("alpha")["receipts"]["alpha-op"]["presentation_digest"]])

    def test_concurrent_move_cannot_split_catalog_snapshot(self):
        tokens = [11]
        root = self.store.create_root("model", tokens, {"status": "idle"})
        self.store.fork("chat", root, namespace_id=self.ns, operation_id="fork")
        target = self.store.append(root, [12], {"status": "stopped"})
        entered, release = threading.Event(), threading.Event()
        original = self.store._read_branch
        first_call = [True]
        def blocked(name):
            result = original(name)
            if first_call[0] and name == "chat":
                first_call[0] = False
                entered.set()
                if not release.wait(5):
                    raise AssertionError("test synchronization timed out")
            return result
        self.store._read_branch = blocked
        catalog_result = []
        reader = threading.Thread(target=lambda: catalog_result.append(self.store.presentation_catalog(self.ns)))
        writer_result = []
        reader.start()
        self.assertTrue(entered.wait(5))
        writer = threading.Thread(target=lambda: writer_result.append(self.store.move("chat", 0, target, "advance")))
        writer.start()
        release.set()
        reader.join(5)
        writer.join(5)
        self.assertFalse(reader.is_alive())
        self.assertFalse(writer.is_alive())
        self.assertEqual(catalog_result[0]["heads"][0]["tip"], root)
        self.assertEqual(writer_result[0].tip, target)


if __name__ == "__main__":
    unittest.main()
