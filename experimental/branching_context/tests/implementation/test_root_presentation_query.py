"""Focused regression coverage for root-only presentation metadata reads."""

import hashlib
import tempfile
import threading
import unittest
from types import SimpleNamespace

from context_time_travel import IntegrityError, InvalidOperation, Store
from context_time_travel.store import _hash, _json_bytes
from serve.cow_adapter import CowAdapter
from serve.cow_text_adapter import CowTextAdapter


RENDERER = hashlib.sha256(b"root-query-renderer-v1").hexdigest()


def root_binding(namespace, tokens, user_id, *, renderer=RENDERER):
    return {"schema": 1, "namespace_id": namespace, "kind": "root",
            "messages": [{"id": user_id, "role": "user", "content": "root prompt"}],
            "source": {"snapshot": None, "revision": 0, "token_count": 0},
            "input_span": {"start": 0, "count": len(tokens),
                           "token_ids_sha256": _hash(_json_bytes(tokens))},
            "output_span": None, "renderer_id": renderer, "turn_complete": False}


def turn_binding(namespace, kind, source, revision, source_tokens, prompt, generated,
                 operation, *, renderer=RENDERER, messages=None):
    return {"schema": 1, "namespace_id": namespace, "kind": kind, "messages": messages or [],
            "source": {"snapshot": source, "revision": revision,
                       "token_count": len(source_tokens)},
            "input_span": {"start": len(source_tokens), "count": len(prompt),
                           "token_ids_sha256": _hash(_json_bytes(prompt))},
            "output_span": {"start": len(source_tokens) + len(prompt), "count": len(generated),
                            "message_id": hashlib.sha256(operation.encode()).hexdigest()[:32]},
            "renderer_id": renderer, "turn_complete": True}


class RootPresentationQueryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="root-presentation-query-")
        self.addCleanup(self.tmp.cleanup)
        self.store = Store(self.tmp.name)
        self.namespace = self.store.create_namespace("Root query", "root-query-ns").namespace_id
        self.tokens = [31, 32, 33]
        self.root = self.store.create_root("model", self.tokens, {"status": "idle"})

    def add_root_branch(self, name, operation, *, renderer=RENDERER, namespace=None):
        namespace = self.namespace if namespace is None else namespace
        descriptor = root_binding(namespace, self.tokens, "1" * 32, renderer=renderer)
        self.store.fork(name, self.root, namespace_id=namespace, operation_id=operation,
                        presentation=descriptor)
        return descriptor

    def add_future_answers(self, count):
        self.add_root_branch("chat", "root-op")
        tip = self.root
        source_tokens = list(self.tokens)
        for index in range(count):
            operation = f"answer-{index}"
            if index == 0:
                prompt = []
                consumed = [100 + index]
                controller = {"status": "stopped", "stream_key": hashlib.sha256(operation.encode()).hexdigest(),
                              "accepted_in_scope": 1, "policy": {"metadata": "x" * 2048}}
                descriptor = turn_binding(self.namespace, "answer", tip, index, source_tokens,
                                          prompt, consumed, operation)
            else:
                prompt = [1000 + index]
                consumed = [1000 + index, 100 + index]
                controller = {"status": "stopped", "stream_key": hashlib.sha256(operation.encode()).hexdigest(),
                              "accepted_in_scope": 1, "policy": {"metadata": "x" * 2048}}
                descriptor = turn_binding(self.namespace, "user_turn", tip, index, source_tokens,
                    [prompt[0]], [consumed[-1]], operation,
                    messages=[{"id": hashlib.sha256((operation + "-user").encode()).hexdigest()[:32],
                               "role": "user", "content": f"question {index}: " + "x" * 2048}])
            target = self.store.append(tip, consumed, controller)
            self.store.move("chat", index, target, operation, presentation=descriptor)
            tip = target
            source_tokens.extend(consumed)

    def test_query_matches_full_catalog_root_projection(self):
        expected = self.add_root_branch("chat", "root-op")
        # This method is intentionally absent in the RED run.
        actual = self.store.root_presentations(self.namespace, self.root)
        self.assertEqual(actual, [expected])
        self.assertIsNot(actual[0], expected)
        actual[0]["messages"][0]["content"] = "mutated detached result"
        self.assertEqual(self.store.root_presentations(self.namespace, self.root)[0], expected)
        catalog = self.store.presentation_catalog(self.namespace)
        catalog_roots = [event["presentation"] for event in catalog["events"]
                         if event["result"]["tip"] == self.root
                         and event["presentation"]["kind"] == "root"]
        self.assertEqual(self.store.root_presentations(self.namespace, self.root), catalog_roots)

    def test_large_future_receipts_are_not_opened_or_snapshotted(self):
        self.add_future_answers(30)
        root_digest = self.store._read_branch("chat")["receipts"]["root-op"]["presentation_digest"]
        future_digests = {receipt["presentation_digest"]
                          for operation, receipt in self.store._read_branch("chat")["receipts"].items()
                          if operation != "root-op"}
        blob_reads = []
        snapshot_reads = []
        read_blob = self.store._read_presentation_blob
        read_snapshot = self.store.get_snapshot

        def track_blob(digest):
            blob_reads.append(digest)
            return read_blob(digest)

        def track_snapshot(snapshot):
            snapshot_reads.append(snapshot)
            return read_snapshot(snapshot)

        self.store._read_presentation_blob = track_blob
        self.store.get_snapshot = track_snapshot
        entries_before = self.store.work_counters()["reference_entries_scanned"]
        selected = self.store.root_presentations(self.namespace, self.root)
        self.assertEqual(len(selected), 1)
        self.assertEqual(blob_reads, [root_digest])
        self.assertTrue(future_digests.isdisjoint(blob_reads))
        self.assertTrue(set(snapshot_reads).issubset({self.root}), snapshot_reads)
        entries_after = self.store.work_counters()["reference_entries_scanned"]
        self.assertEqual(entries_after - entries_before, 32)

    def test_rewind_to_root_does_not_load_its_large_future_source(self):
        self.add_root_branch("chat", "root-op")
        generated = list(range(2000, 4048))
        answer_op = "large-answer-before-rewind"
        controller = {"status": "stopped",
                      "stream_key": hashlib.sha256(answer_op.encode()).hexdigest(),
                      "accepted_in_scope": len(generated),
                      "policy": {"metadata": "large future answer " * 1024}}
        future = self.store.append(self.root, generated, controller)
        answer = turn_binding(self.namespace, "answer", self.root, 0, self.tokens,
                              [], generated, answer_op)
        self.store.move("chat", 0, future, answer_op, presentation=answer)

        rewind = {"schema": 1, "namespace_id": self.namespace, "kind": "rewind",
                  "messages": [],
                  "source": {"snapshot": future, "revision": 1,
                             "token_count": len(self.tokens) + len(generated)},
                  "input_span": None, "output_span": None,
                  "renderer_id": RENDERER, "turn_complete": False}
        self.store.move("chat", 1, self.root, "rewind-to-root", presentation=rewind)

        source_reads = []
        get_snapshot = self.store.get_snapshot

        def track_snapshot(snapshot):
            source_reads.append(snapshot)
            return get_snapshot(snapshot)

        self.store.get_snapshot = track_snapshot
        selected = self.store.root_presentations(self.namespace, self.root)
        self.assertEqual(len(selected), 1)
        self.assertNotIn(future, source_reads,
                         "a rewind receipt points to a large future source that root lookup must skip")

        source_reads.clear()
        catalog = self.store.presentation_catalog(self.namespace)
        self.assertIn(future, source_reads,
                      "the full catalog still validates rewind source history")
        self.assertEqual([event["presentation"]["kind"] for event in catalog["events"]],
                         ["root", "answer", "rewind"])

    def test_future_corruption_is_outside_targeted_query_but_catalog_still_rejects(self):
        self.add_future_answers(2)
        branch = self.store._read_branch("chat")
        digest = branch["receipts"]["answer-1"]["presentation_digest"]
        path = self.store.presentation_bindings / (digest + ".json")
        raw = path.read_bytes()
        path.write_bytes(raw[:-1] + (b" " if raw[-1:] != b" " else b"\n"))
        self.assertEqual(len(self.store.root_presentations(self.namespace, self.root)), 1)
        with self.assertRaises(IntegrityError):
            self.store.presentation_catalog(self.namespace)

    def test_selected_root_blob_corruption_and_symlink_fail_closed(self):
        self.add_root_branch("chat", "root-op")
        digest = self.store._read_branch("chat")["receipts"]["root-op"]["presentation_digest"]
        path = self.store.presentation_bindings / (digest + ".json")
        raw = path.read_bytes()
        path.write_bytes(raw[:-1] + (b" " if raw[-1:] != b" " else b"\n"))
        with self.assertRaises(IntegrityError):
            self.store.root_presentations(self.namespace, self.root)
        path.unlink()
        target = self.store.root / "replacement.json"
        target.write_bytes(raw)
        path.symlink_to(target)
        with self.assertRaises((IntegrityError, PermissionError)):
            self.store.root_presentations(self.namespace, self.root)

    def test_namespace_root_filter_duplicates_and_authorization(self):
        other_namespace = self.store.create_namespace("Other", "root-query-ns-other").namespace_id
        no_annotation_namespace = self.store.create_namespace("No annotation", "root-query-ns-empty").namespace_id
        first = self.add_root_branch("alpha", "root-alpha")
        second = self.add_root_branch("beta", "root-beta", renderer="f" * 64)
        self.add_root_branch("other", "root-other", namespace=other_namespace)
        selected = self.store.root_presentations(self.namespace, self.root)
        self.assertEqual(selected, [first, second])
        self.assertEqual(self.store.root_presentations(other_namespace, self.root),
                         [root_binding(other_namespace, self.tokens, "1" * 32)])
        self.assertEqual(self.store.root_presentations(no_annotation_namespace, self.root), [])
        unrelated_root = self.store.create_root("model", [91, 92], {"status": "idle"})
        unrelated = root_binding(self.namespace, [91, 92], "2" * 32)
        self.store.fork("unrelated", unrelated_root, namespace_id=self.namespace,
                        operation_id="root-unrelated", presentation=unrelated)
        self.assertEqual(self.store.root_presentations(self.namespace, self.root), [first, second])
        self.assertEqual(self.store.root_presentations(self.namespace, unrelated_root), [unrelated])
        text = CowTextAdapter(CowAdapter(SimpleNamespace(store=self.store)),
            SimpleNamespace(token_bytes=lambda token: b""), lambda messages: "", lambda value: [],
            initial_root_mode=lambda canonical, controller: "context_only", renderer_id=RENDERER)
        head = SimpleNamespace(name="alpha", tip=self.root, root=self.root)
        self.assertEqual(text._root_mode_for_source(head, self.tokens, {"status": "idle"}),
                         "unsupported", "conflicting root renderers stay fail-closed")
        restricted = self.store.for_principal("reader")
        with self.assertRaises(PermissionError):
            restricted.root_presentations(self.namespace, self.root)

    def test_unannotated_root_returns_empty_not_inferred_metadata(self):
        self.store.fork("plain", self.root)
        self.assertEqual(self.store.root_presentations(self.namespace, self.root), [])

    def test_cow_adapter_exposes_store_root_query(self):
        expected = [{"kind": "root"}]

        class QueryStore:
            def root_presentations(self, namespace, root):
                self.called = (namespace, root)
                return expected

        store = QueryStore()
        adapter = CowAdapter(SimpleNamespace(store=store))
        self.assertEqual(adapter.root_presentations("ns", "root"), expected)
        self.assertEqual(store.called, ("ns", "root"))

    def test_text_root_mode_prefers_bounded_query_and_uses_legacy_fallback(self):
        descriptor = root_binding(self.namespace, self.tokens, "3" * 32)

        class TokenAdapter:
            def __init__(self, has_query):
                self.calls = []
                self.has_query = has_query

            def namespace(self, branch):
                return self_ns

            def root_presentations(self, namespace, root):
                self.calls.append(("root", namespace, root))
                if not self.has_query:
                    raise AssertionError("legacy adapter must not call absent query")
                return [descriptor]

            def presentation_catalog(self, namespace):
                self.calls.append(("catalog", namespace))
                return {"events": [{"result": {"tip": self_root}, "presentation": descriptor}]}

        class LegacyAdapter:
            def namespace(self, branch):
                return self_ns

            def presentation_catalog(self, namespace):
                self.called = namespace
                return {"events": [{"result": {"tip": self_root}, "presentation": descriptor}]}

        self_ns, self_root = self.namespace, self.root
        renderer = descriptor["renderer_id"]
        tokenizer = SimpleNamespace(token_bytes=lambda token: b"")
        head = SimpleNamespace(name="chat", tip=self.root, root=self.root)
        make = lambda adapter: CowTextAdapter(adapter, tokenizer, lambda messages: "", lambda text: [],
                                               initial_root_mode=lambda canonical, controller: "context_only",
                                               renderer_id=renderer)
        current = TokenAdapter(True)
        self.assertEqual(make(current)._root_mode_for_source(head, self.tokens, {"status": "idle"}),
                         "context_only")
        self.assertEqual(current.calls, [("root", self.namespace, self.root)])
        legacy = LegacyAdapter()
        self.assertEqual(make(legacy)._root_mode_for_source(head, self.tokens, {"status": "idle"}),
                         "context_only")
        self.assertEqual(legacy.called, self.namespace)

    def test_nonroot_snapshot_is_not_a_root_query_key(self):
        self.add_root_branch("chat", "root-op")
        child = self.store.append(self.root, [4], {"status": "stopped"})
        with self.assertRaises(InvalidOperation):
            self.store.root_presentations(self.namespace, child)

    def test_query_is_one_history_snapshot_against_concurrent_move(self):
        descriptor = self.add_root_branch("chat", "root-op")
        target = self.store.append(self.root, [55], {"status": "stopped"})
        query = getattr(self.store, "root_presentations", None)
        self.assertTrue(callable(query), "Store must expose the bounded root-presentation query")
        if not callable(query):
            return
        entered, release = threading.Event(), threading.Event()
        original = self.store._read_branch
        first_call = [True]

        def blocked(name):
            value = original(name)
            if first_call[0] and name == "chat":
                first_call[0] = False
                entered.set()
                if not release.wait(5):
                    raise AssertionError("test synchronization timed out")
            return value

        self.store._read_branch = blocked
        query_result, writer_result = [], []
        reader = threading.Thread(target=lambda: query_result.append(
            query(self.namespace, self.root)))
        reader.start()
        self.assertTrue(entered.wait(5))
        writer = threading.Thread(target=lambda: writer_result.append(
            self.store.move("chat", 0, target, "advance")))
        writer.start()
        release.set()
        reader.join(5)
        writer.join(5)
        self.assertFalse(reader.is_alive())
        self.assertFalse(writer.is_alive())
        self.assertEqual(query_result, [[descriptor]])
        self.assertEqual(writer_result[0].revision, 1)


if __name__ == "__main__":
    unittest.main()
