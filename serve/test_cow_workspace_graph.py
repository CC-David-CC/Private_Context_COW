"""Message graph projection regressions; exercised with the CPU fixture only."""

import hashlib
import unittest

from serve.test_cow_workspace import ASSISTANT, turn_presentation
import serve.test_cow_workspace as _workspace_fixtures
from context_time_travel import NotFound


class CowWorkspaceGraphTests(unittest.TestCase):
    def setUp(self):
        _workspace_fixtures.CowWorkspaceTests.setUp(self)

    def _root(self, *args, **kwargs):
        return _workspace_fixtures.CowWorkspaceTests._root(self, *args, **kwargs)

    def _answer(self, *args, **kwargs):
        return _workspace_fixtures.CowWorkspaceTests._answer(self, *args, **kwargs)

    def test_empty_and_single_chat_graph_counts_messages_not_snapshots(self):
        empty = self.store.create_namespace("Empty graph", "graph-empty").namespace_id
        graph = self.workspace.graph(empty)
        self.assertEqual(graph["branches"], [])
        self.assertEqual([node["role"] for node in graph["nodes"]], ["root"])
        self.assertEqual(graph["shared_prefix_messages"], 0)

        root = self._root()
        self._answer(root, operation="graph-answer", output=(5,), status="stopped")
        graph = self.workspace.graph(self.ns)
        self.assertEqual(len(graph["branches"]), 1)
        branch = graph["branches"][0]
        self.assertEqual(branch["saved_messages"], 2)
        self.assertEqual(branch["snapshot"], self.store.head("chat").tip)
        messages = [node for node in graph["nodes"] if node["role"] != "root"]
        self.assertEqual([node["role"] for node in messages], ["user", "assistant"])
        self.assertEqual(messages[-1]["depth"], 2)
        root_nodes = [node for node in graph["nodes"] if node["role"] == "root"]
        self.assertEqual(len(root_nodes), 2)  # namespace anchor plus canonical root snapshot
        self.assertEqual(root_nodes[1]["action_snapshot"], root)
        self.assertTrue(root_nodes[1]["boundary_available"])

    def test_active_assistant_fork_shares_prefix_then_resume_versions_diverge(self):
        root = self._root()
        target, _head, _binding = self._answer(root, output=(3,), status="active",
                                               operation="graph-active", assistant=ASSISTANT,
                                               pending="e2")
        self.store.fork("sibling", target, namespace_id=self.ns, operation_id="graph-fork",
                        source_branch="chat", source_revision=1, source_tip=target)
        before_resume = self.workspace.graph(self.ns)
        self.assertEqual(len({branch["tip"] for branch in before_resume["branches"]}), 1)
        for branch, token in (("chat", 5), ("sibling", 6)):
            before = self.store.head(branch)
            final = self.store.append(before.tip, [4, token], {
                "status": "stopped", "stream_key": hashlib.sha256(b"graph-active").hexdigest(),
                "accepted_in_scope": 3, "policy": {"temperature": 0, "eos_tokens": []},
                "output_pending_hex": "",
            })
            source_tokens = self.store.lineage(before.tip)[1]
            binding = turn_presentation(self.ns, "resume", before.tip, before.revision,
                list(source_tokens), [], [4, token], f"graph-resume-{branch}", ASSISTANT,
                complete=True)
            self.store.move(branch, before.revision, final, f"graph-resume-{branch}", presentation=binding)

        graph = self.workspace.graph(self.ns)
        self.assertEqual({branch["saved_messages"] for branch in graph["branches"]}, {2})
        assistant_nodes = [node for node in graph["nodes"] if node["role"] == "assistant"]
        self.assertEqual(len(assistant_nodes), 3)
        shared = [node for node in assistant_nodes if node["current_branch_count"] == 2]
        self.assertEqual(len(shared), 1)
        self.assertTrue(shared[0]["shared_partial"])
        self.assertEqual(shared[0]["preview"], "")  # canonical pending UTF-8 prefix remains buffered
        self.assertEqual(shared[0]["current_branch_count"], 2)
        self.assertEqual(len({branch["tip"] for branch in graph["branches"]}), 2)
        self.assertEqual({node["preview"] for node in assistant_nodes
                          if node["current_branch_count"] == 1}, {"€A", "€B"})
        self.assertEqual(graph["shared_prefix_messages"], 1)
        self.assertEqual(graph["shared_partial_messages"], 1)

    def test_rewind_graph_keeps_only_current_message_ancestry(self):
        root = self._root()
        self.store.fork("sibling", root, namespace_id=self.ns, operation_id="graph-copy",
                        source_branch="chat", source_revision=0, source_tip=root)
        self._answer(root, name="chat", operation="graph-original", output=(5,),
                     status="stopped", assistant="4" * 32)
        self._answer(root, name="sibling", operation="graph-sibling", output=(6,),
                     status="stopped", assistant="5" * 32)
        before = self.workspace.graph(self.ns)
        self.assertEqual({branch["saved_messages"] for branch in before["branches"]}, {2})
        self.assertEqual(before["shared_prefix_messages"], 1)
        self.store.move("chat", self.store.head("chat").revision, root, "graph-rewind")
        after = self.workspace.graph(self.ns)
        current = {branch["id"]: branch for branch in after["branches"]}
        self.assertEqual(current["chat"]["saved_messages"], 1)
        self.assertEqual(current["sibling"]["saved_messages"], 2)
        self.assertNotIn("4" * 32, {node["message_id"] for node in after["nodes"]})

    def test_previous_tips_are_faded_paths_not_current_saved_message_progress(self):
        root = self._root()
        answer, _head, _binding = self._answer(root, operation="graph-before-rewind",
            output=(5,), status="stopped", assistant="4" * 32)
        self.store.move("chat", self.store.head("chat").revision, root, "graph-rewind-history")
        current = self.workspace.graph(self.ns)
        self.assertNotIn("4" * 32, {node["message_id"] for node in current["nodes"]})
        with_previous = self.workspace.graph(self.ns, include_previous=True)
        self.assertTrue(with_previous["previous_tips_supported"])
        old_tip = next(item for item in with_previous["previous_tips"]
                       if item["snapshot"] == answer)
        self.assertEqual(with_previous["branches"][0]["saved_messages"], 1)
        self.assertIn("4" * 32, {node["message_id"] for node in with_previous["nodes"]})
        old_node = next(node for node in with_previous["nodes"] if node["id"] == old_tip["node_id"])
        self.assertEqual(old_node["action_snapshot"], answer)
        self.assertEqual(old_node["current_branch_count"], 0)

    def test_graph_preview_is_bounded_and_branch_tip_is_not_snapshot_id(self):
        root = self._root()
        self._answer(root, operation="graph-bounded", output=(5,), status="stopped")
        graph = self.workspace.graph(self.ns, preview_chars=8)
        branch = graph["branches"][0]
        self.assertNotEqual(branch["tip"], branch["snapshot"])
        self.assertTrue(all(len(node["preview"]) <= 8 for node in graph["nodes"]))
        self.assertTrue(all(node["action_snapshot"] is None or node["boundary_available"]
                            for node in graph["nodes"]))

    def test_identical_canonical_root_with_distinct_message_provenance_stays_separate(self):
        self._root(name="chat", messages=[{"id": "1" * 32, "role": "user", "content": "same"}],
                   fork_op="graph-root-a")
        self._root(name="other", messages=[{"id": "2" * 32, "role": "user", "content": "same"}],
                   fork_op="graph-root-b")
        graph = self.workspace.graph(self.ns)
        users = [node for node in graph["nodes"] if node["role"] == "user"]
        self.assertEqual(len(users), 2)
        self.assertEqual({node["message_id"] for node in users}, {"1" * 32, "2" * 32})
        self.assertEqual(graph["shared_prefix_messages"], 0)

    def test_branches_follow_committed_creation_order_not_random_name(self):
        self._root(name="z-first", fork_op="graph-create-first")
        self._root(name="a-second", fork_op="graph-create-second")
        graph = self.workspace.graph(self.ns)
        self.assertEqual([branch["id"] for branch in graph["branches"]], ["z-first", "a-second"])
        self.assertLess(graph["branches"][0]["creation_order"],
                        graph["branches"][1]["creation_order"])

    def test_prepared_transcript_assistant_role_is_a_message_not_a_generated_version(self):
        messages = [
            {"id": "1" * 32, "role": "system", "content": "sys"},
            {"id": "2" * 32, "role": "user", "content": "question"},
            {"id": "3" * 32, "role": "assistant", "content": "prior answer"},
            {"id": "4" * 32, "role": "user", "content": "follow-up"},
        ]
        self._root(messages=messages, tokens=(1, 2, 3))
        graph = self.workspace.graph(self.ns)
        prepared = [node for node in graph["nodes"] if node["role"] != "root"]
        self.assertEqual([node["role"] for node in prepared],
                         ["system", "user", "assistant", "user"])
        self.assertEqual([node["depth"] for node in prepared], [1, 2, 3, 4])
        self.assertEqual([node["action_snapshot"] for node in prepared], [None] * 4)

    def test_completed_shared_assistant_is_not_partial_after_newer_branch_tails(self):
        root = self._root()
        answer, _head, _binding = self._answer(root, operation="graph-completed",
            output=(5,), status="stopped", assistant=ASSISTANT)
        self.store.fork("sibling", answer, namespace_id=self.ns, operation_id="graph-copy-completed",
                        source_branch="chat", source_revision=1, source_tip=answer)
        for branch, user_id, token in (("chat", "6" * 32, 5),
                                       ("sibling", "7" * 32, 6)):
            operation = f"graph-question-{branch}"
            before = self.store.head(branch)
            history = list(self.store.lineage(before.tip)[1])
            target = self.store.append(before.tip, [2, token], {
                "status": "stopped", "stream_key": hashlib.sha256(operation.encode()).hexdigest(),
                "accepted_in_scope": 1, "policy": {"temperature": 0, "eos_tokens": []},
                "output_pending_hex": "",
            })
            binding = turn_presentation(self.ns, "user_turn", before.tip, before.revision,
                history, [2], [token], operation, f"{token:032x}",
                messages=[{"id": user_id, "role": "user", "content": "next"}], complete=True)
            self.store.move(branch, before.revision, target,
                            operation, presentation=binding)
        graph = self.workspace.graph(self.ns)
        self.assertEqual({branch["saved_messages"] for branch in graph["branches"]}, {4})
        self.assertEqual(graph["shared_prefix_messages"], 2)
        self.assertEqual(graph["shared_partial_messages"], 0)
        shared_assistant = next(node for node in graph["nodes"]
            if node["role"] == "assistant" and node["current_branch_count"] == 2)
        self.assertFalse(shared_assistant["shared_partial"])

    def test_partial_then_common_completion_is_not_a_shared_partial(self):
        root = self._root()
        partial, _head, _binding = self._answer(root, output=(3,), status="active",
            operation="graph-partial-common", assistant=ASSISTANT, pending="e2")
        before = self.store.head("chat")
        history = list(self.store.lineage(before.tip)[1])
        final = self.store.append(before.tip, [4, 5], {
            "status": "stopped", "stream_key": hashlib.sha256(b"graph-partial-common").hexdigest(),
            "accepted_in_scope": 3, "policy": {"temperature": 0, "eos_tokens": []},
            "output_pending_hex": "",
        })
        binding = turn_presentation(self.ns, "resume", before.tip, before.revision,
            history, [], [4, 5], "graph-partial-common-resume", ASSISTANT, complete=True)
        self.store.move("chat", before.revision, final, "graph-partial-common-resume",
                        presentation=binding)
        self.store.fork("sibling", final, namespace_id=self.ns,
            operation_id="graph-fork-after-common-completion", source_branch="chat",
            source_revision=2, source_tip=final)
        before_question = self.store.head("sibling")
        question_history = list(self.store.lineage(before_question.tip)[1])
        question = self.store.append(before_question.tip, [2, 6], {
            "status": "stopped", "stream_key": hashlib.sha256(b"sibling-question").hexdigest(),
            "accepted_in_scope": 1, "policy": {"temperature": 0, "eos_tokens": []},
            "output_pending_hex": "",
        })
        question_binding = turn_presentation(self.ns, "user_turn", before_question.tip,
            before_question.revision, question_history, [2], [6], "sibling-question",
            "8" * 32, messages=[{"id": "7" * 32, "role": "user", "content": "next"}],
            complete=True)
        self.store.move("sibling", before_question.revision, question,
                        "sibling-question", presentation=question_binding)

        graph = self.workspace.graph(self.ns)
        self.assertEqual(graph["shared_partial_messages"], 0)
        partial_nodes = [node for node in graph["nodes"]
            if node["role"] == "assistant" and node["id"] != graph["branches"][0]["tip"]
            and node["current_branch_count"] == 2]
        self.assertTrue(partial_nodes)
        self.assertTrue(all(not node["shared_partial"] for node in partial_nodes))

    def test_generated_branch_ids_get_stable_display_labels(self):
        self._root(name="b-op" + "1" * 32, fork_op="graph-generated-main")
        self._root(name="b-" + "2" * 32, fork_op="graph-generated-second")
        self._root(name="manual-review", fork_op="graph-human-name")
        expected = ["Main", "Branch2", "manual-review"]
        listed = self.workspace.list_branches(self.ns)
        graph = self.workspace.graph(self.ns)
        self.assertEqual([branch["name"] for branch in listed], expected)
        self.assertEqual([branch["name"] for branch in graph["branches"]], expected)
        self.assertEqual([branch["id"] for branch in graph["branches"]],
            ["b-op" + "1" * 32, "b-" + "2" * 32, "manual-review"])
        for branch in graph["branches"]:
            self.assertEqual(self.workspace.history(branch["id"])["name"], branch["name"])

    def test_unknown_namespace_graph_is_not_disclosed(self):
        with self.assertRaises(NotFound):
            self.workspace.graph("f" * 32)
