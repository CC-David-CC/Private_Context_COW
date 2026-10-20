"""Regression tests for repeated canonical snapshots on one selected branch."""

import copy
import unittest

from context_time_travel import IntegrityError, Store
from serve.cow_workspace import CowWorkspace
from serve.test_cow_workspace import (
    ASSISTANT,
    CowWorkspaceTests,
    RENDERER,
    root_presentation,
    turn_presentation,
)


class CowWorkspaceRevisitTests(unittest.TestCase):
    def setUp(self):
        CowWorkspaceTests.setUp(self)

    def tearDown(self):
        CowWorkspaceTests.tearDown(self)

    def _revisit_same_resume_snapshot(self):
        """Publish A->B, rewind to A, then publish the same A->B snapshot."""
        root = CowWorkspaceTests._root(self)
        first_tip, first_head, _ = CowWorkspaceTests._answer(
            self, root, output=(5,), status="active", operation="cli-answer-a",
            assistant=ASSISTANT,
        )
        source_controller = self.store.get_snapshot(first_tip)["controller"]
        resumed_controller = {
            **source_controller,
            "accepted_in_scope": 2,
            "output_pending_hex": "",
        }
        second_tip = self.store.append(first_tip, [6], resumed_controller)
        resume_r2 = turn_presentation(
            self.ns, "resume", first_tip, first_head.revision, [1, 5], [], [6],
            "cli-resume-r2", assistant=ASSISTANT,
        )
        head_r2 = self.store.move(
            "chat", first_head.revision, second_tip, "cli-resume-r2",
            presentation=resume_r2,
        )

        rewind = {
            "schema": 1,
            "namespace_id": self.ns,
            "kind": "rewind",
            "messages": [],
            "source": {
                "snapshot": second_tip,
                "revision": head_r2.revision,
                "token_count": 3,
            },
            "input_span": None,
            "output_span": None,
            "renderer_id": resume_r2["renderer_id"],
            "turn_complete": False,
        }
        head_r3 = self.store.move(
            "chat", head_r2.revision, first_tip, "cli-rewind-r3",
            presentation=rewind,
        )

        revisited_tip = self.store.append(first_tip, [6], resumed_controller)
        self.assertEqual(revisited_tip, second_tip)
        resume_r4 = turn_presentation(
            self.ns, "resume", first_tip, head_r3.revision, [1, 5], [], [6],
            "cli-resume-r4", assistant=ASSISTANT,
        )
        head_r4 = self.store.move(
            "chat", head_r3.revision, revisited_tip, "cli-resume-r4",
            presentation=resume_r4,
        )
        return root, first_tip, second_tip, head_r2, head_r3, head_r4

    def test_revisited_snapshot_uses_receipt_at_selected_branch_revision(self):
        _root, first_tip, second_tip, _head_r2, _head_r3, head_r4 = (
            self._revisit_same_resume_snapshot()
        )

        # Reopening exercises catalog reconstruction, rather than relying on
        # process-local ordering or a cached projection.
        reopened = Store(self.tmp.name)
        workspace = CowWorkspace(reopened, self.tokenizer)

        history = workspace.history("chat")
        self.assertEqual(history["revision"], 4)
        self.assertEqual(history["tip"], second_tip)
        self.assertEqual([move["revision"] for move in history["moves"]], [1, 2, 3, 4])
        self.assertEqual(
            [point["operation_id"] for point in history["saved_points"]
             if point["snapshot_id"] == second_tip],
            ["cli-resume-r2", "cli-resume-r4"],
        )
        assistant = next(message for message in history["messages"]
                         if message["role"] == "assistant")
        self.assertEqual(assistant["text"], "AB")
        self.assertEqual(assistant["node_id"], second_tip)

        catalog = reopened.presentation_catalog(self.ns)
        selected_head = next(item for item in catalog["heads"] if item["name"] == "chat")
        selected_events = list(workspace._events_on_ancestry(
            catalog,
            {identity for identity, _record in workspace._canonical_path(selected_head)[2]},
            selected_head,
        ))
        current_operations = [event["operation_id"] for event in selected_events
                              if event["result"]["tip"] == second_tip
                              and event["presentation"]
                              and event["presentation"].get("output_span")]
        self.assertEqual(current_operations, ["cli-resume-r4"])
        historical_head = {**selected_head, "revision": 2, "tip": second_tip}
        historical_events = list(workspace._events_on_ancestry(
            catalog,
            {identity for identity, _record in workspace._canonical_path(historical_head)[2]},
            historical_head,
        ))
        historical_operations = [event["operation_id"] for event in historical_events
                                 if event["result"]["tip"] == second_tip
                                 and event["presentation"]
                                 and event["presentation"].get("output_span")]
        self.assertEqual(historical_operations, ["cli-resume-r2"])

        branch = next(item for item in workspace.list_branches(self.ns)
                      if item["id"] == "chat")
        self.assertEqual((branch["revision"], branch["tip"]), (4, second_tip))
        self.assertEqual(next(item for item in workspace.list_chats()
                              if item["id"] == self.ns)["branch_count"], 1)

        graph = workspace.graph(self.ns, include_previous=True)
        current = next(item for item in graph["branches"] if item["id"] == "chat")
        self.assertEqual((current["revision"], current["snapshot"]),
                         (head_r4.revision, second_tip))
        previous = next(item for item in graph["previous_tips"]
                        if item["branch_id"] == "chat" and item["revision"] == 2)
        self.assertEqual(previous["snapshot"], second_tip)
        historical_node = next(item for item in graph["nodes"]
                               if item["id"] == previous["node_id"])
        self.assertEqual(historical_node["action_snapshot"], second_tip)
        first_point = next(point for point in history["saved_points"]
                           if point["operation_id"] == "cli-answer-a")
        self.assertEqual(first_point["snapshot_id"], first_tip)

    def test_unscoped_conflicting_receipts_remain_ambiguous(self):
        root = self.store.create_root("model", [1], {"status": "idle"})
        for branch, message_id, content in (
            ("producer-a", "1" * 32, "producer A"),
            ("producer-b", "2" * 32, "producer B"),
        ):
            self.store.fork(
                branch, root, namespace_id=self.ns, operation_id=f"fork-{branch}",
                presentation=root_presentation(self.ns, [1], [
                    {"id": message_id, "role": "user", "content": content},
                ]),
            )
        # This legacy/source-less fork has no receipt identifying which
        # conflicting producer annotation it inherited.
        self.store.fork("consumer", root, namespace_id=self.ns,
                        operation_id="fork-consumer")
        workspace = CowWorkspace(self.store, self.tokenizer)
        with self.assertRaisesRegex(IntegrityError, "Ambiguous presentation provenance"):
            workspace.history("consumer")

    def test_selected_explicit_fork_branch_wins_over_captured_parent_receipt(self):
        root = CowWorkspaceTests._root(self)
        first_tip, first_head, _ = CowWorkspaceTests._answer(
            self, root, output=(5,), status="active", operation="parent-answer",
            assistant=ASSISTANT,
        )

        # The parent revisits the same source snapshot at a later branch
        # revision before publishing B, so its valid B descriptor captures a
        # different source revision from the child's publication.
        rewind_parent = {
            "schema": 1, "namespace_id": self.ns, "kind": "rewind", "messages": [],
            "source": {"snapshot": first_tip, "revision": first_head.revision,
                       "token_count": 2},
            "input_span": None, "output_span": None, "renderer_id": RENDERER,
            "turn_complete": False,
        }
        parent_r2 = self.store.move(
            "chat", first_head.revision, root, "parent-rewind", presentation=rewind_parent,
        )
        parent_r3 = self.store.move("chat", parent_r2.revision, first_tip, "parent-return-a")
        source_controller = self.store.get_snapshot(first_tip)["controller"]
        resumed_controller = {
            **source_controller, "accepted_in_scope": 2, "output_pending_hex": "",
        }
        second_tip = self.store.append(first_tip, [6], resumed_controller)
        parent_resume = turn_presentation(
            self.ns, "resume", first_tip, parent_r3.revision, [1, 5], [], [6],
            "parent-resume-b", assistant=ASSISTANT,
        )
        parent_r4 = self.store.move(
            "chat", parent_r3.revision, second_tip, "parent-resume-b",
            presentation=parent_resume,
        )

        child_r0 = self.store.fork(
            "child", second_tip, namespace_id=self.ns, operation_id="fork-child",
            source_branch="chat", source_revision=parent_r4.revision, source_tip=second_tip,
        )
        rewind_child = {
            **rewind_parent,
            "source": {"snapshot": second_tip, "revision": child_r0.revision,
                       "token_count": 3},
        }
        child_r1 = self.store.move(
            "child", child_r0.revision, first_tip, "child-rewind", presentation=rewind_child,
        )
        child_resume = turn_presentation(
            self.ns, "resume", first_tip, child_r1.revision, [1, 5], [], [6],
            "child-resume-b", assistant=ASSISTANT,
        )
        child_r2 = self.store.move(
            "child", child_r1.revision, second_tip, "child-resume-b",
            presentation=child_resume,
        )

        child_history = self.workspace.history("child")
        assistant = next(message for message in child_history["messages"]
                         if message["role"] == "assistant")
        self.assertEqual((child_history["revision"], child_history["tip"]),
                         (child_r2.revision, second_tip))
        self.assertEqual((assistant["text"], assistant["node_id"]), ("AB", second_tip))
        catalog = self.store.presentation_catalog(self.ns)
        child_head = next(item for item in catalog["heads"] if item["name"] == "child")
        child_events = list(self.workspace._events_on_ancestry(
            catalog,
            {identity for identity, _record in self.workspace._canonical_path(child_head)[2]},
            child_head,
        ))
        child_operations = [event["operation_id"] for event in child_events
                            if event["result"]["tip"] == second_tip
                            and event["presentation"]
                            and event["presentation"].get("output_span")]
        self.assertEqual(child_operations, ["child-resume-b"])

    def test_duplicate_scoped_receipts_at_one_revision_fail_closed(self):
        root = CowWorkspaceTests._root(self)
        tip, _head, _ = CowWorkspaceTests._answer(
            self, root, output=(5,), status="stopped", operation="scoped-answer",
        )
        catalog = self.store.presentation_catalog(self.ns)
        branch_head = next(item for item in catalog["heads"] if item["name"] == "chat")
        original = next(event for event in catalog["events"]
                        if event["operation_id"] == "scoped-answer")
        duplicate = copy.deepcopy(original)
        duplicate["operation_id"] = "corrupt-same-revision-receipt"
        duplicate["digest"] = "f" * 64
        catalog["events"].append(duplicate)
        _root_id, _tokens, path, _positions = self.workspace._canonical_path(branch_head)
        ancestry = {identity for identity, _record in path}
        with self.assertRaisesRegex(IntegrityError, "Multiple scoped presentation receipts"):
            list(self.workspace._events_on_ancestry(catalog, ancestry, branch_head))


if __name__ == "__main__":
    unittest.main()
