"""CPU regressions for previous-tip refresh causality in the browser harness."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


SMOKE = Path(__file__).parent / 'web' / 'cow_native_smoke.py'
spec = importlib.util.spec_from_file_location('cow_native_smoke_graph_observation', SMOKE)
smoke = importlib.util.module_from_spec(spec)
spec.loader.exec_module(smoke)


class RewindGraphObservationTests(unittest.TestCase):
    def setUp(self):
        self.after = {'id': 'branch-B', 'chat_id': 'chat-1', 'revision': 2, 'tip': 'root-tip'}
        self.old_tip = 'answer-B-tip'

    def _observation(self, *, revision=2, snapshot='root-tip', selected_revision=2,
                     selected_tip='root-tip', stale=False, previous=None):
        graph = {
            'chat_id': 'chat-1',
            'previous_tips_requested': True,
            'branches': [{'id': 'branch-B', 'revision': revision,
                          'snapshot': snapshot, 'tip': 'graph-node'}],
            'previous_tips': previous if previous is not None else [
                {'branch_id': 'branch-B', 'revision': 1, 'snapshot': self.old_tip,
                 'node_id': 'old-node'}],
        }
        return {'graph': graph, 'stale': stale, 'epoch': 8,
                'selected': {'id': 'branch-B', 'chat_id': 'chat-1',
                             'revision': selected_revision, 'tip': selected_tip},
                'graph_head': {'id': 'branch-B', 'revision': revision,
                               'snapshot': snapshot, 'tip': 'graph-node'}}

    def test_rejects_preexisting_previous_enabled_graph_from_before_rewind(self):
        # The prior graph can already have previous=true and stale=false. Its
        # previous-tip list may even contain the expected snapshot; the head
        # revision/tip binding must still reject that pre-rewind response.
        old_graph = self._observation(revision=1, snapshot=self.old_tip,
                                     selected_revision=1, selected_tip=self.old_tip)
        with self.assertRaisesRegex(RuntimeError, 'different selected branch head'):
            smoke._validate_rewound_previous_graph(old_graph, self.after, 'branch-B',
                                                   self.old_tip, 1)

    def test_accepts_fresh_graph_bound_to_post_rewind_head_and_unique_saved_tip(self):
        fresh = self._observation()
        retained = smoke._validate_rewound_previous_graph(fresh, self.after, 'branch-B',
                                                          self.old_tip, 1)
        self.assertEqual(len(retained), 1)
        self.assertEqual(retained[0]['snapshot'], self.old_tip)

    def test_rejects_duplicate_matching_saved_tip_records(self):
        duplicate = [
            {'branch_id': 'branch-B', 'revision': 1, 'snapshot': self.old_tip},
            {'branch_id': 'branch-B', 'revision': 1, 'snapshot': self.old_tip},
        ]
        with self.assertRaisesRegex(RuntimeError, 'not retained as one'):
            smoke._validate_rewound_previous_graph(self._observation(previous=duplicate),
                self.after, 'branch-B', self.old_tip, 1)

    def test_refresh_expression_returns_the_async_refresh_promise(self):
        expression = smoke._previous_graph_refresh_expression()
        self.assertIn("toggle.checked=true", expression)
        self.assertIn("toggle.dispatchEvent(new Event('change'))", expression)
        self.assertIn('return ContextGraph.refresh().then(', expression)
        self.assertIn('return {graph:graph?', expression)

    def test_freshness_predicate_rejects_superseded_refresh_then_accepts_current_head(self):
        expression = smoke._capture_fresh_rewound_graph_expression(self.after, 'branch-B')
        self.assertIn("g?.chat_id===\"chat-1\"", expression)
        self.assertIn("h?.revision===2", expression)
        self.assertIn("h?.snapshot===\"root-tip\"", expression)
        self.assertIn('g?.previous_tips_requested===true', expression)

        stale = self._observation(revision=1, snapshot=self.old_tip,
                                  selected_revision=2, selected_tip='root-tip')
        current = self._observation()
        after = self.after

        class SupersededRefreshDevtools:
            def __init__(self):
                self.state = stale
                self.attempts = [{'started_epoch': 30, 'settled': stale}]
                self.waited_for_exact_head = False
                self.capture_rounds = 0

            def evaluate(self, source):
                if 'toggle.checked=true' in source:
                    # The awaited refresh settled, but a newer refresh epoch
                    # superseded it before ContextGraph.data was assigned.
                    return {'attempts': list(self.attempts), 'initial': self.state}
                if 'if(!fresh)return null' in source:
                    self.assert_atomic_capture = source == smoke._capture_fresh_rewound_graph_expression(
                        after, 'branch-B')
                    self.capture_rounds += 1
                    if self.state['graph_head']['revision'] == 1:
                        return None
                    self.waited_for_exact_head = True
                    return {'observation': self.state, 'attempts': list(self.attempts),
                            'observer_missing': False}
                if '__cowRewindGraphRefreshObserver' in source:
                    return {'attempts': list(self.attempts), 'final': self.state,
                            'observer_missing': False}
                raise AssertionError('unexpected browser expression')

            def advance_refresh(self):
                # The subsequent refresh wins the epoch race after the first
                # awaited refresh returned while the old graph was still present.
                self.state = current
                self.attempts.append({'started_epoch': 32, 'settled': current})

        devtools = SupersededRefreshDevtools()
        # First capture probe sees the superseded stale graph; the later probe
        # observes the newest refresh. This models the exact browser race.
        original_evaluate = devtools.evaluate

        def evaluate_with_later_refresh(source):
            if 'if(!fresh)return null' in source and devtools.capture_rounds == 1:
                devtools.advance_refresh()
            return original_evaluate(source)

        devtools.evaluate = evaluate_with_later_refresh
        with tempfile.TemporaryDirectory() as temp:
            result = smoke._refresh_rewound_previous_graph(devtools, self.after,
                'branch-B', self.old_tip, 1, Path(temp))
            receipt_path = Path(temp) / 'rewind-previous-graph-observation.json'
            receipt = json.loads(receipt_path.read_text(encoding='utf-8'))

        self.assertTrue(devtools.assert_atomic_capture)
        self.assertTrue(devtools.waited_for_exact_head)
        self.assertEqual(devtools.capture_rounds, 2)
        self.assertEqual(receipt['initial_refresh_observation']['initial']['graph_head']['revision'], 1)
        self.assertEqual(len(receipt['refresh_attempts']), 2)
        self.assertEqual(receipt['final_observation']['graph_head']['revision'], 2)
        self.assertEqual(result['graph_head']['snapshot'], 'root-tip')


if __name__ == '__main__':
    unittest.main()
