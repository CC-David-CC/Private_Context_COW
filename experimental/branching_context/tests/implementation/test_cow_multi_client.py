"""CPU-only real-HTTP tests for the concurrent COW client harness."""
from pathlib import Path
import sys
import tempfile
import threading
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))

from run_cow_multi_client import (  # noqa: E402
    RecoveryFailure,
    _concurrency_evidence,
    _verify_canonical_snapshot,
    run_concurrent_clients,
)
from context_time_travel.controller import consume_many  # noqa: E402
from serve.cow_browser_fixture import CowBrowserFixtureServer  # noqa: E402


class CowMultiClientTests(unittest.TestCase):
    def _seed_active_contexts(self, fixture, count=3):
        namespace = fixture.coordinator.store.create_namespace('multi-client-test', 'multi-ns').namespace_id
        branches = []
        model = fixture.models[0]
        for index in range(count):
            branch = f'ContextA{index}'
            branches.append(branch)
            model.plans[f'multi-answer-{index}'] = (257,)
            root = fixture.text_adapter.prepare(branch,
                [{'role': 'user', 'content': f'independent-context-{index}'}],
                f'multi-root-{index}', namespace_id=namespace)
            root.durable.result(5)
            root_head = fixture.coordinator.store.head(branch)
            answer = fixture.text_adapter.answer(branch, root_head.revision,
                f'multi-answer-{index}', 1)
            answer.durable.result(5)
            final = fixture.coordinator.store.head(branch)
            self.assertEqual(final.root, root_head.root)
            self.assertNotEqual(final.tip, root_head.tip)
            self.assertEqual(fixture.coordinator.store.get_snapshot(final.tip)['controller']['status'],
                             'active')
        return branches

    def test_three_real_http_clients_queue_on_one_worker_and_preserve_sources(self):
        with tempfile.TemporaryDirectory(prefix='cow-multi-client-') as temp:
            run_dir = Path(temp) / 'run'
            run_dir.mkdir(mode=0o700)
            fixture = CowBrowserFixtureServer().start()
            try:
                branches = self._seed_active_contexts(fixture)
                source_heads = {branch: fixture.coordinator.store.head(branch)
                                for branch in branches}
                fixture._gate.clear()

                def release_after_observed_queue(status):
                    active = any(type(value) is int and value > 0
                                 for value in status.get('active_per_gpu', {}).values())
                    if status.get('queued', 0) > 0 and active:
                        fixture._gate.set()

                result = run_concurrent_clients(base_url=fixture.base_url,
                    source_branches=branches, store=fixture.store_path, run_dir=run_dir,
                    max_tokens=2, timeout=10, sample_interval=.005,
                    synthetic_fixture=True, status_observer=release_after_observed_queue,
                    tokenizer=fixture.text_adapter.tokenizer)

                self.assertEqual(result['status'], 'PASS')
                self.assertEqual(result['mode'], 'synthetic')
                self.assertEqual(result['numerical_qualification']['status'], 'NOT_PERFORMED')
                self.assertGreaterEqual(result['positive_queue_overlap_samples'], 1)
                self.assertEqual(result['observed_max_active_per_gpu'], {'0': 1})
                self.assertEqual(result['client_count'], 3)
                for branch in branches:
                    self.assertEqual(fixture.coordinator.store.head(branch), source_heads[branch])
                for row in result['clients'].values():
                    self.assertEqual(row['receipt']['revision'], 1)
                    self.assertEqual(row['counter_deltas']['forward_tokens'], len(row['output_tokens']))
                    self.assertTrue(Path(row['stream_path']).is_file())
                    self.assertGreaterEqual(len(row['output_tokens']), 1)
                    self.assertEqual(row['canonical_validation']['controller_validation'],
                                     'passed_full_state_reconstruction')
            finally:
                fixture._gate.set()
                fixture.close()

    def test_canonical_store_rejects_wrong_parent_and_output_ids_with_same_root(self):
        with tempfile.TemporaryDirectory(prefix='cow-multi-canonical-') as temp:
            fixture = CowBrowserFixtureServer(store_path=Path(temp) / 'store').start()
            try:
                branch = self._seed_active_contexts(fixture, count=1)[0]
                store = fixture.coordinator.store
                source = store.head(branch).tip
                source_record = store.get_snapshot(source)
                root, source_ids = store.lineage(source)
                root_record = store.get_snapshot(root)
                tokenizer = fixture.text_adapter.tokenizer
                output = [257]
                expected_controller = consume_many(source_record['controller'], [
                    (257, tokenizer.token_bytes(257), False)])
                valid_final = store.append(source, output, expected_controller)
                verified = _verify_canonical_snapshot(store, source, valid_final, output, tokenizer)
                self.assertEqual(verified['parent'], source)
                self.assertEqual(verified['source_root'], root)
                self.assertEqual(verified['controller_validation'], 'passed_full_state_reconstruction')

                wrong_parent = store.append(root,
                    source_ids[len(root_record['inputs']):] + output, expected_controller)
                self.assertEqual(store.lineage(wrong_parent)[0], root)
                plausible_receipt = dict(snapshot=wrong_parent,
                    branch=dict(name=branch, revision=3, tip=wrong_parent, root=root))
                self.assertEqual(plausible_receipt['branch']['tip'], plausible_receipt['snapshot'])
                self.assertEqual(plausible_receipt['branch']['root'], root)
                with self.assertRaisesRegex(RecoveryFailure, 'parent differs'):
                    _verify_canonical_snapshot(store, source, wrong_parent, output, tokenizer)
                with self.assertRaisesRegex(RecoveryFailure, 'suffix differs'):
                    _verify_canonical_snapshot(store, source, valid_final, [258], tokenizer)
            finally:
                fixture.close()

    def test_concurrency_guard_rejects_negative_and_over_one_maxima(self):
        final = dict(active_per_gpu={'0': 0}, max_active_per_gpu={'0': 0}, queued=0)
        negative = [dict(active_per_gpu={'0': 1}, max_active_per_gpu={'0': -1}, queued=2)]
        with self.assertRaisesRegex(RecoveryFailure, 'one-worker limit'):
            _concurrency_evidence(negative, final, synthetic_fixture=True)
        over_one = [dict(active_per_gpu={'0': 1}, max_active_per_gpu={'0': 2}, queued=2)]
        with self.assertRaisesRegex(RecoveryFailure, 'one-worker limit'):
            _concurrency_evidence(over_one, final, synthetic_fixture=True)

    def test_requires_three_or_four_distinct_source_branches(self):
        with tempfile.TemporaryDirectory(prefix='cow-multi-validation-') as temp:
            run_dir = Path(temp)
            run_dir.mkdir(exist_ok=True)
            with self.assertRaisesRegex(ValueError, 'three or four'):
                run_concurrent_clients(base_url='http://127.0.0.1:1',
                    source_branches=['A', 'B'], store=run_dir, run_dir=run_dir)
            with self.assertRaisesRegex(ValueError, 'distinct'):
                run_concurrent_clients(base_url='http://127.0.0.1:1',
                    source_branches=['A', 'B', 'A'], store=run_dir, run_dir=run_dir)


if __name__ == '__main__':
    unittest.main()
