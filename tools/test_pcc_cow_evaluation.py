"""Adapter fault tests; native correctness is tested separately on real GPUs."""
import unittest
import tempfile
import hashlib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from tools.pcc_cow_evaluation import CowEvaluationSession, verify_sources


class CowExecutionTests(unittest.TestCase):
    def source_fixture(self, base):
        names = ['tools/pcc_cow_evaluation.py',
            'experimental/branching_context/src/context_time_travel/real.py',
            'experimental/branching_context/src/context_time_travel/qwen35_backend.py',
            'experimental/branching_context/src/context_time_travel/llama_dense_backend.py']
        files = {}
        for name in names:
            p = base/name; p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text('owned source\n'); files[name] = hashlib.sha256(p.read_bytes()).hexdigest()
        return {'native_build': {'selected_runtime_source_closure': {'files_sha256': files}}}

    def test_actual_source_corruption_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp); profile = self.source_fixture(base)
            verify_sources(profile, base)
            (base/'tools/pcc_cow_evaluation.py').write_text('changed source\n')
            with self.assertRaises(ValueError): verify_sources(profile, base)

    def test_missing_closure_and_outside_source_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); base = root/'runtime'; base.mkdir()
            profile = self.source_fixture(base)
            outside = root/'outside.py'; outside.write_text('outside\n')
            profile['native_build']['selected_runtime_source_closure']['files_sha256']['../outside.py'] = hashlib.sha256(outside.read_bytes()).hexdigest()
            with self.assertRaises(ValueError): verify_sources(profile, base)
            profile['native_build']['selected_runtime_source_closure']['files_sha256'] = {}
            with self.assertRaises(ValueError): verify_sources(profile, base)

    def session(self, *, stopped=True):
        session = CowEvaluationSession.__new__(CowEvaluationSession)
        session.closed = session.failed = False
        session.shared_root = None
        session.shared_prefix = ()
        session.requests = 0
        session.profile = dict(execution=dict(capacity=128, eos_tokens=[1]))
        counters = [dict(consumed_tokens=0, forward_calls=0),
                    dict(consumed_tokens=4, forward_calls=3)]
        session.backend = SimpleNamespace(vocab_size=8, work_counters=Mock(side_effect=counters))
        result = dict(snapshot='tail', tokens=[3, 1] if stopped else [3, 4], stopped=stopped)
        session.model = SimpleNamespace(create_with_reuse=Mock(return_value=('root', False)),
            run_greedy_request=Mock(return_value=result), history=Mock(return_value=(5, 6, *result['tokens'])),
            cache={'tail': SimpleNamespace(consumed=4)}, evict_all=Mock())
        return session

    def test_uses_cow_request_and_consumed_eos(self):
        session = self.session()
        result = session.complete([5, 6], maximum=8, stream='case-1')
        session.model.run_greedy_request.assert_called_once_with('root', (), 8, stream='case-1')
        session.model.evict_all.assert_called_once()
        self.assertEqual(result['stop_reason'], 'eos')
        self.assertEqual(result['counters']['consumed_tokens'], 4)

    def test_length_cutoff_keeps_final_consumed_token(self):
        result = self.session(stopped=False).complete([5, 6], maximum=2, stream='case-1')
        self.assertEqual(result['tokens'], [3, 4])
        self.assertEqual(result['stop_reason'], 'length')

    def test_bad_history_or_work_is_failure_and_not_retryable(self):
        for defect in ('history', 'work'):
            session = self.session()
            if defect == 'history': session.model.history.return_value = (5, 6, 3)
            else: session.backend.work_counters.side_effect = [
                dict(consumed_tokens=0), dict(consumed_tokens=3)]
            with self.subTest(defect=defect), self.assertRaises(ValueError):
                session.complete([5, 6], maximum=8, stream='case-1')
            with self.assertRaises(ValueError): session.complete([5, 6], maximum=8, stream='case-1')
            self.assertEqual(session.model.run_greedy_request.call_count, 1)
            session.model.evict_all.assert_called_once()

    def test_invalid_request_does_not_start_model_work(self):
        session = self.session()
        for prefix, maximum in [([], 8), ([True], 8), ([9], 8), ([5]*127, 2), ([5], 0)]:
            with self.subTest(prefix=prefix[:2], maximum=maximum), self.assertRaises(ValueError):
                session.complete(prefix, maximum=maximum, stream='case-1')
        session.model.create_with_reuse.assert_not_called()

    def test_shared_prefix_uses_existing_root_and_only_consumes_question_suffix(self):
        session = self.session()
        session.backend.work_counters.side_effect = [
            dict(consumed_tokens=0, forward_calls=0), dict(consumed_tokens=2, forward_calls=1),
            dict(consumed_tokens=2, forward_calls=1), dict(consumed_tokens=6, forward_calls=4)]
        session.model.cache['root'] = SimpleNamespace(consumed=2)
        session.model.cache['tail'].consumed = 6
        session.model.history.side_effect = [(5, 6), (5, 6, 7, 3, 3, 1)]
        setup = session.prepare_shared_prefix([5, 6])
        answer = session.complete([5, 6, 7, 3], maximum=8, stream='branch-1')
        session.model.create_with_reuse.assert_called_once_with((5, 6))
        session.model.run_greedy_request.assert_called_once_with('root', (7, 3), 8, stream='branch-1')
        self.assertEqual(set(session.model.cache), {'root'})
        self.assertEqual(setup['counters']['consumed_tokens'], 2)
        self.assertEqual(answer['counters']['consumed_tokens'], 4)
        self.assertEqual(answer['shared_prefix_tokens'], 2)
        session.model.evict_all.assert_not_called()

    def test_shared_prefix_mismatch_or_late_prepare_cannot_change_history(self):
        session = self.session(); session.shared_root = 'root'; session.shared_prefix = (5, 6)
        with self.assertRaises(ValueError): session.complete([5, 7], maximum=2, stream='wrong-branch')
        session.model.run_greedy_request.assert_not_called()
        with self.assertRaises(ValueError): session.prepare_shared_prefix([5, 6])
        session.shared_root = None; session.requests = 1
        with self.assertRaises(ValueError): session.prepare_shared_prefix([5, 6])

    def test_failure_releases_shared_materializations_without_retry(self):
        session = self.session(); session.shared_root = 'root'; session.shared_prefix = (5, 6)
        session.model.run_greedy_request.side_effect = RuntimeError('native request failed')
        with self.assertRaises(RuntimeError): session.complete([5, 6], maximum=2, stream='failed')
        session.model.evict_all.assert_called_once()
        with self.assertRaises(ValueError): session.complete([5, 6], maximum=2, stream='failed')
        self.assertEqual(session.model.run_greedy_request.call_count, 1)


if __name__ == '__main__': unittest.main()
