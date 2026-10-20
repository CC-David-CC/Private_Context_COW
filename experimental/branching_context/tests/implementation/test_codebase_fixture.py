"""Exact token partition controls; these do not qualify model inference."""
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts'))
import build_codebase_fixture as fixture


class CodebaseFixtureTests(unittest.TestCase):
    def _three_contexts(self):
        return dict(model_contract_id='contract-v1', contexts=[
            dict(name='coordinator', prefix=[10,11,12], A=[20], B=[21,22],
                 source_files=['experimental/branching_context/src/context_time_travel/coordinator.py'],
                 source_receipts=[dict(path='coordinator.py', sha256='a'*64)]),
            dict(name='model', prefix=[30,31,32,33], A=[40,41], B=[42],
                 source_files=['experimental/branching_context/src/context_time_travel/real.py']),
            dict(name='store', prefix=[50,51], A=[60], B=[61,62],
                 source_files=['experimental/branching_context/src/context_time_travel/store.py']),
        ])

    def test_normalizes_three_real_contexts_to_existing_pool_fixture_shape(self):
        source = self._three_contexts()
        result = fixture.normalize_context_fixture(source, context_capacity=16, output_limit=4)
        self.assertEqual(result['prefixes'], [[10,11,12], [30,31,32,33], [50,51]])
        self.assertEqual(result['suffixes'], [[20], [21,22], [40,41], [42], [60], [61,62]])
        self.assertEqual(result['tree_roots'], [0,0,1,1,2,2])
        self.assertEqual([row['name'] for row in result['context_metadata']],
                         ['coordinator','model','store'])
        self.assertEqual(result['context_metadata'][0]['source_files'],
                         source['contexts'][0]['source_files'])
        self.assertEqual(result['context_metadata'][0]['source_receipts'],
                         source['contexts'][0]['source_receipts'])

    def test_rejects_wrong_context_count_duplicate_prefixes_and_identical_questions(self):
        source = self._three_contexts()
        for contexts in (source['contexts'][:2], source['contexts'] + [source['contexts'][0]]):
            with self.subTest(count=len(contexts)), self.assertRaises(ValueError):
                fixture.normalize_context_fixture(dict(contexts=contexts), 16, 4)
        duplicate = self._three_contexts()
        duplicate['contexts'][1]['prefix'] = list(duplicate['contexts'][0]['prefix'])
        with self.assertRaises(ValueError):
            fixture.normalize_context_fixture(duplicate, 16, 4)
        same_questions = self._three_contexts()
        same_questions['contexts'][1]['B'] = list(same_questions['contexts'][1]['A'])
        with self.assertRaises(ValueError):
            fixture.normalize_context_fixture(same_questions, 16, 4)

    def test_rejects_bool_noninteger_negative_empty_and_overcapacity_tokens(self):
        for bad in ([], [True], [1.0], ['1'], [-1]):
            with self.subTest(bad=bad):
                source = self._three_contexts()
                source['contexts'][0]['A'] = bad
                with self.assertRaises(ValueError):
                    fixture.normalize_context_fixture(source, 16, 4)
        source = self._three_contexts()
        source['contexts'][0]['prefix'] = [1] * 12
        source['contexts'][0]['A'] = [2]
        with self.assertRaises(ValueError):
            fixture.normalize_context_fixture(source, 16, 4)

    def test_capacity_boundary_is_inclusive_and_limits_are_validated(self):
        source = self._three_contexts()
        source['contexts'][0]['prefix'] = [1] * 10
        source['contexts'][0]['A'] = [2,3]
        source['contexts'][0]['B'] = [4,5]
        self.assertEqual(len(fixture.normalize_context_fixture(source, 14, 2)['prefixes'][0]), 10)
        with self.assertRaises(ValueError):
            fixture.normalize_context_fixture(source, 13, 2)
        for capacity, output in ((True, 2), (0, 2), (14, False), (14, 0)):
            with self.subTest(capacity=capacity, output=output), self.assertRaises(ValueError):
                fixture.normalize_context_fixture(self._three_contexts(), capacity, output)


    def test_partition_reconstructs_both_complete_tokenized_requests(self):
        a,b = [10,11,12,13],[10,11,14,15,16]
        result = fixture.split_histories(a,b,2)
        self.assertEqual(result,dict(prefix=[10,11],A=[12,13],B=[14,15,16]))
        self.assertEqual(result['prefix']+result['A'],a)
        self.assertEqual(result['prefix']+result['B'],b)

    def test_text_similarity_cannot_override_different_token_prefixes(self):
        with self.assertRaises(ValueError): fixture.split_histories([10,11,12],[10,14,12],2)

    def test_missing_distinct_continuations_and_invalid_prefix_are_rejected(self):
        for a,b,n in [([1,2],[1,2],1),([1,2],[1,2,3],2),([1,2],[1,3],0),
                      ([1,2],[1,3],True),([1,2],[1,3],3)]:
            with self.subTest(a=a,b=b,n=n),self.assertRaises(ValueError):
                fixture.split_histories(a,b,n)


if __name__ == '__main__': unittest.main()
