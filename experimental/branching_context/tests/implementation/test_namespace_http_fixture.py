"""CPU-only source excerpt selection tests for pinned-GGUF HTTP fixtures."""
import tempfile
import unittest
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parents[2] / 'scripts'
sys.path.insert(0, str(SCRIPT_DIR))
from build_namespace_http_fixture import extract_method, profile_model_variant, source_bundle


class NamespaceHTTPFixtureTests(unittest.TestCase):
    def test_model_variant_comes_from_pinned_profile_path(self):
        self.assertEqual(profile_model_variant({
            'model_path': '/models/UD-Q4_K_XL/model.gguf'}), 'UD-Q4_K_XL')
        self.assertEqual(profile_model_variant({
            'model_path': '/models/Q8_0/model.gguf'}), 'Q8_0')

    def test_actual_source_bundle_is_prioritized_and_contains_fork_projection_seam(self):
        repo_root = Path(__file__).resolve().parents[4]
        body, receipts = source_bundle(repo_root)
        self.assertEqual([item['path'] for item in receipts], [
            'experimental/branching_context/src/context_time_travel/coordinator.py',
            'serve/cow_workspace.py',
        ])
        for marker in ('source_branch', 'source_revision', 'source_tip',
                       'presentation_catalog', 'source_events'):
            self.assertIn(marker, body)
        self.assertNotIn('.github/FUNDING.yml', body)
        self.assertTrue(all(item['line_start'] > 0 and item['line_end'] >= item['line_start']
                            for item in receipts))

    def test_method_excerpt_is_exact_and_file_hash_covers_the_source(self):
        with tempfile.TemporaryDirectory(prefix='cow-fixture-source-') as directory:
            path = Path(directory) / 'sample.py'
            path.write_text('class Example:\n    def selected(self):\n        return "source_tip"\n',
                            encoding='utf-8')
            excerpt, receipt = extract_method(directory, 'sample.py', 'Example', 'selected')
            self.assertEqual(excerpt, '    def selected(self):\n        return "source_tip"\n')
            self.assertEqual((receipt['line_start'], receipt['line_end']), (2, 3))
            self.assertEqual(receipt['source_file_bytes'], path.stat().st_size)
            self.assertEqual(len(receipt['source_file_sha256']), 64)

    def test_missing_source_method_fails_closed(self):
        with tempfile.TemporaryDirectory(prefix='cow-fixture-missing-') as directory:
            (Path(directory) / 'sample.py').write_text('class Example:\n    pass\n', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'expected one Example.absent'):
                extract_method(directory, 'sample.py', 'Example', 'absent')


if __name__ == '__main__':
    unittest.main()
