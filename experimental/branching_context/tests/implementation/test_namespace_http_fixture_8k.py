"""CPU source-selection tests for the expanded namespace HTTP fixture."""
import hashlib
import sys
import unittest
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parents[2] / 'scripts'
sys.path.insert(0, str(SCRIPT_DIR))
from build_namespace_http_fixture import REQUIRED_MARKERS, profile_model_variant
from build_namespace_http_fixture_8k import SOURCE_METHODS_8K, QUESTION_A, QUESTION_B
from build_namespace_http_fixture import source_bundle


class NamespaceHTTPFixture8KTests(unittest.TestCase):
    def test_quantization_label_is_read_from_the_selected_profile(self):
        self.assertEqual(profile_model_variant({
            'model_path': '/models/UD-Q4_K_XL/model.gguf'}), 'UD-Q4_K_XL')
        self.assertEqual(profile_model_variant({
            'model_path': '/models/Q8_0/model.gguf'}), 'Q8_0')

    def test_expanded_bundle_is_request_publication_and_workspace_code_only(self):
        repo_root = Path(__file__).resolve().parents[4]
        body, receipts = source_bundle(repo_root, SOURCE_METHODS_8K, REQUIRED_MARKERS)
        self.assertEqual(len(receipts), len(SOURCE_METHODS_8K))
        self.assertTrue(any(item['path'].endswith('/coordinator.py') for item in receipts))
        self.assertTrue(any(item['path'].endswith('/store.py') for item in receipts))
        self.assertTrue(any(item['path'] == 'serve/cow_adapter.py' for item in receipts))
        self.assertTrue(any(item['path'] == 'serve/cow_text_adapter.py' for item in receipts))
        self.assertTrue(any(item['path'] == 'serve/cow_workspace.py' for item in receipts))
        for marker in REQUIRED_MARKERS:
            self.assertIn(marker.casefold(), body.casefold())
        self.assertNotIn('.github/FUNDING.yml', body)
        self.assertIn('ContextCoordinator._submit', body)
        self.assertIn('Store.presentation_catalog', body)
        self.assertIn('CowWorkspace._events_on_ancestry', body)

    def test_questions_are_distinct_source_review_tasks_with_boundary_requirements(self):
        self.assertNotEqual(QUESTION_A, QUESTION_B)
        self.assertIn('captured revision', QUESTION_A)
        self.assertIn('uncertain publication', QUESTION_A)
        self.assertIn('source-less fork', QUESTION_B)
        self.assertIn('selected canonical ancestry', QUESTION_B)
        self.assertIn('controller is still active', QUESTION_B)

    def test_excerpt_receipts_bind_exact_source_bytes_and_ranges(self):
        repo_root = Path(__file__).resolve().parents[4]
        _, receipts = source_bundle(repo_root, SOURCE_METHODS_8K, REQUIRED_MARKERS)
        for receipt in receipts:
            raw = Path(receipt['absolute_path']).read_bytes()
            self.assertEqual(hashlib.sha256(raw).hexdigest(), receipt['source_file_sha256'])
            self.assertLessEqual(receipt['line_start'], receipt['line_end'])
            self.assertGreater(receipt['source_file_bytes'], 0)


if __name__ == '__main__':
    unittest.main()
