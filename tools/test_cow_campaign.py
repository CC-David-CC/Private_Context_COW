"""Test selection must fail closed before importing an inapplicable native test."""
import tempfile
from pathlib import Path
import unittest
from tools.cow_campaign import classify, validate_manifest


class CampaignTests(unittest.TestCase):
    def test_unknown_and_removed_modules_require_a_manifest_update(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root/'test_new.py').write_text('raise RuntimeError("must not import")')
            with self.assertRaisesRegex(ValueError, 'Unclassified'):
                validate_manifest(root, {})
            with self.assertRaisesRegex(ValueError, 'Missing'):
                validate_manifest(root, {'test_new': {}, 'test_removed': {}})

    def test_missing_native_prerequisite_is_blocked_not_skipped_or_passed(self):
        row = dict(campaign='moe-native', requires=['CTT_PROFILE'])
        self.assertEqual(classify(row, 'moe-native', {})['status'], 'BLOCKED')
        self.assertEqual(classify(row, 'cpu', {})['status'], 'NOT_APPLICABLE')
        self.assertEqual(classify(row, 'moe-native', {'CTT_PROFILE':'profile.json'})['status'], 'READY')


if __name__ == '__main__': unittest.main()
