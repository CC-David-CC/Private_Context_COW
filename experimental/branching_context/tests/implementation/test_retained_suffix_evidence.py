"""Historical retained Q8 evidence; prerequisites selected by cow_campaign."""
import json
import os
from pathlib import Path
import unittest
from test_suffix_frontier_audit import (
    load_suffix_reference, _small_store_fingerprint, _runtime_import_witness,
    preflight_suffix_audit,
)

class RetainedSuffixEvidenceTests(unittest.TestCase):
    def test_retained_q8_case_has_bound_suffix_frontiers(self):
        case = Path(os.environ['COW_SUFFIX_CASE'])
        profile = json.loads(Path(os.environ['COW_SUFFIX_PROFILE']).read_text())
        result = load_suffix_reference(case, profile)
        self.assertIn('root_save', json.loads((case / 'prepare.json').read_text()))
        metadata = _small_store_fingerprint(case / 'store', result['root_snapshot'])
        self.assertIn('references/prefix.json', metadata)
        self.assertIn('references', metadata['$directories'])
        self.assertIn('checkpoints/' + result['root_snapshot'] + '.json', metadata)
        self.assertNotIn('checkpoints/' + json.loads((case / 'prepare.json').read_text())[
            'parent_payload_sha256'] + '.state', metadata)
        runtime = _runtime_import_witness(os.environ['COW_SUFFIX_API_SOURCE'], profile)
        self.assertEqual(len(runtime), 7)
        preflight = preflight_suffix_audit(
            os.environ['COW_SUFFIX_PROFILE'], case,
            api_source=os.environ['COW_SUFFIX_API_SOURCE'])
        self.assertEqual(preflight['status'], 'CPU_PREFLIGHT_PASS')
        self.assertFalse(preflight['parent_payload']['content_sha256_verified'])
        self.assertEqual(result['requests'], [65536, 8335])
        self.assertEqual(result['branches']['A']['suffix_frontier_sha256'],
                         '205c9d07b30f1d86b55236a60cef0724d5ca2f03485737ba5e23cda51c3342c3')
        self.assertEqual(result['branches']['B']['suffix_frontier_sha256'],
                         'c1fb87baf8322da1905a0f53770936775df856e5cc19488204e16cb405b271f0')

