"""Capability claims are bounded by the verified native protocol, not a model name."""
import unittest
from context_time_travel.capabilities import describe_dense, describe_strata, CapabilityError


def profile(backend='qwen35-llama.cpp'):
    return dict(backend=backend, model_contract_id='a'*64, execution_profile_id='b'*64,
        execution=dict(capacity=4096, state_format='qwen35-hybrid-frontier-v1',
            resume_contract='identical-native-batches-v1',
            generated_operator='llama-forward-consumed-v1',
            selected_token_protocol='consumed-selected-receipt-v1',
            sampling_protocol='min-p-top-k-top-p-u53-v2'))


class BackendCapabilityTests(unittest.TestCase):
    def test_dense_reports_serialized_hybrid_state_without_claiming_native_sharing(self):
        value = describe_dense(profile()).to_json()
        self.assertEqual(value['family'], 'dense')
        self.assertEqual(value['inputs'], ['text'])
        self.assertTrue(value['checkpoint_export'])
        self.assertTrue(value['checkpoint_import'])
        self.assertFalse(value['native_residency'])
        self.assertFalse(value['physical_prefix_sharing'])
        self.assertEqual(value['worker_topology'], 'independent')
        self.assertEqual(value['max_context'], 4096)

    def test_unknown_dense_codec_or_boundary_is_rejected(self):
        for key in ('state_format', 'resume_contract', 'selected_token_protocol'):
            value = profile(); value['execution'][key] = 'unknown'
            with self.subTest(key=key), self.assertRaises(CapabilityError):
                describe_dense(value)

    def test_moe_claims_require_matching_native_handshake(self):
        value = profile('strata')
        value['execution'].update(resume_contract='identical-native-operators-v1',
            generated_operator='normal-verifier-decode',
            native_request_protocol='consumed-greedy-ack-v1',
            grammar_selection_protocol='finite-mask-argmax-v1')
        ready = dict(native_request_protocol='consumed-greedy-ack-v1',
            selected_token_protocol='consumed-selected-receipt-v1',
            grammar_selection_protocol='finite-mask-argmax-v1',
            sampling_protocol='min-p-top-k-top-p-u53-v2')
        report = describe_strata(value, ready).to_json()
        self.assertEqual(report['family'], 'moe')
        self.assertTrue(report['native_residency'])
        self.assertFalse(report['physical_prefix_sharing'])
        for key in ready:
            broken = dict(ready); broken[key] = 'unknown'
            with self.subTest(key=key), self.assertRaises(CapabilityError):
                describe_strata(value, broken)

    def test_reports_are_detached_and_do_not_contain_host_paths(self):
        value = profile(); value['model_path'] = '/private/weights.gguf'
        report = describe_dense(value)
        first = report.to_json(); first['inputs'].append('image')
        self.assertEqual(report.to_json()['inputs'], ['text'])
        self.assertNotIn('model_path', report.to_json())


if __name__ == '__main__': unittest.main()
