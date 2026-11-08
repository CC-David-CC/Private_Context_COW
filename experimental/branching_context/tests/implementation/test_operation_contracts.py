"""Independent operation admission: immutable inputs and no implied support."""
from dataclasses import FrozenInstanceError, replace
import unittest
from context_time_travel.operation_contracts import (
    AdmissionError, ContractError, EvidenceReference, NumericLimit,
    OperationDeclaration, OperationRequest, Quantity, OPERATIONS, admit,
)

MODEL = 'a' * 64
PROFILE = 'b' * 64


def declaration(operation='inference', **overrides):
    values = dict(operation=operation, implementation_version='native-v1',
        implementation_status='implemented', qualification_status='unqualified',
        policy_enabled=True, model_id=MODEL, profile_id=PROFILE, codec=None,
        formats=('Q4_K',), devices=('CUDA:sm89',),
        limits=(NumericLimit('context_tokens', 1, 2048),),
        prerequisites=('weights_verified',),
        ownership='caller owns plan; worker owns native lease',
        cancellation='before dispatch: no mutation; after dispatch: consumed boundary',
        evidence=())
    values.update(overrides)
    return OperationDeclaration(**values)


def request(operation='inference', **overrides):
    values = dict(operation=operation, model_id=MODEL, profile_id=PROFILE, codec=None,
        format='Q4_K', device='CUDA:sm89', limits=(Quantity('context_tokens', 128),),
        prerequisites=('weights_verified',))
    values.update(overrides)
    return OperationRequest(**values)


def qualify(d):
    evidence = EvidenceReference('receipts/fixture.json', 'd' * 64, 'e' * 40,
                                 d.scope_sha256)
    return replace(d, qualification_status='qualified', evidence=(evidence,))


class OperationContractTests(unittest.TestCase):
    def reject(self, declarations, req, code, **kwargs):
        with self.assertRaises(AdmissionError) as caught:
            admit(declarations, req, **kwargs)
        self.assertEqual(caught.exception.code, code)
        self.assertTrue(caught.exception.reason)
        self.assertTrue(caught.exception.next_action)
        self.assertEqual(caught.exception.operation, req.operation)
        return caught.exception

    def test_seven_operations_are_independent(self):
        self.assertEqual(set(OPERATIONS), {'inference', 'lora_inference', 'lora_training',
            'full_training', 'optimizer_resume', 'checkpoint_restore', 'reconstruction'})
        for operation in OPERATIONS:
            d = qualify(declaration(operation))
            self.assertEqual(admit((d,), request(operation)).declaration, d)
            for other in set(OPERATIONS) - {operation}:
                self.reject((d,), request(other), 'operation_undeclared')

    def test_default_refuses_unqualified_and_opt_in_does_not_promote(self):
        d = declaration()
        self.reject((d,), request(), 'operation_unqualified')
        plan = admit((d,), request(), allow_unqualified=True)
        self.assertEqual(plan.qualification_status, 'unqualified')
        self.assertTrue(plan.unqualified_opt_in)
        self.assertEqual(d.qualification_status, 'unqualified')

    def test_unknown_requested_operation_is_structured(self):
        self.reject((qualify(declaration()),), request('magic_training'), 'operation_unknown')

    def test_unknown_implementation_differs_from_absent(self):
        for status, code in [('unknown', 'implementation_unknown'), ('absent', 'implementation_absent')]:
            self.reject((declaration(implementation_status=status),), request(), code,
                        allow_unqualified=True)

    def test_policy_disabled_even_when_qualified_or_opted_in(self):
        d = replace(qualify(declaration()), policy_enabled=False)
        self.reject((d,), request(), 'operation_policy_disabled', allow_unqualified=True)

    def test_exact_identity_and_optional_codec(self):
        d = qualify(declaration())
        self.assertIsNone(admit((d,), request()).request.codec)
        for name, value in [('model_id', 'f' * 64), ('profile_id', 'f' * 64), ('codec', 'state-v1')]:
            self.reject((d,), request(**{name: value}), 'operation_identity_mismatch')
        restore = qualify(declaration('checkpoint_restore', codec='state-v1'))
        admit((restore,), request('checkpoint_restore', codec='state-v1'))
        self.reject((restore,), request('checkpoint_restore'), 'operation_identity_mismatch')

    def test_format_and_device_refused_without_fallback(self):
        d = qualify(declaration())
        self.reject((d,), request(format='F16'), 'operation_format_mismatch')
        self.reject((d,), request(device='CPU'), 'operation_device_mismatch')

    def test_limits_require_explicit_values_and_reject_extras(self):
        d = qualify(declaration())
        self.reject((d,), request(limits=()), 'operation_limit_missing')
        self.reject((d,), request(limits=(Quantity('context_tokens', 128), Quantity('rank', 16))),
                    'operation_limit_unknown')
        for value in (0, 2049):
            self.reject((d,), request(limits=(Quantity('context_tokens', value),)),
                        'operation_limit_exceeded')
        for value in (1, 2048):
            admit((d,), request(limits=(Quantity('context_tokens', value),)))

    def test_integer_counts_and_continuous_limits_are_distinct(self):
        self.reject((qualify(declaration()),), request(limits=(Quantity('context_tokens', 1.5),)),
                    'operation_limit_type_mismatch')
        d = qualify(declaration(limits=(NumericLimit('scale', 0.0, 2.0, 'number'),)))
        admit((d,), request(limits=(Quantity('scale', 0.5),)))
        self.reject((qualify(declaration()),), request(limits=(Quantity('context_tokens', 10**400),)),
                    'operation_limit_exceeded')

    def test_prerequisites_checked(self):
        self.reject((qualify(declaration()),), request(prerequisites=()),
                    'operation_prerequisite_missing')

    def test_multiple_profiles_select_exactly(self):
        one = qualify(declaration())
        two = qualify(declaration(profile_id='c' * 64))
        self.assertEqual(admit((one, two), request(profile_id='c' * 64)).declaration, two)

    def test_duplicate_identity_is_rejected_not_first_match(self):
        d = qualify(declaration())
        with self.assertRaises(ContractError):
            admit((d, d), request())
        with self.assertRaises(ContractError):
            admit((d, replace(d, policy_enabled=False)), request())

    def test_qualified_requires_matching_evidence_scope(self):
        with self.assertRaises(ContractError):
            replace(declaration(), qualification_status='qualified')
        d = qualify(declaration())
        for change in ({'model_id': 'c' * 64}, {'codec': 'other-v1'},
                       {'limits': (NumericLimit('context_tokens', 1, 4096),)},
                       {'ownership': 'another owner'}, {'operation': 'full_training'}):
            with self.assertRaises(ContractError):
                replace(d, **change)
        # A policy decision is not a numerical/evidence scope change.
        self.assertEqual(replace(d, policy_enabled=False).scope_sha256, d.scope_sha256)

    def test_unknown_cannot_claim_qualified(self):
        d = qualify(declaration())
        for status in ('unknown', 'absent'):
            with self.assertRaises(ContractError):
                replace(d, implementation_status=status)

    def test_containers_and_records_are_deeply_immutable(self):
        d = qualify(declaration())
        with self.assertRaises(FrozenInstanceError):
            d.policy_enabled = False
        with self.assertRaises(FrozenInstanceError):
            d.limits[0].maximum = 9999
        with self.assertRaises(FrozenInstanceError):
            d.evidence[0].scope_sha256 = '0' * 64
        with self.assertRaises(FrozenInstanceError):
            admit((d,), request()).request.device = 'CPU'
        for field, value in [('formats', ['Q4_K']), ('limits', [NumericLimit('x', 0, 1)]),
                             ('evidence', []), ('prerequisites', ['weights_verified'])]:
            with self.assertRaises(ContractError):
                declaration(**{field: value})
        with self.assertRaises(ContractError):
            admit([d], request())

    def test_bad_numeric_values_never_bypass_comparisons(self):
        for value in (True, float('nan'), float('inf'), -float('inf'), '128'):
            with self.assertRaises(ContractError):
                Quantity('context_tokens', value)
            with self.assertRaises(ContractError):
                NumericLimit('context_tokens', 1, value)
        with self.assertRaises(ContractError):
            NumericLimit('context_tokens', 2, 1)
        with self.assertRaises(ContractError):
            request(limits=(Quantity('context_tokens', 1), Quantity('context_tokens', 2)))

    def test_invalid_schema_fields_rejected(self):
        for change in ({'policy_enabled': 1}, {'implementation_status': 'yes'},
                       {'qualification_status': 'startup-checked'}, {'operation': 'magic'},
                       {'model_id': 'unbound'}, {'ownership': ''}, {'formats': ('Q4_K', 'Q4_K')}):
            with self.assertRaises(ContractError):
                declaration(**change)
        with self.assertRaises(ContractError):
            admit((declaration(),), request(), allow_unqualified='yes')

    def test_plan_retains_exact_request_and_scope(self):
        d = qualify(declaration())
        r = request()
        plan = admit((d,), r)
        self.assertIs(plan.declaration, d)
        self.assertIs(plan.request, r)
        self.assertEqual(plan.scope_sha256, d.scope_sha256)
        self.assertFalse(plan.unqualified_opt_in)

    def test_scope_hash_is_deterministic_and_not_a_verification_claim(self):
        d = declaration()
        self.assertEqual(d.scope_sha256, declaration().scope_sha256)
        self.assertNotEqual(d.scope_sha256, declaration(implementation_version='native-v2').scope_sha256)
        # A structurally valid evidence reference grants no file IO or execution.
        evidence = EvidenceReference('not-a-real-file.json', '1' * 64, '2' * 40, d.scope_sha256)
        plan = admit((replace(d, qualification_status='qualified', evidence=(evidence,)),), request())
        self.assertEqual(plan.qualification_status, 'qualified')
        self.assertFalse(plan.evidence_verified)


if __name__ == '__main__':
    unittest.main()
