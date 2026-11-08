"""Construction selection tests; no native numerical qualification is claimed."""
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch
from context_time_travel import backend_registry as registry
from context_time_travel.real import BranchingModel
from context_time_travel.store import IntegrityError

class BackendRegistryTests(unittest.TestCase):
    def test_unknown_backend_rejects_before_import_or_allocation(self):
        with patch.object(registry, 'import_module') as imports:
            with self.assertRaises(IntegrityError): registry.load_backend({'backend': 'not-registered'})
            imports.assert_not_called()

    def test_builtin_selection_preserves_profile_and_worker(self):
        names = {'qwen35-llama.cpp': 'load_qwen35_backend', 'llama-dense-llama.cpp': 'LlamaDenseBackend', 'strata': 'StrataBackend', 'llama.cpp': 'NativeBackend'}
        for name, constructor_name in names.items():
            with self.subTest(name=name):
                profile = {'backend': name, 'model_contract_id': 'unchanged'}
                constructor = Mock(return_value=object())
                with patch.object(registry, 'import_module', return_value=SimpleNamespace(**{constructor_name: constructor})):
                    result = registry.load_backend(profile)
                constructor.assert_called_once_with(profile)
                self.assertIs(result, constructor.return_value)
                self.assertEqual(profile, {'backend': name, 'model_contract_id': 'unchanged'})
                if name != 'llama.cpp':
                    constructor.reset_mock()
                    with patch.object(registry, 'import_module', return_value=SimpleNamespace(**{constructor_name: constructor})):
                        registry.load_backend(profile, gpu_id=4)
                    constructor.assert_called_once_with(profile, gpu_id=4)

    def test_legacy_default_is_explicit_and_invalid_values_do_not_fall_through(self):
        constructor = Mock()
        with patch.object(registry, 'import_module', return_value=SimpleNamespace(NativeBackend=constructor)):
            registry.load_backend({})
        constructor.assert_called_once_with({})
        for value in (None, '', False, [], 'STRATA'):
            with self.subTest(value=value), self.assertRaises(IntegrityError): registry.load_backend({'backend': value})

    def test_legacy_native_worker_selection_rejects_before_import(self):
        with patch.object(registry, 'import_module') as imports:
            with self.assertRaises(IntegrityError): registry.load_backend({'backend': 'llama.cpp'}, gpu_id=3)
            imports.assert_not_called()

    def test_custom_definition_extends_construction_without_global_mutation(self):
        factory = Mock(return_value='custom')
        local = registry.BackendRegistry((registry.BackendDefinition('fixture-model', factory),))
        profile = {'backend': 'fixture-model'}
        self.assertEqual(local.construct(profile, gpu_id=2), 'custom')
        factory.assert_called_once_with(profile, 2)
        with self.assertRaises(IntegrityError): registry.load_backend(profile)

    def test_duplicate_registration_and_mutable_definition_inputs_reject(self):
        entry = registry.BackendDefinition('fixture', lambda profile, gpu: None)
        with self.assertRaises(ValueError): registry.BackendRegistry((entry, entry))
        with self.assertRaises(TypeError): registry.BackendRegistry([entry])
        with self.assertRaises(ValueError): registry.BackendDefinition('', lambda p, g: None)
        with self.assertRaises(TypeError): registry.BackendDefinition('fixture', None)

    def test_branching_model_uses_shared_constructor(self):
        profile = {'backend': 'strata'}
        with patch.object(registry, 'load_backend', return_value='constructed') as load:
            self.assertEqual(BranchingModel.load_backend(profile), 'constructed')
        load.assert_called_once_with(profile)

    def test_constructor_error_propagates_without_fallback(self):
        for error in (IntegrityError('bad profile'), RuntimeError('native load failed')):
            with self.subTest(error=type(error).__name__):
                constructor = Mock(side_effect=error)
                with patch.object(registry, 'import_module', return_value=SimpleNamespace(StrataBackend=constructor)) as imports:
                    with self.assertRaises(type(error)) as caught:
                        registry.load_backend({'backend': 'strata'}, gpu_id=0)
                self.assertIs(caught.exception, error)
                self.assertEqual(imports.call_count, 1)
                constructor.assert_called_once()

    def test_non_mapping_profile_rejects_before_import(self):
        for profile in (None, [], 'strata', False):
            with self.subTest(profile=profile), patch.object(registry, 'import_module') as imports:
                with self.assertRaises(IntegrityError): registry.load_backend(profile)
                imports.assert_not_called()

if __name__ == '__main__': unittest.main()
