"""CPU durable-manifest controls, not real-GPU restart qualification."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from context_time_travel.real import BranchingModel
from context_time_travel.store import IntegrityError, _json_bytes
from context_time_travel.vision_execution import input_binding, POSITION_CONVENTION, RESUME_CONTRACT
from test_vision_api_binding import MixedCPUBackend, NoArtifacts
from test_vision_execution import spans


class DiskCPUBackend(MixedCPUBackend):
    def restore(self, lease, blob, *, history, frontier):
        value = super().restore(lease, blob, history=history, frontier=frontier)
        self.stats['loaded_file_bytes'] += len(blob)
        return value


class VisionDiskBindingTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.profile = dict(model_contract_id='mixed-model', execution_profile_id='mixed-exec',
            model_weights_sha256='a'*64, execution=dict(resume_contract=RESUME_CONTRACT,
                prefill_chunk=128, embedding_width=4, capacity=128))
        self.backend = DiskCPUBackend(self.profile)
        self.model = BranchingModel(self.tmp.name, self.backend, self.profile)
        self.root = self.model.create_inputs(spans(), artifacts=NoArtifacts(self.model.store))

    def fresh(self):
        backend = DiskCPUBackend(self.profile)
        return BranchingModel(self.tmp.name, backend, self.profile), backend

    def envelope(self, snapshot):
        path = self.model.checkpoint_directory/(snapshot+'.json')
        return path, json.loads(path.read_text())

    def alter(self, path, value):
        value['checksum'] = hashlib.sha256(_json_bytes(value['manifest'])).hexdigest()
        path.write_bytes(_json_bytes(value))

    def test_vc013_manifest_binds_full_inputs_positions_and_schedule(self):
        self.model.checkpoint(self.root)
        manifest = self.model.checkpoint_manifest(self.root)
        self.assertEqual(manifest['schema'], 4)
        self.assertEqual(manifest['positions'], POSITION_CONVENTION)
        for key, value in input_binding(self.model.history(self.root)).items():
            self.assertEqual(manifest[key], value)
        self.assertTrue(manifest['execution_schedule'])

    def test_vc013_new_adapter_strict_restore_needs_no_artifact_owner(self):
        self.model.checkpoint(self.root)
        fresh, backend = self.fresh()
        receipt = fresh.restore_checkpoint(self.root)
        self.assertEqual(receipt['forward_tokens'], 0)
        self.assertEqual(backend.stats['forward_tokens'], 0)
        self.assertGreater(backend.stats['loaded_file_bytes'], 0)
        self.assertEqual(receipt['positions'], POSITION_CONVENTION)
        self.assertEqual(fresh.next_logits(self.root), self.model.next_logits(self.root))

    def test_vc014_consumed_generated_controller_survives_reopen(self):
        active = self.model.start_generation(self.root, dict(temperature=0), 'same-stream')
        generated = self.model.generate(active, 4)
        self.assertEqual(len(generated.accepted_tokens), 4)
        saved = generated.snapshot_id
        self.model.checkpoint(saved)
        fresh, backend = self.fresh()
        receipt = fresh.restore_checkpoint(saved)
        self.assertEqual(receipt['forward_tokens'], 0)
        self.assertEqual(fresh.controller(saved), self.model.controller(saved))
        expected = self.model.generate(saved, 2)
        actual = fresh.generate(saved, 2)
        self.assertEqual(actual.accepted_tokens, expected.accepted_tokens)
        self.assertEqual(fresh.controller(actual.snapshot_id), self.model.controller(expected.snapshot_id))

    def test_altered_full_binding_rejected_before_native_load(self):
        self.model.checkpoint(self.root)
        path, original = self.envelope(self.root)
        for field in ('native_input_sha256', 'position_sha256', 'rotary_cursor', 'positions'):
            with self.subTest(field=field):
                value = copy.deepcopy(original)
                value['manifest'][field] = 999 if field == 'rotary_cursor' else '0'*64
                self.alter(path, value)
                fresh, backend = self.fresh()
                with self.assertRaises(IntegrityError): fresh.restore_checkpoint(self.root)
                self.assertEqual(backend.stats['loaded_file_bytes'], 0)
                self.assertEqual(backend.stats['forward_tokens'], 0)

    def test_missing_position_binding_cannot_load_even_with_valid_checksum(self):
        self.model.checkpoint(self.root)
        path, value = self.envelope(self.root)
        value['manifest'].pop('position_sha256', None)
        self.alter(path, value)
        fresh, backend = self.fresh()
        with self.assertRaises(IntegrityError): fresh.restore_checkpoint(self.root)
        self.assertEqual(backend.stats['loaded_file_bytes'], 0)

    def test_prepared_and_direct_save_have_identical_manifest_contract(self):
        self.model.checkpoint(self.root)
        _, direct = self.envelope(self.root)
        prepared = self.model.prepare_checkpoint(self.root)
        self.assertEqual(json.loads(prepared.envelope), direct)
        self.model.publish_prepared_checkpoint(prepared)
        fresh, _ = self.fresh()
        self.assertEqual(fresh.restore_checkpoint(self.root)['forward_tokens'], 0)


if __name__ == '__main__': unittest.main()
