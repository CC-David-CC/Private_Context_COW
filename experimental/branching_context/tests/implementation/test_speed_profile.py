"""Actual file fingerprints for the portable, unqualified demo profile."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))
from prepare_speed_profile import model_inputs, hash_inventory


class SpeedProfileInputs(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.pack = self.root / 'pack'
        self.pack.mkdir()
        self.model = self.root / 'model.gguf'
        self.model.write_bytes(b'actual fixture bytes')
        self.experts = self.root / 'expert-profile.bin'
        self.experts.write_bytes(b'profile')
        for name in ('dense.bin', 'index.txt', 'native_experts.txt'):
            (self.pack / name).write_bytes(name.encode())
        self.manifest([dict(name=self.model.name, size=self.model.stat().st_size)])

    def manifest(self, shards):
        (self.pack / 'conversions.json').write_text(json.dumps(dict(source_shards=shards)))

    def test_every_weight_pack_and_tokenizer_file_is_bound(self):
        tokenizer = self.pack / 'tokenizer'
        tokenizer.mkdir()
        extra = tokenizer / 'vocab.json'
        extra.write_text('{"word": 1}')
        paths = model_inputs(self.model, self.pack, self.experts)
        self.assertIn(extra.resolve(), paths)
        self.assertIn(self.model.resolve(), paths)
        receipt = hash_inventory(paths)
        self.assertEqual(receipt['model_files'][str(self.model.resolve())]['sha256'],
                         hashlib.sha256(self.model.read_bytes()).hexdigest())
        self.assertEqual(receipt['model_file_hash_receipts_reused'], 0)
        self.assertEqual(receipt['model_bytes_hashed'], sum(p.stat().st_size for p in paths))

    def test_missing_shard_or_wrong_size_is_rejected(self):
        self.manifest([dict(name=self.model.name, size=99)])
        with self.assertRaises(ValueError):
            model_inputs(self.model, self.pack, self.experts)
        self.manifest([dict(name=self.model.name, size=self.model.stat().st_size),
                       dict(name='missing.gguf', size=10)])
        with self.assertRaises(FileNotFoundError):
            model_inputs(self.model, self.pack, self.experts)

    def test_shard_path_cannot_escape_manifest_directory(self):
        for name in ('../model.gguf', '/absolute.gguf', 'folder\\model.gguf'):
            self.manifest([dict(name=name, size=20)])
            with self.assertRaises(ValueError):
                model_inputs(self.model, self.pack, self.experts)

    def test_changed_bytes_invalidate_reused_receipt(self):
        paths = model_inputs(self.model, self.pack, self.experts)
        first = hash_inventory(paths)
        second = hash_inventory(paths, first)
        self.assertEqual(second['model_bytes_hashed'], 0)
        self.assertEqual(second['model_file_hash_receipts_reused'], len(paths))
        self.model.write_bytes(b'a changed, longer fixture')
        third = hash_inventory(paths, first)
        self.assertEqual(third['model_file_hash_receipts_reused'], len(paths)-1)
        self.assertEqual(third['model_bytes_hashed'], self.model.stat().st_size)
        key = str(self.model.resolve())
        self.assertNotEqual(first['model_files'][key]['sha256'], third['model_files'][key]['sha256'])


if __name__ == '__main__':
    unittest.main()
