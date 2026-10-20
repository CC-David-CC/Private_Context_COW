"""Fresh setup must reject unknown files and avoid replacing an existing server."""
import importlib.util
import tempfile
import unittest
from pathlib import Path

class CowServerSetupTests(unittest.TestCase):
    def module(self):
        path=Path(__file__).parents[1]/"tools/cow_server_setup.py"
        spec=importlib.util.spec_from_file_location("cow_setup",path)
        result=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(result)
        return result

    def test_all_split_model_shards_are_required(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/"model-00001-of-00002.gguf"
            p.write_bytes(b"first")
            with self.assertRaisesRegex(ValueError,"shard"):
                self.module().model_shards(p)

    def test_output_directory_is_never_reused(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/"existing"
            p.mkdir()
            keep=p/"keep"
            keep.write_text("saved server")
            with self.assertRaises(FileExistsError):
                self.module().new_output(p)
            self.assertEqual(keep.read_text(),"saved server")

    def test_shards_are_sorted_and_hashes_read_actual_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            a=Path(tmp)/"model-00001-of-00002.gguf"
            b=Path(tmp)/"model-00002-of-00002.gguf"
            a.write_bytes(b"first")
            b.write_bytes(b"second")
            self.assertEqual(self.module().model_shards(a),[a,b])
            x=self.module().fingerprint(a)
            self.assertEqual(x["bytes"],5)
            self.assertEqual(x["sha256"],__import__("hashlib").sha256(b"first").hexdigest())
