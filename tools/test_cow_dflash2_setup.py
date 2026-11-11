"""DFlash2 setup admission; no model, network, or GPU is needed."""
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from tools.cow_server_setup import dflash2_assets


class DFlash2SetupTests(unittest.TestCase):
    def test_ordinary_setup_is_unchanged(self):
        self.assertEqual(dflash2_assets(SimpleNamespace()), [])

    def test_valid_pair_is_resolved(self):
        with tempfile.TemporaryDirectory() as directory:
            pack=Path(directory)/'draft.pack'; vocab=Path(directory)/'vocab.bin'
            pack.write_bytes(b'pack'); vocab.write_bytes(b'vocab')
            self.assertEqual(dflash2_assets(SimpleNamespace(dflash2_pack=pack,dflash2_vocab=vocab)),
                             [pack.resolve(),vocab.resolve()])

    def test_incomplete_invalid_or_unsupported_pair_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            pack=Path(directory)/'draft.pack'; vocab=Path(directory)/'vocab.bin'
            pack.write_bytes(b'pack'); vocab.write_bytes(b'vocab')
            link=Path(directory)/'link'; link.symlink_to(pack)
            cases=[dict(dflash2_pack=pack),
                   dict(dflash2_vocab=vocab),
                   dict(dflash2_pack=Path('relative'),dflash2_vocab=vocab),
                   dict(dflash2_pack=link,dflash2_vocab=vocab),
                   dict(dflash2_pack=pack,dflash2_vocab=Path(directory)/'missing'),
                   dict(dflash2_pack=pack,dflash2_vocab=vocab,gpu=[0,1]),
                   dict(dflash2_pack=pack,dflash2_vocab=vocab,vision_config='fixture')]
            for case in cases:
                with self.subTest(case=case), self.assertRaises(ValueError):
                    dflash2_assets(SimpleNamespace(**case))


if __name__ == '__main__':
    unittest.main()
