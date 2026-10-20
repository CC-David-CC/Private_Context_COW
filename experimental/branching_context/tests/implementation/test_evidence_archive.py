"""Archive selection checks, not inference or gate certification."""
from pathlib import Path
import sys
import tempfile
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts'))
from archive_resident_evidence import collect_files


class EvidenceArchiveTests(unittest.TestCase):
    def test_explicit_versions_are_included_and_missing_roots_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            campaign,probe = root/'campaign',root/'probe'
            for path in (campaign,probe/'binaries-digest-v1',probe/'api-digest-v1/src',probe/'api-digest-v1/scripts'):
                path.mkdir(parents=True)
                (path/'witness.txt').write_text(str(path))
            files = collect_files([('campaign',campaign)],probe,['binaries-digest-v1'],['api-digest-v1'])
            self.assertEqual(set(files),{'campaign/witness.txt','probe/binaries-digest-v1/witness.txt',
                'probe/api-digest-v1/src/witness.txt','probe/api-digest-v1/scripts/witness.txt'})
            with self.assertRaises(FileNotFoundError):
                collect_files([('campaign',campaign)],probe,['binaries-missing'],['api-digest-v1'])

    def test_path_escape_and_duplicate_archive_names_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            campaign = root/'campaign';campaign.mkdir()
            (campaign/'witness.txt').write_text('evidence')
            for bad in ('../outside','binaries/other',''):
                with self.subTest(name=bad),self.assertRaises(ValueError):
                    collect_files([('campaign',campaign)],root,[bad],[])
            with self.assertRaises(ValueError):
                collect_files([('same',campaign),('same',campaign)],root,[],[])


if __name__ == '__main__': unittest.main()
