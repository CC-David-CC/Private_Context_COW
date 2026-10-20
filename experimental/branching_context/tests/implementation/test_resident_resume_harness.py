"""Runner retries must use valid, stable operation identities."""
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts'))
import resident_resume
from context_time_travel.store import _name


class ResidentHarnessTests(unittest.TestCase):
    def test_save_uses_valid_stable_identity_without_truncating_snapshot_identity(self):
        calls = []
        def save(branch, revision, sid, operation):
            _name(operation)
            calls.append((branch,revision,sid,operation))
            return operation
        model = SimpleNamespace(store=SimpleNamespace(head=lambda branch: SimpleNamespace(revision=3)),
                                save_branch=save)
        sid = 'a'*64
        first = resident_resume.save(model,'prepared-A',sid)
        self.assertEqual(resident_resume.save(model,'prepared-A',sid),first)
        self.assertNotEqual(resident_resume.save(model,'prepared-B',sid),first)
        self.assertNotEqual(resident_resume.save(model,'prepared-A','a'*63+'b'),first)
        self.assertEqual(calls[0][:3],('prepared-A',3,sid))


if __name__ == '__main__': unittest.main()
