"""Read/write scope of the real VC-015 campaign's embedding audit."""
import os
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))
from run_vision_rebuild_probe import audit_payload_read


class RebuildProbeAuditTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.managed = self.root / 'checkpoints'
        self.external = self.root / 'old-fixture.sve'
        self.fresh = self.root / 'encoder' / 'output.sve'

    def audit(self, path, mode, flags, event='open'):
        audit_payload_read(event, (str(path), mode, flags),
                           self.managed, {str(self.fresh.resolve())})

    def test_external_old_embedding_reads_are_rejected(self):
        for mode, flags in [('r', os.O_RDONLY), ('r+', os.O_RDWR)]:
            with self.subTest(mode=mode), self.assertRaisesRegex(AssertionError, 'old external'):
                self.audit(self.external, mode, flags)

    def test_native_transport_writes_do_not_count_as_fixture_reads(self):
        self.audit(self.external, 'w', os.O_WRONLY | os.O_CREAT | os.O_TRUNC)

    def test_managed_and_current_encoder_output_reads_are_allowed(self):
        self.audit(self.managed / 'accepted.sve', 'r', os.O_RDONLY)
        self.audit(self.fresh, 'r', os.O_RDONLY)

    def test_nonembedding_and_other_events_are_ignored(self):
        self.audit(self.root / 'original.png', 'r', os.O_RDONLY)
        self.audit(self.external, 'r', os.O_RDONLY, event='os.stat')

    def test_byte_paths_cannot_bypass_the_read_guard(self):
        with self.assertRaisesRegex(AssertionError, 'old external'):
            audit_payload_read('open', (os.fsencode(self.external), 'r', os.O_RDONLY),
                               self.managed, set())


if __name__ == '__main__':
    unittest.main()
