"""Mixed V4 transport guard with synthetic replies, not real GPU evidence."""
import unittest

from context_time_travel.store import InvalidOperation
from context_time_travel.vision_execution import RESUME_CONTRACT
import test_typical_dynatemp_backend as fixtures


class VisionEntropyBackendTests(unittest.TestCase):
    def setUp(self):
        self.fixture=fixtures.EntropyBackendTests();self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.backend=self.fixture.backend
        self.backend.profile['execution']['resume_contract']=RESUME_CONTRACT
        self.backend._ready=dict(sampling_v4_protocol=fixtures.V4)

    def test_mixed_entropy_uses_existing_v4_with_empty_repetition_inputs(self):
        self.assertEqual(self.fixture.sample(),8)
        self.assertEqual(self.fixture.calls[0][0],'sample_mask_v4')
        self.assertEqual(self.fixture.fixture.lease.consumed,4)
        self.assertEqual(list(self.backend._directory.iterdir()),[])

    def test_missing_or_wrong_native_capability_rejects_without_pipe_or_files(self):
        for ready in ({},dict(sampling_v4_protocol='wrong')):
            with self.subTest(ready=ready):
                self.backend._ready=ready
                with self.assertRaisesRegex(InvalidOperation,'checked native V4'):
                    self.fixture.sample()
                self.assertEqual(self.fixture.calls,[])
                self.assertEqual(list(self.backend._directory.iterdir()),[])

    def test_mixed_dry_entropy_is_rejected_before_transient_files(self):
        with self.assertRaisesRegex(InvalidOperation,'Mixed DRY entropy'):
            self.fixture.sample(dry_multiplier=.8,dry_history=(0,1,3,0),dry_breakers=(3,))
        self.assertEqual(self.fixture.calls,[])
        self.assertEqual(list(self.backend._directory.iterdir()),[])


if __name__=='__main__':unittest.main()
