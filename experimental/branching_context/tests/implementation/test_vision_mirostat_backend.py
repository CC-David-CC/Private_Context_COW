"""Mixed Mirostat transport checks use synthetic replies, not GPU inference."""
import unittest

from context_time_travel.store import InvalidOperation
from context_time_travel.vision_execution import RESUME_CONTRACT
import test_mirostat2_backend as fixtures


class VisionMirostatBackendTests(unittest.TestCase):
    def setUp(self):
        self.fixture=fixtures.MirostatBackendTests();self.fixture.setUp();self.addCleanup(self.fixture.doCleanups)
        self.backend=self.fixture.backend;self.backend.profile['execution']['resume_contract']=RESUME_CONTRACT
        self.backend._ready=dict(sampling_protocol=fixtures.V6,mirostat2_protocol=fixtures.V6,
            mirostat2_feedback_protocol='mirostat2-state-v1',mirostat2_feedback_version=1)

    def test_existing_v6_returns_detached_consume_bound_feedback(self):
        result=self.fixture.sample()
        self.assertEqual((result.token,result.draw_start,result.draw_count),(8,7,1))
        self.assertEqual(result.feedback_proposal['mu'],10.5)
        self.assertEqual(result.source_digest,fixtures.SOURCE)
        self.assertEqual(self.fixture.commands[0][0],'sample_mask_mirostat2')
        self.assertEqual(self.fixture.lease.consumed,4);self.assertEqual(list(self.backend._directory.iterdir()),[])

    def test_missing_or_wrong_feedback_query_rejects_before_pipe_or_files(self):
        for ready in ({},dict(mirostat2_protocol='wrong'),dict(mirostat2_protocol=fixtures.V6,
                mirostat2_feedback_protocol='mirostat2-state-v1',mirostat2_feedback_version=True)):
            with self.subTest(ready=ready):
                self.backend._ready=ready
                with self.assertRaisesRegex(InvalidOperation,'Mixed.*Mirostat'):self.fixture.sample()
                self.assertEqual(self.fixture.commands,[]);self.assertEqual(list(self.backend._directory.iterdir()),[])

    def test_dry_feedback_combination_rejects_before_pipe_or_files(self):
        with self.assertRaisesRegex(InvalidOperation,'Mixed.*DRY'):
            self.fixture.sample(dry_multiplier=.8,dry_history=(0,1,3,0),dry_breakers=(3,))
        self.assertEqual(self.fixture.commands,[]);self.assertEqual(list(self.backend._directory.iterdir()),[])


if __name__=='__main__':unittest.main()
