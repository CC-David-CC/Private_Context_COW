"""Image V7 command guards with synthetic replies, not GPU qualification."""
import unittest

from context_time_travel.store import InvalidOperation
from context_time_travel.vision_execution import RESUME_CONTRACT
import test_adaptive_p_backend as fixtures


class VisionAdaptiveBackendTests(unittest.TestCase):
    def setUp(self):
        self.fixture=fixtures.AdaptivePBackendTests();self.fixture.setUp();self.addCleanup(self.fixture.doCleanups)
        self.backend=self.fixture.backend;self.backend.profile['execution']['resume_contract']=RESUME_CONTRACT

    def test_existing_v7_returns_detached_consumption_bound_feedback(self):
        result=self.fixture.sample()
        self.assertEqual((result.token,result.draw_start,result.draw_count),(3,7,1))
        self.assertEqual(result.feedback_proposal['version'],'adaptive-p-state-v1')
        self.assertEqual(result.feedback_proposal['weighted_sum'],self.fixture.response['next_weighted_sum'])
        self.assertEqual(result.source_digest,fixtures.SOURCE)
        self.assertEqual(self.fixture.commands[0][0],'sample_mask_adaptive_p')
        self.assertEqual(self.fixture.lease.consumed,4);self.assertEqual(list(self.backend._directory.iterdir()),[])

    def test_wrong_or_missing_feedback_query_rejects_before_pipe_or_files(self):
        for changes in ({'adaptive_p_protocol':'wrong'},{'adaptive_p_feedback_protocol':'wrong'},
                        {'adaptive_p_feedback_version':True}):
            with self.subTest(changes=changes):
                ready=self.backend._ready;self.backend._ready=dict(ready,**changes)
                with self.assertRaisesRegex(InvalidOperation,'Mixed.*Adaptive'):self.fixture.sample()
                self.assertEqual(self.fixture.commands,[]);self.assertEqual(list(self.backend._directory.iterdir()),[])
                self.backend._ready=ready

    def test_dry_combination_remains_closed_even_when_tail_cell_count_matches(self):
        with self.assertRaisesRegex(InvalidOperation,'Mixed.*DRY'):
            self.fixture.sample(dry_multiplier=.8,dry_history=(0,1,3,0),dry_breakers=(3,))
        self.assertEqual(self.fixture.commands,[]);self.assertEqual(list(self.backend._directory.iterdir()),[])


if __name__=='__main__':unittest.main()
