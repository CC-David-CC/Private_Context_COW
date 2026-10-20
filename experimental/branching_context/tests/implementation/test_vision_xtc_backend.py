"""Mixed proposal transport guards with synthetic replies, not GPU evidence."""
import unittest

from context_time_travel.store import InvalidOperation
from context_time_travel.vision_execution import RESUME_CONTRACT
import test_xtc_backend as fixtures


class VisionXtcBackendTests(unittest.TestCase):
    def setUp(self):
        self.fixture=fixtures.XtcBackendTests();self.fixture.setUp();self.addCleanup(self.fixture.doCleanups)
        self.backend=self.fixture.backend;self.backend.profile['execution']['resume_contract']=RESUME_CONTRACT
        self.backend._ready=dict(sampling_protocol=fixtures.V5)

    def test_existing_v5_returns_a_bound_read_only_proposal(self):
        self.fixture.response=self.fixture._response(xtc_probability=.4)
        result=self.fixture.sample(xtc_probability=.4)
        self.assertEqual((result.token,result.draw_start,result.draw_count),(8,5,1))
        self.assertEqual(result.source_digest,fixtures.SOURCE)
        self.assertEqual(self.fixture.commands[0][0],'sample_mask_v5')
        self.assertEqual(self.fixture.lease.consumed,4)
        self.assertEqual(list(self.backend._directory.iterdir()),[])

    def test_missing_or_wrong_v5_query_rejects_before_pipe_or_files(self):
        for ready in ({},dict(sampling_protocol='wrong')):
            with self.subTest(ready=ready):
                self.backend._ready=ready
                with self.assertRaisesRegex(InvalidOperation,'Mixed.*V5'):self.fixture.sample()
                self.assertEqual(self.fixture.commands,[]);self.assertEqual(list(self.backend._directory.iterdir()),[])

    def test_mixed_dry_proposal_is_closed_even_when_model_cell_tail_matches(self):
        self.fixture.response=self.fixture._response(dry_multiplier=.8,dry_history_count=4,dry_breaker_count=1)
        with self.assertRaisesRegex(InvalidOperation,'Mixed.*DRY'):
            self.fixture.sample(dry_multiplier=.8,dry_history=(0,1,3,0),dry_breakers=(3,))
        self.assertEqual(self.fixture.commands,[]);self.assertEqual(list(self.backend._directory.iterdir()),[])


if __name__=='__main__':unittest.main()
