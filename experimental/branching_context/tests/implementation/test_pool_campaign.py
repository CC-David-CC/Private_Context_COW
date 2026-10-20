"""EOS fixture planning tests; no inference is simulated or qualified here."""
import unittest
from pool_campaign import continuation_limit


class PoolCampaignTests(unittest.TestCase):
    def test_eos_does_not_restart_a_finished_controller(self):
        self.assertEqual(continuation_limit({'controller':{'status':'stopped'}},2,8),0)
    def test_active_controller_only_requests_remaining_outputs(self):
        self.assertEqual(continuation_limit({'controller':{'status':'active'}},4,8),4)
        self.assertEqual(continuation_limit({'controller':{'status':'active'}},8,8),0)


if __name__=='__main__': unittest.main()
