"""Placement decisions with explicit synthetic rates, not speed benchmarks."""
import math
import unittest
from context_time_travel.transfer_costs import TransferCosts


class TransferTests(unittest.TestCase):
    def test_transfer_can_outweigh_faster_compute(self):
        costs=TransferCosts()
        costs.observe_forward(0,100,10); costs.observe_forward(1,100,1)
        costs.observe_restore(1,1000,20)
        self.assertEqual(costs.choose([0,1],[0],1000,100).worker,0)
        costs.observe_restore(1,1000,.01)
        # Use fresh measurements for a different link/device, not overwrite history.
        fast=TransferCosts(); fast.observe_forward(0,100,10); fast.observe_forward(1,100,1)
        fast.observe_restore(1,1000,.01)
        choice=fast.choose([0,1],[0],1000,100)
        self.assertEqual(choice.worker,1); self.assertEqual(choice.reason,'measured')

    def test_missing_measurements_keep_warm_affinity(self):
        costs=TransferCosts(); costs.observe_forward(1,100,1)
        self.assertEqual(costs.choose([0,1],[0],1000,100).worker,0)
        self.assertEqual(costs.choose([0,1],[],1000,100).worker,0)

    def test_invalid_measurements_do_not_poison_scheduler(self):
        costs=TransferCosts()
        for value in [math.nan,math.inf,-1,True]:
            with self.assertRaises(ValueError):costs.observe_forward(0,100,value)
        with self.assertRaises(ValueError):costs.observe_restore(0,0,1)
        self.assertEqual(costs.metrics()['forward'],{})

    def test_byte_count_changes_choice_and_ties_are_stable(self):
        costs=TransferCosts()
        costs.observe_forward(0,100,10); costs.observe_forward(1,100,1)
        costs.observe_restore(1,1000,1)
        self.assertEqual(costs.choose([0,1],[0],100,100).worker,1)
        self.assertEqual(costs.choose([0,1],[0],100000,100).worker,0)
        costs.observe_forward(2,100,10)
        self.assertEqual(costs.choose([2,0],[2,0],0,100).worker,2)


if __name__=='__main__':unittest.main()
