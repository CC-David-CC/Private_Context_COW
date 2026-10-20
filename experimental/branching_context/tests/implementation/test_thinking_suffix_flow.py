"""Receipt checks for the actual thinking question CLI campaign, not GPU substitutes."""
import unittest
from run_thinking_suffix_flow import validate_question, FlowError

class ThinkingSuffixFlowTests(unittest.TestCase):
    def valid(self):
        return dict(prefix_cells=177, suffix_cells=14, generated=5, forward_tokens=19,
                    prefix_unchanged=True, eos_consumed=True, controller_stopped=True,
                    reasoning_tokens=2, thinking_budget=512, source_controller_unchanged=True)
    def test_exact_suffix_only_work(self):
        validate_question(self.valid())
    def test_prefix_replay_is_rejected(self):
        row=self.valid();row['forward_tokens']+=177
        with self.assertRaises(FlowError):validate_question(row)
    def test_unconsumed_eos_and_parent_mutation_rejected(self):
        for key in ('prefix_unchanged','eos_consumed','controller_stopped','source_controller_unchanged'):
            with self.subTest(key=key):
                row=self.valid();row[key]=False
                with self.assertRaises(FlowError):validate_question(row)
    def test_missing_work_and_exceeded_budget_rejected(self):
        for key in ('forward_tokens','reasoning_tokens'):
            row=self.valid();row.pop(key)
            with self.assertRaises(FlowError):validate_question(row)
        row=self.valid();row['reasoning_tokens']=513
        with self.assertRaises(FlowError):validate_question(row)

if __name__=='__main__':unittest.main()
