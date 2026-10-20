"""Mixed operator receipts must cover canonical tokens without weakening batch-only receipts."""
import unittest
from context_time_travel import execution_receipt as receipt


class NativeOperatorReceiptTests(unittest.TestCase):
    def events(self):
        return [dict(operator='batched-prefill',position=0,count=2,
                     tokens_sha256=receipt.tokens_digest([1,2]),chunks=[[0,2]]),
                dict(operator='normal-verifier-decode',position=2,count=2,
                     tokens_sha256=receipt.tokens_digest([3,4]),chunks=[[2,1],[3,1]])]

    def test_mixed_receipt_covers_exact_history(self):
        events = self.events()
        self.assertEqual(receipt.validate_operator_schedule(events,[1,2,3,4],8), receipt.schedule_digest(events))

    def test_original_batch_only_contract_still_rejects_normal_operator(self):
        with self.assertRaises(ValueError):
            receipt.validate_schedule(self.events(),[1,2,3,4],8)

    def test_native_generation_cannot_claim_an_unexecuted_multi_token_window(self):
        events = self.events()
        events[1]['chunks'] = [[2,2]]
        with self.assertRaises(ValueError):
            receipt.validate_operator_schedule(events,[1,2,3,4],8)

    def test_changed_tokens_missing_work_and_unknown_operators_are_rejected(self):
        for change in ('tokens','gap','operator'):
            events = self.events()
            if change == 'tokens': events[1]['tokens_sha256'] = receipt.tokens_digest([4,3])
            elif change == 'gap': events[1]['chunks'] = [[3,1]]
            else: events[1]['operator'] = 'unmeasured-fast-path'
            with self.assertRaises(ValueError):
                receipt.validate_operator_schedule(events,[1,2,3,4],8)


if __name__ == '__main__':
    unittest.main()
