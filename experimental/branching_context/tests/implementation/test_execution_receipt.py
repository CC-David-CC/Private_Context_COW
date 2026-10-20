"""Execution receipts bind completed native requests to canonical token ranges."""
import hashlib
import importlib.util
from pathlib import Path
import sys
import unittest

spec = importlib.util.spec_from_file_location('execution_receipt',
    Path(__file__).resolve().parents[2] / 'src/context_time_travel/execution_receipt.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
validate_schedule, schedule_digest = module.validate_schedule, module.schedule_digest


class ExecutionReceiptTests(unittest.TestCase):
    def test_valid_and_different_partition_identity(self):
        history = (100, 101, 102)
        def event(start, count, chunks):
            import struct
            return dict(operator='batched-prefill', position=start, count=count,
                tokens_sha256=hashlib.sha256(struct.pack('<'+'q'*count, *history[start:start+count])).hexdigest(),
                chunks=chunks)
        a = [event(0,3,[[0,2],[2,1]])]
        b = [event(0,2,[[0,2]]),event(2,1,[[2,1]])]
        validate_schedule(a,history,2)
        validate_schedule(b,history,2)
        self.assertNotEqual(schedule_digest(a),schedule_digest(b))
        for bad in ([], [dict(a[0],operator='normal-decode')],
                    [dict(a[0],position=1)], [dict(a[0],tokens_sha256='0'*64)],
                    [dict(a[0],chunks=[[0,3]])], [dict(a[0],count=True)]):
            with self.assertRaises(ValueError): validate_schedule(bad,history,2)


if __name__ == '__main__': unittest.main()
