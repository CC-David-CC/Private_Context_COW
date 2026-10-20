"""Device routing tests. Fake factories are scheduler tests, never GPU evidence."""
import unittest
from context_time_travel.strata_backend import worker_command


class GPUWorkerSelectionTests(unittest.TestCase):
    def profile(self):
        return {'execution':{'GPU':[1,5,6],'sut_command':['engine','--prefill','8192']}}

    def test_one_explicit_gpu_per_worker(self):
        p = self.profile()
        for gpu in (1,5,6):
            self.assertEqual(worker_command(p,gpu),['engine','--prefill','8192','--gpu',str(gpu)])
        self.assertEqual(p['execution']['sut_command'],['engine','--prefill','8192'])

    def test_unselected_duplicate_and_conflicting_device_rejected(self):
        from context_time_travel.store import InvalidOperation
        with self.assertRaises(InvalidOperation): worker_command(self.profile(),0)
        p = self.profile();p['execution']['GPU'] = [1,1]
        with self.assertRaises(InvalidOperation): worker_command(p,1)
        p = self.profile();p['execution']['sut_command'] += ['--gpu','1,5']
        with self.assertRaises(InvalidOperation): worker_command(p,1)

    def test_legacy_profile_not_silently_rebound(self):
        from context_time_travel.store import InvalidOperation
        p = {'execution':{'sut_command':['engine']}}
        self.assertEqual(worker_command(p,None),['engine'])
        with self.assertRaises(InvalidOperation): worker_command(p,5)

    def test_group_routes_existing_strata_layer_split_as_one_worker(self):
        p = {'execution':{'GPU':[[0,1,2],5,6],
            'sut_command':['engine','--layer-split','auto']}}
        self.assertEqual(worker_command(p,(0,1,2)),['engine','--layer-split','auto','--gpu','0,1,2'])

    def test_current_native_adapter_rejects_unverified_multi_device_capture(self):
        from context_time_travel.strata_backend import StrataBackend
        from context_time_travel.store import InvalidOperation
        p = {'execution':{'GPU':[[0,1,2]],'sut_command':['engine']}}
        with self.assertRaisesRegex(InvalidOperation,'multi-device'):
            StrataBackend(p,gpu_id=(0,1,2))


if __name__ == '__main__': unittest.main()
