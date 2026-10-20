"""CPU protocol fixtures, not GPU inference evidence."""
import copy
import shlex
import struct
import unittest
from pathlib import Path
from context_time_travel import min_p
from context_time_travel.store import IntegrityError, InvalidOperation

V4 = 'full-vocab-typical-dyna-min-p-u53-v4'


class EntropyBackendTests(unittest.TestCase):
    def setUp(self):
        import test_min_p_integration as fixtures
        self.fixture = fixtures.MinPBackendTests('test_native_sample_is_read_only_and_cleans_transient_mask')
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.backend = self.fixture.backend
        self.backend.profile['execution']['sampling_protocol'] = V4
        self.calls = []
        self.response = dict(self.fixture.response,protocol=V4,top_k=0,top_p=1.,
            dry_multiplier=0.,dry_base=1.75,dry_allowed_length=2,dry_penalty_last_n=256,
            dry_history_count=0,dry_breaker_count=0,dry_penalized_candidates=0,
            typical_p=.95,dynatemp_range=.35,dynatemp_exponent=1.,
            effective_temperature=.8,typical_kept_count=1)
        def request(command):
            fields = shlex.split(command)
            self.calls.append(fields)
            self.assertEqual(Path(fields[7]).read_bytes(),b'\0\1')
            if fields[0] == 'sample_mask_v4':
                self.assertEqual(Path(fields[12]).read_bytes(),self.history)
                self.assertEqual(Path(fields[13]).read_bytes(),self.breakers)
            return dict(self.response)
        self.backend._request = request
        self.history = self.breakers = b''

    def sample(self, **changes):
        args = dict(typical_p=.95,dynatemp_range=.35,dynatemp_exponent=1.)
        args.update(changes)
        return self.backend.sample_legal(self.fixture.lease,b'\0\1',.7,.05,123,**args)

    def test_native_ready_requires_exact_pinned_v4_protocol(self):
        from context_time_travel.strata_backend import _validate_sampling_capability
        _validate_sampling_capability(self.backend.profile,{'sampling_protocol':V4})
        for received in ({},{'sampling_protocol':min_p.DRY_PROTOCOL}):
            with self.assertRaises(IntegrityError):
                _validate_sampling_capability(self.backend.profile,received)

    def test_v4_command_binds_settings_and_empty_dry_window(self):
        self.assertEqual(self.sample(),8)
        self.assertEqual(self.calls[0][:7],['sample_mask_v4','4','0.7','0.05','0','1.0','123'])
        self.assertEqual(self.calls[0][14:],['0.95','0.35','1.0'])
        self.assertEqual(list(self.backend._directory.iterdir()),[])
        self.assertEqual(self.fixture.lease.consumed,4)

    def test_active_dry_and_entropy_bind_same_canonical_tail(self):
        self.history = struct.pack('<4I',0,1,3,0)
        self.breakers = struct.pack('<I',3)
        self.response.update(dry_multiplier=.8,dry_history_count=4,dry_breaker_count=1,
                             dry_penalized_candidates=1)
        self.assertEqual(self.sample(dry_multiplier=.8,dry_history=(0,1,3,0),dry_breakers=(3,)),8)
        self.assertEqual(self.calls[0][8],'0.8')
        self.assertEqual(list(self.backend._directory.iterdir()),[])

    def test_old_capability_rejected_before_transient_io(self):
        self.backend.profile['execution']['sampling_protocol'] = min_p.DRY_PROTOCOL
        with self.assertRaises(InvalidOperation): self.sample()
        self.assertEqual(self.calls,[])
        self.assertEqual(list(self.backend._directory.iterdir()),[])

    def test_invalid_receipts_rejected_and_files_removed(self):
        for field,value in [('typical_p',.9),('dynatemp_range',.2),('dynatemp_exponent',True),
                            ('effective_temperature',float('nan')),('effective_temperature',1.1),
                            ('typical_kept_count',True),('typical_kept_count',0),
                            ('dry_history_count',1),('dry_penalized_candidates',1)]:
            original = copy.deepcopy(self.response)
            self.response[field] = value
            with self.subTest(field=field),self.assertRaises(IntegrityError): self.sample()
            self.assertEqual(list(self.backend._directory.iterdir()),[])
            self.response = original

    def test_disabled_entropy_uses_unchanged_v2_protocol(self):
        self.response = dict(self.fixture.response,protocol=min_p.SAMPLING_PROTOCOL_V2,top_k=0,top_p=1.)
        self.assertEqual(self.sample(typical_p=1.,dynatemp_range=0.),8)
        self.assertEqual(self.calls[0][0],'sample_mask_v2')
        self.assertEqual(len(self.calls[0]),8)
        self.assertEqual(list(self.backend._directory.iterdir()),[])


if __name__ == '__main__': unittest.main()
