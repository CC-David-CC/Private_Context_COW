"""Checked mixed entropy/DRY transport with synthetic native replies."""
from pathlib import Path
from contextlib import ExitStack
import shlex
import struct
import tempfile
import unittest
from unittest.mock import patch

from context_time_travel.input_spans import InputHistory
from context_time_travel.store import IntegrityError,InvalidOperation
from context_time_travel.vision_execution import RESUME_CONTRACT
from test_vision_execution import spans
import test_typical_dynatemp_backend as fixtures

PROTOCOL='mixed-text-runs-sampling-u53-v2'


class VisionDryEntropyBackendTests(unittest.TestCase):
    def setUp(self):
        self.fixture=fixtures.EntropyBackendTests();self.fixture.setUp();self.addCleanup(self.fixture.doCleanups)
        self.backend=self.fixture.backend;self.lease=self.fixture.fixture.lease
        self.backend.profile['execution'].update(resume_contract=RESUME_CONTRACT,
            vision_dry_protocol='mixed-text-runs-dry-u53-v1')
        self.backend._ready=dict(sampling_v4_protocol=fixtures.V4,vision_dry_protocol='mixed-text-runs-dry-u53-v1',
            vision_dry_sampling_protocol=PROTOCOL,vision_sampling_protocols=[fixtures.V4])
        self.lease.inputs=InputHistory(spans(rows=1,grid_width=1,grid_height=1,embedding_bytes=36))
        self.fixture.response.update(dry_multiplier=.8,dry_history_count=3,dry_breaker_count=0,
            dry_penalized_candidates=0,vision_dry_sampling_protocol=PROTOCOL,text_tokens=3,
            dry_boundary_count=1,dry_boundaries=[2])
        self.calls=[]
        def request(command):
            fields=shlex.split(command);self.calls.append(fields)
            self.assertEqual(fields[0],'sample_mask_image_v4')
            self.assertEqual(Path(fields[7]).read_bytes(),b'\0\1')
            self.assertEqual(Path(fields[12]).read_bytes(),struct.pack('<3I',1,2,3))
            self.assertEqual(Path(fields[13]).read_bytes(),b'')
            return dict(self.fixture.response)
        self.backend._request=request

    def choose(self,history=(1,2,3)):
        return self.fixture.sample(dry_multiplier=.8,dry_history=history,dry_breakers=())

    def test_checked_command_uses_text_count_instead_of_model_cell_count(self):
        self.assertEqual(self.choose(),8);self.assertEqual(self.lease.consumed,4)
        self.assertEqual(len(self.calls),1);self.assertEqual(list(self.backend._directory.iterdir()),[])

    def test_unavailable_or_wrong_family_rejects_before_transient_io(self):
        for changes in ({'vision_dry_sampling_protocol':'wrong'},{'vision_sampling_protocols':[]}):
            with self.subTest(changes=changes):
                ready=self.backend._ready;self.backend._ready=dict(ready,**changes)
                with self.assertRaisesRegex(InvalidOperation,'Mixed DRY entropy'):self.choose()
                self.assertEqual(self.calls,[]);self.assertEqual(list(self.backend._directory.iterdir()),[])
                self.backend._ready=ready

    def test_model_cell_tail_rejects_before_transient_io(self):
        with self.assertRaises(InvalidOperation):self.choose(self.lease.inputs.native_tokens)
        self.assertEqual(self.calls,[]);self.assertEqual(list(self.backend._directory.iterdir()),[])

    def test_exact_boundary_positions_and_typed_extent_receipts_are_required(self):
        for changes in ({'dry_boundaries':[1]},{'text_tokens':4},{'dry_boundary_count':True},
                        {'vision_dry_sampling_protocol':'wrong'}):
            with self.subTest(changes=changes):
                original=dict(self.fixture.response);self.fixture.response.update(changes)
                with self.assertRaises(IntegrityError):self.choose()
                self.assertEqual(list(self.backend._directory.iterdir()),[]);self.fixture.response=original

    def test_actual_constructor_queries_and_validates_native_sampler_families(self):
        # Exercise __init__, not an object whose _ready dictionary was prefilled.
        # File parsing and the process are CPU protocol fixtures, not inference.
        import context_time_travel.strata_backend as module
        from context_time_travel.vision_execution import VISION_PROTOCOL
        with tempfile.TemporaryDirectory() as temporary:
            reader=Path(temporary)/'reader.py'
            reader.write_text('class GGUFFile:\n def __init__(self,path):\n  self.metadata='+repr({
                'tokenizer.ggml.tokens':['a']*9,'general.architecture':'qwen4exp',
                'tokenizer.ggml.model':'gpt2','qwen4exp.embedding_length':9})+'\n')
            profile=dict(gguf_reader=str(reader),model_path='fixture.gguf',execution=dict(
                sut_command=['fixture-native'],capacity=8,embedding_width=9,resume_contract=RESUME_CONTRACT,
                vision_protocol=VISION_PROTOCOL,vision_dry_protocol='mixed-text-runs-dry-u53-v1'))
            info=dict(vision_protocol=VISION_PROTOCOL,enabled=True,hot_image_snapshots=True,cold_image_snapshots=True,
                vision_dry_protocol='mixed-text-runs-dry-u53-v1',vision_dry_sampling_protocol=PROTOCOL)
            for families,accepted in (([fixtures.V4],True),(['wrong'],False),([fixtures.V4]*2,False),(fixtures.V4,False)):
                with self.subTest(families=families),ExitStack() as stack:
                    stack.enter_context(patch.object(module,'_validate_model',return_value={}))
                    stack.enter_context(patch.object(module.StrataBackend,'validate_execution_profile'))
                    stack.enter_context(patch.object(module.subprocess,'Popen'))
                    stack.enter_context(patch.object(module.StrataBackend,'_read',return_value=dict(status='ready',vocab=9,capacity=8)))
                    stack.enter_context(patch.object(module.StrataBackend,'_refresh_stats'))
                    request=stack.enter_context(patch.object(module.StrataBackend,'_request',return_value=dict(info,vision_sampling_protocols=families)))
                    stack.enter_context(patch.object(module.StrataBackend,'close'))
                    backend=module.StrataBackend.__new__(module.StrataBackend)
                    try:
                        if accepted:
                            module.StrataBackend.__init__(backend,profile)
                            self.assertEqual(backend._ready['vision_sampling_protocols'],[fixtures.V4])
                        else:
                            with self.assertRaisesRegex(IntegrityError,'sampler families'):
                                module.StrataBackend.__init__(backend,profile)
                        request.assert_called_once_with('vision_info')
                    finally:
                        backend._log.close();backend._scratch.cleanup()


if __name__=='__main__':unittest.main()
