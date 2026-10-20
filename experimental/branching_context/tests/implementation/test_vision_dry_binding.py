"""Mixed DRY transport/control validation with a synthetic native reply."""
from pathlib import Path
import shlex
import tempfile
from types import SimpleNamespace
import unittest

from context_time_travel.input_spans import InputHistory
from context_time_travel.min_p import validate_policy
from context_time_travel.store import IntegrityError, InvalidOperation
from context_time_travel.vision_dry import PROTOCOL, sample, text_window
from test_vision_execution import spans


class VisionDryBindingTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.history=InputHistory(spans());self.lease=SimpleNamespace(inputs=self.history,consumed=len(self.history))
        self.policy=validate_policy(dict(temperature=.7,min_p=.05,top_k=0,top_p=1.,
            repetition_penalty=1.,dry_multiplier=.8))
        self.calls=[];self.change=None
        def request(command):
            args=shlex.split(command);self.calls.append(args)
            self.assertEqual(args[0],'sample_mask_image_dry_v1')
            self.assertTrue(Path(args[7]).is_file())
            row=dict(status='ok',operator='full-vocab-min-p',protocol=PROTOCOL,token=0,consumed=9,
                logits_count=8,mask_bytes=1,legal_count=8,draw53=4,top_k=0,kept_count=8,
                temperature=.7,min_p=.05,top_p=1.,dry_multiplier=.8,dry_base=1.75,
                dry_allowed_length=2,dry_penalty_last_n=256,dry_history_count=3,dry_breaker_count=0,
                text_tokens=3,dry_boundary_count=1,dry_penalized_candidates=0)
            if self.change:self.change(row)
            return row
        self.backend=SimpleNamespace(profile=dict(execution=dict(vision_dry_protocol=PROTOCOL)),
            _ready=dict(vision_dry_protocol=PROTOCOL),_check=lambda lease:None,_sequence=0,
            _directory=Path(self.temp.name),_request=request,vocab_size=8,process=SimpleNamespace(pid=1))

    def choose(self,history=(1,2,3),breakers=()):
        return sample(self.backend,self.lease,b'\xff',self.policy,4,history,breakers)

    def test_window_counts_actual_text_and_keeps_image_boundary(self):
        self.assertEqual(text_window(self.history,256),dict(tokens=(1,2,3),boundaries=(2,),text_count=3))
        self.assertEqual(text_window(self.history,2),dict(tokens=(2,3),boundaries=(1,),text_count=3))
        self.assertEqual(text_window(self.history,1),dict(tokens=(3,),boundaries=(0,),text_count=3))

    def test_checked_command_cleans_all_transport_files(self):
        self.assertEqual(self.choose(),0);self.assertEqual(len(self.calls),1)
        self.assertEqual(list(Path(self.temp.name).iterdir()),[])

    def test_missing_native_capability_refuses_before_writing_or_request(self):
        self.backend._ready={}
        with self.assertRaises(InvalidOperation):self.choose()
        self.assertEqual(self.calls,[]);self.assertEqual(list(Path(self.temp.name).iterdir()),[])

    def test_model_cell_tail_cannot_masquerade_as_actual_text(self):
        with self.assertRaises(InvalidOperation):self.choose(self.history.native_tokens)
        self.assertEqual(self.calls,[])

    def test_receipt_text_extent_boundary_or_policy_mismatch_is_rejected(self):
        for key,value in [('text_tokens',9),('dry_boundary_count',0),('temperature',.8),
                ('consumed',3),('draw53',5),('token',False),('kept_count',0)]:
            with self.subTest(key=key):
                self.change=lambda row:row.update({key:value})
                with self.assertRaises(IntegrityError):self.choose()
                self.assertEqual(list(Path(self.temp.name).iterdir()),[])

    def test_noncanonical_breakers_fail_before_request(self):
        for breakers in ((2,1),(1,1),(7,)):
            with self.subTest(breakers=breakers),self.assertRaises(InvalidOperation):self.choose(breakers=breakers)
        self.assertEqual(self.calls,[])


if __name__=='__main__':unittest.main()
