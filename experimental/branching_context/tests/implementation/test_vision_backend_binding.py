"""CPU protocol rejection/ownership tests; native GPU runs remain independent."""
from dataclasses import replace
import tempfile
import threading
from pathlib import Path
import unittest

from context_time_travel.input_spans import InputHistory
from context_time_travel.strata_backend import StrataBackend, StrataImage, _Lease
from context_time_travel.store import IntegrityError, InvalidOperation
from context_time_travel.vision_execution import VISION_PROTOCOL, RESUME_CONTRACT, input_binding
from test_vision_execution import spans, receipt


class VisionBackendBindingTests(unittest.TestCase):
    def setUp(self):
        self.history=InputHistory(spans())
        self.backend=object.__new__(StrataBackend)
        self.backend.profile={'execution':dict(resume_contract=RESUME_CONTRACT,
            prefill_chunk=128,capacity=128,embedding_width=4)}
        self.backend._closed=False; self.backend._process_identity=b'worker'
        self.backend.vocab_size=16
        self.lease=_Lease(self.backend,bound=True,consumed=9,state='ACTIVE',
            _owner_thread=threading.get_ident(),inputs=self.history)
        self.backend._active=self.lease
        self.calls=[]
        self.info=dict(status='ok',enabled=True,vision_protocol=VISION_PROTOCOL,
                       consumed=9,**input_binding(self.history))
        def request(command):
            self.calls.append(command)
            return dict(self.info)
        self.backend._request=request

    def test_native_echo_must_bind_payload_position_and_consumed_extent(self):
        self.backend._check_input_binding(self.history)
        for key in ('native_input_sha256','position_sha256','rotary_cursor','consumed','vision_protocol'):
            with self.subTest(field=key):
                original=self.info.pop(key)
                with self.assertRaises(IntegrityError):self.backend._check_input_binding(self.history)
                self.info[key]='wrong'
                with self.assertRaises(IntegrityError):self.backend._check_input_binding(self.history)
                self.info[key]=original

    def test_same_length_wrong_image_rejected_before_native_restore(self):
        image=StrataImage(b'identity',b'worker',3,9,1,64,(),self.history.digest)
        wrong=InputHistory(spans(embedding_sha256='e'*64))
        with self.assertRaises(IntegrityError):
            self.backend.restore(self.lease,image,history=wrong,frontier=(0.,)*16)
        self.assertEqual(self.calls,[])
        self.assertEqual(self.lease.inputs,self.history)

    def test_legacy_unbound_hot_image_cannot_enter_mixed_restore(self):
        image=StrataImage(b'identity',b'worker',3,9,1,64,())
        with self.assertRaises(IntegrityError):
            self.backend.restore(self.lease,image,history=self.history,frontier=(0.,)*16)
        self.assertEqual(self.calls,[])

    def test_missing_embedding_fails_before_reset_or_forward(self):
        self.lease.bound=False; self.lease.consumed=0; self.lease.inputs=None
        with self.assertRaises(IntegrityError):
            self.backend.consume_inputs(self.lease,spans(),embeddings={})
        self.assertEqual(self.calls,[])

    def test_foreign_owner_cannot_ingest_or_restore_mixed_state(self):
        errors=[]
        def foreign():
            try:self.backend.consume_inputs(self.lease,spans(),embeddings={})
            except BaseException as error:errors.append(error)
        thread=threading.Thread(target=foreign);thread.start();thread.join()
        self.assertEqual(len(errors),1);self.assertIsInstance(errors[0],InvalidOperation)
        self.assertEqual(self.calls,[]);self.assertEqual(self.lease.inputs,self.history)

    def test_bad_completed_work_receipt_invalidates_mixed_execution(self):
        counters=iter(({'forward_tokens':9},{'forward_tokens':11}))
        self.backend._refresh_stats=lambda:next(counters)
        self.backend._request=lambda command:dict(status='ok',consumed=10,operator='batched-prefill',
                                                  position=9,count=1,chunks=[[9,1]])
        with self.assertRaises(IntegrityError):self.backend.consume(self.lease,[4])
        self.assertTrue(getattr(self.backend,'_protocol_failed',False))
        self.assertEqual(self.lease.inputs,self.history)


if __name__=='__main__':unittest.main()
