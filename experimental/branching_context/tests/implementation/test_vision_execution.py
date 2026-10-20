"""Mixed input/receipt contracts. CPU tests do not qualify numerical execution."""
import copy
from dataclasses import FrozenInstanceError
import hashlib
import struct
import unittest

from context_time_travel.input_spans import InputHistory
from context_time_travel.execution_receipt import tokens_digest, validate_input_schedule
from context_time_travel.vision_execution import input_binding
from context_time_travel.store import InvalidOperation


def spans(**changes):
    image = dict(kind='image', original_sha256='a'*64, normalized_sha256='b'*64,
        encoder_profile_id='c'*64, embedding_sha256='d'*64,
        embedding_bytes=20+6*4*4, rows=6, width=4, grid_width=3, grid_height=2,
        pad_token_id=7)
    image.update(changes)
    return [dict(kind='text', tokens=[1,2]), image, dict(kind='text', tokens=[3])]


def receipt(history, *, position=0, operator='batched-prefill'):
    count = len(history)-position
    chunks = [[position,count]] if operator == 'batched-prefill' else [[p,1] for p in range(position,len(history))]
    return dict(operator=operator, position=position, count=count, chunks=chunks,
        tokens_sha256=tokens_digest(history.native_tokens[position:]),
        inputs_sha256=history.digest, **input_binding(history))


class VisionExecutionTests(unittest.TestCase):
    def test_history_has_explicit_native_projection_and_text_runs(self):
        value = InputHistory(spans())
        self.assertEqual(len(value),9)
        self.assertEqual(value.native_tokens,(1,2,7,7,7,7,7,7,3))
        self.assertEqual(value.text_runs,((1,2),(3,)))
        with self.assertRaises(TypeError): list(value)

    def test_detached_immutable_view_does_not_change_canonical_spans(self):
        source=spans(); value=InputHistory(source); digest=value.digest
        source[0]['tokens'][0]=99
        detached=value.spans; detached[1]['embedding_sha256']='f'*64
        self.assertEqual(value.digest,digest)
        self.assertEqual(value.spans,spans())
        with self.assertRaises((FrozenInstanceError,AttributeError)): value._encoded=b'bad'

    def test_append_and_prefix_keep_images_and_reject_partial_image_boundaries(self):
        value=InputHistory(spans()); tail=value+(4,5)
        self.assertEqual(tail.prefix(9),value)
        self.assertEqual(value.prefix(2).native_tokens,(1,2))
        with self.assertRaises(InvalidOperation): value.prefix(3)
        with self.assertRaises(InvalidOperation): value.prefix(True)
        with self.assertRaises(InvalidOperation): value.prefix(10)
        self.assertEqual(value.spans,spans())

    def test_positions_follow_grid_extent_and_not_image_row_count(self):
        value=InputHistory(spans())
        points=[(0,0,0),(1,1,1),(2,2,2),(2,2,3),(2,2,4),
                (2,3,2),(2,3,3),(2,3,4),(5,5,5)]
        data=b'COW-MROPE-v1\0'+struct.pack('<QQ',9,6)
        data+=b''.join(struct.pack('<iii',*p) for p in points)
        actual=input_binding(value)
        self.assertEqual(actual['position_sha256'],hashlib.sha256(data).hexdigest())
        self.assertEqual(actual['rotary_cursor'],6)

    def test_identical_pad_tokens_cannot_alias_another_image(self):
        a=InputHistory(spans()); b=InputHistory(spans(embedding_sha256='e'*64))
        self.assertEqual(a.native_tokens,b.native_tokens)
        self.assertNotEqual(a.digest,b.digest)
        self.assertNotEqual(input_binding(a)['native_input_sha256'],input_binding(b)['native_input_sha256'])
        self.assertEqual(input_binding(a)['position_sha256'],input_binding(b)['position_sha256'])
        with self.assertRaises(ValueError): validate_input_schedule([receipt(a)],b,128)

    def test_native_identity_has_independently_encoded_payload_and_placement(self):
        value=InputHistory(spans())
        raw=b'COW-VISION-INPUT-v1\0'+struct.pack('<Q',9)
        raw+=struct.pack('<9i',*value.native_tokens)+struct.pack('<Q',1)
        raw+=struct.pack('<6q',2,6,3,2,4,7)+b'd'*64
        self.assertEqual(input_binding(value)['native_input_sha256'],hashlib.sha256(raw).hexdigest())

    def test_provenance_is_bound_even_when_model_payload_is_identical(self):
        a=InputHistory(spans()); b=InputHistory(spans(original_sha256='e'*64))
        self.assertEqual(input_binding(a),input_binding(b))
        self.assertNotEqual(a.digest,b.digest)
        with self.assertRaises(ValueError): validate_input_schedule([receipt(a)],b,128)

    def test_schedule_covers_each_consumed_prefix_and_generated_boundary(self):
        prefix=InputHistory(spans()); final=prefix+(4,5)
        events=[receipt(prefix),receipt(final,position=9,operator='normal-verifier-decode')]
        self.assertEqual(len(validate_input_schedule(events,final,128)),64)
        self.assertEqual(events[0]['inputs_sha256'],prefix.digest)

    def test_missing_altered_or_unknown_receipts_fail(self):
        value=InputHistory(spans()); valid=receipt(value)
        for key in ('inputs_sha256','native_input_sha256','position_sha256','rotary_cursor'):
            with self.subTest(key=key):
                bad=copy.deepcopy(valid); bad.pop(key)
                with self.assertRaises(ValueError): validate_input_schedule([bad],value,128)
                bad=copy.deepcopy(valid); bad[key]='0'*64
                with self.assertRaises(ValueError): validate_input_schedule([bad],value,128)
        bad=dict(valid,unexpected=True)
        with self.assertRaises(ValueError): validate_input_schedule([bad],value,128)

    def test_grid_and_placement_changes_invalidate_receipts(self):
        a=InputHistory(spans())
        for source in (spans(grid_width=2,grid_height=3),
                       [dict(kind='text',tokens=[1]),spans()[1],dict(kind='text',tokens=[2,3])]):
            with self.assertRaises(ValueError): validate_input_schedule([receipt(a)],InputHistory(source),128)

    def test_decode_cannot_claim_to_consume_image_rows(self):
        value=InputHistory(spans())
        with self.assertRaises(ValueError):
            validate_input_schedule([receipt(value,operator='normal-verifier-decode')],value,128)

    def test_request_boundary_cannot_end_on_image_row_but_native_chunk_can(self):
        value=InputHistory(spans()); event=receipt(value)
        event['chunks']=[[0,4],[4,4],[8,1]]
        self.assertTrue(validate_input_schedule([event],value,4))
        ending=InputHistory(spans()[:-1])
        with self.assertRaises(ValueError): validate_input_schedule([receipt(ending)],ending,128)


if __name__=='__main__': unittest.main()
