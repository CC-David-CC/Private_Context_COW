"""Independent binding equivalence and complete-receipt rejection, CPU only."""
import copy,random,unittest
from unittest.mock import patch
from context_time_travel.input_spans import InputHistory
from context_time_travel.vision_execution import input_binding
from context_time_travel.execution_receipt import validate_input_schedule,schedule_digest,tokens_digest
from test_vision_execution import spans,receipt


def events_for(history,boundaries):
    events=[];start=0
    for end in boundaries:
        prefix=history.prefix(end);count=end-start
        chunks=[];pos=start
        while pos<end:
            size=min(128,end-pos);chunks.append([pos,size]);pos+=size
        events.append(dict(operator='batched-prefill',position=start,count=count,chunks=chunks,
            tokens_sha256=tokens_digest(history.native_tokens[start:end]),
            inputs_sha256=prefix.digest,**input_binding(prefix)))
        start=end
    return events

class ScheduleProjectionTests(unittest.TestCase):
    def test_complete_validation_does_not_reconstruct_prefix_histories_per_event(self):
        history=InputHistory([dict(kind='text',tokens=list(range(320)))])
        events=events_for(history,range(1,321))
        with patch.object(InputHistory,'prefix',side_effect=AssertionError('full prefix rebuilt per receipt')):
            self.assertEqual(validate_input_schedule(events,history,128),schedule_digest(events))
    def test_projection_matches_unchanged_reference_at_every_text_boundary(self):
        from context_time_travel.input_schedule_projection import _InputScheduleProjection
        cases=[InputHistory([dict(kind='text',tokens=[0,7,2147483647])]),InputHistory(spans()),
            InputHistory([spans()[1],dict(kind='text',tokens=[7,8]),spans(embedding_sha256='e'*64)[1],dict(kind='text',tokens=[9])]),
            InputHistory([dict(kind='text',tokens=[]),dict(kind='text',tokens=[2]),dict(kind='text',tokens=[3])])]
        for history in cases:
            ends=[i for i in range(1,len(history)+1) if self.is_text_end(history,i)]
            projection=_InputScheduleProjection(history,ends)
            for end in ends:
                with self.subTest(history=history.digest,end=end):
                    prefix=history.prefix(end)
                    self.assertEqual(projection.binding(end),dict(inputs_sha256=prefix.digest,**input_binding(prefix)))
            self.assertEqual(projection.native_tokens,history.native_tokens)
    @staticmethod
    def is_text_end(history,count):
        from context_time_travel.store import InvalidOperation
        try:prefix=history.prefix(count)
        except InvalidOperation:return False
        return bool(prefix.spans and prefix.spans[-1]['kind']=='text')
    def test_seeded_mixed_schedules_match_independent_prefix_reference(self):
        rng=random.Random(90217)
        for trial in range(25):
            inputs=[]
            for i in range(rng.randrange(1,6)):
                inputs.append(dict(kind='text',tokens=[rng.randrange(10000) for _ in range(rng.randrange(1,12))]))
                if rng.randrange(2):
                    image=spans()[1].copy();image['grid_width']=rng.randrange(1,5);image['grid_height']=rng.randrange(1,4)
                    image['rows']=image['grid_width']*image['grid_height'];image['embedding_bytes']=20+image['rows']*image['width']*4
                    image['original_sha256']=format(trial+1,'064x');inputs.append(image)
            inputs.append(dict(kind='text',tokens=[7,7,31]))
            history=InputHistory(inputs);ends=[i for i in range(1,len(history)+1) if self.is_text_end(history,i)]
            events=events_for(history,ends)
            self.assertEqual(validate_input_schedule(events,history,128),schedule_digest(events))
    def test_every_binding_on_an_early_receipt_is_checked(self):
        history=InputHistory(spans())+(11,12,13);events=events_for(history,[2,9,10,11,12])
        for key in ['inputs_sha256','native_input_sha256','position_sha256','rotary_cursor']:
            bad=copy.deepcopy(events);bad[0][key]=True if key=='rotary_cursor' else '0'*64
            with self.subTest(key=key),self.assertRaises(ValueError):validate_input_schedule(bad,history,128)
    def test_removed_reordered_and_duplicate_receipts_are_rejected(self):
        history=InputHistory(spans())+(11,12,13);events=events_for(history,[2,9,10,11,12])
        for bad in [events[1:],events[:-1],events[:1]+events[2:3]+events[1:2]+events[3:],events+events[-1:]]:
            with self.subTest(bad=bad),self.assertRaises(ValueError):validate_input_schedule(bad,history,128)
    def test_image_provenance_embedding_geometry_and_placement_remain_distinct(self):
        history=InputHistory(spans());events=events_for(history,[2,9])
        variants=[spans(original_sha256='f'*64),spans(embedding_sha256='e'*64),spans(grid_width=2,grid_height=3),
            [dict(kind='text',tokens=[1]),spans()[1],dict(kind='text',tokens=[2,3])]]
        for value in variants:
            with self.subTest(value=value),self.assertRaises(ValueError):validate_input_schedule(events,InputHistory(value),128)
    def test_decode_cannot_swallow_images_but_prefill_chunks_can_split_rows(self):
        history=InputHistory(spans());event=receipt(history);event['chunks']=[[0,4],[4,4],[8,1]]
        self.assertEqual(validate_input_schedule([event],history,4),schedule_digest([event]))
        with self.assertRaises(ValueError):validate_input_schedule([receipt(history,operator='normal-verifier-decode')],history,128)
    def test_request_boundaries_inside_or_at_end_of_image_are_rejected(self):
        history=InputHistory(spans())
        for end in range(3,9):
            bad=receipt(history);bad['count']=end;bad['chunks']=[[0,end]]
            with self.subTest(end=end),self.assertRaises(ValueError):validate_input_schedule([bad],history,128)
    def test_native_token_and_chunk_validation_not_bypassed(self):
        history=InputHistory(spans());valid=receipt(history)
        for changes in [dict(tokens_sha256='0'*64),dict(operator='unknown'),dict(count=True),dict(position=True),dict(chunks=[]),dict(chunks=[[0,8]]),dict(chunks=[[0,10]])]:
            with self.subTest(changes=changes),self.assertRaises(ValueError):validate_input_schedule([dict(valid,**changes)],history,128)
    def test_empty_and_unknown_receipt_layout_keep_existing_contract(self):
        history=InputHistory([]);self.assertEqual(validate_input_schedule([],history,128),schedule_digest([]))
        history=InputHistory(spans())
        for row in [dict(receipt(history),unknown=True),{},None]:
            with self.subTest(row=row),self.assertRaises(ValueError):validate_input_schedule([row],history,128)
