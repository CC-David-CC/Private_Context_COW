"""Owned, derived receipt validation. CPU proofs do not qualify GPU numerics."""
import copy
import threading
import unittest
from unittest.mock import patch

from context_time_travel.execution_receipt import (
    _InputScheduleCursor, validate_input_schedule, schedule_digest)
from context_time_travel.input_spans import InputHistory
from context_time_travel.vision_execution import input_binding
from test_vision_execution import spans, receipt


class InputScheduleCursorTests(unittest.TestCase):
    def setUp(self):
        self.root = InputHistory(spans())
        self.events = [receipt(self.root)]

    def cursor(self):
        return _InputScheduleCursor(self.events, self.root, 128)

    def test_append_matches_unchanged_full_validator_on_independent_branches(self):
        for tail in ((4,5,6), (11,12,13,14)):
            cur = self.cursor(); history = self.root; events = copy.deepcopy(self.events)
            for token in tail:
                old = len(history); history += (token,)
                event = receipt(history, position=old, operator='normal-verifier-decode')
                events.append(event); cur.append(event, history)
            actual = cur.finish()
            self.assertEqual(actual, tuple(events))
            self.assertEqual(validate_input_schedule(actual, history, 128), schedule_digest(events))
        self.assertEqual(self.root, InputHistory(spans()))

    def test_prefill_extension_can_add_a_complete_new_image(self):
        cur = self.cursor(); final = self.root + InputHistory(spans(embedding_sha256='e'*64))
        event = receipt(final, position=len(self.root)); cur.append(event, final)
        self.assertTrue(validate_input_schedule(cur.finish(), final, 128))

    def test_constructor_rejects_missing_altered_or_noncontiguous_root_schedule(self):
        for events in ([], [dict(self.events[0], position=1)],
                       [dict(self.events[0], inputs_sha256='f'*64)]):
            with self.subTest(events=events), self.assertRaises(ValueError):
                _InputScheduleCursor(events, self.root, 128)

    def test_append_rejects_every_altered_binding_or_native_boundary(self):
        final = self.root + (4,)
        event = receipt(final, position=len(self.root), operator='normal-verifier-decode')
        changes = dict(operator='unknown', position=0, count=True, chunks=[[9,2]],
                       tokens_sha256='f'*64, inputs_sha256='f'*64,
                       native_input_sha256='f'*64, position_sha256='f'*64,
                       rotary_cursor=True, extra='bad')
        for key, value in changes.items():
            with self.subTest(key=key):
                cur = self.cursor(); bad = dict(event, **{key:value})
                with self.assertRaises(ValueError): cur.append(bad, final)
                with self.assertRaises(ValueError): cur.finish()
        for key in event:
            with self.subTest(missing=key):
                cur=self.cursor(); bad=dict(event); bad.pop(key)
                with self.assertRaises(ValueError): cur.append(bad, final)

    def test_same_length_tokens_cannot_rewrite_owned_ancestry(self):
        for source in (spans(embedding_sha256='e'*64), spans(original_sha256='e'*64),
                       [dict(kind='text',tokens=[2,1]),spans()[1],dict(kind='text',tokens=[3])]):
            final=InputHistory(source)+(4,)
            with self.subTest(source=source), self.assertRaises(ValueError):
                self.cursor().append(receipt(final, position=len(self.root)),final)

    def test_decode_rejects_new_images_and_requests_ending_on_image_rows(self):
        for delta, operator in ((InputHistory(spans()),'normal-verifier-decode'),
                                 (InputHistory(spans()[:-1]),'batched-prefill')):
            cur=self.cursor(); final=self.root+delta
            with self.subTest(operator=operator), self.assertRaises(ValueError):
                cur.append(receipt(final,position=len(self.root),operator=operator),final)

    def test_input_and_export_aliases_do_not_mutate_private_proof(self):
        cur=self.cursor(); self.events[0]['chunks'][0][0]=99
        final=self.root+(4,); event=receipt(final,position=len(self.root))
        cur.append(event,final); event['chunks'][0][0]=99
        actual=cur.finish(); self.assertTrue(validate_input_schedule(actual,final,128))
        actual[0]['chunks'][0][0]=88
        with self.assertRaises(ValueError): cur.finish()

    def test_foreign_thread_cannot_append_or_close_but_owner_can_continue(self):
        cur=self.cursor(); final=self.root+(4,); event=receipt(final,position=len(self.root))
        errors=[]
        def foreign():
            for operation in (lambda:cur.append(event,final),cur.finish):
                try: operation()
                except ValueError as error: errors.append(str(error))
        worker=threading.Thread(target=foreign); worker.start(); worker.join(timeout=5)
        self.assertFalse(worker.is_alive()); self.assertEqual(len(errors),2)
        cur.append(event,final); self.assertTrue(validate_input_schedule(cur.finish(),final,128))

    def test_invalid_append_poisoned_and_closed_cursor_cannot_be_reused(self):
        cur=self.cursor()
        with self.assertRaises(ValueError): cur.append(None,self.root)
        final=self.root+(4,)
        with self.assertRaises(ValueError): cur.append(receipt(final,position=9),final)
        with self.assertRaises(ValueError): cur.finish()
        cur=self.cursor(); cur.finish()
        with self.assertRaises(ValueError): cur.append(receipt(final,position=9),final)

    def test_append_hashes_only_current_binding_full_audit_at_open_and_finish(self):
        from context_time_travel import execution_receipt as er, vision_execution as ve
        cur=self.cursor(); history=self.root
        with patch.object(er,'validate_input_schedule',wraps=er.validate_input_schedule) as full:
            with patch.object(ve,'input_binding',wraps=ve.input_binding) as binding:
                for token in range(20,25):
                    old=len(history); history+=(token,); event=receipt(history,position=old)
                    before=binding.call_count;cur.append(event,history)
                    self.assertEqual(binding.call_count-before,1)
                self.assertEqual(full.call_count,0)
            actual=cur.finish(); self.assertEqual(full.call_count,1)
        self.assertTrue(validate_input_schedule(actual,history,128))


class InputScheduleRequestIntegrationTests(unittest.TestCase):
    def setUp(self):
        from test_vision_grammar_request import VisionGrammarRequestTests
        self.fixture=VisionGrammarRequestTests();self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.model=self.fixture.model;self.backend=self.fixture.backend

    def test_generation_uses_incremental_receipts_and_full_validation_before_capture(self):
        from context_time_travel import execution_receipt as er
        old_save=self.model._save;full_audits=[]
        original=er.validate_input_schedule
        def audit(*args,**kwargs):
            full_audits.append(len(args[0]));return original(*args,**kwargs)
        def save(*args,**kwargs):
            self.assertEqual(full_audits[-1],5)
            return old_save(*args,**kwargs)
        self.model._save=save
        with patch.object(er,'validate_input_schedule',side_effect=audit):
            with patch.object(self.model,'_completed_schedule',wraps=self.model._completed_schedule) as completed:
                result=self.fixture.request(ids=[4])
                self.assertEqual(completed.call_count,1)
        self.assertEqual(result['tokens'],[0,0,6])
        self.assertEqual(full_audits,[1,2,2,5])

    def test_bad_native_extension_cannot_publish_and_original_owner_can_retry(self):
        from context_time_travel.store import IntegrityError
        before=set(self.model.store.objects.iterdir());parent=self.model.snapshot_bytes(self.fixture.root)
        consume=self.backend.consume_generated
        def corrupt(*args,**kwargs):
            consume(*args,**kwargs);self.backend.last_execution_receipt['position_sha256']='0'*64
        with patch.object(self.backend,'consume_generated',side_effect=corrupt):
            with self.assertRaises(IntegrityError):self.fixture.request()
        self.assertEqual(set(self.model.store.objects.iterdir()),before)
        self.assertEqual(self.model.snapshot_bytes(self.fixture.root),parent)
        self.assertEqual(self.fixture.request()['tokens'],[0,0,6])

    def test_failed_final_audit_cannot_publish_streamed_progress(self):
        from context_time_travel.store import IntegrityError
        from context_time_travel import execution_receipt as er
        before=set(self.model.store.objects.iterdir());seen=[]
        def reject(_):raise ValueError('synthetic full publication audit failure')
        with patch.object(er._InputScheduleCursor,'finish',reject):
            with self.assertRaises(IntegrityError):self.fixture.request(on_token=seen.append)
        self.assertEqual(seen,[0,0,6])
        self.assertEqual(set(self.model.store.objects.iterdir()),before)
        self.assertEqual(self.fixture.request()['tokens'],[0,0,6])


if __name__=='__main__': unittest.main()
