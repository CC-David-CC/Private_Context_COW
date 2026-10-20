"""Synthetic control-plane tests; GPU and durable resume need separate evidence."""
from contextlib import contextmanager
import hashlib
import json
import unittest
import test_durable_restore as fixtures
from context_time_travel.execution_receipt import tokens_digest, schedule_digest
from context_time_travel.store import InvalidOperation, _json_bytes


def event(operator, position, tokens):
    return dict(operator=operator,position=position,count=len(tokens),tokens_sha256=tokens_digest(tokens),
                chunks=[[position,len(tokens)]] if operator == 'batched-prefill'
                else [[position+i,1] for i in range(len(tokens))])


class ResidentRequestPublicationTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.DurableRestoreControlTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.model, self.backend, self.sid = self.fixture.model,self.fixture.backend,self.fixture.sid
        self.fixture.persist()
        self.model.profile['execution'] = dict(resume_contract='identical-native-operators-v1',
            prefill_chunk=8,capacity=32,eos_tokens=[1],generated_operator='normal-verifier-decode',
            native_request_protocol='consumed-greedy-ack-v1')
        self.model._allocation_fault = lambda *args: None
        self.model._lease_fault = lambda *args: None
        path = self.model.checkpoint_directory/(self.sid+'.json')
        envelope = json.loads(path.read_bytes())
        schedule = [event('batched-prefill',0,[0,1])]
        envelope['manifest'].update(schema=3,execution_schedule=schedule,schedule_sha256=schedule_digest(schedule))
        envelope['checksum'] = hashlib.sha256(_json_bytes(envelope['manifest'])).hexdigest()
        path.write_bytes(_json_bytes(envelope))
        self.backend.piece = lambda token: bytes([65+token])
        self.backend.is_eog = lambda token: token == 1
        self.backend.captures = 0
        self.backend.restores = 0
        original_restore = self.backend.restore
        def restore(*args, **kwargs):
            self.backend.restores += 1
            return original_restore(*args,**kwargs)
        self.backend.restore = restore
        def consume(lease, tokens):
            self.backend.last_execution_receipt = event('batched-prefill',self.backend.count,tokens)
            self.backend.count += len(tokens)
            self.backend.stats['consumed_tokens'] += len(tokens)
        self.backend.consume = consume
        @contextmanager
        def generate(lease, limit):
            start = self.backend.count
            def outputs():
                for token in [0,1][:limit]:
                    self.backend.count += 1
                    self.backend.stats['consumed_tokens'] += 1
                    yield token
                self.backend.last_execution_receipt = event('normal-verifier-decode',start,[0,1][:limit])
            yield outputs()
        self.backend.generate_request = generate
        def export(lease):
            self.backend.captures += 1
            return b'complete-native-state'
        self.backend.export = export
        self.backend.park = lambda lease: None

    def test_consumed_progress_is_published_once_at_request_boundary(self):
        before = len(list(self.model.store.objects.iterdir()))
        observed = []
        def progress(token):
            observed.append((token,self.backend.count))
            self.assertEqual(len(list(self.model.store.objects.iterdir())),before)
        result = self.model.run_greedy_request(self.sid,[0,0],2,stream='test',on_token=progress)
        self.assertEqual(observed,[(0,5),(1,6)])
        self.assertEqual(tuple(result['tokens']),(0,1))
        self.assertEqual(self.model.history(result['snapshot']),(0,1,0,0,0,1))
        self.assertEqual(len(list(self.model.store.objects.iterdir())),before+1)
        self.assertEqual((self.backend.restores,self.backend.captures),(1,1))
        decoder = self.model.controller(result['snapshot'])
        self.assertEqual(decoder['status'],'stopped')
        self.assertEqual(decoder['next_draw'],0)
        self.assertEqual(decoder['accepted_in_scope'],2)
        self.assertEqual(decoder['seen'],{'0':4,'1':2})
        self.assertEqual(len(self.model.cache[result['snapshot']].execution_schedule),3)

    def test_progress_exception_preserves_canonical_parent_and_publishes_nothing(self):
        before = set(self.model.store.objects.iterdir())
        original = self.model.snapshot_bytes(self.sid)
        def abort(token): raise RuntimeError('consumer disconnected')
        with self.assertRaisesRegex(RuntimeError,'consumer disconnected'):
            self.model.run_greedy_request(self.sid,[0],2,stream='test',on_token=abort)
        self.assertEqual(set(self.model.store.objects.iterdir()),before)
        self.assertEqual(self.model.snapshot_bytes(self.sid),original)
        self.assertEqual(self.backend.captures,0)

    def test_empty_suffix_continues_controller_instead_of_resetting_scope(self):
        first = self.model.run_greedy_request(self.sid,[0],1,stream='first')
        decoder = self.model.controller(first['snapshot'])
        decoder['next_draw'] = 37
        resumed = self.model.store.append(first['snapshot'],(),decoder)
        self.model.cache[resumed] = self.model.cache[first['snapshot']]
        second = self.model.run_greedy_request(resumed,[],1,stream='ignored-for-continuation')
        actual = self.model.controller(second['snapshot'])
        self.assertEqual(actual['accepted_in_scope'],2)
        self.assertEqual(actual['next_draw'],37)
        self.assertEqual(actual['stream_key'],decoder['stream_key'])

    def test_unsupported_existing_sampler_is_rejected_before_native_restore(self):
        from context_time_travel import controller
        decoder = controller.start({'temperature':1},'sampled',(0,1),2)
        sid = self.model.store.append(self.sid,(),decoder)
        with self.assertRaises(InvalidOperation):
            self.model.run_greedy_request(sid,[],2,stream='test')
        self.assertEqual(self.backend.restores,0)

    def test_invalid_bound_and_incompatible_contract_fail_before_execution(self):
        for count in (-1,True,100):
            with self.assertRaises(InvalidOperation):
                self.model.run_greedy_request(self.sid,[0],count,stream='test')
        self.model.profile['execution']['resume_contract'] = 'identical-native-batches-v1'
        with self.assertRaises(InvalidOperation):
            self.model.run_greedy_request(self.sid,[0],2,stream='test')
        self.assertEqual(self.backend.restores,0)

    def test_native_operator_checkpoint_does_not_require_a_duplicate_legacy_ram_image(self):
        def refuse_legacy_copy(lease):
            raise MemoryError('legacy RAM cache budget exhausted')
        self.backend.park = refuse_legacy_copy
        result = self.model.run_greedy_request(self.sid,[0],2,stream='test')
        self.assertEqual(self.model.history(result['snapshot']),(0,1,0,0,1))
        self.assertEqual(self.backend.captures,1)


if __name__ == '__main__': unittest.main()
