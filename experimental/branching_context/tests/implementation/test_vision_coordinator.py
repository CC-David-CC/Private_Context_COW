"""VC06 mixed-history serving controls; synthetic CPU backend, never GPU evidence."""
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from context_time_travel.coordinator import ContextCoordinator
from context_time_travel.input_spans import InputHistory
from context_time_travel.pools import PoolConfig
from context_time_travel.presentation_request import binding_for_snapshot
from context_time_travel.real import BranchingModel
from context_time_travel.store import BranchHead, Conflict, IntegrityError, InvalidOperation, _json_bytes
from serve.cow_adapter import CowAdapter
from serve.cow_prefix import RenderedPrefixMismatch
from test_vision_api_binding import MixedCPUBackend, NoArtifacts
from test_vision_execution import spans, receipt


class ServingMixedCPUBackend(MixedCPUBackend):
    @contextmanager
    def generate_request(self,lease,count):
        start=len(lease['inputs'])
        def tokens():
            for _ in range(count):
                token=self.vocab_size-1;self.consume(lease,[token]);yield token
        try:yield tokens()
        finally:self.last_execution_receipt=receipt(lease['inputs'],position=start,operator='normal-verifier-decode')


class VisionCoordinatorTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.profile=dict(model_contract_id='mixed-model',execution_profile_id='mixed-serving',
            model_weights_sha256='a'*64,execution=dict(GPU=[0],resume_contract='identical-mixed-native-operators-v1',
                generated_operator='normal-verifier-decode',native_request_protocol='consumed-greedy-ack-v1',
                prefill_chunk=128,embedding_width=4,capacity=128,eos_tokens=[]))
        self.models=[];self.coordinator=None;self.start()
        self.addCleanup(self.close)

    def close(self):
        if self.coordinator is not None:self.coordinator.close();self.coordinator=None

    def start(self):
        def factory(gpu):
            backend=ServingMixedCPUBackend(self.profile)
            model=BranchingModel(self.tmp.name,backend,self.profile);self.models.append(model)
            if not (model.store.references/'main.json').exists():
                root=model.create_inputs(spans(),artifacts=NoArtifacts(model.store))
                namespace=model.store.create_namespace('Image branches','namespace').namespace_id
                request=dict(schema=1,namespace_id=namespace,kind='root',
                    messages=[dict(id='a'*32,role='user',content='diagram')],renderer_id='b'*64,output_message_id=None)
                head=BranchHead('main',0,root,root)
                binding=binding_for_snapshot(model.store,request,operation_id='root',source_head=head,
                    target_snapshot=root,input_ids=InputHistory(spans()))
                model.store.fork('main',root,namespace_id=namespace,operation_id='root',presentation=binding)
                model.checkpoint(root)
            return model
        self.coordinator=ContextCoordinator(self.tmp.name,
            PoolConfig(gpus=(0,),ram_bytes=1024**2,disk_bytes=1024**2,min_available_bytes=0),factory)
        self.coordinator.ready(5);self.adapter=CowAdapter(self.coordinator)
        self.root=self.coordinator.store.head('main').root
        self.namespace=self.coordinator.store.branch_namespace('main')

    def presentation(self,kind='initial_turn',output='c'*32):
        return dict(schema=1,namespace_id=self.namespace,kind=kind,
            messages=[dict(id='d'*32,role='user',content='question')] if kind=='initial_turn' else [],
            renderer_id='b'*64,output_message_id=output)

    def answer(self,operation='question',branch='main'):
        request=self.adapter.generate(branch,InputHistory(spans())+(4,),0,operation,2,
            presentation=self.presentation())
        durable=request.durable.result(5)
        return {**durable,'answer':request.snapshot.result(5)}

    def test_source_and_adapter_reads_retain_typed_image_identity(self):
        head,history,state=self.coordinator.request_source_state('main',0,'read-source')
        self.assertEqual(head.tip,self.root);self.assertEqual(state,dict(status='idle'))
        self.assertEqual(history,InputHistory(spans()))
        self.assertEqual(self.adapter.history('main'),history)

    def test_queued_suffix_evaluates_only_new_text_and_publishes_mixed_receipt(self):
        before=self.models[-1].backend.stats['forward_tokens'];result=self.answer()
        self.assertEqual(self.models[-1].backend.stats['forward_tokens']-before,3)
        self.assertEqual(result['answer']['tokens'],[15,15])
        record=self.coordinator.committed_generation('question')
        self.assertEqual(record['generation']['schema'],8)
        self.assertEqual(record['generation']['source_input_kind'],'canonical-mixed-spans-v1')
        self.assertEqual(record['generation']['source_history_sha256'],InputHistory(spans()).digest)
        self.assertEqual(record['generation']['ids'],[4])
        self.assertEqual(record['presentation']['schema'],1)  # Original display request.
        self.assertEqual(record['receipt']['presentation']['schema'],3)  # Consumed binding.
        history=self.models[-1].history(result['branch']['tip'])
        self.assertEqual(history,InputHistory(spans())+(4,15,15))
        controller=self.models[-1].controller(result['branch']['tip'])
        self.assertNotIn('7',controller['seen'])

    def test_identical_native_pad_ids_cannot_substitute_a_different_image(self):
        changed=InputHistory(spans(embedding_sha256='e'*64))+(4,)
        self.assertEqual(changed.native_tokens,(InputHistory(spans())+(4,)).native_tokens)
        before=self.models[-1].backend.stats['forward_tokens']
        with self.assertRaises(RenderedPrefixMismatch):
            self.adapter.generate('main',changed,0,'wrong-image',2,presentation=self.presentation())
        self.assertEqual(self.models[-1].backend.stats['forward_tokens'],before)
        self.assertFalse((Path(self.tmp.name)/'pool-operations/wrong-image.json').exists())

    def test_lost_ack_reopen_replays_receipt_without_model_work(self):
        first=self.answer();self.close();self.start()
        before=self.models[-1].backend.stats['forward_tokens'];again=self.answer()
        self.assertEqual(self.models[-1].backend.stats['forward_tokens'],before)
        self.assertEqual(again['branch'],first['branch'])
        self.assertEqual(again['answer']['tokens'],[15,15])
        self.assertEqual(self.coordinator.store.head('main').revision,1)

    def test_flattened_source_digest_is_rejected_even_with_a_consistent_envelope(self):
        self.answer();path=Path(self.tmp.name)/'pool-operations/question.json';original=path.read_bytes()
        value=json.loads(original);generation=value['intent']['generation']
        generation['source_history_sha256']=hashlib.sha256(_json_bytes(list(InputHistory(spans()).native_tokens))).hexdigest()
        value['checksum']=hashlib.sha256(_json_bytes(value['intent'])).hexdigest();path.write_bytes(_json_bytes(value))
        try:
            with self.assertRaises(IntegrityError):self.coordinator.committed_generation('question')
        finally:path.write_bytes(original)

    def test_mixed_greedy_descriptor_cannot_claim_a_sampled_policy(self):
        self.answer();path=Path(self.tmp.name)/'pool-operations/question.json';original=path.read_bytes()
        value=json.loads(original)
        value['intent']['generation']['sampling']=dict(temperature=0.7,min_p=0.05,top_k=0,top_p=1.,repetition_penalty=1.)
        value['checksum']=hashlib.sha256(_json_bytes(value['intent'])).hexdigest();path.write_bytes(_json_bytes(value))
        try:
            with self.assertRaisesRegex(IntegrityError,'descriptor|greedy'):
                self.coordinator.committed_generation('question')
        finally:path.write_bytes(original)

    def test_text_continuation_does_not_accept_new_unleased_image_inputs(self):
        history=InputHistory(spans()+[spans()[1],dict(kind='text',tokens=[4])])
        before=self.models[-1].backend.stats['forward_tokens']
        with self.assertRaisesRegex(InvalidOperation,'text|image'):
            self.adapter.generate('main',history,0,'unleased-image',2,presentation=self.presentation())
        self.assertEqual(self.models[-1].backend.stats['forward_tokens'],before)
        self.assertFalse((Path(self.tmp.name)/'pool-operations/unleased-image.json').exists())

    def test_fork_and_typed_retry_keep_the_old_tip_and_full_image_prefix(self):
        first=self.answer();tip=first['branch']['tip']
        self.adapter.fork('sibling',self.root,source_branch='main',operation_id='fork-image')
        sibling=self.answer('sibling-question','sibling')
        self.assertEqual(sibling['answer']['tokens'],[15,15])
        result=self.adapter.rewind('main',1,self.root,'retry-root')
        self.assertEqual(result.tip,self.root)
        self.assertEqual(self.coordinator.store.get_snapshot(tip)['schema'],2)
        binding=self.coordinator.store.presentation_receipt('retry-root')['presentation']
        self.assertEqual(binding['schema'],3);self.assertEqual(binding['source']['token_count'],12)

    def test_stale_revision_rejects_before_model_or_intent_changes(self):
        self.answer();before=self.models[-1].backend.stats['forward_tokens']
        with self.assertRaises(Conflict):
            self.adapter.generate('main',InputHistory(spans())+(4,),0,'stale',2,presentation=self.presentation())
        self.assertEqual(self.models[-1].backend.stats['forward_tokens'],before)
        self.assertFalse((Path(self.tmp.name)/'pool-operations/stale.json').exists())


if __name__=='__main__':unittest.main()
