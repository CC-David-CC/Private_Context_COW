"""VC06 canonical publication prerequisites; metadata/controller CPU fixtures."""
import copy
import tempfile
import unittest

from context_time_travel import controller
from context_time_travel.input_spans import InputHistory, history_for_snapshot
from context_time_travel.presentation_binding import validate_binding
from context_time_travel.presentation_request import binding_for_snapshot
from context_time_travel.store import Store, BranchHead, InvalidOperation, _hash, _json_bytes
from test_vision_input_history import image_span


class VisionPresentationTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.store=Store(self.tmp.name)
        self.namespace=self.store.create_namespace('Image fixture','namespace').namespace_id
        self.spans=[dict(kind='text',tokens=[1,2]),image_span(),dict(kind='text',tokens=[3])]
        self.root=self.store.create_input_root('mixed-model',self.spans,dict(status='idle'))
        self.head=BranchHead('main',0,self.root,self.root)
        self.request=dict(schema=1,namespace_id=self.namespace,kind='root',
            messages=[dict(id='a'*32,role='user',content='The retained diagram')],
            renderer_id='b'*64,output_message_id=None)

    def binding(self,request=None,head=None,target=None,ids=()):
        return binding_for_snapshot(self.store,request or self.request,operation_id='op',
            source_head=head or self.head,target_snapshot=target or self.root,
            input_ids=InputHistory(self.spans) if request is None else ids)

    def validate(self,value,target=None,head=None,namespace=None):
        return validate_binding(self.store,head or self.head,target or self.root,value,
            namespace_id=namespace or self.namespace,operation_id='op')

    def generated(self,delta=(),kind='answer'):
        state=controller.start(dict(temperature=0,eos_tokens=[]),'op',[1,2,3,*delta],32)
        state=controller.consume_many(state,[(7,b'a',False),(8,b'b',False)])
        target=self.store.append_inputs(self.root,[dict(kind='text',tokens=[*delta,7,8])],state)
        request={**self.request,'kind':kind,'messages':[], 'output_message_id':'c'*32}
        if kind=='initial_turn':request['messages']=[dict(id='d'*32,role='user',content='new question')]
        return target,request

    def test_root_binds_full_image_identity_and_consumed_cells(self):
        value=self.binding()
        self.assertEqual(value['schema'],3)
        self.assertEqual(value['input_span'],dict(start=0,count=9,inputs_sha256=InputHistory(self.spans).digest))
        self.assertEqual(self.validate(value),value)
        self.assertEqual(self.store.get_snapshot(self.root)['inputs'],self.spans)

    def test_equal_pad_tokens_cannot_substitute_another_embedding(self):
        value=self.binding();changed=copy.deepcopy(self.spans)
        changed[1]['embedding_sha256']='e'*64
        other=self.store.create_input_root('mixed-model',changed,dict(status='idle'))
        self.assertEqual(InputHistory(changed).native_tokens,InputHistory(self.spans).native_tokens)
        with self.assertRaisesRegex(InvalidOperation,'digest'):
            self.validate(value,other,BranchHead('main',0,other,other))

    def test_legacy_token_binding_cannot_claim_a_mixed_root(self):
        value=self.binding();value['schema']=1
        value['input_span']=dict(start=0,count=9,token_ids_sha256=_hash(_json_bytes(list(InputHistory(self.spans).native_tokens))))
        with self.assertRaises(InvalidOperation):self.validate(value)

    def test_root_request_cannot_silently_drop_a_generation_seed(self):
        request={**self.request,'schema':2,'seed':17}
        with self.assertRaisesRegex(InvalidOperation,'seed'):
            self.binding(request,ids=InputHistory(self.spans))

    def test_answer_output_counts_only_actual_generated_text(self):
        target,request=self.generated()
        value=self.binding(request,target=target)
        self.assertEqual(value['source']['token_count'],9)
        self.assertEqual(value['input_span'],dict(start=9,count=0,inputs_sha256=InputHistory([]).digest))
        self.assertEqual(value['output_span']['start'],9)
        self.assertEqual(value['output_span']['count'],2)
        self.assertEqual(self.validate(value,target),value)

    def test_new_question_binds_its_exact_text_delta_after_the_image_prefix(self):
        target,request=self.generated((4,5),'initial_turn')
        value=self.binding(request,target=target,ids=(4,5))
        self.assertEqual(value['input_span'],dict(start=9,count=2,
            inputs_sha256=InputHistory([dict(kind='text',tokens=[4,5])]).digest))
        self.assertEqual(value['output_span']['start'],11)
        self.assertEqual(value['output_span']['count'],2)
        self.assertEqual(self.validate(value,target),value)

    def test_generated_output_cannot_hide_an_image_span(self):
        target,request=self.generated()
        value=self.binding(request,target=target)
        state=self.store.get_snapshot(target)['controller']
        bad=self.store.append_inputs(self.root,[image_span(),dict(kind='text',tokens=[7,8])],state)
        value['output_span']['count']=8
        with self.assertRaisesRegex(InvalidOperation,'output|text'):
            self.validate(value,bad)

    def test_history_segments_keep_identity_and_reject_partial_images(self):
        history=history_for_snapshot(self.store,self.root)
        self.assertIsInstance(history,InputHistory)
        self.assertEqual(history.segment(2,8),InputHistory([self.spans[1]]))
        self.assertEqual(history.segment(8,9),InputHistory([dict(kind='text',tokens=[3])]))
        for start,end in ((1,3),(3,8),(3,4)):
            with self.subTest(start=start,end=end),self.assertRaises(InvalidOperation):history.segment(start,end)

    def test_root_and_answer_publish_and_reopen_as_the_same_canonical_receipts(self):
        value=self.binding()
        head=self.store.fork('main',self.root,namespace_id=self.namespace,operation_id='root',presentation=value)
        target,request=self.generated()
        answer=self.binding(request,head=head,target=target)
        self.store.move('main',0,target,'op',presentation=answer)
        receipt=Store(self.tmp.name).presentation_receipt('op')
        self.assertEqual(receipt['presentation'],answer)
        self.assertEqual(receipt['result']['tip'],target)
        self.assertEqual(Store(self.tmp.name).head('main').revision,1)

    def test_namespace_or_changed_digest_rejections_do_not_modify_history(self):
        value=self.binding();before={p.name:p.read_bytes() for p in self.store.objects.glob('*.json')}
        for modified,namespace in ((value,'f'*32),({**value,'input_span':dict(start=0,count=9,inputs_sha256='f'*64)},None)):
            with self.assertRaises(InvalidOperation):self.validate(modified,namespace=namespace)
        self.assertEqual(before,{p.name:p.read_bytes() for p in self.store.objects.glob('*.json')})

    def test_legacy_history_view_and_root_descriptor_remain_unchanged(self):
        root=self.store.create_root('legacy',[1,2,3],dict(status='idle'))
        head=BranchHead('legacy',0,root,root)
        self.assertEqual(history_for_snapshot(self.store,root),(1,2,3))
        value=binding_for_snapshot(self.store,self.request,operation_id='op',source_head=head,
            target_snapshot=root,input_ids=(1,2,3))
        self.assertEqual(value['schema'],1)
        self.assertEqual(value['input_span'],dict(start=0,count=3,token_ids_sha256=_hash(_json_bytes([1,2,3]))))


if __name__=='__main__':unittest.main()
