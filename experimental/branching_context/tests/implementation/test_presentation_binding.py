"""Typed presentation spans refer to canonical consumed history, not another head."""
import copy
import hashlib
import tempfile
import unittest

from context_time_travel import Store, InvalidOperation
from context_time_travel import controller
from context_time_travel.store import _hash, _json_bytes
from context_time_travel.presentation_binding import validate_binding


class PresentationBindingTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.store = Store(self.tmp.name)
        self.namespace = 'a'*32
        self.operation = 'answer-next'
        stopped = controller.start(dict(temperature=0,eos_tokens=[14]), 'prior', [1,2],16)
        stopped = controller.consume_many(stopped, [(14,b'',True)])
        self.root = self.store.create_root('model',[1,2,14],stopped)
        self.head = self.store.fork('main',self.root)
        active = controller.start(dict(temperature=0,eos_tokens=[14]),self.operation,[1,2,14,3,4],16)
        active = controller.consume_many(active,[(5,b'a',False),(6,b'b',False)])
        self.target = self.store.append(self.root,[3,4,5,6],active)
        self.binding = dict(schema=1,namespace_id=self.namespace,kind='user_turn',
            messages=[dict(id='b'*32,role='user',content='next question')],
            source=dict(snapshot=self.root,revision=0,token_count=3),
            input_span=dict(start=3,count=2,token_ids_sha256=_hash(_json_bytes([3,4]))),
            output_span=dict(start=5,count=2,message_id='c'*32),
            renderer_id='d'*64,turn_complete=False)

    def validate(self,binding=None,target=None,head=None,operation=None):
        return validate_binding(self.store,head or self.head,target or self.target,
            binding or self.binding,namespace_id=self.namespace,operation_id=operation or self.operation)

    def test_valid_user_turn_is_detached_and_does_not_move_a_head(self):
        validated = self.validate()
        self.assertEqual(validated,self.binding)
        validated['messages'][0]['content']='changed'
        self.assertEqual(self.binding['messages'][0]['content'],'next question')
        self.assertEqual(self.store.head('main'),self.head)

    def test_rejects_wrong_namespace_source_revision_spans_or_digest(self):
        changes = [dict(namespace_id='e'*32), dict(source=dict(snapshot=self.root,revision=1,token_count=3)),
            dict(input_span=dict(start=3,count=2,token_ids_sha256='e'*64)),
            dict(output_span=dict(start=4,count=2,message_id='c'*32)),
            dict(output_span=dict(start=5,count=1,message_id='c'*32))]
        for change in changes:
            with self.subTest(change=change), self.assertRaises(InvalidOperation):
                self.validate({**self.binding,**change})

    def test_rejects_hidden_fields_roles_and_non_plain_message_content(self):
        for change in (dict(extra='hidden'),dict(messages=[dict(id='b'*32,role='tool',content='x')]),
                       dict(messages=[dict(id='b'*32,role='user',content=['image'])])):
            with self.subTest(change=change),self.assertRaises(InvalidOperation):
                self.validate({**self.binding,**change})

    def test_resume_preserves_scope_and_does_not_invent_a_user_message(self):
        self.head=self.store.move('main',0,self.target,'prior-move')
        state=self.store.get_snapshot(self.target)['controller']
        later=controller.consume_many(state,[(7,b'c',False)])
        target=self.store.append(self.target,[7],later)
        binding={**self.binding,'kind':'resume','messages':[],
            'source':dict(snapshot=self.target,revision=1,token_count=7),
            'input_span':dict(start=7,count=0,token_ids_sha256=_hash(_json_bytes([]))),
            'output_span':dict(start=7,count=1,message_id='c'*32)}
        self.assertEqual(self.validate(binding,target=target,operation='resume-op'),binding)
        invalid=copy.deepcopy(binding);invalid['messages']=self.binding['messages']
        with self.assertRaises(InvalidOperation):self.validate(invalid,target=target)
        changed=copy.deepcopy(later);changed['stream_key']=hashlib.sha256(b'new-scope').hexdigest()
        wrong=self.store.append(self.target,[7],changed)
        with self.assertRaises(InvalidOperation):self.validate(binding,target=wrong)

    def test_claimed_turn_completion_matches_consumed_controller(self):
        with self.assertRaises(InvalidOperation):self.validate({**self.binding,'turn_complete':True})
        state=self.store.get_snapshot(self.target)['controller']
        stopped=controller.consume_many(state,[(14,b'',True)])
        ended=self.store.append(self.root,[3,4,5,6,14],stopped)
        binding={**self.binding,'turn_complete':True,
                 'output_span':dict(start=5,count=3,message_id='c'*32)}
        self.assertEqual(self.validate(binding,target=ended),binding)

    def test_output_scope_cannot_belong_to_a_different_operation(self):
        with self.assertRaises(InvalidOperation):self.validate(operation='different-op')

    def test_boolean_controller_progress_cannot_masquerade_as_one_output(self):
        state=controller.start(dict(temperature=0,eos_tokens=[14]),self.operation,[1,2,14,3,4],16)
        state=controller.consume_many(state,[(5,b'a',False)])
        state['accepted_in_scope']=True
        target=self.store.append(self.root,[3,4,5],state)
        binding={**self.binding,'output_span':dict(start=5,count=1,message_id='c'*32)}
        with self.assertRaises(InvalidOperation):self.validate(binding,target=target)

    def test_root_binding_refers_to_actual_prefix_without_output(self):
        root=self.store.create_root('model',[1,2],{'status':'idle'})
        head=self.store.fork('prefix',root)
        binding={**self.binding,'kind':'root',
            'source':dict(snapshot=None,revision=0,token_count=0),
            'input_span':dict(start=0,count=2,token_ids_sha256=_hash(_json_bytes([1,2]))),
            'output_span':None}
        self.assertEqual(self.validate(binding,target=root,head=head),binding)
        invalid=copy.deepcopy(binding);invalid['output_span']=self.binding['output_span']
        with self.assertRaises(InvalidOperation):self.validate(invalid,target=root,head=head)

    def test_initial_answer_can_consume_only_output_after_prepared_prompt(self):
        root=self.store.create_root('model',[1,2],{'status':'idle'})
        head=self.store.fork('prepared-answer',root)
        state=controller.start(dict(temperature=0,eos_tokens=[14]),self.operation,[1,2],16)
        state=controller.consume_many(state,[(5,b'a',False)])
        target=self.store.append(root,[5],state)
        binding={**self.binding,'kind':'answer','messages':[],
            'source':dict(snapshot=root,revision=0,token_count=2),
            'input_span':dict(start=2,count=0,token_ids_sha256=_hash(_json_bytes([]))),
            'output_span':dict(start=2,count=1,message_id='c'*32)}
        self.assertEqual(self.validate(binding,target=target,head=head),binding)

    def test_first_question_appends_to_idle_codebase_prefix(self):
        root=self.store.create_root('model',[1,2],{'status':'idle'})
        head=self.store.fork('first-question',root)
        state=controller.start(dict(temperature=0,eos_tokens=[14]),self.operation,[1,2,3,4],16)
        state=controller.consume_many(state,[(5,b'a',False),(6,b'b',False)])
        target=self.store.append(root,[3,4,5,6],state)
        binding={**self.binding,'kind':'initial_turn',
            'source':dict(snapshot=root,revision=0,token_count=2),
            'input_span':dict(start=2,count=2,token_ids_sha256=_hash(_json_bytes([3,4]))),
            'output_span':dict(start=4,count=2,message_id='c'*32)}
        self.assertEqual(self.validate(binding,target=target,head=head),binding)

    def test_rewind_requires_a_canonical_ancestor_and_no_new_spans(self):
        head=self.store.move('main',0,self.target,'prior-move')
        binding={**self.binding,'kind':'rewind','messages':[],
            'source':dict(snapshot=self.target,revision=1,token_count=7),
            'input_span':None,'output_span':None,'turn_complete':True}
        self.assertEqual(self.validate(binding,target=self.root,head=head),binding)
        sibling=self.store.append(self.root,[8],{'status':'active'})
        with self.assertRaises(InvalidOperation):self.validate(binding,target=sibling,head=head)


if __name__=='__main__':unittest.main()
