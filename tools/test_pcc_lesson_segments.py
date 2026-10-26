import copy,tempfile,unittest,json
from pathlib import Path
from unittest.mock import patch
from tools.pcc_lesson_segments import tokenize_segments,validate_segments_lesson
from tools.pcc_lessons import LessonService,validate_lesson

def lesson(policy='verified_thinking_actions_answer'):
    spans=[('user','question','Q:',False),('assistant','thinking','reason;',True),('assistant','action','read;',True),('tool','observation','value;',False),('user','question','continue;',False),('assistant','answer','done!',True)]
    return dict(schema=2,title='Owned multi-turn demonstration',model_contract_id='a'*64,scope='owned-training',target_policy=policy,segments=[dict(role=r,kind=k,rendered=t,verified=v,scope='practice') for r,k,t,v in spans],evidence=dict(method='sss-verified-episode-v1',archive_sha256='b'*64,verification_sha256='c'*64,renderer_sha256='d'*64,tokenizer_sha256='e'*64))

class Segments(unittest.TestCase):
    def test_thinking_targets_are_optional_context_is_identical(self):
        full=tokenize_segments(lesson(),lambda s:list(s.encode()),256)
        answer=tokenize_segments(lesson('actions_answer'),lambda s:list(s.encode()),256)
        self.assertEqual(full['tokens'],answer['tokens']);self.assertEqual(full['targets'],answer['targets'])
        self.assertEqual(sum(full['loss_mask'])-sum(answer['loss_mask']),len('reason;'))
        self.assertEqual(sum(answer['loss_mask']),len('read;done!'))
    def test_verification_scope_and_boundaries(self):
        x=lesson();x['segments'][1]['verified']=False
        with self.assertRaises(ValueError):validate_lesson(x)
        x['target_policy']='actions_answer';validate_lesson(x)
        x['segments'][0]['scope']='qualification'
        with self.assertRaises(ValueError):validate_lesson(x)
        with self.assertRaises(ValueError):tokenize_segments(lesson(),lambda s:[len(s)],256)
        with self.assertRaises(ValueError):tokenize_segments(lesson(),lambda s:list(s.encode()),8)
    def test_real_lesson_store_reconciles_v2(self):
        with tempfile.TemporaryDirectory() as root:
            service=LessonService(root,'a'*64,lambda s:list(s.encode()),lambda *args:None)
            try:
                request=dict(operation_id='record-multiturn',lesson=lesson())
                first=service.submit('/cow/training/lessons',request)
                self.assertEqual(first,service.submit('/cow/training/lessons',request))
                saved=service.lesson(first['lesson_id'])
                self.assertEqual(saved['lesson']['schema'],2)
                self.assertEqual(saved['example']['loss_mask'],tokenize_segments(lesson(),lambda s:list(s.encode()),256)['loss_mask'])
            finally:service.close()

    def test_configured_rendered_spans_use_control_tokens_without_changing_legacy_text(self):
        from tools.pcc_lessons import configured_service
        class Tokenizer:
            def encode(self,text,parse_special=False):
                if parse_special:
                    return [250 if char=='~' else ord(char) for char in text.replace('<U>','~')]
                return list(text.encode())
        with tempfile.TemporaryDirectory() as root:
            root=Path(root);model=root/'model.gguf';model.write_bytes(b'owned-not-loaded')
            config=dict(schema=1,directory=str(root/'store'),max_tokens=256,max_steps=2,runner=dict(model=str(model)))
            (root/'config.json').write_text(json.dumps(config))
            with patch('tools.pcc_lessons.NativeRunner',return_value=lambda *args:None):
                service=configured_service(root/'config.json',dict(model_path=str(model),model_contract_id='a'*64),Tokenizer())
            try:
                item=lesson();item['segments'][0]['rendered']='<U>Q:'
                accepted=service.submit('/cow/training/lessons',dict(operation_id='rendered-control-proof',lesson=item))
                saved=service.lesson(accepted['lesson_id'])['example']
                self.assertEqual(saved['tokens'][0],250)
                legacy=service._example(dict(schema=1,prompt='<U>Q:',completion='ok',evidence={}), 'b'*64)
                self.assertEqual(legacy['tokens'][:3],list(b'<U>'))
                self.assertNotIn(250,legacy['tokens'])
            finally:service.close()


class TemplateFraming(unittest.TestCase):
    @staticmethod
    def token_ids(text):
        # Explicit synthetic EOS is one token; no model or native engine runs.
        result=[]
        while text:
            if text.startswith('<EOS>'):
                result.append(256);text=text[5:]
            else:
                result.append(ord(text[0]));text=text[1:]
        return result

    def sample(self,policy):
        value=lesson(policy)
        value['segments']=[dict(role='user',kind='question',rendered='Q:',verified=False,scope='practice'),
            dict(role='assistant',kind='answer',rendered='answer<EOS>',verified=True,scope='practice'),
            dict(role='assistant',kind='framing',rendered='\n',verified=False,scope='practice')]
        return value

    def test_template_tail_is_retained_but_never_a_target(self):
        for policy in ('actions_answer','verified_thinking_actions_answer'):
            with self.subTest(policy=policy):
                value=tokenize_segments(self.sample(policy),self.token_ids,256)
                self.assertEqual(value['tokens'],self.token_ids('Q:answer<EOS>'))
                self.assertEqual(value['targets'],self.token_ids(':answer<EOS>\n'))
                self.assertEqual(value['loss_mask'],[False]+[True]*7+[False])
                self.assertEqual(value['targets'][-2],256)
                self.assertTrue(value['loss_mask'][-2])
                self.assertFalse(value['loss_mask'][-1])

    def test_framing_cannot_claim_verification_or_another_role(self):
        for field,value in [('verified',True),('role','user'),('scope','development')]:
            item=self.sample('actions_answer');item['segments'][-1][field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):validate_segments_lesson(item)
        item=self.sample('actions_answer');item['segments'][1]['verified']=False
        with self.assertRaises(ValueError):validate_segments_lesson(item)
        item=self.sample('actions_answer');item['segments'][1]['kind']='framing';item['segments'][1]['verified']=False
        with self.assertRaises(ValueError):validate_segments_lesson(item)

    def test_real_service_admission_preserves_mask_and_idempotency(self):
        with tempfile.TemporaryDirectory() as root:
            service=LessonService(root,'a'*64,self.token_ids,lambda *args:None,rendered_tokenize=self.token_ids)
            try:
                request=dict(operation_id='record-framed-answer',lesson=self.sample('actions_answer'))
                first=service.submit('/cow/training/lessons',request)
                self.assertEqual(first,service.submit('/cow/training/lessons',request))
                saved=service.lesson(first['lesson_id'])['example']
                expected=tokenize_segments(request['lesson'],self.token_ids,256)
                for key in ('tokens','targets','loss_mask'):self.assertEqual(saved[key],expected[key])
            finally:service.close()
