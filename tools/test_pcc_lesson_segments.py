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
