import copy,tempfile,unittest
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
