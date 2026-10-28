"""Canonical text receipt lookup after coordinator reconstruction; CPU fixture only.

These are actual Store/HTTP integration checks, not process-crash/GPU qualification.
"""
import json,threading,unittest,urllib.request,urllib.error
from unittest.mock import patch
from pathlib import Path
import serve.test_cow_presentation_flow as fixture
from serve.cow_adapter import CowAdapter
from serve.cow_http import make_server
class PersistedTextStatusTests(unittest.TestCase):
 def setUp(self):
  self.case=fixture.CowPresentationFlowTests();self.addCleanup(self.case.doCleanups);self.case.setUp();self.server=None
 def persist_checkpoint_metadata(self):
  # FlowModel predates durable checkpoint annotations. Exercise the production
  # Store.move metadata contract without changing that shared legacy fixture.
  model=self.case.models[0];move=model.store.move
  def annotated(name,revision,target,operation,**options):
   cp=model.cache[target]
   options['checkpoint_receipt']=dict(schema=1,snapshot=target,
    payload_sha256=cp.checksum,model_contract_id='flow-model',
    execution_profile_id='flow-profile')
   return move(name,revision,target,operation,**options)
  model.store.move=annotated
 def commit(self):
  self.persist_checkpoint_metadata()
  c=self.case;c.text.prepare('chat',[dict(role='user',content='hello')],'root-op',namespace_id=c.namespace1).durable.result(5)
  head=c.coordinator.store.head('chat');request=c.text.answer('chat',head.revision,'answer-op',1);receipt=request.durable.result(5)
  self.assertEqual(request.text_result(),'hi');return receipt
 def lookup(self,operation,path='/cow/operation'):
  if self.server is None:
   self.server=make_server(CowAdapter(self.case.coordinator),api_key='status-key',text_adapter=self.case.text)
   thread=threading.Thread(target=self.server.serve_forever,daemon=True);thread.start()
   def stop():self.server.shutdown();self.server.server_close();thread.join(3)
   self.addCleanup(stop)
  url='http://127.0.0.1:'+str(self.server.server_port)+path+'?operation='+operation
  request=urllib.request.Request(url,headers={'Authorization':'Bearer status-key'})
  try:
   with urllib.request.urlopen(request,timeout=5) as r:return r.status,json.load(r)
  except urllib.error.HTTPError as error:
   with error:return error.code,json.loads(error.read())
 def assert_read_only_lookup(self,expected):
  before=dict(self.case.models[0].stats);head=self.case.coordinator.store.head('chat');code,value=self.lookup('answer-op')
  self.assertEqual(code,200,value);self.assertTrue(value['durable']);self.assertEqual(value['receipt'],expected);self.assertEqual(value['operation_id'],'answer-op');self.assertEqual(value['scope'],'canonical-presentation')
  self.assertEqual(self.case.models[0].stats,before);self.assertEqual(self.case.coordinator.store.head('chat'),head)
 def test_committed_receipt_survives_coordinator_reconstruction(self):
  expected=self.commit();self.case.new_coordinator();self.assert_read_only_lookup(expected)
 def test_lookup_returns_original_result_not_later_tip(self):
  expected=self.commit();head=self.case.coordinator.store.head('chat');self.case.text.resume('chat',head.revision,'resume-op',2).durable.result(5)
  self.assertNotEqual(self.case.coordinator.store.head('chat').tip,expected['snapshot']);self.case.new_coordinator();self.assert_read_only_lookup(expected)
 def test_lookup_needs_no_numerical_checkpoint(self):
  expected=self.commit();self.case.coordinator.close();directory=Path(self.case.temp.name)/'checkpoints'
  for path in directory.iterdir():
   self.assertTrue(path.resolve().is_relative_to(Path(self.case.temp.name).resolve()))
   if path.is_file():path.unlink()
  self.case.new_coordinator();self.assert_read_only_lookup(expected)
 def test_unknown_operation_is_not_a_commit_or_abort_proof(self):
  self.commit();self.case.new_coordinator();before=dict(self.case.models[0].stats);code,value=self.lookup('unknown-operation');self.assertEqual(code,404);self.assertNotIn('receipt',value);self.assertEqual(self.case.models[0].stats,before)
 def test_failed_generation_status_is_process_local_after_reconstruction(self):
  self.root_commit();c=self.case;head=c.coordinator.store.head('chat')
  with patch.object(c.models[0],'run_greedy_request',side_effect=RuntimeError('synthetic inference failure')):
   request=c.text.answer('chat',head.revision,'failed-answer',1)
   with self.assertRaisesRegex(RuntimeError,'synthetic inference failure'):request.durable.result(5)
  live=c.coordinator.operation_status('failed-answer')
  self.assertEqual(live['state'],'failed');self.assertFalse(live['durable'])
  self.assertEqual(live['scope'],'coordinator-process')
  c.new_coordinator();code,value=self.lookup('failed-answer')
  self.assertEqual(code,404);self.assertNotIn('receipt',value)
  self.assertEqual(c.coordinator.store.head('chat'),head)
 def test_live_failure_diagnostic_is_read_only_and_process_scoped(self):
  self.root_commit();c=self.case;head=c.coordinator.store.head('chat')
  with patch.object(c.models[0],'run_greedy_request',side_effect=RuntimeError('synthetic inference failure')):
   request=c.text.answer('chat',head.revision,'diagnostic-failure',1)
   with self.assertRaises(RuntimeError):request.durable.result(5)
  before=dict(c.models[0].stats)
  code,value=self.lookup('diagnostic-failure','/cow/operation-diagnostic')
  self.assertEqual(code,200,value);self.assertEqual(value['state'],'failed')
  self.assertEqual(value['diagnostic_scope'],'coordinator-process-only')
  self.assertEqual(value['failure'],dict(type='RuntimeError',message='synthetic inference failure'))
  self.assertFalse(value['durable']);self.assertEqual(c.models[0].stats,before)
  self.assertEqual(c.coordinator.store.head('chat'),head)
  url='http://127.0.0.1:'+str(self.server.server_port)+'/cow/operation-diagnostic?operation=diagnostic-failure'
  with self.assertRaises(urllib.error.HTTPError) as denied:urllib.request.urlopen(url,timeout=5)
  self.assertEqual(denied.exception.code,401)
  code,value=self.lookup('unknown-operation','/cow/operation-diagnostic')
  self.assertEqual(code,404);self.assertNotIn('failure',value)
  from context_time_travel.store import NotFound
  c.new_coordinator()
  with self.assertRaises(NotFound):c.coordinator.operation_diagnostic('diagnostic-failure')
 def test_context_budget_is_snapshot_bound_and_does_not_run_model(self):
  self.root_commit();c=self.case;model=c.models[0];model.profile['execution']['capacity']=1024
  head=c.coordinator.store.head('chat');before=dict(model.stats)
  from context_time_travel.input_spans import history_for_snapshot
  count=len(history_for_snapshot(c.coordinator.store,head.tip))
  self.lookup('unknown-operation') # starts the authenticated HTTP fixture
  url='http://127.0.0.1:'+str(self.server.server_port)+'/cow/context-budget?branch=chat'
  req=urllib.request.Request(url,headers={'Authorization':'Bearer status-key'})
  for capacity in (1024,8192,24576,32768,65536,131072):
   model.profile['execution']['capacity']=capacity
   with urllib.request.urlopen(req,timeout=5) as response:value=json.load(response)
   self.assertEqual(value['branch'],'chat');self.assertEqual(value['revision'],head.revision)
   self.assertEqual(value['snapshot'],head.tip);self.assertEqual(value['capacity_tokens'],capacity)
   self.assertEqual(value['used_tokens'],count);self.assertEqual(value['remaining_tokens'],capacity-count)
   self.assertEqual(value['execution_profile_id'],'flow-profile')
  self.assertEqual(model.stats,before);self.assertEqual(c.coordinator.store.head('chat'),head)
  with self.assertRaises(urllib.error.HTTPError) as denied:urllib.request.urlopen(url,timeout=5)
  self.assertEqual(denied.exception.code,401)
  model.profile['execution']['capacity']=count-1
  with self.assertRaises(urllib.error.HTTPError) as overfull:urllib.request.urlopen(req,timeout=5)
  self.assertEqual(overfull.exception.code,409)
  del model.profile['execution']['capacity']
  with self.assertRaises(urllib.error.HTTPError) as unavailable:urllib.request.urlopen(req,timeout=5)
  self.assertEqual(unavailable.exception.code,409)
 def test_text_preflight_counts_boundary_and_rejects_stale_snapshot(self):
  self.root_commit();c=self.case;model=c.models[0];head=c.coordinator.store.head('chat')
  from context_time_travel.input_spans import history_for_snapshot
  count=len(history_for_snapshot(c.coordinator.store,head.tip));model.profile['execution']['capacity']=count+8
  self.lookup('unknown-operation');before=dict(model.stats)
  url='http://127.0.0.1:'+str(self.server.server_port)+'/cow/text/context-preflight'
  def post(data):
   request=urllib.request.Request(url,data=json.dumps(data).encode(),headers={'Authorization':'Bearer status-key','Content-Type':'application/json'})
   with urllib.request.urlopen(request,timeout=5) as response:return json.load(response)
  body=dict(branch='chat',revision=head.revision,snapshot=head.tip,mode='answer',content=None,max_tokens=8)
  value=post(body);self.assertTrue(value['fits']);self.assertEqual(value['input_tokens'],0)
  self.assertEqual(value['available_output_tokens'],8)
  body['max_tokens']=9;self.assertFalse(post(body)['fits'])
  body['snapshot']='0'*64
  with self.assertRaises(urllib.error.HTTPError) as stale:post(body)
  self.assertEqual(stale.exception.code,409)
  self.assertEqual(model.stats,before);self.assertEqual(c.coordinator.store.head('chat'),head)
 def test_text_preflight_counts_actual_next_user_suffix(self):
  self.root_commit();c=self.case;head=c.coordinator.store.head('chat')
  c.text.answer('chat',head.revision,'answer-op',1).durable.result(5)
  head=c.coordinator.store.head('chat');c.models[0].profile['execution']['capacity']=1024
  before=dict(c.models[0].stats);continuation=c.text.context_preflight('chat',head.revision,head.tip,'continue',None,2)
  self.assertEqual(continuation['input_tokens'],0);self.assertTrue(continuation['fits']);self.assertEqual(c.models[0].stats,before)
  c.text.resume('chat',head.revision,'resume-op',2).durable.result(5)
  head=c.coordinator.store.head('chat');model=c.models[0];model.profile['execution']['capacity']=1024
  from context_time_travel.input_spans import history_for_snapshot
  canonical=history_for_snapshot(c.coordinator.store,head.tip);controller=c.coordinator.store.get_snapshot(head.tip)['controller']
  content='A literal question with spaces.\n';expected=c.text.append_user_suffix(canonical,controller,content)
  before=dict(model.stats);result=c.text.context_preflight('chat',head.revision,head.tip,'append_user',content,1)
  self.assertEqual(result['input_tokens'],len(expected));self.assertTrue(result['fits'])
  self.assertEqual(model.stats,before);self.assertEqual(c.coordinator.store.head('chat'),head)
 def test_text_preflight_counts_initial_question_suffix(self):
  c=self.case;c.text.create('code-a',[{'role':'user','content':'saved context'}],
   'code-root',namespace_id=c.namespace1).durable.result(5)
  model=c.models[0];model.profile['execution']['capacity']=1024;head=c.coordinator.store.head('code-a')
  from context_time_travel.input_spans import history_for_snapshot
  canonical=history_for_snapshot(c.coordinator.store,head.tip);controller=c.coordinator.store.get_snapshot(head.tip)['controller']
  content='First question';expected=c.text.append_initial_user_suffix(canonical,controller,content)
  before=dict(model.stats);result=c.text.context_preflight('code-a',head.revision,head.tip,'ask',content,1)
  self.assertEqual(result['input_tokens'],len(expected));self.assertTrue(result['fits'])
  self.assertEqual(model.stats,before);self.assertEqual(c.coordinator.store.head('code-a'),head)
 def test_commit_then_failure_is_recovered_as_durable_after_reconstruction(self):
  self.root_commit();c=self.case;head=c.coordinator.store.head('chat');model=c.models[0]
  save=model.save_prepared_branch
  def commit_then_fail(*args,**kwargs):
   save(*args,**kwargs)
   raise RuntimeError('synthetic lost publication acknowledgement')
  with patch.dict(model.plans,{'commit-then-fail':(257,)}), patch.object(model,'save_prepared_branch',side_effect=commit_then_fail):
   request=c.text.answer('chat',head.revision,'commit-then-fail',1)
   with self.assertRaisesRegex(RuntimeError,'synthetic lost publication acknowledgement'):
    request.durable.result(5)
  live=c.coordinator.operation_status('commit-then-fail')
  self.assertEqual(live['state'],'failed');self.assertFalse(live['durable'])
  c.new_coordinator();code,value=self.lookup('commit-then-fail')
  self.assertEqual(code,200,value);self.assertTrue(value['durable'])
  self.assertEqual(value['scope'],'canonical-presentation')
  self.assertEqual(value['receipt']['branch']['revision'],head.revision+1)
 def test_corrupt_intent_is_rejected_even_with_committed_reference(self):
  self.commit();self.case.new_coordinator();path=Path(self.case.temp.name)/'pool-operations/answer-op.json';record=json.loads(path.read_text());record['intent']['signature']='0'*64;path.write_text(json.dumps(record))
  code,value=self.lookup('answer-op');self.assertEqual(code,409,value);self.assertNotIn('receipt',value)
 def root_commit(self):
  self.persist_checkpoint_metadata()
  return self.case.text.prepare('chat',[dict(role='user',content='hello')],'root-op',namespace_id=self.case.namespace1).durable.result(5)
 def test_root_receipt_survives_reconstruction_and_later_append(self):
  expected=self.root_commit();head=self.case.coordinator.store.head('chat')
  self.case.text.answer('chat',head.revision,'answer-op',1).durable.result(5)
  self.case.new_coordinator();before=dict(self.case.models[0].stats)
  code,value=self.lookup('root-op')
  self.assertEqual(code,200,value);self.assertEqual(value['receipt'],expected)
  self.assertEqual(value['scope'],'canonical-root');self.assertTrue(value['durable'])
  self.assertNotEqual(self.case.coordinator.store.head('chat').tip,expected['snapshot'])
  self.assertEqual(self.case.models[0].stats,before)
 def test_root_lookup_does_not_require_optional_checkpoint_files(self):
  expected=self.root_commit();self.case.coordinator.close()
  directory=Path(self.case.temp.name)/'checkpoints'
  for path in directory.iterdir():
   self.assertTrue(path.resolve().is_relative_to(Path(self.case.temp.name).resolve()))
   if path.is_file():path.unlink()
  self.case.new_coordinator()
  code,value=self.lookup('root-op');self.assertEqual(code,200,value);self.assertEqual(value['receipt'],expected)
  self.assertEqual(self.case.models[0].stats['forward_tokens'],0)
 def test_corrupt_root_intent_cannot_prove_commit(self):
  self.root_commit();self.case.new_coordinator()
  path=Path(self.case.temp.name)/'pool-operations/root-op.json';record=json.loads(path.read_text());record['intent']['signature']='0'*64;path.write_text(json.dumps(record))
  code,value=self.lookup('root-op');self.assertEqual(code,409,value);self.assertNotIn('receipt',value)
 def test_original_root_request_replays_after_reconstruction_without_forward_work(self):
  expected=self.root_commit();self.case.new_coordinator();before=dict(self.case.models[0].stats)
  replay=self.case.text.prepare('chat',[dict(role='user',content='hello')],'root-op',namespace_id=self.case.namespace1)
  self.assertEqual(replay.durable.result(5),expected)
  self.assertEqual(replay.snapshot.result(5)['tokens'],[])
  self.assertEqual(self.case.models[0].stats,before)
if __name__=='__main__':unittest.main()
