"""Actual HTTP control protocol with synthetic requests; not GPU qualification."""
from concurrent.futures import Future
import json,threading,time,unittest
import serve.test_cow_http as fixtures
from context_time_travel.coordinator import Request

class PauseHTTPTests(unittest.TestCase):
 def setUp(self):
  self.f=fixtures.CowHTTPTests();self.f.setUp();self.addCleanup(self.f.tearDown)
 def pending(self,*,supported=True,failure=False):
  handle=Request('request-a','signature',pause_supported=supported)
  self.release=threading.Event();self.addCleanup(self.release.set)
  self.handle=handle
  def generate(*args,**kwargs):
   def finish():
    if supported:
     handle.pause_event.wait(5)
    else:self.release.wait(5)
    handle.events.put(dict(event='token',token=5))
    handle.snapshot.set_result(dict(tokens=[5],stopped=False))
    if supported:self.release.wait(5)
    if failure:handle.durable.set_exception(OSError('disk full after pause'))
    else:handle.durable.set_result(dict(snapshot='saved',branch=dict(revision=1)))
   threading.Thread(target=finish,daemon=True).start();return handle
  self.f.adapter.generate=generate
  response={};worker=threading.Thread(target=lambda:response.update(value=self.f.call('/cow/generate',self.f.request())))
  worker.start();self.addCleanup(lambda:worker.join(6))
  return response,worker
 def pause_active(self):
  until=time.monotonic()+5
  while True:
   code,body=self.f.call('/cow/pause',{'operation_id':'request-a'});value=json.loads(body)
   if code!=200 or value.get('active'):return code,value
   if time.monotonic()>until:self.fail('active stream was never registered')
   threading.Event().wait(.01)
 def test_requested_pause_is_not_durable_acknowledgment(self):
  response,worker=self.pending();code,value=self.pause_active()
  self.assertEqual(code,200,value);self.assertEqual(value,dict(operation_id='request-a',active=True,pause_requested=True))
  self.assertTrue(self.handle.pause_event.is_set());self.assertFalse(self.handle.durable.done());self.assertTrue(worker.is_alive())
  self.release.set();worker.join(5);self.assertFalse(worker.is_alive())
  events=[json.loads(row) for row in response['value'][1].splitlines()]
  self.assertEqual(events[-1]['event'],'complete');self.assertEqual(events[-1]['answer']['tokens'],[5])
  code,body=self.f.call('/cow/pause',{'operation_id':'request-a'})
  self.assertEqual(json.loads(body),dict(operation_id='request-a',active=False,pause_requested=False))
 def test_publication_failure_after_pause_never_emits_complete(self):
  response,worker=self.pending(failure=True);code,value=self.pause_active();self.assertEqual(code,200,value)
  self.release.set();worker.join(5);self.assertFalse(worker.is_alive())
  events=[json.loads(row) for row in response['value'][1].splitlines()]
  self.assertEqual(events[-1]['event'],'error');self.assertNotIn('complete',[r['event'] for r in events])
 def test_unsupported_request_rejects_pause_without_setting_event(self):
  response,worker=self.pending(supported=False);code,value=self.pause_active();self.assertEqual(code,409,value)
  self.assertFalse(self.handle.pause_event.is_set());self.release.set();worker.join(5)
 def test_auth_origin_id_and_unknown_request(self):
  self.assertEqual(self.f.call('/cow/pause',{'operation_id':'x'},key='wrong')[0],401)
  self.assertEqual(self.f.call('/cow/pause',{'operation_id':'x'},origin='http://evil.example')[0],403)
  for body in ({},{'operation_id':'../bad'},{'operation_id':'x','extra':True}):
   self.assertEqual(self.f.call('/cow/pause',body)[0],400)
  code,body=self.f.call('/cow/pause',{'operation_id':'missing'})
  self.assertEqual(code,200);self.assertEqual(json.loads(body),dict(operation_id='missing',active=False,pause_requested=False))
  self.assertEqual(self.f.adapter.calls,[])
