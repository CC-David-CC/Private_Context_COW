"""CPU checks ensure the live HTTP probe cannot accept unrelated errors."""
import importlib.util,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
class VisionAuthorizationProbeTests(unittest.TestCase):
 def module(self):
  spec=importlib.util.spec_from_file_location('run_vision_authorization_probe',ROOT/'scripts/run_vision_authorization_probe.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
 def test_exact_status_and_expected_guard_reason_are_required(self):
  m=self.module();m.require_rejection(401,{'error':'Missing or wrong API key'},401,'API key')
  for code,value in [(200,{'error':'API key'}),(400,{'error':'unrelated malformed request'}),(401,{'error':'unrelated'}),(401,{})]:
   with self.subTest(code=code,value=value),self.assertRaises(m.ProbeFailure):m.require_rejection(code,value,401,'API key')
 def test_effects_cannot_be_hidden_by_an_expected_http_error(self):
  m=self.module()
  baseline={'head':{'revision':1,'tip':'abc'},'snapshot':{'controller':{'status':'idle'}},'jobs':1,'encoders':1}
  m.require_unchanged(baseline,dict(baseline))
  for key,value in [('head',{'revision':2,'tip':'abc'}),('snapshot',{'controller':{'status':'active'}}),('jobs',2),('encoders',2)]:
   with self.subTest(key=key),self.assertRaises(m.ProbeFailure):m.require_unchanged(baseline,{**baseline,key:value})
 def test_real_http_exchange_sends_body_only_when_requested(self):
  m=self.module()
  import http.server,threading,json
  seen=[]
  class Handler(http.server.BaseHTTPRequestHandler):
   def log_message(self,*args):pass
   def do_POST(self):
    length=int(self.headers['Content-Length'])
    raw=self.rfile.read(length) if self.path=='/body' else None
    seen.append((self.path,length,raw,self.headers['Authorization']))
    value={'error':'Missing or wrong API key'} if raw is None else {'bytes':len(raw)}
    payload=json.dumps(value).encode();self.send_response(401 if raw is None else 200);self.send_header('Content-Length',str(len(payload)));self.end_headers();self.wfile.write(payload)
  server=http.server.HTTPServer(('127.0.0.1',0),Handler);thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
  try:
   url='http://127.0.0.1:'+str(server.server_port)
   status,value,elapsed=m.http_exchange(url,'/header',b'not transmitted',key='wrong',header_only=True)
   self.assertEqual(status,401);self.assertIn('API key',value['error']);self.assertLess(elapsed,3)
   status,value,_=m.http_exchange(url,'/body',b'complete payload',key='right')
   self.assertEqual((status,value),(200,{'bytes':16}));self.assertEqual(seen,[('/header',15,None,'Bearer wrong'),('/body',16,b'complete payload','Bearer right')])
  finally:server.shutdown();server.server_close();thread.join(5)

if __name__=='__main__':unittest.main()
