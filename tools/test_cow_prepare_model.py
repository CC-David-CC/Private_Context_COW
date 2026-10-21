"""Installer resource checks. These do not qualify GPU inference."""
import contextlib,hashlib,http.server,io,json,tempfile,threading,unittest
from pathlib import Path
from tools import cow_prepare_model as prepare

class ModelPreparationTests(unittest.TestCase):
 def asset(self,body,url='https://example.test/model.gguf'):
  return prepare.Asset('model.gguf',url,len(body),hashlib.sha256(body).hexdigest())
 def test_default_plan_has_four_pinned_q4_shards_and_no_mtp(self):
  plan=prepare.model_plan()
  self.assertEqual(len(plan),4)
  self.assertTrue(all('/resolve/38bb39ee97821de2c9009abb7e93950eec396e66/' in a.url for a in plan))
  self.assertTrue(all('UD-Q4_K_XL' in a.name and len(a.sha256)==64 for a in plan))
  self.assertFalse(any('mtp' in a.url.lower() for a in plan))
 def test_cached_model_is_rehashed_without_network_or_deletion(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'model.gguf';p.write_bytes(b'correct')
   prepare.fetch_verified(self.asset(b'correct'),Path(d),downloader=lambda *a: self.fail('network'))
   self.assertEqual(p.read_bytes(),b'correct')
 def test_same_size_corruption_with_a_done_marker_is_rejected(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'model.gguf';p.write_bytes(b'wrong!!');p.with_name(p.name+'.done').write_text('finished')
   with self.assertRaisesRegex(ValueError,'SHA-256'):
    prepare.fetch_verified(self.asset(b'correct'),Path(d),downloader=lambda *a:self.fail('network'))
   self.assertEqual(p.read_bytes(),b'wrong!!')
 def test_symlink_model_is_rejected(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d);(p/'outside').write_bytes(b'right');(p/'model.gguf').symlink_to(p/'outside')
   with self.assertRaisesRegex(ValueError,'symlink'):
    prepare.fetch_verified(self.asset(b'right'),p,downloader=lambda *a:self.fail('network'))
 def test_real_http_download_resumes_partial_bytes(self):
  body=b'seeded-model-resource-'+bytes(range(256))*64;seen=[]
  class Handler(http.server.BaseHTTPRequestHandler):
   def log_message(self,*a):pass
   def do_HEAD(self):
    self.send_response(200);self.send_header('Content-Length',str(len(body)));self.end_headers()
   def do_GET(self):
    start=int(self.headers.get('Range','bytes=0-').split('=')[1].split('-')[0]);seen.append(start)
    self.send_response(206);self.send_header('Content-Length',str(len(body)-start));self.send_header('Content-Range',f'bytes {start}-{len(body)-1}/{len(body)}');self.end_headers();self.wfile.write(body[start:])
  with http.server.ThreadingHTTPServer(('127.0.0.1',0),Handler) as server,tempfile.TemporaryDirectory() as d:
   thread=threading.Thread(target=server.serve_forever);thread.start()
   try:
    root=Path(d);(root/'model.gguf.part').write_bytes(body[:137])
    with contextlib.redirect_stdout(io.StringIO()):prepare.fetch_verified(self.asset(body,f'http://127.0.0.1:{server.server_port}/model.gguf'),root)
    self.assertEqual((root/'model.gguf').read_bytes(),body);self.assertEqual(seen,[137]);self.assertFalse((root/'model.gguf.part').exists())
   finally:server.shutdown();thread.join()
 def builder(self,path):
  (path/'tokenizer').mkdir()
  for name in ('dense.bin','index.txt','native_experts.txt','tokenizer/vocab.json'):(path/name).write_bytes(('fixture '+name).encode())
 def test_pack_publishes_complete_receipt_and_reuses_verified_bytes(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);pack=prepare.publish_pack(root,[],self.builder)
   self.assertTrue((pack/'cow-prepared.json').is_file())
   self.assertEqual(prepare.publish_pack(root,[],lambda p:self.fail('rebuild')),pack)
 def test_failed_pack_builder_publishes_no_final_pack(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d)
   def fail(path):(path/'dense.bin').write_bytes(b'partial');raise RuntimeError('builder failed')
   with self.assertRaisesRegex(RuntimeError,'builder failed'):prepare.publish_pack(root,[],fail)
   self.assertFalse((root/'pack-ud-q4-k-xl').exists());self.assertEqual(list(root.iterdir()),[])
 def test_extra_nested_receipt_named_file_is_not_exempt_from_integrity(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);pack=prepare.publish_pack(root,[],self.builder)
   (pack/'tokenizer/cow-prepared.json').write_text('unexpected extra resource')
   with self.assertRaisesRegex(ValueError,'changed'):
    prepare.publish_pack(root,[],lambda p:self.fail('rebuild'))
 def test_altered_pack_is_rejected_without_rebuilding_or_deleting(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);pack=prepare.publish_pack(root,[],self.builder);(pack/'dense.bin').write_bytes(b'changed')
   with self.assertRaisesRegex(ValueError,'changed'):prepare.publish_pack(root,[],lambda p:self.fail('rebuild'))
   self.assertEqual((pack/'dense.bin').read_bytes(),b'changed')

if __name__=='__main__':unittest.main()
