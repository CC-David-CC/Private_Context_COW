
"""CPU launcher tests; these do not qualify GPU vision or any numerical gate."""
from pathlib import Path
import tempfile,unittest
from tools import cow_server

class ConfiguredVisionTests(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
  self.config=dict(profile='/p',store='/s',host='127.0.0.1',port=8099,ram_gib=16,disk_gib=16,min_available_gib=12,checkpoint_idle_seconds=1800)
 def test_default_arguments_remain_identical(self):
  self.assertEqual(cow_server.configured_argv(self.root,self.config,'test-key'),['cow-http','--profile','/p','--store','/s','--host','127.0.0.1','--port','8099','--api-key','test-key','--ram-gib','16','--disk-gib','16','--min-available-gib','12','--checkpoint-idle-seconds','1800','--text','--grammar-store',str(self.root/'grammars')])
 def test_option_looking_api_keys_reach_argparse_unchanged(self):
  import argparse
  parser=argparse.ArgumentParser();parser.add_argument('--api-key')
  for key in ('-leading-dash', '--text', '--', '-9', 'plain-key'):
   with self.subTest(key=key):
    argv=cow_server.configured_argv(self.root,self.config,key)
    parsed,remaining=parser.parse_known_args(argv[1:])
    self.assertEqual(parsed.api_key,key)
    self.assertIn('--text',remaining)
 def test_weighted_scheduler_is_forwarded_only_when_opted_in(self):
  args=cow_server.configured_argv(self.root,dict(self.config,scheduling='weighted-prefix'),'test-key')
  self.assertEqual(args[-2:],['--scheduling','weighted-prefix'])
  self.assertEqual(args.count('--scheduling'),1)
  self.assertEqual(cow_server.configured_argv(self.root,dict(self.config,scheduling='fifo'),'test-key'),cow_server.configured_argv(self.root,self.config,'test-key'))
 def test_explicit_vision_configuration_is_forwarded_once(self):
  p=self.root/'vision.json';p.write_text('{}');config=dict(self.config,vision_config=str(p));args=cow_server.configured_argv(self.root,config,'test-key')
  self.assertEqual(args[-2:],['--vision-config',str(p)]);self.assertEqual(args.count('--vision-config'),1);self.assertEqual(config['vision_config'],str(p))
 def test_missing_vision_configuration_fails_before_launch(self):
  with self.assertRaises(ValueError):cow_server.configured_argv(self.root,dict(self.config,vision_config=str(self.root/'missing.json')),'test-key')
 def test_relative_vision_configuration_rejected(self):
  with self.assertRaises(ValueError):cow_server.configured_argv(self.root,dict(self.config,vision_config='relative.json'),'test-key')
 def test_linked_vision_configuration_rejected(self):
  p=self.root/'actual.json';p.write_text('{}');link=self.root/'link.json';link.symlink_to(p)
  with self.assertRaises(ValueError):cow_server.configured_argv(self.root,dict(self.config,vision_config=str(link)),'test-key')
if __name__=='__main__':unittest.main()
