import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from tools.cow_server_setup import compatibility_config


class CompatibilitySetupTests(unittest.TestCase):
    def test_thinking_is_explicit_and_default_argv_stays_unchanged(self):
        a=SimpleNamespace(port=8099,openai_port=8100,openai_thinking=True)
        self.assertTrue(compatibility_config(a)['openai_thinking'])
        with self.assertRaises(ValueError):compatibility_config(SimpleNamespace(port=8099,openai_thinking=True))
        with self.assertRaises(ValueError):compatibility_config(SimpleNamespace(port=8099,openai_thinking='yes'))
        from tools.cow_server import configured_argv
        cfg=dict(profile='/owned/profile',store='/owned/store',host='127.0.0.1',port=8099,ram_gib=10,disk_gib=10,checkpoint_idle_seconds=1800,openai_port=8100)
        before=configured_argv(Path('/owned'),cfg,'fixture')
        self.assertNotIn('--openai-thinking',before)
        self.assertIn('--openai-thinking',configured_argv(Path('/owned'),dict(cfg,openai_thinking=True),'fixture'))

    def test_default_adds_no_configuration(self):
        self.assertEqual(compatibility_config(SimpleNamespace(port=8099)), {})

    def test_separate_port_model_and_output_limit(self):
        a = SimpleNamespace(port=8099, openai_port=8100, openai_model='wen-1.1', openai_max_output=4096)
        self.assertEqual(compatibility_config(a), dict(openai_port=8100, openai_model='wen-1.1', openai_max_output=4096))
        a.openai_model = None
        self.assertEqual(compatibility_config(a)['openai_model'], 'cow')

    def test_rejects_colliding_invalid_or_incomplete_configuration(self):
        for options in [dict(openai_port=8099), dict(openai_port=0), dict(openai_port=65536),
                        dict(openai_model='wen'), dict(openai_port=8100, openai_model='bad\nname'),
                        dict(openai_port=8100, openai_max_output=0)]:
            with self.subTest(options=options), self.assertRaises(ValueError):
                compatibility_config(SimpleNamespace(port=8099, **options))

    @unittest.skipUnless(os.name == 'posix', 'Existing COW launcher uses POSIX locks')
    def test_legacy_launcher_argv_is_unchanged_until_opted_in(self):
        from tools.cow_server import configured_argv
        cfg = dict(profile='/owned/profile.json', store='/owned/store', host='127.0.0.1', port=8099,
                   ram_gib=100, disk_gib=100, checkpoint_idle_seconds=1800)
        before = configured_argv(Path('/owned'), cfg, 'test-only-key')
        self.assertEqual(before, ['cow-http','--profile','/owned/profile.json','--store','/owned/store',
            '--host','127.0.0.1','--port','8099','--api-key','test-only-key', '--ram-gib','100',
            '--disk-gib','100','--min-available-gib','16','--checkpoint-idle-seconds','1800',
            '--text','--grammar-store','/owned/grammars'])
        after = configured_argv(Path('/owned'), dict(cfg, openai_port=8100, openai_model='wen-1.1'), 'test-only-key')
        self.assertEqual(after, before+['--openai-port','8100','--openai-model','wen-1.1','--openai-max-output','2048'])
        fabric=configured_argv(Path('/owned'),dict(cfg,cache_inventory=True,scheduling='transfer-aware'),'test-only-key')
        self.assertEqual(fabric,before+['--scheduling','transfer-aware','--cache-inventory'])
        with self.assertRaises(ValueError):
            configured_argv(Path('/owned'),dict(cfg,cache_inventory='false'),'test-only-key')


if __name__ == '__main__': unittest.main()
