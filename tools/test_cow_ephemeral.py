"""Real filesystem and process-lifetime tests for temporary server sessions."""
import os
import json
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import unittest
from tools.cow_server import session_storage,clean_abandoned_sessions


class EphemeralSessionTests(unittest.TestCase):
    def test_default_store_is_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);store=root/'store';store.mkdir();(store/'keep').write_text('saved')
            config={'store':str(store)}
            with session_storage(root,config) as (directory,value):
                self.assertEqual(directory,root);self.assertIs(value,config)
            self.assertTrue((store/'keep').exists())

    def test_session_files_removed_but_config_and_existing_store_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);original=root/'store';original.mkdir();(original/'keep').write_text('saved')
            config={'ephemeral':True,'store':str(original)}
            with session_storage(root,config) as (session,value):
                store=Path(value['store']);store.mkdir();(store/'spill').write_bytes(b'checkpoint')
                (session/'grammars').mkdir()
                with session_storage(root,config) as (other,_):
                    self.assertTrue(store.exists());self.assertNotEqual(session,other)
            self.assertFalse(session.exists());self.assertTrue((original/'keep').exists())
            self.assertEqual(config['store'],str(original))

    def test_exception_cleans_session(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(RuntimeError,'failure'):
                with session_storage(Path(tmp),{'ephemeral':True}) as (session,_):raise RuntimeError('failure')
            self.assertFalse(session.exists())

    def test_crash_session_is_reaped_on_next_start(self):
        with tempfile.TemporaryDirectory() as tmp:
            code="from pathlib import Path; from tools.cow_server import session_storage; import sys,time\nwith session_storage(Path(sys.argv[1]),{'ephemeral':True}) as (session,_):\n (session/'spill').write_bytes(b'checkpoint');print(session,flush=True);time.sleep(60)\n"
            process=subprocess.Popen([sys.executable,'-c',code,tmp],stdout=subprocess.PIPE,text=True)
            try:
                stale=Path(process.stdout.readline().strip());self.assertTrue((stale/'spill').is_file())
                process.kill();process.wait(timeout=10)
                with session_storage(Path(tmp),{'ephemeral':True}):self.assertFalse(stale.exists())
            finally:
                if process.poll() is None:process.kill();process.wait(timeout=10)
                process.stdout.close()

    def test_launcher_sigterm_unwinds_temporary_store(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);key=root/'key';key.write_text('test key');key.chmod(0o600)
            profile=root/'profile.json';profile.write_text(json.dumps({'execution':{'GPU':[0],'environment':{}}}))
            config=dict(ephemeral=True,profile=str(profile),store=str(root/'store'),api_key_file=str(key),
                        host='127.0.0.1',port=18180,ram_gib=1,disk_gib=1,checkpoint_idle_seconds=1800)
            (root/'server.json').write_text(json.dumps(config))
            code="import os,sys,time; from tools import cow_server\nos.environ['HOME']=sys.argv[1]\ndef idle(*args,**kwargs):\n print('ready',flush=True);time.sleep(60)\ncow_server.runpy.run_module=idle\nsys.argv=['cow-server',sys.argv[1]]\ncow_server.main()\n"
            process=subprocess.Popen([sys.executable,'-c',code,tmp],stdout=subprocess.PIPE,stderr=subprocess.DEVNULL,text=True)
            try:
                self.assertEqual(process.stdout.readline().strip(),'ready')
                self.assertEqual(len(list((root/'.ephemeral-sessions').glob('session-*'))),1)
                process.terminate();process.wait(timeout=10)
                self.assertEqual(list((root/'.ephemeral-sessions').glob('session-*')),[])
            finally:
                if process.poll() is None:process.kill();process.wait(timeout=10)
                process.stdout.close()

    def test_symlinks_and_unmarked_directories_are_not_deleted(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);external=root/'external';external.mkdir();(external/'keep').write_text('saved')
            sessions=root/'sessions';sessions.mkdir();(sessions/'session-link').symlink_to(external,target_is_directory=True)
            unmarked=sessions/'session-unmarked';unmarked.mkdir();(unmarked/'keep').write_text('saved')
            clean_abandoned_sessions(sessions)
            self.assertTrue((external/'keep').exists());self.assertTrue((unmarked/'keep').exists())
            (root/'.ephemeral-sessions').symlink_to(external,target_is_directory=True)
            with self.assertRaisesRegex(ValueError,'symlink'):
                with session_storage(root,{'ephemeral':True}):self.fail('linked root admitted')


if __name__=='__main__':unittest.main()
