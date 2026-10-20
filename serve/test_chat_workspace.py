"""Durable branch catalog invariants; these are not native/GPU gate tests."""
import copy
from contextlib import closing
import sqlite3
import tempfile
import unittest
from pathlib import Path

from serve.chat_workspace import ChatWorkspace, WorkspaceConflict


class WorkspaceTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.path=Path(self.tmp.name)/'chats.sqlite3'
        self.store=ChatWorkspace(self.path)
        self.chat=self.store.create_chat('First chat','create-one')
        self.branch=self.chat['branch']['id']

    def tearDown(self):
        self.store.close()
        self.tmp.cleanup()

    def add(self,text,branch=None,role='user',op=None):
        branch=branch or self.branch
        head=self.store.history(branch)
        return self.store.append(branch,head['revision'],{'role':role,'text':text,'time':1},op or text)

    def test_fork_shares_prefix_and_keeps_suffixes_isolated(self):
        prefix=self.add('P')
        b=self.store.fork(self.branch,prefix['tip'],'Alternative','fork-b')
        self.add('A',role='assistant')
        self.add('B',b['id'],role='assistant')
        a_hist=self.store.history(self.branch);b_hist=self.store.history(b['id'])
        self.assertEqual([m['text'] for m in a_hist['messages']],['P','A'])
        self.assertEqual([m['text'] for m in b_hist['messages']],['P','B'])
        self.assertEqual(a_hist['messages'][0]['node_id'],b_hist['messages'][0]['node_id'])
        with closing(sqlite3.connect(self.path)) as db:
            self.assertEqual(db.execute('SELECT count(*) FROM messages').fetchone()[0],3)

    def test_restart_preserves_chats_branches_and_source(self):
        literal='<tool_response>\\(x_1\\)</tool_response>'
        tip=self.add(literal)
        self.store.close();self.store=ChatWorkspace(self.path)
        self.assertEqual(self.store.history(self.branch)['messages'][0]['text'],literal)
        self.assertEqual(self.store.history(self.branch)['tip'],tip['tip'])
        self.assertEqual(self.store.list_chats()[0]['title'],'First chat')

    def test_cas_rejects_stale_writer_and_aba(self):
        old=self.store.history(self.branch)
        self.add('P')
        self.store.rewind(self.branch,1,old['tip'],'back')
        with self.assertRaises(WorkspaceConflict):
            self.store.append(self.branch,0,{'role':'user','text':'stale'},'stale')
        self.assertEqual(self.store.history(self.branch)['revision'],2)

    def test_retry_is_idempotent_even_after_later_move(self):
        first=self.add('P',op='same-op')
        self.add('later')
        retry=self.store.append(self.branch,0,{'role':'user','text':'P','time':1},'same-op')
        self.assertEqual(retry,first)
        self.assertEqual(len(self.store.history(self.branch)['messages']),2)
        with self.assertRaises(ValueError):
            self.store.append(self.branch,0,{'role':'user','text':'different'},'same-op')

    def test_rewind_and_clear_leave_old_history_recoverable(self):
        first=self.add('P');old=self.add('A',role='assistant')
        self.store.rewind(self.branch,2,first['tip'],'rewind')
        clear=self.store.clear(self.branch,3,'clear')
        self.assertEqual(self.store.history(self.branch)['messages'],[])
        self.store.rewind(self.branch,clear['revision'],old['tip'],'recover')
        self.assertEqual([m['text'] for m in self.store.history(self.branch)['messages']],['P','A'])

    def test_cross_chat_node_is_refused(self):
        other=self.store.create_chat('Other','other-chat')['branch']['id']
        target=self.add('Other prefix',other)
        with self.assertRaises(ValueError):
            self.store.fork(self.branch,target['tip'],'Wrong','cross')

    def test_message_and_node_rows_cannot_be_updated_or_deleted(self):
        self.add('P')
        with closing(sqlite3.connect(self.path)) as db:
            for sql in ['UPDATE messages SET text="changed"','DELETE FROM messages',
                        'UPDATE history_nodes SET parent_id=NULL','DELETE FROM history_nodes']:
                with self.subTest(sql=sql),self.assertRaises(sqlite3.IntegrityError):
                    db.execute(sql)

    def test_ordered_attachments_and_tools_survive_without_branch_copies(self):
        message={'role':'assistant','text':'Result','time':2,'reasoning':'why',
                 'files':[{'name':'notes.txt','text':'literal </aaa>'}],
                 'images':[{'name':'image.png','url':'data:image/png;base64,AA=='}],
                 'tools':[{'id':'call-1','name':'calc__add','server':'calc','tool':'add',
                           'arguments':{'a':2,'b':3},'result':'5','ok':True,'state':'done','round':0,'at':0,'rat':0}]}
        before=copy.deepcopy(message)
        tip=self.store.append(self.branch,0,message,'rich')
        message['tools'][0]['arguments']['a']=99
        stored=self.store.history(self.branch)['messages'][0]
        for key in before:self.assertEqual(stored[key],before[key])
        self.store.fork(self.branch,tip['tip'],'Copy reference','ref')
        with closing(sqlite3.connect(self.path)) as db:
            self.assertEqual(db.execute('SELECT count(*) FROM messages').fetchone()[0],1)
            self.assertEqual(db.execute('SELECT count(*) FROM message_assets').fetchone()[0],2)
            self.assertEqual(db.execute('SELECT count(*) FROM message_tools').fetchone()[0],1)

    def test_rename_and_archive_are_metadata_only(self):
        tip=self.add('P')
        self.store.rename_chat(self.chat['id'],'Renamed')
        self.store.rename_branch(self.branch,'Main idea')
        self.assertEqual(self.store.history(self.branch)['tip'],tip['tip'])
        self.store.archive_chat(self.chat['id'],True)
        self.assertEqual(self.store.list_chats(),[])
        self.assertEqual(len(self.store.list_chats(True)),1)
        self.assertEqual(self.store.history(self.branch)['messages'][0]['text'],'P')

    def test_generation_reservation_blocks_only_its_branch(self):
        job=self.store.claim(self.branch,0,{'role':'user','text':'Question','time':3},{},'request')
        with self.assertRaises(WorkspaceConflict):self.store.clear(self.branch,1,'clear-busy')
        b=self.store.fork(self.branch,job['input_node'],'B','fork-pending')
        self.add('independent',b['id'])
        self.store.finish('request',{'role':'assistant','text':'Answer','time':4},'completed')
        self.assertEqual([m['text'] for m in self.store.history(self.branch)['messages']],['Question','Answer'])

    def test_restart_marks_interrupted_request_and_retains_input(self):
        self.store.claim(self.branch,0,{'role':'user','text':'Question','time':3},{},'request')
        self.store.close();self.store=ChatWorkspace(self.path)
        self.assertEqual(self.store.request('request')['status'],'interrupted')
        self.assertEqual([m['text'] for m in self.store.history(self.branch)['messages']],['Question'])
        self.add('next')

    def test_sql_constraints_and_foreign_keys_are_enabled(self):
        with self.store.connection() as db:
            self.assertEqual(db.execute('PRAGMA foreign_keys').fetchone()[0],1)
            self.assertEqual(db.execute('PRAGMA foreign_key_check').fetchall(),[])
            self.assertEqual(db.execute('PRAGMA integrity_check').fetchone()[0],'ok')
        with self.assertRaises(ValueError):self.store.create_chat('','empty')

    def test_second_server_cannot_recover_a_live_request(self):
        self.store.claim(self.branch,0,{'role':'user','text':'Question','time':3},{},'request')
        with self.assertRaises(WorkspaceConflict):ChatWorkspace(self.path)
        self.assertEqual(self.store.request('request')['status'],'pending')

    def test_archived_chat_preserves_text_and_forces_checkpoint_off(self):
        self.add('saved text')
        self.store.archive_chat(self.chat['id'],True)
        job=self.store.claim(self.branch,1,{'role':'user','text':'Read archived chat'}, {'strata_checkpoint':True},'archived-request')
        self.assertIs(job['settings']['strata_checkpoint'],False)
        self.assertEqual(self.store.history(self.branch)['messages'][0]['text'],'saved text')
        with self.assertRaises(WorkspaceConflict):self.store.archive_chat(self.chat['id'],False)
        self.store.finish('archived-request',None,'failed')
        self.store.archive_chat(self.chat['id'],False)
        next_job=self.store.claim(self.branch,2,{'role':'user','text':'Continue'}, {},'active-request')
        self.assertNotIn('strata_checkpoint',next_job['settings'])
