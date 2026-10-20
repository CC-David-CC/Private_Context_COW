"""History graph checks against real SQLite; no numerical/backend gate claims."""
from contextlib import contextmanager
import sqlite3
import tempfile
import threading
import time
import unittest
from pathlib import Path

from serve.chat_workspace import ChatWorkspace, WorkspaceNotFound, identity


class GraphTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = ChatWorkspace(Path(self.tmp.name) / 'graph.sqlite3')
        self.chat = self.store.create_chat('Graph', 'create')
        self.main = self.chat['branch']['id']

    def tearDown(self):
        self.store.close()
        self.tmp.cleanup()

    def add(self, text, branch=None):
        branch = branch or self.main
        head = self.store.history(branch)
        return self.store.append(branch, head['revision'], {'role': 'user', 'text': text, 'time': 1}, text + branch)

    def test_empty_and_single_branch_counts(self):
        graph = self.store.graph(self.chat['id'])
        self.assertEqual(len(graph['nodes']), 1)
        self.assertEqual(graph['branches'][0]['saved_messages'], 0)
        self.assertEqual(graph['shared_prefix_messages'], 0)
        self.add('one'); self.add('two')
        graph = self.store.graph(self.chat['id'])
        self.assertEqual(graph['branches'][0]['saved_messages'], 2)
        self.assertEqual(graph['shared_prefix_messages'], 0)
        self.assertEqual(sorted(n['depth'] for n in graph['nodes']), [0, 1, 2])
        summary = self.store.list_chats()[0]
        self.assertEqual((summary['branch_count'], summary['max_saved_messages']), (1, 2))

    def test_nested_shared_cores_and_different_lengths(self):
        p = self.add('common')
        c = self.store.fork(self.main, p['tip'], 'C', 'fork-c')
        q = self.add('shared by A and B')
        b = self.store.fork(self.main, q['tip'], 'B', 'fork-b')
        self.add('A ending'); self.add('B ending', b['id']); self.add('B longer', b['id'])
        self.add('C ending', c['id'])
        graph = self.store.graph(self.chat['id'])
        nodes = {n['id']: n for n in graph['nodes']}
        self.assertEqual(graph['shared_prefix_messages'], 1)
        self.assertEqual(nodes[p['tip']]['current_branch_count'], 3)
        self.assertEqual(nodes[q['tip']]['current_branch_count'], 2)
        self.assertEqual({x['name']: x['saved_messages'] for x in graph['branches']}, {'Main': 3, 'C': 2, 'B': 4})
        self.assertEqual(len(graph['nodes']), 7)

    def test_identical_tips_are_distinct_branches(self):
        p = self.add('common')
        self.store.fork(self.main, p['tip'], 'B', 'fork')
        graph = self.store.graph(self.chat['id'])
        self.assertEqual(len(graph['branches']), 2)
        self.assertEqual(len(graph['nodes']), 2)
        self.assertEqual(graph['shared_prefix_messages'], 1)

    def test_root_fork_has_no_shared_messages(self):
        root=self.chat['branch']['tip'];self.add('one')
        self.store.fork(self.main,root,'Empty alternative','root-fork')
        graph=self.store.graph(self.chat['id'])
        self.assertEqual(graph['shared_prefix_messages'],0)
        self.assertEqual(sorted(b['saved_messages'] for b in graph['branches']),[0,1])
        self.assertEqual(self.store.list_chats()[0]['shared_prefix_messages'],0)

    def test_equal_text_on_different_paths_is_not_shared(self):
        p = self.add('common')
        b = self.store.fork(self.main, p['tip'], 'B', 'fork')
        self.add('A'); self.add('B', b['id'])
        a_tip = self.add('same'); b_tip = self.add('same', b['id'])
        graph = self.store.graph(self.chat['id'])
        self.assertNotEqual(a_tip['tip'], b_tip['tip'])
        self.assertEqual(graph['shared_prefix_messages'], 1)
        self.assertEqual(sum(n['preview'] == 'same' for n in graph['nodes']), 2)

    def test_rewind_previous_tips_clear_and_restore(self):
        p = self.add('one'); old = self.add('two')
        self.store.rewind(self.main, 2, p['tip'], 'rewind')
        graph = self.store.graph(self.chat['id'])
        self.assertNotIn(old['tip'], {n['id'] for n in graph['nodes']})
        graph = self.store.graph(self.chat['id'], include_previous=True)
        self.assertEqual([x['node_id'] for x in graph['previous_tips']], [old['tip']])
        old_node = next(n for n in graph['nodes'] if n['id'] == old['tip'])
        self.assertEqual(old_node['current_branch_count'], 0)
        self.store.clear(self.main, 3, 'clear')
        self.assertEqual(self.store.list_chats()[0]['max_saved_messages'], 0)
        self.store.rewind(self.main, 4, old['tip'], 'restore')
        self.assertEqual(self.store.graph(self.chat['id'])['branches'][0]['saved_messages'], 2)

    def test_chat_boundaries_and_missing_chat(self):
        self.add('one')
        other = self.store.create_chat('Other', 'other')
        graph = self.store.graph(other['id'])
        self.assertEqual(len(graph['nodes']), 1)
        self.assertNotEqual(graph['root_id'], self.store.graph(self.chat['id'])['root_id'])
        with self.assertRaises(WorkspaceNotFound): self.store.graph('missing')

    def test_pending_answer_only_counts_durable_input(self):
        job = self.store.claim(self.main, 0, {'role': 'user', 'text': 'question'}, {}, 'pending')
        graph = self.store.graph(self.chat['id'])
        self.assertEqual(graph['branches'][0]['saved_messages'], 1)
        self.assertEqual(graph['branches'][0]['pending_request'], 'pending')
        self.store.finish(job['request_id'], {'role': 'assistant', 'text': 'answer'}, 'completed')
        graph = self.store.graph(self.chat['id'])
        self.assertEqual(graph['branches'][0]['saved_messages'], 2)
        self.assertIsNone(graph['branches'][0]['pending_request'])

    def test_previews_are_bounded_and_do_not_load_attachments(self):
        self.store.append(self.main, 0, {'role': 'user', 'text': 'x' * 1024 * 1024,
            'files': [{'name': 'large.txt', 'text': 'y' * 1024 * 1024}], 'time': 1}, 'large')
        original = self.store._read_message
        self.store._read_message = lambda *args: self.fail('graph loaded a complete message')
        try:
            graph = self.store.graph(self.chat['id'])
        finally: self.store._read_message = original
        node = next(n for n in graph['nodes'] if n['role'] == 'user')
        self.assertEqual(len(node['preview']), 160)
        self.assertNotIn('files', node)
        self.assertLess(len(str(graph)), 2500)

    def test_reads_do_not_change_database(self):
        self.add('one')
        with self.store.connection() as db: before = '\n'.join(db.iterdump())
        self.store.graph(self.chat['id']); self.store.list_chats()
        with self.store.connection() as db: after = '\n'.join(db.iterdump())
        self.assertEqual(before, after)

    def test_snapshot_remains_consistent_during_append(self):
        self.add('one')
        connection = self.store.connection
        fired = False
        errors = []
        @contextmanager
        def instrument(write=False):
            nonlocal fired
            with connection(write) as db:
                def trace(sql):
                    nonlocal fired
                    if not write and not fired and sql.lstrip().startswith('WITH RECURSIVE'):
                        fired = True
                        def writer():
                            try: self.add('concurrent')
                            except BaseException as e: errors.append(e)
                        t = threading.Thread(target=writer); t.start(); t.join(5)
                        if t.is_alive(): errors.append(RuntimeError('writer blocked'))
                db.set_trace_callback(trace)
                yield db
        self.store.connection = instrument
        try: graph = self.store.graph(self.chat['id'])
        finally: self.store.connection = connection
        self.assertTrue(fired); self.assertEqual(errors, [])
        self.assertEqual(graph['branches'][0]['saved_messages'], 1)
        self.assertEqual(len(graph['nodes']), 2)
        self.assertEqual(self.store.graph(self.chat['id'])['branches'][0]['saved_messages'], 2)

    def test_ten_thousand_messages_use_bounded_queries(self):
        # Build a large canonical fixture in one transaction, without quadratic history reads.
        root = self.chat['branch']['tip']; parent = root; nodes = []
        with self.store.connection(True) as db:
            for i in range(10000):
                mid = self.store._message(db, {'role': 'user', 'text': str(i), 'time': 1})
                node = identity([parent, mid]); db.execute('INSERT INTO history_nodes VALUES(?,?,?)', (node, parent, mid))
                nodes.append(node); parent = node
            db.execute('UPDATE branch_heads SET tip=? WHERE branch_id=?', (parent, self.main))
            for i in range(31):
                branch = 'fixture-' + str(i)
                db.execute('INSERT INTO branches VALUES(?,?,?,?)', (branch, self.chat['id'], 'B' + str(i), i))
                db.execute('INSERT INTO branch_heads VALUES(?,?,0)', (branch, nodes[9000 + i]))
        connection = self.store.connection; statements = []
        @contextmanager
        def instrument(write=False):
            with connection(write) as db:
                db.set_trace_callback(statements.append); yield db
        self.store.connection = instrument
        try:
            started=time.perf_counter()
            graph = self.store.graph(self.chat['id'])
            graph_seconds=time.perf_counter()-started;started=time.perf_counter()
            summaries = self.store.list_chats()
            summary_seconds=time.perf_counter()-started
        finally: self.store.connection = connection
        self.assertEqual(len(graph['nodes']), 10001)
        self.assertEqual(len(graph['branches']), 32)
        self.assertEqual(graph['shared_prefix_messages'], 9001)
        self.assertEqual(summaries[0]['max_saved_messages'], 10000)
        self.assertLessEqual(sum(s.lstrip().upper().startswith(('SELECT', 'WITH')) for s in statements), 10)
        self.measurements={'message_nodes':10000,'returned_nodes_including_root':len(graph['nodes']),
            'branches':len(graph['branches']),'graph_seconds':graph_seconds,'summary_seconds':summary_seconds,
            'select_statements_graph_plus_summary':sum(s.lstrip().upper().startswith(('SELECT','WITH')) for s in statements)}


if __name__ == '__main__': unittest.main()
