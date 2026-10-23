"""Metadata recovery controls, independent of native inference correctness."""
import copy
import random
import unittest

from context_time_travel.cache_inventory import InventoryJournal, InventoryView, RecoveryRequired


def entry(snapshot='s', tier='ram', worker='pool'):
    return dict(model='m', execution='e', snapshot=snapshot, tier=tier,
                worker=worker, checksum='a'*64, bytes=4096)


class InventoryTests(unittest.TestCase):
    def test_bounded_pool_notifications_require_rescan_after_overrun(self):
        from context_time_travel.cache_inventory import PoolNotifications
        changes=PoolNotifications(limit=2)
        initial=changes.read(None)
        self.assertTrue(initial['reset'])
        changes.record('ram',('m','e','one'))
        one=changes.read(initial['seq'])
        self.assertFalse(one['reset'])
        self.assertEqual(one['events'],[('ram',('m','e','one'))])
        changes.record('disk','two'); changes.record('disk','three')
        self.assertTrue(changes.read(initial['seq'])['reset'])
        self.assertEqual(changes.read(one['seq'])['events'],[('disk','two'),('disk','three')])

    def test_paged_recovery_is_frozen_and_visible_only_when_complete(self):
        source=InventoryJournal('node')
        rows=[entry(str(i)) for i in range(5)]; source.reconcile(rows)
        view=InventoryView('node')
        first=source.read(page_size=2)
        view.apply(first); self.assertEqual(view.entries,[])
        source.reconcile([entry('new')])
        second=source.read(page_size=2,cursor=first['next_cursor'])
        view.apply(second); view.apply(second)
        self.assertEqual(view.entries,[])
        last=source.read(page_size=2,cursor=second['next_cursor'])
        view.apply(last); view.apply(last)
        self.assertEqual(view.entries,rows)
        view.apply(source.read(**view.cursor))
        self.assertEqual(view.entries,[entry('new')])

    def test_page_gap_and_corruption_never_replace_a_live_view(self):
        source=InventoryJournal('node'); source.reconcile([entry('old')])
        view=InventoryView('node'); view.apply(source.read()); before=view.entries
        source.reconcile([entry(str(i)) for i in range(4)])
        first=source.read(page_size=2)
        last=source.read(page_size=2,cursor=first['next_cursor'])
        with self.assertRaises(RecoveryRequired):view.apply(last)
        self.assertEqual(view.entries,before)
        view.apply(first)
        bad=copy.deepcopy(last); bad['entries'][0]['bytes']=-1
        with self.assertRaises(ValueError):view.apply(bad)
        self.assertEqual(view.entries,before)
        view.apply(last)
        self.assertEqual(len(view.entries),4)

    def test_paged_sessions_are_bounded_expire_and_reject_cursor_changes(self):
        now=[0.]
        source=InventoryJournal('node',snapshot_limit=1,snapshot_ttl=5,clock=lambda:now[0])
        source.reconcile([entry(str(i)) for i in range(3)])
        first=source.read(page_size=1)
        with self.assertRaises(ValueError):source.read(page_size=2,cursor=first['next_cursor'])
        second=source.read(page_size=1)
        with self.assertRaises(RecoveryRequired):source.read(page_size=1,cursor=first['next_cursor'])
        now[0]=6
        with self.assertRaises(RecoveryRequired):source.read(page_size=1,cursor=second['next_cursor'])
        for size in (True,0,-1,1025):
            with self.subTest(size=size),self.assertRaises(ValueError):source.read(page_size=size)

    def test_snapshot_delta_removal_and_repeated_delivery(self):
        source=InventoryJournal('node', event_limit=4)
        view=InventoryView('node')
        source.reconcile([entry()]); view.apply(source.read())
        cursor=view.cursor
        source.reconcile([entry('other')])
        delta=source.read(**cursor)
        view.apply(delta); view.apply(delta)
        self.assertEqual(view.entries, [entry('other')])
        self.assertEqual(source.read(**view.cursor)['events'], [])

    def test_event_overrun_returns_complete_recovery(self):
        source=InventoryJournal('node', event_limit=1)
        source.reconcile([entry()]); view=InventoryView('node')
        view.apply(source.read()); old=view.cursor
        source.reconcile([entry('two')]); source.reconcile([entry('three')])
        recovery=source.read(**old)
        self.assertEqual(recovery['kind'], 'snapshot')
        view.apply(recovery); self.assertEqual(view.entries, [entry('three')])

    def test_restart_changes_epoch_and_cannot_keep_old_locations(self):
        first=InventoryJournal('node'); first.reconcile([entry()])
        view=InventoryView('node'); view.apply(first.read())
        second=InventoryJournal('node')
        self.assertNotEqual(first.read()['epoch'], second.read()['epoch'])
        view.apply(second.read(**view.cursor)); self.assertEqual(view.entries, [])

    def test_gap_and_malformed_batch_do_not_partially_change_view(self):
        source=InventoryJournal('node'); source.reconcile([entry()])
        view=InventoryView('node'); view.apply(source.read()); before=view.entries
        source.reconcile([entry('two')]); delta=source.read(**view.cursor)
        bad=copy.deepcopy(delta); bad['events'][0]['seq']+=1
        with self.assertRaises(RecoveryRequired):view.apply(bad)
        self.assertEqual(view.entries,before)
        bad=copy.deepcopy(delta); bad['events'][-1]['entry']['bytes']=-1
        with self.assertRaises(ValueError):view.apply(bad)
        self.assertEqual(view.entries,before)

    def test_capacity_and_identity_domains_are_enforced_atomically(self):
        source=InventoryJournal('node', entry_limit=1); source.reconcile([entry()])
        before=source.read()
        with self.assertRaises(ValueError):source.reconcile([entry(),entry('two')])
        self.assertEqual(source.read(),before)
        wrong=entry(); wrong['execution']='different'
        source.reconcile([wrong]); self.assertEqual(source.read()['entries'],[wrong])
        with self.assertRaises(ValueError):source.read(epoch=before['epoch'],since=True)

    def test_returned_metadata_cannot_mutate_source_or_view(self):
        source=InventoryJournal('node'); source.reconcile([entry()])
        packet=source.read(); view=InventoryView('node'); view.apply(packet)
        packet['entries'][0]['bytes']=1
        view.entries[0]['bytes']=2
        self.assertEqual(source.read()['entries'],[entry()])
        self.assertEqual(view.entries,[entry()])

    def test_randomized_dropped_polls_recover_exact_current_locations(self):
        rng=random.Random(7375)
        source=InventoryJournal('node',event_limit=4,entry_limit=32)
        view=InventoryView('node'); current={}
        for i in range(1000):
            key=str(rng.randrange(32))
            if rng.randrange(2):current[key]=entry(key)
            else:current.pop(key,None)
            source.reconcile(list(current.values()))
            if rng.randrange(5)==0:
                view.apply(source.read(**view.cursor))
                self.assertEqual({r['snapshot'] for r in view.entries},set(current))
        view.apply(source.read(**view.cursor))
        self.assertEqual({r['snapshot'] for r in view.entries},set(current))

    def test_retired_epoch_cannot_reappear_after_restart(self):
        first=InventoryJournal('node'); second=InventoryJournal('node')
        view=InventoryView('node'); old=first.read()
        view.apply(old); view.apply(second.read())
        with self.assertRaises(RecoveryRequired):view.apply(old)


if __name__=='__main__':unittest.main()
