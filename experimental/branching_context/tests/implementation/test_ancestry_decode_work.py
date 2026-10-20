"""Derived ancestry metadata must save parsing while checking every saved byte."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from context_time_travel.store import Store, IntegrityError, NotFound
import context_time_travel.store as store_module


class AncestryDecodeWorkTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.store = Store(self.temp.name)
        self.root = self.store.create_root('model',[1,2],{'status':'idle'})
        self.tip = self.store.append(self.root,[3],{'status':'idle','large_policy':list(range(10000))})

    def test_historical_controllers_are_not_reparsed_on_repeat_ancestry_reads(self):
        with patch.object(store_module,'_decode',wraps=store_module._decode) as decode:
            self.assertEqual(self.store.lineage(self.tip),(self.root,[1,2,3]))
            first = decode.call_count
            for _ in range(5): self.assertEqual(self.store.lineage(self.tip),(self.root,[1,2,3]))
            self.assertEqual(decode.call_count,first,'Repeated ancestry must reuse only derived metadata')

    def test_warmed_ancestor_corruption_is_still_detected(self):
        self.store.lineage(self.tip)
        path = self.store.objects/(self.root+'.json')
        original = path.read_bytes()
        path.write_bytes(original.replace(b'"inputs":[1,2]',b'"inputs":[9,2]'))
        self.assertNotEqual(path.read_bytes(),original)
        with self.assertRaises(IntegrityError): self.store.lineage(self.tip)

    def test_warmed_ancestor_deletion_is_still_detected(self):
        self.store.lineage(self.tip)
        (self.store.objects/(self.root+'.json')).unlink()
        with self.assertRaises(NotFound): self.store.lineage(self.tip)

    def test_repeated_current_snapshot_reads_reuse_parsing_but_return_detached_values(self):
        with patch.object(store_module,'_decode',wraps=store_module._decode) as decode:
            first = self.store.get_snapshot(self.tip)
            count = decode.call_count
            first['controller']['large_policy'][0] = -999
            for _ in range(5):
                record = self.store.get_snapshot(self.tip)
                self.assertEqual(record['controller']['large_policy'][0],0)
            self.assertEqual(decode.call_count,count,'Verified snapshot bytes need only be parsed once')

    def test_warmed_snapshot_corruption_is_still_detected(self):
        self.store.get_snapshot(self.tip)
        path = self.store.objects/(self.tip+'.json')
        raw = path.read_bytes()
        path.write_bytes(raw.replace(b'"inputs":[3]',b'"inputs":[9]'))
        self.assertNotEqual(path.read_bytes(),raw)
        with self.assertRaises(IntegrityError): self.store.get_snapshot(self.tip)

    def test_cached_atomic_support_and_nested_grammar_are_detached(self):
        sid = self.store.append(self.root,[7],{'policy':{'allowed':list(range(10000)),
            'grammar':{'choices':[1,2]}}})
        changed = self.store.get_snapshot(sid)
        changed['controller']['policy']['allowed'][0] = 999
        changed['controller']['policy']['grammar']['choices'].append(3)
        unchanged = self.store.get_snapshot(sid)
        self.assertEqual(unchanged['controller']['policy']['allowed'][0],0)
        self.assertEqual(unchanged['controller']['policy']['grammar']['choices'],[1,2])

    def test_general_json_support_is_not_shallow_copied(self):
        sid = self.store.append(self.root,[7],{'policy':{'allowed':[{'nested':[1]}]}})
        changed = self.store.get_snapshot(sid)
        changed['controller']['policy']['allowed'][0]['nested'].append(2)
        self.assertEqual(self.store.get_snapshot(sid)['controller']['policy']['allowed'],[{'nested':[1]}])


if __name__ == '__main__': unittest.main()
