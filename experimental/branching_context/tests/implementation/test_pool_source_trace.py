"""Campaign planning assertions; these do not qualify GPU inference."""
import unittest
import json
from context_time_travel.store import BranchHead
from pool_campaign import tree_root, validate_rewind, rewind_record


class SourceTraceTests(unittest.TestCase):
    def test_native_branch_move_result_is_serializable_evidence(self):
        before=BranchHead('working',2,'answer','root')
        after=BranchHead('working',3,'root','root')
        sibling=BranchHead('saved-A',0,'answer','root')
        row=rewind_record(0,1,before,after,after,sibling,'new-answer')
        decoded=json.loads(json.dumps(row,allow_nan=False))
        self.assertEqual(decoded['receipt'],dict(name='working',revision=3,tip='root',root='root'))
        self.assertEqual(decoded['before']['tip'],'answer')
        self.assertEqual(decoded['snapshot'],'new-answer')

    def test_explicit_mapping_keeps_each_pair_on_its_own_source(self):
        fixture = {'tree_roots':[0,0,1,1,2,2]}
        self.assertEqual([tree_root(fixture,t,3) for t in range(6)], [0,0,1,1,2,2])

    def test_original_synthetic_mapping_is_preserved(self):
        self.assertEqual([tree_root({},t,2) for t in range(6)], [0,1,0,1,0,1])

    def test_invalid_mapping_is_rejected(self):
        for mapping in ([True],[-1],[3],['0'],[]):
            with self.subTest(mapping=mapping), self.assertRaises(ValueError):
                tree_root({'tree_roots':mapping},0,3)

    def test_rewind_moves_only_the_working_reference(self):
        before = {'tip':'answer','revision':1,'root':'prefix','name':'working'}
        after = dict(before,tip='prefix',revision=2)
        sibling = dict(before,name='saved-A')
        validate_rewind(before,after,'prefix',sibling,sibling)
        for bad in (dict(after,revision=1),dict(after,tip='answer'),dict(after,root='other')):
            with self.subTest(bad=bad),self.assertRaises(AssertionError):
                validate_rewind(before,bad,'prefix',sibling,sibling)
        with self.assertRaises(AssertionError):
            validate_rewind(before,after,'prefix',sibling,dict(sibling,tip='prefix'))


if __name__ == '__main__': unittest.main()
