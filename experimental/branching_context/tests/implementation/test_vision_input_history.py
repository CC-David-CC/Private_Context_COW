"""VC01 canonical mixed-input tests. Metadata fixtures do not simulate inference."""
import tempfile
import unittest

from context_time_travel import Conflict, IntegrityError, InvalidOperation, Store


def image_span(**changes):
    value = dict(kind='image', original_sha256='a'*64, normalized_sha256='b'*64,
                 encoder_profile_id='c'*64, embedding_sha256='d'*64,
                 embedding_bytes=20 + 6*4*4, rows=6, width=4,
                 grid_width=3, grid_height=2, pad_token_id=123456)
    value.update(changes)
    return value


class VisionInputHistoryTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.store = Store(self.directory.name)
        self.controller = {'counter': 0, 'policy': {'allowed': [1, 2]}}

    def root(self, spans=None):
        return self.store.create_input_root('mixed-model-v1',
            [dict(kind='text', tokens=[1, 2]), image_span()] if spans is None else spans,
            self.controller)

    def test_vc001_different_pixels_with_same_pad_count_have_different_identity(self):
        first = self.root()
        second = self.root([dict(kind='text', tokens=[1, 2]),
                            image_span(original_sha256='e'*64, embedding_sha256='f'*64)])
        self.assertNotEqual(first, second)
        self.assertEqual(self.store.input_lineage(first)[1][-1]['rows'], 6)
        self.assertEqual(self.store.input_lineage(second)[1][-1]['rows'], 6)

    def test_vc001_equivalent_span_fragmentation_and_separate_calls_share_identity(self):
        first = self.root([dict(kind='text', tokens=[1]), dict(kind='text', tokens=[2]), image_span()])
        self.assertEqual(first, self.root())
        self.assertEqual(first, Store(self.directory.name).create_input_root(
            'mixed-model-v1', [dict(kind='text', tokens=[1, 2]), image_span()], self.controller))

    def test_vc002_legacy_history_remains_schema_one_and_exact_tokens(self):
        root = self.store.create_root('text-v1', [1, 2], self.controller)
        tail = self.store.append(root, [3], self.controller)
        self.assertEqual(self.store.get_snapshot(root)['schema'], 1)
        self.assertEqual(self.store.lineage(tail), (root, [1, 2, 3]))
        self.assertEqual(self.store.input_lineage(tail),
                         (root, [dict(kind='text', tokens=[1, 2, 3])]))

    def test_vc002_image_metadata_and_text_lists_are_detached_from_callers(self):
        spans = [dict(kind='text', tokens=[1, 2]), image_span()]
        root = self.root(spans)
        spans[0]['tokens'][0] = 99
        spans[1]['embedding_sha256'] = 'f'*64
        record = self.store.get_snapshot(root)
        record['inputs'][0]['tokens'][0] = 88
        record['inputs'][1]['embedding_sha256'] = 'e'*64
        # Exercise both the snapshot parse cache and the ancestry cache.
        lineage = self.store.input_lineage(root)[1]
        lineage[0]['tokens'][0] = 77
        lineage[1]['embedding_sha256'] = 'e'*64
        self.assertEqual(self.store.input_lineage(root)[1],
                         [dict(kind='text', tokens=[1, 2]), image_span()])

    def test_vc002_forks_append_one_canonical_mixed_history_and_keep_parent(self):
        root = self.root()
        a = self.store.append_inputs(root, [dict(kind='text', tokens=[10])], self.controller)
        b = self.store.append_inputs(root, [dict(kind='text', tokens=[20])], self.controller)
        self.store.fork('A', a)
        self.store.fork('B', b)
        reopened = Store(self.directory.name)
        self.assertEqual(reopened.input_lineage(a)[1][-1], dict(kind='text', tokens=[10]))
        self.assertEqual(reopened.input_lineage(b)[1][-1], dict(kind='text', tokens=[20]))
        self.assertEqual(reopened.input_lineage(root)[1][-1], image_span())
        self.assertEqual(reopened.head('A').tip, a)

    def test_vc002_text_only_entrypoints_cannot_discard_image_identity(self):
        root = self.root()
        with self.assertRaisesRegex(InvalidOperation, 'mixed|Mixed'):
            self.store.lineage(root)
        with self.assertRaisesRegex(InvalidOperation, 'mixed|Mixed'):
            self.store.append(root, [7], self.controller)
        text = self.store.create_root('text-v1', [1], self.controller)
        with self.assertRaisesRegex(InvalidOperation, 'mixed|Mixed'):
            self.store.append_inputs(text, [image_span()], self.controller)

    def test_vc003_changed_encoder_grid_or_placement_changes_identity(self):
        root = self.root()
        for changed in (image_span(encoder_profile_id='e'*64),
                        image_span(grid_width=2, grid_height=3)):
            with self.subTest(changed=changed):
                self.assertNotEqual(root, self.root([dict(kind='text', tokens=[1, 2]), changed]))
        self.assertNotEqual(root, self.root([image_span(), dict(kind='text', tokens=[1, 2])]))

    def test_vc003_invalid_image_metadata_rejected_before_publication(self):
        bad = [image_span(rows=5), image_span(width=True), image_span(grid_width=0),
               image_span(embedding_bytes=1), image_span(embedding_sha256='not-a-hash'),
               image_span(encoder_profile_id='C'*64), image_span(pad_token_id=-1),
               image_span(rows=2**63), dict(kind='unknown', tokens=[1])]
        for span in bad:
            with self.subTest(span=span):
                before = set(self.store.objects.iterdir())
                with self.assertRaises(InvalidOperation):
                    self.root([span])
                self.assertEqual(before, set(self.store.objects.iterdir()))

    def test_vc003_input_count_counts_image_rows_without_sampler_tokens(self):
        from context_time_travel.input_spans import input_count, text_runs
        spans = [dict(kind='text', tokens=[1, 2]), image_span(),
                 dict(kind='text', tokens=[3])]
        self.assertEqual(input_count(spans), 9)
        self.assertEqual(text_runs(spans), ((1, 2), (3,)))

    def test_vc003_unknown_stored_version_is_rejected(self):
        root = self.root()
        record = self.store.get_snapshot(root)
        record['schema'] = 3
        unknown = self.store._put_object(record)
        with self.assertRaisesRegex(IntegrityError, 'version'):
            self.store.get_snapshot(unknown)

    def test_vc003_cross_version_ancestry_is_rejected(self):
        text = self.store.create_root('mixed-model-v1', [1], self.controller)
        root = self.root()
        record = self.store.get_snapshot(root)
        record['parent'] = text
        malformed = self.store._put_object(record, text)
        with self.assertRaisesRegex(IntegrityError, 'cross-input-version'):
            self.store.input_lineage(malformed)

    def test_vc003_branch_move_retry_and_stale_revision_work_with_image_history(self):
        root = self.root()
        head = self.store.fork('main', root)
        child = self.store.append_inputs(root, [dict(kind='text', tokens=[9])], self.controller)
        moved = self.store.move('main', head.revision, child, 'mixed-move')
        self.assertEqual(moved.tip, child)
        self.assertEqual(moved, self.store.move('main', head.revision, child, 'mixed-move'))
        with self.assertRaises(Conflict):
            self.store.move('main', head.revision, root, 'stale-move')
        self.assertEqual(self.store.head('main'), moved)

    def test_vc003_missing_derived_root_index_is_repaired_from_mixed_ancestry(self):
        root = self.root()
        child = self.store.append_inputs(root, [dict(kind='text', tokens=[9])], self.controller)
        (self.store.root_index/(child+'.json')).unlink()
        reopened = Store(self.directory.name)
        head = reopened.fork('repaired', child)
        self.assertEqual(head.root, root)
        self.assertEqual(head.tip, child)
        self.assertEqual(reopened.work_counters()['root_index_repairs'], 1)

    def test_vc003_collection_keeps_live_mixed_ancestors_and_other_text_roots(self):
        root = self.root()
        child = self.store.append_inputs(root, [dict(kind='text', tokens=[9])], self.controller)
        orphan = self.store.append_inputs(root, [dict(kind='text', tokens=[8])], self.controller)
        text = self.store.create_root('legacy', [5], {})
        self.store.fork('main', child)
        self.store.release_history(child)
        self.store.release_history(orphan)
        with self.store.read_lease(orphan):
            self.assertEqual(self.store.collect_history(root)['deleted'], [])
        self.assertEqual(self.store.collect_history(root)['deleted'], [orphan])
        self.assertEqual(self.store.input_lineage(child)[0], root)
        self.assertEqual(self.store.lineage(text), (text, [5]))

    def test_vc003_move_rejects_cross_version_parent_without_moving_reference(self):
        root = self.root()
        text = self.store.create_root('mixed-model-v1', [1], self.controller)
        record = self.store.get_snapshot(root)
        record['parent'] = text
        malformed = self.store._put_object(record, text)
        head = self.store.fork('main', text)
        with self.assertRaisesRegex(IntegrityError, 'cross-input-version'):
            self.store.move('main', head.revision, malformed, 'invalid-mixed-move')
        self.assertEqual(self.store.head('main'), head)


if __name__ == '__main__':
    unittest.main()
