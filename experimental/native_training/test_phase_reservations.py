"""CPU/filesystem qualification only; no model or live phase budget is qualified."""
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock
from types import SimpleNamespace

from experimental.native_training import phase_reservations as pr


EVIDENCE = hashlib.sha256(b'controlled test allocation cap, not a live policy').hexdigest()
POLICIES = (pr.PhasePolicy('native', 70, 'QUALIFIED', EVIDENCE),
            pr.PhasePolicy('evaluate', 10, 'QUALIFIED', EVIDENCE),
            pr.PhasePolicy('large-evaluate', 60, 'QUALIFIED', EVIDENCE),
            pr.PhasePolicy('unqualified', 1))


def _open(metadata, pool):
    return pr.PhaseReservations(metadata, pool, floor_bytes=0, policies=POLICIES)


def _fake_capacity(value):
    return lambda path: SimpleNamespace(f_bavail=value.value, f_frsize=1)


def _reserve_process(metadata, pool, operation, free, ready, start, output):
    budget = _open(metadata, pool)
    ready.set()
    start.wait(10)
    pr.os.statvfs = _fake_capacity(free)
    try:
        budget.reserve(operation, 'native')
        output.put('admitted')
    except pr.CapacityUnavailable:
        output.put('refused')


def _hold_inherited(fd, ready, release):
    # fd is an inherited open file description, not a freshly acquired lock.
    ready.set()
    release.wait(10)
    os.close(fd)


def _crash_transition(metadata, pool, artifact, when):
    budget = _open(metadata, pool)
    original = pr.os.replace
    def replace(source, destination):
        if when == 'before':
            os._exit(21)
        original(source, destination)
        os._exit(22)
    pr.os.replace = replace
    budget.transition('original', revision=0, next_phase='evaluate', artifacts=[artifact])
    os._exit(99)


class PhaseReservationTests(unittest.TestCase):
    def setUp(self):
        if not Path('/dev/shm').is_dir():
            self.skipTest('Linux tmpfs required for separate-filesystem qualification')
        self.meta_parent = tempfile.TemporaryDirectory(prefix='pcc-phase-metadata-')
        self.pool_temp = tempfile.TemporaryDirectory(prefix='pcc-phase-pool-', dir='/dev/shm')
        self.addCleanup(self.pool_temp.cleanup)
        self.addCleanup(self.meta_parent.cleanup)
        self.pool = Path(self.pool_temp.name)
        self.metadata = Path(self.meta_parent.name)/'ledger'
        if self.pool.stat().st_dev == self.metadata.parent.stat().st_dev:
            self.skipTest('Separate metadata/pool filesystems required')
        self.budget = pr.PhaseReservations.create(self.metadata, self.pool, floor_bytes=0, policies=POLICIES)
        self.context = mp.get_context('fork')

    def artifact(self, value=b'finished output', name='candidate.bin'):
        path = self.pool/name
        path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        path.write_bytes(value)
        path.chmod(0o400)
        return pr.Artifact(name, len(value), hashlib.sha256(value).hexdigest())

    def start(self, target, *args):
        child = self.context.Process(target=target, args=args)
        child.start()
        def cleanup():
            if child.is_alive():
                child.terminate()
            child.join(5)
        self.addCleanup(cleanup)
        return child

    def test_materialized_claim_shrinks_and_finishes_with_original_provenance(self):
        self.budget.reserve('original', 'native')
        artifact = self.artifact()
        event = self.budget.transition('original', revision=0, next_phase='evaluate', artifacts=[artifact])
        self.assertEqual((event['previous_bytes'], event['remaining_bytes']), (70, 10))
        self.assertEqual(self.budget.snapshot()['active']['original']['revision'], 1)
        self.budget.finish('original', revision=1, artifacts=[artifact])
        state = _open(self.metadata, self.pool).snapshot()
        self.assertFalse(state['active'])
        self.assertEqual(len(state['completed']['original']['events']), 2)
        with self.assertRaisesRegex(pr.ReservationError, 'never redispatch'):
            self.budget.reserve('original', 'native')

    def test_unqualified_policies_cannot_admit_or_shrink(self):
        with self.assertRaisesRegex(pr.ReservationError, 'no qualified'):
            self.budget.reserve('unknown', 'unqualified')
        self.budget.reserve('original', 'native')
        with self.assertRaisesRegex(pr.ReservationError, 'no qualified'):
            self.budget.transition('original', revision=0, next_phase='unqualified', artifacts=[self.artifact()])
        self.assertEqual(self.budget.snapshot()['active']['original']['remaining_bytes'], 70)

    def test_real_capacity_sampler_refuses_impossible_allocation(self):
        capacity = os.statvfs(self.pool)
        maximum = capacity.f_bavail*capacity.f_frsize + 1
        policies = [pr.PhasePolicy('too-large', maximum, 'QUALIFIED', EVIDENCE)]
        other = pr.PhaseReservations.create(Path(self.meta_parent.name)/'other', self.pool,
                                            floor_bytes=0, policies=policies)
        with self.assertRaises(pr.CapacityUnavailable):
            other.reserve('impossible', 'too-large')

    def test_concurrent_process_admission_uses_one_serialized_pool_snapshot(self):
        free = self.context.Value('q', 100)
        start, output = self.context.Event(), self.context.Queue()
        ready = [self.context.Event(), self.context.Event()]
        children = [self.start(_reserve_process, self.metadata, self.pool, str(i), free, ready[i], start, output)
                    for i in range(2)]
        for event in ready:
            self.assertTrue(event.wait(5))
        start.set()
        outcomes = [output.get(timeout=10) for child in children]
        for child in children:
            child.join(5)
            self.assertEqual(child.exitcode, 0)
        self.assertEqual(sorted(outcomes), ['admitted', 'refused'])
        self.assertEqual(len(self.budget.snapshot()['active']), 1)

    def test_capacity_is_sampled_after_waiting_for_ledger_lock(self):
        free = self.context.Value('q', 100)
        ready, start, output = self.context.Event(), self.context.Event(), self.context.Queue()
        child = self.start(_reserve_process, self.metadata, self.pool, 'late', free, ready, start, output)
        self.assertTrue(ready.wait(5))
        with self.budget._locked():
            start.set()
            free.value = 50
        self.assertEqual(output.get(timeout=10), 'refused')
        child.join(5)
        self.assertEqual(child.exitcode, 0)
        self.assertFalse(self.budget.snapshot()['active'])

    def test_capacity_sampler_runs_while_global_lock_is_held(self):
        import fcntl
        def sample(path):
            fd = os.open(self.metadata/'ledger.lock', os.O_RDWR)
            try:
                with self.assertRaises(BlockingIOError):
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            finally:
                os.close(fd)
            return SimpleNamespace(f_bavail=100, f_frsize=1)
        with mock.patch.object(pr.os, 'statvfs', side_effect=sample):
            self.budget.reserve('original', 'native')
            self.budget.transition('original', revision=0, next_phase='evaluate', artifacts=[self.artifact()])

    def test_owner_floor_is_retained_in_every_admission(self):
        other = pr.PhaseReservations.create(Path(self.meta_parent.name)/'with-floor', self.pool,
                                            floor_bytes=40, policies=POLICIES)
        with mock.patch.object(pr.os, 'statvfs', return_value=SimpleNamespace(f_bavail=100, f_frsize=1)):
            with self.assertRaises(pr.CapacityUnavailable):
                other.reserve('native-denied', 'native')
            other.reserve('evaluation-fits', 'evaluate')
            self.assertEqual(other.snapshot()['floor_bytes'], 40)

    def test_partial_ledger_write_error_retains_old_claim(self):
        self.budget.reserve('original', 'native')
        artifact = self.artifact()
        original = pr.os.fdopen
        class PartialFile:
            def __init__(self, stream):
                self.stream = stream
            def __enter__(self):
                return self
            def __exit__(self, *args):
                self.stream.close()
            def write(self, raw):
                self.stream.write(raw[:7])
                self.stream.flush()
                raise OSError('controlled partial metadata write')
        def fdopen(fd, mode):
            stream = original(fd, mode)
            return PartialFile(stream) if mode == 'wb' else stream
        with mock.patch.object(pr.os, 'fdopen', side_effect=fdopen):
            with self.assertRaisesRegex(OSError, 'partial metadata write'):
                self.budget.transition('original', revision=0, next_phase='evaluate', artifacts=[artifact])
        self.assertEqual(self.budget.snapshot()['active']['original']['remaining_bytes'], 70)
        self.assertEqual(len(next(self.metadata.glob('.ledger-*.pending')).read_bytes()), 7)

    def test_byte_correct_unsealed_partial_output_still_cannot_release_claim(self):
        self.budget.reserve('original', 'native')
        artifact = self.artifact()
        (self.pool/artifact.relative_path).chmod(0o600)
        with self.assertRaises(pr.ReservationError):
            self.budget.finish('original', revision=0, artifacts=[artifact])
        self.assertIn('original', self.budget.snapshot()['active'])

    def test_callers_cannot_supply_a_stale_free_capacity_sample(self):
        with self.assertRaises(TypeError):
            self.budget.reserve('bad', 'native', free_ram_bytes=2**60)

    def test_inherited_child_writer_blocks_shrink_after_parent_fd_closes(self):
        self.budget.reserve('original', 'native')
        artifact = self.artifact()
        ready, release = self.context.Event(), self.context.Event()
        with self.budget.writer_lease('original', revision=0) as fd:
            child = self.start(_hold_inherited, fd, ready, release)
            self.assertTrue(ready.wait(5))
        with self.assertRaises(pr.WriterBusy):
            self.budget.transition('original', revision=0, next_phase='evaluate', artifacts=[artifact])
        self.assertEqual(self.budget.snapshot()['active']['original']['remaining_bytes'], 70)
        release.set()
        child.join(5)
        self.assertEqual(child.exitcode, 0)
        self.budget.transition('original', revision=0, next_phase='evaluate', artifacts=[artifact])

    def test_writer_rejects_stale_revision_after_transition(self):
        self.budget.reserve('original', 'native')
        self.budget.transition('original', revision=0, next_phase='evaluate', artifacts=[self.artifact()])
        with self.assertRaises(pr.StaleRevision):
            with self.budget.writer_lease('original', revision=0):
                self.fail('stale writer entered')
        with self.budget.writer_lease('original', revision=1):
            pass

    def test_missing_partial_or_wrong_digest_artifacts_retain_original_claim(self):
        self.budget.reserve('original', 'native')
        artifact = self.artifact()
        invalid = [[], [pr.Artifact('absent.bin', 0, hashlib.sha256(b'').hexdigest())],
                   [pr.Artifact('candidate.bin', artifact.bytes+1, artifact.sha256)],
                   [pr.Artifact('candidate.bin', artifact.bytes, '0'*64)]]
        for manifest in invalid:
            with self.assertRaises((pr.ReservationError, FileNotFoundError)):
                self.budget.transition('original', revision=0, next_phase='evaluate', artifacts=manifest)
            self.assertEqual(self.budget.snapshot()['active']['original']['remaining_bytes'], 70)

    def test_unsealed_artifact_refuses_transition(self):
        self.budget.reserve('original', 'native')
        artifact = self.artifact()
        (self.pool/artifact.relative_path).chmod(0o600)
        with self.assertRaisesRegex(pr.ReservationError, 'sealed'):
            self.budget.transition('original', revision=0, next_phase='evaluate', artifacts=[artifact])

    def test_symlink_and_hardlink_artifacts_are_refused(self):
        self.budget.reserve('original', 'native')
        artifact = self.artifact()
        (self.pool/'link.bin').symlink_to(self.pool/artifact.relative_path)
        with self.assertRaises(OSError):
            self.budget.transition('original', revision=0, next_phase='evaluate',
                                   artifacts=[pr.Artifact('link.bin', artifact.bytes, artifact.sha256)])
        os.link(self.pool/artifact.relative_path, self.pool/'hard.bin')
        with self.assertRaisesRegex(pr.ReservationError, 'sealed'):
            self.budget.transition('original', revision=0, next_phase='evaluate', artifacts=[artifact])

    def test_artifact_changed_during_hash_refuses_transition(self):
        self.budget.reserve('original', 'native')
        artifact = self.artifact()
        original = hashlib.file_digest
        def mutate(stream, algorithm):
            digest = original(stream, algorithm)
            target = self.pool/artifact.relative_path
            target.chmod(0o600)
            target.write_bytes(b'x'*artifact.bytes)
            target.chmod(0o400)
            return digest
        with mock.patch.object(pr.hashlib, 'file_digest', side_effect=mutate):
            with self.assertRaisesRegex(pr.ReservationError, 'identity differ'):
                self.budget.transition('original', revision=0, next_phase='evaluate', artifacts=[artifact])

    def test_exact_duplicate_transition_is_idempotent_changed_intent_refused(self):
        self.budget.reserve('original', 'native')
        artifact = self.artifact()
        first = self.budget.transition('original', revision=0, next_phase='evaluate', artifacts=[artifact])
        again = self.budget.transition('original', revision=0, next_phase='evaluate', artifacts=[artifact])
        self.assertEqual(first, again)
        with self.assertRaises(pr.StaleRevision):
            self.budget.transition('original', revision=0, next_phase='large-evaluate', artifacts=[artifact])
        self.budget.finish('original', revision=1, artifacts=[artifact])
        self.assertEqual(first, self.budget.transition('original', revision=0, next_phase='evaluate', artifacts=[artifact]))

    def test_capacity_increase_requires_fresh_admission(self):
        self.budget.reserve('original', 'native')
        artifact = self.artifact()
        self.budget.transition('original', revision=0, next_phase='evaluate', artifacts=[artifact])
        with mock.patch.object(pr.os, 'statvfs', return_value=SimpleNamespace(f_bavail=40, f_frsize=1)):
            with self.assertRaises(pr.CapacityUnavailable):
                self.budget.transition('original', revision=1, next_phase='large-evaluate', artifacts=[artifact])
        self.assertEqual(self.budget.snapshot()['active']['original']['remaining_bytes'], 10)

    def test_crash_before_replace_keeps_original_claim_and_partial_is_not_receipt(self):
        self.budget.reserve('original', 'native')
        artifact = self.artifact()
        child = self.start(_crash_transition, self.metadata, self.pool, artifact, 'before')
        child.join(10)
        self.assertEqual(child.exitcode, 21)
        self.assertEqual(_open(self.metadata, self.pool).snapshot()['active']['original']['remaining_bytes'], 70)
        self.assertTrue(list(self.metadata.glob('.ledger-*.pending')))
        self.budget.transition('original', revision=0, next_phase='evaluate', artifacts=[artifact])

    def test_process_crash_after_replace_retains_durable_new_identity_without_redispatch(self):
        self.budget.reserve('original', 'native')
        artifact = self.artifact()
        child = self.start(_crash_transition, self.metadata, self.pool, artifact, 'after')
        child.join(10)
        self.assertEqual(child.exitcode, 22)
        state = _open(self.metadata, self.pool).snapshot()
        self.assertEqual(state['active']['original']['remaining_bytes'], 10)
        self.assertEqual(state['active']['original']['revision'], 1)
        self.assertEqual(self.budget.transition('original', revision=0, next_phase='evaluate', artifacts=[artifact]),
                         state['active']['original']['events'][0])

    def test_truncated_pending_file_does_not_override_last_good_ledger(self):
        self.budget.reserve('original', 'native')
        (self.metadata/'.ledger-abandoned.pending').write_bytes(b'{')
        self.assertEqual(self.budget.snapshot()['active']['original']['remaining_bytes'], 70)
        with (self.metadata/'ledger.json').open('wb') as stream:
            stream.write(b'{')
        with self.assertRaises(json.JSONDecodeError):
            self.budget.snapshot()

    def test_negative_or_understated_persisted_claim_refuses_new_admission(self):
        self.budget.reserve('original', 'native')
        data = self.budget.snapshot()
        data['active']['original']['remaining_bytes'] = -1
        self.budget._save(data)
        with self.assertRaisesRegex(pr.ReservationError, 'phase/claim differs'):
            self.budget.reserve('not-admitted', 'native')

    def test_current_phase_only_corruption_cannot_lower_original_claim(self):
        self.budget.reserve('original', 'native')
        data = self.budget.snapshot()
        data['active']['original'].update(phase='evaluate', remaining_bytes=10)
        self.budget._save(data)
        with self.assertRaisesRegex(pr.ReservationError, 'original transition chain'):
            self.budget.reserve('not-admitted', 'native')

    def test_admission_phase_cannot_be_rewritten_without_original_digest(self):
        self.budget.reserve('original', 'native')
        data = self.budget.snapshot()
        data['active']['original']['admission']['phase'] = 'evaluate'
        self.budget._save(data)
        with self.assertRaisesRegex(pr.ReservationError, 'original admission differs'):
            self.budget.snapshot()

    def test_corrupted_transition_phase_and_bounds_refused_even_with_rehashed_payload(self):
        self.budget.reserve('original', 'native')
        artifact = self.artifact()
        self.budget.transition('original', revision=0, next_phase='evaluate', artifacts=[artifact])
        original = self.budget.snapshot()
        cases = [('from_phase', 'evaluate'), ('previous_bytes', 10),
                 ('remaining_bytes', 1), ('next_phase', 'large-evaluate')]
        for key, value in cases:
            with self.subTest(key=key):
                data = json.loads(json.dumps(original))
                event = data['active']['original']['events'][0]
                event[key] = value
                intent = {k: event[k] for k in ('from_revision', 'next_phase', 'artifacts')}
                event['intent_sha256'] = pr._digest(intent)
                event['event_sha256'] = pr._digest({k: v for k, v in event.items() if k != 'event_sha256'})
                self.budget._save(data)
                with self.assertRaisesRegex(pr.ReservationError, 'chain differs|bound/capacity differs'):
                    self.budget.snapshot()
        self.budget._save(original)
        expected = original['active']['original']['events'][0]
        self.assertEqual(self.budget.transition('original', revision=0, next_phase='evaluate', artifacts=[artifact]), expected)

    def test_completed_registry_requires_actual_terminal_event(self):
        self.budget.reserve('original', 'native')
        self.budget.transition('original', revision=0, next_phase='evaluate', artifacts=[self.artifact()])
        data = self.budget.snapshot()
        data['completed']['original'] = data['active'].pop('original')
        self.budget._save(data)
        with self.assertRaisesRegex(pr.ReservationError, 'original transition chain'):
            self.budget.snapshot()

    def test_current_phase_must_match_last_event_next_phase(self):
        self.budget.reserve('original', 'native')
        self.budget.transition('original', revision=0, next_phase='large-evaluate', artifacts=[self.artifact()])
        data = self.budget.snapshot()
        data['active']['original'].update(phase='evaluate', remaining_bytes=10)
        self.budget._save(data)
        with self.assertRaisesRegex(pr.ReservationError, 'original transition chain'):
            self.budget.snapshot()

    def test_pinned_floor_and_registry_cannot_be_relaxed_on_reopen(self):
        with self.assertRaisesRegex(pr.ReservationError, 'Pinned'):
            pr.PhaseReservations(self.metadata, self.pool, floor_bytes=1, policies=POLICIES)
        changed = [pr.PhasePolicy('native', 1, 'QUALIFIED', EVIDENCE)]
        with self.assertRaisesRegex(pr.ReservationError, 'Pinned'):
            pr.PhaseReservations(self.metadata, self.pool, floor_bytes=0, policies=changed)

    def test_metadata_cannot_consume_guarded_working_storage_pool(self):
        with self.assertRaisesRegex(ValueError, 'separate filesystem'):
            pr.PhaseReservations.create(self.pool/'metadata', self.pool, floor_bytes=0, policies=POLICIES)

    def test_normal_flow_never_dispatches_subprocesses(self):
        with mock.patch('subprocess.Popen', side_effect=AssertionError('dispatch forbidden')):
            self.budget.reserve('original', 'native')
            with self.budget.writer_lease('original', revision=0):
                pass
            self.budget.finish('original', revision=0, artifacts=[self.artifact()])

    def test_qualified_policy_requires_bound_evidence(self):
        with self.assertRaisesRegex(ValueError, 'pinned bound evidence'):
            pr.PhasePolicy('native', 1, 'QUALIFIED').validate()

    def test_generator_policy_registry_can_be_reopened(self):
        generated = pr.PhaseReservations.create(Path(self.meta_parent.name)/'generated', self.pool,
                                                floor_bytes=0, policies=(x for x in POLICIES))
        self.assertEqual(generated.snapshot()['policies']['native']['maximum_additional_bytes'], 70)


if __name__ == '__main__':
    unittest.main()
