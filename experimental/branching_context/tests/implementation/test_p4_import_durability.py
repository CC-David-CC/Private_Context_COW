"""P4 arm imports checkpoint files durably before disk-tier measurements."""
import hashlib
from pathlib import Path
import struct
import tempfile
import unittest
from unittest import mock

from context_time_travel.store import Store, _json_bytes
from experimental.branching_context.scripts import p4_return_latency as p4


class P4ImportDurabilityTests(unittest.TestCase):
    def make_source(self, root: Path):
        source = root / 'source'
        source_store = Store(source)
        contract = 'contract-test'
        execution = 'profile-test'
        sid = source_store.create_root(contract, [11, 12], {'status': 'idle'})
        _, history = source_store.lineage(sid)

        state = b'native-state'
        logits = (1.25, -2.5)
        payload = state + struct.pack('<2f', *logits)
        payload_sha = hashlib.sha256(payload).hexdigest()
        manifest = dict(
            schema=1,
            snapshot=sid,
            input_identity=hashlib.sha256(_json_bytes(history)).hexdigest(),
            model_contract_id=contract,
            execution_profile_id=execution,
            consumed=len(history),
            positions='absolute-zero-based',
            logits_count=len(logits),
            state_bytes=len(state),
            payload_sha256=payload_sha,
        )
        checkpoints = source / 'checkpoints'
        checkpoints.mkdir()
        (checkpoints / (payload_sha + '.state')).write_bytes(payload)
        envelope = dict(manifest=manifest,
                        checksum=hashlib.sha256(_json_bytes(manifest)).hexdigest())
        (checkpoints / (sid + '.json')).write_bytes(_json_bytes(envelope))
        state_row = dict(snapshot=sid, history=history,
                         controller=source_store.get_snapshot(sid)['controller'],
                         manifest=manifest)
        profile = dict(model_contract_id=contract, execution_profile_id=execution,
                       execution=dict(resume_contract='legacy'))
        return source, state_row, profile, payload_sha, sid

    def test_import_syncs_checkpoint_files_before_atomic_publish(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            source, root_state, profile, payload_sha, snapshot = self.make_source(base)
            target = base / 'arm' / 'store'
            target.parent.mkdir()
            real_fsync = p4.os.fsync
            real_replace = p4.os.replace
            events = []
            synced_paths = []

            def record_fsync(fd):
                try:
                    path = Path(f'/proc/self/fd/{fd}').resolve()
                    synced_paths.append(path)
                    events.append(('fsync', path))
                except OSError:
                    pass
                return real_fsync(fd)

            def record_replace(source, destination):
                result = real_replace(source, destination)
                if Path(destination) == target:
                    events.append(('publish_rename', Path(source), Path(destination)))
                return result

            with mock.patch.object(p4.os, 'fsync', side_effect=record_fsync), \
                    mock.patch.object(p4.os, 'replace', side_effect=record_replace):
                p4.import_minimal_store(source, target, [root_state], profile)

            self.assertEqual(hashlib.sha256(
                (target / 'checkpoints' / (payload_sha + '.state')).read_bytes()
            ).hexdigest(), payload_sha)
            self.assertTrue((target / 'checkpoints' / (snapshot + '.json')).is_file())
            payload_sync = next(i for i, path in enumerate(synced_paths)
                                if path.name == payload_sha + '.state')
            manifest_sync = next(i for i, path in enumerate(synced_paths)
                                 if path.name == snapshot + '.json')
            checkpoint_dir_sync = next(i for i, path in enumerate(synced_paths)
                                       if path.name == 'checkpoints')
            store_dir_sync = max(i for i, path in enumerate(synced_paths)
                                 if path.name == 'store' and
                                 path.parent.name.startswith('.store.staging-'))
            publish_dir_sync = max(i for i, path in enumerate(synced_paths)
                                   if path == target.parent)
            self.assertLess(payload_sync, checkpoint_dir_sync)
            self.assertLess(manifest_sync, checkpoint_dir_sync)
            self.assertLess(checkpoint_dir_sync, store_dir_sync)
            rename_index = next(i for i, event in enumerate(events)
                                if event[0] == 'publish_rename')
            store_event = max(i for i, event in enumerate(events)
                              if event[0] == 'fsync' and event[1].name == 'store'
                              and event[1].parent.name.startswith('.store.staging-'))
            parent_event = max(i for i, event in enumerate(events)
                               if event[0] == 'fsync' and event[1] == target.parent)
            checkpoint_event = next(i for i, event in enumerate(events)
                                    if event[0] == 'fsync' and event[1].name == 'checkpoints')
            self.assertLess(checkpoint_event, store_event)
            self.assertLess(store_event, rename_index)
            self.assertGreater(parent_event, rename_index)

    def test_store_directory_fsync_failure_does_not_publish_target(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            source, root_state, profile, _, _ = self.make_source(base)
            target = base / 'arm' / 'store'
            target.parent.mkdir()
            real_fsync = p4.os.fsync

            def fail_store_fsync(fd):
                path = Path(f'/proc/self/fd/{fd}').resolve()
                if path.name == 'store' and (path / 'checkpoints').is_dir():
                    raise OSError('simulated store-directory sync failure')
                return real_fsync(fd)

            with mock.patch.object(p4.os, 'fsync', side_effect=fail_store_fsync):
                with self.assertRaisesRegex(OSError, 'store-directory sync failure'):
                    p4.import_minimal_store(source, target, [root_state], profile)
            self.assertFalse(target.exists())


if __name__ == '__main__':
    unittest.main()
