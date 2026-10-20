"""Deterministic idle expiry for derived RAM and disk checkpoints."""
from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
import unittest
from dataclasses import replace

from context_time_travel.pools import DiskCheckpointPool, PoolConfig, SharedCheckpointPool
from context_time_travel.real import Checkpoint
from context_time_travel.store import Store


def checkpoint(blob: bytes, identity: str | None = None) -> Checkpoint:
    frontier = (1.25, -2.5)
    return Checkpoint("model", "execution", 1, blob, frontier,
                      Checkpoint.digest(blob, frontier), identity or blob.hex())


class FakeClock:
    def __init__(self, value: float):
        self.value = value

    def __call__(self):
        return self.value


class PoolConfigIdleExpiryTests(unittest.TestCase):
    def test_default_is_disabled_and_valid_durations_are_retained(self):
        self.assertIsNone(PoolConfig().checkpoint_idle_seconds)
        self.assertEqual(PoolConfig(checkpoint_idle_seconds=1800).checkpoint_idle_seconds, 1800)
        self.assertEqual(PoolConfig(checkpoint_idle_seconds=0.25).checkpoint_idle_seconds, 0.25)

    def test_rejects_nonpositive_nonfinite_boolean_and_nonnumber_durations(self):
        for value in (0, -1, 0.0, -0.1, True, False, float("inf"), float("nan"), "1800"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                PoolConfig(checkpoint_idle_seconds=value)


class SharedPoolIdleExpiryTests(unittest.TestCase):
    def make_pool(self, clock: FakeClock):
        return SharedCheckpointPool(4096, checkpoint_idle_seconds=10, clock=clock)

    def test_idle_age_uses_strict_cutoff_and_get_refreshes(self):
        clock = FakeClock(100.0)
        pool = self.make_pool(clock)
        pool.admit("a", checkpoint(b"a"), durable=True)
        clock.value = 109.0
        self.assertEqual(pool.get("a").blob, b"a")
        clock.value = 119.0
        self.assertEqual(pool.collect_expired(), 0, "exactly idle is retained")
        clock.value = 119.01
        self.assertEqual(pool.collect_expired(), 1)
        self.assertNotIn("a", pool)

    def test_metrics_and_misses_do_not_refresh_and_clock_rollback_is_conservative(self):
        clock = FakeClock(10.0)
        pool = self.make_pool(clock)
        pool.admit("a", checkpoint(b"a"), durable=True)
        with self.assertRaises(KeyError):
            pool.get("missing")
        pool.metrics()
        clock.value = 5.0
        self.assertEqual(pool.collect_expired(), 0)
        clock.value = 20.01
        self.assertEqual(pool.collect_expired(), 1)

    def test_long_pin_starts_idle_interval_on_release(self):
        clock = FakeClock(0.0)
        pool = self.make_pool(clock)
        pool.admit("a", checkpoint(b"a"), durable=True)
        clock.value = 9.0
        with pool.pin("a"):
            clock.value = 100.0
            self.assertEqual(pool.collect_expired(), 0)
            self.assertIn("a", pool)
        clock.value = 110.0
        self.assertEqual(pool.collect_expired(), 0, "exactly idle after release is retained")
        clock.value = 110.01
        self.assertEqual(pool.collect_expired(), 1)

    def test_pin_clock_error_does_not_leak_pin(self):
        clock = FakeClock(0.0)
        pool = self.make_pool(clock)
        pool.admit("a", checkpoint(b"a"), durable=True)
        clock.value = True
        with self.assertRaises(ValueError):
            with pool.pin("a"):
                self.fail("pin should reject invalid clock before yielding")
        self.assertEqual(pool.metrics()["pins"], 0)

    def test_dirty_entries_are_protected_and_failed_entries_can_expire(self):
        clock = FakeClock(0.0)
        pool = self.make_pool(clock)
        pool.admit("dirty", checkpoint(b"d"), durable=False)
        pool.admit("failed", checkpoint(b"f"), durable=False)
        pool.mark_failed("failed")
        clock.value = 10.01
        self.assertEqual(pool.collect_expired(), 1)
        self.assertIn("dirty", pool)
        self.assertNotIn("failed", pool)

    def test_aliases_have_independent_expiry_and_shared_bytes_remain(self):
        clock = FakeClock(0.0)
        pool = self.make_pool(clock)
        item = checkpoint(b"shared")
        pool.admit("old", item, durable=True)
        clock.value = 5.0
        pool.admit("new", item, durable=True)
        clock.value = 9.0
        pool.get("new")
        clock.value = 10.01
        self.assertEqual(pool.collect_expired(), 1)
        self.assertNotIn("old", pool)
        self.assertIn("new", pool)
        self.assertEqual(pool.metrics()["resident_bytes"], len(item.blob) + 4 * len(item.frontier))

    def test_invalid_admission_does_not_refresh_existing_entry(self):
        clock = FakeClock(0.0)
        pool = self.make_pool(clock)
        valid = checkpoint(b"same")
        pool.admit("a", valid, durable=True)
        bad = replace(valid, checksum="0" * 64)
        clock.value = 9.0
        with self.assertRaises(ValueError):
            pool.admit("a", bad, durable=True)
        clock.value = 10.01
        self.assertEqual(pool.collect_expired(), 1)


class DiskPoolIdleExpiryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.store = Store(self.tmp.name)
        self.root = self.store.create_root("model", (1,), {"status": "idle"})
        self.s1 = self.store.append(self.root, (2,), {"status": "idle"})
        self.s2 = self.store.append(self.s1, (3,), {"status": "idle"})
        self.clock = FakeClock(0)
        self.pool = DiskCheckpointPool(self.store, 1 << 20, checkpoint_idle_seconds=10,
                                       clock_ns=self.clock)

    def add_checkpoint(self, snapshot, payload, stamp_ns):
        digest = hashlib.sha256(payload).hexdigest()
        state = self.pool.directory / (digest + ".state")
        state.write_bytes(payload)
        manifest = self.pool.directory / (snapshot + ".json")
        manifest.write_text(json.dumps({"manifest": {"payload_sha256": digest}}), encoding="utf-8")
        os.utime(manifest, ns=(stamp_ns, stamp_ns))
        return manifest, state

    def test_persisted_touch_is_visible_after_pool_restart_and_peer_use(self):
        manifest, state = self.add_checkpoint(self.s1, b"old", 1_000_000_000)
        manifest_bytes = manifest.read_bytes()
        peer = DiskCheckpointPool(self.store, 1 << 20, checkpoint_idle_seconds=10,
                                  clock_ns=self.clock)
        self.clock.value = 5_000_000_000
        peer.touch(self.s1)
        self.assertEqual(manifest.read_bytes(), manifest_bytes, "touch persists only mtime metadata")
        restarted = DiskCheckpointPool(self.store, 1 << 20, checkpoint_idle_seconds=10,
                                       clock_ns=self.clock)
        self.clock.value = 14_000_000_000
        self.assertEqual(restarted.collect_expired(), 0)
        self.clock.value = 15_000_000_001
        self.assertEqual(self.pool.collect_expired(), 1)
        self.assertFalse(manifest.exists())
        self.assertFalse(state.exists())

    def test_disk_cutoff_is_strict_and_clock_rollback_retains_future_stamp(self):
        manifest, _ = self.add_checkpoint(self.s1, b"strict", 1_000_000_000)
        self.clock.value = 0
        self.assertEqual(self.pool.collect_expired(), 0, "clock rollback is conservative")
        self.clock.value = 11_000_000_000
        self.assertEqual(self.pool.collect_expired(), 0, "exact idle duration is retained")
        self.clock.value = 11_000_000_001
        self.assertEqual(self.pool.collect_expired(), 1)
        self.assertFalse(manifest.exists())

    def test_read_lease_protects_expired_checkpoint_and_history_is_unchanged(self):
        manifest, state = self.add_checkpoint(self.s1, b"leased", 1)
        history_before = sorted((p.name, p.read_bytes()) for p in (self.store.root / "objects").glob("*.json"))
        self.clock.value = 10_000_000_002
        with self.store.read_lease(self.s1):
            self.assertEqual(self.pool.collect_expired(), 0)
            self.assertTrue(manifest.exists())
            self.assertTrue(state.exists())
        self.assertEqual(self.pool.collect_expired(), 1)
        history_after = sorted((p.name, p.read_bytes()) for p in (self.store.root / "objects").glob("*.json"))
        self.assertEqual(history_after, history_before)

    def test_expired_alias_manifest_does_not_remove_shared_live_payload(self):
        first, state = self.add_checkpoint(self.s1, b"shared", 1)
        second = self.pool.directory / (self.s2 + ".json")
        second.write_text(first.read_text(encoding="utf-8"), encoding="utf-8")
        os.utime(second, ns=(5_000_000_000, 5_000_000_000))
        self.clock.value = 10_000_000_002
        self.assertEqual(self.pool.collect_expired(), 1)
        self.assertFalse(first.exists())
        self.assertTrue(second.exists())
        self.assertTrue(state.exists())
        self.clock.value = 15_000_000_001
        self.assertEqual(self.pool.collect_expired(), 1)
        self.assertFalse(second.exists())
        self.assertFalse(state.exists())

    def test_active_payload_reservation_survives_idle_manifest_expiry(self):
        manifest, state = self.add_checkpoint(self.s1, b"reserved", 1)
        digest = hashlib.sha256(b"reserved").hexdigest()
        self.clock.value = 10_000_000_002
        with self.pool.reserve(0, payload_sha256=digest):
            self.assertEqual(self.pool.collect_expired(), 1)
            self.assertFalse(manifest.exists())
            self.assertTrue(state.exists(), "live reservation protects the payload")
        self.pool.collect_expired()
        self.assertFalse(state.exists(), "unreferenced payload becomes reclaimable after release")

    def test_missing_or_invalid_touch_does_not_create_or_refresh_metadata(self):
        self.add_checkpoint(self.s1, b"valid", 1)
        with self.assertRaises(FileNotFoundError):
            self.pool.touch(self.s2)
        before = (self.pool.directory / (self.s1 + ".json")).stat().st_mtime_ns
        self.clock.value = 10_000_000_000
        self.pool.metrics()
        self.assertEqual((self.pool.directory / (self.s1 + ".json")).stat().st_mtime_ns, before)


if __name__ == "__main__":
    unittest.main()
