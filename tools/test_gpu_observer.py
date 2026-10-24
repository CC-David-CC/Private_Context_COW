"""Read-only telemetry must not report missing or malformed data as free GPUs."""
import json
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch
import subprocess
from tools.pcc_gpu_observer import parse_sample, sample, observe, QUERY


class GPUObserver(unittest.TestCase):
    def test_independent_capacity_and_activity_per_uuid(self):
        rows = parse_sample('2, GPU-b, 81920, 1024, 80896, 99, 30\n0, GPU-a, 81920, 80000, 1920, 0, 0\n')
        self.assertEqual([r['index'] for r in rows], [0, 2])
        self.assertEqual(rows[1]['gpu_percent'], 99)
        self.assertEqual(rows[0]['free_mib'], 1920)
        self.assertFalse(any('eligible' in row for row in rows))

    def test_missing_invalid_or_duplicate_data_is_unavailable(self):
        for raw in ['', '0, GPU-a, 81920, 0, N/A, 0, 0',
                    '0, GPU-a, 81920, 0, 81920, 101, 0',
                    '0, GPU-a, 81920, 0, 81920, 0',
                    '0, GPU-a, 81920, 0, 81920, 0, 0\n1, GPU-a, 81920, 0, 81920, 0, 0']:
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                parse_sample(raw)

    def test_driver_failure_is_recorded_without_claiming_capacity(self):
        with patch('tools.pcc_gpu_observer.subprocess.run', side_effect=subprocess.TimeoutExpired(QUERY, 15)):
            value = sample()
        self.assertEqual(value['status'], 'unavailable')
        self.assertNotIn('gpus', value)

    def test_single_query_has_no_mutating_arguments(self):
        result = subprocess.CompletedProcess(QUERY, 0, '0, GPU-a, 81920, 0, 81920, 0, 0\n', '')
        with patch('tools.pcc_gpu_observer.subprocess.run', return_value=result) as run:
            value = sample()
        self.assertEqual(value['status'], 'observed')
        self.assertEqual(run.call_args.args[0], QUERY)
        self.assertEqual(len(QUERY), 3)

    def test_existing_observation_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp, self.assertRaises(FileExistsError):
            observe(Path(tmp), duration_seconds=1)

    def test_unbounded_monitoring_is_refused(self):
        for interval, duration in [(0, 10), (30, 0), (30, 259201)]:
            with self.assertRaises(ValueError):
                observe('/unused', interval_seconds=interval, duration_seconds=duration)

    def test_bounded_run_preserves_unavailable_observation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)/'new-run'
            with patch('tools.pcc_gpu_observer.sample', return_value={'status': 'unavailable', 'error_type': 'TimeoutExpired'}):
                result = observe(root, interval_seconds=1, duration_seconds=1)
            self.assertEqual(result['state'], 'complete')
            self.assertEqual(result['samples'], 1)
            self.assertEqual(result['unavailable'], 1)
            self.assertFalse(json.loads((root/'identity.json').read_text())['allocates_gpus'])
            self.assertEqual(len((root/'samples-private.jsonl').read_text().splitlines()), 1)
