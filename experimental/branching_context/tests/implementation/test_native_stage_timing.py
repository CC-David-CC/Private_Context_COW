"""Generated-driver instrumentation checks; actual timings require GPU runs."""
from pathlib import Path
import os
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts'))
from build_native_request_probe import instrument_request_timing


class NativeStageTimingTests(unittest.TestCase):
    def source(self):
        path = Path(os.environ.get('CTT_NATIVE_DRIVER_SOURCE',
            str(Path(__file__).resolve().parents[4]/'src/core/branch_driver.cpp')))
        return path.read_text()

    def test_only_generation_is_instrumented_without_moving_the_consumed_event(self):
        source = self.source()
        actual = instrument_request_timing(source)
        marker = '} else if (command == "generate")'
        self.assertEqual(actual.split(marker)[0],source.split(marker)[0])
        for key in ('decode_prepare_ns','decode_compute_ns','decode_finish_ns','decode_timed_tokens'):
            self.assertIn(key,actual.split(marker)[1].split('} else if (command == "capture")')[0])
        self.assertLess(actual.index('const bool finished = session.finish'),actual.index('++decode_timed_tokens'))
        self.assertLess(actual.index('++decode_timed_tokens'),actual.index('completed_positions.push_back(position)'))
        self.assertIn('forward.run(1, &token, position, pool, user, &unused, error)',actual)

    def test_missing_or_duplicate_generation_handler_and_changed_operator_are_rejected(self):
        source = self.source()
        for broken in (source.replace('command == "generate"','command == "unknown"'),
                       source+source,source.replace('forward.run(1, &token','forward.run(2, &token')):
            with self.subTest(),self.assertRaises(ValueError): instrument_request_timing(broken)


if __name__ == '__main__': unittest.main()
