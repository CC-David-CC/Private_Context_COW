"""Measurement protocol tests; these do not qualify native inference."""
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts'))
from client_stream_speed import RequestTimeline, counter_delta


class ClientStreamMetricsTests(unittest.TestCase):
    def test_counter_delta_excludes_error_labels_without_hiding_native_work(self):
        self.assertEqual(counter_delta(dict(forward_tokens=2,live_bytes=100,vmm_last_failed_operation=''),
                                       dict(forward_tokens=5,live_bytes=80,vmm_last_failed_operation='')),
                         dict(forward_tokens=3,live_bytes=-20))
        with self.assertRaises(ValueError): counter_delta(dict(forward_tokens=2),dict(forward_tokens='2'))
    def test_received_tokens_and_durable_ack_are_separate(self):
        row = RequestTimeline()
        row.accept(dict(event='request_started'),10.)
        row.accept(dict(event='token',index=0,token=3),12.)
        row.accept(dict(event='token',index=1,token=4),12.5)
        row.accept(dict(event='snapshot',tokens=[3,4]),13.)
        self.assertFalse(row.summary()['durable'])
        row.accept(dict(event='durable'),15.)
        result = row.summary()
        self.assertEqual(result['outputs'],2)
        self.assertEqual(result['time_to_first_token_seconds'],2.)
        self.assertEqual(result['steady_received_tps'],2.)
        self.assertAlmostEqual(result['request_tps'],2/3)
        self.assertAlmostEqual(result['durable_request_tps'],2/5)

    def test_out_of_order_duplicate_and_premature_publication_fail(self):
        for events in ([dict(event='token',index=0,token=1)],
                       [dict(event='request_started'),dict(event='durable')],
                       [dict(event='request_started'),dict(event='token',index=1,token=1)],
                       [dict(event='request_started'),dict(event='token',index=0,token=1),
                        dict(event='token',index=0,token=1)],
                       [dict(event='request_started'),dict(event='token',index=0,token=1),
                        dict(event='snapshot',tokens=[2])]):
            with self.subTest(events=events):
                row = RequestTimeline()
                with self.assertRaises(ValueError):
                    for event in events: row.accept(event,1.)

    def test_one_output_has_no_fabricated_steady_rate_and_clock_must_increase(self):
        row = RequestTimeline()
        row.accept(dict(event='request_started'),1.)
        row.accept(dict(event='token',index=0,token=1),2.)
        row.accept(dict(event='snapshot',tokens=[1]),3.)
        self.assertIsNone(row.summary()['steady_received_tps'])
        with self.assertRaises(ValueError): row.accept(dict(event='durable'),2.)

    def test_token_gap_distribution_excludes_first_token_wait(self):
        row = RequestTimeline()
        row.accept(dict(event='request_started'),0.)
        for i,now in enumerate((10.,10.01,10.03,12.03)):
            row.accept(dict(event='token',index=i,token=i),now)
        row.accept(dict(event='snapshot',tokens=list(range(4))),13.)
        result = row.summary()
        self.assertEqual(result['token_gap_count'],3)
        self.assertAlmostEqual(result['mean_tpot_seconds'],2.03/3)
        self.assertAlmostEqual(result['median_tpot_seconds'],0.02)
        self.assertAlmostEqual(result['p95_tpot_seconds'],2.)
        self.assertAlmostEqual(result['max_tpot_seconds'],2.)
        self.assertEqual(result['time_to_first_token_seconds'],10.)


if __name__ == '__main__': unittest.main()
