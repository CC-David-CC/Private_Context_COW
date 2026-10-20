"""Tier labels require observed sources and work, not latency guesses."""
import unittest
from run_vision_latency import validate_tier, summarize_repetitions, LatencyError

class VisionLatencyTests(unittest.TestCase):
    def request(self):
        return dict(mode='resident',prefix_cells=370,suffix_cells=14,outputs=32,forward_tokens=46,
                    source_native=True,source_ram=True,source_disk=True,loaded_file_bytes=0,
                    disk_preload_bytes=0,encoder_calls=0,image_cached=True)
    def test_resident_requires_hot_source_and_no_cold_read(self):
        validate_tier(self.request())
        for key,value in [('source_native',False),('loaded_file_bytes',1),('forward_tokens',416)]:
            with self.subTest(key=key),self.assertRaises(LatencyError):validate_tier(dict(self.request(),**{key:value}))
    def test_ram_and_disk_are_distinguished_by_actual_pool_read(self):
        row=dict(self.request(),mode='ram',source_native=False,loaded_file_bytes=100)
        validate_tier(row)
        with self.assertRaises(LatencyError):validate_tier(dict(row,disk_preload_bytes=100))
        disk=dict(row,mode='disk',source_ram=False,disk_preload_bytes=100)
        validate_tier(disk)
        with self.assertRaises(LatencyError):validate_tier(dict(disk,disk_preload_bytes=0))
    def test_embedding_reconstruction_and_cold_encoding_require_prefix_work(self):
        row=dict(self.request(),mode='embedding',suffix_cells=0,outputs=0,forward_tokens=370,
                 source_native=False,source_ram=False,source_disk=False)
        validate_tier(row)
        with self.assertRaises(LatencyError):validate_tier(dict(row,forward_tokens=0))
        validate_tier(dict(row,mode='cold',image_cached=False,encoder_calls=1))
        with self.assertRaises(LatencyError):validate_tier(dict(row,mode='cold'))
    def test_warmup_is_excluded_and_three_measurements_required(self):
        rows=[dict(warmup=True,seconds=1000),dict(warmup=False,seconds=3),dict(warmup=False,seconds=1),dict(warmup=False,seconds=2)]
        result=summarize_repetitions(rows)
        self.assertEqual(result['individual_seconds'],[3,1,2]);self.assertEqual(result['median_seconds'],2)
        with self.assertRaises(LatencyError):summarize_repetitions(rows[:-1])

    def test_encoder_calls_come_from_actual_encoding_receipts(self):
        from run_vision_latency import encoding_calls
        self.assertEqual(encoding_calls({'encodings':[{'rows':300,'seconds':0.1}]}),1)
        self.assertEqual(encoding_calls({'encodings':[]}),0)
        with self.assertRaises(LatencyError):encoding_calls({'encoder_calls':1})

    def test_generated_cleanup_matches_only_derived_file_names(self):
        import ast,re
        from pathlib import Path
        from run_vision_latency import install_tier_observer
        body=install_tier_observer("importlib.import_module('serve.cow_http').main()\n",Path('/tmp/cow-fixture'))
        literals=[node.value for node in ast.walk(ast.parse(body)) if isinstance(node,ast.Constant) and isinstance(node.value,str)]
        patterns=[value for value in literals if value.startswith('[0-9a-f]{64}')]
        self.assertTrue(patterns)
        for suffix in ('.json','.state','.sve'):
            self.assertTrue(any(re.fullmatch(pattern,'a'*64+suffix) for pattern in patterns))
        for name in ('objects.json','notes.md','.stage-123','a'*64+'.input'):
            self.assertFalse(any(re.fullmatch(pattern,name) for pattern in patterns))

    def test_reference_text_uses_plain_feed_and_images_use_image_ingest(self):
        from run_vision_latency import reference_prefix_command
        self.assertEqual(reference_prefix_command((1,2),'/a.manifest','/a.i32',0),'feed 1 2')
        self.assertEqual(reference_prefix_command((1,2),'/a.manifest','/a.i32',1),'feed_image "/a.manifest" "/a.i32"')
        for images in (-1,True,9):
            with self.subTest(images=images),self.assertRaises(LatencyError):
                reference_prefix_command((1,2),'/a','/b',images)

    def test_reference_cleanup_records_dead_process_instead_of_masking_primary_error(self):
        from run_vision_latency import close_reference
        import subprocess,sys
        from types import SimpleNamespace
        process=subprocess.Popen([sys.executable,'-c','raise SystemExit(7)']);process.wait(timeout=10)
        def broken():raise BrokenPipeError('synthetic quit race after native rejection')
        value=close_reference(SimpleNamespace(proc=process,close=broken))
        self.assertEqual(value['exit_code'],7);self.assertTrue(value['process_absent'])
        self.assertIn('BrokenPipeError',value['shutdown_error'])

    def test_reference_cleanup_cannot_claim_retirement_of_a_live_process(self):
        from run_vision_latency import close_reference
        from types import SimpleNamespace
        process=SimpleNamespace(poll=lambda:None,pid=1)
        def broken():raise BrokenPipeError('still live')
        with self.assertRaises(BrokenPipeError):close_reference(SimpleNamespace(proc=process,close=broken))

if __name__=='__main__':unittest.main()
