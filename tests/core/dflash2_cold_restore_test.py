"""Real Flash/CUDA regression: mixed transport must admit text-only cold state.

Requires STRATA_BRANCH_COMMAND (JSON argv with --vision), STRATA_DFLASH2_PACK,
STRATA_DFLASH2_VOCAB, and STRATA_TEST_EMBEDDING_WIDTH for the selected model.
This test starts its own native process and must run on an idle GPU.
"""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest


class DFlash2ColdRestoreTest(unittest.TestCase):
    def test_text_only_cold_restore_and_image_rejection(self):
        command = json.loads(os.environ['STRATA_BRANCH_COMMAND'])
        width = int(os.environ['STRATA_TEST_EMBEDDING_WIDTH'])
        self.assertIn('--vision', command)
        self.assertTrue(os.environ.get('STRATA_DFLASH2_PACK'))
        self.assertTrue(os.environ.get('STRATA_DFLASH2_VOCAB'))
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            with (directory/'native.stderr').open('w') as log:
                process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                           stderr=log, text=True, bufsize=1)
                def read():
                    line = process.stdout.readline()
                    self.assertTrue(line, 'native process ended before response')
                    return json.loads(line)
                def request(command):
                    process.stdin.write(command+'\n')
                    process.stdin.flush()
                    return read()
                def quoted(path):
                    return json.dumps(str(path))
                try:
                    ready = read()
                    self.assertEqual(ready.get('dflash2_protocol'), 'verified-window-v1')
                    self.assertEqual(request('feed 1 2 3')['consumed'], 3)
                    state, metadata, frontier = (directory/name for name in ('state', 'inputs', 'frontier'))
                    self.assertEqual(request('logits '+str(frontier))['status'], 'ok')
                    expected = frontier.read_bytes()
                    self.assertEqual(request('write '+quoted(state))['status'], 'ok')
                    before = request('stats')['forward_tokens']
                    self.assertEqual(request('reset')['consumed'], 0)
                    metadata.write_text('COW_VISION_RESTORE_V1 3 0\n1 2 3\n')
                    load = 'load_image '+ ' '.join(map(quoted, (state, metadata, frontier)))
                    self.assertEqual(request(load)['consumed'], 3)
                    self.assertEqual(request('stats')['forward_tokens'], before)
                    self.assertEqual(request('logits '+str(frontier))['status'], 'ok')
                    self.assertEqual(frontier.read_bytes(), expected)
                    self.assertEqual(request('decode 4')['consumed'], 4)
                    self.assertEqual(request('decode 5')['consumed'], 5)
                    self.assertGreater(request('stats')['dflash2_offered'], 0)

                    # A valid mixed restore manifest with an actual placement
                    # must fail before touching the current text-only session.
                    metadata.write_text('COW_VISION_RESTORE_V1 4 1\n1 2 3 4\n'
                                        f'1 1 1 1 {width} 2 '+ '0'*64 + '\n')
                    denied = request(load)
                    self.assertEqual(denied['status'], 'error')
                    self.assertIn('does not support image histories', denied['message'])
                    self.assertEqual(request('stats')['consumed'], 5)
                    denied = request('feed_image '+quoted(directory/'missing'))
                    self.assertEqual(denied['status'], 'error')
                    self.assertIn('does not support image histories', denied['message'])
                    self.assertEqual(request('quit')['status'], 'closed')
                    self.assertEqual(process.wait(timeout=30), 0)
                finally:
                    if process.poll() is None:
                        process.kill()
                        process.wait(timeout=30)
                    process.stdin.close()
                    process.stdout.close()


if __name__ == '__main__':
    unittest.main()
