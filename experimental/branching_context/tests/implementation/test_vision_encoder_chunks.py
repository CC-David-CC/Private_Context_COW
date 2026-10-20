"""Actual C++ chunk selector; synthetic chunk kinds, not image/GPU evidence."""
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[4]


class VisionEncoderChunkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory(prefix='cow-image-chunks-');cls.addClassCleanup(cls.temp.cleanup)
        cls.exe=Path(cls.temp.name)/'probe'
        compiler=shutil.which('g++')
        if compiler is None:raise RuntimeError('Actual native compiler is required')
        run=subprocess.run([compiler,'-std=c++17','-Wall','-Wextra','-Werror','-I',str(ROOT),
            str(ROOT/'experimental/branching_context/native/vision_encoder_chunk_probe.cpp'),
            '-o',str(cls.exe)],capture_output=True,text=True,timeout=60)
        if run.returncode:raise AssertionError(run.stderr)

    def select(self,*kinds):
        run=subprocess.run([str(self.exe),*kinds],capture_output=True,text=True,timeout=5)
        self.assertEqual(run.returncode,0,run.stderr)
        return run.stdout.strip()

    def test_single_image_with_text_chunks_selects_exact_index(self):
        self.assertEqual(self.select('image'),'OK 0')
        self.assertEqual(self.select('text','image','text'),'OK 1')

    def test_multiple_images_are_rejected_instead_of_retaining_last(self):
        for kinds in (('image','image'),('image','text','image'),('text','image','image','text')):
            with self.subTest(kinds=kinds):self.assertIn('multiple image chunks',self.select(*kinds))

    def test_missing_image_rejects(self):
        self.assertIn('no image chunk',self.select())
        self.assertIn('no image chunk',self.select('text','text'))


if __name__=='__main__':unittest.main()
