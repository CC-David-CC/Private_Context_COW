"""Compiled native input digest versus independently specified byte encoding."""
import hashlib
from pathlib import Path
import shutil
import struct
import subprocess
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[4]


class VisionNativeBindingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        compiler=shutil.which('g++') or shutil.which('c++')
        if compiler is None:raise RuntimeError('Native input binding needs a real C++ compiler')
        cls.tmp=tempfile.TemporaryDirectory(); cls.addClassCleanup(cls.tmp.cleanup)
        cls.exe=Path(cls.tmp.name)/'binding'
        subprocess.run([compiler,'-std=c++20','-O2','-Wall','-Wextra','-Werror',
            '-I',str(ROOT/'include'),str(ROOT/'experimental/branching_context/native/vision_layout_probe.cpp'),
            '-o',str(cls.exe)],check=True,capture_output=True,timeout=60)

    def digest(self,ids,images):
        request=[32,128,4,1024,len(ids),len(images),*ids]
        raw=b'COW-VISION-INPUT-v1\0'+struct.pack('<Q',len(ids))
        raw+=struct.pack('<'+'i'*len(ids),*ids)+struct.pack('<Q',len(images))
        for dimensions,sha in images:
            request.extend((*dimensions,sha))
            raw+=struct.pack('<6q',*dimensions)+sha.encode()
        actual=subprocess.run([str(self.exe),'--input-digest'],input=' '.join(map(str,request))+'\n',
                              text=True,capture_output=True,check=True,timeout=10).stdout.strip()
        self.assertEqual(actual,'OK '+hashlib.sha256(raw).hexdigest())
        return actual

    def test_empty_text_and_literal_image_pad_ids(self):
        for ids in ([],[1,2,3],[7]*9):self.digest(ids,[])

    def test_full_embedding_digest_distinguishes_equal_pad_sequences(self):
        ids=[1,2]+[7]*6+[3]; dimensions=(2,6,3,2,4,7)
        self.assertNotEqual(self.digest(ids,[(dimensions,'a'*64)]),
                            self.digest(ids,[(dimensions,'b'*64)]))

    def test_same_payload_different_placement_and_grid(self):
        ids=[7]*9
        a=self.digest(ids,[((1,6,3,2,4,7),'a'*64)])
        b=self.digest(ids,[((2,6,3,2,4,7),'a'*64)])
        c=self.digest(ids,[((1,6,2,3,4,7),'a'*64)])
        self.assertEqual(len({a,b,c}),3)

    def test_two_images_preserve_order_and_text_identity(self):
        images=[((1,2,2,1,4,7),'a'*64),((4,2,1,2,4,7),'b'*64)]
        a=self.digest([1,7,7,2,7,7,3],images)
        b=self.digest([1,7,7,4,7,7,3],images)
        self.assertNotEqual(a,b)


if __name__=='__main__':unittest.main()
