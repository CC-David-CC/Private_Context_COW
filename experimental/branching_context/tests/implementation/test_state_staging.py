"""Filesystem selection tests; GPU staging measurements remain separate."""
import tempfile
from pathlib import Path
import unittest
from context_time_travel.strata_backend import _tmpfs_mount


class StateStagingTests(unittest.TestCase):
    def test_only_exact_tmpfs_mount_is_accepted(self):
        with tempfile.TemporaryDirectory() as root:
            target = Path(root)
            self.assertTrue(_tmpfs_mount('tmpfs '+root+' tmpfs rw 0 0\n',target))
            self.assertFalse(_tmpfs_mount('/dev/disk '+root+' ext4 rw 0 0\n',target))
            self.assertFalse(_tmpfs_mount('tmpfs '+root+'/other tmpfs rw 0 0\n',target))
            self.assertFalse(_tmpfs_mount('garbage\n',target))

    def test_mount_escape_and_missing_directory(self):
        with tempfile.TemporaryDirectory(prefix='with space ') as root:
            text = root.replace(' ','\\040')
            self.assertTrue(_tmpfs_mount('tmpfs '+text+' tmpfs rw 0 0\n',Path(root)))
            self.assertFalse(_tmpfs_mount('tmpfs '+text+' tmpfs rw 0 0\n',Path(root)/'missing'))


if __name__ == '__main__': unittest.main()
