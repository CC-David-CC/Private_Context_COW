"""Exercise the shipped client archive's documentation, including moved README links."""
import json
from pathlib import Path, PurePosixPath
import posixpath
import re
import subprocess
import sys
import tempfile
import unittest
from urllib.parse import unquote, urlsplit
import zipfile


class PrivateClientPackageTests(unittest.TestCase):
    def test_every_packaged_markdown_link_resolves_after_readme_relocation(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)/'client.zip'
            completed = subprocess.run([sys.executable,str(root/'tools/package_cow_cli.py'),
                '--output',str(output)],capture_output=True,text=True)
            self.assertEqual(completed.returncode,0,completed.stderr)
            json.loads(completed.stdout)
            with zipfile.ZipFile(output) as archive:
                self.assertIsNone(archive.testzip())
                names = set(archive.namelist())
                self.assertIn('context-cow-cli/README.md',names)
                readme = archive.read('context-cow-cli/README.md').decode('utf-8')
                self.assertIn('(docs/context-cow/CLI.md#recover-an-uncertain-operation)',readme)
                status = archive.read('context-cow-cli/experimental/branching_context/evidence/STATUS.md').decode('utf-8')
                self.assertIn('Client package note:',status)
                self.assertIn('full source checkout:',status)
                self.assertFalse(any(name.startswith('context-cow-cli/src/') for name in names))
                for name in sorted(names):
                    if not name.endswith('.md'):
                        continue
                    text = archive.read(name).decode('utf-8')
                    for link in re.findall(r'\]\(([^)]+)\)',text):
                        parsed = urlsplit(link)
                        if parsed.scheme or parsed.netloc or not parsed.path:
                            continue
                        target = posixpath.normpath(str(PurePosixPath(name).parent/unquote(parsed.path)))
                        with self.subTest(document=name,link=link):
                            self.assertIn(target,names,'Local guide link is absent from the client ZIP')
                            self.assertTrue(target.startswith('context-cow-cli/'))


if __name__ == '__main__':
    unittest.main()
