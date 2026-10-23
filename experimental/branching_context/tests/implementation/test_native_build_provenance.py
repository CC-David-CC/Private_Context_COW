import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[2] / 'scripts' / 'native_build_provenance.py'
SPEC = importlib.util.spec_from_file_location('native_build_provenance', SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)
SCRIPTS = SCRIPT.parent
sys.path.insert(0, str(SCRIPTS))
import build_native_request_probe as REQUEST_BUILDER


class NativeBuildProvenanceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name) / 'source'
        self.root.mkdir()
        self.build = self.root / 'build'
        self.build.mkdir()
        (self.build / 'CMakeCache.txt').write_text(
            'CMAKE_CXX_COMPILER:FILEPATH=/usr/bin/c++\n'
            'CMAKE_CUDA_COMPILER:FILEPATH=/usr/local/cuda/bin/nvcc\n'
            'CMAKE_BUILD_TYPE:STRING=Release\n')
        recipe = self.build / 'CMakeFiles' / 'strata-branch.dir'
        recipe.mkdir(parents=True)
        (recipe / 'flags.make').write_text(
            'CXX_DEFINES = -DSTRATA_BRANCH_DRIVER=1\nCXX_INCLUDES = -Iinclude\nCXX_FLAGS = -O3\n')
        (recipe / 'link.txt').write_text('/usr/bin/c++ '
            'CMakeFiles/strata-branch.dir/src/program/generate.cpp.o '
            'CMakeFiles/strata-branch.dir/src/core/branch_driver.cpp.o '
            '-o strata-branch lib/libengine.a\n')
        self.script = self.root / 'build_native_request_probe.py'
        self.script.write_text('from helper_one import build\n')
        (self.root / 'helper_one.py').write_text('from helper_two import value\n')
        (self.root / 'helper_two.py').write_text('value = 1\n')
        libdir = self.build / 'lib'
        libdir.mkdir()
        (libdir / 'libengine.a').write_bytes(b'engine archive')
        (self.build / 'args.rsp').write_text('lib/libengine.a')
        self.commands = [{
            'argv': ['/usr/bin/c++', '-o', str(self.build / 'probe'), '@args.rsp'],
            'exit_code': 0,
        }]

    def tearDown(self):
        self.temp.cleanup()

    def collect(self):
        return MODULE.collect_build_provenance(
            source_root=self.root,
            build_dir=self.build,
            commands=self.commands,
            generator_entrypoints=[self.script],
            declared_sources=[self.script],
        )

    def test_git_source_and_observed_inputs_are_pinned_but_not_hermetic(self):
        subprocess.run(['git', 'init', '-q'], cwd=self.root, check=True)
        subprocess.run(['git', 'config', 'user.email', 'test@example.invalid'], cwd=self.root, check=True)
        subprocess.run(['git', 'config', 'user.name', 'Test'], cwd=self.root, check=True)
        subprocess.run(['git', 'add', '.'], cwd=self.root, check=True)
        subprocess.run(['git', 'commit', '-qm', 'fixture'], cwd=self.root, check=True)
        first = self.collect()
        self.assertEqual(first['source']['status'], 'clean')
        self.assertFalse(first['reproducible'])
        self.assertEqual(first['reproducibility_status'], 'unproven_external_toolchain_closure')
        self.assertTrue(first['observed_inputs_complete'])
        self.assertFalse(first['provenance_complete'])
        self.assertEqual(first['generator_closure']['files'][str(self.root / 'helper_two.py')],
                         MODULE.sha256_file(self.root / 'helper_two.py'))
        self.assertEqual(first['linked_archives'][str(self.build / 'lib/libengine.a')],
                         MODULE.sha256_file(self.build / 'lib/libengine.a'))
        self.assertEqual(first['link_response_files'][str(self.build / 'args.rsp')],
                         MODULE.sha256_file(self.build / 'args.rsp'))
        self.assertEqual(first['cmake_toolchain']['cache_sha256'],
                         MODULE.sha256_file(self.build / 'CMakeCache.txt'))

    def test_dirty_and_non_git_are_explicitly_not_reproducible(self):
        subprocess.run(['git', 'init', '-q'], cwd=self.root, check=True)
        subprocess.run(['git', 'config', 'user.email', 'test@example.invalid'], cwd=self.root, check=True)
        subprocess.run(['git', 'config', 'user.name', 'Test'], cwd=self.root, check=True)
        subprocess.run(['git', 'add', '.'], cwd=self.root, check=True)
        subprocess.run(['git', 'commit', '-qm', 'fixture'], cwd=self.root, check=True)
        (self.root / 'helper_two.py').write_text('value = 2\n')
        (self.root / 'untracked.txt').write_text('untracked')
        dirty = self.collect()
        self.assertEqual(dirty['source']['status'], 'dirty')
        self.assertFalse(dirty['reproducible'])
        self.assertEqual(dirty['source']['tracked_changes'], ['helper_two.py'])
        self.assertEqual(dirty['source']['untracked'], ['untracked.txt'])
        shutil = __import__('shutil')
        shutil.rmtree(self.root / '.git')
        nongit = self.collect()
        self.assertEqual(nongit['source']['status'], 'non_git')
        self.assertFalse(nongit['reproducible'])

    def test_imported_generator_and_linked_archive_changes_change_hashes(self):
        subprocess.run(['git', 'init', '-q'], cwd=self.root, check=True)
        subprocess.run(['git', 'config', 'user.email', 'test@example.invalid'], cwd=self.root, check=True)
        subprocess.run(['git', 'config', 'user.name', 'Test'], cwd=self.root, check=True)
        subprocess.run(['git', 'add', '.'], cwd=self.root, check=True)
        subprocess.run(['git', 'commit', '-qm', 'fixture'], cwd=self.root, check=True)
        before = self.collect()
        (self.root / 'helper_two.py').write_text('value = 2\n')
        (self.build / 'lib/libengine.a').write_bytes(b'changed archive')
        after = self.collect()
        self.assertNotEqual(before['generator_closure']['files'], after['generator_closure']['files'])
        self.assertNotEqual(before['linked_archives'], after['linked_archives'])
        self.assertFalse(after['reproducible'])

    def test_relative_link_paths_and_response_files_resolve_from_link_cwd(self):
        self.commands = [{'argv': ['/usr/bin/c++', '-o', 'probe', '@args.rsp'], 'exit_code': 0}]
        result = self.collect()
        self.assertEqual(list(result['linked_archives']), [str(self.build / 'lib/libengine.a')])

    def test_attach_failure_cannot_leave_successful_receipt(self):
        output = self.root / 'out'
        output.mkdir()
        receipt = {
            'status': 'BUILT_NOT_EXECUTED',
            'commands': [{'argv': ['/usr/bin/c++', '-o', 'probe',
                                   str(self.build / 'lib/missing.a')], 'exit_code': 0}],
            'source_sha256': {str(self.script): MODULE.sha256_file(self.script)},
            'generated_sha256': {},
        }
        (output / 'build.json').write_text(json.dumps(receipt))
        with self.assertRaises(FileNotFoundError):
            REQUEST_BUILDER.attach_provenance(self.root, output, receipt, [self.script])
        written = json.loads((output / 'build.json').read_text())
        self.assertEqual(written['status'], 'BUILD_PROVENANCE_FAILED')
        self.assertIn('provenance_error', written)

    def test_missing_response_or_declared_archive_fails_closed(self):
        (self.build / 'args.rsp').unlink()
        with self.assertRaises((FileNotFoundError, ValueError)):
            self.collect()
        (self.build / 'args.rsp').write_text(str(self.build / 'lib/missing.a'))
        with self.assertRaises((FileNotFoundError, ValueError)):
            self.collect()

    def test_nonzero_or_malformed_command_record_fails_closed(self):
        self.commands = [{'argv': ['/usr/bin/c++', '-o', 'probe', '@args.rsp'], 'exit_code': 1}]
        with self.assertRaisesRegex(ValueError, 'did not succeed'):
            self.collect()
        self.commands = [['/usr/bin/c++', '-o', 'probe', '@args.rsp']]
        with self.assertRaisesRegex(ValueError, 'receipt object'):
            self.collect()


if __name__ == '__main__':
    unittest.main()
