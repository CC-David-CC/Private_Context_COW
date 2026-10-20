"""CPU-only tests for explicit external native-probe build directories."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock


BRANCHING = Path(__file__).resolve().parents[2]
SCRIPTS = BRANCHING / 'scripts'
sys.path.insert(0, str(SCRIPTS))
import build_digest_reuse_probe as DIGEST_BUILDER
import build_fast_prefill_probe as FAST_BUILDER
import build_native_request_probe as REQUEST_BUILDER


GENERATE_SOURCE = r'''#include "strata/core/branch_driver.hpp"
int main() {
  drive_pool_multi();
  branch_driver(&drive_pool_multi, &drive, drive.d, pcie_mode);
}
'''

DRIVER_SOURCE = r'''#include "strata/core/branch_driver.hpp"
int branch_driver(ExpertDispatch& dispatch, int pcie_mode) {
    std::string line;
    if (command == "prefill") {
            bool ok = true;
            std::cout << "{\"status\":\"ok\",\"consumed\":";
    } else if (command == "generate") {
        for (int i = 0; i < 1; ++i) {
            int32_t token = 0; int32_t unused = 0;
                if (!session.begin(position, 1, error) ||
                    !forward.run(1, &token, position, pool, user, &unused, error) || pool_failed ||
                    !forward.commit(1, error) || !forward.copy_logits(0, logits.data()) ||
                    !session.finish({token}, logits, error)) { ok = false; break; }
        }
        std::cout << ",\"forward_tokens\":" << (forward.completed_forward_tokens);
        } else if (command == "capture") {
    }
    << ",\"forward_tokens\":" << forward.completed_forward_tokens
}
'''

HEADER_SOURCE = '''namespace strata::core {
void branch_driver(ExpertDispatch&, int pcie_mode);
}
'''


class ExternalBuildDirTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.base = Path(self.temp.name)
        self.root = self.base / 'source'
        self.root.mkdir()
        self.external_build = self.base / 'build-outside-source'
        self._write_source_tree(self.root)
        self._write_build_tree(self.external_build)
        self.calls = []

    def tearDown(self):
        self.temp.cleanup()

    @staticmethod
    def _write_source_tree(root):
        files = {
            'include/strata/core/branch_driver.hpp': HEADER_SOURCE,
            'include/strata/core/branch_session.hpp': 'struct BranchSession {};\n',
            'include/strata/core/branch_pages.hpp': 'struct BranchPages {};\n',
            'include/strata/core/branch_digest.hpp': 'struct BranchDigest {};\n',
            'include/strata/core/branch_sha256.hpp': 'struct BranchSha256 {};\n',
            'src/program/generate.cpp': GENERATE_SOURCE,
            'src/core/branch_driver.cpp': DRIVER_SOURCE,
            'src/core/branch_session.cpp': 'int branch_session_fixture;\n',
            'src/core/conversation_snapshot.cpp': 'int conversation_snapshot_fixture;\n',
            'src/core/conversation_state.cpp': 'int conversation_state_fixture;\n',
            'src/core/conversation_file.cpp': 'int conversation_file_fixture;\n',
            'src/core/conversation_copy.hpp': 'struct ConversationCopy {};\n',
            'src/core/branch_digest.cu': 'int branch_digest_fixture;\n',
            'src/core/branch_pages.cpp': 'int branch_pages_fixture;\n',
            'src/core/vmm.cpp': 'int vmm_fixture;\n',
            'src/core/layer.cpp': 'int layer_fixture;\n',
            'src/core/expert_cache.cpp': 'int expert_cache_fixture;\n',
            'include/strata/core/vmm.hpp': 'struct VmmRange {};\n',
            'include/strata/core/layer.hpp': 'struct LayerFixture {};\n',
            'include/strata/core/expert_cache.hpp': 'struct ExpertCacheFixture {};\n',
        }
        for relative, content in files.items():
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content)

    @staticmethod
    def _write_build_tree(build):
        recipe = build / 'CMakeFiles' / 'strata-branch.dir'
        recipe.mkdir(parents=True)
        (build / 'CMakeCache.txt').write_text(
            'CMAKE_CXX_COMPILER:FILEPATH=/usr/bin/c++\n'
            'CMAKE_CUDA_COMPILER:FILEPATH=/usr/local/cuda/bin/nvcc\n'
            'CMAKE_BUILD_TYPE:STRING=Release\n'
            'CMAKE_CUDA_ARCHITECTURES:STRING=120\n')
        (recipe / 'flags.make').write_text(
            'CXX_DEFINES = -DTEST_BUILD=1\n'
            'CXX_INCLUDES = -I/usr/include\n'
            'CXX_FLAGS = -O2 -std=c++20\n')
        engine_recipe = build / 'CMakeFiles/strata_engine.dir'
        engine_recipe.mkdir()
        (engine_recipe / 'flags.make').write_text(
            'CXX_DEFINES = -DTEST_ENGINE_BUILD=1\n'
            'CXX_INCLUDES = -I/usr/include\n'
            'CXX_FLAGS = -O2 -std=c++20\n')
        (recipe / 'link.txt').write_text(
            '/usr/bin/c++ -o strata-branch '
            'CMakeFiles/strata-branch.dir/src/program/generate.cpp.o '
            'CMakeFiles/strata-branch.dir/src/core/branch_driver.cpp.o '
            'lib/libengine.a @link.rsp\n')
        (build / 'lib').mkdir()
        (build / 'lib' / 'libengine.a').write_bytes(b'external build archive fixture')
        (build / 'link.rsp').write_text('-pthread\n')

    def _fake_compiler(self, command, cwd=None, stdout=None, stderr=None, **_kwargs):
        command = [str(arg) for arg in command]
        resolved_cwd = Path(cwd).resolve()
        self.calls.append({'argv': command, 'cwd': str(resolved_cwd)})
        if '-o' not in command:
            raise AssertionError(f'compiler command has no output path: {command}')
        output = Path(command[command.index('-o') + 1])
        if not output.is_absolute():
            output = resolved_cwd / output
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(('mock-compiler-output\n' + '\0'.join(command)).encode())
        return subprocess.CompletedProcess(command, 0)

    def _patch_compilers(self):
        proxy = types.SimpleNamespace(run=self._fake_compiler)
        return (mock.patch.object(FAST_BUILDER, 'subprocess', proxy),
                mock.patch.object(DIGEST_BUILDER, 'subprocess', proxy))

    def test_digest_chain_uses_one_external_build_dir_for_recipes_cwd_and_provenance(self):
        output = self.base / 'probe-output'
        patches = self._patch_compilers()
        builder_state = (FAST_BUILDER.replace_once, FAST_BUILDER.FRESH,
                         FAST_BUILDER.INIT, FAST_BUILDER.FEED)
        with patches[0], patches[1]:
            DIGEST_BUILDER.main([str(self.root), str(output), '--build-dir', str(self.external_build)])
        self.assertEqual((FAST_BUILDER.replace_once, FAST_BUILDER.FRESH,
                          FAST_BUILDER.INIT, FAST_BUILDER.FEED), builder_state)

        report = json.loads((output / 'build.json').read_text())
        selected = str(self.external_build.resolve())
        self.assertEqual(report['status'], 'BUILT_NOT_EXECUTED')
        generated_driver = (output / 'branch_driver.cpp').read_text()
        self.assertIn('forward.completed_forward_tokens + prefill_tokens', generated_driver)
        self.assertIn('decode_prepare_ns', generated_driver)
        self.assertIn('decode_compute_ns', generated_driver)
        self.assertIn('decode_finish_ns', generated_driver)
        generated_fresh = (output / 'fresh_prefill.cpp').read_text()
        self.assertIn('normal_forward_tokens', generated_fresh)
        self.assertIn('decode.completed_forward_tokens', generated_fresh)
        self.assertEqual(report['build_dir'], selected)
        self.assertEqual(report['link_working_directory'], selected)
        self.assertTrue(self.calls)
        self.assertEqual({call['cwd'] for call in self.calls}, {selected})
        self.assertEqual({command['cwd'] for command in report['commands']}, {selected})
        provenance = report['build_provenance']
        self.assertEqual(provenance['cmake_toolchain']['cache_path'], str(self.external_build / 'CMakeCache.txt'))
        selected_inputs = set(provenance['declared_build_inputs'])
        self.assertIn(str(self.external_build / 'CMakeFiles/strata-branch.dir/flags.make'), selected_inputs)
        self.assertIn(str(self.external_build / 'CMakeFiles/strata-branch.dir/link.txt'), selected_inputs)
        self.assertEqual(set(provenance['linked_archives']), {str(self.external_build / 'lib/libengine.a')})
        self.assertEqual(set(provenance['link_response_files']), {str(self.external_build / 'link.rsp')})

    def test_missing_external_link_recipe_fails_before_compilation(self):
        missing = self.base / 'build-missing-link'
        self._write_build_tree(missing)
        (missing / 'CMakeFiles' / 'strata-branch.dir' / 'link.txt').unlink()
        output = self.base / 'must-not-be-created'
        patches = self._patch_compilers()
        with patches[0], patches[1]:
            with self.assertRaises(FileNotFoundError):
                DIGEST_BUILDER.main([str(self.root), str(output), '--build-dir', str(missing)])
        self.assertEqual(self.calls, [])
        self.assertFalse(output.exists())
        self.assertFalse(output.with_name(output.name + '-core').exists())

    def test_cold_restore_reader_is_rebuilt_before_both_probe_links(self):
        output = self.base / 'stream-reader-output'
        patches = self._patch_compilers()
        with patches[0], patches[1]:
            DIGEST_BUILDER.main([str(self.root), str(output), '--build-dir', str(self.external_build)])
        source = str(self.root / 'src/core/conversation_file.cpp')
        obj = str(output / 'conversation_file.o')
        compiles = [i for i, call in enumerate(self.calls) if '-c' in call['argv'] and source in call['argv']]
        self.assertEqual(len(compiles), 1, 'Streaming reader must replace the old archive member')
        links = [i for i, call in enumerate(self.calls) if '-c' not in call['argv']]
        self.assertEqual(len(links), 2)
        for index in links:
            self.assertLess(compiles[0], index)
            self.assertIn(obj, self.calls[index]['argv'])
            self.assertLess(self.calls[index]['argv'].index(obj), self.calls[index]['argv'].index('lib/libengine.a'))
        report = json.loads((output / 'build.json').read_text())
        self.assertIn(source, report['source_sha256'])
        self.assertIn(source, report['build_provenance']['declared_build_inputs'])

    def test_digest_cuda_compilation_uses_selected_cmake_compiler(self):
        cache = self.external_build / 'CMakeCache.txt'
        selected = '/opt/test-cuda-13.2/bin/nvcc'
        cache.write_text(cache.read_text().replace('/usr/local/cuda/bin/nvcc', selected))
        output = self.base / 'selected-compiler-output'
        patches = self._patch_compilers()
        with patches[0], patches[1]:
            DIGEST_BUILDER.main([str(self.root), str(output), '--build-dir', str(self.external_build)])
        cuda = [call for call in self.calls if str(self.root / 'src/core/branch_digest.cu') in call['argv']]
        self.assertEqual(len(cuda), 1)
        self.assertEqual(cuda[0]['argv'][0], selected)

    def test_vmm_layout_consumers_are_rebuilt_with_engine_flags(self):
        output = self.base / 'vmm-closure-output'
        patches = self._patch_compilers()
        with patches[0], patches[1]:
            DIGEST_BUILDER.main([str(self.root), str(output), '--build-dir', str(self.external_build)])
        links = [call['argv'] for call in self.calls if '-c' not in call['argv']]
        for name in ('vmm', 'layer', 'expert_cache'):
            source = str(self.root / 'src/core' / (name + '.cpp'))
            compiles = [call['argv'] for call in self.calls if '-c' in call['argv'] and source in call['argv']]
            self.assertEqual(len(compiles), 1, name + ' must use the new VMM layout')
            self.assertIn('-DTEST_ENGINE_BUILD=1', compiles[0])
            self.assertNotIn('-DTEST_BUILD=1', compiles[0])
            obj = str(output.with_name(output.name + '-core') / (name + '.o'))
            for link in links:
                self.assertIn(obj, link)
                self.assertLess(link.index(obj), link.index('lib/libengine.a'))
        report = json.loads((output / 'build.json').read_text())
        self.assertIn(str(self.external_build / 'CMakeFiles/strata_engine.dir/flags.make'), report['source_sha256'])

    def test_malformed_external_link_recipe_fails_before_compilation(self):
        malformed = self.base / 'build-malformed-link'
        self._write_build_tree(malformed)
        link = malformed / 'CMakeFiles' / 'strata-branch.dir' / 'link.txt'
        link.write_text('/usr/bin/c++ -o strata-branch only-one-object.o\n')
        output = self.base / 'malformed-must-not-be-created'
        patches = self._patch_compilers()
        with patches[0], patches[1]:
            with self.assertRaisesRegex(ValueError, 'link recipe must contain exactly one'):
                DIGEST_BUILDER.main([str(self.root), str(output), '--build-dir', str(malformed)])
        self.assertEqual(self.calls, [])
        self.assertFalse(output.exists())
        self.assertFalse(output.with_name(output.name + '-core').exists())

    def test_request_builder_restores_module_globals_after_build_exception(self):
        before = (FAST_BUILDER.replace_once, FAST_BUILDER.FRESH,
                  FAST_BUILDER.INIT, FAST_BUILDER.FEED)
        output = self.base / 'injected-failure-output'
        with mock.patch.object(FAST_BUILDER, 'build', side_effect=RuntimeError('injected build failure')):
            with self.assertRaisesRegex(RuntimeError, 'injected build failure'):
                REQUEST_BUILDER.build(self.root, output, build_dir=self.external_build)
        self.assertEqual((FAST_BUILDER.replace_once, FAST_BUILDER.FRESH,
                          FAST_BUILDER.INIT, FAST_BUILDER.FEED), before)

    def test_native_and_fast_prefill_clis_accept_explicit_external_build_dir(self):
        selected = str(self.external_build.resolve())
        patches = self._patch_compilers()
        with patches[0], patches[1]:
            fast_output = self.base / 'fast-output'
            FAST_BUILDER.main([str(self.root), str(fast_output), '--build-dir', str(self.external_build)])
            fast_report = json.loads((fast_output / 'build.json').read_text())
            self.assertEqual(fast_report['build_dir'], selected)
            self.assertEqual(fast_report['link_working_directory'], selected)

            request_output = self.base / 'request-output'
            REQUEST_BUILDER.main([str(self.root), str(request_output), '--build-dir', str(self.external_build)])
            request_report = json.loads((request_output / 'build.json').read_text())
            self.assertEqual(request_report['build_dir'], selected)
            self.assertEqual(request_report['link_working_directory'], selected)
            self.assertEqual(request_report['build_provenance']['cmake_toolchain']['cache_path'],
                             str(self.external_build / 'CMakeCache.txt'))

    def test_omitted_build_dir_keeps_root_build_default(self):
        default_build = self.root / 'build'
        self._write_build_tree(default_build)
        output = self.base / 'default-output'
        patches = self._patch_compilers()
        with patches[0], patches[1]:
            FAST_BUILDER.main([str(self.root), str(output)])
        report = json.loads((output / 'build.json').read_text())
        self.assertEqual(report['build_dir'], str(default_build.resolve()))
        self.assertEqual(report['link_working_directory'], str(default_build.resolve()))


if __name__ == '__main__':
    unittest.main()
