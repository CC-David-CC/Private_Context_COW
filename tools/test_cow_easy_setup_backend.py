"""CPU diagnostics only; these do not qualify AMD COW inference."""
import sys
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import cow_easy_setup as setup

class BackendDiagnosticTests(unittest.TestCase):
    def test_amd_display_vendor_is_detected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); p=root/'card1/device';p.mkdir(parents=True)
            (p/'vendor').write_text('0x1002\n')
            self.assertTrue(setup.amd_display_present(root))
    def test_other_vendor_is_not_called_amd(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); p=root/'card0/device';p.mkdir(parents=True)
            (p/'vendor').write_text('0x10de\n')
            self.assertFalse(setup.amd_display_present(root))
    def test_absent_display_tree_is_not_called_amd(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertFalse(setup.amd_display_present(Path(tmp)/'missing'))
    def test_amd_only_error_names_missing_cow_backend(self):
        with patch.object(setup.shutil,'which',side_effect=lambda n: None if n=='nvidia-smi' else '/usr/bin/'+n),patch.object(setup,'amd_display_present',return_value=True),patch.object(setup.subprocess,'check_output',side_effect=AssertionError('must stop before native commands')):
            with self.assertRaisesRegex(setup.SetupError,'snapshot-session'):
                setup.inspect_host()
    def test_unsupported_host_creates_no_data_and_does_not_prompt(self):
        with tempfile.TemporaryDirectory() as tmp:
            data=Path(tmp)/'not-created'
            with patch.object(setup.shutil,'which',side_effect=lambda n: None if n=='nvidia-smi' else '/usr/bin/'+n),patch.object(setup,'amd_display_present',return_value=True),patch('builtins.input',side_effect=AssertionError('must not prompt')),patch.object(setup,'run_step',side_effect=AssertionError('must not install')):
                with self.assertRaisesRegex(setup.SetupError,'AMD'):
                    setup.main(['--yes','--data-dir',str(data)])
            self.assertFalse(data.exists())
    def test_nvidia_path_is_preserved_on_mixed_vendor_host(self):
        def output(command,**kwargs):
            if command[0]=='cmake':return 'cmake version 4.2.3\n'
            if '--query-compute-apps=pid' in command:return ''
            return 'Example NVIDIA GPU, 24576, 24000\n'
        with patch.object(setup.shutil,'which',side_effect=lambda n:'/usr/bin/'+n),patch.object(setup,'amd_display_present',side_effect=AssertionError('NVIDIA remains selected')),patch.object(setup.subprocess,'check_output',side_effect=output),patch.object(setup.subprocess,'run'):
            hardware=setup.inspect_host()
        self.assertEqual(hardware.gpu_name,'Example NVIDIA GPU')
        self.assertEqual(hardware.gpu_free,24000*1024**2)

if __name__=='__main__':unittest.main(verbosity=2)
