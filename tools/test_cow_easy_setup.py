"""CPU setup-policy tests; live GPU installation is recorded separately."""
import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'tools'))
import cow_easy_setup as setup
G=1024**3

class SetupPolicyTests(unittest.TestCase):
    def hardware(self,ram=32,available=24,vram=24,free=None):
        return setup.Hardware(ram*G,available*G,vram*G,(vram if free is None else free)*G,12,'Example GPU','/usr/local/cuda/bin/nvcc')
    def test_small_server_budgets_fit_available_ram(self):
        p=setup.plan_for(self.hardware())
        self.assertLessEqual(p.resident_gib+p.ram_gib+8,24)
        self.assertEqual(p.context,32768)
        self.assertEqual(p.prefill,8192)
        self.assertGreaterEqual(p.resident_gib,4)
        self.assertGreaterEqual(p.ram_gib,2)
        self.assertLess(p.vram_reserve_mib,24*1024)
    def test_friend_5090_is_not_given_reference_server_vram(self):
        p=setup.plan_for(self.hardware(ram=128,available=110,vram=32))
        self.assertLess(p.vram_reserve_mib,32*1024)
        self.assertGreater(p.resident_gib,setup.plan_for(self.hardware()).resident_gib)
        self.assertLessEqual(p.resident_gib+p.ram_gib+8,110)
    def test_smaller_gpu_uses_a_smaller_prefill_batch(self):
        self.assertEqual(setup.plan_for(self.hardware(vram=16)).prefill,2048)
    def test_memory_pressure_reduces_jobs_and_residency(self):
        a=setup.plan_for(self.hardware(available=24));b=setup.plan_for(self.hardware(available=17))
        self.assertLess(b.resident_gib,a.resident_gib)
        self.assertLessEqual(b.build_jobs,a.build_jobs)
    def test_insufficient_ram_fails_before_setup(self):
        with self.assertRaisesRegex(ValueError,'RAM'):setup.plan_for(self.hardware(available=7))
    def test_insufficient_gpu_memory_fails_before_download(self):
        with self.assertRaisesRegex(ValueError,'GPU'):setup.plan_for(self.hardware(vram=8))
    def test_nonfinite_resource_sizes_not_accepted(self):
        with self.assertRaises((ValueError,OverflowError)):setup.plan_for(self.hardware(available=float('nan')))
    def test_failed_command_keeps_log_and_returns_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            log=Path(tmp)/'failure.log'
            with self.assertRaisesRegex(setup.SetupError,'failure.log'):
                setup.run_step('test', [sys.executable,'-c','print("diagnostic");raise SystemExit(7)'],log,ROOT)
            self.assertIn('diagnostic',log.read_text())
    def test_commands_preserve_spaces_without_shell_interpretation(self):
        with tempfile.TemporaryDirectory() as tmp:
            log=Path(tmp)/'argument log.txt'
            setup.run_step('argument',[sys.executable,'-c','import sys;print(sys.argv[1])','a b; $HOME'],log,ROOT)
            self.assertIn('a b; $HOME',log.read_text())
    def test_bootstrap_script_shell_syntax(self):
        subprocess.run(['sh','-n',str(ROOT/'cow-setup')],check=True)

class SetupInteractionTests(unittest.TestCase):
    def test_declining_installation_creates_no_data_folder(self):
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as tmp:
            data=Path(tmp)/'not-created'
            hardware=setup.Hardware(32*G,24*G,24*G,24*G,12,'Example GPU','/cuda/nvcc')
            with patch.object(setup,'inspect_host',return_value=hardware),patch.object(setup.shutil,'disk_usage',return_value=__import__('collections').namedtuple('Disk','total used free')(1024*G,0,1024*G)),patch('builtins.input',return_value='no'):
                self.assertEqual(setup.main(['--data-dir',str(data)]),1)
            self.assertFalse(data.exists())
    def test_check_only_creates_no_data_folder(self):
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as tmp:
            data=Path(tmp)/'not-created'
            hardware=setup.Hardware(32*G,24*G,24*G,24*G,12,'Example GPU','/cuda/nvcc')
            with patch.object(setup,'inspect_host',return_value=hardware),patch.object(setup.shutil,'disk_usage',return_value=__import__('collections').namedtuple('Disk','total used free')(1024*G,0,1024*G)),patch('builtins.input',side_effect=AssertionError('must not prompt')):
                self.assertEqual(setup.main(['--check','--data-dir',str(data)]),0)
            self.assertFalse(data.exists())

class HeadroomPolicyTests(unittest.TestCase):
    def test_small_server_reserves_guard_and_native_overhead(self):
        h=setup.Hardware(32*G,24*G,24*G,24*G,12,'Example','/cuda/nvcc')
        p=setup.plan_for(h)
        self.assertEqual(p.min_available_gib,8)
        self.assertLessEqual(p.resident_gib+p.ram_gib+p.min_available_gib+6,24)
    def test_large_server_retains_sixteen_gib_guard(self):
        p=setup.plan_for(setup.Hardware(128*G,110*G,32*G,32*G,12,'Example','/cuda/nvcc'))
        self.assertEqual(p.min_available_gib,16)
    def test_http_pool_receives_positive_configured_guard(self):
        from types import SimpleNamespace
        sys.path.insert(0,str(ROOT/'experimental/branching_context/src'))
        from serve.cow_http import pool_config_from_args
        args=SimpleNamespace(ram_gib=2,disk_gib=16,min_available_gib=8,checkpoint_idle_seconds=1800)
        self.assertEqual(pool_config_from_args(args).min_available_bytes,8*G)
        args.min_available_gib=0
        with self.assertRaisesRegex(ValueError,'positive'):pool_config_from_args(args)

class ConfigurationRetryTests(unittest.TestCase):
    def test_failed_configuration_leaves_no_unusable_server_folder(self):
        from unittest.mock import patch
        from types import SimpleNamespace
        import cow_server_setup
        with tempfile.TemporaryDirectory() as tmp:
            output=Path(tmp)/'server'
            def fail(args,directory):
                (directory/'partial.json').write_text('partial')
                raise ValueError('deliberate configuration failure')
            with patch.object(cow_server_setup,'_make_profile',fail):
                with self.assertRaisesRegex(ValueError,'deliberate'):
                    cow_server_setup.make_profile(SimpleNamespace(output=output))
            self.assertFalse(output.exists())
    def test_existing_configuration_is_never_removed_or_overwritten(self):
        from types import SimpleNamespace
        import cow_server_setup
        with tempfile.TemporaryDirectory() as tmp:
            output=Path(tmp)/'server';output.mkdir();keep=output/'keep';keep.write_bytes(b'keep')
            with self.assertRaises(FileExistsError):cow_server_setup.make_profile(SimpleNamespace(output=output))
            self.assertEqual(keep.read_bytes(),b'keep')

if __name__=='__main__':unittest.main(verbosity=2)
