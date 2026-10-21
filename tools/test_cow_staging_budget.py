"""Checkpoint working-set planning; the 2 GiB live failure is retained separately."""
from pathlib import Path
import sys,tempfile,unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parent))
import cow_easy_setup as setup
G=1024**3
class WorkingSetPolicyTests(unittest.TestCase):
 def hardware(self,available):return setup.Hardware(32*G,available*G,24*G,24*G,12,'Fixture GPU','/cuda/nvcc')
 def test_small_server_reserves_transfer_and_capture_working_set(self):
  p=setup.plan_for(self.hardware(24))
  self.assertGreaterEqual(p.ram_gib,6)
  self.assertLessEqual(p.resident_gib+p.ram_gib+p.min_available_gib+6,24)
  self.assertEqual(p.context,32768)
 def test_guided_setup_refuses_insufficient_staging_budget_before_install(self):
  with tempfile.TemporaryDirectory() as tmp:
   data=Path(tmp)/'new'
   with patch.object(setup,'inspect_host',return_value=self.hardware(17)),patch.object(setup,'run_step',side_effect=AssertionError('must not build or download')):
    with self.assertRaisesRegex(setup.SetupError,'checkpoint staging'):
     setup.main(['--model','IQ2_XS','--data-dir',str(data),'--yes'])
   self.assertFalse(data.exists())
if __name__=='__main__':unittest.main(verbosity=2)
