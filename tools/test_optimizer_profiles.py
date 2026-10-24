import copy,unittest
from tools.pcc_optimizer_profiles import profiles,select,identity

class OptimizerProfileTests(unittest.TestCase):
 def profile(self):return dict(schema=1,title='Owned native policy',optimizer='muon',learning_rate=.001,optimizer_settings=dict(schema=1),state_policy='fresh',training_budget_seconds=1.)
 def test_registered_profile_is_detached_and_cannot_change_under_same_id(self):
  p=self.profile();key=identity(p);owned=profiles({key:p},sequence_profile='dense-llama-short-v1',deadline=60);p['optimizer_settings']['schema']=9
  request=dict(mode='full',learning_rate=.001,optimizer_profile_id=key,parent_job_id=None);self.assertEqual(select(owned,request)['optimizer_settings']['schema'],1)
  owned[key]['learning_rate']=.002
  with self.assertRaises(ValueError):select(owned,request)
 def test_unsupported_shape_budget_and_identity_refused(self):
  p=self.profile()
  for change in [dict(schema=True),dict(optimizer='sgd'),dict(state_policy='guess'),dict(training_budget_seconds=60.),dict(training_budget_seconds=True),dict(learning_rate=0.)]:
   d=dict(p,**change)
   with self.assertRaises(ValueError):profiles({identity(d):d},sequence_profile='dense-llama-short-v1',deadline=60)
  with self.assertRaises(ValueError):profiles({'a'*64:p},sequence_profile='dense-llama-short-v1',deadline=60)
  with self.assertRaises(ValueError):profiles({identity(p):p},sequence_profile='short-v1',deadline=60)
 def test_no_profile_keeps_legacy_path_but_null_is_not_a_profile(self):
  self.assertIsNone(select({},dict(mode='full',learning_rate=.001)))
  with self.assertRaises(ValueError):select({},dict(mode='full',learning_rate=.001,optimizer_profile_id=None))
if __name__=='__main__':unittest.main()
