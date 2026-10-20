"""Declared source pins reject deployment mismatch before model/store startup."""
import hashlib,json,os,subprocess,sys,tempfile,unittest
from pathlib import Path
from context_time_travel import strata_backend as backend
from context_time_travel.store import IntegrityError

class RuntimeSourceIdentityTests(unittest.TestCase):
 def setUp(self):
  self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)/'runtime'
  (self.root/'serve').mkdir(parents=True);self.file=self.root/'serve/example.py';self.file.write_text('VALUE = 1\n')
  self.profile=dict(native_build=dict(selected_runtime_source_closure=dict(source_commit='a'*40,files_sha256={'/old/location/serve/example.py':hashlib.sha256(self.file.read_bytes()).hexdigest()})))
 def verify(self):return backend.verify_runtime_source_closure(self.profile,self.root)
 def test_relocated_declared_sources_match_by_exact_bytes(self):
  result=self.verify();self.assertEqual(result['status'],'verified');self.assertEqual(result['files'],1)
 def test_windows_declared_paths_relocate_without_changing_bytes(self):
  pins=self.profile['native_build']['selected_runtime_source_closure']['files_sha256']
  expected=next(iter(pins.values()))
  self.profile['native_build']['selected_runtime_source_closure']['files_sha256']={'C:\\old\\runtime\\serve\\example.py':expected}
  self.assertEqual(self.verify()['status'],'verified')
 def test_modified_source_rejected(self):
  self.file.write_text('VALUE = 2\n')
  with self.assertRaises(IntegrityError):self.verify()
 def test_missing_source_rejected(self):
  self.file.unlink()
  with self.assertRaises(IntegrityError):self.verify()
 def test_invalid_hash_rejected(self):
  self.profile['native_build']['selected_runtime_source_closure']['files_sha256']['/old/location/serve/example.py']=True
  with self.assertRaises(IntegrityError):self.verify()
 def test_parent_escape_rejected(self):
  self.profile['native_build']['selected_runtime_source_closure']['files_sha256']={'serve/../outside.py':'a'*64}
  with self.assertRaises(IntegrityError):self.verify()
 def test_duplicate_relative_aliases_rejected(self):
  pins=self.profile['native_build']['selected_runtime_source_closure']['files_sha256'];pins['serve/example.py']=next(iter(pins.values()))
  with self.assertRaises(IntegrityError):self.verify()
 def test_symlink_is_not_a_pinned_source_file(self):
  saved=self.root/'elsewhere.py';self.file.rename(saved);self.file.symlink_to(saved)
  with self.assertRaises(IntegrityError):self.verify()
 def test_missing_declaration_is_reported_as_unverified(self):
  result=backend.verify_runtime_source_closure(dict(native_build={}),self.root)
  self.assertEqual(result['status'],'not_declared');self.assertEqual(result['files'],0)
 def test_normal_server_rejects_declared_mismatch_before_creating_store(self):
  repo=Path(__file__).resolve().parents[1]
  profile=Path(self.temp.name)/'profile.json';store=Path(self.temp.name)/'server-store'
  profile.write_text(json.dumps(dict(model_contract_id='model',execution_profile_id='profile',execution=dict(GPU=[0]),native_build=dict(selected_runtime_source_closure=dict(source_commit='a'*40,files_sha256={'serve/cow_http.py':'0'*64})))))
  r=subprocess.run([sys.executable,'-B','-m','serve.cow_http','--profile',str(profile),'--store',str(store)],cwd=repo,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'),capture_output=True,text=True,timeout=20)
  self.assertNotEqual(r.returncode,0)
  self.assertIn('Runtime source fingerprint mismatch',r.stderr)
  self.assertFalse(store.exists(),'source validation must precede store/model startup')
