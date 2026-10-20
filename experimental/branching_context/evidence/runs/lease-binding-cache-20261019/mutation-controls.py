import io,json,unittest
from unittest.mock import patch
from context_time_travel.strata_backend import StrataBackend,_Lease
from test_vision_binding_cache import BindingCacheTests
original=StrataBackend._expected_input_binding
exit_original=_Lease.__exit__
def object_only(self,history):
 lease=self._active
 if lease._input_binding_history is history:return lease._input_binding_value
 return original(self,history)
def no_native_comparison(self,history):return self._request('vision_info')
def retain_closed(self,*args):
 saved=(self._input_binding_history,self._input_binding_encoded,self._input_binding_value)
 result=exit_original(self,*args)
 self._input_binding_history,self._input_binding_encoded,self._input_binding_value=saved
 return result
results=[]
for name,target,attribute,mutant,method in [
 ('history-object-only-key',StrataBackend,'_expected_input_binding',object_only,'test_same_object_changed_encoded_bytes_cannot_hit_old_binding'),
 ('native-comparison-omitted',StrataBackend,'_check_input_binding',no_native_comparison,'test_cached_expected_still_rejects_each_wrong_native_binding'),
 ('closed-lease-retains-derived-history',_Lease,'__exit__',retain_closed,'test_cache_is_lease_local_and_release_discards_derived_history')]:
 stream=io.StringIO()
 with patch.object(target,attribute,mutant):
  outcome=unittest.TextTestRunner(stream=stream,verbosity=2).run(unittest.TestSuite([BindingCacheTests(method)]))
 assert outcome.failures and not outcome.errors,stream.getvalue()
 results.append(dict(mutant=name,status='DETECTED',failed_assertions=len(outcome.failures),errors=0,detecting_method=method,log=stream.getvalue()))
print(json.dumps(dict(status='ALL_THREE_HOST_PROTOCOL_MUTANTS_DETECTED',gpu_evidence=False,mutants=results),indent=2))
