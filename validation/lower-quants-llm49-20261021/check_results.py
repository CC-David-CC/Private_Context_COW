from pathlib import Path
import json
root=Path(__file__).resolve().parent
j=json.loads((root/'result.json').read_text())
assert j['numerical_max_error'] is None and j['oracle_comparisons']==0
assert j['setup_tests']['unit_tests']==44 and j['setup_tests']['G0_tests']==51
assert j['setup_tests']['failures']==j['setup_tests']['skips']==0
for run in j['runs']:
 assert run['official_model_hashes_match'] and run['kv_dtype']=='FP16'
 for row in run['stages'][1:]:
  c=row['native_counter_deltas'];assert c['prefill_tokens']==19 and c['prefill_chunks']==1
  assert c['forward_tokens']==row['generated_tokens']+19
  assert c['decode_timed_tokens']==row['generated_tokens'] and c['integrity_mismatches']==0
long=j['runs'][2]
assert long['prefix_tokens']==22092 and long['stages'][0]['native_counter_deltas']['prefill_chunks']==3
assert long['stages'][0]['ram_reservation_bytes']>2*1024**3
assert j['fresh_install']['exit_code']==0 and j['fresh_cli_chat']['passed']
assert j['actual_checkpoint_directory_file_bytes']==j['final_pool']['disk']['file_bytes']
assert j['state_file_names_match_independent_sha256']
assert len(j['disk_state_files'])==5
states=[c['status'] for c in j['cases']]
assert states.count('PASS')==10 and states.count('FAIL')==2 and states.count('TIMEOUT')==1
print('Recorded evidence checks PASS. Two custom failures and one timeout remain recorded; G1-G5 are not qualified.')
