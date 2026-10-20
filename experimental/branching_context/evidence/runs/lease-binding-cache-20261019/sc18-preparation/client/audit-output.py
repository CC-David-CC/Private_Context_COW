from pathlib import Path
import json,hashlib,shutil
from branchwrite.execution import Executor,Grant
D=Path(__file__).parent
assert (D/'command.exit').read_text().strip()=='0'
result=json.loads((D/'output/result.json').read_text());assert result['recovered_exact_answer'] and result['workflow_ledger_unchanged_by_reconcile']
base=json.loads((D/'baseline-inventory.json').read_text());source=D/'source';workspace=D/'state/world/workspace'
assert all(hashlib.sha256((source/n).read_bytes()).hexdigest()==sha for n,sha in base.items())
actual={str(p.relative_to(workspace)):hashlib.sha256(p.read_bytes()).hexdigest() for p in workspace.rglob('*') if p.is_file()}
assert set(actual)==set(base)
changed=[n for n in actual if actual[n]!=base[n]];assert changed==['intervals.py'],changed
isolated=D/'independent-source';shutil.copytree(workspace,isolated)
independent="""import unittest,random,copy
from intervals import merge_intervals
class Generated(unittest.TestCase):
 def test_seeded_coverage(self):
  rng=random.Random(181019)
  for index in range(1000):
   rows=[]
   for _ in range(rng.randrange(20)):
    a,b=sorted([rng.randrange(-20,21),rng.randrange(-20,21)]);rows.append([a,b])
   rng.shuffle(rows);before=copy.deepcopy(rows)
   points=sorted({x for a,b in rows for x in range(2*a,2*b+1)})
   runs=[]
   for x in points:
    if not runs or x>runs[-1][1]+1:runs.append([x,x])
    else:runs[-1][1]=x
   expected=[(a//2,b//2) for a,b in runs]
   output=merge_intervals(rows);self.assertIs(type(output),list)
   self.assertTrue(all(type(pair) is tuple and len(pair)==2 and all(type(v) is int for v in pair) for pair in output))
   self.assertEqual(output,expected,index);self.assertEqual(rows,before,index)
"""
(isolated/'tests/test_independent.py').write_text(independent)
e=Executor.create(isolated,D/'independent-world');tested=e.run_tests('sc18-independent',Grant((),True),timeout=30)
(D/'independent.stdout').write_bytes(tested.stdout.read_bytes());(D/'independent.stderr').write_bytes(tested.stderr.read_bytes())
r=dict(status='PASS' if tested.exit_code==0 else 'FAIL',exit_code=tested.exit_code,changed=changed,original_source_unchanged=True,protected_tests_unchanged=True,generated_cases=1000,seed=181019,scope='New interval repair after real lost acknowledgment; not process-kill/tool-crash qualification')
(D/'audit-output-result.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r));raise SystemExit(tested.exit_code!=0)
