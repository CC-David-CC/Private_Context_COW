from pathlib import Path
import time,sys,json,hashlib,shutil,difflib,random
d=Path(__file__).parent
while not (d/'client-run/command.exit').exists():time.sleep(1)
if (d/'client-run/command.exit').read_text()!='0':
 (d/'audit-result.json').write_text(json.dumps(dict(status='HARNESS_OR_WORKFLOW_FAILURE',note='Original client result retained; no completion claim')))
 raise SystemExit(1)
repo=Path('/home/dflanag3/src/branchwrite-grammar-client-20261019');sys.path.insert(0,str(repo))
from branchwrite.records import RuntimeLog,Artifact,decode
from branchwrite.kernel import FrameKernel
from branchwrite.execution import Executor,Grant
from branchwrite.terminal import ShellConfig
def hashes(root):return {str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in root.rglob('*') if p.is_file()}
ledger=d/'state/ledger';before=hashes(ledger);copy=d/'offline-ledger-copy';shutil.copytree(ledger,copy)
tools=[];models=[]
with RuntimeLog(copy) as log:
 for frame in FrameKernel(log).frames():
  if frame['state']!='Closed' or frame['contract'][0] not in ['ToolOutcome','ModelOutput']:continue
  value=decode(log.read_artifact(Artifact(*frame['result'])))[2]
  inputs=decode(log.read_artifact(Artifact(*frame['inputs'])))
  if frame['contract'][0]=='ToolOutcome':
   tools.append(dict(tool=value['tool'],arguments=inputs[4],disposition=value['disposition'],exit_code=value['exit_code'],stdout=log.read_artifact(Artifact(*value['stdout'])).decode('utf8','replace'),stderr=log.read_artifact(Artifact(*value['stderr'])).decode('utf8','replace'),details=value['details']))
  else:models.append(dict(tokens=value['consumed_output_tokens'],grammar_state=inputs.get('config',{}).get('grammar_state')))
assert hashes(ledger)==before
failures=[];world=d/'state/world/workspace'
manifest=json.loads((d/'source-before.json').read_text())
if hashes(d/'source')!={k:v['sha256'] for k,v in manifest.items()}:failures.append('original_source_changed')
for name,record in manifest.items():
 p=world/name
 if not p.is_file():failures.append('missing_file:'+name);continue
 if name!='ledger.py' and (hashlib.sha256(p.read_bytes()).hexdigest()!=record['sha256'] or oct(p.stat().st_mode&0o777)!=record['mode']):failures.append('protected_file_changed:'+name)
if not tools or tools[-1]['tool']!='finish':failures.append('missing_finish')
command='python3 -m unittest discover -s tests -v'
runs=[t for t in tools if t['tool']=='run_shell' and t['arguments'].get('command')==command]
if not any(t['disposition']=='exited' and t['exit_code']==0 and 'Ran 8 tests' in t['stderr'] and 'OK' in t['stderr'] for t in runs):failures.append('missing_actual_required_tests_pass')
for t in runs:
 w=t['details'].get('isolation',{})
 if w.get('worker_exited') is not True or w.get('namespace_members_after')!=[]:failures.append('unproven_command_retirement')
registrations=[json.loads(p.read_text()) for p in (d/'state/cow-operations').glob('*.json') if json.loads(p.read_text()).get('path')=='/cow/grammars']
if len(registrations)!=1 or registrations[0]['state']!='complete':failures.append('missing_durable_registration')
else:
 import base64
 body=json.loads(base64.b64decode(registrations[0]['body_b64']))
 definition={k:v for k,v in body.items() if k!='operation_id'}
 if definition!=json.loads((d/'grammar.json').read_text()):failures.append('registered_definition_differs_from_host_grammar')
 states=set(registrations[0]['receipt']['states'].values())
 if any(m['grammar_state'] not in states for m in models):failures.append('model_used_unregistered_state')
# Independent original test execution through another actual readonly namespace.
reads=tuple(['README.md','ledger.py','tests/__init__.py','tests/test_ledger.py'])
e=Executor.create(world,d/'operator-visible-world',shell=ShellConfig(timeout=12))
v=e.run_shell('operator-visible',Grant((),False,read_paths=reads,allow_shell=True),command)
(d/'operator-visible.stdout').write_bytes(v.stdout.read_bytes());(d/'operator-visible.stderr').write_bytes(v.stderr.read_bytes())
if v.disposition!='exited' or v.exit_code!=0:failures.append('independent_visible_tests_failed')
# Hidden generator derives expected sums before serializing input records.
h=d/'operator-heldout-source';h.mkdir();(h/'ledger.py').write_bytes((world/'ledger.py').read_bytes())
rng=random.Random(7375);cases=[]
for index in range(64):
 names=['café','雪','oak','space name',"quote'name",'literal$HOME']
 chosen=[(rng.choice(names),rng.randrange(1000)) for _ in range(rng.randrange(1,25))]
 expected={}
 for name,count in chosen:expected[name]=expected.get(name,0)+count
 lines=[' # synthetic fixture','']+['  '+name+' : '+str(count)+'  ' for name,count in chosen]
 cases.append([lines,expected])
(h/'cases.json').write_text(json.dumps(cases,ensure_ascii=False))
(h/'test_generated.py').write_text("import json,unittest\nfrom pathlib import Path\nfrom ledger import summarize\nclass Generated(unittest.TestCase):\n def test_seeded_totals(self):\n  for i,(lines,expected) in enumerate(json.loads(Path('cases.json').read_text())):\n   with self.subTest(case=i):self.assertEqual(summarize(iter(lines)),expected)\n")
e=Executor.create(h,d/'operator-heldout-world',shell=ShellConfig(timeout=12))
q=e.run_shell('operator-heldout',Grant((),False,read_paths=('ledger.py','cases.json','test_generated.py'),allow_shell=True),'python3 -m unittest discover -v')
(d/'operator-heldout.stdout').write_bytes(q.stdout.read_bytes());(d/'operator-heldout.stderr').write_bytes(q.stderr.read_bytes())
if q.disposition!='exited' or q.exit_code!=0:failures.append('heldout_seeded_totals_failed')
result=json.loads((d/'output/result.json').read_text())
out=dict(status='BOUNDED_AUTONOMOUS_SC01_PASS' if not failures else 'TASK_FAILURE',failures=failures,model='Q8_0',kv='FP16',sampler='greedy',thinking_tokens=4096,domain_examples_added=0,model_calls=len(models),output_tokens=sum(m['tokens'] for m in models),whole_cli_seconds=result['positive']['seconds'],visible_test_methods=8,heldout_methods=1,heldout_generated_cases=64,seed=7375,ledger_unchanged=True,tools=tools,model_requests=models,final_answer=tools[-1]['arguments'].get('answer','') if tools else '',diff=''.join(difflib.unified_diff((d/'source/ledger.py').read_text().splitlines(True),(world/'ledger.py').read_text().splitlines(True),fromfile='ledger.py before',tofile='ledger.py after')),scope='One model-selected coding trajectory,not general certification or fresh-logit equivalence')
(d/'audit-result.json').write_text(json.dumps(out,indent=2)+'\n')
(d/'audit.exit').write_text('0' if not failures else '1')
print(json.dumps({k:v for k,v in out.items() if k not in ['tools','model_requests']},indent=2))
raise SystemExit(bool(failures))
