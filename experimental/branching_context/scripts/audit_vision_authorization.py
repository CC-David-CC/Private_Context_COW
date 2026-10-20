"""Independently re-read retained authorization evidence without loading a model.

This checks saved bytes, native receipts, canonical controller bindings and exits.
Per-denial before/after inventories were checked live but not retained separately;
this auditor cannot independently reconstruct those intermediate inventories.
"""
import argparse, array, hashlib, json, math, sys
from pathlib import Path
class AuditFailure(RuntimeError):pass

def require(ok,message):
 if not ok:raise AuditFailure(message)
def load(path):return json.loads(Path(path).read_bytes())
def digest(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
 return h.hexdigest()
def fingerprint(path):
 p=Path(path);return dict(path=str(p),bytes=p.stat().st_size,sha256=digest(p))
def verify_pin(row):
 require(fingerprint(row['path'])==row,'Pinned file changed: '+row['path'])

def compare_vectors(expected,actual,values=248320):
 a=Path(expected).read_bytes();b=Path(actual).read_bytes()
 require(len(a)==len(b)==values*4,'Wrong or truncated frontier size')
 aa=array.array('f');bb=array.array('f');aa.frombytes(a);bb.frombytes(b)
 if sys.byteorder!='little':aa.byteswap();bb.byteswap()
 require(aa.itemsize==4,'Host float width differs')
 require(all(math.isfinite(x) for x in aa) and all(math.isfinite(x) for x in bb),'Nonfinite frontier')
 error=max((abs(x-y) for x,y in zip(aa,bb)),default=0)
 require(a==b and error==0,'Frontiers differ')
 return dict(values=values,bytes_identical=True,max_abs_error=error,expected=fingerprint(expected),actual=fingerprint(actual))

def validate_work_receipts(rows,total):
 position=0
 for r in rows:
  require(r['status']=='ok' and r['position']==position and type(r['count']) is int and r['count']>0,'Failed or out-of-order forward receipt')
  chunks=r['chunks'];cursor=position
  for start,count in chunks:
   require(start==cursor and type(count) is int and count>0,'Invalid native chunk');cursor+=count
  require(cursor==position+r['count']==r['consumed'],'Consumed position/chunks differ')
  position=cursor
 require(position==total,'Incomplete forward receipts')
 return position

def validate_denial(row,status,fragment,body_sent):
 require(row['status']==status and fragment.casefold() in row['response']['error'].casefold(),'Wrong denial guard')
 require(row['native_jobs_delta']==row['encoder_calls_delta']==0 and row['protected_state_unchanged'] is True,'Denial effects claim failed')
 require(row['body_sent'] is body_sent,'Wrong header/body fixture')
 require(math.isfinite(row['seconds']) and row['seconds']>=0 and (body_sent or row['seconds']<3),'Header denial waited for body')

def audit(run,wrapper):
 run=Path(run).resolve(strict=True);wrapper=Path(wrapper).resolve(strict=True);r=load(run/'result.json')
 require(r['status']=='BOUNDED_Q4_REAL_HTTP_REJECTIONS_AUTHORIZED_REUSE_AND_RECOVERY_PASS','Live run did not pass')
 for name in ['command.exit','wrapper.exit']:require((wrapper/name).read_text().strip()=='0','Nonzero/unknown '+name)
 verify_pin(r['profile']);profile=load(r['profile']['path'])
 require(profile['execution_profile_id']==r['execution_profile_id'] and profile['model_contract_id']==r['model_contract_id'],'Profile binding differs')
 denials=[(401,'API key',False),(403,'Foreign Origin',False),(404,'namespace',False),(400,'digest does not match',True),(400,'Invalid',True),(409,'not uploaded in this namespace',True),(409,'At most eight',True)]
 require(len(r['rejections'])==len(denials),'Missing denial')
 for i,(row,expected) in enumerate(zip(r['rejections'],denials),1):
  require(row['id'].startswith('VA-'+str(i).zfill(3)+'-'),'Wrong denial ordering');validate_denial(row,*expected)
 jobs=[json.loads(x) for x in (run/'jobs.jsonl').read_text().splitlines()]
 require([x['operation'] for x in jobs]==['auth-context-A','auth-context-B','auth-valid-question'],'Unexpected/missing native job')
 initial=r['controls']['initial'];dup=r['controls']['authorized_duplicate'];cont=r['controls']['post_denial_continuation']
 sid=initial['result']['receipt']['snapshot'];final=cont['result']['receipt']['snapshot'];tokens=cont['result']['answer']['tokens'];suffix=cont['suffix']
 require(initial['cells']==360 and len(suffix)==19 and len(tokens)==45,'Fixture counts differ')
 require([x['forward_tokens'] for x in jobs]==[360,0,64],'Actual job prefix/suffix work differs')
 for j in jobs:
  require(j['after']['forward_tokens']-j['before']['forward_tokens']==j['forward_tokens'],'Job counter delta differs')
  require(digest(run/j['frontier_file'])==j['frontier_sha256'],'Job frontier hash differs')
 require(jobs[0]['snapshot']==jobs[1]['snapshot']==sid and jobs[2]['snapshot']==final and jobs[2]['source']==sid,'Wrong branch source')
 require(initial['namespace']!=dup['namespace'] and dup['result']['receipt']['snapshot']==sid,'Duplicate namespace/source differs')
 require(dup['model_forward_tokens']==dup['encoder_calls']==0 and dup['native_jobs']==1,'Duplicate did work')
 compare_vectors(run/jobs[0]['frontier_file'],run/jobs[1]['frontier_file'])
 for branch,namespace,tip in [('auth-A',initial['namespace'],sid),('auth-B',dup['namespace'],final)]:
  ref=load(run/'store/references'/ (branch+'.json'))
  require(ref['head']['tip']==tip and ref['namespace_id']==namespace,'Current branch tip/namespace differs')
  # Namespace authorization binding is in the retained presentation registry,
  # not inferred from matching text or from a branch revision.
  grants=list((run/'store/image-upload-grants').glob(namespace+'-*.json'))
  require(len(grants)==1,'Missing distinct authorized upload grant')
  grant=load(grants[0])['grant'];require(grant['namespace_id']==namespace,'Wrong grant namespace')
  require(digest(run/'store/image-originals'/(grant['image_id']+'.input'))==grant['image_id'],'Original image hash differs')
 snapshot=load(run/'store/objects'/(final+'.json'))
 require(snapshot['controller']==cont['controller'],'Canonical controller binding differs')
 manifest=load(run/'store/checkpoints'/(final+'.json'))['manifest']
 require(manifest['execution_schedule']==cont['schedule'],'Stored execution schedule differs')
 require(len(tokens)==cont['forward_tokens']-len(suffix),'Generation work differs')
 verify_pin(r['parent_payload_unchanged'])
 comparisons=[]
 require(len(r['comparisons'])==2,'Missing independent comparison')
 for c in r['comparisons']:comparisons.append(dict(label=c['label'],**compare_vectors(c['expected'],c['actual'])))
 require(len(r['fresh_references'])==1 and len(r['services'])==1,'Unexpected process count')
 fresh=r['fresh_references'][0];verify_pin(fresh['binary'])
 require(fresh['exit']['exit_code']==0 and fresh['exit']['process_absent'] is True and fresh['work']['forward_tokens']==424,'Fresh reference did not finish complete work')
 ref_rows=[json.loads(x) for x in (run/'reference/native.stdout').read_text().splitlines()]
 require(ref_rows[0].get('reference')=='fresh-no-checkpoints' and ref_rows[-2]==fresh['work'] and ref_rows[-1]=={'status':'closed'},'Independent raw reference control/work differs')
 service=r['services'][0];trace=run/'native-1.jsonl'
 require(digest(run/'pool-1.json')==service['worker_shutdown']['trace_sha256'],'Pool shutdown trace changed')
 pool=load(run/'pool-1.json');require(pool['overflowed'] is False and pool['event_count']==len(pool['events']),'Pool trace incomplete')
 exits=[x for x in pool['events'] if x.get('event')=='native_worker_closed'];require(len(exits)==1 and exits[0]['returncode']==0 and exits[0]['pid']==initial['native_pid'],'Pool close differs')
 rows=[json.loads(x) for x in trace.read_text().splitlines()]
 forwards=[x['response'] for x in rows if x.get('command','').split(' ')[0] in ['feed_image','feed','decode']]
 validate_work_receipts(forwards,424)
 require(forwards[0]['chunks']==initial['manifest']['execution_schedule'][0]['chunks'],'Prefix arrangement differs')
 require([x['operator'] for x in forwards]==['batched-prefill']*2+['normal-verifier-decode']*45,'Operator arrangement differs')
 require([int(x['command'].split()[1]) for x in rows if x.get('command','').startswith('decode ')]==tokens,'Consumed tokens differ')
 closed=[x for x in rows if x.get('closed') is True];require(len(closed)==1,'Missing native closure')
 counters=closed[0]['counters'];require(counters['forward_tokens']==424 and counters['contexts_created']==counters['contexts_freed']==1,'Work or native lifetime leak')
 require(service['stop']['exit_code']==service['worker_shutdown']['native_exit_code']==0,'Service/native exit failed')
 pids=[service['pid'],initial['native_pid'],fresh['pid']]
 require(len(set(pids))==3 and all(not Path('/proc',str(pid)).exists() for pid in pids),'Owned process still present or PID reused; inspect manually')
 for pin in service['source_imports'].values():require(digest(pin['path'])==pin['sha256'],'Imported source changed')
 require(len((run/'encoder-entries.jsonl').read_text().splitlines())==1,'Unexpected encoder work')
 return dict(schema=1,status='OFFLINE_AUTHORIZATION_AUDIT_PASS',live_result=fingerprint(run/'result.json'),execution_profile_id=r['execution_profile_id'],model_contract_id=r['model_contract_id'],hardware=profile['hardware'],weights_sha256=profile['model_weights_sha256'],comparisons=comparisons,denial_ids=[x['id'] for x in r['rejections']],native_forward_tokens=424,prefix_cells=360,suffix_tokens=19,generated_tokens=45,duplicate_forward_tokens=0,encoder_calls=1,parent_payload=r['parent_payload_unchanged'],pids_absent=pids,controller='Exact retained canonical record equality; not an independent controller replay.',scope='Existing Q4 single-key authorized namespaces; forced-token fresh prefix/final reference, no new inference.',limits=['Per-denial inventory assertions executed live; intermediate inventories not separately retained, so this audit cannot independently re-prove them.','Reference full tokens are forced: not independent greedy token-choice/grammar/RNG replay.','Not multi-tenant isolation, process restart, every-token oracle, Q8, full VC09 or original G1-G5 certification.'])

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--wrapper',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 result=audit(a.run,a.wrapper)
 with a.output.open('x',encoding='utf8',newline='\n') as f:json.dump(result,f,sort_keys=True,indent=2);f.write('\n')
 print(result['status'])
if __name__=='__main__':main()
