from pathlib import Path
import json,hashlib,statistics
D=Path(__file__).parent
result=json.loads((D/'campaign.json').read_text());assert result['status']=='PASS',result
records={name:json.loads((D/name/'result.json').read_text()) for name in ['baseline','candidate','restart']}
summary={'status':'PASS','scope':result['scope'],'comparisons':result['comparisons'],'timings':{},'controller_sha256':{},'repetition_exactness':{},'raw_files':{}}
for kind,r in records.items():
 assert r['status']=='PASS' and r['native_exit']['exit_code']==0 and r['native_exit']['absent']
 summary['timings'][kind]={}
 summary['controller_sha256'][kind]={}
 for side in ['A','B']:
  items=[r['branches'][side+str(i)] for i in range(3)]
  assert all(x['tokens']==items[0]['tokens'] for x in items)
  vectors=[(D/kind/(side+str(i)+'.f32')).read_bytes() for i in range(3)]
  assert all(x==vectors[0] for x in vectors)
  rows=[]
  for i,x in enumerate(items):
   count=len(x['tokens']);a=x['work_before'];b=x['work_after']
   elapsed=sum(b[k]-a[k] for k in ['decode_prepare_ns','decode_compute_ns','decode_finish_ns'])/1e9
   rows.append(dict(output_tokens=count,request_seconds=x['request_seconds'],generation_seconds=x['generation_seconds'],generation_tps=count/x['generation_seconds'],native_timed_tps=count/elapsed,forward_tokens=b['forward_tokens']-a['forward_tokens'],integrity_mismatches=b['integrity_mismatches']-a['integrity_mismatches']))
   assert rows[-1]['integrity_mismatches']==0
   summary['controller_sha256'][kind][side+str(i)]=hashlib.sha256(json.dumps(x['controller'],sort_keys=True).encode()).hexdigest()
  summary['timings'][kind][side]={'individual':rows,'median_request_seconds':statistics.median(x['request_seconds'] for x in rows),'median_generation_tps':statistics.median(x['generation_tps'] for x in rows),'median_native_timed_tps':statistics.median(x['native_timed_tps'] for x in rows)}
  summary['repetition_exactness'][kind+side]=True
 for path in (D/kind).glob('*.f32'):summary['raw_files'][str(path.relative_to(D))]={'bytes':path.stat().st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
 summary['raw_files'][kind+'/result.json']={'bytes':(D/kind/'result.json').stat().st_size,'sha256':hashlib.sha256((D/kind/'result.json').read_bytes()).hexdigest()}
for kind in ['candidate','restart']:assert summary['controller_sha256'][kind]==summary['controller_sha256']['baseline']
restore=records['restart']['restore'];assert restore['forward_tokens']==0
summary['restore']={k:restore[k] for k in ['consumed','seconds','forward_tokens','native_pid','execution_profile_id']}
summary['request_change_percent']={side:100*(summary['timings']['candidate'][side]['median_request_seconds']/summary['timings']['baseline'][side]['median_request_seconds']-1) for side in ['A','B']}
summary['limits']='Sequential baseline/candidate stages, three repetitions per question, capped thinking-inclusive output. Not randomized timing, final-answer throughput, HTTP/CLI performance or independent full-history oracle.'
(D/'analysis.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps({'status':summary['status'],'request_change_percent':summary['request_change_percent'],'restore':summary['restore'],'timings':summary['timings']},indent=2))
