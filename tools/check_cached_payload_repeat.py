"""Compare one identical 4K serving payload cold and warm; run after a server restart."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import urllib.request

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from serve.workspace_messages import model_messages


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base',required=True);parser.add_argument('--campaign',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--key-env',default='STRATA_API_KEY')
    args=parser.parse_args();headers={'Content-Type':'application/json','Authorization':'Bearer '+os.environ.get(args.key_env,'')}
    def call(path,body=None):
        req=urllib.request.Request(args.base.rstrip('/')+path,headers=headers,data=json.dumps(body).encode() if body is not None else None)
        with urllib.request.urlopen(req,timeout=240) as response:return json.loads(response.read())
    case=next(c for c in json.loads(args.campaign.read_text())['cases'] if c['target_prefix_tokens']==4096)
    branch=next(b for b in call('/workspace/branches?chat='+case['chat_id'])['branches'] if b['name']=='A')
    history=call('/workspace/history?branch='+branch['id'])['messages']
    body={'model':'strata','messages':model_messages(history[:-1]),'temperature':0,'reasoning_effort':'none','max_tokens':128,'stream':False,'strata_mcp':False}
    report={'scope':'identical HTTP payload; first request after restart and two warm repeats; caches remain enabled; no full-logit oracle',
            'payload_sha256':hashlib.sha256(json.dumps(body,sort_keys=True).encode()).hexdigest(),'runs':[]}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    def save():args.output.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    try:
        for index in range(3):
            start=time.perf_counter();response=call('/v1/chat/completions',body)
            report['runs'].append({'index':index,'response':response,'wall_s':time.perf_counter()-start});save()
        answers=[r['response']['choices'][0]['message']['content'] for r in report['runs']]
        report.update(cold_vs_warm_match=answers[0]==answers[1],warm_vs_warm_match=answers[1]==answers[2],
                      first_run_cached_tokens=report['runs'][0]['response']['usage']['prompt_tokens_details']['cached_tokens'])
        report['status']='PASS' if report['cold_vs_warm_match'] and report['warm_vs_warm_match'] and report['first_run_cached_tokens']==0 else 'FAIL'
    except Exception as error:report.update(status='FAIL',error=repr(error));raise
    finally:save()
    return 0 if report['status']=='PASS' else 1


if __name__=='__main__':sys.exit(main())
