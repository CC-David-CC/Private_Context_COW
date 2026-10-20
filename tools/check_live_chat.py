"""Measure exact-copy behavior of a real model; keep mismatches, do not repair output."""
import argparse
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from serve.test_literal_boundaries import LITERALS


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base',required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--key-env',default='STRATA_API_KEY')
    a=parser.parse_args()
    headers={'Content-Type':'application/json'}
    if os.environ.get(a.key_env):
        headers['Authorization']='Bearer '+os.environ[a.key_env]
    report={'status':'RUNNING','scope':'real model exact-copy measurement, not a numerical oracle',
            'requests':[]}
    def save():
        a.output.parent.mkdir(parents=True,exist_ok=True)
        a.output.write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    save()
    for index,literal in enumerate(LITERALS):
        stream=index % 2 == 0
        body={'model':'strata','messages':[{'role':'user','content':
              'Copy only the following text exactly. Do not explain it or add formatting:\n'+literal}],
              'reasoning_effort':'none','temperature':0,'max_tokens':128,'stream':stream}
        start=time.perf_counter()
        row={'literal':literal,'stream':stream,'index':index}
        try:
            req=urllib.request.Request(a.base.rstrip('/')+'/v1/chat/completions',
                data=json.dumps(body).encode(),headers=headers)
            with urllib.request.urlopen(req,timeout=120) as response:
                row['http_status']=response.status
                if stream:
                    chunks=[]
                    for line in response:
                        if line.startswith(b'data: ') and line.strip()!=b'data: [DONE]':
                            chunks.append(json.loads(line[6:]))
                    row['response']=chunks
                    row['answer']=''.join(c.get('choices',[{}])[0].get('delta',{}).get('content','') or ''
                                          for c in chunks if c.get('choices'))
                else:
                    row['response']=json.loads(response.read())
                    row['answer']=row['response']['choices'][0]['message'].get('content','') or ''
            row['exact_copy']=row['answer']==literal
        except Exception as error:
            row['error']=repr(error)
            row['exact_copy']=False
        row['seconds']=time.perf_counter()-start
        report['requests'].append(row)
        save()
        print(json.dumps({k:v for k,v in row.items() if k!='response'},ensure_ascii=True),flush=True)
    report.update(status='MEASURED',exact_copies=sum(r['exact_copy'] for r in report['requests']),
                  requests_completed=len(report['requests']),
                  transport_errors=sum('error' in r for r in report['requests']))
    save()
    return 1 if report['transport_errors'] else 0


if __name__=='__main__':
    raise SystemExit(main())
