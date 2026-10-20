"""Real serving/cache measurements for saved 1K/4K branch histories; no logit oracle."""
import argparse
import json
import os
from pathlib import Path
import sys
import time
import urllib.request
import uuid

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from serve.workspace_messages import model_messages


def run(base,output,key):
    report={'status':'RUNNING','scope':'real ordinary serving and SQLite branches with native prompt/conversation caches; no full-logit oracle','cases':[]}
    headers={'Content-Type':'application/json','Authorization':'Bearer '+key}
    def save():
        output.parent.mkdir(parents=True,exist_ok=True)
        output.write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    def call(path,body=None):
        request=urllib.request.Request(base.rstrip('/')+path,headers=headers,data=json.dumps(body).encode() if body is not None else None)
        with urllib.request.urlopen(request,timeout=240) as response:return json.loads(response.read())
    def op():return uuid.uuid4().hex
    def history(branch):return call('/workspace/history?branch='+branch)
    def filler(rows):
        return '\n'.join(f'Record {i:05d}: The orchard has red apples, green pears, and a blue gate.' for i in range(rows))
    def count(text):
        return call('/v1/messages/count_tokens',{'model':'strata','max_tokens':128,'messages':[{'role':'user','content':text}],'thinking':{'type':'disabled'}})['input_tokens']
    def sized(target):
        lo,hi=1,target
        while lo<hi:
            middle=(lo+hi)//2
            if count(filler(middle))<target:lo=middle+1
            else:hi=middle
        text=filler(lo)+'\nKeep this reference for later. Reply only READY.'
        return text,count(text)
    def generate(branch,text,row,label):
        head=history(branch);start=time.perf_counter()
        response=call('/workspace/generate',{'branch_id':branch,'revision':head['revision'],'operation_id':op(),'message':{'role':'user','text':text},'stream':False,'temperature':0,'reasoning_effort':'none','max_tokens':128,'strata_mcp':False})
        item={'stage':label,'wall_s':time.perf_counter()-start,'response':response}
        row['requests'].append(item);save()
        print(json.dumps({'target':row['target_prefix_tokens'],'stage':label,'wall_s':item['wall_s'],'usage':response.get('usage'),'timings':response.get('timings')},ensure_ascii=True),flush=True)
        return response['choices'][0]['message'].get('content','') or ''
    save()
    try:
        for target in (1024,4096):
            row={'target_prefix_tokens':target,'requests':[],'checks':{}};report['cases'].append(row);save()
            prefix,measured=sized(target);row['count_endpoint_prefix_tokens']=measured
            chat=call('/workspace/chats/create',{'title':f'Long context {target} test','operation_id':op()});main=chat['branch']['id'];row['chat_id']=chat['id']
            generate(main,prefix,row,'cold_prefix');shared=history(main)
            branches={}
            for name in ('A','B'):
                branch=call('/workspace/branches/fork',{'branch_id':main,'node_id':shared['tip'],'name':name,'operation_id':op()})['id'];branches[name]=branch
                suffix=filler(20)+f'\nThis is branch {name}. Start your answer with BRANCH_{name}. Then write twelve short numbered observations about this reference.'
                answer=generate(branch,suffix,row,'branch_'+name)
                saved=history(branch);start=time.perf_counter()
                control=call('/v1/chat/completions',{'model':'strata','messages':model_messages(saved['messages'][:-1]),'temperature':0,'reasoning_effort':'none','max_tokens':128,'stream':False,'strata_mcp':False})
                row['requests'].append({'stage':'ordinary_control_'+name,'wall_s':time.perf_counter()-start,'response':control});save()
                row['checks']['branch_'+name+'_matches_same_history_control']=answer==(control['choices'][0]['message'].get('content','') or '')
                row['checks']['branch_'+name+'_retains_shared_nodes']=[m['node_id'] for m in saved['messages'][:2]]==[m['node_id'] for m in shared['messages']]
                row['checks']['branch_'+name+'_suffix_is_separate']=saved['messages'][2]['text']==suffix
                if name=='A':
                    call('/workspace/branches/rewind',{'branch_id':branch,'revision':saved['revision'],'node_id':shared['tip'],'operation_id':op()})
                    repeated=generate(branch,suffix,row,'rewind_retry_A');row['checks']['rewind_retry_output_matches_A']=repeated==answer
            row['checks']['main_is_unchanged']=history(main)['messages']==shared['messages']
            row['metrics_after']=call('/metrics');row['status']='PASS' if all(row['checks'].values()) else 'FAIL';save()
        report['status']='PASS' if all(c['status']=='PASS' for c in report['cases']) else 'FAIL'
    except Exception as error:
        report.update(status='FAIL',error=repr(error));raise
    finally:save()
    return 0 if report['status']=='PASS' else 1


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--base',required=True);parser.add_argument('--output',type=Path,required=True);parser.add_argument('--key-env',default='STRATA_API_KEY');args=parser.parse_args()
    sys.exit(run(args.base,args.output,os.environ.get(args.key_env,'')))
