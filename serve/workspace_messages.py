"""Presentation history -> ordinary serving requests; no native snapshot alias."""
import time


def model_messages(history):
    out=[]
    for message in history:
        if message['role']=='user':
            text=message.get('text','')
            for file in message.get('files',[]):
                if file.get('text') is not None:
                    fence='`'*(max([2]+[len(s) for s in __import__('re').findall(r'`+',file['text'])])+1)
                    text += f"\n\nFile: {file['name']}\n{fence}\n{file['text']}\n{fence}"
            images=[{'type':'image_url','image_url':{'url':image['url']}} for image in message.get('images',[]) if image.get('url')]
            out.append({'role':'user','content':[{'type':'text','text':text},*images] if images else text})
            continue
        tools=[t for t in message.get('tools',[]) if t.get('round') is not None and t.get('result') is not None and t.get('state')!='skipped']
        if not tools:
            item={'role':'assistant','content':message.get('text','')}
            if not item['content'] and message.get('reasoning'):item['reasoning_content']=message['reasoning']
            out.append(item);continue
        text=message.get('text','');pos=0
        for round_id in dict.fromkeys(t['round'] for t in tools):
            calls=[t for t in tools if t['round']==round_id]
            at=min(max(pos,calls[0].get('at',0)),len(text))
            out.append({'role':'assistant','content':text[pos:at].strip(),'tool_calls':[
                {'id':t['id'],'type':'function','function':{'name':t['name'],'arguments':t.get('arguments',{})}} for t in calls]})
            out.extend({'role':'tool','tool_call_id':t['id'],'content':t['result']} for t in calls)
            pos=at
        if text[pos:].strip():out.append({'role':'assistant','content':text[pos:].strip()})
    return out


class Answer:
    def __init__(self):
        self.message={'role':'assistant','text':'','reasoning':'','time':time.time()*1000}
        self.usage=None

    def accept(self,chunk):
        message=self.message
        event=chunk.get('strata_mcp')
        if event:
            if event['event']=='limit':message['limit']=event['max_rounds']
            else:
                tools=message.setdefault('tools',[])
                tool=next((t for t in tools if t['id']==event['id']),None)
                if tool is None:
                    tool={'id':event['id'],'name':event.get('name'),'at':len(message['text']),'rat':len(message['reasoning']),'state':'writing'};tools.append(tool)
                if event['event']=='call':
                    tool.update({k:event.get(k) for k in ['name','server','tool','arguments','round']});tool['state']='running'
                elif event['event']=='result':
                    tool.update(result=event.get('text',''),ok=event.get('ok',False),chars=event.get('chars'),truncated=event.get('truncated'),ms=event.get('ms'),
                                state='skipped' if event.get('skipped') else 'done' if event.get('ok') else 'error')
        delta=(chunk.get('choices') or [{}])[0].get('delta',{})
        for field,key,offset in [('text','content','at'),('reasoning','reasoning_content','rat')]:
            value=delta.get(key)
            if value:
                last=(message.get('tools') or [None])[-1]
                if last and message[field] and last.get(offset)==len(message[field]):message[field]+='\n\n'
                message[field]+=value
        if chunk.get('usage'):self.usage=chunk['usage']

    def finish(self,reason):
        if self.usage: self.message['meta']=f"{self.usage['completion_tokens']} tokens"
        if reason=='length':self.message['meta']=(self.message.get('meta','')+' · token limit reached').strip(' ·')
        if reason in ('stopped','failed','length'):self.message['stopped']=True
        for tool in self.message.get('tools',[]):
            if tool['state'] in ('writing','running'):tool['state']='skipped'
        return self.message


def track_chunks(chunks,workspace,request_id,cancel,answer):
    committed=False
    try:
        for chunk in chunks:
            if chunk is not None:
                answer.accept(chunk)
                reason=(chunk.get('choices') or [{}])[0].get('finish_reason')
                if reason is not None:
                    status='stopped' if cancel.is_set() or reason=='length' else 'completed'
                    head=workspace.finish(request_id,answer.finish('length' if reason=='length' else status),status)
                    committed=True
                    chunk={**chunk,'context_workspace':{'branch_id':head['id'],'revision':head['revision'],'tip':head['tip'],'saved':True,'execution_mode':'ordinary-serving-history-replay'}}
            yield chunk
    finally:
        if not committed:
            workspace.finish(request_id,answer.finish('stopped' if cancel.is_set() else 'failed'),'stopped' if cancel.is_set() else 'failed')
        chunks.close()
