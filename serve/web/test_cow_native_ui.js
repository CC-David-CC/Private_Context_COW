"use strict";
// Inject before app.js in a copy of index.html for a local, model-free DOM test.
(() => {
  const originalFetch = window.fetch.bind(window);
  const calls = [];
  const state = {namespace:null,branch:null,revision:0,status:'idle',messages:[],answerAttempts:0,durableOperation:null,
    rootMode:'unsupported',contextEvents:[],contextAttempts:0};
  let rejectWorkspaceList=false,healthAttempts=0;
  const legacy=JSON.stringify([{role:'user',text:'legacy local message'}]);
  const localValues=new Map();
  const sessionValues=new Map();
  let rejectSessionWrites=false;
  Object.defineProperty(window,'localStorage',{configurable:true,value:{getItem:key=>key==='strata.chat'?legacy:(localValues.has(key)?localValues.get(key):null),setItem:(key,value)=>localValues.set(key,String(value))}});
  Object.defineProperty(window,'sessionStorage',{configurable:true,value:{getItem:key=>sessionValues.get(key)||null,setItem:(key,value)=>{if(rejectSessionWrites)throw Error('quota');sessionValues.set(key,String(value));},removeItem:key=>sessionValues.delete(key)}});
  const json = (value,status=200) => new Response(JSON.stringify(value),{status,headers:{'Content-Type':'application/json'}});
  const stream = (event) => new Response(JSON.stringify(event)+'\n',{headers:{'Content-Type':'application/x-ndjson'}});
  const history = () => ({id:state.branch,chat_id:state.namespace,title:'Browser test',name:state.branch,
    revision:state.revision,tip:'snap-'+state.revision,root_id:'root',root:'root',messages:state.messages,
    moves:[],saved_points:state.revision?[{snapshot_id:'snap-'+state.revision,node_id:'snap-'+state.revision,
      revision:state.revision,kind:'answer',on_selected_ancestry:true,current:true,boundary_available:true}]:[],
    controller_status:state.status,initial_root_mode:state.rootMode,pending_controller:{active:state.status==='active',status:state.status,
      accepted_in_scope:0,pending_utf8:false},pending_request:null});
  window.fetch=async(url,options={})=>{
    const path=String(url);let body={};if(options.body){try{body=JSON.parse(options.body);}catch{}}
    calls.push({path,method:options.method||'GET',body});
    if(path==='health'){healthAttempts++;if(healthAttempts===1)return json({error:'temporary discovery failure'},503);return json({service:'context-cow',model:'Context COW',images:false,capabilities:{greedy:true,native_cow:true,tools:false}});}
    if(path==='mcp')return json({servers:[],tools:0});
    if(path.includes('/cow/workspace/chats'))return rejectWorkspaceList?json({error:'Missing or wrong API key'},401):json({chats:state.namespace?[{id:state.namespace,chat_id:state.namespace,title:'Browser test',branch_count:state.branch?1:0,max_saved_messages:state.messages.length}]:[]});
    if(path.includes('/cow/workspace/branches'))return json({branches:state.branch?[{id:state.branch,name:state.branch,chat_id:state.namespace,revision:state.revision,controller_status:state.status}]:[]});
    if(path.includes('/cow/workspace/history'))return json(history());
    if(path==='/cow/namespace'&&options.method==='POST'){
      state.namespace='1'.repeat(32);return json({namespace_id:state.namespace,title:body.title,operation_id:body.operation_id});
    }
    if(path==='/cow/text/prepare'&&options.method==='POST'){
      state.branch=body.branch;state.revision=1;state.status='idle';
      state.rootMode='prepared_answer';
      state.messages=body.messages.filter(m=>m.role==='user').map((m,i)=>({id:`u${i}`,role:'user',text:m.content,content:m.content,node_id:null,boundary_available:false}));
      return stream({event:'complete',answer:{snapshot:'snap-1'},receipt:{branch:{name:state.branch,revision:1,tip:'snap-1',root:'root'},snapshot:'snap-1'}});
    }
    if(path==='/cow/text/answer'&&options.method==='POST'){
      state.answerAttempts++;
      if(state.answerAttempts===1)return json({error:'model worker unavailable'},409);
      state.status='stopped';state.revision=2;state.messages.push({id:'a1',role:'assistant',text:'answer',content:'answer',node_id:'snap-2',boundary_available:true,snapshot_boundaries:['snap-2']});
      state.rootMode='unsupported';
      return stream({event:'complete',answer:{snapshot:'snap-2',text:'answer',tip:'snap-2'},receipt:{branch:{name:state.branch,revision:2,tip:'snap-2',root:'root'},snapshot:'snap-2'}});
    }
    if(path==='/cow/text/append-user'&&options.method==='POST'){
      if(body.content==='ack lost turn'){
        state.revision++;state.status='stopped';state.messages.push({id:'u-ack',role:'user',text:body.content,content:body.content,node_id:null,boundary_available:false},{id:'a-ack',role:'assistant',text:'ack recovered answer',content:'ack recovered answer',node_id:'snap-'+state.revision,boundary_available:true});
        state.durableOperation=body.operation_id;return new Response('{"event":"text","text":"partial"}\n',{headers:{'Content-Type':'application/x-ndjson'}});
      }
      return json({error:'synthetic rejection'},409);
    }
    if(path==='/cow/text/context'&&options.method==='POST'){
      state.contextAttempts++;
      const digest=await crypto.subtle.digest('SHA-256',new TextEncoder().encode(body.operation_id));
      const rootOperation='root-'+Array.from(new Uint8Array(digest),b=>b.toString(16).padStart(2,'0')).join('').slice(0,32);
      if(state.contextAttempts===1){
        state.branch=body.branch;
        state.contextEvents=[{ordinal:1,branch:body.branch,operation_id:rootOperation,type:'fork',
          presentation:{schema:1,namespace_id:body.namespace_id,kind:'root',renderer_id:'f'.repeat(64),
            messages:body.messages,source:null,input_span:{start:0,count:3}},
          result:{name:body.branch,revision:0,tip:'context-root',root:'context-root'}}];
        return json({error:'synthetic response lost after root fork but before durable publication'},503);
      }
      state.branch=body.branch;state.revision=0;state.status='idle';state.rootMode='context_only';
      state.messages=body.messages.map((m,i)=>({id:'context-'+i,role:m.role,text:m.content,
        content:m.content,node_id:null,boundary_available:false}));
      state.contextEvents.push({ordinal:2,branch:body.branch,operation_id:body.operation_id,type:'move',
        presentation:null,result:{name:body.branch,revision:1,tip:'context-root',root:'context-root'}});
      return new Response('{"event":"text","text":"server committed; final acknowledgement lost"}\n',{headers:{'Content-Type':'application/x-ndjson'}});
    }
    if(path==='/cow/text/ask'&&options.method==='POST'){
      state.revision=1;state.status='stopped';state.rootMode='unsupported';
      state.messages.push({id:'question-context',role:'user',text:body.content,content:body.content,node_id:null,boundary_available:false},
        {id:'answer-context',role:'assistant',text:'context answer',content:'context answer',node_id:'context-answer',boundary_available:true});
      return stream({event:'complete',answer:{snapshot:'context-answer',text:'context answer',tip:'context-answer'},receipt:{branch:{name:state.branch,revision:1,tip:'context-answer',root:'context-root'},snapshot:'context-answer'}});
    }
    if(path.startsWith('/cow/presentation?namespace='))return json({events:state.contextEvents});
    if(path.startsWith('/cow/operation?operation='))return state.durableOperation===decodeURIComponent(path.split('operation=')[1])?json({operation_id:state.durableOperation,branch:state.branch,state:'durable',scope:'coordinator-process',durable:true,publication_retryable:false,receipt:{snapshot:'snap-'+state.revision}}):json({error:'not retained'},404);
    if(path.includes('metrics'))return json({});
    if(path==='config'||path==='settings')return json({});
    return json({error:`unmocked ${path}`},404);
  };
  const delay=ms=>new Promise(resolve=>setTimeout(resolve,ms));
  const waitFor=async(predicate,label)=>{for(let i=0;i<100;i++){if(predicate())return;await delay(25);}throw Error(`timed out: ${label}`);};
  const report=(ok,message)=>{let out=document.getElementById('cow-ui-test-report');if(!out){out=document.createElement('pre');out.id='cow-ui-test-report';document.body.append(out);}out.textContent=`${ok?'PASS':'FAIL'} ${message}`;document.title=out.textContent;};
  window.cowNativeUiTest=(async()=>{
    try{
      await waitFor(()=>ContextWorkspace.nativeReady,'native workspace init');
      if(healthAttempts<2)throw Error('failed service discovery was not retried');
      if(messages.length)throw Error('legacy localStorage messages appeared in COW view');
      if($('chat').textContent.includes('legacy local message'))throw Error('legacy localStorage text was displayed as COW history');
      if(calls.some(c=>c.path.startsWith('workspace/')))throw Error('ordinary SQLite workspace route was called');
      $('input').value='first question';await send();
      await waitFor(()=>!busy&&ContextWorkspace.current?.controller_status==='idle'&&$('cow-resume-btn')&&!$('cow-resume-btn').hidden,'prepared prompt recovery state');
      if(!$('cow-resume-btn')||$('cow-resume-btn').hidden||$('cow-resume-btn').textContent!=='Answer prepared prompt')throw Error(`prepared prompt answer control missing: ${$('cow-resume-btn')?.hidden} / ${$('cow-resume-btn')?.textContent} / ${ContextWorkspace.current?.controller_status} / ${JSON.stringify(messages.map(m=>m.role))}`);
      if(state.answerAttempts!==1)throw Error('first answer failure was not exercised');
      $('cow-resume-btn').click();
      await waitFor(()=>!busy&&ContextWorkspace.current?.controller_status==='stopped'&&messages.at(-1)?.text==='answer'&&$('cow-resume-btn').hidden,'recovered answer');
      if(ContextWorkspace.current.messages.at(-1)?.text!=='answer')throw Error('durable answer projection missing');
      const equation={role:'assistant',text:'The identity is $x^2$.',reasoning:'',time:Date.now()};messages.push(equation);renderChat();
      const equationEl=$('chat').lastElementChild;
      $('sampling-btn').click();$('s-math').click();$('s-math-engine').value='mathjax';$('s-math-engine').dispatchEvent(new Event('change'));
      $('s-show').click();$('s-apply').click();
      const savedSampling=JSON.parse(localValues.get('strata.sampling'));
      if(savedSampling.math!==true||savedSampling.mathEngine!=='mathjax'||savedSampling.show!==false)throw Error('native Apply discarded local math/display settings');
      if(!equationEl.querySelector('.chat-math'))throw Error('enabling local math did not repaint the equation');
      if(settings.mathEngine!=='mathjax'||settings.show!==false)throw Error('local rendering settings were not applied');
      $('sampling-btn').click();$('s-math').click();$('s-apply').click();
      const disabledSampling=JSON.parse(localValues.get('strata.sampling'));
      if(disabledSampling.math!==false||disabledSampling.mathEngine!=='mathjax')throw Error('native Apply failed to preserve math disable/engine selection');
      if(equationEl.querySelector('.chat-math')||!equationEl.querySelector('.st-bubble').textContent.includes('$x^2$'))throw Error('disabling local math did not restore plain equation text');
      if(!$('s-esp').disabled)throw Error('unsupported speed projection remained enabled in native mode');
      messages.pop();renderChat();
      $('input').value='ack lost turn';await send();
      await waitFor(()=>!busy&&messages.some(m=>m.role==='assistant'&&m.text==='ack recovered answer'),'receipt lookup after lost acknowledgement');
      if($('input').value!=='')throw Error('a status-confirmed durable request was restored as an unsent draft');
      $('input').value='first question';await send();
      await waitFor(()=>!busy&&$('input').value==='first question','failed repeated question draft restoration');
      if(messages.filter(m=>m.role==='user'&&m.text==='first question').length!==1)throw Error('failed repeated question duplicated or discarded the canonical turn');
      const kinds=calls.filter(c=>c.path.startsWith('/cow/text/')).map(c=>c.path.split('/').at(-1));
      if(kinds.join(',')!=='prepare,answer,answer,append-user,append-user')throw Error(`unexpected COW flow ${kinds.join(',')}`);
      if(!calls.some(c=>c.path.startsWith('/cow/operation?operation=')))throw Error('acknowledgement loss was not checked against operation status');
      rejectWorkspaceList=false;await ContextWorkspace.init();
      attachments=[{kind:'file',name:'src.py',text:'literal <|im_start|> in file'}];renderAttachments();
      $('input').value='independent question';
      rejectSessionWrites=true;
      await ContextWorkspace.saveFilesAsContext();
      if(state.contextAttempts!==0||attachments.length!==1||$('input').value!=='independent question')
        throw Error('context request was submitted without retaining its retry identity');
      attachments[0].text='edited file text after refused submit';
      $('input').value='edited question after refused submit';
      rejectSessionWrites=false;
      await ContextWorkspace.saveFilesAsContext();
      const pending=ContextWorkspace.pendingContextSave;
      if(!pending||state.contextAttempts!==1||attachments.length!==1||$('input').value!=='edited question after refused submit'
        ||!pending.messages[0].content.includes('edited file text after refused submit'))
        throw Error('failed context save did not preserve file and question drafts for retry');
      const op=pending.operation_id,branch=pending.branch;
      ContextWorkspace.pendingContextSave=null;attachments=[];$('input').value='';
      await ContextWorkspace.recoverReusableContext();
      if(ContextWorkspace.pendingContextSave?.operation_id!==op||attachments[0]?.text!=='edited file text after refused submit'
        ||$('input').value!=='edited question after refused submit')throw Error('reload did not restore exact context operation and drafts');
      await ContextWorkspace.saveFilesAsContext();
      if(state.contextAttempts!==2||state.branch!==branch
        ||calls.filter(c=>c.path==='/cow/text/context').map(c=>c.body.operation_id).join(',')!==op+','+op)
        throw Error('context retry changed operation identity or branch');
      const contextBody=calls.filter(c=>c.path==='/cow/text/context').at(-1).body;
      if(contextBody.messages.length!==1||contextBody.messages[0].role!=='user'
        ||contextBody.messages[0].content.includes('edited question after refused submit'))throw Error('context was not user data separate from the question');
      if($('input').value!=='edited question after refused submit'||attachments.length)throw Error('context save consumed the question or retained saved files');
      await send();
      await waitFor(()=>!busy&&state.status==='stopped'&&messages.at(-1)?.text==='context answer','first question on context-only root');
      if(calls.filter(c=>c.path.startsWith('/cow/text/')).map(c=>c.path.split('/').at(-1)).slice(-2).join(',')!=='context,ask')
        throw Error('context-only first question did not use the ask endpoint');
      rejectWorkspaceList=true;await ContextWorkspace.init();
      if(ContextWorkspace.nativeReady||!$('input').disabled||!$('send-btn').disabled)throw Error('native auth failure did not fail closed');
      if(calls.some(c=>c.path.startsWith('workspace/')))throw Error('native auth failure fell back to ordinary workspace');
      report(true,'default prepare/answer unchanged; user-role reusable context, retry/reload identity, suffix ask, fail-closed auth');
    }catch(error){report(false,error.stack||String(error));}
  })();
  window.fetch=window.fetch.bind(window);
  void originalFetch;
})();
