// The server owns durable history. Browser messages are a rendering projection.
"use strict";
async function consumeCowEvents(response,onText){
  const reader=response.body.getReader(),decoder=new TextDecoder();let buffer='',complete=null;
  const handle=line=>{if(!line.trim())return;if(complete)throw Error('COW event arrived after complete.');let event;try{event=JSON.parse(line);}catch{throw Error('Malformed COW event stream.');}
    if(event.event==='error')throw Error(event.error||'COW request failed.');
    if(event.event==='text')onText(event.text||'');
    if(event.event==='complete'){
      const answer=event.answer,receipt=event.receipt,branch=receipt?.branch;
      if(!answer||typeof answer!=='object'||typeof answer.snapshot!=='string'||!receipt||typeof receipt!=='object'
          ||!branch||typeof branch.name!=='string'||!Number.isInteger(branch.revision)||branch.revision<0
          ||typeof branch.tip!=='string'||typeof branch.root!=='string'||typeof receipt.snapshot!=='string'
          ||answer.snapshot!==receipt.snapshot||branch.tip!==receipt.snapshot)
        throw Error('Malformed COW completion receipt.');
      complete=event;
    }
  };
  for(;;){const {value,done}=await reader.read();if(done)break;buffer+=decoder.decode(value,{stream:true});let n;while((n=buffer.indexOf('\n'))>=0){handle(buffer.slice(0,n));buffer=buffer.slice(n+1);}}
  buffer+=decoder.decode();if(buffer.trim())handle(buffer);
  if(!complete)throw Error('Stream ended without a durable COW receipt.');return complete;
}
if(typeof window!=='undefined')window.CowProtocol={consumeCowEvents};
const ContextWorkspace = {
  enabled:false,nativeMode:false,nativeReady:false,current:null,currentChat:null,chats:[],branches:[],loading:false,drafts:new Map(),selectionEpoch:0,
  icons:{
    fork:'<path d="M7 19V5m0 7c0-5 10-1 10-7"/><path d="m14 7 3-3 3 3"/><circle cx="7" cy="20" r="2"/><circle cx="7" cy="4" r="2"/>',
    rewind:'<path d="M4 10a8 8 0 1 1 1 9M4 4v6h6"/>',
    add:'<path d="M12 5v14M5 12h14"/>',
    minus:'<path d="M5 12h14"/>',
    fit:'<path d="M8 4H4v4m12-4h4v4M4 16v4h4m12-4v4h-4"/>',
    export:'<path d="M12 3v12m-4-4 4 4 4-4M4 15v5h16v-5"/>',
    pencil:'<path d="m4 16 12-12 4 4L8 20H4zM13 7l4 4"/>',
    name:'<path d="M3 4h9l9 9-8 8-10-10z"/><circle cx="7" cy="8" r="1"/><path d="m12 12 4 4"/>',
    clear:'<path d="M4 7h16M9 7V4h6v3M6 7l1 14h10l1-14M10 11v6M14 11v6"/>',
    archive:'<path d="m7 7 10 10M17 7 7 17"/>',
    unarchive:'<path d="M5 12h14m-6-6 6 6-6 6"/>',
    teach:'<path d="M3 5h7l2 2 2-2h7v14h-7l-2 2-2-2H3zM12 7v14"/>'
  },
  iconButton(button,label,icon) {
    button.replaceChildren();button.classList.add('cow-icon');button.setAttribute('aria-label',label);button.removeAttribute('title');
    const svg=document.createElementNS('http://www.w3.org/2000/svg','svg');svg.setAttribute('viewBox','0 0 24 24');svg.setAttribute('aria-hidden','true');svg.innerHTML=this.icons[icon];button.append(svg);
    const span=document.createElement('span');span.className='control-label';span.textContent=label;button.append(span);return button;
  },
  appearance() {
    for(const [id,label,icon] of [['new-btn','New chat','add'],['export-btn','Export','export'],['rename-chat-btn','Name chat','pencil'],['rename-branch-btn','Name branch','name'],['clear-branch-btn','Clear branch','clear'],['restore-point-btn','Restore point','rewind'],['teaching-btn','Show control labels','teach']])
      this.iconButton($(id),label,icon);
    const teaching=store.get('workspace.teaching',false);document.documentElement.dataset.teaching=String(teaching);$('teaching-btn').setAttribute('aria-pressed',String(teaching));
    $('teaching-btn').onclick=()=>{const on=document.documentElement.dataset.teaching!=='true';document.documentElement.dataset.teaching=String(on);store.set('workspace.teaching',on);$('teaching-btn').setAttribute('aria-pressed',String(on));};
  },
  async request(path,body) {
    const response=await fetch('workspace/'+path,{method:body===undefined?'GET':'POST',headers:headers(body!==undefined),body:body===undefined?undefined:JSON.stringify(body)});
    let data;try { data=await response.json(); } catch(e) { throw Error('The server did not return a chat response.'); }
    if(!response.ok) { const error=Error(data.error?.message || `HTTP ${response.status}`);error.status=response.status;throw error; }
    return data;
  },
  operation() { return crypto.randomUUID(); },
  nativeOperation() { return `op-${crypto.randomUUID()}`; },
  async init() {
    if(this.loading || busy || !serviceHealthKnown)return;
    this.appearance();
    this.loading=true;
    try {
      if(health.service==='context-cow') { this.nativeMode=true;this.nativeReady=false;await this.initNative(); return; }
      this.nativeMode=false;
      this.chats=(await this.request('chats')).chats;
      this.enabled=true;
      ContextGraph.init();
      for(const id of ['workspace-sidebar','workspace-controls','workspace-state'])$(id).hidden=false;
      $('new-btn').onclick=()=>this.create();
      $('branch-select').onchange=()=>this.select($('branch-select').value).catch(e=>this.error(e));
      $('rename-chat-btn').onclick=()=>this.renameChat();
      $('rename-branch-btn').onclick=()=>this.renameBranch();
      $('clear-branch-btn').onclick=()=>this.clear();
      $('restore-point-btn').onclick=()=>this.rewind($('restore-point').value);
      $('restore-point').onchange=()=>this.savedPointControls();
      $('show-archived').onchange=()=>this.refreshList().catch(e=>this.error(e));
      if(messages.length && !store.get('workspace.imported',false)) {
        let operation=store.get('workspace.import-operation',null);
        if(!operation){operation=this.operation();store.set('workspace.import-operation',operation);}
        const imported=await this.request('import',{title:'Imported browser chat',messages,operation_id:operation});
        store.set('workspace.imported',true);store.set('workspace.branch',imported.branch.id);
        await this.refreshList();
      }
      const remembered=store.get('workspace.branch',null);
      if(remembered) {
        try { await this.select(remembered);return; } catch(e) { if(e.status!==404)throw e; }
      }
      if(this.chats.length)await this.openChat(this.chats[0].id);
      else await this.makeChat('My first chat');
    } catch(e) {
      if(this.nativeMode){this.nativeReady=false;$('send-btn').disabled=true;$('input').disabled=true;$('attach-btn').disabled=true;$('new-btn').disabled=true;$('branch-select').disabled=true;
        toast('error','COW workspace unavailable',e.status===401?'Add the server API key under About > Settings, then save it to retry.':e.message||String(e),7000);
      } else if(e.status!==404)this.error(e);
    } finally {this.loading=false;}
  },
  async nativeRequest(path,body) {
    const response=await fetch('/cow/'+path,{method:body===undefined?'GET':'POST',headers:headers(body!==undefined),body:body===undefined?undefined:JSON.stringify(body)});
    let data;try { data=await response.json(); } catch(e) { throw Error('The COW service did not return JSON.'); }
    if(!response.ok){const error=Error(response.status===401?'This server needs an API key: add it under About > Settings.':data.error?.message||data.error||`HTTP ${response.status}`);error.status=response.status;throw error;}
    return data;
  },
  async initNative() {
    this.nativeMode=true;this.enabled=true;
    $('new-btn').disabled=true;$('branch-select').disabled=true;
    // Native text COW currently guarantees greedy, text-only generation.
    settings.temperature=0;settings.seed='';
    $('sampling-btn').disabled=false;$('sampling-btn').title='COW output limit and local display settings.';
    $('composer-hint').textContent=`COW · greedy text only · output limit ${settings.max||128} tokens · text files only`;
    for(const id of ['s-temp','s-topp','s-topk','s-seed','s-reset']){$(id).disabled=true;$(id).title='This COW endpoint supports greedy text generation only.';}
    for(const button of $('s-thinking').querySelectorAll('button')){button.disabled=true;button.title='Thinking controls are not supported by native COW text generation.';}
    for(const id of ['s-mcp','s-share','s-esp']){$(id).disabled=true;$(id).title='This COW workspace does not expose this action.';}
    this.chats=(await this.nativeRequest('workspace/chats')).chats;
    for(const id of ['workspace-sidebar','workspace-controls','workspace-state'])$(id).hidden=false;
    $('new-btn').onclick=()=>this.createNative();
    $('cow-context-btn').onclick=()=>this.saveFilesAsContext().catch(e=>this.error(e));
    $('branch-select').onchange=()=>this.selectNative($('branch-select').value).catch(e=>this.error(e));
    $('rename-chat-btn').onclick=()=>this.nativeUnsupported('Chat naming');
    $('rename-branch-btn').onclick=()=>this.nativeUnsupported('Branch naming');
    $('clear-branch-btn').onclick=()=>this.nativeUnsupported('Clearing a branch');
    for(const [id,reason] of [['rename-chat-btn','COW namespace rename is not available yet.'],['rename-branch-btn','COW branch rename is not available yet.'],['clear-branch-btn','COW branch clear is not available yet.']]){$(id).disabled=true;$(id).title=reason;}
    $('show-archived').title='COW namespaces do not support archive state yet.';
    $('restore-point-btn').onclick=()=>this.rewindNative($('restore-point').value).catch(e=>this.error(e));
    $('restore-point').onchange=()=>this.savedPointControls();
    $('show-archived').disabled=true;
    $('archive-chat-btn')?.setAttribute('disabled','');
    ContextGraph.init();$('graph-scroll').hidden=false;
    this.ensureResumeButton();
    if(this.chats.length) await this.openChatNative(this.chats[0].id);
    else await this.makeNativeChat('New chat');
    this.nativeReady=true;$('new-btn').disabled=false;$('send-btn').disabled=!!busy;$('input').disabled=false;$('attach-btn').disabled=false;
    this.updateContextButton();
    this.renderNativeList();
    await this.recoverReusableContext();
  },
  updateContextButton(){
    const button=$('cow-context-btn');if(!button)return;
    button.hidden=!this.nativeMode;
    const hasFiles=attachments.some(item=>item.kind==='file'&&typeof item.text==='string');
    button.disabled=!this.nativeReady||!!busy||!hasFiles||attachments.some(item=>item.kind!=='file')
      ||!!this.current?.pending_request||this.current?.controller_status==='active';
    button.textContent=this.pendingContextSave?'Retry context save':'Save files as context';
    button.title=this.pendingContextSave
      ?'Retry the same context operation and branch identity.'
      :'Save files as user-provided context; your question stays in the composer.';
  },
  _contextPendingKey(namespace){return 'cow-reusable-context-pending:'+namespace;},
  _readPendingContext(namespace){
    try{
      const raw=sessionStorage.getItem(this._contextPendingKey(namespace));
      const value=raw?JSON.parse(raw):null;
      return value&&value.namespace_id===namespace&&typeof value.operation_id==='string'
        &&typeof value.branch==='string'&&Array.isArray(value.messages)?value:null;
    }catch{return null;}
  },
  _writePendingContext(value){
    try{sessionStorage.setItem(this._contextPendingKey(value.namespace_id),JSON.stringify(value));}
    catch(error){
      this.updateContextButton();
      throw Error('The browser could not retain this context operation for reload-safe retry. Free session storage space and retry; no request was submitted.');
    }
    this.pendingContextSave=value;
    this.updateContextButton();
  },
  _clearPendingContext(value){
    this.pendingContextSave=null;
    try{sessionStorage.removeItem(this._contextPendingKey(value.namespace_id));}catch{}
    this.updateContextButton();
  },
  async _contextRootReceipt(value){
    const catalog=await this.nativeRequest('presentation?namespace='+encodeURIComponent(value.namespace_id));
    const events=Array.isArray(catalog.events)?catalog.events:[];
    const roots=events.filter(event=>event.type==='fork'&&event.branch===value.branch
      &&event.presentation?.schema===1&&event.presentation?.namespace_id===value.namespace_id
      &&event.presentation?.kind==='root'&&event.result?.name===value.branch
      &&typeof event.result?.tip==='string'&&event.result.tip===event.result.root);
    return events.find(event=>event.operation_id===value.operation_id&&event.branch===value.branch
      &&event.type==='move'&&event.result?.name===value.branch
      &&typeof event.result?.tip==='string'&&event.result.tip===event.result.root
      &&roots.some(root=>root.result.tip===event.result.tip));
  },
  async recoverReusableContext(){
    const value=this._readPendingContext(this.currentChat);
    this.pendingContextSave=value;
    if(!value){this.updateContextButton();return;}
    this.pendingContextSave=value;
    const receipt=await this._contextRootReceipt(value).catch(()=>null);
    if(receipt){
      this._clearPendingContext(value);
      this.drafts.set(value.branch,{text:value.draft_text||'',attachments:[]});
      await this.selectNative(value.branch);
      $('input').value=value.draft_text||'';attachments=[];autosize();renderAttachments();
    }else{
      $('input').value=value.draft_text||'';
      attachments=structuredClone(value.attachments||[]);autosize();renderAttachments();
      this.updateContextButton();
    }
  },
  async saveFilesAsContext(){
    if(!this.nativeReady||!this.currentChat||busy)return;
    if(this.current?.pending_request||this.current?.controller_status==='active'){
      toast('info','Wait for the current request','Finish the active request before saving a reusable context.');return;
    }
    let value=this.pendingContextSave;
    if(!value){
      const files=attachments.filter(item=>item.kind==='file');
      if(!files.length||files.length!==attachments.length||files.some(file=>typeof file.text!=='string')){
        toast('info','Attach text files first','This action saves text files as user-provided context; it does not send a question.');return;
      }
      const operation=this.nativeOperation();
      const content=files.map(file=>userText({text:'',files:[file]})).join('\n\n');
      value={namespace_id:this.currentChat,operation_id:operation,
        branch:'b-'+operation.replaceAll('-',''),messages:[{role:'user',content}],
        draft_text:$('input').value,attachments:structuredClone(files)};
    }
    if(value.namespace_id!==this.currentChat){toast('info','Different chat selected','Open the chat with the pending context save to retry it.');return;}
    try{this._writePendingContext(value);}catch(error){
      toast('error','Context save not submitted',error.message+' Your question and file attachments remain here.',7000);
      return;
    }
    const controller=new AbortController();busy={controller,msg:null};setBusy(true);
    $('stop-btn').hidden=true;$('composer-hint').textContent='Saving context; it finishes before the durable receipt is shown.';
    try{
      const response=await fetch('/cow/text/context',{method:'POST',headers:headers(true),
        body:JSON.stringify({branch:value.branch,namespace_id:value.namespace_id,
          messages:value.messages,operation_id:value.operation_id,stream:true})});
      if(!response.ok){let error='HTTP '+response.status;try{const data=await response.json();error=data.error?.message||data.error||error;}catch{}throw Error(error);}
      const complete=await consumeCowEvents(response,()=>{});
      this._clearPendingContext(value);
      this.drafts.set(value.branch,{text:value.draft_text||'',attachments:[]});
      await this.selectNative(complete.receipt.branch.name,{allowBusy:true});
      $('input').value=value.draft_text||'';attachments=[];autosize();renderAttachments();
      toast('info','Reusable context saved','Your question remains in the composer. Send it to ask on this code-only branch.');
    }catch(error){
      const receipt=await this._contextRootReceipt(value).catch(()=>null);
      if(receipt){
        this._clearPendingContext(value);
        this.drafts.set(value.branch,{text:value.draft_text||'',attachments:[]});
        await this.selectNative(value.branch,{allowBusy:true});
        $('input').value=value.draft_text||'';attachments=[];autosize();renderAttachments();
        toast('info','Context save confirmed','The canonical root was saved; your question remains in the composer.');
      }else{
        this.pendingContextSave=value;
        $('input').value=value.draft_text||'';attachments=structuredClone(value.attachments||[]);
        autosize();renderAttachments();
        toast('error','Context save not confirmed',(error.message||String(error))+' The exact operation is retained for retry; your draft is unchanged.',7000);
      }
    }finally{
      busy=null;setBusy(false);$('stop-btn').hidden=true;
      this.updateContextButton();
      try{await this.refreshNativeList();}catch{}
    }
  },
  nativeUnsupported(label) { toast('info',`${label} unavailable`,'This COW workspace does not expose that durable action yet.'); },
  ensureResumeButton() {
    if(!$('cow-resume-btn')){
      const button=document.createElement('button');button.id='cow-resume-btn';button.type='button';button.className='st-btn st-btn--secondary';
      button.textContent='Resume answer';button.hidden=true;button.onclick=()=>this.resumeNative().catch(e=>this.error(e));$('workspace-state').append(button);
    }
    if(!$('cow-save-retry-btn')){
      const retry=document.createElement('button');retry.id='cow-save-retry-btn';retry.type='button';retry.className='st-btn st-btn--secondary';
      retry.textContent='Finish saving';retry.hidden=true;retry.onclick=()=>this.retryNativeSave().catch(e=>this.error(e));$('workspace-state').append(retry);
    }
  },
  async retryNativeSave(){
    const pending=this.current?.pending_request;
    if(!pending?.publication_retryable||!pending.operation_id)return;
    const button=$('cow-save-retry-btn');button.disabled=true;
    try{await this.nativeRequest('publication/retry',{operation_id:pending.operation_id});await this.selectNative(this.current.id);}
    finally{button.disabled=false;}
  },
  async refreshNativeList(epoch=this.selectionEpoch) {
    const data=await this.nativeRequest('workspace/chats');if(epoch!==this.selectionEpoch)return;
    this.chats=data.chats;this.renderNativeList();
  },
  renderNativeList() {
    const list=$('chat-list');list.replaceChildren();
    for(const chat of this.chats){const button=document.createElement('button');button.type='button';button.className='workspace-chat';button.setAttribute('aria-current',String(this.current?.chat_id===chat.id));
      const title=document.createElement('strong');title.textContent=chat.title;button.append(title);const small=document.createElement('small');small.textContent=`${chat.branch_count} ${chat.branch_count===1?'branch':'branches'} · ${chat.max_saved_messages} messages`;button.append(small);
      button.onclick=()=>this.openChatNative(chat.id).catch(e=>this.error(e));const row=document.createElement('div');row.className='workspace-chat-row';row.append(button);
      const archive=document.createElement('button');archive.type='button';archive.className='st-btn workspace-chat-archive';archive.disabled=true;this.iconButton(archive,'Archive unavailable','archive');archive.title='COW namespaces do not support archiving yet.';row.append(archive);list.append(row);}
  },
  async makeNativeChat(title) {
    this.saveDraft();
    const ns=await this.nativeRequest('namespace',{title,operation_id:this.nativeOperation()});
    this.currentChat=ns.namespace_id||ns.id;this.current=null;this.messagesReset();
    this.pendingContextSave=this._readPendingContext(this.currentChat);
    await this.refreshNativeList();this.showNativeEmpty();
    this.updateContextButton();
  },
  createNative() { this.modal('New chat','Your current chat stays in the list.','New chat',name=>this.makeNativeChat(name),'Create chat'); },
  messagesReset(){messages=[];renderChat();},
  showNativeEmpty(){
    messages=[];renderChat();
    ContextGraph.beginChat(this.currentChat);$('graph-summary').textContent='No fork yet';
    $('graph-status').textContent='Send a message to start this history.';$('graph-refresh').hidden=true;
    const draft=this.drafts.get(`namespace:${this.currentChat}`);$('input').value=draft?.text||'';attachments=structuredClone(draft?.attachments||[]);autosize();renderAttachments();
    $('workspace-title').textContent=this.chats.find(c=>c.id===this.currentChat)?.title||'New chat';
    $('workspace-save-state').textContent='Empty namespace · no branch created until first message';
    $('branch-select').replaceChildren();$('branch-select').disabled=true;$('rename-branch-btn').disabled=true;$('clear-branch-btn').disabled=true;
    $('rename-chat-btn').disabled=true;$('restore-point').replaceChildren();const o=document.createElement('option');o.value='';o.textContent='Saved points';$('restore-point').append(o);this.savedPointControls();
    $('cow-resume-btn').hidden=true;this.renderNativeList();
  },
  async openChatNative(chatId){
    if(busy)throw Error('Wait for the answer before switching chats.');
    this.saveDraft();
    this.currentChat=chatId;const epoch=++this.selectionEpoch;const data=await this.nativeRequest('workspace/branches?chat='+encodeURIComponent(chatId));if(epoch!==this.selectionEpoch)return;this.branches=data.branches||[];
    if(!this.branches.length){this.current=null;this.messagesReset();this.showNativeEmpty();await this.recoverReusableContext();return;}
    await this.selectNative(this.branches[0].id);
    await this.recoverReusableContext();
  },
  async selectNative(branch,{allowBusy=false}={}){
    if(busy&&!allowBusy)throw Error('Wait for the answer before switching branches.');
    this.saveDraft();const epoch=++this.selectionEpoch;const history=await this.nativeRequest('workspace/history?branch='+encodeURIComponent(branch));if(epoch!==this.selectionEpoch)return;
    ContextGraph.beginChat(history.chat_id);
    this.current=history;this.currentChat=history.chat_id;messages=history.messages||[];
    const draft=this.drafts.get(branch);$('input').value=draft?.text||'';attachments=structuredClone(draft?.attachments||[]);autosize();renderAttachments();renderChat();
    const branchData=await this.nativeRequest('workspace/branches?chat='+encodeURIComponent(history.chat_id));if(epoch!==this.selectionEpoch)return;this.branches=branchData.branches||[];
    const selector=$('branch-select');selector.replaceChildren();for(const b of this.branches){const o=document.createElement('option');o.value=b.id;o.textContent=b.name;selector.append(o);}selector.value=branch;selector.disabled=!!history.pending_request;
    $('workspace-title').textContent=this.chats.find(c=>c.id===history.chat_id)?.title||'Chat';$('rename-chat-btn').disabled=true;$('rename-branch-btn').disabled=true;$('clear-branch-btn').disabled=true;
    $('workspace-save-state').textContent=`COW ${history.controller_status||'idle'} · ${messages.length} projected messages`;$('workspace-save-state').title='Only canonical COW snapshots and receipts are shown as saved history.';
    const points=$('restore-point');points.replaceChildren();const prompt=document.createElement('option');prompt.value='';prompt.textContent='Saved points';points.append(prompt);
    for(const point of (history.saved_points||[]).filter(p=>p.boundary_available&&p.on_selected_ancestry)){const o=document.createElement('option');o.value=point.snapshot_id;o.textContent=`${point.kind} · rev ${point.revision}${point.current?' · current':''}`;points.append(o);}
    this.savedPointControls();const prepared=history.initial_root_mode==='prepared_answer';
    const pending=history.pending_request;
    $('cow-save-retry-btn').hidden=!pending?.publication_retryable;
    $('cow-save-retry-btn').disabled=!!pending&&!pending.publication_retryable;
    if(pending&&!pending.publication_retryable){$('cow-save-retry-btn').hidden=false;$('cow-save-retry-btn').textContent=pending.state==='persisting'?'Saving checkpoint…':'Request in progress…';}
    $('cow-resume-btn').hidden=!!pending||!(history.controller_status==='active'||prepared);$('cow-resume-btn').textContent=prepared?'Answer prepared prompt':'Resume answer';
    await this.refreshNativeList(epoch);
    if(epoch===this.selectionEpoch)await ContextGraph.refresh();
  },
  async sendNative(){
    if(!this.nativeReady){toast('error','COW workspace unavailable','Fix the connection or API key to retry; no message was sent.');return;}
    const input=$('input'),raw=input.value.trim();if((!raw&&!attachments.length)||busy)return;
    if(this.current?.pending_request){toast('info','Request still being saved','Finish saving or wait for the existing COW request before sending another message; your draft is unchanged.');return;}
    if(this.current?.controller_status==='active'){toast('info','Answer still active','Resume the current answer before asking another question; your draft is unchanged.');return;}
    if(this.current?.initial_root_mode==='prepared_answer'){
      toast('info','Prepared prompt needs an answer','Use Answer prepared prompt first; your draft is unchanged.');return;
    }
    if(attachments.some(a=>a.kind!=='file')){toast('error','Images unavailable','Remove image attachments before sending; your draft is still here.');return;}
    if(settings.temperature!==0||settings.seed){toast('error','Unsupported COW options','COW mode supports greedy text only. Your draft is still here.');return;}
    if(!this.readNativeMaxTokens())return;
    if(attachments.some(a=>a.text==null)){toast('error','Unsupported attachment','COW mode accepts text files only; this attachment is unavailable. Your draft is still here.');return;}
    let content=raw;for(const file of attachments)content+=(content?'\n\n':'')+userText({text:'',files:[file]});
    await this.runNative(content,{consumeDraft:true});
  },
  readNativeMaxTokens(){
    const value=String(settings.max||'').trim(),number=value?Number(value):128;
    if(!Number.isSafeInteger(number)||number<1||number>8192){toast('error','Invalid output limit','Choose a whole number from 1 to 8192 tokens. Your draft is unchanged.');return null;}
    return number;
  },
  async resumeNative(){if(!this.current)return;if(this.current.pending_request){toast('info','Request still being saved','Wait for the existing COW request to finish before resuming.');return;}if(!this.readNativeMaxTokens())return;await this.runNative(null,{resume:true});},
  async runNative(content,{consumeDraft=false,resume=false}={}){
    if(busy)return;if(!this.currentChat)throw Error('Create a chat first.');
    const maxTokens=this.readNativeMaxTokens();if(!maxTokens)return;
    const preparedBefore=this.current?.initial_root_mode==='prepared_answer';
    const draftText=$('input').value,draftAttachments=structuredClone(attachments);let durable=false;
    const user=content!==null?{role:'user',text:content,time:Date.now()}:null;
    if(consumeDraft){messages=[...messages,user];attachments=[];renderAttachments();$('input').value='';autosize();}
    const m={role:'assistant',text:'',reasoning:'',time:Date.now()};messages=[...messages,m];renderChat();let activeEl=$('chat').lastElementChild;
    const controller=new AbortController();busy={controller,msg:m};setBusy(true);$('cow-resume-btn').hidden=true;
    const op=this.nativeOperation();let activeOperation=op;
    const request=async(path,body)=>{
      activeOperation=body.operation_id||op;
      const response=await fetch('/cow/text/'+path,{method:'POST',headers:headers(true),body:JSON.stringify({...body,operation_id:activeOperation,stream:true}),signal:controller.signal});
      if(!response.ok){let error=`HTTP ${response.status}`;try{const j=await response.json();error=j.error?.message||j.error||error;}catch{}throw Error(error);}
      const complete=await consumeCowEvents(response,text=>{m.text+=text;updateAssistant(activeEl,m,true);scrollDown();});durable=true;return complete;
    };
    try{
      let result;
      if(!this.current){const branch='b-'+op.replaceAll('-','');const full=messages.filter(x=>x!==m).map(x=>({role:x.role,content:userText(x)}));
        result=await request('prepare',{branch,namespace_id:this.currentChat,messages:full,operation_id:op});
        durable=true;
        const branchName=result.receipt.branch.name;
        await this.selectNative(branchName,{allowBusy:true});
        messages=[...messages,m];renderChat();activeEl=$('chat').lastElementChild;
        result=await request('answer',{branch:branchName,revision:this.current.revision,max_tokens:maxTokens,settings:{temperature:0},operation_id:this.nativeOperation()});
      }else if(resume&&preparedBefore){result=await request('answer',{branch:this.current.id,revision:this.current.revision,max_tokens:maxTokens,settings:{temperature:0},operation_id:op});}
      else if(resume){result=await request('resume',{branch:this.current.id,revision:this.current.revision,max_tokens:maxTokens,operation_id:op});}
      else if(this.current.controller_status==='active'){throw Error('This answer is active. Use Resume answer before asking a new question.');}
      else if(this.current.controller_status==='stopped'){result=await request('append-user',{branch:this.current.id,revision:this.current.revision,content,max_tokens:maxTokens,settings:{temperature:0},operation_id:op});}
      else if(preparedBefore){throw Error('Prepared prompts must be answered with the explicit answer control.');}
      else {result=await request('ask',{branch:this.current.id,revision:this.current.revision,content,max_tokens:maxTokens,settings:{temperature:0},operation_id:op});}
      durable=true;
      m.text=result.answer?.text??m.text;m.node_id=result.answer?.tip||result.receipt?.snapshot||null;m.boundary_available=!!m.node_id;updateAssistant(activeEl,m,false);
    }catch(e){
      m.error=e.message||String(e);
      try{const status=await this.nativeRequest('operation?operation='+encodeURIComponent(activeOperation));if(status.durable){durable=true;if(!this.current&&status.branch)await this.selectNative(status.branch,{allowBusy:true});}}
      catch{}
      toast('error','COW request incomplete',m.error,7000);
    }
    busy=null;setBusy(false);updateAssistant(activeEl,m,false);
    try{await this.refreshNativeList();if(this.current)await this.selectNative(this.current.id);else this.showNativeEmpty();}
    catch(e){this.error(e);}
    if(!durable&&content!==null){$('input').value=draftText;attachments=draftAttachments;autosize();renderAttachments();}
  },
  error(error) {toast('error','Chat action failed',error.message,7000);},
  modal(title,help,value,action,button='Save') {
    if(busy){toast('info','Wait for this answer','Stop or finish it before changing branches.');return;}
    const dialog=$('workspace-dialog');$('workspace-dialog-title').textContent=title;$('workspace-dialog-help').textContent=help;
    $('workspace-name').hidden=value===null;$('workspace-name-label').hidden=value===null;
    $('workspace-name').value=value || '';$('workspace-confirm').textContent=button;
    $('workspace-cancel').onclick=()=>dialog.close();
    $('workspace-confirm').onclick=async()=>{
      const name=$('workspace-name').value.trim();
      if(value!==null&&!name){$('workspace-name').focus();return;}
      $('workspace-confirm').disabled=true;
      try {await action(name);dialog.close();}catch(e){this.error(e);}finally{$('workspace-confirm').disabled=false;}
    };
    $('workspace-name').onkeydown=e=>{if(e.key==='Enter'){e.preventDefault();$('workspace-confirm').click();}};
    dialog.showModal();if(value!==null){$('workspace-name').focus();$('workspace-name').select();}
  },
  async refreshList() {
    const epoch=this.listEpoch=(this.listEpoch || 0)+1;
    const chats=(await this.request('chats'+($('show-archived').checked?'?archived=true':''))).chats;
    if(epoch!==this.listEpoch)return;
    this.chats=chats;
    const list=$('chat-list');list.replaceChildren();
    for(const chat of this.chats) {
      const button=document.createElement('button');button.type='button';button.className='workspace-chat';button.setAttribute('aria-current',String(this.current?.chat_id===chat.id));
      const title=document.createElement('strong');title.textContent=chat.title;button.append(title);
      const subtitle=document.createElement('small');subtitle.textContent=`${chat.branch_count} ${chat.branch_count===1?'branch':'branches'} · ${chat.max_saved_messages} messages${chat.archived?' · Archived':''}`;button.append(subtitle);
      button.onclick=()=>this.openChat(chat.id).catch(e=>this.error(e));
      const row=document.createElement('div');row.className='workspace-chat-row';row.append(button);
      const archive=document.createElement('button');archive.type='button';archive.className='st-btn workspace-chat-archive';archive.dataset.archiveChat=chat.id;
      if(this.current?.chat_id===chat.id)archive.id='archive-chat-btn';
      this.iconButton(archive,chat.archived?'Unarchive chat':'Archive chat',chat.archived?'unarchive':'archive');
      archive.onclick=()=>this.archive(chat).catch(e=>this.error(e));row.append(archive);list.append(row);
    }
  },
  async openChat(chat) {
    if(busy)throw Error('Wait for the answer before switching chats.');
    if(this.moving)throw Error('Wait for the branch change before switching chats.');
    const epoch=++this.selectionEpoch;
    const branches=(await this.request('branches?chat='+encodeURIComponent(chat))).branches;
    if(epoch!==this.selectionEpoch)return;
    if(!branches.length)throw Error('This chat has no retained branch.');
    await this.select(branches[0].id);
  },
  saveDraft() {
    if(this.nativeMode&&!this.current&&this.currentChat)this.drafts.set(`namespace:${this.currentChat}`,{text:$('input').value,attachments:structuredClone(attachments)});
    else if(this.current)this.drafts.set(this.current.id,{text:$('input').value,attachments:structuredClone(attachments)});
  },
  savedPoints(graph) {
    if(!this.current || !Array.isArray(this.current.moves))return;
    const nodes=new Map(graph.nodes.map(n=>[n.id,n])),points=$('restore-point'),previous=points.value;
    points.replaceChildren();const prompt=document.createElement('option');prompt.value='';prompt.textContent='Saved points';points.append(prompt);
    const added=new Set();
    for(const move of this.current.moves){
      if(added.has(move.from_node))continue;added.add(move.from_node);
      const option=document.createElement('option');option.value=move.from_node;const node=nodes.get(move.from_node);
      option.textContent=move.from_node===graph.root_id?'Start of chat':node?`${node.depth} ${node.depth===1?'message':'messages'} · ${(node.preview || node.role).replace(/\s+/g,' ').slice(0,42)}`:`Saved ${new Date(move.at*1000).toLocaleString(undefined,{month:'short',day:'numeric',hour:'numeric',minute:'2-digit',second:'2-digit'})}`;
      points.append(option);
    }
    if(added.has(previous))points.value=previous;
    this.savedPointControls();
  },
  savedPointControls() {
    if(this.nativeMode){const available=!!this.current&&$('restore-point').options.length>1;$('restore-point').disabled=!available||!!busy;$('restore-point-btn').disabled=!available||!!busy||!$('restore-point').value;return;}
    $('restore-point').disabled=this.moving || !!busy || $('restore-point').options.length<2;
    $('restore-point-btn').disabled=this.moving || !!busy || !$('restore-point').value;
  },
  async select(branch) {
    if(this.nativeMode)return this.selectNative(branch);
    if(busy)throw Error('Wait for the answer before switching branches.');
    const epoch=++this.selectionEpoch;
    this.saveDraft();
    const history=await this.request('history?branch='+encodeURIComponent(branch));
    if(epoch!==this.selectionEpoch)return;
    const [branchList,catalogData]=await Promise.all([
      this.request('branches?chat='+encodeURIComponent(history.chat_id)),this.request('chats?archived=true')]);
    if(epoch!==this.selectionEpoch)return;
    this.saveDraft();
    ContextGraph.beginChat(history.chat_id);
    this.current=history;messages=history.messages;
    const draft=this.drafts.get(branch);$('input').value=draft?.text || '';attachments=structuredClone(draft?.attachments || []);
    autosize();renderAttachments();renderChat();
    store.set('workspace.branch',branch);
    this.branches=branchList.branches;
    const selector=$('branch-select');selector.replaceChildren();
    for(const b of this.branches){const option=document.createElement('option');option.value=b.id;option.textContent=b.name;selector.append(option);}selector.value=branch;
    // Include archived metadata without forcing the archived list open.
    const catalog=catalogData.chats;
    const chat=catalog.find(c=>c.id===history.chat_id);$('workspace-title').textContent=chat?.title || 'Chat';
    $('workspace-save-state').textContent=history.pending_request?'Answer in progress in another tab':`${messages.length} saved ${messages.length===1?'message':'messages'}`;
    $('workspace-save-state').title='Serving receives the selected history and may reuse its prompt/conversation checkpoints. Native GPU COW handles are not connected to this interface yet.';
    const points=$('restore-point');points.replaceChildren();const prompt=document.createElement('option');prompt.value='';prompt.textContent='Saved points';points.append(prompt);
    const added=new Set();
    for(const move of history.moves){if(added.has(move.from_node))continue;added.add(move.from_node);const option=document.createElement('option');option.value=move.from_node;option.textContent='Before change '+move.revision;points.append(option);}
    await this.refreshList();
    if(epoch!==this.selectionEpoch)return;
    await ContextGraph.refresh();
  },
  async makeChat(title) {
    const chat=await this.request('chats/create',{title,operation_id:this.operation()});await this.select(chat.branch.id);
  },
  create() {this.modal('New chat','Your current chat stays in the list.','New chat',name=>this.makeChat(name),'Create chat');},
  async direct(action) {
    if(busy){toast('info','Wait for this answer','Stop or finish it before changing branches.');return;}
    if(this.moving)return;
    this.moving=true;
    ContextGraph.disableActions();
    const controls=['branch-select','clear-branch-btn','restore-point','restore-point-btn'].map($);
    controls.forEach(control=>control.disabled=true);
    try {
      if(ContextGraph.stale)throw Error('Refresh paths before changing this branch.');
      await action();
    } catch(e){
      if(e.status===409 && this.current){try {await this.select(this.current.id);} catch(refreshError){ContextGraph.failed(refreshError);}}
      this.error(e);
    } finally {this.moving=false;controls.forEach(control=>control.disabled=false);this.savedPointControls();ContextGraph.disableActions();}
  },
  fork(node) {
    if(this.nativeMode)return this.forkNative(node);
    return this.direct(async()=>{
      const names=new Set(this.branches.map(b=>b.name));let number=1;
      while(names.has(`Alternative ${number}`))number++;
      const name=`Alternative ${number}`;
      const branch=await this.request('branches/fork',{branch_id:this.current.id,node_id:node,name,operation_id:this.operation()});await this.select(branch.id);
      $('input').focus();
    });
  },
  rewind(node) {
    if(this.nativeMode)return this.rewindNative(node);
    if(!node)return;
    return this.direct(async()=>{
      const branch=await this.request('branches/rewind',{branch_id:this.current.id,revision:this.current.revision,node_id:node,operation_id:this.operation()});await this.select(branch.id);
      $('input').focus();
    });
  },
  async forkNative(snapshot) {
    if(!this.current||!snapshot||busy||this.current.pending_request)return;
    const operation=this.nativeOperation(),name=`b-${operation.slice(3).replaceAll('-','')}`;
    const result=await this.nativeRequest('fork',{source_branch:this.current.id,snapshot,name,operation_id:operation});
    const branch=result.branch?.name||result.branch?.id||result.name;
    if(!branch)throw Error('Fork response did not include the new branch name.');
    await this.refreshNativeList();await this.selectNative(branch);$('input').focus();
  },
  async rewindNative(snapshot) {
    if(!this.current||!snapshot||busy||this.current.pending_request)return;
    await this.nativeRequest('rewind',{branch:this.current.id,revision:this.current.revision,snapshot,operation_id:this.nativeOperation()});
    await this.selectNative(this.current.id);$('input').focus();
  },
  clear() {
    return this.direct(async()=>{
      const branch=await this.request('branches/clear',{branch_id:this.current.id,revision:this.current.revision,operation_id:this.operation()});await this.select(branch.id);
      $('input').focus();
    });
  },
  renameChat() {this.modal('Name this chat','Use a name you can find in the chat list.',$('workspace-title').textContent,async title=>{await this.request('chats/rename',{chat_id:this.current.chat_id,title});await this.select(this.current.id);});},
  renameBranch() {this.modal('Name this branch','The name does not change its history.',this.current.name,async name=>{await this.request('branches/rename',{branch_id:this.current.id,name});await this.select(this.current.id);});},
  async archive(chat) {
    if(busy)throw Error('Wait for the answer before archiving a chat.');
    await this.request('chats/archive',{chat_id:chat.id,archived:!chat.archived});await this.refreshList();
    if(chat.id===this.current?.chat_id)await ContextGraph.refresh();
    toast('info',chat.archived?'Chat restored':'Chat archived','Its saved messages remain in SQLite.');
  },
  messageActions(message) {
    if(!this.enabled||!message.node_id)return null;
    if(this.nativeMode){
      if(!message.boundary_available||this.current?.pending_request)return null;
      const actions=document.createElement('div');actions.className='workspace-message-actions';
      for(const [text,kind,callback] of [['Fork here','fork',()=>this.forkNative(message.node_id).catch(e=>this.error(e))],['Retry here','rewind',()=>this.rewindNative(message.node_id).catch(e=>this.error(e))]]) {
        if(kind==='fork'&&message.fork_available===false)continue;
        const button=document.createElement('button');button.type='button';button.className='st-btn st-btn--secondary';this.iconButton(button,text,kind);button.dataset['workspace'+kind[0].toUpperCase()+kind.slice(1)]=message.node_id;button.onclick=callback;actions.append(button);
      }
      return actions;
    }
    const actions=document.createElement('div');actions.className='workspace-message-actions';
    for(const [text,kind,callback] of [['Fork here','fork',()=>this.fork(message.node_id)],['Retry here','rewind',()=>this.rewind(message.node_id)]]) {
      const button=document.createElement('button');button.type='button';button.className='st-btn st-btn--secondary';this.iconButton(button,text,kind);button.dataset['workspace'+kind[0].toUpperCase()+kind.slice(1)]=message.node_id;
      button.onclick=callback;actions.append(button);
    }
    return actions;
  },
  generation(body,user) {
    if(!this.enabled)return 'v1/chat/completions';
    if(!this.current)throw Error('Select a chat before sending a message.');
    Object.assign(body,{branch_id:this.current.id,revision:this.current.revision,message:user,operation_id:this.operation()});delete body.messages;
    $('workspace-save-state').textContent='Answer in progress · not saved yet';return 'workspace/generate';
  },
  async afterGeneration() {
    if(!this.enabled)return;
    await this.select(this.current.id);
  }
};
