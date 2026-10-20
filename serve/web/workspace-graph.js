"use strict";
// A transient view of canonical history. No messages or branch heads are stored here.
const ContextGraph = {
  data:null,epoch:0,stale:false,timer:null,initialized:false,zoom:null,scale:1,
  init() {
    if(this.initialized)return;this.initialized=true;
    const viewer=$('branch-viewer');viewer.hidden=false;viewer.open=store.get('workspace.graph-open',true);
    viewer.ontoggle=()=>store.set('workspace.graph-open',viewer.open);
    $('graph-previous').checked=store.get('workspace.graph-previous',false);
    $('graph-previous').onchange=()=>{store.set('workspace.graph-previous',$('graph-previous').checked);this.refresh();};
    $('graph-refresh').onclick=()=>ContextWorkspace.select(ContextWorkspace.current.id).catch(e=>this.failed(e));
    $('graph-zoom-in').onclick=()=>{this.zoom=Math.min(2,this.scale*1.25);this.render();};
    $('graph-zoom-out').onclick=()=>{this.zoom=Math.max(.1,this.scale/1.25);this.render();};
    $('graph-fit').onclick=()=>{this.zoom='fit';this.render();};
    for(const [id,label,icon] of [['graph-zoom-in','Zoom in','add'],['graph-zoom-out','Zoom out','minus'],['graph-fit','Fit paths','fit']])ContextWorkspace.iconButton($(id),label,icon);
    const scroll=$('graph-scroll');let pan=null;
    scroll.onpointerdown=e=>{
      if(e.pointerType!=='mouse' || e.button!==0 || e.target.closest('button'))return;
      pan={x:e.clientX,y:e.clientY,left:scroll.scrollLeft,top:scroll.scrollTop};scroll.setPointerCapture(e.pointerId);scroll.classList.add('is-panning');
    };
    scroll.onpointermove=e=>{if(pan){scroll.scrollLeft=pan.left-(e.clientX-pan.x);scroll.scrollTop=pan.top-(e.clientY-pan.y);}};
    scroll.onpointerup=scroll.onpointercancel=()=>{pan=null;scroll.classList.remove('is-panning');};
    this.observer=new ResizeObserver(()=>{
      const width=$('graph-scroll').clientWidth,height=$('graph-scroll').clientHeight;
      if(this.data && width>0 && (width!==this.viewportWidth || height!==this.viewportHeight))this.render();
    });this.observer.observe($('graph-scroll'));
  },
  failed(error) {
    this.stale=true;$('graph-status').textContent='Paths need refreshing: '+error.message;$('graph-refresh').hidden=false;
    this.disableActions();
  },
  beginChat(chat) {
    if(this.data?.chat_id===chat)return;
    this.epoch++;this.data=null;this.zoom=null;this.stale=true;this.focusLayoutKey=null;
    if(this.motionFrame)cancelAnimationFrame(this.motionFrame);
    this.motionEpoch=(this.motionEpoch || 0)+1;this.pathGeometry=new Map();
    $('graph-canvas').replaceChildren();$('graph-canvas').dataset.motion='settled';$('graph-inspector').hidden=true;
    $('graph-summary').textContent='Loading saved history…';$('graph-status').textContent='';
  },
  disableActions() {
    for(const button of $('graph-inspector').querySelectorAll('.workspace-message-actions button'))button.disabled=this.stale || !!busy || !!ContextWorkspace.moving;
    for(const button of $('graph-canvas').querySelectorAll('[data-graph-branch]'))button.disabled=this.stale || !!busy || !!ContextWorkspace.moving;
  },
  async refresh() {
    const current=ContextWorkspace.current;if(!current)return;
    const epoch=++this.epoch;const branch=current.id;const chat=current.chat_id;
    if(this.data?.chat_id!==chat){this.data=null;this.zoom=null;$('graph-canvas').replaceChildren();$('graph-inspector').hidden=true;}
    try {
      const path='graph?chat='+encodeURIComponent(chat)+($('graph-previous').checked?'&previous=true':'');
      const data=await (ContextWorkspace.nativeMode?ContextWorkspace.nativeRequest('workspace/'+path):ContextWorkspace.request(path));
      if(epoch!==this.epoch || ContextWorkspace.current?.id!==branch)return;
      this.data=data;
      const head=data.branches.find(b=>b.id===branch);
      this.stale=!busy && (!head || head.revision!==ContextWorkspace.current.revision);
      $('graph-status').textContent=this.stale?'This branch changed. Refresh paths before editing.':'';
      $('graph-refresh').hidden=!this.stale;
      $('graph-inspector').hidden=true;this.render();
    } catch(e) {if(epoch===this.epoch && ContextWorkspace.current?.id===branch)this.failed(e);}
  },
  setBusy(on) {
    if(!ContextWorkspace.enabled)return;
    ContextWorkspace.savedPointControls();
    if(this.timer){clearInterval(this.timer);this.timer=null;}
    if(on)this.timer=setInterval(()=>this.refresh(),2000);
    if(this.data)this.render();
    this.disableActions();
  },
  model(data) {
    const nodes=new Map(data.nodes.map(n=>[n.id,n]));const children=new Map(data.nodes.map(n=>[n.id,[]]));
    for(const node of data.nodes)if(node.parent_id)children.get(node.parent_id).push(node.id);
    const tips=new Set(data.branches.map(b=>b.tip));for(const p of data.previous_tips)tips.add(p.node_id);
    const critical=new Set(data.nodes.filter(n=>n.parent_id===null || children.get(n.id).length!==1 || tips.has(n.id)).map(n=>n.id));
    // Branch references at an identical tip share one group and one graph point.
    const groups=new Map(),badgePositions=new Map();
    const group=(tip,key)=>{if(!groups.has(tip))groups.set(tip,[]);groups.get(tip).push(key);};
    data.branches.forEach(b=>group(b.tip,b.id));data.previous_tips.forEach((p,i)=>group(p.node_id,'previous-'+i));
    const lane=new Map();let groupIndex=0;
    for(const tip of groups.keys())lane.set(tip,groupIndex++);
    for(let i=data.nodes.length-1;i>=0;i--){const n=data.nodes[i];for(const child of children.get(n.id))lane.set(n.id,Math.min(lane.get(n.id)??Infinity,lane.get(child)??Infinity));}
    for(const list of children.values())list.sort((a,b)=>(lane.get(a)??Infinity)-(lane.get(b)??Infinity) || a.localeCompare(b));
    const anchors=[data.root_id],edges=[];
    for(const id of anchors){
      for(const first of children.get(id)){
        const chain=[first];let end=first;
        while(!critical.has(end)){end=children.get(end)[0];chain.push(end);}
        anchors.push(end);edges.push({from:id,to:end,chain});
      }
    }
    const selected=new Set();let tip=data.branches.find(b=>b.id===ContextWorkspace.current?.id)?.tip;
    while(tip && nodes.has(tip)){selected.add(tip);tip=nodes.get(tip).parent_id;}
    const position=new Map();let band=16,maxRail=28;
    for(const id of anchors){
      const refs=groups.get(id) || [],x=28+24*(lane.get(id)||0),y=band+(refs.length?52:24);
      maxRail=Math.max(maxRail,x);position.set(id,{x,y});
      refs.forEach((key,i)=>badgePositions.set(key,{top:band+29+i*52,y:band+52+i*52}));
      band+=refs.length?refs.length*52+42:id===data.root_id?44:64;
    }
    const badgeX=maxRail+24;
    return {nodes,children,anchors,edges,position,selected,badgePositions,groups,vertical:true,badgeX,width:badgeX+205+24,height:band};
  },
  render() {
    if(!this.data)return;
    const data=this.data,model=this.model(data),canvas=$('graph-canvas');
    if(this.motionFrame)cancelAnimationFrame(this.motionFrame);
    const previousPaths=this.pathGeometry || new Map();this.pathGeometry=new Map();
    const motionEpoch=this.motionEpoch=(this.motionEpoch || 0)+1;
    const movingPaths=[],reduced=matchMedia('(prefers-reduced-motion:reduce)').matches;
    if(this.vertical!==model.vertical){this.vertical=model.vertical;this.zoom=null;this.focusLayoutKey=null;}
    const old=new Map();let focusKey=null;
    const key=b=>b.dataset.graphBranch?'branch:'+b.dataset.graphBranch:b.dataset.graphNode?'node:'+b.dataset.graphNode:b.dataset.graphSegment?'segment:'+b.dataset.graphSegment:null;
    for(const b of canvas.querySelectorAll('button')){const id=key(b);if(id){old.set(id,{x:b.offsetLeft,y:b.offsetTop});if(document.activeElement===b)focusKey=id;}}
    canvas.replaceChildren();canvas.dataset.orientation=model.vertical?'vertical':'horizontal';
    $('graph-summary').textContent=data.branches.length<=1?'No fork yet':data.shared_prefix_messages?`Shared core: ${data.shared_prefix_messages} messages`:'No shared messages';
    if(data.shared_partial_messages)$('graph-summary').textContent+=` · ${data.shared_partial_messages} shared answer ${data.shared_partial_messages===1?'prefix':'prefixes'}`;
    if(!ContextWorkspace.nativeMode && data.chat_id && data.chat_id===ContextWorkspace.current.chat_id)ContextWorkspace.savedPoints(data);
    const scroll=$('graph-scroll');this.viewportWidth=scroll.clientWidth;this.viewportHeight=scroll.clientHeight;
    const maxHeight=matchMedia('(min-width:1101px)').matches?Math.max(160,scroll.clientHeight-20):parseFloat(getComputedStyle(scroll).maxHeight)-16;
    const fit=Math.min(1,Math.max(200,scroll.clientWidth-20)/model.width,maxHeight/model.height);
    this.scale=this.zoom==='fit'?Math.max(.05,fit):typeof this.zoom==='number'?this.zoom:Math.max(.85,fit);
    $('graph-scale').textContent=Math.round(this.scale*100)+'%';
    canvas.style.width=Math.ceil(model.width*this.scale)+'px';canvas.style.height=Math.ceil(model.height*this.scale)+'px';
    const drawing=document.createElement('div');drawing.className='graph-drawing';drawing.style.width=model.width+'px';drawing.style.height=model.height+'px';drawing.style.transform=`scale(${this.scale})`;canvas.append(drawing);
    const svg=document.createElementNS('http://www.w3.org/2000/svg','svg');svg.setAttribute('width',model.width);svg.setAttribute('height',model.height);svg.setAttribute('aria-hidden','true');drawing.append(svg);
    const curve=(from,to,connector)=>model.vertical?(connector?`M ${from.x} ${from.y} C ${to.x-12} ${from.y}, ${to.x-12} ${to.y}, ${to.x} ${to.y}`:`M ${from.x} ${from.y} C ${from.x} ${from.y+36}, ${to.x} ${to.y-36}, ${to.x} ${to.y}`):to.x-from.x>90?`M ${from.x} ${from.y} C ${from.x+50} ${from.y}, ${from.x+50} ${to.y}, ${from.x+90} ${to.y} L ${to.x} ${to.y}`:`M ${from.x} ${from.y} C ${from.x+20} ${from.y}, ${to.x-20} ${to.y}, ${to.x} ${to.y}`;
    const path=(from,to,classes,connector=false,id='')=>{
      const line=document.createElementNS(svg.namespaceURI,'path');
      const prior=previousPaths.get(id),target={from,to,vertical:model.vertical};
      const moves=prior && prior.vertical===model.vertical && (prior.from.x!==from.x || prior.from.y!==from.y || prior.to.x!==to.x || prior.to.y!==to.y);
      line.setAttribute('d',curve(moves&&!reduced?prior.from:from,moves&&!reduced?prior.to:to,connector));line.setAttribute('class',classes);svg.append(line);
      this.pathGeometry.set(id,moves&&!reduced?prior:target);
      if(moves&&!reduced)movingPaths.push({id,line,prior,target,connector});
      else if(!prior&&!reduced&&previousPaths.size)line.animate([{opacity:0},{opacity:classes.includes('graph-selected')?1:classes.includes('graph-old')?.55:.6}],{duration:220,easing:'ease-out'});
    };
    const button=(label,x,y,classes,callback)=>{
      const b=document.createElement('button');b.type='button';b.className=classes;b.textContent=label;b.style.left=x+'px';b.style.top=y+'px';b.onclick=callback;drawing.append(b);return b;
    };
    for(const edge of model.edges){
      const from=model.position.get(edge.from),to=model.position.get(edge.to),node=model.nodes.get(edge.to);
      const faded=!node.current_branch_count;const selected=model.selected.has(edge.to);
      path(from,to,'graph-line'+(faded?' graph-old':selected?' graph-selected':''),false,'edge:'+edge.to);
      const owners=node.current_branch_count;const count=node.depth-model.nodes.get(edge.from).depth;
      let text=count===0?(node.role==='root'?'Prepared prefix':owners>1?`Shared answer continuation · ${owners} branches`:'Answer continued'):
        owners>1?`${count} shared · ${owners} branches`:`${count} ${count===1?'message':'messages'}`;
      if(count>0 && owners>1 && node.shared_partial){
        const whole=count-1;
        text=(whole?`${whole} ${whole===1?'message':'messages'} + answer prefix`:'Answer prefix')+` · ${owners} branches`;
      }
      const b=button(text,model.vertical?model.badgeX:(from.x+to.x)/2-70,model.vertical?to.y-(model.groups.has(edge.to)?56:12):to.y-16,'graph-segment'+(faded?' graph-old':''),()=>this.inspect(edge.chain,model));
      b.setAttribute('aria-label',count===0?(node.role==='root'?'Inspect prepared prefix':`Inspect continuation of saved message ${node.depth}`):
        `Inspect ${count} saved ${count===1?'message':'messages'}${owners>1?', shared by '+owners+' branches':''}`);b.dataset.graphSegment=edge.to;
    }
    for(const id of model.anchors){
      const node=model.nodes.get(id),point=model.position.get(id);
      const b=button(node.depth?'':'Start',point.x-8,point.y-8,'graph-dot'+(model.selected.has(id)?' graph-selected':'')+(!node.current_branch_count?' graph-old':''),()=>this.inspect(node.depth?[id]:[],model));
      b.setAttribute('aria-label',node.depth?`Inspect message ${node.depth}: ${node.role}`:'Inspect chat start');b.dataset.graphNode=id;
    }
    data.branches.forEach(branch=>{
      const point=model.position.get(branch.tip),badge=model.badgePositions.get(branch.id),y=badge.y;
      const x=model.vertical?model.badgeX:point.x+24;
      path(point,{x,y},branch.id===ContextWorkspace.current.id?'graph-line graph-selected':'graph-line',true,'branch:'+branch.id);
      const answering=!!branch.pending_request || (!!busy && ContextWorkspace.current.id===branch.id);
      const b=button(branch.name,x,badge.top,'graph-tip'+(branch.id===ContextWorkspace.current.id?' graph-selected':''),()=>ContextWorkspace.select(branch.id).catch(e=>ContextWorkspace.error(e)));
      b.dataset.graphTipGroup=branch.tip;
      const name=document.createElement('span');name.className='graph-tip-name';name.textContent=branch.name;b.replaceChildren(name);
      const count=`${branch.saved_messages} saved ${branch.saved_messages===1?'message':'messages'}`;
      const small=document.createElement('small');small.textContent=count+(answering?' · answering…':'');b.append(small);
      b.setAttribute('aria-label',`Open ${branch.name}, ${count}`);b.setAttribute('aria-current',String(branch.id===ContextWorkspace.current.id));b.dataset.graphBranch=branch.id;
    });
    data.previous_tips.forEach((tip,i)=>{
      const point=model.position.get(tip.node_id),badge=model.badgePositions.get('previous-'+i),y=badge.y;const branch=data.branches.find(b=>b.id===tip.branch_id);
      const x=model.vertical?model.badgeX:point.x+24;path(point,{x,y},'graph-line graph-old',true,'previous:'+i);
      const label=`Previous ${branch?.name || 'path'} · ${model.nodes.get(tip.node_id).depth} messages`;
      const b=button(label,x,badge.top,'graph-tip graph-old',()=>this.inspect([tip.node_id],model));b.dataset.graphPrevious=tip.node_id;
      const name=document.createElement('span');name.className='graph-tip-name';name.textContent='Previous '+(branch?.name || 'path');
      const small=document.createElement('small');small.textContent=`${model.nodes.get(tip.node_id).depth} saved messages`;b.replaceChildren(name,small);b.setAttribute('aria-label',label);
    });
    this.disableActions();
    const layoutKey=ContextWorkspace.current.id+':'+model.width+':'+model.height+':'+this.scale+':'+model.vertical+':'+this.viewportWidth+':'+this.viewportHeight+':'+data.branches.map(b=>b.id+':'+b.tip).join(',');
    if($('graph-inspector').hidden && layoutKey!==this.focusLayoutKey){
      this.focusLayoutKey=layoutKey;
      $('branch-viewer').scrollTop=0;scroll.scrollTop=0;scroll.scrollLeft=0;
      const selected=drawing.querySelector('[data-graph-branch][aria-current="true"]');
      if(selected){
        const b=selected.getBoundingClientRect(),v=scroll.getBoundingClientRect();
        if(b.bottom>v.bottom-8)scroll.scrollTop=Math.max(0,(selected.offsetTop+selected.offsetHeight)*this.scale-scroll.clientHeight+16);
        if(b.right>v.right-8)scroll.scrollLeft=Math.max(0,(selected.offsetLeft+selected.offsetWidth)*this.scale-scroll.clientWidth+16);
      }
    }
    for(const b of drawing.querySelectorAll('button')){
      const id=key(b),before=old.get(id);
      if(!reduced && id && old.size){
        const dx=before?before.x-b.offsetLeft:0,dy=before?before.y-b.offsetTop:8;
        const opacity=b.classList.contains('graph-old')?.55:1;
        if(dx || dy || !before)b.animate([{transform:`translate(${dx}px,${dy}px)`,opacity:before?opacity:0},{transform:'translate(0,0)',opacity}],{duration:220,easing:'cubic-bezier(.2,.7,.2,1)'});
      }
      if(id && id===focusKey)b.focus({preventScroll:true});
    }
    canvas.dataset.motion=movingPaths.length?'running':'settled';
    if(movingPaths.length){
      const started=performance.now(),lerp=(a,b,t)=>({x:a.x+(b.x-a.x)*t,y:a.y+(b.y-a.y)*t});
      const frame=now=>{
        if(motionEpoch!==this.motionEpoch)return;
        this.motionFramesTotal=(this.motionFramesTotal || 0)+1;
        const progress=Math.min(1,(now-started)/220),t=1-Math.pow(1-progress,3);
        for(const move of movingPaths){
          const from=lerp(move.prior.from,move.target.from,t),to=lerp(move.prior.to,move.target.to,t);
          move.line.setAttribute('d',curve(from,to,move.connector));this.pathGeometry.set(move.id,{from,to,vertical:model.vertical});
        }
        if(progress<1)this.motionFrame=requestAnimationFrame(frame);else {this.motionFrame=null;canvas.dataset.motion='settled';}
      };this.motionFrame=requestAnimationFrame(frame);
    }
  },
  inspect(ids,model,page=0) {
    const panel=$('graph-inspector');panel.replaceChildren();panel.hidden=false;
    const header=document.createElement('div');header.className='graph-inspector-head';
    const title=document.createElement('strong');title.textContent=ids.length?`Saved messages ${model.nodes.get(ids[0]).depth}–${model.nodes.get(ids.at(-1)).depth}`:'Chat start';header.append(title);
    const close=document.createElement('button');close.type='button';close.className='st-btn';close.textContent='×';close.setAttribute('aria-label','Close preview');close.onclick=()=>{panel.hidden=true;$('branch-viewer').scrollTop=0;};header.append(close);panel.append(header);
    if(!ids.length){const note=document.createElement('p');note.textContent='No messages before this point.';panel.append(note);const actions=this.messageActions(model.nodes.get(this.data.root_id),model);if(actions)panel.append(actions);this.disableActions();panel.scrollIntoView({block:'nearest',inline:'nearest'});return;}
    for(const id of ids.slice(page*10,page*10+10)){
      const node=model.nodes.get(id),row=document.createElement('div');row.className='graph-preview';
      const role=node.role==='user'?'You':node.role==='assistant'?'Answer':node.role.charAt(0).toUpperCase()+node.role.slice(1);
      const label=document.createElement('strong');label.textContent=`${node.depth}. ${role}${node.current_branch_count>1?' · shared by '+node.current_branch_count+' branches':''}`;row.append(label);
      const preview=document.createElement('p');preview.textContent=node.preview || '(No text)';row.append(preview);
      const actions=this.messageActions(node,model);if(actions)row.append(actions);panel.append(row);
    }
    if(ids.length>10){
      const controls=document.createElement('div');controls.className='graph-pages';
      for(const [text,offset,disabled] of [['Previous',-1,page===0],['Next',1,(page+1)*10>=ids.length]]){
        const b=document.createElement('button');b.type='button';b.className='st-btn st-btn--secondary';b.textContent=text;b.disabled=disabled;b.onclick=()=>this.inspect(ids,model,page+offset);controls.append(b);
      }
      const count=document.createElement('span');count.textContent=`${page*10+1}–${Math.min((page+1)*10,ids.length)} of ${ids.length}`;controls.append(count);panel.append(controls);
    }
    this.disableActions();
    panel.scrollIntoView({block:'nearest',inline:'nearest'});
  },
  messageActions(node,model) {
    if(!node)return null;
    if(ContextWorkspace.nativeMode){
      const selected=model.selected.has(node.id);
      const previous=(ContextWorkspace.current?.saved_points||[]).some(point=>point.boundary_available && point.snapshot_id===node.action_snapshot);
      if(!selected&&!previous)return null;
      return ContextWorkspace.messageActions({node_id:node.action_snapshot,fork_available:selected,
        boundary_available:node.boundary_available===true && typeof node.action_snapshot==='string'});
    }
    return ContextWorkspace.messageActions({node_id:node.id});
  }
};
