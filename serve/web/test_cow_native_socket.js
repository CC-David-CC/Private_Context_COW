"use strict";
// Runs in the real same-origin CowBrowserFixtureServer; fetch is never mocked.
(async()=>{
  const delay=ms=>new Promise(resolve=>setTimeout(resolve,ms));
  const checks=[];
  const waitFor=async(predicate,label)=>{for(let i=0;i<240;i++){if(predicate())return;await delay(50);}throw Error(`timed out: ${label}`);};
  const report=(ok,message)=>{const pre=document.createElement('pre');pre.id='cow-native-socket-report';pre.textContent=`${ok?'PASS':'FAIL'} ${message}`;document.body.append(pre);document.title=pre.textContent;};
  try{
    await waitFor(()=>ContextWorkspace.nativeReady,'native service discovery and workspace init');checks.push('UI-BOOT: PASS same-origin native discovery and workspace initialization');
    if(messages.length||$('chat').textContent.includes('must not import this'))throw Error('legacy browser history leaked into COW projection');
    if(performance.getEntriesByType('resource').some(r=>new URL(r.name).pathname.startsWith('/workspace/')))throw Error('ordinary workspace route was used');
    checks.push('UI-ISOLATION: PASS legacy local history not imported; no ordinary workspace routes');
    $('sampling-btn').click();
    if(!$('s-temp').disabled||!$('s-seed').disabled||$('s-mcp').disabled===false)throw Error('unsupported stochastic/tool settings were not visibly disabled');
    $('s-max').value='2048';$('s-apply').click();
    if(settings.max!=='2048'||!$('composer-hint').textContent.includes('2048'))throw Error('native output limit did not accept the selected bound');
    checks.push('UI-LIMIT: PASS editable 2048-token bound; unsupported stochastic/tool controls disabled');
    $('input').value='socket-backed first question';await send();
    await waitFor(()=>!busy&&ContextWorkspace.current?.controller_status==='active'&&!$('cow-resume-btn').hidden,'durable active first answer');
    if(!messages.some(m=>m.role==='user'&&m.text==='socket-backed first question'))throw Error('first user turn not projected');
    if(!messages.some(m=>m.role==='assistant'&&m.text==='hi'))throw Error(`active assistant output missing: ${messages.map(m=>m.text).join('|')}`);
    checks.push('UI-PREPARE: PASS first send creates canonical prepared branch and active assistant snapshot');
    $('cow-resume-btn').click();
    await waitFor(()=>!busy&&ContextWorkspace.current?.controller_status==='stopped'&&messages.some(m=>m.role==='assistant'&&m.text==='hi!'),'explicit resume and complete answer');
    checks.push('UI-RESUME: PASS explicit resume completes the active answer');
    const firstAssistant=$('chat').querySelector('.st-msg--assistant');
    const fork=Array.from(firstAssistant.querySelectorAll('.workspace-message-actions button')).find(b=>b.textContent.includes('Fork here'));
    if(!fork)throw Error('saved assistant boundary did not expose Fork here');
    const oldBranch=ContextWorkspace.current.id;fork.click();
    await waitFor(()=>!busy&&ContextWorkspace.current?.id!==oldBranch&&ContextWorkspace.branches.length===2,'canonical branch fork');
    checks.push('UI-FORK: PASS branch fork selected a distinct canonical branch');
    $('input').value='follow-up question';await send();
    await waitFor(()=>!busy&&messages.some(m=>m.role==='assistant'&&m.text==='!'),'second durable turn');
    checks.push('UI-APPEND: PASS stopped branch accepts a new user turn and projected answer');
    const oldAssistant=$('chat').querySelector('.st-msg--assistant');
    const rewind=Array.from(oldAssistant.querySelectorAll('.workspace-message-actions button')).find(b=>b.textContent.includes('Retry here'));
    if(!rewind)throw Error('earlier assistant boundary did not expose Retry here');
    rewind.click();
    await waitFor(()=>!busy&&messages.filter(m=>m.role==='assistant').length===1,'canonical rewind');
    checks.push('UI-REWIND: PASS rewind refreshes canonical projection');
    const requests=performance.getEntriesByType('resource').map(r=>new URL(r.name).pathname);
    if(requests.some(path=>path.startsWith('/workspace/')))throw Error('ordinary SQLite workspace request observed');
    if(!requests.some(path=>path==='/cow/text/prepare')||!requests.some(path=>path==='/cow/text/answer')||!requests.some(path=>path==='/cow/text/resume')||!requests.some(path=>path==='/cow/text/append-user')||!requests.includes('/cow/fork')||!requests.includes('/cow/rewind'))throw Error(`incomplete native HTTP flow: ${requests.filter(p=>p.startsWith('/cow/')).join(',')}`);
    checks.push('UI-ROUTES: PASS actual fetch reached prepare/answer/resume/append-user/fork/rewind; no SQLite workspace routes');
    report(true,'real same-origin synthetic COW HTTP\n'+checks.join('\n'));
  }catch(error){report(false,error.stack||String(error));}
})();
