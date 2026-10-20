#!/usr/bin/env python3
"""Real UI/CSS and bundled math overflow checks; synthetic HTTP, no inference."""
from __future__ import annotations
import argparse,hashlib,json,os,sys,time,traceback
from pathlib import Path
from playwright.sync_api import sync_playwright
from serve.cow_browser_fixture import CowBrowserFixtureServer

def main():
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument('--output-dir',type=Path,required=True)
 parser.add_argument('--browser',default='/snap/chromium/3554/usr/lib/chromium-browser/chrome')
 args=parser.parse_args();out=args.output_dir.resolve();out.mkdir(mode=0o700,parents=True,exist_ok=False)
 root=Path(__file__).resolve().parents[2]
 report=dict(schema=1,status='RUNNING',scope='Actual COW DOM/CSS and actual bundled KaTeX/MathJax; synthetic loopback HTTP fixture,no inference or GPU qualification.',browser_sandbox=True,checks=[],page_errors=[],request_failures=[])
 stage='fixture';server=None
 try:
  with CowBrowserFixtureServer() as server:
   saved_store=server.store_path
   with sync_playwright() as pw:
    browser=pw.chromium.launch(executable_path=args.browser,headless=True,chromium_sandbox=True,args=['--disable-gpu'])
    report['browser']=browser.version;report['browser_binary_sha256']=hashlib.sha256(Path(args.browser).read_bytes()).hexdigest()
    context=browser.new_context(viewport=dict(width=412,height=915))
    page=context.new_page();requests=[]
    page.on('pageerror',lambda error:report['page_errors'].append(str(error)))
    page.on('request',lambda request:requests.append(dict(method=request.method,path=request.url.split(server.base_url)[-1])))
    page.on('requestfailed',lambda request:report['request_failures'].append(dict(url=request.url,failure=request.failure)))
    context.route('**/*',lambda route:route.continue_() if route.request.url.startswith(server.base_url+'/') else route.abort())
    page.goto(server.base_url+'/',wait_until='networkidle')
    font=page.request.get(server.base_url+'/fonts/outfit-latin-wght.woff2')
    report['app_font']=dict(status=font.status,sha256=hashlib.sha256(font.body()).hexdigest(),expected_sha256=hashlib.sha256((root/'serve/web/fonts/outfit-latin-wght.woff2').read_bytes()).hexdigest())
    page.wait_for_function("typeof ContextWorkspace!=='undefined'&&ContextWorkspace.nativeReady===true")
    page.evaluate("ContextWorkspace.makeNativeChat('Overflow browser fixture')")
    page.wait_for_function("ContextWorkspace.currentChat&&ContextWorkspace.loading===false")
    namespace=page.evaluate('ContextWorkspace.currentChat')
    before=page.evaluate("ContextWorkspace.nativeRequest('workspace/branches?chat='+ContextWorkspace.currentChat)")
    formula='y='+ '+'.join(r'\frac{a_{'+str(i)+r'}+b_{'+str(i)+r'}}{c_{'+str(i)+r'}+d_{'+str(i)+r'}}' for i in range(1,33))
    source='A long inline expression: $'+formula+'$.\n\nA long display expression:\n\n$$'+formula+'$$\n\nThe equation scrolls locally; the page stays within the viewport.'
    wide_formula=r'z=\frac{\text{'+('UnbrokenFormula'*20)+r'}}{2}'
    source+='\n\nA wide inline fraction: $'+wide_formula+'$.\n\nA wide display fraction:\n\n$$'+wide_formula+'$$'
    tools=[dict(id='overflow-tool',name='firecrawl__firecrawl_search_feedback_'+'long_namespace_'*24,tool='firecrawl__firecrawl_search_feedback_'+'long_namespace_'*24,server='local_mcp',arguments=dict(query='an_unbroken_query_'*24),result='an_unbroken_result_'*80,ok=True,chars=1600,ms=1250,state='done',at=0,open=False)]
    stage='actual render matrix'
    for engine in ['katex','mathjax']:
     page.evaluate("engine=>{settings.math=true;settings.mathEngine=engine;StrataMathEngines.select(engine)}",engine)
     for width in [320,360,412,768,1280]:
      page.set_viewport_size(dict(width=width,height=915))
      for state in ['writing','running','done','error','skipped']:
       for opened in [False,True]:
        page.evaluate("""data=>{messages.splice(0,messages.length,{role:'assistant',text:data.source,reasoning:'',time:0,tools:[{...data.tool,state:data.state,open:data.opened}]});renderChat()}""",dict(source=source,tool=tools[0],state=state,opened=opened))
        selector='.chat-math .katex' if engine=='katex' else '.chat-math mjx-container'
        page.wait_for_function("selector=>document.querySelectorAll(selector).length===4",arg=selector,timeout=30000)
        page.evaluate('document.fonts.ready')
        page.wait_for_timeout(30)
        values=page.evaluate("""()=>{const rect=e=>{const r=e.getBoundingClientRect();return {left:r.left,right:r.right,width:r.width,height:r.height}};const summary=document.querySelector('.tool-call > summary'),name=document.querySelector('.tool-call__name'),badge=summary.querySelector('.st-badge'),chev=summary.querySelector('.st-chev');return {outside:[...document.querySelectorAll('body *')].filter(e=>{const r=e.getBoundingClientRect();return r.width>0&&r.right>innerWidth+.5}).slice(0,25).map(e=>({tag:e.tagName,id:e.id,cls:e.className,right:e.getBoundingClientRect().right})),viewport:innerWidth,rootWidth:document.documentElement.scrollWidth,chatWidth:document.getElementById('chat-scroll').scrollWidth,chatClient:document.getElementById('chat-scroll').clientWidth,summary:rect(summary),name:{...rect(name),client:name.clientWidth,scroll:name.scrollWidth,overflow:getComputedStyle(name).overflowX,ellipsis:getComputedStyle(name).textOverflow,title:name.title},badge:rect(badge),chevron:rect(chev),math:[...document.querySelectorAll('.chat-math')].map(e=>({display:e.dataset.display,client:e.clientWidth,scroll:e.scrollWidth,...rect(e),overflow:getComputedStyle(e).overflowX,source:decodeURIComponent(e.dataset.math),error:e.dataset.mathError||null})),messageSource:messages[0].text,mathErrors:document.querySelectorAll('.katex-error,mjx-merror').length}}""")
        failures=[]
        if values['rootWidth']>width:failures.append('page widened')
        if values['chatWidth']>values['chatClient']:failures.append('chat scroll area widened')
        for name in ['badge','chevron']:
         if values[name]['right']>values['summary']['right']+0.5 or values[name]['left']<values['summary']['left']-0.5:failures.append(name+' outside summary')
        if values['name']['width']<16:failures.append('tool name hidden by metadata')
        if values['name']['right']>values['summary']['right']+0.5:failures.append('tool name outside summary')
        if values['name']['ellipsis']!='ellipsis' or values['name']['overflow']!='hidden':failures.append('tool name lacks local ellipsis')
        for index,math in enumerate(values['math']):
         if math['right']>width+0.5 or math['left']<-.5:failures.append('math widens message')
         if math['source']!=(formula if index<2 else wide_formula) or math['error']:failures.append('math source changed or render failed')
         if math['display']=='1' and (math['client']<=0 or math['scroll']<=math['client']):failures.append('wide equation not locally scrollable')
         if math['display']=='1' and math['overflow']!='auto':failures.append('wide equation lacks local scroll')
        if values['mathErrors'] or values['messageSource']!=source:failures.append('math error or transcript changed')
        item=dict(id=f'{engine}-{width}-{state}-'+('open' if opened else 'closed'),status='FAIL' if failures else 'PASS',failures=failures,metrics=values)
        report['checks'].append(item)
        if state=='error' and opened and width in [412,1280]:
         page.locator('.tool-call > summary').scroll_into_view_if_needed()
         page.screenshot(path=str(out/f'{engine}-{width}-tool.png'),full_page=True)
         page.locator('.chat-math--display').first.scroll_into_view_if_needed()
         page.screenshot(path=str(out/f'{engine}-{width}-math.png'),full_page=True)
    stage='summary interaction'
    page.locator('.tool-call > summary').click()
    report['summary_click_toggles']=not page.locator('.tool-call').evaluate('(e)=>e.open')
    page.wait_for_load_state('networkidle')
    stage='math disabled'
    page.evaluate('settings.math=false;renderChat()')
    page.wait_for_function("document.querySelectorAll('.chat-math').length===0")
    report['math_disabled_source_preserved']=page.evaluate('messages[0].text')==source
    after=page.evaluate("ContextWorkspace.nativeRequest('workspace/branches?chat='+ContextWorkspace.currentChat)")
    report['canonical_branches_unchanged']=before==after
    report['inference_requests']=[v for v in requests if v['method']=='POST' and any(s in v['path'] for s in ['/text/ask','/text/answer','/text/resume','/text/append-user','/generate'])]
    report['synthetic_model_execution_trace']=server.trace
    report['source_sha256']={name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in ['serve/cow_http.py','serve/web/app.css','serve/web/components.css','serve/web/app.js','serve/web/math.js','serve/web/math-adapter.js','serve/web/math-katex.js','serve/web/math-mathjax.js','serve/web/cow_overflow_browser_test.py']}
    page.evaluate('document.fonts.ready')
    page.wait_for_load_state('networkidle')
    browser.close()
  report['fixture_store_removed']=not saved_store.exists()
  report['failed_cases']=sum(v['status']!='PASS' for v in report['checks'])
  report['status']='PASS' if (report['failed_cases']==0 and report['summary_click_toggles'] and report['app_font']['status']==200 and report['app_font']['sha256']==report['app_font']['expected_sha256'] and not report['page_errors'] and not report['request_failures'] and report['math_disabled_source_preserved'] and report['canonical_branches_unchanged'] and not report['inference_requests'] and not report['synthetic_model_execution_trace'] and report['fixture_store_removed']) else 'FAIL'
 except Exception as error:
  report.update(status='ERROR',stage=stage,error=dict(type=type(error).__name__,message=str(error)),traceback=traceback.format_exc())
 finally:
  (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
 print(json.dumps(dict(status=report['status'],checks=len(report['checks']),failed_cases=report.get('failed_cases'),report=str(out/'report.json'),error=report.get('error'))))
 return int(report['status']!='PASS')
if __name__=='__main__':raise SystemExit(main())
