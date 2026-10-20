"""Real-browser integration for two selectable math viewers; no model/GPU claim."""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def run(browser_name,output):
    from playwright.sync_api import sync_playwright
    from serve.frontend import ChatTemplate
    from serve.server import ByteTokenizer,MockEngine,Service,serve
    tok=ByteTokenizer();server=serve(Service(MockEngine(tok,'ok'),tok,ChatTemplate(ROOT/'serve/chat_template.jinja')),port=0)
    base=f'http://127.0.0.1:{server.server_port}/'
    report=dict(browser=browser_name,status='RUNNING',scope='actual browser UI; mock inference, no GPU',checks=[],engines={})
    output.mkdir(parents=True,exist_ok=True)
    def check(name,ok):
        if not ok:raise AssertionError(name)
        report['checks'].append(name)
    try:
        with sync_playwright() as pw:
            browser=getattr(pw,browser_name).launch()
            context=browser.new_context(viewport=dict(width=390,height=844))
            context.route('**/*',lambda r:r.continue_() if r.request.url.startswith(base) else r.abort())
            page=context.new_page();errors=[];requests=[]
            page.on('pageerror',lambda e:errors.append(str(e)))
            page.on('request',lambda r:requests.append(r.url))
            page.goto(base);page.wait_for_function("typeof settings !== 'undefined'")
            check('rendering_starts_disabled',not page.evaluate('settings.math'))
            check('no_vendor_load_when_disabled',not any('/vendor/' in r for r in requests))
            check('renderer_selector_exists',page.locator('#s-math-engine').count()==1)
            source=r'Answer: \(\frac{1}{2}\), $x_1$, and $$\sum_{n=1}^{3}n=6$$.'
            page.evaluate("text=>{messages=[{role:'assistant',text,time:Date.now()}];saveChat();renderChat();}",source)
            baseline=page.locator('.st-bubble').inner_html()
            for engine,selector in [('katex','.katex'),('mathjax','mjx-container'),('katex','.katex')]:
                page.evaluate("globalThis.liveMathMessage=document.querySelector('.st-msg--assistant');busy={msg:messages[0]}")
                page.click('#sampling-btn');page.select_option('#s-math-engine',engine)
                if page.get_attribute('#s-math','aria-checked')!='true':page.click('#s-math')
                started=time.perf_counter();page.click('#s-apply')
                check(engine+'_switch_keeps_streaming_element',page.evaluate("liveMathMessage===document.querySelector('.st-msg--assistant')"))
                page.wait_for_function("s=>document.querySelectorAll('[data-math]').length===3 && [...document.querySelectorAll('[data-math]')].every(e=>e.querySelector(s)&&!e.dataset.mathError)",arg=selector)
                report['engines'].setdefault(engine,[]).append(dict(chat_ms=(time.perf_counter()-started)*1000))
                check(engine+'_adapter_selected',page.evaluate('settings.mathEngine')==engine)
                check(engine+'_source_preserved_while_streaming',page.evaluate('messages[0].text')==source)
                check(engine+'_persisted_choice',page.evaluate("JSON.parse(localStorage.getItem('strata.sampling')).mathEngine")==engine)
                page.evaluate('busy=null;updateAssistant(liveMathMessage,messages[0],false)')
                check(engine+'_completed_api_source_preserved',page.evaluate('apiMessages()[0].content')==source)
            page.evaluate("""async()=>{
              const container=document.createElement('div');document.body.append(container);
              let finish;globalThis.StrataMathAdapter={initialize:async()=>{},render:()=>new Promise(r=>finish=r)};
              StrataMath.reset();container.innerHTML=StrataMath.markdown('\\\\(pending\\\\)',markdown);
              StrataMath.hydrate(container);
              while(!finish)await new Promise(r=>setTimeout(r,0));
              StrataMathEngines.select('mathjax');
              let completed=false;const next=StrataMath.result('x+1',false).then(r=>{completed=true;return r});
              await new Promise(r=>setTimeout(r,40));
              if(completed)throw Error('conversion overlapped across engine switch');
              const stale=document.createElement('strong');stale.textContent='STALE';finish({ok:true,element:stale});
              const rendered=await next;await new Promise(r=>setTimeout(r,0));
              if(container.querySelector('strong'))throw Error('old engine result hydrated after switching');
              if(!rendered.ok||!rendered.element.matches('mjx-container'))throw Error('new engine result missing');
              container.remove();
            }""")
            check('switch_discards_old_results_and_keeps_queue_serial',True)
            page.click('#sampling-btn');page.select_option('#s-math-engine','mathjax');page.click('#s-apply')
            page.reload();page.wait_for_function("settings.mathEngine==='mathjax' && [...document.querySelectorAll('[data-math]')].length===3 && [...document.querySelectorAll('[data-math]')].every(e=>e.querySelector('mjx-container'))")
            check('reload_restores_selected_viewer_and_source',page.evaluate('messages[0].text')==source)
            page.evaluate("store.set('sampling',{...settings,mathEngine:'unsupported-viewer'})")
            page.reload();page.wait_for_function("settings.mathEngine==='katex' && [...document.querySelectorAll('[data-math]')].length===3 && [...document.querySelectorAll('[data-math]')].every(e=>e.querySelector('.katex'))")
            check('unknown_saved_viewer_falls_back_to_katex',True)
            page.evaluate("document.documentElement.dataset.theme='dark'")
            page.evaluate('async()=>{await document.fonts.ready}')
            check('narrow_viewport_has_no_page_overflow',page.evaluate('document.documentElement.scrollWidth<=innerWidth'))
            page.screenshot(path=str(output/(browser_name+'-dual-math.png')),full_page=True)
            page.click('#sampling-btn');page.click('#s-math');page.click('#s-apply')
            check('disabled_round_trip_uses_original_markdown',page.locator('.st-bubble').inner_html()==baseline)
            check('no_browser_errors',not errors)
            check('no_external_asset_requests',all(r.startswith(base) for r in requests))
            context.close();browser.close()
        report['status']='PASS'
    except BaseException as e:
        report.update(status='FAIL',error=repr(e));raise
    finally:
        server.shutdown();server.server_close()
        (output/(browser_name+'-dual-results.json')).write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8',newline='\n')
        print(json.dumps(report,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--browser',choices=['chromium','firefox','webkit'],default='chromium')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();run(args.browser,args.output)
