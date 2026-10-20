"""Real-browser math controls must update and persist without pressing Apply."""
import argparse
import json
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def run(browser_name,output):
    from playwright.sync_api import sync_playwright
    from serve.frontend import ChatTemplate
    from serve.server import ByteTokenizer,MockEngine,Service,serve
    output.mkdir(parents=True,exist_ok=True)
    tok=ByteTokenizer();engine=MockEngine(tok,'ok')
    server=serve(Service(engine,tok,ChatTemplate(ROOT/'serve/chat_template.jinja')),port=0)
    base=f'http://127.0.0.1:{server.server_port}/'
    report={'browser':browser_name,'status':'RUNNING','scope':'real browser, scripted fixture, no GPU','checks':[]}
    def check(name,ok):
        if not ok:raise AssertionError(name)
        report['checks'].append(name)
    try:
        with sync_playwright() as pw:
            browser=getattr(pw,browser_name).launch()
            page=browser.new_page(viewport={'width':1000,'height':800})
            errors=[];page.on('pageerror',lambda e:errors.append(str(e)))
            page.goto(base);page.wait_for_function("typeof settings !== 'undefined'")
            source=r'One half is \(\frac{1}{2}=0.5\).'
            page.evaluate("text=>{messages=[{role:'assistant',text,time:Date.now()}];saveChat();renderChat();}",source)
            baseline=page.locator('.st-bubble').inner_html()
            page.click('#sampling-btn');page.click('#s-math');page.click('#s-apply')
            page.wait_for_selector('.st-bubble .katex',timeout=15000)
            page.click('#sampling-btn');page.click('#s-math')
            check('disable_updates_existing_answer_without_apply',page.locator('.st-bubble .katex,.st-bubble mjx-container').count()==0)
            check('disabled_source_is_original_markdown',page.locator('.st-bubble').inner_html()==baseline)
            check('disabled_setting_saved_immediately',page.evaluate("JSON.parse(localStorage.getItem('strata.sampling')).math===false"))
            page.fill('#s-max','42')
            page.select_option('#s-math-engine','mathjax')
            check('viewer_choice_saved_while_disabled',page.evaluate("JSON.parse(localStorage.getItem('strata.sampling')).mathEngine==='mathjax'"))
            check('changing_viewer_keeps_math_disabled',page.locator('[data-math]').count()==0)
            check('pending_sampling_edits_are_not_saved_by_math',page.evaluate("settings.max!== '42'"))
            page.evaluate("globalThis.liveElement=document.querySelector('.st-msg--assistant');busy={msg:messages[0]}")
            page.click('#s-math')
            page.wait_for_selector('.st-bubble mjx-container',timeout=15000)
            check('enable_uses_selected_viewer_without_apply',page.locator('.st-bubble .katex').count()==0)
            page.select_option('#s-math-engine','katex')
            page.wait_for_selector('.st-bubble .katex',timeout=15000)
            check('switch_viewer_updates_without_apply',page.locator('.st-bubble mjx-container').count()==0)
            check('streaming_element_is_preserved',page.evaluate("liveElement===document.querySelector('.st-msg--assistant')"))
            check('raw_answer_is_preserved',page.evaluate('messages[0].text')==source)
            page.click('#s-reset')
            check('reset_disables_math_immediately',page.locator('[data-math]').count()==0)
            check('reset_persists_math_choice',page.evaluate("!settings.math && settings.mathEngine==='katex'"))
            page.click('#s-math');page.wait_for_selector('.st-bubble .katex',timeout=15000)
            page.evaluate('busy=null')
            page.reload();page.wait_for_selector('.st-bubble .katex',timeout=15000)
            check('reload_preserves_immediate_choices',page.evaluate("settings.math && settings.mathEngine==='katex' && messages[0].text") == source)
            page.click('#sampling-btn');page.click('#s-math')
            page.reload();page.wait_for_function("typeof settings !== 'undefined'")
            check('reload_keeps_disabled_math_raw',page.locator('[data-math]').count()==0 and page.locator('.st-bubble').inner_html()==baseline)
            check('no_model_request_for_viewer_changes',engine.turns==0)
            check('no_browser_errors',not errors)
            page.screenshot(path=str(output/(browser_name+'-math-disabled.png')),full_page=True)
            browser.close()
        report['status']='PASS'
    except BaseException as error:
        report.update(status='FAIL',error=repr(error));raise
    finally:
        server.shutdown();server.server_close()
        (output/(browser_name+'-immediate-results.json')).write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--browser',choices=['chromium','firefox','webkit'],default='chromium')
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();run(args.browser,args.output)
