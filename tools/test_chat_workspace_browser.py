"""Exercise the real branch-aware browser against a temporary SQLite workspace."""
import argparse
import json
from pathlib import Path
import sys
import tempfile

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))


def run(browser_name,output):
    from playwright.sync_api import sync_playwright
    from serve.chat_workspace import ChatWorkspace
    from serve.frontend import ChatTemplate
    from serve.server import ByteTokenizer,MockEngine,Service,serve
    output.mkdir(parents=True,exist_ok=True);tmp=tempfile.TemporaryDirectory();tok=ByteTokenizer()
    engine=MockEngine(tok,'</think>\n\nA half is \\(\\frac{1}{2}=0.5\\).')
    svc=Service(engine,tok,ChatTemplate(ROOT/'serve/chat_template.jinja'));svc.workspace=ChatWorkspace(Path(tmp.name)/'chats.sqlite3')
    server=serve(svc,port=0);base=f'http://127.0.0.1:{server.server_port}/'
    report={'browser':browser_name,'status':'RUNNING','scope':'real UI and SQLite; mock model, no native COW','checks':[]}
    def check(name,value):
        if not value:raise AssertionError(name)
        report['checks'].append(name)
    def dialog(page,text):
        page.locator('#workspace-name').fill(text);page.locator('#workspace-confirm').click()
    try:
        with sync_playwright() as pw:
            browser=getattr(pw,browser_name).launch();page=browser.new_page(viewport={'width':1280,'height':900})
            errors=[];page.on('pageerror',lambda e:errors.append(str(e)))
            page.goto(base)
            page.wait_for_selector('#workspace-sidebar:not([hidden])',timeout=8000)
            check('chat_list_is_visible',page.get_by_role('heading',name='Chats',exact=True).count()==1)
            page.click('#new-btn');dialog(page,'Calculus notes')
            page.wait_for_function("ContextWorkspace.current && document.querySelector('#workspace-title').textContent==='Calculus notes'")
            page.wait_for_selector('#workspace-dialog',state='hidden')
            check('named_chat_created',page.locator('#chat-list').inner_text().find('Calculus notes')>=0)
            page.fill('#input','Explain one half.');page.click('#send-btn')
            page.wait_for_function("!busy && messages.length===2 && messages.every(m=>m.node_id)")
            check('answer_is_durably_saved',len(svc.workspace.history(page.evaluate('ContextWorkspace.current.id'))['messages'])==2)
            check('sleek_icons_are_default',page.evaluate("getComputedStyle(document.querySelector('[data-workspace-fork] .control-label')).position==='absolute'"))
            check('icons_have_accessible_names',page.get_by_role('button',name='Fork here',exact=True).count()==2)
            page.locator('[data-workspace-fork]').last.hover();page.wait_for_timeout(120)
            check('brief_hover_does_not_show_tooltip',page.evaluate("getComputedStyle(document.querySelectorAll('[data-workspace-fork]')[1],'::after').opacity==='0'"))
            page.wait_for_timeout(950)
            check('held_hover_shows_tooltip',page.evaluate("Number(getComputedStyle(document.querySelectorAll('[data-workspace-fork]')[1],'::after').opacity)>0.9"))
            page.click('#teaching-btn')
            check('teaching_mode_shows_labels',page.evaluate("getComputedStyle(document.querySelector('[data-workspace-fork] .control-label')).position!=='absolute'"))
            page.click('#teaching-btn')
            page.click('#sampling-btn');page.click('#s-math');page.click('#s-apply')
            page.wait_for_selector('.st-bubble .katex',timeout=15000)
            check('math_in_saved_branch_renders',page.locator('.st-bubble .katex').count()==1)
            page.locator('[data-workspace-fork]').last.click()
            page.wait_for_function("ContextWorkspace.current && ContextWorkspace.current.name==='Alternative 1'")
            page.wait_for_selector('#workspace-dialog',state='hidden')
            alternative=page.evaluate('ContextWorkspace.current.id')
            check('fork_does_not_open_confirmation',not page.locator('#workspace-dialog').is_visible())
            check('rename_and_clear_have_distinct_icons',page.locator('#rename-branch-btn svg').inner_html()!=page.locator('#clear-branch-btn svg').inner_html())
            check('fork_retains_prefix_messages',page.evaluate('messages.length')==2)
            page.fill('#input','Use a pizza example.');page.click('#send-btn')
            page.wait_for_function("!busy && messages.length===4 && messages.every(m=>m.node_id)")
            check('fork_has_its_own_suffix',page.evaluate("messages[2].text==='Use a pizza example.'"))
            branches=svc.workspace.list_branches(page.evaluate('ContextWorkspace.current.chat_id'))
            main=next(b for b in branches if b['name']=='Main')
            page.select_option('#branch-select',main['id']);page.wait_for_function('messages.length===2')
            check('parent_branch_is_unchanged',page.evaluate("!messages.some(m=>m.text.includes('pizza'))"))
            page.select_option('#branch-select',alternative);page.wait_for_function('messages.length===4')
            page.locator('[data-workspace-rewind]').nth(1).click()
            page.wait_for_function('messages.length===2 && !ContextWorkspace.moving')
            check('rewind_does_not_open_confirmation',not page.locator('#workspace-dialog').is_visible())
            page.select_option('#restore-point',page.locator('#restore-point option').nth(1).get_attribute('value'))
            page.click('#restore-point-btn');page.wait_for_function('messages.length===4')
            page.click('#clear-branch-btn')
            page.wait_for_function('!busy && messages.length===0 && !ContextWorkspace.moving')
            page.wait_for_selector('#workspace-dialog',state='hidden')
            check('clear_empties_only_selected_branch',len(svc.workspace.history(main['id'])['messages'])==2)
            page.select_option('#restore-point',page.locator('#restore-point option').nth(1).get_attribute('value'))
            page.click('#restore-point-btn');page.wait_for_function('messages.length===4')
            check('restore_does_not_open_confirmation',not page.locator('#workspace-dialog').is_visible())
            page.wait_for_selector('#workspace-dialog',state='hidden')
            check('clear_is_recoverable',page.evaluate("messages[2].text==='Use a pizza example.'"))
            page.reload();page.wait_for_function("ContextWorkspace.enabled && ContextWorkspace.current && messages.length===4")
            check('reload_keeps_selected_branch_and_history',page.evaluate('ContextWorkspace.current.id')==alternative)
            page.click('#rename-branch-btn');dialog(page,'Pizza version')
            page.wait_for_function("ContextWorkspace.current.name==='Pizza version'")
            page.wait_for_selector('#workspace-dialog',state='hidden')
            check('branch_can_be_named',page.locator('#branch-select').input_value()==alternative)
            page.click('#archive-chat-btn')
            page.wait_for_function("document.querySelector('#chat-list').textContent.indexOf('Calculus notes')<0")
            page.check('#show-archived');page.wait_for_function("document.querySelector('#chat-list').textContent.includes('Calculus notes')")
            check('archived_chat_is_recoverable',True)
            check('archiving_keeps_sqlite_history',len(svc.workspace.history(alternative)['messages'])==4)
            page.screenshot(path=str(output/(browser_name+'-workspace.png')),full_page=True)
            page.set_viewport_size({'width':390,'height':844})
            report['overflow_elements']=page.evaluate("Array.from(document.querySelectorAll('body *')).filter(e=>e.getBoundingClientRect().right>innerWidth+1).slice(0,10).map(e=>({id:e.id,cls:e.className,right:e.getBoundingClientRect().right}))")
            report['mobile_layout']=page.evaluate("({width:innerWidth,scroll:document.documentElement.scrollWidth,teaching:document.documentElement.dataset.teaching,buttons:Array.from(document.querySelectorAll('.cow-icon')).map(e=>({id:e.id,x:e.getBoundingClientRect().x,right:e.getBoundingClientRect().right,tooltip:getComputedStyle(e,'::after').width}))})")
            page.screenshot(path=str(output/(browser_name+'-mobile.png')),full_page=True)
            check('mobile_has_no_page_overflow',page.evaluate('document.documentElement.scrollWidth<=innerWidth'))
            check('no_browser_errors',not errors)
            browser.close()
        report['status']='PASS'
    except BaseException as error:
        report.update(status='FAIL',error=repr(error))
        raise
    finally:
        server.shutdown();server.server_close();svc.workspace.close();tmp.cleanup()
        (output/(browser_name+'-workspace-results.json')).write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--browser',choices=['chromium','firefox','webkit'],default='chromium');p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();run(a.browser,a.output)
