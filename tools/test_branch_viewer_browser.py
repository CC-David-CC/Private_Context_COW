"""Interactive branch graph checks: real SQLite, mock inference, no GPU claims."""
import argparse
import json
from pathlib import Path
import sys
import tempfile
import time

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))


def run(browser_name,output):
    from playwright.sync_api import sync_playwright
    from serve.chat_workspace import ChatWorkspace
    from serve.frontend import ChatTemplate
    from serve.server import ByteTokenizer,MockEngine,Service,serve
    output.mkdir(parents=True,exist_ok=True);tmp=tempfile.TemporaryDirectory();tok=ByteTokenizer()
    svc=Service(MockEngine(tok,'</think>\n\nA saved answer.'),tok,ChatTemplate(ROOT/'serve/chat_template.jinja'))
    svc.workspace=ChatWorkspace(Path(tmp.name)/'chats.sqlite3');store=svc.workspace
    chat=store.create_chat('Three paths','create');main=chat['branch']['id']
    serial=0
    def add(branch,text,role='user'):
        nonlocal serial
        serial+=1;head=store.history(branch)
        return store.append(branch,head['revision'],{'role':role,'text':text,'time':serial},str(serial))
    add(main,'Explain fractions.');p=add(main,'A fraction is part of a whole.','assistant')
    c=store.fork(main,p['tip'],'Pictures','fork-c')['id']
    add(main,'Compare two fractions.');q=add(main,'Use the same denominator.','assistant')
    b=store.fork(main,q['tip'],'Pizza','fork-b')['id']
    add(main,'Use a number line.');add(main,'Mark zero, one half and one.','assistant')
    add(b,'Use pizza slices.');old_b=add(b,'Two of four slices is one half.','assistant')['tip']
    add(c,'Draw a circle.');add(c,'Shade half the circle.','assistant')
    server=serve(svc,port=0);base=f'http://127.0.0.1:{server.server_port}/'
    report={'browser':browser_name,'status':'RUNNING','scope':__doc__,'checks':[]}
    def check(name,value):
        if not value:raise AssertionError(name)
        report['checks'].append(name)
    def ready(page,branch):
        page.wait_for_function('(id)=>ContextWorkspace.current?.id===id && ContextGraph.data?.chat_id===ContextWorkspace.current.chat_id && !ContextWorkspace.moving && document.querySelector(`[data-graph-branch="${id}"]`)?.getAttribute("aria-current")==="true"',arg=branch)
        page.locator('[data-graph-branch="'+branch+'"]').wait_for(state='attached')
        page.wait_for_function("document.querySelector('#graph-canvas').getAnimations({subtree:true}).every(a=>a.playState==='finished')")
        page.wait_for_function("document.querySelector('#graph-canvas').dataset.motion!=='running'")
    try:
        with sync_playwright() as pw:
            browser=getattr(pw,browser_name).launch();page=browser.new_page(viewport={'width':1360,'height':1000})
            errors=[];page.on('pageerror',lambda e:errors.append(str(e)))
            page.goto(base);ready(page,main)
            check('viewer_opens_by_default',page.locator('#branch-viewer').get_attribute('open') is not None)
            check('viewer_stays_visible_beside_scrolling_messages',page.locator('#branch-viewer').evaluate('(e)=>e.getBoundingClientRect().top>=0 && e.getBoundingClientRect().bottom<innerHeight') and page.evaluate("document.querySelector('#branch-viewer').getBoundingClientRect().left>=document.querySelector('.chat-session').getBoundingClientRect().right && document.querySelector('#graph-canvas').dataset.orientation==='vertical'"))
            check('overview_has_branch_and_message_counts','3 branches · 6 messages' in page.locator('#chat-list').inner_text())
            check('shared_core_is_exact',page.locator('#graph-summary').inner_text()=='Shared core: 2 messages')
            check('subgroup_shared_core_is_visible',page.get_by_role('button',name='Inspect 2 saved messages, shared by 2 branches',exact=True).count()==1)
            check('branch_tips_have_counts',page.get_by_role('button',name='Open Pizza, 6 saved messages',exact=True).count()==1)
            check('shared_nodes_drawn_once',page.locator('[data-graph-node]').count()==6)
            check('segment_labels_do_not_overlap_branch_badges',page.evaluate("""() => {
                const tips=Array.from(document.querySelectorAll('[data-graph-branch]')).map(e=>e.getBoundingClientRect());
                return Array.from(document.querySelectorAll('[data-graph-segment]')).every(e=>{const r=e.getBoundingClientRect();return tips.every(t=>Math.min(r.right,t.right)<=Math.max(r.left,t.left) || Math.min(r.bottom,t.bottom)<=Math.max(r.top,t.top));});
            }"""))
            page.screenshot(path=str(output/(browser_name+'-three-paths.png')),full_page=True)
            page.locator('[data-graph-node="'+q['tip']+'"]').hover();page.wait_for_timeout(120)
            check('graph_point_tooltip_is_delayed',page.locator('[data-graph-node="'+q['tip']+'"]').evaluate("e=>getComputedStyle(e,'::after').opacity")=='0')
            page.wait_for_timeout(950)
            check('graph_point_tooltip_appears_after_hold',page.locator('[data-graph-node="'+q['tip']+'"]').evaluate("e=>Number(getComputedStyle(e,'::after').opacity)>0.9"))
            page.get_by_role('button',name='Inspect 2 saved messages, shared by 2 branches',exact=True).click()
            check('inspection_does_not_move_head',store.history(main)['revision']==6 and page.locator('#graph-inspector .graph-preview').count()==2)
            check('inspection_scrolls_preview_into_view',page.evaluate("() => {const row=document.querySelector('#graph-inspector .graph-preview').getBoundingClientRect(),viewer=document.querySelector('#branch-viewer').getBoundingClientRect();return row.top>=viewer.top && row.bottom<=viewer.bottom;}"))
            check('preview_has_fork_and_rewind',page.locator('#graph-inspector [data-workspace-fork]').count()==2 and page.locator('#graph-inspector [data-workspace-rewind]').count()==2)
            page.fill('#input','Unsent Main draft')
            page.locator('#file').set_input_files({'name':'draft.txt','mimeType':'text/plain','buffer':b'Keep this attachment.'})
            page.wait_for_function('attachments.length===1')
            page.locator('[data-graph-branch="'+b+'"]').click();ready(page,b)
            check('new_branch_has_its_own_draft',page.locator('#input').input_value()=='' and page.locator('#attachments').is_hidden())
            page.fill('#input','Unsent Pizza draft')
            page.locator('[data-graph-branch="'+main+'"]').click();ready(page,main)
            check('draft_text_is_preserved',page.locator('#input').input_value()=='Unsent Main draft')
            check('draft_attachment_is_preserved','draft.txt' in page.locator('#attachments').inner_text())
            page.locator('[data-graph-branch="'+b+'"]').click();ready(page,b)
            check('second_draft_is_preserved',page.locator('#input').input_value()=='Unsent Pizza draft')
            page.locator('[data-graph-branch="'+c+'"]').press('Enter');ready(page,c)
            check('keyboard_selects_branch',page.locator('[data-graph-branch="'+c+'"]').get_attribute('aria-current')=='true')
            page.locator('#branch-viewer > summary').click();page.reload(wait_until='domcontentloaded');ready(page,c)
            check('collapsed_preference_survives_reload',page.locator('#branch-viewer').get_attribute('open') is None)
            page.locator('#branch-viewer > summary').click()
            page.locator('[data-graph-branch="'+b+'"]').click();ready(page,b)
            page.locator('[data-graph-node="'+q['tip']+'"]').click()
            page.locator('#graph-inspector [data-workspace-rewind]').click()
            page.wait_for_function('messages.length===4 && !ContextWorkspace.moving');ready(page,b)
            check('rewind_is_immediate_and_recoverable',page.locator('#workspace-dialog').is_hidden() and old_b not in {n['id'] for n in store.graph(chat['id'])['nodes']})
            page.check('#graph-previous');page.locator('[data-graph-previous="'+old_b+'"]').wait_for()
            page.wait_for_function("document.querySelector('#graph-canvas').getAnimations({subtree:true}).every(a=>a.playState==='finished')")
            check('previous_tip_is_faded',page.locator('[data-graph-previous="'+old_b+'"]').evaluate('(e)=>getComputedStyle(e).opacity')=='0.55')
            page.screenshot(path=str(output/(browser_name+'-previous-tip.png')),full_page=True)
            page.locator('[data-graph-previous="'+old_b+'"]').click();page.locator('#graph-inspector [data-workspace-rewind]').click()
            page.wait_for_function('messages.length===6 && !ContextWorkspace.moving');ready(page,b)
            check('previous_tip_can_be_restored',store.history(b)['tip']==old_b)
            # A foreign tab appends: old revision must fail and refresh without moving the branch.
            before=store.history(b);external=add(b,'External tab update.')
            page.locator('[data-graph-node="'+q['tip']+'"]').click();page.locator('#graph-inspector [data-workspace-rewind]').click()
            page.wait_for_function('messages.length===7 && !ContextWorkspace.moving');ready(page,b)
            check('stale_action_refreshes_without_rewind',store.history(b)['tip']==external['tip'] and store.history(b)['revision']==before['revision']+1)
            page.locator('[data-graph-branch="'+main+'"]').click();ready(page,main)
            # A delayed old selection must not replace a newer selection.
            held=[];pattern='**/workspace/history?branch='+b
            page.route(pattern,lambda route:held.append(route))
            page.locator('[data-graph-branch="'+b+'"]').click();page.wait_for_timeout(100)
            check('delayed_selection_fixture_started',len(held)==1)
            page.locator('[data-graph-branch="'+c+'"]').click();ready(page,c)
            held[0].fulfill(status=200,content_type='application/json',body=json.dumps(store.history(b)))
            page.wait_for_timeout(100);page.unroute(pattern)
            check('late_response_does_not_replace_selection',page.evaluate('ContextWorkspace.current.id')==c)
            # Read failure does not prevent typing; refresh recovers the viewer and draft.
            page.route('**/workspace/graph?*',lambda route:route.fulfill(status=503,content_type='application/json',body='{"error":{"message":"Test network failure"}}'))
            page.uncheck('#graph-previous');page.wait_for_function('ContextGraph.stale')
            page.fill('#input','Draft while graph is offline')
            check('graph_failure_leaves_chat_usable',page.evaluate('messages.length')==4 and page.locator('#graph-refresh').is_visible())
            page.unroute('**/workspace/graph?*');page.click('#graph-refresh');page.wait_for_function('!ContextGraph.stale');ready(page,c)
            check('refresh_preserves_draft',page.locator('#input').input_value()=='Draft while graph is offline')
            page.get_by_role('button',name='Inspect 2 saved messages, shared by 3 branches',exact=True).click()
            page.locator('#graph-inspector [data-workspace-fork]').last.click()
            page.wait_for_function("ContextWorkspace.current.name==='Alternative 1' && !ContextWorkspace.moving")
            check('inspector_fork_is_immediate',page.locator('#workspace-dialog').is_hidden() and page.evaluate('messages.length')==2)
            check('inspector_fork_keeps_parent_history',len(store.history(c)['messages'])==4)
            page.screenshot(path=str(output/(browser_name+'-graph.png')),full_page=True)
            page.set_viewport_size({'width':390,'height':844})
            check('mobile_has_no_page_overflow',page.evaluate('document.documentElement.scrollWidth<=innerWidth'))
            check('graph_scrolls_inside_its_container',page.evaluate("(()=>{const e=document.querySelector('#graph-scroll');return e.scrollWidth>e.clientWidth || e.scrollHeight>e.clientHeight;})()"))
            page.screenshot(path=str(output/(browser_name+'-graph-mobile.png')),full_page=True)
            pending_chat=store.create_chat('Answer in progress','pending-chat');pending_branch=pending_chat['branch']['id']
            job=store.claim(pending_branch,0,{'role':'user','text':'Pending question.'},{},'pending-fixture')
            page.reload(wait_until='domcontentloaded');page.wait_for_selector('#workspace-sidebar:not([hidden])')
            page.locator('#chat-list .workspace-chat').filter(has_text='Answer in progress').click();ready(page,pending_branch)
            check('pending_answer_has_indicator','answering' in page.locator('[data-graph-branch="'+pending_branch+'"]').inner_text())
            check('pending_counts_only_durable_input',page.get_by_role('button',name='Open Main, 1 saved message',exact=True).count()==1 and len(store.history(pending_branch)['messages'])==1)
            store.finish(job['request_id'],{'role':'assistant','text':'Completed answer.'},'completed')
            page.locator('#graph-previous').set_checked(not page.locator('#graph-previous').is_checked())
            page.wait_for_function('ContextGraph.data.branches[0].saved_messages===2')
            check('completed_answer_increases_saved_count','answering' not in page.locator('[data-graph-branch="'+pending_branch+'"]').inner_text())
            page.click('#graph-refresh');page.wait_for_function('!ContextGraph.stale')
            check('completed_answer_is_in_saved_history',page.evaluate('messages.length')==2)
            # Isolated layout fixture complements the real 10K SQLite store test.
            report['large_layout']=page.evaluate("""() => {
                const old=ContextGraph.data,previous=ContextWorkspace.current;
                const nodes=[{id:'0',parent_id:null,depth:0,role:null,preview:'',current_branch_count:32}];
                for(let i=1;i<=10000;i++)nodes.push({id:String(i),parent_id:String(i-1),depth:i,role:'user',preview:'Large fixture',current_branch_count:Math.max(1,32-Math.max(0,i-9001))});
                const branches=Array.from({length:32},(_,i)=>({id:'b'+i,name:'Branch '+i,tip:String(i===31?10000:9001+i),saved_messages:i===31?10000:9001+i}));
                const fixture={root_id:'0',nodes,branches,previous_tips:[],shared_prefix_messages:9001};
                ContextGraph.data=fixture;ContextWorkspace.current={id:'b31',tip:'10000'};
                const started=performance.now();ContextGraph.render();
                const result={milliseconds:performance.now()-started,message_nodes:10000,branches:32,rendered_points:document.querySelectorAll('[data-graph-node]').length,rendered_segments:document.querySelectorAll('[data-graph-segment]').length};
                document.querySelector('[data-graph-segment]').click();result.preview_rows=document.querySelectorAll('#graph-inspector .graph-preview').length;
                ContextGraph.data=old;ContextWorkspace.current=previous;ContextGraph.render();document.querySelector('#graph-inspector').hidden=true;return result;
            }""")
            check('ten_thousand_messages_render_compactly',report['large_layout']['rendered_points']==33 and report['large_layout']['rendered_segments']==32)
            check('large_segment_preview_is_paged',report['large_layout']['preview_rows']==10)
            check('no_browser_errors',not errors)
            browser.close()
        report['status']='PASS'
    except BaseException as error:
        report.update(status='FAIL',error=repr(error))
        raise
    finally:
        server.shutdown();server.server_close();store.close();tmp.cleanup()
        (output/(browser_name+'-graph-results.json')).write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--browser',choices=['chromium','firefox','webkit'],default='chromium');parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();run(args.browser,args.output)
