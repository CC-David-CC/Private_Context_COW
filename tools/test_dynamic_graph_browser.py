"""Human-facing geometry and motion checks for the same-tip fork regression."""
import argparse
import json
from pathlib import Path
import sys
import tempfile

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))


def run(name,output):
    from playwright.sync_api import sync_playwright
    from serve.chat_workspace import ChatWorkspace
    from serve.frontend import ChatTemplate
    from serve.server import ByteTokenizer,MockEngine,Service,serve
    output.mkdir(parents=True,exist_ok=True);tmp=tempfile.TemporaryDirectory();tok=ByteTokenizer()
    svc=Service(MockEngine(tok,'</think>\n\nA new independent ending.'),tok,ChatTemplate(ROOT/'serve/chat_template.jinja'))
    svc.workspace=ChatWorkspace(Path(tmp.name)/'history.sqlite3');store=svc.workspace
    chat=store.create_chat('Branch design review','create');main=chat['branch']['id'];seq=0
    def add(branch,text,role='user'):
        nonlocal seq
        seq+=1;head=store.history(branch)
        return store.append(branch,head['revision'],{'role':role,'text':text,'time':seq},str(seq))
    add(main,'Keep the orchard reference.');p=add(main,'READY','assistant')
    a=store.fork(main,p['tip'],'A','fork-a')['id'];b=store.fork(main,p['tip'],'B','fork-b')['id']
    add(main,'Check the saved history.');add(main,'The history is saved.','assistant')
    add(main,'Continue the original path.');same=add(main,'This is the original ending.','assistant')
    clone=store.fork(main,same['tip'],'Alternative 1','same-tip')['id']
    add(a,'Use an apple example.');add(a,'Apples belong to path A.','assistant')
    add(b,'Use a pear example.');add(b,'Pears belong to path B.','assistant')
    server=serve(svc,port=0);url=f'http://127.0.0.1:{server.server_port}/'
    report={'browser':name,'status':'RUNNING','scope':'real SQLite and browser; mock model; no GPU qualification','checks':[]}
    def check(label,value):
        if not value:raise AssertionError(label)
        report['checks'].append(label)
    def settle(page):
        page.wait_for_function("ContextGraph.data && !ContextWorkspace.moving && ContextGraph.viewportWidth===document.querySelector('#graph-scroll').clientWidth && ContextGraph.viewportHeight===document.querySelector('#graph-scroll').clientHeight && document.querySelector('#graph-canvas').dataset.motion!=='running' && ContextGraph.data.branches.find(b=>b.id===ContextWorkspace.current.id)?.tip===ContextWorkspace.current.tip && document.querySelector('[data-graph-branch][aria-current=\"true\"]')?.dataset.graphBranch===ContextWorkspace.current.id && document.querySelector('#graph-canvas').getAnimations({subtree:true}).every(a=>a.playState==='finished')")
    def visible_badges(page):
        return page.evaluate("""() => {const v=document.querySelector('#graph-scroll').getBoundingClientRect();return Array.from(document.querySelectorAll('[data-graph-branch]')).every(b=>{const r=b.getBoundingClientRect();return r.top>=v.top && r.bottom<=v.bottom && r.left>=v.left && r.right<=v.right;});}""")
    def geometry(page):
        return page.evaluate("""() => Object.fromEntries(['branch-viewer','graph-scroll','chat-scroll','composer'].map(id=>{const e=document.getElementById(id),r=e.getBoundingClientRect();return [id,{top:r.top,bottom:r.bottom,width:r.width,height:r.height}];}).concat([['scale',ContextGraph.scale],['model', {width:ContextGraph.model(ContextGraph.data).width,height:ContextGraph.model(ContextGraph.data).height}]]))""")
    try:
        with sync_playwright() as pw:
            browser=getattr(pw,name).launch();options={'viewport':{'width':1500,'height':950}}
            if name=='chromium':options.update(record_video_dir=str(output/'video'),record_video_size={'width':1500,'height':950})
            page=browser.new_page(**options);video=page.video
            errors=[];page.on('pageerror',lambda e:errors.append(str(e)));page.goto(url);settle(page)
            check('vertical_viewer_is_beside_chat',page.evaluate("document.querySelector('#graph-canvas').dataset.orientation==='vertical' && document.querySelector('#branch-viewer').getBoundingClientRect().left>=document.querySelector('.chat-session').getBoundingClientRect().right"))
            check('four_branch_badges_are_fully_visible',visible_badges(page))
            check('same_tip_branches_use_one_point',page.locator('[data-graph-node="'+same['tip']+'"]').count()==1)
            check('same_tip_badges_are_grouped',page.evaluate("""([a,b])=>{const x=document.querySelector(`[data-graph-branch="${a}"]`),y=document.querySelector(`[data-graph-branch="${b}"]`);return x.dataset.graphTipGroup===y.dataset.graphTipGroup && Math.abs(x.offsetTop-y.offsetTop)===52;}""",[main,clone]))
            check('chat_has_room_to_read',page.locator('#chat-scroll').evaluate('(e)=>e.clientHeight>400'))
            report['desktop_geometry']=geometry(page)
            page.locator('[data-graph-branch="'+clone+'"]').click();page.wait_for_function('(id)=>ContextWorkspace.current.id===id',arg=clone);settle(page)
            check('selected_fork_is_not_clipped',visible_badges(page))
            check('unused_restore_action_is_disabled',page.locator('#restore-point-btn').is_disabled())
            page.get_by_role('button',name='Switch between light and dark',exact=True).click()
            page.wait_for_timeout(250);settle(page)
            page.screenshot(path=str(output/(name+'-desktop-dark.png')),full_page=True)
            frames=page.evaluate('ContextGraph.motionFramesTotal || 0');page.locator('#chat [data-workspace-fork]').last.click()
            page.wait_for_function("ContextWorkspace.current.name==='Alternative 2' && !ContextWorkspace.moving");settle(page)
            check('new_fork_reflows_without_clipping',visible_badges(page) and page.locator('[data-graph-branch]').count()==5)
            executed_frames=page.evaluate('ContextGraph.motionFramesTotal || 0')-frames
            check('connecting_lines_execute_motion_frames',executed_frames>0)
            report['motion']={'same_tip_fork_svg_frames':executed_frames,'duration_setting_ms':220}
            check('fork_keeps_original_messages',len(store.history(main)['messages'])==6)
            page.screenshot(path=str(output/(name+'-five-branches.png')),full_page=True)
            page.fill('#input','Give this branch its own ending.');page.click('#send-btn')
            page.wait_for_function('!busy && messages.length===8 && messages.every(m=>m.node_id)');settle(page)
            check('new_ending_reflows_graph',page.locator('[data-graph-node]').count()==6 and len(store.history(main)['messages'])==6)
            check('active_ending_is_in_view',page.locator('[data-graph-branch][aria-current="true"]').evaluate("e=>{const r=e.getBoundingClientRect(),v=document.querySelector('#graph-scroll').getBoundingClientRect();return r.top>=v.top && r.bottom<=v.bottom && r.left>=v.left && r.right<=v.right;}"))
            page.screenshot(path=str(output/(name+'-new-ending.png')),full_page=True)
            before=page.locator('#graph-scale').inner_text();page.click('#graph-zoom-in');settle(page)
            check('zoom_in_changes_scale',page.locator('#graph-scale').inner_text()!=before)
            page.click('#graph-zoom-in');settle(page)
            bounds=page.locator('#graph-scroll').bounding_box();before_pan=page.locator('#graph-scroll').evaluate('e=>e.scrollTop')
            page.mouse.move(bounds['x']+10,bounds['y']+16);page.mouse.down();page.mouse.move(bounds['x']+10,bounds['y']+56,steps=4);page.mouse.up()
            after_pan=page.locator('#graph-scroll').evaluate('e=>e.scrollTop')
            check('background_drag_pans_without_editing',after_pan!=before_pan and len(store.history(main)['messages'])==6)
            page.evaluate('ContextGraph.render()');settle(page)
            check('unchanged_refresh_preserves_pan',page.locator('#graph-scroll').evaluate('e=>e.scrollTop')==after_pan)
            page.click('#graph-fit');settle(page)
            check('fit_shows_all_branch_badges',visible_badges(page))
            page.set_viewport_size({'width':980,'height':900});settle(page)
            check('tablet_keeps_vertical_history_direction',page.locator('#graph-canvas').get_attribute('data-orientation')=='vertical')
            check('tablet_viewer_moves_above_chat',page.locator('#branch-viewer').evaluate("e=>e.getBoundingClientRect().bottom<=document.querySelector('.chat-session').getBoundingClientRect().top+1"))
            page.screenshot(path=str(output/(name+'-tablet.png')),full_page=True)
            page.set_viewport_size({'width':390,'height':844});settle(page)
            report['phone_geometry']=geometry(page);page.screenshot(path=str(output/(name+'-phone.png')),full_page=True)
            check('phone_does_not_overflow',page.evaluate('document.documentElement.scrollWidth<=innerWidth'))
            check('phone_keeps_vertical_history_direction',page.locator('#graph-canvas').get_attribute('data-orientation')=='vertical')
            check('phone_composer_is_visible',page.locator('#composer').evaluate('(e)=>e.getBoundingClientRect().bottom<=innerHeight'))
            check('phone_active_branch_is_in_view',page.locator('[data-graph-branch][aria-current="true"]').evaluate("e=>{const r=e.getBoundingClientRect(),v=document.querySelector('#graph-scroll').getBoundingClientRect();return r.top>=v.top && r.bottom<=v.bottom && r.left>=v.left && r.right<=v.right;}"))
            check('phone_keeps_readable_chat_space',page.locator('#chat-scroll').evaluate('(e)=>e.clientHeight>=80'))
            page.screenshot(path=str(output/(name+'-phone.png')),full_page=True)
            page.emulate_media(reduced_motion='reduce');page.set_viewport_size({'width':1500,'height':950});settle(page)
            frames=page.evaluate('ContextGraph.motionFramesTotal || 0');page.locator('#chat [data-workspace-fork]').last.click();page.wait_for_function("ContextWorkspace.current.name==='Alternative 3' && !ContextWorkspace.moving");settle(page)
            check('reduced_motion_has_no_layout_animation',page.evaluate("document.querySelector('#graph-canvas').getAnimations({subtree:true}).length===0"))
            check('reduced_motion_has_no_svg_motion_frames',page.evaluate('ContextGraph.motionFramesTotal || 0')==frames)
            report['motion']['reduced_motion_svg_frames']=page.evaluate('ContextGraph.motionFramesTotal || 0')-frames
            page.click('#graph-fit');settle(page);page.get_by_role('button',name='Inspect chat start',exact=True).click()
            check('chat_start_has_fork_and_rewind',page.locator('#graph-inspector [data-workspace-fork]').count()==1 and page.locator('#graph-inspector [data-workspace-rewind]').count()==1)
            page.locator('#graph-inspector [data-workspace-fork]').click();page.wait_for_function("ContextWorkspace.current.name==='Alternative 4' && !ContextWorkspace.moving");settle(page)
            check('root_fork_starts_empty_and_keeps_parent',page.evaluate('messages.length')==0 and len(store.history(main)['messages'])==6)
            check('no_browser_errors',not errors);page.context.close()
            if video:video.save_as(str(output/(name+'-branch-motion.webm')))
            browser.close()
        report['status']='PASS'
    except BaseException as e:
        report.update(status='FAIL',error=repr(e));raise
    finally:
        server.shutdown();server.server_close();store.close();tmp.cleanup()
        (output/(name+'-dynamic-results.json')).write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--browser',choices=['chromium','firefox','webkit'],default='chromium');p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();run(a.browser,a.output)
