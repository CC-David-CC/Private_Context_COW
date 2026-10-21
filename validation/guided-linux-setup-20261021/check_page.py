def structure(page,target,expect):
    expect(page.locator('h1')).to_be_visible()
    assert 'Buddy' in page.locator('body').inner_text()
    assert page.locator('meta[name="viewport"]').count()==1
    assert page.evaluate('document.documentElement.scrollWidth <= innerWidth'), 'horizontal page overflow'

def interact(page,target,expect,actions):
    page.locator('button').first.click()
    expect(page.locator('#helloMessage')).to_be_visible()
    message=page.locator('#helloMessage').inner_text()
    assert 'hello' in message.lower() and 'Buddy' in message
    actions.append(['real greeting click',message[:256]])
    weights=page.evaluate("""() => {
      const out=[]; const walker=document.createTreeWalker(document.body,NodeFilter.SHOW_TEXT);
      while(walker.nextNode()) {
        const node=walker.currentNode, p=node.parentElement;
        if(!node.nodeValue.includes('Buddy') || !p || ['SCRIPT','STYLE','TITLE'].includes(p.tagName))continue;
        if(p.getClientRects().length && getComputedStyle(p).visibility !== 'hidden')
          out.push({text:node.nodeValue,weight:getComputedStyle(p).fontWeight});
      }
      return out;
    }""")
    assert weights and all(int(row['weight'])>=600 for row in weights), repr(weights)
    actions.append(['visible dog names bold',repr(weights)[:256]])

def pixels(path,page,target,revision):
    from PIL import Image
    with Image.open(path) as image:
        assert image.width>0 and image.height>0
        assert len(image.convert('RGB').getcolors(image.width*image.height) or [])>8
