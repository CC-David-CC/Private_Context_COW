"use strict";
const test=require('node:test');
const assert=require('node:assert/strict');
const fs=require('node:fs');
const vm=require('node:vm');
const source=fs.readFileSync(require('node:path').join(__dirname,'workspace.js'),'utf8');
const prefix=source.slice(0,source.indexOf('const ContextWorkspace ='));
const context={TextDecoder,window:{}};vm.runInNewContext(prefix,context);
const consume=context.window.CowProtocol.consumeCowEvents;
function response(chunks){let i=0;return{body:{getReader(){return{async read(){return i<chunks.length?{value:Buffer.from(chunks[i++]),done:false}:{done:true};}};}};}}

const complete={event:'complete',answer:{snapshot:'snap-2',text:'hello'},receipt:{branch:{name:'b1',revision:2,tip:'snap-2',root:'root'},snapshot:'snap-2'}};

test('split NDJSON text remains provisional until a complete receipt',async()=>{
  const seen=[];
  const result=await consume(response(['{"event":"text","text":"hel','lo"}\n'+JSON.stringify(complete)]),x=>seen.push(x));
  assert.deepEqual(seen,['hello']);assert.deepEqual(JSON.parse(JSON.stringify(result)),complete);
});

test('EOF with provisional text does not report success',async()=>{
  const seen=[];await assert.rejects(()=>consume(response(['{"event":"text","text":"partial"}\n']),x=>seen.push(x)),/without a durable COW receipt/);
  assert.deepEqual(seen,['partial']);
});

test('server error event rejects even after provisional text',async()=>{
  await assert.rejects(()=>consume(response(['{"event":"text","text":"partial"}\n{"event":"error","error":"cancelled"}\n']),()=>{}),/cancelled/);
});

test('final complete event may omit trailing newline',async()=>{
  const value=await consume(response([JSON.stringify({...complete,receipt:{...complete.receipt,branch:{...complete.receipt.branch,revision:1}}})]),()=>{});
  assert.equal(value.receipt.branch.revision,1);
});

test('malformed completion and duplicate completion are rejected',async()=>{
  await assert.rejects(()=>consume(response(['{"event":"complete","receipt":{"revision":1}}']),()=>{}),/receipt/);
  await assert.rejects(()=>consume(response([JSON.stringify({...complete,answer:{...complete.answer,snapshot:'other'}})]),()=>{}),/receipt/);
  await assert.rejects(()=>consume(response([JSON.stringify({...complete,receipt:{...complete.receipt,branch:{...complete.receipt.branch,revision:-1}}})]),()=>{}),/receipt/);
  await assert.rejects(()=>consume(response([JSON.stringify(complete)+'\n'+JSON.stringify(complete)]),()=>{}),/after complete/);
  await assert.rejects(()=>consume(response([JSON.stringify(complete)+'\n{"event":"text","text":"late"}']),()=>{}),/after complete/);
});
