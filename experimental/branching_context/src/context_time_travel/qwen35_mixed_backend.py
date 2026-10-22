"""Versioned mixed-input extension of the pinned Qwen adapter.

The existing immutable InputHistory and controller remain canonical. Native
version 2 adds logical/rotary cursors and streamed execution proofs to its
optional derived checkpoint; it never imports version 1 text checkpoint bytes.
"""
import ctypes as C
from .qwen35_backend import Qwen35Backend,Qwen35FreshOracle
from .input_spans import InputHistory
from .store import IntegrityError,InvalidOperation
from .vision_execution import VISION_PROTOCOL,RESUME_CONTRACT,POSITION_CONVENTION,input_binding,validate_boundary
from .vision_embeddings import Embedding,parse_sve
from .qwen35_mixed_binding import native_binding
from .execution_receipt import tokens_digest,validate_input_schedule

class Qwen35MixedBackend(Qwen35Backend):
 STATE_FORMAT='qwen35-hybrid-mrope-frontier-v2'
 RESUME_CONTRACT=RESUME_CONTRACT
 def __init__(self,profile,gpu_id=None):
  e=profile['execution']
  if e.get('vision_protocol')!=VISION_PROTOCOL or e.get('native_binding_protocol')!='qwen35-streamed-cells-mrope-v2':raise IntegrityError('Unsupported mixed Qwen binding protocol')
  self._inputs=None
  super().__init__(profile,gpu_id=gpu_id)
  self.lib.ctt_state_version.restype=C.c_int
  if self.lib.ctt_state_version()!=2:self.close();raise IntegrityError('Native mixed state version mismatch')
  self.lib.ctt_proof.argtypes=[C.c_void_p,C.c_void_p,C.POINTER(C.c_int),C.POINTER(C.c_int)];self.lib.ctt_proof.restype=C.c_int
  self.lib.ctt_temporal_position.argtypes=[C.c_void_p];self.lib.ctt_temporal_position.restype=C.c_int
  self.lib.ctt_poison.argtypes=[C.c_void_p];self.lib.ctt_poison.restype=None
  self.lib.ctt_consume_image.argtypes=[C.c_void_p,C.c_void_p,C.c_size_t,C.c_int,C.c_char_p];self.lib.ctt_consume_image.restype=C.c_int
 def contract_fields(self):
  return dict(super().contract_fields(),positions=POSITION_CONVENTION,input_kind='canonical-mixed-spans-v1')
 def _acquire(self):
  lease=super()._acquire();self._inputs=InputHistory([])
  try:self._check_binding(lease,self._inputs);return lease
  except BaseException:super()._release(lease);self._inputs=None;raise
 def _release(self,lease):
  super()._release(lease);self._inputs=None
 def _native_binding(self,lease):
  self._check(lease);out=C.create_string_buffer(64);count=C.c_int();cursor=C.c_int()
  if self.lib.ctt_proof(lease,out,C.byref(count),C.byref(cursor)):raise IntegrityError(self.lib.ctt_error().decode())
  return dict(consumed=count.value,rotary_cursor=cursor.value,cells_sha256=out.raw[:32].hex(),positions_sha256=out.raw[32:].hex())
 def _check_binding(self,lease,history):
  actual=self._native_binding(lease);expected=native_binding(history)
  if actual!=expected:raise IntegrityError('Native mixed inputs/positions differ from canonical history')
  if history and history.spans[-1]['kind']=='text' and self.lib.ctt_temporal_position(lease)!=expected['rotary_cursor']-1:raise IntegrityError('Native temporal memory cursor differs from history')
 def _bind_receipt(self,lease,history,receipt):
  self._check_binding(lease,history);self._inputs=history
  self.last_execution_receipt=dict(receipt,inputs_sha256=history.digest,**input_binding(history))
 def consume(self,lease,ids):
  self._check(lease)
  if not isinstance(self._inputs,InputHistory):raise IntegrityError('Missing active mixed input boundary')
  ids=tuple(ids)
  if not ids or any(type(t) is not int or not 0<=t<self.vocab_size for t in ids) or len(self._inputs)+len(ids)>self.profile['execution']['capacity']:raise InvalidOperation('Invalid tokens or mixed capacity exceeded')
  history=self._inputs+ids
  try:
   super().consume(lease,ids);self._bind_receipt(lease,history,self.last_execution_receipt)
  except BaseException:self.lib.ctt_poison(lease);self._inputs=None;raise
 consume_generated=consume
 def consume_inputs(self,lease,spans,*,embeddings):
  self._check(lease)
  if not isinstance(self._inputs,InputHistory):raise IntegrityError('Missing active mixed input boundary')
  prior=self._inputs;delta=InputHistory(spans);history=prior+delta;e=self.profile['execution']
  validate_boundary(history,capacity=e['capacity'],vocab=self.vocab_size,width=e['embedding_width'])
  payloads={}
  for span in delta.spans:
   if span['kind']!='image':continue
   item=embeddings.get(span['embedding_sha256'])
   if not isinstance(item,Embedding):raise IntegrityError('Missing leased image embedding')
   checked=parse_sve(item.payload,expected_sha256=span['embedding_sha256'],expected_width=e['embedding_width'],max_rows=1024)
   if (checked.rows,checked.grid_width,checked.grid_height,checked.nbytes)!=(span['rows'],span['grid_width'],span['grid_height'],span['embedding_bytes']):raise IntegrityError('Image metadata differs from payload')
   payloads[span['embedding_sha256']]=checked.payload
  chunks=[]
  try:
   for span in delta.spans:
    if span['kind']=='text':
     super().consume(lease,span['tokens']);chunks.extend(self.last_execution_receipt['chunks'])
    else:
     start=self.position(lease)+1;payload=payloads[span['embedding_sha256']];data=C.create_string_buffer(payload,len(payload));before=self.lib.ctt_work(lease);calls=self.lib.ctt_calls(lease)
     if self.lib.ctt_consume_image(lease,data,len(payload),span['pad_token_id'],span['embedding_sha256'].encode()):raise IntegrityError(self.lib.ctt_error().decode())
     work=self.lib.ctt_work(lease)-before;n_calls=self.lib.ctt_calls(lease)-calls
     if work!=span['rows'] or n_calls!=(span['rows']+e['prefill_chunk']-1)//e['prefill_chunk']:raise IntegrityError('Native image work accounting mismatch')
     self.stats['consumed_tokens']+=work;self.stats['forward_calls']+=n_calls
     chunks.extend([[start+i,min(e['prefill_chunk'],span['rows']-i)] for i in range(0,span['rows'],e['prefill_chunk'])])
   receipt=dict(operator='batched-prefill',position=len(prior),count=len(delta),tokens_sha256=tokens_digest(delta.native_tokens),chunks=chunks)
   self._bind_receipt(lease,history,receipt)
  except BaseException:self.lib.ctt_poison(lease);self._inputs=None;raise
 def restore(self,lease,blob,*,history,frontier):
  self._check(lease)
  if not isinstance(history,InputHistory):raise IntegrityError('Mixed restore requires canonical InputHistory')
  validate_boundary(history,capacity=self.profile['execution']['capacity'],vocab=self.vocab_size,width=self.profile['execution']['embedding_width'])
  try:
   result=super().restore(lease,blob,history=history,frontier=frontier);self._check_binding(lease,history);self._inputs=history;return result
  except BaseException:self.lib.ctt_poison(lease);self._inputs=None;raise
 def export(self,lease):
  if not isinstance(self._inputs,InputHistory):raise IntegrityError('Missing consumed mixed state')
  self._check_binding(lease,self._inputs);return super().export(lease)

class Qwen35MixedOracle(Qwen35FreshOracle):
 def __init__(self,profile,*,embeddings,gpu_id=None):
  import tempfile
  super().__init__(profile,gpu_id=gpu_id);self._directory=tempfile.TemporaryDirectory(prefix='qwen35-oracle-');self._paths={}
  from pathlib import Path
  for key,item in embeddings.items():
   p=Path(self._directory.name)/(key+'.sve');p.write_bytes(item.payload);self._paths[key]=str(p)
 def forward_inputs(self,history,schedule):
  import json,struct,hashlib
  validate_input_schedule(schedule,history,self.profile['execution']['prefill_chunk'])
  chunks=[chunk for event in schedule for chunk in event['chunks']]
  manifest=dict(schema=1,spans=history.spans,chunks=chunks,embedding_paths=self._paths)
  self.process.stdin.write(('M '+json.dumps(manifest,separators=(',',':'))+'\n').encode());self.process.stdin.flush()
  n,work,position=struct.unpack('<iQi',self._read(16));expected=input_binding(history)
  if n!=self.profile['vocab_size'] or not 0<n<=2**20:raise IntegrityError('Invalid mixed oracle vocabulary')
  if work!=len(history) or position!=expected['rotary_cursor']-1:raise IntegrityError('Mixed oracle omitted work or positions')
  raw=self._read(n*4);self.requests+=1;self.tokens_computed+=work;self.last=dict(computed_tokens=work,position_max=position,logits_sha256=hashlib.sha256(raw).hexdigest(),chunks=chunks)
  return struct.unpack('<'+str(n)+'f',raw)
 def close(self):
  try:super().close()
  finally:self._directory.cleanup()
