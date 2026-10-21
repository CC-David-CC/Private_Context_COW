"""Independent host check of the native streamed execution proof.

This is derived from canonical InputHistory, never a second stored history.
Native image identity covers the exact SVE1 bytes; original-upload provenance
remains in the canonical snapshot and checkpoint manifest independently.
"""
import hashlib,struct
from .input_spans import InputHistory
from .store import InvalidOperation

def native_binding(history):
 if not isinstance(history,InputHistory):raise InvalidOperation('Expected canonical mixed input view')
 cells=hashlib.sha256(b'COW-QWEN35-CELLS-v2\0');positions=hashlib.sha256(b'COW-QWEN35-POS-v2\0');cursor=0
 for span in history.spans:
  if span['kind']=='text':
   for token in span['tokens']:
    cells.update(b'T'+struct.pack('<i',token));positions.update(struct.pack('<iii',cursor,cursor,cursor));cursor+=1
  else:
   cells.update(b'I'+struct.pack('<5i',*(span[k] for k in ['rows','grid_width','grid_height','width','pad_token_id']))+span['embedding_sha256'].encode('ascii'))
   for y in range(span['grid_height']):
    for x in range(span['grid_width']):positions.update(struct.pack('<iii',cursor,cursor+y,cursor+x))
   cursor+=max(span['grid_width'],span['grid_height'])
 return dict(consumed=len(history),rotary_cursor=cursor,cells_sha256=cells.hexdigest(),positions_sha256=positions.hexdigest())
