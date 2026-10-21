"""Compare native selectors with independent arithmetic on a stored frontier."""
from pathlib import Path
import sys,argparse,ctypes as C,struct,time,json,hashlib
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from context_time_travel.min_p import choose

def main():
 p=argparse.ArgumentParser();p.add_argument('--library',type=Path,required=True);p.add_argument('--frontier',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 raw=a.frontier.read_bytes();assert raw and len(raw)%4==0
 values=struct.unpack('<'+str(len(raw)//4)+'f',raw);nv=len(values);array=(C.c_float*nv).from_buffer_copy(raw)
 lib=C.CDLL(str(a.library));f=lib.ctt_select_logits_v2;f.restype=C.c_int;f.argtypes=[C.c_void_p,C.c_size_t,C.c_void_p,C.c_size_t,C.c_double,C.c_double,C.c_uint64,C.c_size_t,C.c_double]
 rows=[]
 for label,pattern in [('all',255),('even',85)]:
  mask=bytearray([pattern])*((nv+7)//8)
  if nv%8:mask[-1]&=(1<<(nv%8))-1
  mask=bytes(mask)
  for temp,p,k,topp,draw in [(0,0,0,1,0),(.7,.05,0,1,123456789),(.7,.05,60,.95,2**52),(.8,0,1,.95,2**53-1)]:
   data=C.create_string_buffer(mask,len(mask));expected=choose(values,mask,temperature=temp,min_p=p,draw53=draw,top_k=k,top_p=topp)['token'];actual=f(array,nv,data,len(mask),temp,p,draw,k,topp)
   assert actual==expected,(label,temp,p,k,topp,expected,actual);rows.append(dict(mask=label,temperature=temp,min_p=p,top_k=k,top_p=topp,draw=draw,token=actual))
 empty=bytes((nv+7)//8);data=C.create_string_buffer(empty,len(empty));assert f(array,nv,data,len(empty),0,0,0,0,1)==-1
 mask=bytearray([255])*((nv+7)//8)
 if nv%8:mask[-1]&=(1<<(nv%8))-1
 mask=bytes(mask);data=C.create_string_buffer(mask,len(mask));start=time.perf_counter()
 for _ in range(100):f(array,nv,data,len(mask),0,0,0,0,1)
 native=(time.perf_counter()-start)/100;start=time.perf_counter()
 for _ in range(3):choose(values,mask,temperature=0,min_p=0,draw53=0)
 python=(time.perf_counter()-start)/3
 result=dict(status='PASS',scope='CPU selection on recorded frontier; not GPU inference evidence',library_sha256=hashlib.sha256(a.library.read_bytes()).hexdigest(),frontier_sha256=hashlib.sha256(raw).hexdigest(),cases=rows,empty_mask_rejected=True,native_selection_seconds=native,python_reference_seconds=python)
 a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='cases'}))
if __name__=='__main__':main()
