"""Real native encoder protocol, geometry, finite payload and repeatability probe."""
import argparse,hashlib,json,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from context_time_travel.vision_embeddings import parse_sve

def main():
 p=argparse.ArgumentParser();p.add_argument('--binary',type=Path,required=True);p.add_argument('--model',type=Path,required=True);p.add_argument('--projector',type=Path,required=True);p.add_argument('--image',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--gpu',action='store_true');a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
 cmd=[str(a.binary),'--model',str(a.model),'--mmproj',str(a.projector),'--threads','12','--max-tokens','64','--flash-attn','on']+(['--gpu'] if a.gpu else [])
 results=[]
 for i in range(2):
  dst=a.output/('image-'+str(i)+'.sve');started=time.monotonic();x=subprocess.run(cmd,input='ENC '+str(a.image)+' '+str(dst)+'\nQUIT\n',text=True,capture_output=True,timeout=120)
  (a.output/('encoder-'+str(i)+'.stderr')).write_text(x.stderr);assert x.returncode==0,(x.returncode,x.stderr[-2000:]);lines=x.stdout.splitlines();assert len(lines)==2 and lines[0]=='READY 4096',lines
  b=dst.read_bytes();e=parse_sve(b,expected_sha256=hashlib.sha256(b).hexdigest(),expected_width=4096,max_rows=64);f=lines[1].split();assert f[0]=='OK' and tuple(map(int,f[1:4]))==(e.rows,e.grid_width,e.grid_height)
  results.append(dict(bytes=len(b),sha256=e.sha256,rows=e.rows,grid=[e.grid_width,e.grid_height],seconds=time.monotonic()-started,stdout=x.stdout))
 assert results[0]['sha256']==results[1]['sha256'],'Independently initialized encoders differ'
 result=dict(status='PASS',scope='Encoding only; decoder/image checkpoint continuation not yet tested',gpu=a.gpu,runs=results);(a.output/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
