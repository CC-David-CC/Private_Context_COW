"""Reproducible PNG/JPEG orientation inputs and independent raster witnesses.

No encoder or model runs here. Expected rasters use literal row-major index
permutations, not the serving normalizer. JPEG references start from decoded
pixels, so this tests geometric normalization rather than JPEG losslessness.
"""
import hashlib,io,json
from pathlib import Path
from PIL import Image
ORDER={1:(3,2,(0,1,2,3,4,5)),2:(3,2,(2,1,0,5,4,3)),
 3:(3,2,(5,4,3,2,1,0)),4:(3,2,(3,4,5,0,1,2)),
 5:(2,3,(0,3,1,4,2,5)),6:(2,3,(3,0,4,1,5,2)),
 7:(2,3,(5,2,4,1,3,0)),8:(2,3,(2,5,1,4,0,3))}
PIXELS=((11,27,189),(220,9,41),(17,203,83),(251,130,6),(67,31,244),(5,160,191))
def prepare(directory):
 directory=Path(directory);directory.mkdir(mode=0o700,parents=True,exist_ok=True)
 if any(directory.iterdir()):raise FileExistsError('Fixture directory must be empty')
 rows=[]
 for format in ('PNG','JPEG'):
  for orientation,(width,height,order) in ORDER.items():
   image=Image.new('RGB',(3,2));image.putdata(PIXELS);exif=Image.Exif();exif[274]=orientation
   stream=io.BytesIO();image.save(stream,format=format,exif=exif,**({'quality':100,'subsampling':0} if format=='JPEG' else {}));raw=stream.getvalue()
   with Image.open(io.BytesIO(raw)) as decoded:
    decoded.load();pixels=list(decoded.convert('RGB').getdata())
   path=directory/(format.lower()+'-orientation-'+str(orientation)+('.png' if format=='PNG' else '.jpg'));path.write_bytes(raw)
   row=dict(name=path.stem,format=format,orientation=orientation,original=str(path.resolve()),original_sha256=hashlib.sha256(raw).hexdigest(),original_bytes=len(raw),expected_size=[width,height],expected_pixels=[list(pixels[i]) for i in order]);rows.append(row)
 (directory/'manifest.json').write_text(json.dumps(dict(schema=1,status='PREPARED_NOT_GPU_EXECUTED',cases=rows),indent=2,sort_keys=True)+'\n',newline='\n');return rows

def check_normalized(row,payload):
 try:
  with Image.open(io.BytesIO(payload)) as image:
   image.load();pixels=[list(x) for x in image.convert('RGB').getdata()]
   if list(image.size)!=row['expected_size'] or pixels!=row['expected_pixels']:raise ValueError('Normalized pixel raster differs from independent fixture')
   if image.getexif().get(274) not in (None,1):raise ValueError('Normalized image retains active orientation')
 except (OSError,SyntaxError) as error:raise ValueError('Invalid normalized pixel payload') from error
 return dict(width=row['expected_size'][0],height=row['expected_size'][1],pixels=len(pixels),payload_sha256=hashlib.sha256(payload).hexdigest())

def case_plan(rows):
 if (len(rows)!=16 or len({x['name'] for x in rows})!=16
     or {(x['format'],x['orientation']) for x in rows}!={(f,o) for f in ('PNG','JPEG') for o in range(1,9)}):
  raise ValueError('The matrix requires all16 unique PNG/JPEG orientation cases')
 by_key={(x['format'],x['orientation']):x['name'] for x in rows}
 a=by_key[('PNG',1)];b=by_key[('JPEG',6)]
 return [dict(name=x['name'],images=[x['name']]) for x in rows]+[
  dict(name='ordered-png-jpeg',images=[a,b]),dict(name='ordered-jpeg-png',images=[b,a])]
