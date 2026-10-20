"""Prepare exact-count numeric fixtures from retained real encoder output.

No encoder/inference runs here. Encoder provenance and the immutable SVE bytes
are retained inputs; cold live encoding is a separate VC09 measurement path.
"""
import argparse,hashlib,json,struct,sys
from pathlib import Path
from vision_fixture import fingerprint,reference_positions,fixture_parts

def prepare(args):
 out=args.output.resolve();out.mkdir(mode=0o700,parents=True,exist_ok=False)
 profile=json.loads(args.profile.read_bytes());source=json.loads(args.images.read_bytes())
 count,suffix,outputs={'tiny':(128,32,8),'medium':(1024,64,16),'target':(8192,512,128)}[args.size]
 images=source['images'][:1 if args.size=='tiny' else 2]
 for image in images:
  for field in ['original','embedding']:
   if fingerprint(image[field]['path'])!=image[field]:raise ValueError('retained image fixture changed')
 sys.path.insert(0,str(args.source/'tools'))
 from strata_tokenizer import Tokenizer
 tokenizer=Tokenizer.from_gguf(profile['model_path'])
 files=[args.source/'include/strata/core/vision_layout.hpp',args.source/'serve/cow_text_adapter.py',args.source/'experimental/branching_context/src/context_time_travel/coordinator.py']
 code=tokenizer.encode('\n'.join(p.read_text() for p in files),parse_special=False)
 opening=tokenizer.encode('<|im_start|>user\nStudy the following code and diagrams. A question will follow the shared material.\n',parse_special=True)
 vs=tokenizer.encode('<|vision_start|>',parse_special=True);ve=tokenizer.encode('<|vision_end|>',parse_special=True);pad=tokenizer.encode('<|image_pad|>',parse_special=True)
 assert len(vs)==len(ve)==len(pad)==1 and len(code)>count
 starts=[64] if args.size=='tiny' else [125,1030 if args.size=='medium' else 4094]
 text_counts=[starts[0]]
 if len(images)==2:text_counts.append(starts[1]-(starts[0]+images[0]['rows']))
 text_counts.append(count-sum(text_counts));cursor=0;ids=[];placements=[]
 for index,image in enumerate(images):
  prefix=opening if index==0 else ve
  n=text_counts[index]-len(prefix)-len(vs)
  assert n>0
  ids+=prefix+code[cursor:cursor+n]+vs;cursor+=n
  assert len(ids)==starts[index]
  placements.append(dict(start=len(ids),rows=image['rows'],grid_width=image['grid_width'],grid_height=image['grid_height']))
  ids+=pad*image['rows']
 n=text_counts[-1]-len(ve);ids+=ve+code[cursor:cursor+n]
 close=tokenizer.encode('<|im_end|>\n<|im_start|>assistant\n<think>\n',parse_special=True)
 suffixes={}
 for name,side in [('A','left'),('B','right')]:
  question=(f'Focus on the {side} half of each diagram. Explain the colors, shapes and spatial relationships, then relate your observations to the supplied code. '
   'Consider coordinate transforms, immutable state, ownership, cache lifetime and how a fresh continuation should be checked. '
   'Describe the top and bottom markers and compare the two diagrams carefully. ')
  body=tokenizer.encode(question+('\nAdditional relevant code for this question:\n'+'\n'.join(p.read_text() for p in files)),parse_special=False)
  suffixes[name]=body[:suffix-len(close)]+close
 assert len(ids)==count+sum(x['rows'] for x in images) and suffixes['A']!=suffixes['B']
 record=dict(schema=1,status='PREPARED_NOT_EXECUTED',size=args.size,text_control_tokens=count,image_rows=sum(x['rows'] for x in images),
  total_cells=len(ids),input_tokens=ids,suffixes=suffixes,suffix_tokens=suffix,outputs=outputs,images=images,
  prefill_chunk=profile['execution']['prefill_chunk'],execution_profile_id=profile['execution_profile_id'],
  encoder_profile_id=source['encoder_profile_id'],encoder_provenance=fingerprint(args.images),source_files=[fingerprint(p) for p in files],
  source_description='Real source-code text and retained independently repeated real SVE inputs; explicit assistant thinking prefix, bounded numeric generation, not answer-quality scoring')
 position=out/'reference-positions.i32';points=reference_positions(len(ids),placements,profile['execution']['capacity']);position.write_bytes(struct.pack('<%di'%(len(points)*3),*(v for point in points for v in point)))
 record['positions']=fingerprint(position)
 lines=[f'COW_VISION_INPUT_V1 0 {len(ids)} {len(images)}',' '.join(map(str,ids))]
 for place,image in zip(placements,images):lines.append(' '.join(map(str,[place['start'],image['rows'],image['grid_width'],image['grid_height'],image['width'],pad[0],image['embedding']['sha256']]))+' '+json.dumps(image['embedding']['path']))
 (out/'input.manifest').write_text('\n'.join(lines)+'\n',newline='\n');(out/'result.json').write_text(json.dumps(record,indent=2)+'\n',newline='\n')
 fixture_parts(out/'result.json',[Path(x['original']['path']) for x in images])
 print(json.dumps({k:record[k] for k in ['status','size','text_control_tokens','image_rows','total_cells','suffix_tokens','outputs']},sort_keys=True))

if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__)
 for name in ['profile','images','source','output']:p.add_argument('--'+name,type=Path,required=True)
 p.add_argument('--size',choices=['tiny','medium','target'],required=True);prepare(p.parse_args())
