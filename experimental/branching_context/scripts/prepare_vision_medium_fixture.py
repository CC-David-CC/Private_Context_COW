"""Prepare real, repeat-encoded images and independent positions for VC04/05.

This runs the pinned encoder, not the language model. Its PREPARED result is an
input fixture, not a passing model-continuation result.
"""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import sys

from PIL import Image, ImageDraw, __version__ as pillow_version

from run_vision_rebuild_probe import Encoder
from run_vision_ingest_probe import save, fingerprint
from vision_fixture import fixture_parts, reference_positions


def prepare(args):
    out = args.output.resolve(); out.mkdir(parents=True, exist_ok=False)
    profile = json.loads(args.profile.read_text())
    record = dict(status='RUNNING',scope='medium fixture preparation, not model qualification',
                  pillow_version=pillow_version,images=[],encoder_runs=[])
    try:
        encoder = Encoder(dict(encoder_baseline=str(args.encoder_baseline.resolve())),out,record)
        for index, size in enumerate((224,1024)):
            image = Image.new('RGB',(size,size),'white'); draw = ImageDraw.Draw(image)
            draw.rectangle((0,0,size//2-1,size-1),fill=(255,32*index,0))
            draw.rectangle((size//2,0,size-1,size-1),fill=(0,40,255-32*index))
            draw.rectangle((size//8,size//8,size//4,size//4),fill='green')
            draw.ellipse((size*3//4,size*3//4,size*7//8,size*7//8),fill='yellow')
            original = out/f'image-{index}.png'; image.save(original)
            payloads = []
            for repeat in range(2):
                normalized,payload = encoder.encode(original.read_bytes())
                assert normalized == original.read_bytes()
                payloads.append(payload)
            assert payloads[0] == payloads[1], 'Independent encoder repeats differ'
            payload = out/f'image-{index}.sve'; payload.write_bytes(payloads[0])
            magic,rows,nx,ny,width = struct.unpack('<5i',payloads[0][:20])
            assert magic == 0x31455653 and rows == nx*ny and rows <= 1024
            assert (nx,ny) == ((7,7) if index==0 else (32,32)), 'Retain unexpected encoder geometry before changing fixture'
            record['images'].append(dict(original=fingerprint(original),embedding=fingerprint(payload),
                rows=rows,grid_width=nx,grid_height=ny,width=width))
            save(out/'result.json',record)
        baseline = json.loads(args.encoder_baseline.read_text())
        encoder_contract = dict(argv=encoder.argv,files={path:baseline['files'][path] for path in
            (encoder.argv[0],encoder.argv[encoder.argv.index('--mmproj')+1])},
            model_metadata=fingerprint(profile['model_path']),normalization='generated-RGB-PNG-v1',
            encoder_baseline=fingerprint(args.encoder_baseline))
        record['encoder_profile_id'] = hashlib.sha256(json.dumps(encoder_contract,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        record['encoder_contract'] = encoder_contract
        sys.path.insert(0,str(Path(__file__).resolve().parents[3]/'tools'))
        from strata_tokenizer import Tokenizer
        tok = Tokenizer.from_gguf(profile['model_path'])
        root = Path(__file__).resolve().parents[3]
        files = [root/'include/strata/core/vision_layout.hpp',root/'serve/cow_text_adapter.py']
        source = '\n'.join(path.read_text() for path in files)
        body = tok.encode(source,parse_special=False)
        opening = tok.encode('<|im_start|>user\nStudy this code and these two diagrams.\n',parse_special=True)
        vstart = tok.encode('<|vision_start|>',parse_special=True)
        vend = tok.encode('<|vision_end|>',parse_special=True)
        pad = tok.encode('<|image_pad|>',parse_special=True)
        assert len(vstart)==len(vend)==len(pad)==1
        starts = [125,1030]; before = 125-len(opening)-len(vstart)
        first = opening+body[:before]+vstart
        middle_count = starts[1]-(len(first)+record['images'][0]['rows'])
        middle = vend+body[before:before+middle_count-2]+vstart
        tail_count = 1024-len(first)-len(middle)
        tail = vend+body[before+middle_count:before+middle_count+tail_count-1]
        assert len(first)+len(middle)+len(tail)==1024
        ids = first+pad*record['images'][0]['rows']+middle+pad*record['images'][1]['rows']+tail
        assert len(ids)==2097
        close = tok.encode('<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n',parse_special=True)
        suffixes = {}
        for name, side in (('A','left'),('B','right')):
            question = tok.encode(f'Explain the {side} half of both diagrams in two detailed paragraphs. '
                'Compare the shapes, colors and positions. Describe what stays the same and what differs. '
                'Include enough detail for a reader who cannot see either image and connect this to the code example. '
                'Explain the top and bottom markers, their relative sizes, the contrasting backgrounds, '
                'and the spatial relationship between all of these features.',parse_special=False)
            assert len(question)>=64-len(close)
            suffixes[name]=question[:64-len(close)]+close
        placements = [dict(start=start,rows=image['rows'],grid_width=image['grid_width'],
            grid_height=image['grid_height']) for start,image in zip(starts,record['images'])]
        positions = reference_positions(len(ids),placements,profile['execution']['capacity'])
        flat = [value for point in positions for value in point]
        reference = out/'reference-positions.i32'
        reference.write_bytes(struct.pack('<'+'i'*len(flat),*flat))
        lines = [f'COW_VISION_INPUT_V1 0 {len(ids)} 2',' '.join(map(str,ids))]
        for start,image in zip(starts,record['images']):
            lines.append(f"{start} {image['rows']} {image['grid_width']} {image['grid_height']} {image['width']} {pad[0]} "+
                image['embedding']['sha256']+' '+json.dumps(image['embedding']['path']))
        (out/'input.manifest').write_text('\n'.join(lines)+'\n')
        record.update(input_tokens=ids,suffixes=suffixes,text_control_tokens=1024,image_rows=1073,
            total_cells=len(ids),prefill_chunk=512,require_boundaries=True,source_files=[fingerprint(path) for path in files],
            positions=fingerprint(reference),image_starts=starts,status='PREPARED')
        save(out/'result.json',record)
        fixture_parts(out/'result.json',[Path(x['original']['path']) for x in record['images']])
    except BaseException as error:
        record.update(status='FAIL',error=repr(error));raise
    finally:save(out/'result.json',record)
    print(json.dumps(dict(status=record['status'],total_cells=len(ids),text_tokens=1024,image_rows=1073,
                         grids=[[x['grid_width'],x['grid_height']] for x in record['images']])),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('profile','encoder-baseline','output'):parser.add_argument('--'+name,type=Path,required=True)
    prepare(parser.parse_args())
