"""Rebuild the pinned vision driver with a checked mixed-text DRY command.

Retains every existing command. No model kernel, session object, checkpoint
format or normal text sampler is replaced. The fresh model oracle stays separate.
"""
import argparse
import json
from pathlib import Path
import shlex
import subprocess
import time

from build_min_p_selector_probe import sha256

PROTOCOL='mixed-text-runs-dry-u53-v1'


def once(source,old,new):
    if source.count(old)!=1:raise ValueError('Expected exactly one source anchor: '+old[:100])
    return source.replace(old,new,1)


def instrument(source):
    anchor='        } else if (command == "sample_mask_v3") {'
    if source.count(anchor)!=1:raise ValueError('Expected one retained V3 command')
    begin=source.index(anchor);end=source.index('        } else if (command ==',begin+len(anchor))
    command=source[begin:end]
    command=command.replace('sample_mask_v3','sample_mask_image_dry_v1')
    command=once(command,'''            const size_t expected_history = dry_last_n_arg == 0 ? 0 :
                std::min<uint64_t>(actual_consumed, dry_last_n_arg);''', '''            const auto window = cow_vision_dry::project(session.ids(), session.input_images(),
                static_cast<size_t>(dry_last_n_arg));
            const size_t expected_history = window.tokens.size();''')
    command=once(command,'static_cast<uint32_t>(native_ids[native_ids.size() - history.size() + i])','window.tokens[i]')
    command=once(command,'            const auto& native_ids = session.ids();\n','')
    command=once(command,'''                selection = min_p_select::select_with_dry(frontier.data(), vocab, mask.data(), mask.size(),
                    temperature, min_p, draw53, static_cast<size_t>(top_k_arg), top_p,
                    history, breakers, min_p_select::DryConfig{dry_multiplier, dry_base,
                        static_cast<uint32_t>(dry_allowed_arg), static_cast<uint32_t>(dry_last_n_arg)});''', '''                size_t affected = 0;
                const auto adjusted = min_p_select::apply_dry_penalties(frontier.data(), vocab,
                    history, breakers, min_p_select::DryConfig{dry_multiplier, dry_base,
                        static_cast<uint32_t>(dry_allowed_arg), static_cast<uint32_t>(dry_last_n_arg)},
                    &affected, &window.boundaries);
                selection = min_p_select::select(adjusted.data(), vocab, mask.data(), mask.size(),
                    temperature, min_p, draw53, static_cast<size_t>(top_k_arg), top_p);
                selection.dry_penalized_candidates = affected;''')
    command=command.replace('full-vocab-dry-min-p-u53-v3',PROTOCOL)
    for field in ('calls','ns','logit_bytes','mask_bytes'):
        command=command.replace('sampling_v3_'+field,'sampling_mixed_dry_'+field)
    command=once(command,r'<< ",\"dry_history_count\":" << history.size()',
        r'<< ",\"text_tokens\":" << window.text_count << ",\"dry_boundary_count\":" << window.boundaries.size() '+
        r'<< ",\"dry_history_count\":" << history.size()')
    command=once(command,r'<< ",\"top_p\":" << top_p',
        r'<< ",\"temperature\":" << temperature << ",\"min_p\":" << min_p << ",\"top_p\":" << top_p')
    source=once(source,'#include "vision_ingest.h"','#include "vision_ingest.h"\n#include "vision_dry.h"')
    source=once(source,'    std::string line;',
        '    uint64_t sampling_mixed_dry_calls=0,sampling_mixed_dry_ns=0,\n'
        '        sampling_mixed_dry_logit_bytes=0,sampling_mixed_dry_mask_bytes=0;\n    std::string line;')
    source=once(source,anchor,command+anchor)
    source=once(source,r'<< ",\"validated_embedding_bytes\":" << vision_validated_bytes',
        r'<< ",\"vision_dry_protocol\":\"'+PROTOCOL+r'\"" '+
        r'<< ",\"validated_embedding_bytes\":" << vision_validated_bytes')
    stats=r'<< ",\"sampling_v3_ns\":" << sampling_v3_ns'
    source=once(source,stats,stats+
        ''.join(r' << ",\"sampling_mixed_dry_'+field+r'\":" << sampling_mixed_dry_'+field
            for field in ('calls','ns','logit_bytes','mask_bytes')))
    return source


def build(receipt_path,root,out):
    receipt=json.loads(receipt_path.read_text());root=root.resolve(strict=True);out=out.resolve()
    if receipt['status']!='BUILT_NOT_EXECUTED':raise ValueError('Unexpected pinned build receipt')
    for group in receipt['dependencies'].values():
        for path,digest in group.items():
            if sha256(Path(path))!=digest:raise ValueError('Changed pinned dependency: '+path)
    for path,digest in receipt['verified_unchanged_inputs'].items():
        if sha256(Path(path))!=digest:raise ValueError('Changed pinned link input: '+path)
    compile_row=next(row for row in receipt['commands'] if row['label']=='cow-compile')
    link_row=next(row for row in receipt['commands'] if row['label']=='cow-link')
    old_source=Path(compile_row['argv'][compile_row['argv'].index('-c')+1]);old_obj=compile_row['argv'][compile_row['argv'].index('-o')+1]
    if sha256(old_source)!=receipt['dependencies']['cow'][str(old_source)]:raise ValueError('Changed pinned driver')
    out.mkdir(parents=True,exist_ok=False)
    cpp=out/'branch_driver.cpp';cpp.write_text(instrument(old_source.read_text()),newline='\n')
    headers={}
    for name in ('min_p_select.h','vision_dry.h'):
        path=root/'experimental/branching_context/native'/name;(out/name).write_bytes(path.read_bytes())
        headers[str(path)]=sha256(path)
    obj=out/'branch_driver.o';binary=out/'strata-vision-dry-cow';dep=out/'branch_driver.d'
    compile_args=[str(cpp) if arg==str(old_source) else str(obj) if arg==old_obj else arg for arg in compile_row['argv']]
    compile_args[1:1]=['-I'+str(out)]
    compile_args[compile_args.index('-MF')+1]=str(dep)
    link_args=[str(obj) if arg==old_obj else arg for arg in link_row['argv']]
    link_args[link_args.index('-o')+1]=str(binary)
    report=dict(status='BUILDING',source_receipt=dict(path=str(receipt_path),sha256=sha256(receipt_path)),
        base_driver=dict(path=str(old_source),sha256=sha256(old_source)),headers=headers,
        generated_driver=dict(path=str(cpp),sha256=sha256(cpp)),commands=[],protocol=PROTOCOL)
    def save():(out/'build.json').write_text(json.dumps(report,indent=2)+'\n')
    save()
    try:
        for label,argv,cwd in [('compile',compile_args,compile_row['cwd']),('link',link_args,link_row['cwd'])]:
            begin=time.monotonic();result=subprocess.run(argv,cwd=cwd,capture_output=True)
            (out/(label+'.stdout')).write_bytes(result.stdout);(out/(label+'.stderr')).write_bytes(result.stderr)
            report['commands'].append(dict(label=label,argv=argv,cwd=cwd,exit_code=result.returncode,seconds=time.monotonic()-begin));save()
            if result.returncode:raise RuntimeError(label+' failed; retained logs')
        dependencies=shlex.split(dep.read_text().replace('\\\n',' ').split(':',1)[1])
        report.update(status='BUILT_NOT_EXECUTED',binary=dict(path=str(binary),sha256=sha256(binary),bytes=binary.stat().st_size),
            dependencies={str(Path(p).resolve()):sha256(Path(p)) for p in dependencies})
    except BaseException as error:report.update(status='BUILD_FAILED',error=repr(error));raise
    finally:save()
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('receipt','root','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();print(json.dumps(build(a.receipt,a.root,a.output)))
