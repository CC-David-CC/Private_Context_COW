"""Add checked image DRY/filter commands without changing frozen selectors.

Projection and penalties are derived from native IDs/image placements. The
existing selectors receive the adjusted temporary frontier with DRY disabled,
so no repetition penalty is applied twice. Retained commands stay unchanged.
"""
import argparse
import json
from pathlib import Path
import shlex
import subprocess
import time

from build_vision_dry_probe import once
from build_min_p_selector_probe import sha256

PROTOCOL='mixed-text-runs-sampling-u53-v2'
V4='full-vocab-typical-dyna-min-p-u53-v4'


def command_v4(command):
    if command.count('} else if (command == "sample_mask_v4") {')!=1:
        raise ValueError('Expected exactly one retained V4 command')
    command=command.replace('sample_mask_v4','sample_mask_image_v4')
    command=once(command,'''            const size_t expected_history = dry_active ?
                std::min<uint64_t>(actual_consumed, dry_last_n_arg) : 0;''','''            const auto window = cow_vision_dry::project(session.ids(), session.input_images(),
                dry_active ? static_cast<size_t>(dry_last_n_arg) : 0);
            const size_t expected_history = window.tokens.size();
            std::string boundary_json = "[";
            for (size_t i = 0; i < window.boundaries.size(); ++i) {
                if (i != 0) boundary_json += ",";
                boundary_json += std::to_string(window.boundaries[i]);
            }
            boundary_json += "]";''')
    command=once(command,'            const auto& native_ids = session.ids();\n','')
    command=once(command,'static_cast<uint32_t>(native_ids[native_ids.size() - history.size() + i])','window.tokens[i]')
    command=once(command,'''                selection = min_p_select::select_v4(frontier.data(), vocab, mask.data(), mask.size(),
                    temperature, min_p, static_cast<size_t>(top_k_arg), top_p, draw53,
                    history, breakers, min_p_select::DryConfig{dry_multiplier, dry_base,
                        static_cast<uint32_t>(dry_allowed_arg), static_cast<uint32_t>(dry_last_n_arg)},
                    typical_p, dyna_range, dyna_exponent);''','''                size_t segmented_affected = 0;
                const auto adjusted = min_p_select::apply_dry_penalties(frontier.data(), vocab,
                    history, breakers, min_p_select::DryConfig{dry_multiplier, dry_base,
                        static_cast<uint32_t>(dry_allowed_arg), static_cast<uint32_t>(dry_last_n_arg)},
                    &segmented_affected, &window.boundaries);
                selection = min_p_select::select_v4(adjusted.data(), vocab, mask.data(), mask.size(),
                    temperature, min_p, static_cast<size_t>(top_k_arg), top_p, draw53,
                    {}, {}, min_p_select::DryConfig{0.0, dry_base,
                        static_cast<uint32_t>(dry_allowed_arg), static_cast<uint32_t>(dry_last_n_arg)},
                    typical_p, dyna_range, dyna_exponent);
                selection.dry_penalized_candidates = segmented_affected;''')
    command=once(command,'++sampling_calls; ++sampling_v4_calls;',
        '++sampling_calls; ++sampling_v4_calls; ++sampling_image_dry_combination_calls;')
    anchor=r'<< ",\"dry_history_count\":" << history.size()'
    command=once(command,anchor,
        r'<< ",\"vision_dry_sampling_protocol\":\"'+PROTOCOL+r'\"" '+
        r'<< ",\"text_tokens\":" << window.text_count << ",\"dry_boundary_count\":" << window.boundaries.size() '+
        r'<< ",\"dry_boundaries\":" << boundary_json '+anchor)
    return command


def instrument(source):
    anchor='        } else if (command == "sample_mask_v4") {'
    if source.count(anchor)!=1 or 'command == "sample_mask_image_v4"' in source:
        raise ValueError('Expected one unextended retained V4 command')
    begin=source.index(anchor);end=source.index('        } else if (command ==',begin+len(anchor))
    extended=command_v4(source[begin:end])
    source=once(source,anchor,extended+anchor)
    source=once(source,'    std::string line;',
        '    uint64_t sampling_image_dry_combination_calls=0;\n    std::string line;')
    info=r'<< ",\"validated_embedding_bytes\":" << vision_validated_bytes'
    source=once(source,info,
        r'<< ",\"vision_dry_sampling_protocol\":\"'+PROTOCOL+r'\"" '+
        r'<< ",\"vision_sampling_protocols\":[\"'+V4+r'\"]" '+info)
    stats=r'<< ",\"sampling_v4_ns\":" << sampling_v4_ns'
    source=once(source,stats,stats+
        r' << ",\"sampling_image_dry_combination_calls\":" << sampling_image_dry_combination_calls')
    return source


def build(receipt_path,root,out):
    receipt=json.loads(receipt_path.read_text());root=root.resolve(strict=True);out=out.resolve()
    if receipt['status']!='BUILT_NOT_EXECUTED':raise ValueError('Unexpected pinned build receipt')
    for path,digest in receipt['dependencies'].items():
        if sha256(Path(path))!=digest:raise ValueError('Changed pinned dependency: '+path)
    parent_path=Path(receipt['source_receipt']['path'])
    if sha256(parent_path)!=receipt['source_receipt']['sha256']:raise ValueError('Changed parent build receipt')
    parent=json.loads(parent_path.read_text())
    links=parent['verified_unchanged_inputs']
    for path,digest in links.items():
        if sha256(Path(path))!=digest:raise ValueError('Changed pinned link input: '+path)
    compile_row=next(row for row in receipt['commands'] if row['label']=='compile')
    link_row=next(row for row in receipt['commands'] if row['label']=='link')
    old_source=Path(compile_row['argv'][compile_row['argv'].index('-c')+1]);old_obj=compile_row['argv'][compile_row['argv'].index('-o')+1]
    if sha256(old_source)!=receipt['generated_driver']['sha256']:raise ValueError('Changed pinned generated driver')
    out.mkdir(parents=True,exist_ok=False);cpp=out/'branch_driver.cpp';cpp.write_text(instrument(old_source.read_text()),newline='\n')
    headers={}
    for old_path,digest in receipt['headers'].items():
        path=root/'experimental/branching_context/native'/Path(old_path).name
        if sha256(path)!=digest:raise ValueError('Frozen selector/projection header changed: '+str(path))
        (out/path.name).write_bytes(path.read_bytes());headers[str(path)]=digest
    obj=out/'branch_driver.o';binary=out/'strata-vision-dry-combinations-cow';dep=out/'branch_driver.d'
    compile_args=[str(cpp) if arg==str(old_source) else str(obj) if arg==old_obj else arg for arg in compile_row['argv']]
    compile_args[1:1]=['-I'+str(out)];compile_args[compile_args.index('-MF')+1]=str(dep)
    link_args=[str(obj) if arg==old_obj else arg for arg in link_row['argv']];link_args[link_args.index('-o')+1]=str(binary)
    report=dict(status='BUILDING',source_receipt=dict(path=str(receipt_path),sha256=sha256(receipt_path)),
        base_driver=dict(path=str(old_source),sha256=sha256(old_source)),headers=headers,
        generated_driver=dict(path=str(cpp),sha256=sha256(cpp)),verified_unchanged_inputs=links,
        commands=[],protocol=PROTOCOL,sampling_protocols=[V4])
    def save():(out/'build.json').write_text(json.dumps(report,indent=2)+'\n')
    save()
    try:
        for label,argv,cwd in [('compile',compile_args,compile_row['cwd']),('link',link_args,link_row['cwd'])]:
            begin=time.monotonic();result=subprocess.run(argv,cwd=cwd,capture_output=True)
            (out/(label+'.stdout')).write_bytes(result.stdout);(out/(label+'.stderr')).write_bytes(result.stderr)
            report['commands'].append(dict(label=label,argv=argv,cwd=cwd,exit_code=result.returncode,seconds=time.monotonic()-begin));save()
            if result.returncode:raise RuntimeError(label+' failed; retained logs')
        dependencies=shlex.split(dep.read_text().replace('\\\n',' ').split(':',1)[1])
        report.update(status='BUILT_NOT_EXECUTED',binary=dict(path=str(binary),bytes=binary.stat().st_size,sha256=sha256(binary)),
            dependencies={str(Path(p).resolve()):sha256(Path(p)) for p in dependencies})
    except BaseException as error:report.update(status='BUILD_FAILED',error=repr(error));raise
    finally:save()
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('receipt','root','output'):parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();print(json.dumps(build(args.receipt,args.root,args.output)))
