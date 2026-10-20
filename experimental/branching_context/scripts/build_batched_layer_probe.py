"""Instrument only Prefill's mixed activations in a separate diagnostic binary."""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess

from build_fast_prefill_probe import replace_once, sha


HOOK = r'''
                if (const char* trace = std::getenv("COW_BATCH_LAYER_TRACE")) {
                    const char* wanted = std::getenv("COW_BATCH_TRACE_POSITION");
                    if (!wanted) { err = "batch trace: target position missing"; return false; }
                    const int64_t target = std::atoll(wanted);
                    if (p0 <= target && target < p0 + T) {
                        if (l == 0 && half == 0)
                            std::fprintf(stderr, "batch trace expected_layers=%lld width=%lld\n",
                                         (long long) g.n_layers, (long long) N);
                        std::vector<float> row((size_t) N);
                        if (cudaStreamSynchronize(m.cs) != cudaSuccess ||
                            cudaMemcpy(row.data(), m.mixed + (target-p0)*N, N*4,
                                       cudaMemcpyDeviceToHost) != cudaSuccess) {
                            err = "batch trace: mixed readback failed"; return false;
                        }
                        std::FILE* file = std::fopen(trace, "ab");
                        if (!file) { err = "batch trace: cannot open output"; return false; }
                        const int64_t header[4] = {target, l, half, N};
                        const bool written = std::fwrite(header, sizeof(header), 1, file) == 1 &&
                                             std::fwrite(row.data(), 4, row.size(), file) == row.size();
                        const bool closed = std::fclose(file) == 0;
                        if (!written || !closed) { err = "batch trace: incomplete output"; return false; }
                        if (l == 0 && half == 0) {
                            std::vector<uint16_t> images((size_t) N*2);
                            if (cudaMemcpy(images.data(),m.mixed_h+(target-p0)*N,N*2,cudaMemcpyDeviceToHost)!=cudaSuccess ||
                                cudaMemcpy(images.data()+N,m.mixed_bf+(target-p0)*N,N*2,cudaMemcpyDeviceToHost)!=cudaSuccess) {
                                err = "batch trace: projection input image readback failed"; return false;
                            }
                            const std::string name = std::string(trace)+".input-f16-bf16";
                            std::FILE* input = std::fopen(name.c_str(),"wb");
                            if (!input) { err = "batch trace: input image open failed"; return false; }
                            const bool saved = std::fwrite(images.data(),2,images.size(),input)==images.size();
                            const bool done = std::fclose(input)==0;
                            if (!saved || !done) { err = "batch trace: input image write failed"; return false; }
                        }
                    }
                }
'''

GDN_HOOK = r'''
                    auto trace_gdn = [&](int64_t stage, const float* data, int64_t width, int64_t stride) -> bool {
                        const char* file_name = std::getenv("COW_BATCH_GDN_TRACE");
                        if (!file_name || l != 0) return true;
                        const char* wanted = std::getenv("COW_BATCH_TRACE_POSITION");
                        if (!wanted) { err = "gdn trace: target missing"; return false; }
                        const int64_t target = std::atoll(wanted);
                        if (target < p0 || target >= p0+T) return true;
                        std::vector<float> row((size_t) width);
                        if (cudaStreamSynchronize(m.cs) != cudaSuccess ||
                            cudaMemcpy(row.data(), data+(target-p0)*stride, width*4,
                                       cudaMemcpyDeviceToHost) != cudaSuccess) {
                            err = "gdn trace: readback failed"; return false;
                        }
                        std::FILE* f = std::fopen(file_name,"ab");
                        if (!f) { err = "gdn trace: open failed"; return false; }
                        const int64_t header[4] = {target,l,stage,width};
                        const bool ok = std::fwrite(header,sizeof(header),1,f)==1 &&
                                        std::fwrite(row.data(),4,row.size(),f)==row.size();
                        const bool closed = std::fclose(f)==0;
                        if (!ok || !closed) { err = "gdn trace: write failed"; return false; }
                        return true;
                    };
'''


def add_gdn_trace(source):
    anchor = '                    if (!native_proj(m.gemm, wqkv, m.mixed_h, m.qkv, T, v.name("attn_qkv.weight"), err)) return false;'
    source = replace_once(source,anchor,GDN_HOOK+'\n'+anchor+'\n                    if (!trace_gdn(2,m.qkv,C,C)) return false;')
    for anchor,call in [
        ('                    if (!native_proj(m.gemm, wg, m.mixed_h, m.z, T, v.name("attn_gate.weight"), err)) return false;',
         'trace_gdn(3,m.z,ZV,ZV)'),
        ('                    if (!bf16_proj(m.gemm, wb, m.mixed_bf, m.ab + HV, T, v.name("ssm_beta.weight"), err, 2 * HV, m.mixed_bf_lo)) return false;',
         'trace_gdn(4,m.ab,2*HV,2*HV)'),
        ('                    gdn_gates(m.ab, (const float*) wdt->data, (const float*) wsa->data, m.gate, m.beta, T, m.cs);',
         'trace_gdn(5,m.gate,HV,HV) || !trace_gdn(6,m.beta,HV,HV)'),
        ('                    gdn_conv(conv, m.qkv, (const float*) wc->data, m.hbuf, T, EPS, m.cs);',
         'trace_gdn(7,m.hbuf,C,C)'),
        ('                    pt.mark(kPfGdnOut, cs);',
         'trace_gdn(8,m.y,ZV,ZV)'),
        ('                    if (!native_proj(m.gemm, wo, m.y_h, m.bo, T, v.name("ssm_out.weight"), err, 0, ld_y)) return false;',
         'trace_gdn(9,m.bo,N,N)')]:
        source=replace_once(source,anchor,anchor+'\n                    if (!'+call+') return false;')
    return source


def build(root, base, output):
    root, base, output = map(lambda p: Path(p).resolve(), (root,base,output))
    output.mkdir(parents=True,exist_ok=False)
    original = root/'src/prefill/prefill.cpp'
    source = replace_once(original.read_text(),
        '                if (half == 0 && !core::is_qsa_layer(g, l)) {',
        HOOK+'\n                if (half == 0 && !core::is_qsa_layer(g, l)) {')
    source = add_gdn_trace(source)
    generated = output/'prefill.cpp';generated.write_text(source)
    receipt = json.loads((base/'build.json').read_text())
    if receipt.get('continuation_operator') != 'batched-prefill':
        raise ValueError('Batches-only base required')
    for name,digest in receipt['binaries'].items():
        if sha(base/name) != digest: raise ValueError('Base binary changed')
    flags_path = root/'build/CMakeFiles/strata_prefill.dir/flags.make'
    flags={}
    for line in flags_path.read_text().splitlines():
        if line.startswith(('CXX_DEFINES =','CXX_INCLUDES =','CXX_FLAGS =')):
            key,value=line.split('=',1);flags[key.strip()]=shlex.split(value)
    obj=output/'prefill.o'
    commands=[['/usr/bin/c++','-I'+str(root/'src/prefill'),*flags['CXX_DEFINES'],*flags['CXX_INCLUDES'],*flags['CXX_FLAGS'],
               '-c',str(generated),'-o',str(obj)]]
    link=list(receipt['commands'][-1]['argv'])
    index=link.index('-o');link[index+1]=str(output/'strata-batched-layer-fresh')
    link.insert(1,str(obj));commands.append(link)
    report=dict(status='BUILDING',scope='diagnostic readback, no model arithmetic changes',
        source_sha256={str(original):sha(original),str(generated):sha(generated),str(flags_path):sha(flags_path)},
        base_build_sha256=sha(base/'build.json'),commands=[])
    for i,cmd in enumerate(commands):
        with (output/(str(i)+'.stdout')).open('wb') as out,(output/(str(i)+'.stderr')).open('wb') as err:
            run=subprocess.run(cmd,cwd=root/'build',stdout=out,stderr=err)
        report['commands'].append(dict(argv=cmd,exit_code=run.returncode))
        (output/'build.json').write_text(json.dumps(report,indent=2)+'\n')
        if run.returncode: raise RuntimeError('Diagnostic build failed: '+str(output/(str(i)+'.stderr')))
    report.update(status='BUILT_NOT_EXECUTED',binary_sha256=sha(output/'strata-batched-layer-fresh'))
    (output/'build.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('root','base','output'): parser.add_argument(name,type=Path)
    args=parser.parse_args();build(args.root,args.base,args.output)
