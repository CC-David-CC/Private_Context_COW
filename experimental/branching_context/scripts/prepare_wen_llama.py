"""Apply and verify the optional Wen naming patch at the inspected llama.cpp pin."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

PIN='972d2313bc0bf0a45f634f77d95c9fb03aeab12c'
REPLACEMENTS={
 'src/llama-arch.h':[
  ('    const char * suffix;\n\n    std::string operator()(llm_kv kv) const;',
   '    const char * suffix;\n    const char * prefix = nullptr;\n\n    std::string operator()(llm_kv kv) const;')],
 'src/llama-arch.cpp':[
  ('::format(LLM_KV_NAMES.at(kv), LLM_ARCH_NAMES.at(arch))',
   '::format(LLM_KV_NAMES.at(kv), prefix ? prefix : LLM_ARCH_NAMES.at(arch))'),
  ('llm_arch llm_arch_from_string(const std::string & name) {',
   'llm_arch llm_arch_from_string(const std::string & name) {\n    if (name == "wen1") return LLM_ARCH_QWEN35;')],
 'src/llama-model-loader.cpp':[
  ('        llm_kv = LLM_KV(llm_arch_from_string(arch_name));',
   '        llm_kv = LLM_KV(llm_arch_from_string(arch_name));\n        if (arch_name == "wen1") llm_kv.prefix = "wen1";')],
 'src/llama-vocab.cpp':[
  ('tokenizer_pre == "qwen35") {','tokenizer_pre == "qwen35" || tokenizer_pre == "wen1") {')]
}

# Optional diagnostic/training precision profile. The BF16/F16 storage remains
# unchanged; route its matrix products through FP32 cuBLAS instead of casting
# F32 activations down to the stored weight type in MMF, or executing explicit
# TF32 MMA for F32 operands. Default dispatch is
# untouched. This is a separate, fingerprinted patch at the same inspected pin.
F32_REPLACEMENTS={
 'ggml/src/ggml-cuda/ggml-cuda.cu':[
  ('const ggml_tensor * src1, ggml_tensor * dst) {\n    const int cc = ggml_cuda_info().devices[ctx.device].cc;\n    ggml_type compute_type = src0->type;',
   'const ggml_tensor * src1, ggml_tensor * dst, bool force_f32 = false) {\n    const int cc = ggml_cuda_info().devices[ctx.device].cc;\n    ggml_type compute_type = src0->type;'),
  ('    switch (compute_type) {\n        case GGML_TYPE_F32:',
   '    if (force_f32) compute_type = GGML_TYPE_F32;\n\n    switch (compute_type) {\n        case GGML_TYPE_F32:'),
  ('static void ggml_cuda_mul_mat(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {\n    GGML_TENSOR_BINARY_OP_LOCALS',
   'static void ggml_cuda_mul_mat(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {\n    GGML_TENSOR_BINARY_OP_LOCALS\n\n    const char * f32_env = getenv("GGML_CUDA_F32_ACTIVATIONS");\n    if (f32_env && std::string(f32_env) == "1"\n            && (src0->type == GGML_TYPE_BF16 || src0->type == GGML_TYPE_F16 || src0->type == GGML_TYPE_F32)\n            && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {\n        ggml_cuda_mul_mat_cublas(ctx, src0, src1, dst, true);\n        return;\n    }')]
}

def git(root,*args):
 return subprocess.check_output(['git','-C',str(root),*args])

def expected_sources(root,*,f32_activations=False):
 result={}
 recipes=dict(REPLACEMENTS,**(F32_REPLACEMENTS if f32_activations else {}))
 for name,replacements in recipes.items():
  original=git(root,'show',PIN+':'+name);text=original.decode()
  for old,new in replacements:
   if old not in text or (name in F32_REPLACEMENTS and text.count(old)!=1):raise ValueError('Pinned source no longer uniquely matches Wen recipe: '+name)
   text=text.replace(old,new)
  result[name]=(original,text.encode())
 return result

def validate_checkout(root,*,f32_activations=False):
 root=Path(root)
 if git(root,'rev-parse','HEAD').decode().strip()!=PIN:raise ValueError('Requires inspected llama.cpp pin')
 if git(root,'diff','--cached','--name-only') or git(root,'ls-files','--others','--exclude-standard'):
  raise ValueError('Requires clean index and no untracked source files')
 dirty=set(git(root,'diff','--name-only').decode().splitlines())
 if not dirty:return {}
 expected_paths=set(REPLACEMENTS)|(set(F32_REPLACEMENTS) if f32_activations else set())
 if dirty!=expected_paths:raise ValueError('Unexpected llama.cpp source changes')
 sources=expected_sources(root,f32_activations=f32_activations)
 for name,(_,expected) in sources.items():
  if (root/name).is_symlink() or (root/name).read_bytes()!=expected:
   raise ValueError('Uninspected Wen source change: '+name)
 hashes={name:hashlib.sha256(value[1]).hexdigest() for name,value in sources.items() if name in REPLACEMENTS}
 result={'wen_patch_sha256':hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest(),
         'wen_sources_sha256':hashes}
 if f32_activations:
  precision={name:hashlib.sha256(sources[name][1]).hexdigest() for name in F32_REPLACEMENTS}
  result.update(f32_activations_patch_sha256=hashlib.sha256(json.dumps(precision,sort_keys=True).encode()).hexdigest(),
      f32_activations_sources_sha256=precision)
 return result

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--llama',type=Path,required=True)
 p.add_argument('--f32-activations',action='store_true',help='Apply the separate opt-in CUDA activation precision patch')
 a=p.parse_args()
 try:state=validate_checkout(a.llama,f32_activations=a.f32_activations)
 except ValueError:
  if not a.f32_activations:raise
  # Only a clean pin or the exact prior Wen recipe can be upgraded.
  validate_checkout(a.llama)
  state=None
 if not state:
  for name,(_,expected) in expected_sources(a.llama,f32_activations=a.f32_activations).items():(a.llama/name).write_bytes(expected)
 print(json.dumps(validate_checkout(a.llama,f32_activations=a.f32_activations),indent=2))

if __name__=='__main__':main()
