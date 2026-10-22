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

def git(root,*args):
 return subprocess.check_output(['git','-C',str(root),*args])

def expected_sources(root):
 result={}
 for name,replacements in REPLACEMENTS.items():
  original=git(root,'show',PIN+':'+name);text=original.decode()
  for old,new in replacements:
   if old not in text:raise ValueError('Pinned source no longer matches Wen recipe: '+name)
   text=text.replace(old,new)
  result[name]=(original,text.encode())
 return result

def validate_checkout(root):
 root=Path(root)
 if git(root,'rev-parse','HEAD').decode().strip()!=PIN:raise ValueError('Requires inspected llama.cpp pin')
 if git(root,'diff','--cached','--name-only') or git(root,'ls-files','--others','--exclude-standard'):
  raise ValueError('Requires clean index and no untracked source files')
 dirty=set(git(root,'diff','--name-only').decode().splitlines())
 if not dirty:return {}
 if dirty!=set(REPLACEMENTS):raise ValueError('Unexpected llama.cpp source changes')
 sources=expected_sources(root)
 for name,(_,expected) in sources.items():
  if (root/name).is_symlink() or (root/name).read_bytes()!=expected:
   raise ValueError('Uninspected Wen source change: '+name)
 hashes={name:hashlib.sha256(value[1]).hexdigest() for name,value in sources.items()}
 return {'wen_patch_sha256':hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest(),
         'wen_sources_sha256':hashes}

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--llama',type=Path,required=True);a=p.parse_args()
 state=validate_checkout(a.llama)
 if not state:
  for name,(_,expected) in expected_sources(a.llama).items():(a.llama/name).write_bytes(expected)
 print(json.dumps(validate_checkout(a.llama),indent=2))

if __name__=='__main__':main()
