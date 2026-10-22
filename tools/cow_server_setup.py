"""Create a new, fingerprinted Context COW server configuration."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import secrets
import shutil
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]

def parse_gpus(value):
    """Use the coordinator's existing flat-worker or nested-group topology."""
    sys.path[:0]=[str(ROOT/'experimental/branching_context/src')]
    from context_time_travel.pools import normalize_gpu_selection
    try:
        selection=json.loads(value) if value.strip().startswith('[') else [int(n) for n in value.split(',')]
        return list(normalize_gpu_selection(selection))
    except (ValueError,TypeError) as exc:raise argparse.ArgumentTypeError(str(exc)) from exc

def fingerprint(path):
    path=Path(path)
    with path.open("rb") as stream:
        before=os.fstat(stream.fileno())
        digest=hashlib.file_digest(stream,"sha256").hexdigest()
        after=os.fstat(stream.fileno())
    if (before.st_size,before.st_mtime_ns,before.st_ctime_ns)!=(after.st_size,after.st_mtime_ns,after.st_ctime_ns):
        raise ValueError("File changed while hashing: "+str(path))
    return dict(bytes=after.st_size,mtime_ns=after.st_mtime_ns,
                ctime_ns=after.st_ctime_ns,sha256=digest)

def model_shards(path):
    path=Path(path)
    match=re.fullmatch(r"(.*)-(\d{5})-of-(\d{5})\.gguf",path.name)
    if match is None:
        if not path.is_file(): raise ValueError("Missing model file: "+str(path))
        return [path]
    stem,part,count=match.groups()
    if int(part)!=1: raise ValueError("Choose the first model shard.")
    files=[path.with_name(f"{stem}-{i:05d}-of-{int(count):05d}.gguf") for i in range(1,int(count)+1)]
    if not all(p.is_file() for p in files): raise ValueError("Missing model shard; complete the model download first.")
    return files

def new_output(path):
    path=Path(path)
    path.mkdir(mode=0o700,parents=True,exist_ok=False)
    return path.resolve()

def write_json(path,value):
    with Path(path).open("x") as stream:
        json.dump(value,stream,indent=2)
        stream.write("\n")
    Path(path).chmod(0o600)

def _make_profile(args,output):
    sys.path[:0]=[str(ROOT),str(ROOT/"experimental/branching_context/src")]
    from tools.gguf_reader import GGUFFile
    from context_time_travel.strata_backend import _digest
    model=args.model.resolve(strict=True)
    md=GGUFFile(model).metadata
    if md.get("general.architecture") in ("qwen35","wen1"):
        if args.pack is not None or args.expert_profile is not None:
            raise ValueError("Dense Qwen does not use a Flash expert pack/profile")
        from experimental.branching_context.scripts.prepare_qwen35_profile import make_profile as qwen_profile
        profile=qwen_profile(model,args.native.resolve(strict=True).parent,output/"profile.json",
            args.context,args.prefill,os.cpu_count(),gpus=getattr(args,'gpu',None))
        if Path(profile["bridge_library"]).resolve()!=args.native.resolve():
            raise ValueError("--native must select the built Qwen bridge library")
        vision=args.vision_config
        if vision is not None:
            if profile['execution'].get('resume_contract')!='identical-mixed-native-operators-v1':
                raise ValueError('Image encoding requires the versioned Qwen mixed-input build')
            if not vision.is_absolute() or vision.is_symlink() or not vision.is_file():
                raise ValueError('--vision-config requires an existing absolute regular file')
        key=output/"api-key.txt";key.write_text(secrets.token_urlsafe(32)+"\n");key.chmod(0o600)
        configuration=dict(profile=str(output/"profile.json"),store=str(output/"store"),
            api_key_file=str(key),host=args.host,port=args.port,ram_gib=args.ram_gib,
            disk_gib=args.disk_gib,min_available_gib=args.min_available_gib,checkpoint_idle_seconds=1800)
        if vision is not None:configuration['vision_config']=str(vision)
        if getattr(args,'ephemeral',False):configuration['ephemeral']=True
        write_json(output/"server.json",configuration)
        print("Qwen server configured in "+str(output));return profile
    if args.pack is None:raise ValueError("Flash requires --pack")
    pack=args.pack.resolve(strict=True)
    binary=args.native.resolve(strict=True)
    if not os.access(binary,os.X_OK): raise ValueError("The native engine is not executable.")
    if not (pack/"dense.bin").is_file(): raise ValueError("The model pack has no dense.bin.")
    md=GGUFFile(model).metadata
    if md.get("general.architecture")!="qwen4exp" or md.get("tokenizer.ggml.model")!="gpt2":
        raise ValueError("This preview supports the tested qwen4exp GPT-2 tokenizer arrangement.")
    files=model_shards(model)+sorted(p for p in pack.rglob("*") if p.is_file())
    if args.expert_profile: files.append(args.expert_profile.resolve(strict=True))
    if any(p.is_symlink() for p in files): raise ValueError("Use actual model files, not symlinks.")
    reuse={}
    if args.reuse_model_receipt:
        reuse=json.loads(args.reuse_model_receipt.read_text())["model_files"]
    rows={}
    hashed=reused=0
    for i,path in enumerate(dict.fromkeys(files)):
        stat=path.stat()
        prior=reuse.get(str(path),{})
        if ((stat.st_size,stat.st_mtime_ns,stat.st_ctime_ns)==
            (prior.get("bytes"),prior.get("mtime_ns"),prior.get("ctime_ns"))
            and re.fullmatch("[a-f0-9]{64}",str(prior.get("sha256","")))):
            rows[str(path)]=dict(prior)
            reused+=1
        else:
            print(f"Hashing model file {i+1}/{len(files)}: {path.name}",flush=True)
            rows[str(path)]=fingerprint(path)
            hashed+=stat.st_size
    receipt=output/"model-receipt.json"
    write_json(receipt,dict(model_files=rows,hashed_bytes=hashed,reused_files=reused,
                           reused_receipt=str(args.reuse_model_receipt) if args.reuse_model_receipt else None))
    weights_sha=_digest({name:dict(bytes=x["bytes"],sha256=x["sha256"]) for name,x in rows.items()})
    contract=dict(weights_sha256=weights_sha,architecture=md["general.architecture"],
        rope_base=str(md["qwen4exp.rope.freq_base"]),yarn="model-default",
        tokenizer_identity=weights_sha,template_identity="exact-token-ids/no-template",
        attention="causal-qsa-indexed",positions="qwen-image-grid-mrope-v1",
        input_kind="canonical-mixed-spans-v1",adapters=[],state_kind="qwen4exp-gdn-qsa-ple-v1")
    command=[str(binary),"--pack",str(pack),"--native",str(model),
        "--max-context",str(args.context),"--prefill",str(args.prefill),
        "--spec","2","--suffix-draft","0","--kv","fp16",
        "--prompt-cache","0","--conversation-cache-mib","0","--stats","--no-prefill-borrow",
        "--resident-budget-gib",str(args.resident_gib),"--expert-cache","auto",
        "--ple-io","direct","--ple-row-cache","1048576",
        "--vram-reserve-mib",str(args.vram_reserve_mib),"--pcie-frac","0.55",
        "--max-new","128","--tokens","1","--vision","--gpu","0"]
    if args.expert_profile: command+=["--expert-profile",str(args.expert_profile.resolve())]
    execution=dict(GPU=[0],capacity=args.context,kv_dtype="FP16",
        checkpoint_cache=dict(byte_limit=min(16,args.ram_gib)*1024**3,entry_limit=256),
        embedding_width=md["qwen4exp.embedding_length"],environment={"CUDA_VISIBLE_DEVICES":"0"},
        eos_tokens=[md["tokenizer.ggml.tokens"].index(t) for t in ("<|endoftext|>","<|im_end|>")],
        generated_operator="normal-verifier-decode",grammar_selection_protocol="finite-mask-argmax-v1",
        native_request_protocol="consumed-greedy-ack-v1",
        operator="batched-input/native-greedy-request/per-write-sha256",
        prefill_chunk=args.prefill,resume_contract="identical-mixed-native-operators-v1",
        sampling_protocol="full-vocab-adaptive-p-u53-v7",
        selected_token_protocol="consumed-selected-receipt-v1",sut_command=command,
        text_template_kwargs=dict(enable_thinking=True),vision_dry_protocol="mixed-text-runs-dry-u53-v1",
        vision_protocol="explicit-image-rows-mrope-v3")
    if getattr(args,'gpu',None) is not None:
        from context_time_travel.pools import normalize_gpu_selection
        workers=normalize_gpu_selection(args.gpu)
        if any(type(gpu) is not int for gpu in workers):
            raise ValueError('Native split-worker checkpoints are not supported')
        execution['GPU']=list(workers)
        execution['environment']={'CUDA_DEVICE_ORDER':'PCI_BUS_ID','CUDA_VISIBLE_DEVICES':','.join(map(str,workers))}
        # Each native worker receives its own physical --gpu from worker_command.
        command=execution['sut_command'];i=command.index('--gpu');del command[i:i+2]
    revision=subprocess.check_output(["git","-C",str(ROOT),"rev-parse","HEAD"],text=True).strip()
    prefixes=("serve/","tools/","experimental/branching_context/src/","experimental/branching_context/scripts/")
    tracked=subprocess.check_output(["git","-C",str(ROOT),"ls-files","-z"]).decode().split("\0")
    closure={name:fingerprint(ROOT/name)["sha256"] for name in tracked if name.endswith(".py") and name.startswith(prefixes)}
    helper=ROOT/"tools/gguf_reader.py"
    build=dict(revision=revision,sha256={str(p):fingerprint(p)["sha256"] for p in (binary,helper)},
        selected_runtime_source_closure=dict(source_commit=revision,files_sha256=closure))
    hardware=subprocess.check_output(["nvidia-smi","--query-gpu=name,uuid,driver_version,memory.total","--format=csv,noheader"],text=True).strip()
    profile=dict(backend="strata",backend_revision=revision,model_path=str(model),
        model_files=rows,model_weights_sha256=weights_sha,model_hash_receipt=str(receipt),
        model_hash_receipt_sha256=fingerprint(receipt)["sha256"],
        model_contract=contract,model_contract_id=_digest(contract),
        native_build=build,gguf_reader=str(helper),execution=execution,hardware=hardware,
        execution_profile_id=_digest(dict(build=build,execution=execution,hardware=hardware)),
        calibration_status="FRESH_INSTALL_PREVIEW_NOT_NUMERICALLY_QUALIFIED",
        numerical_gate_qualified=False)
    write_json(output/"profile.json",profile)
    key=output/"api-key.txt"
    key.write_text(secrets.token_urlsafe(32)+"\n")
    key.chmod(0o600)
    server_config=dict(profile=str(output/"profile.json"),store=str(output/"store"),
        api_key_file=str(key),host=args.host,port=args.port,ram_gib=args.ram_gib,
        disk_gib=args.disk_gib,min_available_gib=getattr(args,"min_available_gib",16),checkpoint_idle_seconds=1800)
    vision=getattr(args,"vision_config",None)
    if vision is not None:
        path=Path(vision)
        if not path.is_absolute() or path.is_symlink() or not path.is_file():
            raise ValueError("--vision-config requires an existing absolute regular file")
        server_config["vision_config"]=str(path)
    if getattr(args,'ephemeral',False):server_config['ephemeral']=True
    write_json(output/"server.json",server_config)
    print("Server setup saved in "+str(output))
    print("Show the key privately with: cat "+str(key))
    print("Start the server with: "+str(sys.executable)+" "+str(ROOT/"tools/cow_server.py")+" "+str(output))
    return profile

def make_profile(args):
    output=new_output(args.output)
    try:
        return _make_profile(args,output)
    except BaseException:
        # Only this invocation's newly created, unpublished configuration.
        # A pre-existing folder is rejected by new_output before entering here.
        shutil.rmtree(output)
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model",type=Path,required=True)
    parser.add_argument("--pack",type=Path,help="Required for Flash; omit for dense Qwen")
    parser.add_argument("--native",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--gpu",type=parse_gpus,help='Physical PCI GPU IDs: 4 or 4,6 for independent workers. Grouped native checkpoints remain unsupported. Omit to preserve GPU 0 defaults.')
    parser.add_argument('--ephemeral',action='store_true',help='Use temporary session history/checkpoints and remove them on server exit')
    parser.add_argument("--expert-profile",type=Path)
    parser.add_argument("--vision-config",type=Path,help="Optional existing pinned image-producer configuration; omit for text/restore-only serving.")
    parser.add_argument("--reuse-model-receipt",type=Path,help="Optional trusted prior hash receipt; changed file stamps force hashing.")
    parser.add_argument("--context",type=int,default=98304)
    parser.add_argument("--prefill",type=int,default=512)
    parser.add_argument("--resident-gib",type=int,default=40)
    parser.add_argument("--vram-reserve-mib",type=int,default=8192)
    parser.add_argument("--ram-gib",type=int,default=48)
    parser.add_argument("--disk-gib",type=int,default=64)
    parser.add_argument("--min-available-gib",type=int,default=16)
    parser.add_argument("--host",default="127.0.0.1")
    parser.add_argument("--port",type=int,default=8099)
    args=parser.parse_args()
    if min(args.context,args.prefill,args.resident_gib,args.ram_gib,args.disk_gib,args.vram_reserve_mib,args.min_available_gib)<=0:
        parser.error("Context, batch and pool sizes must be positive.")
    if not 1<=args.port<=65535: parser.error("Invalid port.")
    os.umask(0o077)
    make_profile(args)

if __name__=="__main__":main()
