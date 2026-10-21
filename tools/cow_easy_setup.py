"""Guided Linux/NVIDIA installation using the existing COW build and serializer.

Budgets are conservative starting allocations, not a hardware qualification.
No ordinary Strata server, MTP, driver replacement or background service is installed.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict, dataclass
import fcntl
import json
import math
import os
from pathlib import Path
import platform
import re
import shlex
import shutil
import signal
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
G=1024**3

class SetupError(RuntimeError):
    pass

@dataclass(frozen=True)
class Hardware:
    ram_total:int
    ram_available:int
    gpu_total:int
    gpu_free:int
    cpus:int
    gpu_name:str
    nvcc:str

@dataclass(frozen=True)
class Plan:
    resident_gib:int
    ram_gib:int
    disk_gib:int
    vram_reserve_mib:int
    context:int
    prefill:int
    build_jobs:int

def plan_for(hardware):
    values=[hardware.ram_total,hardware.ram_available,hardware.gpu_total,hardware.gpu_free,hardware.cpus]
    if any(not math.isfinite(x) or x<=0 for x in values):
        raise ValueError('Invalid hardware resource measurements')
    available=int(hardware.ram_available//G)
    if available<12:raise ValueError('Not enough available RAM: close other programs; at least 12 GiB must be available for this initial profile.')
    if hardware.gpu_free<12*G:raise ValueError('Not enough free GPU memory: this initial profile requires at least 12 GiB free; this is a preflight floor, not a qualified minimum.')
    ram=max(1,min(16,available//12))
    resident=max(4,min(72,available-ram-8))
    reserve=min(8192,max(4096,int(hardware.gpu_free/G)*128))
    # Compilation and serving occur separately. Keep headroom for the OS and
    # cap compiler concurrency by available memory, not only the CPU count.
    jobs=max(1,min(hardware.cpus,(available-6)//3))
    return Plan(resident,ram,16,reserve,8192,512,jobs)

def inspect_host():
    if platform.system()!='Linux' or platform.machine() not in ('x86_64','amd64'):
        raise SetupError('This guided installation supports 64-bit Intel/AMD Linux.')
    missing=[name for name in ('git','cmake','c++','nvidia-smi') if not shutil.which(name)]
    if missing:
        raise SetupError('Missing tools: '+', '.join(missing)+'. On Ubuntu, install build tools with: sudo apt update && sudo apt install -y git cmake build-essential python3-venv. NVIDIA driver/CUDA installation is separate.')
    nvcc=shutil.which('nvcc') or ('/usr/local/cuda/bin/nvcc' if Path('/usr/local/cuda/bin/nvcc').is_file() else None)
    if not nvcc:raise SetupError('CUDA compiler not found. Install the NVIDIA CUDA Toolkit, then run setup again. A graphics driver alone does not include nvcc.')
    version=subprocess.check_output(['cmake','--version'],text=True).splitlines()[0]
    match=re.search(r'(\d+)\.(\d+)',version)
    if not match or tuple(map(int,match.groups()))<(3,24):raise SetupError('CMake 3.24 or newer is required.')
    mem={}
    for line in Path('/proc/meminfo').read_text().splitlines():
        parts=line.split();mem[parts[0].rstrip(':')]=int(parts[1])*1024
    row=subprocess.check_output(['nvidia-smi','-i','0','--query-gpu=name,memory.total,memory.free','--format=csv,noheader,nounits'],text=True).strip().split(',')
    if len(row)!=3:raise SetupError('Could not read GPU 0 memory from nvidia-smi.')
    busy=subprocess.check_output(['nvidia-smi','-i','0','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip()
    if busy:raise SetupError('GPU 0 already has a compute process. Stop your own model server before setup; no process was stopped automatically.')
    subprocess.run([nvcc,'--version'],check=True,stdout=subprocess.DEVNULL)
    return Hardware(mem['MemTotal'],mem['MemAvailable'],int(row[1])*1024**2,int(row[2])*1024**2,os.cpu_count() or 1,row[0].strip(),nvcc)

def run_step(label,command,log,cwd):
    log=Path(log)
    print(label+' ...',flush=True)
    with log.open('w') as stream:
        process=subprocess.Popen(command,cwd=cwd,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        started=last=time.monotonic()
        try:
            while process.poll() is None:
                time.sleep(0.1)
                if time.monotonic()-last>=30:
                    print(f'{label}: still running ({time.monotonic()-started:.0f}s). Log: {log}',flush=True);last=time.monotonic()
        except BaseException:
            if process.poll() is None:
                os.killpg(process.pid,signal.SIGTERM)
                try:process.wait(timeout=5)
                except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()
            raise
    if process.returncode:
        tail='\n'.join(log.read_text(errors='replace').splitlines()[-12:])
        raise SetupError(label+' failed. Log: '+str(log)+'\n'+tail)
    print(label+' complete.',flush=True)

def nearest_existing(path):
    while not path.exists():path=path.parent
    return path

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir',type=Path,default=Path.home()/'ContextCOW-data')
    parser.add_argument('--existing-models',type=Path,help='Use and hash existing pinned Q4 shards without copying or downloading them.')
    parser.add_argument('--yes',action='store_true',help='Approve installation and the displayed large model download.')
    parser.add_argument('--check',action='store_true',help='Inspect hardware and print a plan; do not build, install or download.')
    parser.add_argument('--port',type=int,default=8099)
    args=parser.parse_args(argv)
    if not 1<=args.port<=65535:parser.error('Port must be between 1 and 65535.')
    os.umask(0o077)
    data=args.data_dir.expanduser().absolute()
    if data.is_symlink():raise SetupError('Choose a real data folder, not a symlink.')
    config=data/'server/server.json'
    if config.exists() and not args.check:
        saved=json.loads(config.read_text())
        if not Path(saved['profile']).is_file() or not Path(saved['api_key_file']).is_file():
            raise SetupError('An incomplete server configuration exists; existing files were preserved: '+str(config))
        print('This server is already configured. Its model, profile and API key were preserved.')
        print('Start it with: '+shlex.join([str(ROOT/'cow-start'),str(data)]))
        return 0
    hardware=inspect_host();plan=plan_for(hardware)
    sys.path.insert(0,str(ROOT/'tools'))
    from cow_prepare_model import model_plan,PACK_NAME
    assets=model_plan()
    models=args.existing_models.expanduser().resolve(strict=True) if args.existing_models else data/'models/UD-Q4_K_XL'
    if args.existing_models and any(not (models/a.name).is_file() for a in assets):
        raise SetupError('The existing folder does not contain all four pinned Q4 shards.')
    missing=sum(a.size for a in assets if not (models/a.name).exists())
    disk=shutil.disk_usage(nearest_existing(data))
    required=missing+(plan.disk_gib+6)*G
    print('Context COW guided setup')
    print(f'GPU 0: {hardware.gpu_name}; {hardware.gpu_free/G:.1f} GiB free')
    print(f'RAM: {hardware.ram_total/G:.1f} GiB total; {hardware.ram_available/G:.1f} GiB available')
    print(f'Initial settings: {plan.context:,} context tokens, FP16 KV, {plan.resident_gib} GiB expert RAM, {plan.ram_gib} GiB context RAM, {plan.disk_gib} GiB disk pool.')
    print(f'Build jobs: {plan.build_jobs}; model files: {sum(a.size for a in assets)/1e9:.1f} GB; missing download: {missing/1e9:.1f} GB.')
    print('Settings are initial budgets, not numerical or performance qualification. GPU 0 serves one request at a time.')
    print('Server listens on your trusted LAN with an API key. Do not forward it to the Internet.')
    if disk.free<required:raise SetupError(f'Not enough disk space: allow {required/G:.1f} GiB free here, including missing model files and the disk-pool budget.')
    if args.check:return 0
    if not args.yes and input('Install using these settings? Type YES: ').strip()!='YES':
        print('Canceled; no installation started.');return 1
    data.mkdir(mode=0o700,parents=True,exist_ok=True)
    with (data/'.setup.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise SetupError('Another setup owns this data folder.')
        if config.exists():raise SetupError('A server configuration appeared during setup; it was not replaced.')
        logs=data/'setup-logs';logs.mkdir(mode=0o700,exist_ok=True)
        venv=data/'venv';python=venv/'bin/python';build=data/'build'
        (logs/'hardware-plan.json').write_text(json.dumps(dict(hardware=asdict(hardware),plan=asdict(plan)),indent=2)+'\n')
        if not python.exists():run_step('Create Python environment',[sys.executable,'-m','venv',str(venv)],logs/'venv.log',ROOT)
        run_step('Install Python packages',[str(python),'-m','pip','install','-r',str(ROOT/'requirements-cow-server.txt')],logs/'packages.log',ROOT)
        run_step('Configure engine',['cmake','-S',str(ROOT),'-B',str(build),'-DSTRATA_ENABLE_CUDA=ON','-DCMAKE_CUDA_COMPILER='+hardware.nvcc,'-DCMAKE_CUDA_ARCHITECTURES=native','-DCMAKE_BUILD_TYPE=Release','-DPython3_EXECUTABLE='+str(python)],logs/'configure.log',ROOT)
        run_step('Build engine',['cmake','--build',str(build),'--target','strata-branch','-j',str(plan.build_jobs)],logs/'build.log',ROOT)
        command=[str(python),str(ROOT/'tools/cow_prepare_model.py'),'--data-dir',str(data),'--yes']
        if args.existing_models:command+=['--existing-models',str(models)]
        run_step('Prepare Q4 model',command,logs/'model.log',ROOT)
        run_step('Configure server',[str(python),str(ROOT/'tools/cow_server_setup.py'),'--model',str(models/assets[0].name),'--pack',str(data/PACK_NAME),'--native',str(build/'strata-branch'),'--expert-profile',str(ROOT/'data/expert-profile.bin'),'--output',str(data/'server'),'--host','0.0.0.0','--port',str(args.port),'--context',str(plan.context),'--prefill',str(plan.prefill),'--resident-gib',str(plan.resident_gib),'--ram-gib',str(plan.ram_gib),'--disk-gib',str(plan.disk_gib),'--vram-reserve-mib',str(plan.vram_reserve_mib)],logs/'server.log',ROOT)
    print('Context COW setup complete. The model server has not been started.')
    print('Start: '+shlex.join([str(ROOT/'cow-start'),str(data)]))
    print('API key: '+shlex.join(['cat',str(data/'server/api-key.txt')]))
    print('Connection address: http://YOUR-SERVER-LAN-IP:'+str(args.port))
    return 0

if __name__=='__main__':
    try:raise SystemExit(main())
    except (SetupError,ValueError,OSError,subprocess.CalledProcessError) as error:raise SystemExit('Setup stopped: '+str(error))
