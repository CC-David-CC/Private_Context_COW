"""Build the isolated native probe and portable head backward, without a GPU."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

PIN = '972d2313bc0bf0a45f634f77d95c9fb03aeab12c'
ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--llama', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    llama = args.llama.resolve()
    assert subprocess.check_output(['git','-C',str(llama),'rev-parse','HEAD'], text=True).strip() == PIN
    args.output.mkdir(parents=True, exist_ok=True)
    lib = llama/'build/bin'
    bridge = ROOT.parent/'branching_context/native'
    common = ['c++','-std=c++17','-O2', '-I'+str(llama/'include'),
              '-I'+str(llama/'ggml/include'), '-I'+str(llama/'vendor'), '-I'+str(bridge)]
    commands = [common+['-shared','-fPIC',str(ROOT/'head_backward.cpp'),'-o',str(args.output/'libwen-head-backward.so')],
                common+[str(ROOT/'native_probe.cpp'),'-L'+str(lib),'-Wl,-rpath,'+str(lib),
                        '-o',str(args.output/'wen-native-training-probe'),'-lllama','-lggml','-lggml-base']]
    for command in commands:
        subprocess.run(command, check=True)
    sources = [ROOT/'head_backward.cpp', ROOT/'native_probe.cpp', bridge/'qwen35_device.h',
               bridge/'qwen35_math.h', bridge/'qwen35_lora.h', llama/'include/llama.h']
    sources += list(lib.glob('*.so'))
    receipt = dict(llama_pin=PIN, commands=commands,
                   sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})
    (args.output/'build.json').write_text(json.dumps(receipt, indent=2)+'\n')


if __name__ == '__main__':
    main()
