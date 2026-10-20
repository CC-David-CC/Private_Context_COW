#!/usr/bin/env python3
"""Build the isolated native model-free CUDA harness without configuring Strata."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
SOURCES = [ROOT / "src/core/vmm.cpp", ROOT / "src/core/branch_pages.cpp", ROOT / "src/core/branch_digest.cu"]
ORACLE_SOURCE = HERE / "fixture_cpu.cpp"


def run(command: list[str], cwd: Path) -> None:
    print("$ " + " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=("llm60", "llm49", "r7300"), default="llm60")
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--arch", help="Override CUDA architecture, e.g. sm_120")
    args = parser.parse_args()
    missing = [str(path) for path in SOURCES + [ORACLE_SOURCE, HERE / "seeded_cow_test.cu", HERE / "fixture_pressure.cu", HERE / "fixture_data.hpp"] if not path.is_file()]
    if missing:
        parser.error("missing build input(s): " + ", ".join(missing))
    nvcc = os.environ.get("NVCC", "/usr/local/cuda-13.2/bin/nvcc" if args.profile == "llm60" else "/usr/bin/nvcc")
    cxx = os.environ.get("CXX", "/usr/bin/g++-12" if args.profile == "r7300" else "/usr/bin/g++")
    arch = args.arch or {"llm60": "sm_120", "llm49": "sm_89", "r7300": "sm_61"}[args.profile]
    out = args.build_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    common = ["-std=c++17", "-O2", "-I" + str(ROOT / "include"), "-I" + str(HERE)]
    objects: list[Path] = []
    for source in SOURCES[:2]:
        obj = out / (source.stem + ".o")
        run([cxx, *common, "-I/usr/local/cuda/include", "-c", str(source), "-o", str(obj)], out)
        objects.append(obj)
    oracle_binary = out / "fixture_cpu"
    run([cxx, "-std=c++17", "-O2", str(ORACLE_SOURCE), "-o", str(oracle_binary)], out)
    digest_obj = out / "branch_digest.o"
    run([nvcc, "-ccbin=" + cxx, "-std=c++17", "-O2", "-arch=" + arch, "-I" + str(ROOT / "include"), "-c", str(SOURCES[2]), "-o", str(digest_obj)], out)
    objects.append(digest_obj)
    binaries = [oracle_binary]
    for source, name in ((HERE / "seeded_cow_test.cu", "seeded_cow_test"), (HERE / "fixture_pressure.cu", "fixture_pressure")):
        binary = out / name
        run([nvcc, "-ccbin=" + cxx, "-std=c++17", "-O2", "-arch=" + arch, "-I" + str(ROOT / "include"), "-I" + str(HERE),
             str(source), *(str(item) for item in objects), "-lcudart", "-o", str(binary)], out)
        binaries.append(binary)
    receipt = {"profile": args.profile, "arch": arch, "nvcc": nvcc, "cxx": cxx,
               "binaries": {binary.name: {"path": str(binary), "sha256": sha256(binary)} for binary in binaries},
               "inputs": {str(path.relative_to(ROOT)): sha256(path) for path in SOURCES + [ORACLE_SOURCE, HERE / "seeded_cow_test.cu", HERE / "fixture_pressure.cu", HERE / "fixture_data.hpp", HERE / "build_model_free.py"]}}
    receipt_path = out / "build_receipt.json"
    receipt_path.write_text(json.dumps(receipt, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
