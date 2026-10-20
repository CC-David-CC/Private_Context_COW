"""Build the audit-only interleaved raw Prefill/Verifier driver from pinned inputs."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess


PINS = {
    "base_source": "85f9c1942f697ede539f977674db4ffe9af9f0800a30ace4dfe7baf718223b5c",
    "base_build": "d0fd5cc7981e29677f08c0bffadb1c951581d7e67cc8c3149562c83aab9f898f",
    "original_build_receipt": "d251cd11267d2e32a57144d25588a7f6450bd466df72f1cd828b21b36a2c7b0c",
    "compile_link": "7b26945aef71ed8ad46e3c8c030b9eebe3381a0b5fe6dab9f152667c6ac6aa30",
    "input_verification": "9ec1bb03b22e8b18449dc62775c57b84658b84181adb9f8c2787ef75ce8af6e1",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _rewrite_compile(argv, source_path, object_path):
    result = list(argv)
    source_suffix = "fresh_prefill_segmented_forced.cpp"
    source_indexes = [i for i, value in enumerate(result) if value.endswith(source_suffix)]
    if len(source_indexes) != 1 or result.count("-o") != 1:
        raise ValueError("pinned compile command has ambiguous source/output anchors")
    result[source_indexes[0]] = str(source_path)
    out_index = result.index("-o") + 1
    if out_index >= len(result):
        raise ValueError("pinned command has malformed -o argument")
    result[out_index] = str(object_path)
    return result


def _rewrite_link(argv, object_path, binary_path):
    result = list(argv)
    object_indexes = [i for i, value in enumerate(result)
                      if value.endswith("fresh_prefill_segmented_forced.o")]
    if len(object_indexes) != 1 or result.count("-o") != 1:
        raise ValueError("pinned link command has ambiguous object/output anchors")
    result[object_indexes[0]] = str(object_path)
    out_index = result.index("-o") + 1
    if out_index >= len(result):
        raise ValueError("pinned link command has malformed -o argument")
    result[out_index] = str(binary_path)
    return result


def _normalize_step(step):
    argv = list(step["argv"])
    for index, value in enumerate(argv):
        if value.endswith(".cpp"):
            argv[index] = "<source.cpp>"
        elif value.endswith(".o"):
            argv[index] = "<object.o>"
    output = argv.index("-o") + 1
    if output >= len(argv):
        raise ValueError("pinned build command has malformed output")
    argv[output] = "<output>"
    return {"stage": step.get("stage"), "cwd": step.get("cwd"), "argv": argv}


def build(*, source: Path, base_source: Path, build_receipt: Path,
          compile_link_receipt: Path, input_report: Path, output_dir: Path) -> dict:
    for label, path, pin in (
        ("base_source", base_source, PINS["base_source"]),
        ("base_build", build_receipt, PINS["base_build"]),
        ("compile_link", compile_link_receipt, PINS["compile_link"]),
        ("input_verification", input_report, PINS["input_verification"]),
    ):
        if sha256_file(path) != pin:
            raise ValueError(f"pinned {label} bytes differ")
    base = json.loads(build_receipt.read_bytes())
    recipe = json.loads(compile_link_receipt.read_bytes())
    verification = json.loads(input_report.read_bytes())
    if (base.get("source_original_sha256") != PINS["base_source"]
            or base.get("compile_link_receipt_sha256") != PINS["compile_link"]
            or base.get("input_verification_sha256") != PINS["input_verification"]
            or base.get("verified_input_count") != 53
            or base.get("verified_input_mismatches") != []):
        raise ValueError("base build receipt provenance is inconsistent")
    recipe_steps = recipe.get("compiler_and_link_steps")
    if recipe.get("original_build_receipt_sha256") != PINS["original_build_receipt"]:
        raise ValueError("compile/link receipt points to an unrelated build")
    if (recipe.get("segmented_source_sha256") != PINS["base_source"]
            or recipe.get("verified_input_count") != 53
            or recipe.get("verified_input_mismatches") != []
            or recipe.get("verified_input_report_sha256") != PINS["input_verification"]
            or not isinstance(recipe_steps, list) or len(recipe_steps) != 2):
        raise ValueError("compile/link receipt provenance is incomplete")
    for base_step, recipe_step in zip(base.get("compiler_and_link_steps", []), recipe_steps):
        if _normalize_step(base_step) != _normalize_step(recipe_step):
            raise ValueError("build receipt commands differ from pinned compile/link receipt")
    rows = verification.get("rows")
    if verification.get("checked_files") != 53 or verification.get("mismatches") != []:
        raise ValueError("input verification report is incomplete")
    if not isinstance(rows, list) or len(rows) != 53:
        raise ValueError("input verification report must have 53 rows")
    seen = set()
    for row in rows:
        path = Path(row["path"]).resolve(strict=True)
        if path in seen:
            raise ValueError("duplicate pinned native input")
        seen.add(path)
        if sha256_file(path) != row["expected_sha256"]:
            raise ValueError(f"pinned native input changed: {path}")
    steps = base.get("compiler_and_link_steps")
    if (not isinstance(steps, list) or len(steps) != 2
            or steps[0].get("stage") != "compile" or steps[1].get("stage") != "link"):
        raise ValueError("base build receipt lacks the reviewed compile/link pair")
    if output_dir.exists():
        raise FileExistsError(f"refusing existing output directory: {output_dir}")
    output_dir.mkdir(parents=True)
    generated = output_dir / "fresh_prefill_segmented_forced.cpp"
    generated.write_bytes(source.read_bytes())
    object_path = output_dir / "fresh_prefill_segmented_forced.o"
    binary_path = output_dir / "strata-batched-fresh-interleaved-raw-audit"
    commands = [
        _rewrite_compile(steps[0]["argv"], generated, object_path),
        _rewrite_link(steps[1]["argv"], object_path, binary_path),
    ]
    results = []
    for phase, argv, step in zip(("compile", "link"), commands, steps):
        completed = subprocess.run(argv, cwd=step["cwd"], text=True,
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                   check=False)
        result = {"phase": phase, "argv": argv, "cwd": step["cwd"],
                  "exit_code": completed.returncode,
                  "stdout": completed.stdout, "stderr": completed.stderr}
        (output_dir / f"{phase}.json").write_text(
            json.dumps(result, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        results.append(result)
        if completed.returncode != 0:
            raise RuntimeError(f"audit-only native {phase} failed ({completed.returncode})")
    receipt = {
        "schema": "cli-seed-stop-interleaved-raw-oracle-build-v1",
        "status": "BUILT_NOT_EXECUTED", "runtime_executed": False,
        "gpu_launched": False, "base_source_sha256": PINS["base_source"],
        "base_build_receipt_sha256": PINS["base_build"],
        "compile_link_receipt_sha256": PINS["compile_link"],
        "input_verification_sha256": PINS["input_verification"],
        "verified_input_count": 53, "verified_input_mismatches": [],
        "audit_driver_source_sha256": sha256_file(generated),
        "binary_sha256": sha256_file(binary_path),
        "compile_link_steps": results,
        "binary": str(binary_path), "source": str(generated),
        "static_input_paths": sorted(map(str, seen)),
    }
    target = output_dir / "build.json"
    target.write_text(json.dumps(receipt, sort_keys=True, indent=2) + "\n",
                      encoding="utf-8")
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--base-source", type=Path, required=True)
    parser.add_argument("--build-receipt", type=Path, required=True)
    parser.add_argument("--compile-link-receipt", type=Path, required=True)
    parser.add_argument("--input-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    result = build(source=args.source.resolve(strict=True),
                   base_source=args.base_source.resolve(strict=True),
                   build_receipt=args.build_receipt.resolve(strict=True),
                   compile_link_receipt=args.compile_link_receipt.resolve(strict=True),
                   input_report=args.input_report.resolve(strict=True),
                   output_dir=args.output_dir.resolve(strict=False))
    print(json.dumps({key: result[key] for key in (
        "status", "audit_driver_source_sha256", "binary_sha256", "binary")},
        sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
