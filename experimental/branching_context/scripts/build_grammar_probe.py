#!/usr/bin/env python3
"""Build an isolated grammar probe binary with an explicit selected-token ack.

This does not modify Strata sources or the frozen build. It transforms the
receipt-pinned generated driver, compiles that one translation unit, and
relinks the exact recorded objects/libraries to a new output directory.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


READY_OLD = ',\\"native_request_protocol\\":\\"consumed-greedy-ack-v1\\"}'
READY_NEW = (',\\"native_request_protocol\\":\\"consumed-greedy-ack-v1\\"'
             ',\\"selected_token_protocol\\":\\"consumed-selected-receipt-v1\\"}')
DECODE_OLD = ('std::cout << "{\\"status\\":\\"ok\\",\\"consumed\\":" '
              '<< session.ids().size() << "}" << std::endl;')
DECODE_NEW = ('std::cout << "{\\"status\\":\\"ok\\",\\"operator\\":'
              '\\"normal-verifier-decode\\",\\"position\\":" << position'
              ' << ",\\"count\\":1,\\"consumed\\":" << session.ids().size()'
              ' << ",\\"chunks\\":[[" << position << ",1]]}" << std::endl;')


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def instrument_selected_token_receipt(source: str) -> str:
    """Add a separate readiness capability and receipt after successful finish."""
    if source.count(READY_OLD) != 1:
        raise ValueError("ready capability anchor must occur exactly once")
    decode_at = source.index('} else if (command == "decode") {')
    capture_at = source.index('} else if (command == "capture") {', decode_at)
    decode_segment = source[decode_at:capture_at]
    if decode_segment.count(DECODE_OLD) != 1:
        raise ValueError("decode success receipt must be inside the decode handler")
    if "selected_token_protocol" in source or "consumed-selected-receipt-v1" in source:
        raise ValueError("driver already contains a selected-token capability")
    if decode_segment.count("const int64_t position = session.ids().size();") != 1:
        raise ValueError("decode handler must capture one pre-operation position")
    if decode_segment.count("session.finish({token}, logits, error)") != 1:
        raise ValueError("decode handler must have one selected-token finish boundary")
    if decode_segment.index("session.finish({token}, logits, error)") > decode_segment.index(DECODE_OLD):
        raise ValueError("decode receipt must be emitted after successful session.finish")
    source = source.replace(READY_OLD, READY_NEW, 1)
    decode_at = source.index('} else if (command == "decode") {')
    capture_at = source.index('} else if (command == "capture") {', decode_at)
    decode_segment = source[decode_at:capture_at].replace(DECODE_OLD, DECODE_NEW, 1)
    result = source[:decode_at] + decode_segment + source[capture_at:]
    if result.count("selected_token_protocol") != 1 or result.count(DECODE_NEW) != 1:
        raise ValueError("transformed driver did not contain exactly one capability and receipt")
    return result


def _command_rows(receipt: dict, label: str) -> list[dict]:
    rows = [row for row in receipt.get("commands", []) if row.get("label") == label]
    if len(rows) != 1 or not isinstance(rows[0].get("argv"), list) or rows[0].get("exit_code") != 0:
        raise ValueError(f"build receipt must contain one successful {label} command row")
    return rows


def _replace_once(argv: list[str], old: str, new: str, what: str) -> list[str]:
    if argv.count(old) != 1:
        raise ValueError(f"expected exactly one {what} in recorded argv")
    return [new if arg == old else arg for arg in argv]


def _verify_inputs(receipt: dict) -> dict[str, str]:
    inputs = receipt.get("input_sha256")
    if not isinstance(inputs, dict) or not inputs:
        raise ValueError("build receipt has no pinned input_sha256 map")
    checked = {}
    for raw_path, expected in sorted(inputs.items()):
        path = Path(raw_path)
        if not path.is_file():
            raise ValueError(f"pinned build input is missing: {path}")
        actual = sha256(path)
        if actual != expected:
            raise ValueError(f"pinned build input changed: {path}")
        checked[str(path)] = actual
    return checked


def build(receipt_path: Path, driver_path: Path, output_dir: Path) -> dict:
    receipt_path = receipt_path.resolve(strict=True)
    driver_path = driver_path.resolve(strict=True)
    if output_dir.exists():
        raise FileExistsError(f"output directory already exists: {output_dir}")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    checked_inputs = _verify_inputs(receipt)
    if receipt.get("status") != "BUILT_NOT_EXECUTED":
        raise ValueError("source receipt is not a frozen BUILT_NOT_EXECUTED receipt")
    compile_row = _command_rows(receipt, "compile-branch_driver")[0]
    link_row = _command_rows(receipt, "link-sut")[0]
    source_sha = sha256(driver_path)
    expected_source = receipt["input_sha256"].get(str(driver_path))
    if expected_source is None or source_sha != expected_source:
        raise ValueError("driver path/hash is not the source pinned by the build receipt")
    source = driver_path.read_text(encoding="utf-8")
    transformed = instrument_selected_token_receipt(source)

    output_dir.mkdir(parents=True, exist_ok=False)
    generated_cpp = output_dir / "branch_driver.cpp"
    generated_obj = output_dir / "branch_driver.o"
    executable = output_dir / "strata-batched-branch-lazy-grammar"
    generated_cpp.write_text(transformed, encoding="utf-8", newline="\n")
    compile_argv = list(compile_row["argv"])
    compile_argv = _replace_once(compile_argv, str(driver_path), str(generated_cpp), "driver source")
    old_obj_candidates = [arg for arg in compile_argv if arg.endswith("/branch_driver.o")]
    if len(old_obj_candidates) != 1:
        raise ValueError("compile command must name exactly one branch_driver.o output")
    compile_argv = _replace_once(compile_argv, old_obj_candidates[0], str(generated_obj), "driver object output")
    link_argv = list(link_row["argv"])
    old_obj = old_obj_candidates[0]
    link_argv = _replace_once(link_argv, old_obj, str(generated_obj), "linked driver object")
    output_index = [i for i, arg in enumerate(link_argv[:-1]) if arg == "-o"]
    if len(output_index) != 1:
        raise ValueError("link command must have exactly one -o output")
    old_executable = link_argv[output_index[0] + 1]
    link_argv = _replace_once(link_argv, old_executable, str(executable), "linked executable output")
    cwd = Path(compile_row.get("cwd", "")).resolve(strict=True)
    if Path(link_row.get("cwd", "")).resolve(strict=True) != cwd:
        raise ValueError("compile and link commands must use the same recorded cwd")

    commands = []
    for label, argv in (("compile-instrumented-driver", compile_argv), ("link-instrumented-sut", link_argv)):
        started = time.monotonic()
        proc = subprocess.run(argv, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        elapsed = time.monotonic() - started
        stdout_path = output_dir / f"{label}.stdout.log"
        stderr_path = output_dir / f"{label}.stderr.log"
        stdout_path.write_bytes(proc.stdout)
        stderr_path.write_bytes(proc.stderr)
        commands.append({"label": label, "argv": argv, "cwd": str(cwd),
                         "exit_code": proc.returncode, "seconds": elapsed,
                         "stdout_sha256": sha256(stdout_path), "stderr_sha256": sha256(stderr_path)})
        if proc.returncode != 0:
            raise RuntimeError(f"{label} failed with exit {proc.returncode}; see {stderr_path}")

    if not executable.is_file():
        raise RuntimeError("linker returned zero without creating the requested executable")
    result = {
        "status": "BUILT_NOT_EXECUTED",
        "scope": "isolated generated-driver receipt extension; all other link inputs reused from pinned receipt",
        "numerical_qualification": "UNRUN",
        "source_build_receipt": {"path": str(receipt_path), "sha256": sha256(receipt_path)},
        "pinned_driver": {"path": str(driver_path), "sha256": source_sha},
        "instrumented_driver": {"path": str(generated_cpp), "sha256": sha256(generated_cpp)},
        "instrumented_driver_object": {"path": str(generated_obj), "sha256": sha256(generated_obj),
                                       "bytes": generated_obj.stat().st_size},
        "verified_unchanged_inputs": checked_inputs,
        "commands": commands,
        "binary": {"path": str(executable), "sha256": sha256(executable), "bytes": executable.stat().st_size},
    }
    (output_dir / "build.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path, required=True, help="frozen successful native build receipt")
    parser.add_argument("--driver", type=Path, required=True, help="generated driver source named in receipt")
    parser.add_argument("--output-dir", type=Path, required=True, help="new external output directory; must not exist")
    args = parser.parse_args(argv)
    try:
        result = build(args.receipt, args.driver, args.output_dir)
    except Exception as exc:
        print(f"grammar probe build failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
