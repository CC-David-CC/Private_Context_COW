#!/usr/bin/env python3
"""Build an isolated native masked-argmax command from a frozen build receipt."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time


PROTOCOL = "finite-mask-argmax-v1"
READY_ANCHOR = ',\\"selected_token_protocol\\":\\"consumed-selected-receipt-v1\\"}'
STATS_ANCHOR = r'<< ",\"prefill_chunks\":" << prefill_chunks'


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def select_reference(scores, mask: bytes) -> int:
    vocab = len(scores)
    need = vocab // 8 + (vocab % 8 != 0)
    if vocab <= 0 or len(mask) != need:
        raise ValueError("mask width mismatch")
    used = vocab % 8
    if used and mask[-1] & ~((1 << used) - 1):
        raise ValueError("nonzero mask padding bits")
    import math
    if any(not math.isfinite(float(value)) for value in scores):
        raise ValueError("nonfinite logits")
    allowed = [i for i in range(vocab) if mask[i // 8] & (1 << (i % 8))]
    if not allowed:
        raise ValueError("empty grammar support")
    return max(allowed, key=lambda i: (float(scores[i]), -i))


def instrument_grammar_selector(source: str) -> str:
    if source.count(READY_ANCHOR) != 1:
        raise ValueError("ready capability anchor must occur exactly once")
    if source.count(STATS_ANCHOR) != 1:
        raise ValueError("stats receipt anchor must occur exactly once")
    if 'command == "select_mask"' in source or "grammar_selection_protocol" in source:
        raise ValueError("driver already contains grammar selector protocol")
    ready = READY_ANCHOR[:-1] + ',\\"grammar_selection_protocol\\":\\"' + PROTOCOL + '\\"}'
    source = source.replace(READY_ANCHOR, ready, 1)
    include_anchor = '#include <fstream>'
    if source.count(include_anchor) != 1:
        raise ValueError("fstream include anchor must occur exactly once")
    source = source.replace(include_anchor, include_anchor + '\n#include "grammar_select.h"', 1)
    logits_anchor = '    std::vector<float> logits((size_t)forward.vocab());'
    if source.count(logits_anchor) != 1:
        raise ValueError("host logits anchor must occur exactly once")
    source = source.replace(logits_anchor,
        logits_anchor +
        '\n    uint64_t grammar_selection_calls = 0, grammar_selection_logit_bytes = 0, grammar_selection_mask_bytes = 0;', 1)

    command_anchor = '} else if (command == "generate") {'
    if source.count(command_anchor) != 1:
        raise ValueError("generate command anchor must occur exactly once")
    command = r''' } else if (command == "select_mask") {
            int64_t expected_consumed = 0;
            std::string mask_path, extra;
            if (!(in >> expected_consumed >> std::quoted(mask_path)) || (in >> extra)) {
                error_reply("select_mask requires expected consumed count and mask path"); continue;
            }
            const auto actual_consumed = session.ids().size();
            if (actual_consumed == 0 || expected_consumed <= 0 ||
                static_cast<uint64_t>(expected_consumed) != actual_consumed) {
                error_reply("select_mask consumed count mismatch"); continue;
            }
            const auto& frontier = session.logits();
            const size_t vocab = static_cast<size_t>(forward.vocab());
            if (vocab == 0 || frontier.size() != vocab) {
                error_reply("select_mask frontier extent mismatch"); continue;
            }
            const size_t mask_bytes = vocab / 8 + (vocab % 8 != 0 ? 1 : 0);
            std::ifstream mask_file(mask_path, std::ios::binary);
            if (!mask_file) { error_reply("select_mask could not open mask file"); continue; }
            std::vector<uint8_t> mask(mask_bytes);
            mask_file.read(reinterpret_cast<char*>(mask.data()), static_cast<std::streamsize>(mask_bytes));
            if (mask_file.gcount() != static_cast<std::streamsize>(mask_bytes)) {
                error_reply("select_mask truncated mask file"); continue;
            }
            char extra_byte = 0;
            if (mask_file.get(extra_byte)) { error_reply("select_mask extra mask bytes"); continue; }
            if (!mask_file.eof()) { error_reply("select_mask mask read failed"); continue; }
            grammar_select::Result selection{};
            try {
                selection = grammar_select::select(frontier.data(), vocab, mask.data(), mask.size(),
                    actual_consumed, static_cast<size_t>(expected_consumed));
            } catch (const std::exception& e) { error_reply(e.what()); continue; }
            ++grammar_selection_calls;
            grammar_selection_logit_bytes += static_cast<uint64_t>(vocab) * sizeof(float);
            grammar_selection_mask_bytes += selection.mask_bytes;
            std::cout << "{\"status\":\"ok\",\"selection_protocol\":\"finite-mask-argmax-v1\",\"consumed\":"
                      << actual_consumed << ",\"token\":" << selection.token
                      << ",\"logits_count\":" << selection.logits_count
                      << ",\"mask_bytes\":" << selection.mask_bytes << "}" << std::endl;
        } else if (command == "generate") {'''
    source = source.replace(command_anchor, command, 1)
    if source.count(STATS_ANCHOR) != 1:
        raise ValueError("stats receipt anchor lost during transformation")
    stats_new = STATS_ANCHOR \
        + r' << ",\"grammar_selection_calls\":" << grammar_selection_calls' \
        + r' << ",\"grammar_selection_logit_bytes\":" << grammar_selection_logit_bytes' \
        + r' << ",\"grammar_selection_mask_bytes\":" << grammar_selection_mask_bytes'
    source = source.replace(STATS_ANCHOR, stats_new, 1)
    if source.count("grammar_selection_protocol") != 1 or source.count('command == "select_mask"') != 1:
        raise ValueError("selector transformation did not produce unique protocol anchors")
    return source


def _command(receipt: dict, label: str) -> dict:
    rows = [r for r in receipt.get("commands", []) if r.get("label") == label]
    if len(rows) != 1 or rows[0].get("exit_code") != 0 or not isinstance(rows[0].get("argv"), list):
        raise ValueError(f"receipt requires one successful {label} command")
    return rows[0]


def _verify_inputs(receipt: dict) -> dict[str, str]:
    items = receipt.get("verified_unchanged_inputs")
    if not isinstance(items, dict) or not items:
        raise ValueError("receipt lacks verified unchanged input hashes")
    verified = {}
    for name, expected in items.items():
        path = Path(name)
        if not path.is_file() or sha256(path) != expected:
            raise ValueError(f"pinned input missing or changed: {path}")
        verified[str(path)] = expected
    return verified


def build(receipt_path: Path, driver_path: Path, header_path: Path, output_dir: Path) -> dict:
    receipt_path = receipt_path.resolve(strict=True)
    driver_path = driver_path.resolve(strict=True)
    header_path = header_path.resolve(strict=True)
    if output_dir.exists():
        raise FileExistsError(f"output directory already exists: {output_dir}")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("status") != "BUILT_NOT_EXECUTED":
        raise ValueError("source receipt is not frozen BUILT_NOT_EXECUTED")
    inputs = _verify_inputs(receipt)
    pinned = receipt.get("instrumented_driver", {})
    if pinned.get("path") != str(driver_path) or pinned.get("sha256") != sha256(driver_path):
        raise ValueError("driver does not match receipt's selected-receipt driver bytes")
    compile_row = _command(receipt, "compile-instrumented-driver")
    link_row = _command(receipt, "link-instrumented-sut")
    source = instrument_grammar_selector(driver_path.read_text(encoding="utf-8"))
    output_dir.mkdir(parents=True)
    generated_cpp, generated_obj = output_dir / "branch_driver.cpp", output_dir / "branch_driver.o"
    executable = output_dir / "strata-batched-branch-lazy-grammar-selector"
    copied_header = output_dir / "grammar_select.h"
    copied_header.write_bytes(header_path.read_bytes())
    generated_cpp.write_text(source, encoding="utf-8", newline="\n")
    compile_argv = list(compile_row["argv"])
    old_cpp = str(receipt["instrumented_driver"]["path"])
    old_obj = str(receipt["instrumented_driver_object"]["path"])
    old_obj_out = [x for x in compile_argv if x.endswith("/branch_driver.o")]
    if len(old_obj_out) != 1:
        raise ValueError("recorded compile argv must have one driver object output")
    compile_argv[compile_argv.index(old_cpp)] = str(generated_cpp)
    compile_argv[compile_argv.index(old_obj_out[0])] = str(generated_obj)
    compile_argv.extend(["-I", str(output_dir)])
    link_argv = list(link_row["argv"])
    if link_argv.count(old_obj) != 1:
        raise ValueError("recorded link argv must contain pinned driver object exactly once")
    link_argv[link_argv.index(old_obj)] = str(generated_obj)
    output_markers = [i for i, value in enumerate(link_argv[:-1]) if value == "-o"]
    if len(output_markers) != 1:
        raise ValueError("recorded link argv must have one output")
    previous_binary = link_argv[output_markers[0] + 1]
    link_argv[output_markers[0] + 1] = str(executable)
    cwd = Path(compile_row["cwd"]).resolve(strict=True)
    if Path(link_row["cwd"]).resolve(strict=True) != cwd:
        raise ValueError("compile and link cwd differ")
    command_rows = []
    for label, argv in (("compile-selector-driver", compile_argv), ("link-selector-sut", link_argv)):
        start = time.monotonic()
        proc = subprocess.run(argv, cwd=cwd, capture_output=True, check=False)
        elapsed = time.monotonic() - start
        out, err = output_dir / f"{label}.stdout.log", output_dir / f"{label}.stderr.log"
        out.write_bytes(proc.stdout); err.write_bytes(proc.stderr)
        command_rows.append({"label": label, "argv": argv, "cwd": str(cwd), "exit_code": proc.returncode,
                             "seconds": elapsed, "stdout_sha256": sha256(out), "stderr_sha256": sha256(err)})
        if proc.returncode:
            raise RuntimeError(f"{label} failed; inspect {err}")
    result = {"status": "BUILT_NOT_EXECUTED", "numerical_qualification": "UNRUN",
              "scope": "new selector command in generated driver; all other linked objects/libraries unchanged",
              "builder_source": {"path": str(Path(__file__).resolve()), "sha256": sha256(Path(__file__).resolve())},
              "source_receipt": {"path": str(receipt_path), "sha256": sha256(receipt_path)},
              "source_driver": {"path": str(driver_path), "sha256": sha256(driver_path)},
              "selector_header": {"path": str(copied_header), "sha256": sha256(copied_header)},
              "generated_driver": {"path": str(generated_cpp), "sha256": sha256(generated_cpp)},
              "generated_object": {"path": str(generated_obj), "sha256": sha256(generated_obj)},
              "verified_unchanged_inputs": inputs, "previous_binary": previous_binary,
              "commands": command_rows,
              "binary": {"path": str(executable), "sha256": sha256(executable), "bytes": executable.stat().st_size}}
    (output_dir / "build.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--receipt", type=Path, required=True)
    p.add_argument("--driver", type=Path, required=True)
    p.add_argument("--header", type=Path, default=Path(__file__).resolve().parents[1] / "native" / "grammar_select.h")
    p.add_argument("--output-dir", type=Path, required=True)
    a = p.parse_args(argv)
    try:
        result = build(a.receipt, a.driver, a.header, a.output_dir)
    except Exception as e:
        print(f"grammar selector build failed: {type(e).__name__}: {e}", file=sys.stderr)
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
