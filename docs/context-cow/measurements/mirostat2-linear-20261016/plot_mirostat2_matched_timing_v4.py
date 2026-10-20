#!/usr/bin/env python3
"""Create a matched before/after Mirostat2 Q4 8K performance figure."""
from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


BASE = Path("/home/dflanag3/spatial-artifacts/sampler-extensions-20261016-v1")
BACKUP = Path("/home/dflanag3/spatial-artifacts/mirostat-backup-v1")
OUT = Path("/home/dflanag3/spatial-artifacts/mirostat-linear-plot-20261016-v4")
BASE_RESULT = BASE / "gpu-mirostat2-q4-8k-timing-v1/resume/result.json"
CAND_RESULT = BASE / "gpu-mirostat2-linear-q4-8k-timing-v2/resume/result.json"
COMPARISON = BACKUP / "linear-timing-comparison-v2.json"
PROFILE = BASE / "q4-profile-mirostat2-linear-v1/profile.json"
EXPECTED = {
    "baseline_result_sha256": "6647a80e544364531837330301d23f7ed4fb60c7d3a691e076ba2984dd1dfdbf",
    "candidate_result_sha256": "12b10e9b785d10682dff406667bd98c2ab7e686d1d9042737b21ebbdcbb3c1aa",
    "comparison_sha256": "50aed31ee883b97646b2f4145bd1268ba8786ee605ac4c150b2dcbf7c1f59c39",
    "fixture_source_sha256": "6f517dbf9380416f581a1ac222434d99f97567f5b1f79f8a834615f3b7b28a9d",
    "selected_fixture_sha256": "78553f0140428c589e12e7fb01728e119f461ea216e239edaf4379875bd19296",
    "baseline_profile_sha256": "cdeb462cbd8154322849838e2dc0a79315d7b7941a208c613b7d1c6ae50875be",
    "candidate_profile_sha256": "09ec62ea167691d6a790f4f019142f25675b2887c6011c7672263045cb24a6d4",
    "candidate_profile_file_sha256": "09ec62ea167691d6a790f4f019142f25675b2887c6011c7672263045cb24a6d4",
    "candidate_elf_sha256": "b545e0f0c2c344ce7138dd9a8bd239668b2a03d81791c3b1e579a3b943f97a65",
    "model_contract_id": "ff923fd4e0fb068eea3a39a79f130860fdbdda8f686ecb0c470fa09a27808827",
}
COLORS = {"baseline": "#777777", "candidate": "#007F86"}
LABELS = {"baseline": "Baseline", "candidate": "Linear selector"}


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def read_json(path: Path) -> dict:
    return json.loads(path.read_bytes())


def extract_rows(path: Path, variant: str) -> list[dict]:
    result = read_json(path)
    require(result.get("status") == "PASS", f"{variant} run is not PASS")
    require(result.get("native_exit_code") == 0 and result.get("native_pid_absent") is True,
            f"{variant} native process exit/absence not proven")
    measured = [r for r in result.get("timed_rows", []) if r.get("phase") == "measured"]
    require(len(measured) == 6, f"{variant} expected six measured rows; found {len(measured)}")
    require({r.get("branch") for r in measured} == {"A", "B"},
            f"{variant} branch labels differ")
    require(result.get("controller_scope", {}).get("grammar") is False and
            result.get("controller_scope", {}).get("policy") == {
                "min_p": 0.0, "mirostat": 2, "mirostat_eta": 0.05,
                "mirostat_tau": 4.0, "repetition_penalty": 1.0,
                "temperature": 0.8, "top_k": 0, "top_p": 1.0},
            f"{variant} controller policy/scope changed")
    for branch in ("A", "B"):
        branch_rows = [r for r in measured if r.get("branch") == branch]
        require(len(branch_rows) == 3 and sorted(r.get("repetition") for r in branch_rows) == [1, 2, 3],
                f"{variant}/{branch} must contain measured repetitions 1-3 only")
    result_rows = []
    for row in measured:
        before, after = row["counters_before"], row["counters_after"]
        calls = after["sampling_mirostat2_calls"] - before["sampling_mirostat2_calls"]
        sampling_ns = after["sampling_mirostat2_ns"] - before["sampling_mirostat2_ns"]
        require(calls == row["output_count"] == 128, f"{variant} sampler calls/output count mismatch")
        require(after["forward_tokens"] - before["forward_tokens"] == 640,
                f"{variant} timed forward work must be 512 suffix + 128 outputs")
        require(after["prefill_tokens"] - before["prefill_tokens"] == 512,
                f"{variant} timed suffix prefill changed")
        require(after["loaded_file_bytes"] - before["loaded_file_bytes"] == 0,
                f"{variant} had a timed disk load")
        require(math.isfinite(row["generation_tps"]) and math.isfinite(row["request_seconds"]),
                f"{variant} has nonfinite timing")
        result_rows.append({
            "variant": variant,
            "branch": row["branch"],
            "repetition": row["repetition"],
            "generation_tps": row["generation_tps"],
            "host_sampler_ms_per_output": sampling_ns / calls / 1_000_000,
            "request_seconds": row["request_seconds"],
            "output_count": row["output_count"],
            "suffix_prefill_tokens": 512,
            "total_forward_tokens": 640,
            "sampler_calls": calls,
            "loaded_file_bytes": 0,
            "frontier_sha256": row["final_frontier_sha256"],
        })
    return result_rows


def median(values: list[float]) -> float:
    ordered = sorted(values)
    return ordered[len(ordered) // 2]


def main() -> int:
    OUT.mkdir(mode=0o700, parents=True, exist_ok=True)
    require(sha_file(BASE_RESULT) == EXPECTED["baseline_result_sha256"], "baseline result hash changed")
    require(sha_file(CAND_RESULT) == EXPECTED["candidate_result_sha256"], "candidate result hash changed")
    require(sha_file(COMPARISON) == EXPECTED["comparison_sha256"], "matched comparison hash changed")
    require(sha_file(PROFILE) == EXPECTED["candidate_profile_file_sha256"],
            "candidate profile file hash changed")
    profile = read_json(PROFILE)
    require(profile["execution"]["GPU"] == [0] and
            profile["execution"]["sampling_protocol"] == "full-vocab-mirostat2-u53-v6" and
            profile["execution"]["resume_contract"] == "identical-native-operators-v1" and
            profile["execution"]["prefill_chunk"] == 8192,
            "candidate execution profile does not match requested GPU/operator/prefill scope")
    require("RTX PRO 6000 Blackwell" in profile["hardware"] and
            profile["model_contract"]["architecture"] == "qwen4exp",
            "unexpected candidate hardware/model architecture")
    comparison = read_json(COMPARISON)
    require(comparison.get("status") == "PASS", "matched comparison not PASS")
    require(comparison["source_fixture_sha256"] == EXPECTED["fixture_source_sha256"],
            "fixture source hash changed")
    require(comparison["selected_fixture_sha256"] == EXPECTED["selected_fixture_sha256"],
            "selected fixture digest changed")
    require(comparison["baseline"]["profile_sha256"] == EXPECTED["baseline_profile_sha256"],
            "baseline profile hash changed")
    require(comparison["candidate"]["profile_sha256"] == EXPECTED["candidate_profile_sha256"],
            "candidate profile hash changed")
    require(comparison["candidate"]["native_elf_sha256_actual"] == EXPECTED["candidate_elf_sha256"],
            "candidate ELF hash changed")
    require(comparison["answer_equivalence"]["A"]["exact_match"] is True and
            comparison["answer_equivalence"]["B"]["exact_match"] is True,
            "A/B answer equivalence not proven")
    require(comparison["answer_equivalence"]["A"]["model_contract_id"] == EXPECTED["model_contract_id"] and
            comparison["answer_equivalence"]["B"]["model_contract_id"] == EXPECTED["model_contract_id"],
            "matched model contract differs")
    workload = comparison["workload"]
    require(workload == {"branches": ["A", "B"], "measured_repetitions_per_branch": 3,
                         "output_limit": 128, "prefix_tokens": 8192,
                         "suffix_tokens": 512, "warmups_per_branch_excluded": 1},
            f"unexpected matched workload: {workload}")
    policy = comparison["policy"]
    require(policy == {"min_p": 0.0, "mirostat": 2, "mirostat_eta": 0.05,
                       "mirostat_tau": 4.0, "repetition_penalty": 1.0,
                       "temperature": 0.8, "top_k": 0, "top_p": 1.0},
            "policy changed between matched runs")

    rows = extract_rows(BASE_RESULT, "baseline") + extract_rows(CAND_RESULT, "candidate")
    for branch in ("A", "B"):
        left = [r["frontier_sha256"] for r in rows if r["branch"] == branch and r["variant"] == "baseline"]
        right = [r["frontier_sha256"] for r in rows if r["branch"] == branch and r["variant"] == "candidate"]
        require(left == right, f"{branch} measured frontier hashes differ")

    csv_path = OUT / "matched-measurements.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    metrics = [
        ("generation_tps", "Generation throughput", "tokens / second · higher is better"),
        ("host_sampler_ms_per_output", "Host sampler cost", "ms / output token · lower is better"),
        ("request_seconds", "Complete request", "seconds · lower is better"),
    ]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.titlesize": 13, "axes.labelsize": 10,
                         "svg.fonttype": "none", "figure.facecolor": "white"})
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.2), constrained_layout=False)
    branch_centers = [0.85, 2.05]
    offsets = {"baseline": -0.18, "candidate": 0.18}
    bar_width = 0.31
    for ax, (metric, title, unit) in zip(axes, metrics):
        for branch, center in zip(("A", "B"), branch_centers):
            for variant in ("baseline", "candidate"):
                subset = [r for r in rows if r["branch"] == branch and r["variant"] == variant]
                values = [r[metric] for r in subset]
                med = median(values)
                x = center + offsets[variant]
                ax.bar(x, med, width=bar_width, color=COLORS[variant], alpha=0.85,
                       edgecolor="white", linewidth=0.8, zorder=2)
                # Fixed symmetric jitter shows all three measured repetitions
                # without implying a distribution estimate.
                jitter = [-0.055, 0.0, 0.055]
                ax.scatter([x + d for d in jitter], values, s=24, color=COLORS[variant],
                           edgecolor="white", linewidth=0.6, zorder=4)
                ax.scatter([x], [med], marker="D", s=43, color="white",
                           edgecolor=COLORS[variant], linewidth=1.4, zorder=5)
                ax.annotate(f"{med:.2f}", (x, med), xytext=(0, 8),
                            textcoords="offset points", ha="center", va="bottom",
                            fontsize=8.5, color="#222222")
        ax.set_title(title, pad=11, fontweight="semibold")
        ax.set_ylabel(unit)
        ax.set_xticks(branch_centers, ["A", "B"])
        ax.set_xlim(0.35, 2.55)
        ax.set_ylim(bottom=0)
        ax.grid(axis="y", color="#d9dee2", linewidth=0.8, alpha=0.9)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#9199a0")
        ax.spines["bottom"].set_color("#9199a0")

    legend = [
        plt.Rectangle((0, 0), 1, 1, color=COLORS["baseline"], label="Baseline"),
        plt.Rectangle((0, 0), 1, 1, color=COLORS["candidate"], label="Linear selector"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#555555",
               markeredgecolor="white", markersize=6, label="Measured repetition"),
        Line2D([0], [0], marker="D", color="none", markerfacecolor="white",
               markeredgecolor="#555555", markersize=6, label="Median"),
    ]
    fig.legend(handles=legend, loc="upper center", bbox_to_anchor=(0.5, 0.865),
               ncol=4, frameon=False, fontsize=9)
    fig.suptitle("Mirostat2 selector: matched Q4 8K timing", y=0.985,
                 fontsize=17, fontweight="bold")
    fig.text(0.5, 0.925,
             "8192-token prefix · 512-token suffix · 128 outputs · 1 warmup excluded + 3 measured per branch",
             ha="center", va="center", fontsize=10, color="#343a40")
    fig.text(0.5, 0.015,
             "Resident Python COW API · RTX PRO 6000 Blackwell, GPU 0 · same Q4_K_XL model, policy, schedule, tokens, and sampled frontiers (A/B exact match). Each request includes 512 suffix-prefill tokens; saved prefix is not replayed and no disk bytes are loaded. Model startup and checkpoint write are outside the timed request.",
             ha="center", va="bottom", fontsize=8.2, color="#3e474e", wrap=True)
    fig.subplots_adjust(left=0.055, right=0.99, top=0.73, bottom=0.18, wspace=0.34)
    for suffix, kwargs in (("png", {"dpi": 220}), ("svg", {}), ("pdf", {})):
        fig.savefig(OUT / f"mirostat2-matched-q4-8k.{suffix}", bbox_inches="tight", **kwargs)
    plt.close(fig)

    meta = {
        "status": "PASS",
        "comparison_sha256": sha_file(COMPARISON),
        "baseline_result_sha256": sha_file(BASE_RESULT),
        "candidate_result_sha256": sha_file(CAND_RESULT),
        "fixture_source_sha256": EXPECTED["fixture_source_sha256"],
        "selected_fixture_sha256": EXPECTED["selected_fixture_sha256"],
        "baseline_profile_sha256": EXPECTED["baseline_profile_sha256"],
        "candidate_profile_sha256": EXPECTED["candidate_profile_sha256"],
        "candidate_profile_file_sha256": sha_file(PROFILE),
        "candidate_native_elf_sha256": EXPECTED["candidate_elf_sha256"],
        "plot_script_sha256": sha_file(Path(__file__)),
        "matplotlib_version": matplotlib.__version__,
        "baseline_source_commit": comparison["baseline"]["source_commit"],
        "candidate_source_commit": comparison["candidate"]["source_commit"],
        "model_contract_id": EXPECTED["model_contract_id"],
        "policy": policy,
        "workload": workload,
        "hardware": profile["hardware"],
        "model_architecture": profile["model_contract"]["architecture"],
        "execution_operator": profile["execution"]["operator"],
        "rows": rows,
        "exclusions": ["model startup", "checkpoint creation/write", "warmup repetitions",
                       "saved-prefix replay", "disk loads"],
        "scope_limit": "Matched Q4 Mirostat2 control-plane timing only; not CLI/HTTP overhead, Q8 throughput, longer contexts, or another sampler policy.",
        "outputs": ["mirostat2-matched-q4-8k.png", "mirostat2-matched-q4-8k.svg",
                    "mirostat2-matched-q4-8k.pdf", "matched-measurements.csv"],
    }
    meta["output_sha256"] = {name: sha_file(OUT / name) for name in meta["outputs"]}
    (OUT / "plot-inputs.json").write_text(json.dumps(meta, sort_keys=True, indent=2) + "\n",
                                          encoding="utf-8", newline="\n")
    print(json.dumps({"status": "PASS", "output_dir": str(OUT),
                      "rows": len(rows), "versions": ["baseline", "candidate"],
                      "branches": ["A", "B"], "measured_reps_per_cell": 3}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
