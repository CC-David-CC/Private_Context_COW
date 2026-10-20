#!/usr/bin/env python3
"""Audit CPU-side GPU selection binding for the prepared Q8 profile.

This imports the candidate backend and asks its command builder to resolve
GPU 0. It never constructs a backend or launches a native executable.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-root", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--expected-profile-sha256", required=True)
    parser.add_argument("--expected-backend-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    profile_bytes = args.profile.read_bytes()
    if hashlib.sha256(profile_bytes).hexdigest() != args.expected_profile_sha256:
        raise SystemExit("profile SHA mismatch")
    backend_path = args.api_root / "src/context_time_travel/strata_backend.py"
    backend_sha = sha256(backend_path)
    if backend_sha != args.expected_backend_sha256:
        raise SystemExit("candidate backend SHA mismatch")
    campaign_path = args.api_root / "scripts/pool_campaign.py"
    probe_path = args.api_root / "scripts/native_request_probe.py"
    campaign_text = campaign_path.read_text(encoding="utf-8")
    probe_text = probe_path.read_text(encoding="utf-8")
    if "StrataBackend(profile,gpu_id=gpu)" not in campaign_text:
        raise SystemExit("campaign factory no longer passes its selected GPU id")
    if "StrataBackend(profile)" not in probe_text:
        raise SystemExit("standalone probe no longer uses the recorded no-id constructor")

    sys.path.insert(0, str(args.api_root / "src"))
    from context_time_travel.strata_backend import worker_command

    profile = json.loads(profile_bytes)
    template = list(profile["execution"]["sut_command"])
    effective = worker_command(profile, 0)
    expected = template + ["--gpu", "0"]
    if effective != expected:
        raise SystemExit("effective SUT command does not append exactly --gpu 0")
    if profile["execution"].get("GPU") != [0]:
        raise SystemExit("profile does not pin exactly GPU 0")

    # The standalone probe calls StrataBackend(profile), which is equivalent
    # to worker_command(profile, None); distinguish it from the campaign.
    direct = worker_command(profile, None)
    if direct != template or "--gpu" in direct:
        raise SystemExit("unexpected gpu_id=None command behavior")

    output = {
        "schema": 1,
        "status": "PASS_CPU_COMMAND_BINDING_ONLY",
        "scope": "Imported command-builder audit only; no backend construction, native process, GPU call, inference, or timing.",
        "api_root": str(args.api_root.resolve()),
        "profile_path": str(args.profile.resolve()),
        "profile_sha256": args.expected_profile_sha256,
        "execution_profile_id": profile["execution_profile_id"],
        "backend_path": str(backend_path.resolve()),
        "backend_sha256": backend_sha,
        "campaign_factory_path": str(campaign_path.resolve()),
        "campaign_factory_sha256": sha256(campaign_path),
        "campaign_factory_call": "StrataBackend(profile,gpu_id=gpu)",
        "standalone_probe_path": str(probe_path.resolve()),
        "standalone_probe_sha256": sha256(probe_path),
        "standalone_probe_call": "StrataBackend(profile)",
        "profile_gpu_selection": profile["execution"]["GPU"],
        "pool_factory_invocation": "StrataBackend(profile, gpu_id=gpu)",
        "effective_gpu0_sut_command": effective,
        "effective_gpu0_flag_count": effective.count("--gpu"),
        "standalone_probe_invocation": "StrataBackend(profile)",
        "gpu_id_none_sut_command": direct,
        "gpu_id_none_has_gpu_flag": "--gpu" in direct,
        "qualification_boundary": "This establishes command construction for an explicit pool-factory GPU id. It does not establish native execution or the behavior of a separately launched probe binary.",
    }
    payload = (json.dumps(output, indent=2, sort_keys=True) + "\n").encode()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as stream:
        stream.write(payload)
        stream.flush()
    print(json.dumps({"status": output["status"], "output": str(args.output),
                      "sha256": hashlib.sha256(payload).hexdigest()}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
