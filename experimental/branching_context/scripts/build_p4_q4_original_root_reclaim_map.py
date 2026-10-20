#!/usr/bin/env python3
"""Prepare a separate read-only map for the three original Q4 campaign roots."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
PREP = BASE / "evidence/P4-Q4-EXECUTION-PREP-2026-10-15.json"
BEFORE = BASE / "evidence/P4-Q4-ORIGINAL-ROOT-SOURCE-INVENTORY-2026-10-15.json"
AFTER = BASE / "evidence/P4-Q4-ORIGINAL-ROOT-SOURCE-INVENTORY-AFTER-RECLAIM-2026-10-15.json"
OUTPUT = BASE / "evidence/P4-Q4-ORIGINAL-ROOT-RECLAIM-MAP-2026-10-15.json"
ARCHIVE = "/srv/data/git/private_context_cow.evidence/20261015-p5-q4-full96/p5-q4-full96-20261015.tar"
ARCHIVE_SHA256 = "5e7339ce9d751afe64e5bf166c5c3f0ea6b7cc62255cacca7021ee0f8771d0ce"
MEMBER_LISTING_SHA256 = "d766cb7a4d84c2db07c2fc32c17657f442b30374f7636d36173e619bd66333b3"
ARCHIVE_BYTES = 41838049280


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    prep_bytes, before_bytes, after_bytes = PREP.read_bytes(), BEFORE.read_bytes(), AFTER.read_bytes()
    prep, before, after = json.loads(prep_bytes), json.loads(before_bytes), json.loads(after_bytes)
    rootmap = prep["root_payload_restore"]
    if rootmap["map_status"] != "READ_ONLY_INPUTS_VALIDATED" or rootmap["archive_sha256"] != ARCHIVE_SHA256:
        raise SystemExit("P4 preparation archive pins/status changed")
    if before["status"] != "READ_ONLY_Q4_SOURCE_ROOT_INVENTORY_PASS" or after["status"] != before["status"]:
        raise SystemExit("source inventory failed")
    if before["files"] != after["files"] or len(before["files"]) != 3:
        raise SystemExit("original Q4 source-root files changed or wrong set")
    prep_rows = {row["payload_sha256"]: row for row in rootmap["payloads"]}
    archive_member_results = {
        "cbc82b900ab53e37a0b823b59d48a7c6a202455b87584a3c4b7b4a1868afe4ca": {"exit_code": 0, "sha256": "cbc82b900ab53e37a0b823b59d48a7c6a202455b87584a3c4b7b4a1868afe4ca"},
        "9e05cf1b02251f5f70ef7fe3f55cd928b5784b9f537bd663c0f02502eecea9f7": {"exit_code": 0, "sha256": "9e05cf1b02251f5f70ef7fe3f55cd928b5784b9f537bd663c0f02502eecea9f7"},
        "43d65bdc1877e01b9d3bbf9fb7d6b88e594bc946607fc61cd70961cf4721a3a7": {"exit_code": 0, "sha256": "43d65bdc1877e01b9d3bbf9fb7d6b88e594bc946607fc61cd70961cf4721a3a7"},
    }
    rows = []
    for source in before["files"]:
        digest = source["sha256"]
        prep_row = prep_rows[digest]
        archive_result = archive_member_results[digest]
        if digest != Path(source["source_state_path"]).stem or digest != archive_result["sha256"]:
            raise SystemExit(f"payload filename/hash/archive mismatch: {digest}")
        if source["bytes"] != prep_row["payload_bytes"] or source["metadata_sha256"] != prep_row["manifest_sha256"]:
            raise SystemExit(f"source payload/metadata differ from pinned prep: {digest}")
        if prep_row["archive_member"].split("/")[-1] != Path(source["source_state_path"]).name:
            raise SystemExit(f"archive member path mismatch: {digest}")
        rows.append({
            "root_snapshot": source["snapshot"],
            "checkpoint_metadata_path": source["metadata_path"],
            "checkpoint_metadata_sha256": source["metadata_sha256"],
            "checkpoint_metadata_bytes": source["metadata_bytes"],
            "source_state_path": source["source_state_path"],
            "source_state_bytes": source["bytes"],
            "source_state_sha256": digest,
            "source_state_mode": source["mode"],
            "source_state_nlink": source["nlink"],
            "source_device": source["device"],
            "source_inode": source["inode"],
            "source_mtime_ns": source["mtime_ns"],
            "source_stable_during_hash": source["stable_during_hash"],
            "archive_member": prep_row["archive_member"],
            "archive_member_stream_sha256": archive_result["sha256"],
            "archive_member_stream_exit_code": archive_result["exit_code"],
            "source_matches_archive_member": True,
        })
    total = sum(row["source_state_bytes"] for row in rows)
    result = {
        "schema": "p4-q4-original-root-readonly-map-v1",
        "status": "READ_ONLY_ARCHIVE_BOUND_NOT_RECLAIMED",
        "scope": "Exactly three original Q4 64k campaign root checkpoint payloads. Kept separate from the nine resident-arm output payloads; Q4 profiling still uses these roots.",
        "verified_archive": {
            "path": ARCHIVE,
            "sha256": ARCHIVE_SHA256,
            "bytes": ARCHIVE_BYTES,
            "member_listing_sha256": MEMBER_LISTING_SHA256,
            "source_prep_path": str(PREP.relative_to(BASE.parent.parent)),
            "source_prep_sha256": sha(PREP),
            "archive_integrity_scope": "Previously verified full archive SHA/member listing, plus three fresh tar -xOf|sha256sum member streams with pipefail; this legacy tar has no root MANIFEST.json.",
        },
        "source_inventory_before_path": str(BEFORE.relative_to(BASE.parent.parent)),
        "source_inventory_before_sha256": sha(BEFORE),
        "source_inventory_after_resident_reclaim_path": str(AFTER.relative_to(BASE.parent.parent)),
        "source_inventory_after_resident_reclaim_sha256": sha(AFTER),
        "source_rows_unchanged_across_resident_reclaim": True,
        "source_payload_count": len(rows),
        "source_payload_bytes": total,
        "source_payloads": rows,
        "capacity_note": "These original Q4 source payloads remain present and were not removed. Their 5,491,806,696 bytes are only a future capacity possibility after Q4 profiling and separate authorization.",
        "prohibition": "Read-only preparation only: no source checkpoint was deleted, restored, or modified.",
    }
    OUTPUT.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "output": str(OUTPUT), "sha256": sha(OUTPUT), "count": len(rows), "bytes": total}, sort_keys=True))


if __name__ == "__main__":
    main()
