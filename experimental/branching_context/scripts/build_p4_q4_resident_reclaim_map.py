#!/usr/bin/env python3
"""Bind a read-only resident checkpoint inventory to its verified archive receipt."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
INVENTORY = BASE / "evidence/P4-Q4-RESIDENT-RECLAIM-INVENTORY-2026-10-15-v2.json"
ARCHIVE_PROOF = BASE / "evidence/P4-Q4-RESIDENT-ARCHIVE-MATCH-PROOF-2026-10-15.json"
OUTPUT = BASE / "evidence/P4-Q4-RESIDENT-RECLAIM-MAP-2026-10-15.json"
ARCHIVE_PATH = "/srv/data/git/private_context_cow.evidence/20261015-p4-q4-resident-return/p4-q4-resident-return-20261015.tar"
ARCHIVE_SHA256 = "afea6d6117cc9d7df8c3fbc70225d01ceb3dc61a10224192cf5dd3e694b43a60"
RECEIPT_SHA256 = "a35dc82d155dc3598f87ee13f4b333b055ac0c189ce090b3e05047b18e05a235"
MANIFEST_SHA256 = "31787d8bc498ee35a4aeeab3756e1712d473ec99fe17227b7657722b005c7111"
LISTING_SHA256 = "77dfa83047605f111c924243d1e7dba0ac5a9b93e04512f5bc0c332258693ced"
RESIDENT_MEMBER_PREFIX = "resident-arm/output/"
STATE_MEMBERS = [
    ("0bb35ceaf12586db929c4664c39d1a9e5f774f7a5731ae2f177c8658348669ea", 2051119588),
    ("3d3ff630ac37b5c823c198ea892fd7102bc30215e568e7ea251bee57a46bc6ca", 2048403516),
    ("43d65bdc1877e01b9d3bbf9fb7d6b88e594bc946607fc61cd70961cf4721a3a7", 1830602232),
    ("4a0f6795dc1427ae981a01ee5505a21c49361e4823bfe995f0a7f115695386bd", 2047782888),
    ("4f5386bb29cac43cc7e270333a923e2791b78a557db160063ae5ff74c2f7c8db", 2051230200),
    ("9e05cf1b02251f5f70ef7fe3f55cd928b5784b9f537bd663c0f02502eecea9f7", 1830602232),
    ("a72f1893376fe84c19a4c73a074999bb3dbc93e6532368a94b0f73e4779158ba", 2047985660),
    ("cbc82b900ab53e37a0b823b59d48a7c6a202455b87584a3c4b7b4a1868afe4ca", 1830602232),
    ("e4b0b4cd88b52f27ac839c02491eeb4cfcc0ec9b486dfa8d7f44405d485d56d0", 2051648056),
]


def main() -> None:
    inventory_bytes = INVENTORY.read_bytes()
    inventory = json.loads(inventory_bytes)
    proof_bytes = ARCHIVE_PROOF.read_bytes()
    proof = json.loads(proof_bytes)
    if inventory["status"] != "READ_ONLY_INVENTORY_PASS" or inventory["candidate_count"] != 9:
        raise SystemExit("inventory is incomplete or failed")
    if proof.get("status") != "PASS_EXACT_ARCHIVE_MANIFEST_SOURCE_MATCH" or proof.get("execution", {}).get("exit_code") != 0:
        raise SystemExit("executed archive/source comparison proof missing or failed")
    if not proof.get("result", {}).get("archive_manifest_exact_candidate_match"):
        raise SystemExit("executed archive/source comparison did not establish exact match")
    candidates = inventory["candidates"]
    if len(candidates) != len(STATE_MEMBERS):
        raise SystemExit("wrong candidate count")
    by_name = {Path(row["path"]).name for row in candidates}
    expected_names = {f"{digest}.state" for digest, _ in STATE_MEMBERS}
    if by_name != expected_names:
        raise SystemExit("inventory candidate set differs from archived member set")
    member_rows = []
    for digest, size in STATE_MEMBERS:
        name = f"{digest}.state"
        row = next(item for item in candidates if Path(item["path"]).name == name)
        if row["sha256"] != digest or row["size"] != size or not row["stable_during_hash"]:
            raise SystemExit(f"live source candidate mismatch: {name}")
        matched = next((item for item in proof["result"]["matched_rows"] if item["source_path"] == row["path"]), None)
        if not matched or matched["sha256"] != digest or matched["size"] != size:
            raise SystemExit(f"executed archive proof does not match candidate: {name}")
        member_rows.append({
            "archive_member": matched["archive_path"],
            "source_path": row["path"],
            "bytes": size,
            "sha256": digest,
            "source_mode": row["mode"],
            "source_nlink": row["nlink"],
            "source_device": row["device"],
            "source_inode": row["inode"],
            "source_mtime_ns": row["mtime_ns"],
            "source_stable_during_hash": row["stable_during_hash"],
            "archive_manifest_match": "verified by executed proof",
        })
    if sum(row["bytes"] for row in member_rows) != inventory["candidate_total_bytes"]:
        raise SystemExit("source total differs from archive-matched payloads")
    q8_restore_bytes = 5_491_806_696
    resident_arm_reserve = 41_512_139_080
    combined_requirement = q8_restore_bytes + resident_arm_reserve
    current_free = inventory["free_bytes_at_inventory"]
    candidate_bytes = inventory["candidate_total_bytes"]
    result = {
        "schema": "p4-q4-resident-readonly-reclaim-map-v1",
        "status": "PREPARED_READ_ONLY_NOT_UNLINKED",
        "scope": "Only resident-arm output store/checkpoints/*.state files; original Q4 source roots and all protected output metadata are excluded.",
        "source_inventory": {
            "path": str(INVENTORY.relative_to(BASE.parent.parent)),
            "sha256": hashlib.sha256(inventory_bytes).hexdigest(),
            "candidate_count": len(candidates),
            "candidate_total_bytes": candidate_bytes,
            "candidate_max_bytes": inventory["candidate_max_bytes"],
            "device_inode_mtime_nlink_captured": True,
            "all_source_files_mode_0600_nlink_1_and_sha_named": True,
        },
        "verified_archive": {
            "path": ARCHIVE_PATH,
            "sha256": ARCHIVE_SHA256,
            "bytes": 17814958080,
            "receipt_sha256": RECEIPT_SHA256,
            "receiver_status": "PASS",
            "manifest_sha256": MANIFEST_SHA256,
            "manifest_raw_sha256": proof["inputs"]["manifest_raw_sha256"],
            "manifest_raw_bytes": proof["inputs"]["manifest_raw_bytes"],
            "member_listing_sha256": LISTING_SHA256,
            "comparison_proof_path": str(ARCHIVE_PROOF.relative_to(BASE.parent.parent)),
            "comparison_proof_sha256": hashlib.sha256(proof_bytes).hexdigest(),
            "receiver_receipt_sha256": proof["inputs"]["receiver_receipt_sha256"],
            "candidate_archive_members": len(member_rows),
            "candidate_archive_bytes": candidate_bytes,
            "all_candidates_match_manifest_path_size_sha256": True,
        },
        "candidates": member_rows,
        "protected_output": {
            "root": inventory["root"],
            "result_json_sha256": "0c160914ed28406bd2cfee935726630e1cae974c0c8f5c4bb83f2e9baee2446e",
            "protected_file_count": inventory["protected_count"],
            "protected_bytes": inventory["protected_bytes"],
            "fingerprint_algorithm": "SHA256(canonical JSON array of rows sorted by relative_path, retaining relative_path,size,mode,nlink,device,inode,mtime_ns,sha256)",
            "fingerprint_sha256": inventory["protected_fingerprint_sha256"],
            "inventory_rows": inventory["protected"],
        },
        "capacity_projection_bytes": {
            "free_at_read_only_inventory": current_free,
            "q8_three_root_restore_payloads": q8_restore_bytes,
            "resident_arm_envelope": resident_arm_reserve,
            "combined_q8_restore_plus_resident_envelope": combined_requirement,
            "candidate_reclaim_potential_only_not_performed": candidate_bytes,
            "projected_free_after_candidate_reclaim_only": current_free + candidate_bytes,
            "projected_surplus_over_combined_requirement_after_candidate_reclaim_only": current_free + candidate_bytes - combined_requirement,
        },
        "q8_restore_manifest": {
            "path": "experimental/branching_context/evidence/P5-Q8-ROOT-RESTORE-INPUTS-2026-10-15.json",
            "note": "snapshot_count is 18; protected_metadata_file_count is 56; three payload bytes total 5,491,806,696.",
        },
        "prohibition": "This map is evidence only. No source payload was deleted, no Q8 payload was restored, and no GPU workload was started.",
    }
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "output": str(OUTPUT), "sha256": hashlib.sha256(OUTPUT.read_bytes()).hexdigest(), "candidate_bytes": candidate_bytes, "protected_fingerprint": inventory["protected_fingerprint_sha256"]}, sort_keys=True))


if __name__ == "__main__":
    main()
