#!/usr/bin/env python3
"""Bind live resident state inventory to the receiver-verified archive manifest."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path("/home/dflanag3/spatial-artifacts/p4-q4-three-root-return-20261015-01/resident-reclaim-readonly-20261015-01")
INVENTORY = ROOT / "inventory-v2.json"
MANIFEST = ROOT / "MANIFEST.json"
RECEIPT = ROOT / "receiver-receipt.json"
OUTPUT = ROOT / "archive-match-proof-v2.json"
EXPECTED_ARCHIVE = "/srv/data/git/private_context_cow.evidence/20261015-p4-q4-resident-return/p4-q4-resident-return-20261015.tar"
EXPECTED_ARCHIVE_SHA256 = "afea6d6117cc9d7df8c3fbc70225d01ceb3dc61a10224192cf5dd3e694b43a60"
EXPECTED_ARCHIVE_BYTES = 17814958080
EXPECTED_RECEIPT_SHA256 = "a35dc82d155dc3598f87ee13f4b333b055ac0c189ce090b3e05047b18e05a235"
EXPECTED_MANIFEST_SHA256 = "31787d8bc498ee35a4aeeab3756e1712d473ec99fe17227b7657722b005c7111"
EXPECTED_LISTING_SHA256 = "77dfa83047605f111c924243d1e7dba0ac5a9b93e04512f5bc0c332258693ced"


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> None:
    inventory_bytes = INVENTORY.read_bytes()
    manifest_bytes = MANIFEST.read_bytes()
    receipt_bytes = RECEIPT.read_bytes()
    inventory = json.loads(inventory_bytes)
    manifest = json.loads(manifest_bytes)
    receipt = json.loads(receipt_bytes)
    raw_manifest_sha = sha(manifest_bytes)
    receipt_sha = sha(receipt_bytes)
    canonical_manifest_sha = sha(json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    listed = manifest.get("files")
    if not isinstance(listed, list) or not listed:
        raise SystemExit("archive manifest has no file rows")
    names = [row.get("path") for row in listed]
    if len(names) != len(set(names)):
        raise SystemExit("archive manifest contains duplicate paths")
    listing_sha = sha(json.dumps(sorted(listed, key=lambda row: row["path"]), sort_keys=True, separators=(",", ":")).encode("utf-8"))
    pins = (receipt.get("target"), receipt.get("archive_sha256"), receipt.get("archive_bytes"), receipt.get("status"))
    if pins != (EXPECTED_ARCHIVE, EXPECTED_ARCHIVE_SHA256, EXPECTED_ARCHIVE_BYTES, "PASS"):
        raise SystemExit(f"receiver receipt archive pins failed: {pins}")
    if receipt_sha != EXPECTED_RECEIPT_SHA256:
        raise SystemExit("receiver receipt raw-byte SHA does not match archived evidence")
    if canonical_manifest_sha != EXPECTED_MANIFEST_SHA256 or receipt.get("manifest_sha256") != canonical_manifest_sha:
        raise SystemExit("canonical manifest digest does not match receiver receipt")
    if listing_sha != EXPECTED_LISTING_SHA256 or receipt.get("member_listing_sha256") != listing_sha:
        raise SystemExit("member listing digest does not match receiver receipt")
    if inventory.get("status") != "READ_ONLY_INVENTORY_PASS" or inventory.get("candidate_count") != 9:
        raise SystemExit("live inventory is not a complete successful read-only scan")
    prefix = "resident-arm/output/store/checkpoints/"
    archive_rows = [row for row in listed if row["path"].startswith(prefix) and row["path"].endswith(".state")]
    source_by_name = {Path(row["path"]).name: row for row in inventory["candidates"]}
    if len(archive_rows) != 9 or len(source_by_name) != 9:
        raise SystemExit("unexpected resident .state member/source count")
    matched = []
    for archive_row in archive_rows:
        name = archive_row["path"].split("/")[-1]
        source = source_by_name.get(name)
        if source is None:
            raise SystemExit(f"archive member has no live source candidate: {name}")
        if (source.get("size"), source.get("sha256")) != (archive_row.get("size"), archive_row.get("sha256")):
            raise SystemExit(f"source/archive mismatch: {name}")
        if not source.get("stable_during_hash") or source.get("mode") != "0o600" or source.get("nlink") != 1:
            raise SystemExit(f"live candidate invariant failed: {name}")
        matched.append({"archive_path": archive_row["path"], "source_path": source["path"], "size": source["size"], "sha256": source["sha256"]})
    protected = inventory.get("protected", [])
    protected_sha = sha(json.dumps(protected, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    result = {
        "status": "PASS_EXACT_ARCHIVE_MANIFEST_SOURCE_MATCH",
        "execution": {"exit_code": 0, "host": "llm-60", "gpu_used": False},
        "inputs": {
            "inventory_sha256": sha(inventory_bytes),
            "manifest_path": str(MANIFEST),
            "manifest_raw_bytes": len(manifest_bytes),
            "manifest_raw_sha256": raw_manifest_sha,
            "manifest_canonical_sha256": canonical_manifest_sha,
            "receiver_receipt_sha256": receipt_sha,
            "archive_sha256": receipt["archive_sha256"],
            "member_listing_sha256": listing_sha,
        },
        "result": {
            "archive_manifest_exact_candidate_match": True,
            "matched_candidate_count": len(matched),
            "matched_candidate_bytes": sum(row["size"] for row in matched),
            "matched_rows": matched,
            "protected_metadata_file_count": len(protected),
            "protected_metadata_fingerprint_sha256": protected_sha,
            "no_restore_or_unlink_performed": True,
        },
    }
    OUTPUT.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "output": str(OUTPUT), "matched": len(matched), "bytes": result["result"]["matched_candidate_bytes"], "raw_manifest_sha256": raw_manifest_sha, "canonical_manifest_sha256": canonical_manifest_sha}, sort_keys=True))


if __name__ == "__main__":
    main()
