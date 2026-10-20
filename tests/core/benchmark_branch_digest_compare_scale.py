"""Generated-memory benchmark for the unchanged production page-compare kernel."""
import argparse
import hashlib
import json
import random
import subprocess
import time
from pathlib import Path


PAGE_BYTES = 65536
TILE_BYTES = 4096


def generated_fixture(mib):
    total = mib * 1024 * 1024
    if total % PAGE_BYTES:
        raise ValueError("MiB fixture must contain whole 64 KiB pages")
    data = bytearray(total)
    rng = random.Random(20261015)
    for offset in range(0, total, PAGE_BYTES):
        data[offset:offset + PAGE_BYTES] = rng.randbytes(PAGE_BYTES)
    seals = bytearray()
    view = memoryview(data)
    for offset in range(0, total, TILE_BYTES):
        seals.extend(hashlib.sha256(view[offset:offset + TILE_BYTES]).digest())
    return data, seals


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--mib", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.mib not in (256, 2560):
        parser.error("only the predeclared 256 MiB and 2560 MiB inputs are supported")
    if args.output.exists():
        raise FileExistsError(args.output)

    report = {
        "status": "RUNNING",
        "scope": "isolated production hash_compare_pages CUDA kernel; no model/inference; 1 warmup + 3 measured full-byte comparisons",
        "fixture_seed": 20261015,
        "fixture_bytes": args.mib * 1024 * 1024,
        "binary_sha256": hashlib.sha256(args.binary.read_bytes()).hexdigest(),
        "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "repetition": None,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        data, expected_seals = generated_fixture(args.mib)
        report["fixture_sha256"] = hashlib.sha256(data).hexdigest()
        report["expected_seals_sha256"] = hashlib.sha256(expected_seals).hexdigest()
        report["expected_tiles"] = len(expected_seals) // 32

        started = time.monotonic()
        process = subprocess.Popen(
            [str(args.binary.resolve()), str(len(data))],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        try:
            # Write in bounded chunks; never concatenate a second multi-GiB copy.
            for payload in (memoryview(data), memoryview(expected_seals)):
                for offset in range(0, len(payload), 8 * 1024 * 1024):
                    process.stdin.write(payload[offset:offset + 8 * 1024 * 1024])
            process.stdin.close()
            stdout = process.stdout.read()
            stderr = process.stderr.read()
            code = process.wait()
        except BaseException:
            process.kill()
            process.wait()
            raise
        if code != 0:
            raise RuntimeError(f"CUDA compare driver exited {code}: {stderr.decode('utf-8', 'replace')[-2000:]}")
        measured = json.loads(stdout)
        expected_bytes = len(data) * 3
        if (measured.get("status") != "PASS"
                or measured.get("bytes_per_compare") != len(data)
                or measured.get("tiles_per_compare") != len(expected_seals) // 32
                or measured.get("fresh_hash_bytes") != expected_bytes
                or measured.get("kernel_launches") != 3
                or measured.get("warmup_compares") != 1
                or measured.get("measured_compares") != 3
                or measured.get("mismatch") is not None):
            raise AssertionError("GPU comparison did not match every independent CPU SHA-256 tile")
        measured["process_wall_seconds"] = time.monotonic() - started
        report["repetition"] = measured
        report["status"] = "PASS"
    except BaseException as error:
        report.update(status="FAIL", error=repr(error))
        raise
    finally:
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
