#!/usr/bin/env python3
import csv
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

DIM = 128
K = 10
QUERY_COUNT = 100
COUNTS = (10_000, 100_000)
EFC_VALUES = (64, 200)
EFS = (32, 64, 128)
MAX_RECALL_REGRESSION = 0.005
DIAGNOSTIC_N = 100_000
DIAGNOSTIC_EFC = 64

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / ".artifacts" / "crosslang-data"
OUT = ROOT / ".artifacts" / "simd256-crosslang.csv"
DIAG_OUT = ROOT / ".artifacts" / "simd256-diagnostics.csv"
RUNNER = ROOT / ".artifacts" / "simd256-go"


def dataset(n):
    DATA.mkdir(parents=True, exist_ok=True)
    path = DATA / f"vectors-{n}-{DIM}.f32"
    if not path.exists():
        rng = np.random.default_rng(20260913 + n)
        x = rng.random((n, DIM), dtype=np.float32)
        x.tofile(path)
    return path, np.memmap(path, dtype=np.float32, mode="r", shape=(n, DIM))


def quality_data(n, x):
    query_path = DATA / f"queries-{n}-{DIM}.f32"
    truth_path = DATA / f"truth-{n}-{K}.u32"
    if not query_path.exists():
        np.random.default_rng(20261913 + n).random(
            (QUERY_COUNT, DIM), dtype=np.float32
        ).tofile(query_path)
    queries = np.memmap(
        query_path, dtype=np.float32, mode="r", shape=(QUERY_COUNT, DIM)
    )
    if not truth_path.exists():
        truth = np.empty((QUERY_COUNT, K), dtype=np.uint32)
        for i, query in enumerate(queries):
            distances = np.einsum("ij,ij->i", x - query, x - query, optimize=True)
            ids = np.argpartition(distances, K - 1)[:K]
            truth[i] = ids[np.argsort(distances[ids])]
        truth.tofile(truth_path)
    return query_path, truth_path


def run(mode, vectors, n, efc, queries, truth, suffix="matrix", extra_env=None):
    output = DATA / f"simd256-{suffix}-{mode}-{n}-{efc}.csv"
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)
    subprocess.run(
        [
            str(RUNNER),
            mode,
            str(vectors),
            str(n),
            str(DIM),
            str(efc),
            str(queries),
            str(truth),
            str(output),
        ],
        cwd=ROOT,
        env=env,
        check=True,
    )
    with output.open() as f:
        return list(csv.DictReader(f))


def write_matrix(rows):
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with OUT.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run_diagnostics():
    vectors, x = dataset(DIAGNOSTIC_N)
    queries, truth = quality_data(DIAGNOSTIC_N, x)

    counted = {}
    for mode in ("scalar", "simd256"):
        result = run(
            mode,
            vectors,
            DIAGNOSTIC_N,
            DIAGNOSTIC_EFC,
            queries,
            truth,
            suffix="count",
            extra_env={"HNSW_DISTANCE_COUNT": "1"},
        )
        counted[mode] = int(result[0]["distance_calls"])

    profile_paths = {}
    for mode in ("scalar", "simd256"):
        profile_path = ROOT / ".artifacts" / f"{mode}-100k-ef64.cpu.pprof"
        run(
            mode,
            vectors,
            DIAGNOSTIC_N,
            DIAGNOSTIC_EFC,
            queries,
            truth,
            suffix="profile",
            extra_env={"HNSW_CPU_PROFILE": str(profile_path)},
        )
        profile_paths[mode] = profile_path

    scalar_calls = counted["scalar"]
    simd_calls = counted["simd256"]
    delta = simd_calls - scalar_calls
    ratio = simd_calls / scalar_calls if scalar_calls else float("nan")

    diag_rows = [
        {
            "mode": mode,
            "vectors": DIAGNOSTIC_N,
            "dimensions": DIM,
            "efConstruction": DIAGNOSTIC_EFC,
            "distance_calls": counted[mode],
            "distance_calls_per_vector": f"{counted[mode] / DIAGNOSTIC_N:.3f}",
            "profile": str(profile_paths[mode].relative_to(ROOT)),
        }
        for mode in ("scalar", "simd256")
    ]
    with DIAG_OUT.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(diag_rows[0]))
        writer.writeheader()
        writer.writerows(diag_rows)

    return scalar_calls, simd_calls, delta, ratio


def main():
    if not RUNNER.exists():
        raise SystemExit(f"missing runner: {RUNNER}")

    rows = []
    failures = []

    for n in COUNTS:
        vectors, x = dataset(n)
        queries, truth = quality_data(n, x)
        for efc in EFC_VALUES:
            scalar = run("scalar", vectors, n, efc, queries, truth)
            simd = run("simd256", vectors, n, efc, queries, truth)
            by_ef_scalar = {int(r["efSearch"]): r for r in scalar}
            by_ef_simd = {int(r["efSearch"]): r for r in simd}

            scalar_seconds = float(scalar[0]["seconds"])
            simd_seconds = float(simd[0]["seconds"])
            speedup = scalar_seconds / simd_seconds

            for ef in EFS:
                s = by_ef_scalar[ef]
                v = by_ef_simd[ef]
                scalar_recall = float(s["recall_at_10"])
                simd_recall = float(v["recall_at_10"])
                delta = simd_recall - scalar_recall
                row = {
                    "vectors": n,
                    "dimensions": DIM,
                    "efConstruction": efc,
                    "efSearch": ef,
                    "scalar_seconds": f"{scalar_seconds:.6f}",
                    "simd256_seconds": f"{simd_seconds:.6f}",
                    "build_speedup": f"{speedup:.3f}",
                    "scalar_vectors_per_second": s["vectors_per_second"],
                    "simd256_vectors_per_second": v["vectors_per_second"],
                    "scalar_total_alloc_mib": s["total_alloc_mib"],
                    "simd256_total_alloc_mib": v["total_alloc_mib"],
                    "scalar_recall_at_10": f"{scalar_recall:.4f}",
                    "simd256_recall_at_10": f"{simd_recall:.4f}",
                    "recall_delta": f"{delta:+.4f}",
                }
                rows.append(row)
                if delta < -MAX_RECALL_REGRESSION:
                    failures.append(
                        f"N={n} efC={efc} efSearch={ef}: recall delta {delta:+.4f} "
                        f"is below {-MAX_RECALL_REGRESSION:.4f}"
                    )

    write_matrix(rows)
    scalar_calls, simd_calls, call_delta, call_ratio = run_diagnostics()

    print("## SIMD256 serial build impact")
    print()
    print("| N | efC | Scalar build | SIMD256 build | Speedup | Scalar vec/s | SIMD256 vec/s |")
    print("| ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    seen = set()
    for row in rows:
        key = (row["vectors"], row["efConstruction"])
        if key in seen:
            continue
        seen.add(key)
        print(
            f"| {row['vectors']} | {row['efConstruction']} | "
            f"{float(row['scalar_seconds']):.3f}s | {float(row['simd256_seconds']):.3f}s | "
            f"{float(row['build_speedup']):.2f}x | {row['scalar_vectors_per_second']} | "
            f"{row['simd256_vectors_per_second']} |"
        )

    print()
    print("## DistanceFunc call-count diagnostic")
    print()
    print("| Scenario | Scalar calls | SIMD256 calls | Delta | SIMD/scalar |")
    print("| :--- | ---: | ---: | ---: | ---: |")
    print(
        f"| {DIAGNOSTIC_N} x {DIM}d, efC={DIAGNOSTIC_EFC} | {scalar_calls} | "
        f"{simd_calls} | {call_delta:+d} | {call_ratio:.4f}x |"
    )
    print()
    print(
        "> Counts come from separate instrumented builds; the main timing matrix is uninstrumented. "
        "CPU profiles are separate uninstrumented 100k/efC64 builds."
    )

    print()
    print("## Common-ground-truth Recall@10: scalar vs SIMD256")
    print()
    print("| N | efC | efSearch | Scalar | SIMD256 | Delta |")
    print("| ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in rows:
        print(
            f"| {row['vectors']} | {row['efConstruction']} | {row['efSearch']} | "
            f"{row['scalar_recall_at_10']} | {row['simd256_recall_at_10']} | "
            f"{row['recall_delta']} |"
        )

    print()
    print(
        f"> Quality gate: SIMD256 may not regress Recall@10 by more than "
        f"{MAX_RECALL_REGRESSION:.3f} in any deterministic matrix cell."
    )

    if failures:
        print("\nSIMD256 quality gate failed:", file=sys.stderr)
        for failure in failures:
            print(f"- {failure}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
