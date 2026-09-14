#!/usr/bin/env python3
import csv
import statistics
import sys

import simd256

N = 100_000
EFC_VALUES = (64, 200)
CYCLES = 3
ORDER = ("scalar", "simd256", "simd256", "scalar")
MIN_MEDIAN_SPEEDUP = 1.10
RAW_OUT = simd256.ROOT / ".artifacts" / "simd256-paired.csv"
SUMMARY_OUT = simd256.ROOT / ".artifacts" / "simd256-paired-summary.csv"


def main():
    if not simd256.RUNNER.exists():
        raise SystemExit(f"missing runner: {simd256.RUNNER}")

    vectors, x = simd256.dataset(N)
    queries, truth = simd256.quality_data(N, x)
    raw_rows = []
    summary_rows = []
    failures = []

    for efc in EFC_VALUES:
        pair_speedups = []
        for cycle in range(1, CYCLES + 1):
            measurements = []
            for position, mode in enumerate(ORDER, start=1):
                result = simd256.run(
                    mode,
                    vectors,
                    N,
                    efc,
                    queries,
                    truth,
                    suffix=f"paired-c{cycle}-p{position}",
                )
                seconds = float(result[0]["seconds"])
                measurements.append(seconds)
                raw_rows.append(
                    {
                        "vectors": N,
                        "dimensions": simd256.DIM,
                        "efConstruction": efc,
                        "cycle": cycle,
                        "position": position,
                        "mode": mode,
                        "seconds": f"{seconds:.6f}",
                        "vectors_per_second": result[0]["vectors_per_second"],
                    }
                )

            # A-B-B-A gives two local comparisons with opposite ordering.
            pair_speedups.append(measurements[0] / measurements[1])
            pair_speedups.append(measurements[3] / measurements[2])

        median_speedup = statistics.median(pair_speedups)
        summary_rows.append(
            {
                "vectors": N,
                "dimensions": simd256.DIM,
                "efConstruction": efc,
                "cycles": CYCLES,
                "pairs": len(pair_speedups),
                "median_speedup": f"{median_speedup:.4f}",
                "min_speedup": f"{min(pair_speedups):.4f}",
                "max_speedup": f"{max(pair_speedups):.4f}",
                "pair_speedups": ";".join(f"{v:.4f}" for v in pair_speedups),
            }
        )
        if median_speedup < MIN_MEDIAN_SPEEDUP:
            failures.append(
                f"100k efC={efc}: median speedup {median_speedup:.3f}x "
                f"is below {MIN_MEDIAN_SPEEDUP:.2f}x"
            )

    with RAW_OUT.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(raw_rows[0]))
        writer.writeheader()
        writer.writerows(raw_rows)
    with SUMMARY_OUT.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)

    print("## Paired ABBA SIMD256 build benchmark")
    print()
    print("| N | efC | cycles | pairs | median speedup | min | max |")
    print("| ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in summary_rows:
        print(
            f"| {row['vectors']} | {row['efConstruction']} | {row['cycles']} | "
            f"{row['pairs']} | {float(row['median_speedup']):.3f}x | "
            f"{float(row['min_speedup']):.3f}x | {float(row['max_speedup']):.3f}x |"
        )
        print(f"Pair ratios efC={row['efConstruction']}: {row['pair_speedups']}")

    print()
    print(
        f"> Promotion gate: median scalar/SIMD256 speedup must be >= "
        f"{MIN_MEDIAN_SPEEDUP:.2f}x for both efConstruction values. "
        "Each of three cycles runs scalar -> SIMD256 -> SIMD256 -> scalar."
    )

    if failures:
        print("\nSIMD256 paired benchmark gate failed:", file=sys.stderr)
        for failure in failures:
            print(f"- {failure}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
