#!/usr/bin/env python3

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import random
import shutil
import statistics
import subprocess
import sys
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description="Noise-aware paired HNSW performance comparison")
    p.add_argument("--base-bin", required=True)
    p.add_argument("--head-bin", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--count", type=int, required=True)
    p.add_argument("--dim", type=int, default=128)
    p.add_argument("--efc", default="64,200", help="comma-separated efConstruction values")
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--pairs", type=int, default=6, help="number of adjacent base/head pairs per efC")
    p.add_argument("--warmups", type=int, default=1, help="warm-up runs per variant per efC")
    p.add_argument("--min-effect", type=float, default=0.02, help="minimum effect for win/regression classification")
    p.add_argument("--bootstrap-samples", type=int, default=5000)
    p.add_argument("--out-dir", default=".artifacts/methodology")
    p.add_argument("--label-base", default="base")
    p.add_argument("--label-head", default="head")
    p.add_argument("--no-pin", action="store_true")
    p.add_argument("--pin-cpus", default="", help="explicit CPU list, e.g. 2 or 2,4,6,8")
    p.add_argument("--require-pinning", action="store_true")
    p.add_argument("--counter-pass", action="store_true", help="run one untimed instrumented build per variant/efC")
    p.add_argument("--require-win", action="store_true", help="exit non-zero unless every efC is a win")
    p.add_argument("--fail-regression", action="store_true", help="exit non-zero if any efC is a regression")
    return p.parse_args()


def geomean(values):
    if not values or any(v <= 0 for v in values):
        raise ValueError("geomean requires positive values")
    return math.exp(sum(math.log(v) for v in values) / len(values))


def bootstrap_ci(values, samples=5000, seed=20260927):
    if not values:
        raise ValueError("bootstrap_ci requires values")
    if len(values) == 1:
        return values[0], values[0]
    rng = random.Random(seed)
    n = len(values)
    draws = []
    for _ in range(samples):
        draws.append(geomean([values[rng.randrange(n)] for _ in range(n)]))
    draws.sort()
    lo = draws[int(0.025 * (samples - 1))]
    hi = draws[int(0.975 * (samples - 1))]
    return lo, hi


def classify(estimate, ci_low, ci_high, min_effect):
    win_threshold = 1.0 + min_effect
    regression_threshold = 1.0 / win_threshold
    if estimate >= win_threshold and ci_low > 1.0:
        return "win"
    if estimate <= regression_threshold and ci_high < 1.0:
        return "regression"
    return "inconclusive"


def cv(values):
    if len(values) < 2:
        return 0.0
    mean = statistics.mean(values)
    return statistics.stdev(values) / mean if mean else 0.0


def parse_cpu_list(spec):
    cpus = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = map(int, part.split("-", 1))
            cpus.extend(range(a, b + 1))
        else:
            cpus.append(int(part))
    return cpus


def choose_distinct_core_cpus(n):
    allowed = set(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else set(range(os.cpu_count() or 1))
    chosen = []
    seen_cores = set()
    try:
        cp = subprocess.run(
            ["lscpu", "-p=CPU,CORE,SOCKET"],
            check=True,
            text=True,
            capture_output=True,
        )
        for line in cp.stdout.splitlines():
            if not line or line.startswith("#"):
                continue
            cpu_s, core_s, socket_s = line.split(",")[:3]
            cpu = int(cpu_s)
            key = (int(socket_s), int(core_s))
            if cpu in allowed and key not in seen_cores:
                chosen.append(cpu)
                seen_cores.add(key)
                if len(chosen) == n:
                    return chosen
    except Exception:
        pass

    for cpu in sorted(allowed):
        if cpu not in chosen:
            chosen.append(cpu)
            if len(chosen) == n:
                break
    return chosen


def command_output(cmd):
    try:
        return subprocess.run(cmd, check=True, text=True, capture_output=True).stdout.strip()
    except Exception as exc:
        return f"unavailable: {exc}"


def read_lscpu_fields():
    fields = {}
    try:
        payload = json.loads(command_output(["lscpu", "-J"]))
        for item in payload.get("lscpu", []):
            field = item.get("field", "").rstrip(":")
            fields[field] = item.get("data", "")
    except Exception:
        pass
    return fields


def hardware_metadata(selected_cpus, pinned):
    fields = read_lscpu_fields()
    fingerprint_fields = {
        key: fields.get(key, "")
        for key in ("Vendor ID", "Model name", "CPU family", "Model", "Stepping")
    }
    raw_fingerprint = json.dumps(fingerprint_fields, sort_keys=True).encode()
    governors = {}
    for cpu in selected_cpus:
        path = Path(f"/sys/devices/system/cpu/cpu{cpu}/cpufreq/scaling_governor")
        if path.exists():
            try:
                governors[str(cpu)] = path.read_text().strip()
            except OSError:
                pass

    return {
        "hardware_fingerprint": hashlib.sha256(raw_fingerprint).hexdigest()[:16],
        "fingerprint_fields": fingerprint_fields,
        "selected_cpus": selected_cpus,
        "pinned": pinned,
        "governors": governors,
        "lscpu": fields,
        "uname": platform.uname()._asdict(),
        "python": platform.python_version(),
        "runner": {
            "name": os.getenv("RUNNER_NAME", ""),
            "os": os.getenv("RUNNER_OS", ""),
            "arch": os.getenv("RUNNER_ARCH", ""),
            "github_run_id": os.getenv("GITHUB_RUN_ID", ""),
            "github_run_attempt": os.getenv("GITHUB_RUN_ATTEMPT", ""),
            "github_sha": os.getenv("GITHUB_SHA", ""),
        },
    }


def run_one(binary, data, count, dim, efc, threads, out, cpus, instrument=False):
    env = os.environ.copy()
    env["GOMAXPROCS"] = str(threads)
    if instrument:
        env["HNSW_BENCH_INSTRUMENT"] = "1"
    else:
        env.pop("HNSW_BENCH_INSTRUMENT", None)

    cmd = [str(binary), str(data), str(count), str(dim), str(efc), str(threads), str(out)]
    if cpus:
        cmd = ["taskset", "-c", ",".join(map(str, cpus))] + cmd
    subprocess.run(cmd, check=True, env=env)
    with out.open() as f:
        return next(csv.DictReader(f))


def numeric(row, key, default=0.0):
    try:
        return float(row.get(key, default))
    except (TypeError, ValueError):
        return default


def integer(row, key, default=0):
    try:
        return int(row.get(key, default))
    except (TypeError, ValueError):
        return default


def main():
    args = parse_args()
    base_bin = Path(args.base_bin).resolve()
    head_bin = Path(args.head_bin).resolve()
    data = Path(args.data).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    efcs = [int(x.strip()) for x in args.efc.split(",") if x.strip()]

    if args.pairs < 2:
        raise SystemExit("--pairs must be >= 2")
    if args.threads < 1:
        raise SystemExit("--threads must be >= 1")

    cpus = []
    if not args.no_pin:
        if args.pin_cpus:
            cpus = parse_cpu_list(args.pin_cpus)
        else:
            cpus = choose_distinct_core_cpus(args.threads)
        if len(cpus) < args.threads:
            cpus = []
    pinned = bool(cpus and shutil.which("taskset"))
    if cpus and not shutil.which("taskset"):
        cpus = []
    if args.require_pinning and not pinned:
        raise SystemExit("CPU pinning required but taskset/CPU affinity is unavailable")

    metadata = hardware_metadata(cpus, pinned)
    metadata.update({
        "count": args.count,
        "dim": args.dim,
        "efConstruction": efcs,
        "threads": args.threads,
        "pairs": args.pairs,
        "warmups": args.warmups,
        "min_effect": args.min_effect,
        "bootstrap_samples": args.bootstrap_samples,
        "base_label": args.label_base,
        "head_label": args.label_head,
    })
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")

    variants = {
        "base": (args.label_base, base_bin),
        "head": (args.label_head, head_bin),
    }
    rows = []
    counter_rows = []
    summaries = []

    for efc in efcs:
        for warmup in range(args.warmups):
            order = ("base", "head") if warmup % 2 == 0 else ("head", "base")
            for variant in order:
                label, binary = variants[variant]
                out = out_dir / f"warmup-{efc}-{warmup}-{variant}.csv"
                run_one(binary, data, args.count, args.dim, efc, args.threads, out, cpus)

        pair_rows = []
        for pair in range(args.pairs):
            order = ("base", "head") if pair % 2 == 0 else ("head", "base")
            current = {}
            for position, variant in enumerate(order):
                label, binary = variants[variant]
                out = out_dir / f"timed-{efc}-pair{pair}-{position}-{variant}.csv"
                row = run_one(binary, data, args.count, args.dim, efc, args.threads, out, cpus)
                row.update({
                    "variant": variant,
                    "label": label,
                    "pair": str(pair),
                    "position": str(position),
                    "order": "->".join(order),
                    "hardware_fingerprint": metadata["hardware_fingerprint"],
                })
                rows.append(row)
                current[variant] = row
            pair_rows.append(current)

        base_seconds = [numeric(p["base"], "seconds") for p in pair_rows]
        head_seconds = [numeric(p["head"], "seconds") for p in pair_rows]
        ratios = [numeric(p["base"], "seconds") / numeric(p["head"], "seconds") for p in pair_rows]
        estimate = geomean(ratios)
        ci_low, ci_high = bootstrap_ci(ratios, args.bootstrap_samples, seed=20260927 + efc)
        verdict = classify(estimate, ci_low, ci_high, args.min_effect)

        base_alloc = [numeric(p["base"], "total_alloc_mib") for p in pair_rows]
        head_alloc = [numeric(p["head"], "total_alloc_mib") for p in pair_rows]
        base_cpu_util = [numeric(p["base"], "cpu_utilization") for p in pair_rows]
        head_cpu_util = [numeric(p["head"], "cpu_utilization") for p in pair_rows]
        base_checksums = {p["base"].get("graph_checksum", "") for p in pair_rows}
        head_checksums = {p["head"].get("graph_checksum", "") for p in pair_rows}
        base_deterministic = len(base_checksums) == 1
        head_deterministic = len(head_checksums) == 1
        if not base_deterministic or not head_deterministic:
            verdict = "invalid:nondeterministic-graph"

        alloc_change = 0.0
        base_alloc_median = statistics.median(base_alloc)
        head_alloc_median = statistics.median(head_alloc)
        if base_alloc_median:
            alloc_change = (head_alloc_median / base_alloc_median - 1.0) * 100.0

        summaries.append({
            "efc": efc,
            "base_median": statistics.median(base_seconds),
            "head_median": statistics.median(head_seconds),
            "speedup": estimate,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "base_cv": cv(base_seconds),
            "head_cv": cv(head_seconds),
            "alloc_change": alloc_change,
            "base_cpu_util": statistics.median(base_cpu_util),
            "head_cpu_util": statistics.median(head_cpu_util),
            "verdict": verdict,
            "base_deterministic": base_deterministic,
            "head_deterministic": head_deterministic,
            "ratios": ratios,
        })

        if args.counter_pass:
            for variant in ("base", "head"):
                label, binary = variants[variant]
                out = out_dir / f"counter-{efc}-{variant}.csv"
                row = run_one(binary, data, args.count, args.dim, efc, args.threads, out, cpus, instrument=True)
                row.update({
                    "variant": variant,
                    "label": label,
                    "hardware_fingerprint": metadata["hardware_fingerprint"],
                })
                counter_rows.append(row)

    raw_fields = []
    for row in rows:
        for key in row:
            if key not in raw_fields:
                raw_fields.append(key)
    with (out_dir / "raw.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=raw_fields)
        w.writeheader()
        w.writerows(rows)

    if counter_rows:
        counter_fields = []
        for row in counter_rows:
            for key in row:
                if key not in counter_fields:
                    counter_fields.append(key)
        with (out_dir / "counters.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=counter_fields)
            w.writeheader()
            w.writerows(counter_rows)

    lines = [
        "## Benchmark methodology v2",
        "",
        f"Hardware fingerprint: `{metadata['hardware_fingerprint']}`",
        f"CPU: `{metadata['fingerprint_fields'].get('Model name', '')}`",
        f"Pinned CPUs: `{','.join(map(str, cpus)) if pinned else 'no'}`",
        f"Pairs per efC: `{args.pairs}`; warmups per variant: `{args.warmups}`",
        "",
        "| efC | Base median | Head median | Paired speedup | 95% bootstrap CI | Base CV | Head CV | Alloc change | CPU util base/head | Verdict |",
        "| ---: | ---: | ---: | ---: | :--- | ---: | ---: | ---: | :--- | :--- |",
    ]
    for s in summaries:
        lines.append(
            f"| {s['efc']} | {s['base_median']:.3f}s | {s['head_median']:.3f}s | "
            f"**{s['speedup']:.4f}x** | [{s['ci_low']:.4f}, {s['ci_high']:.4f}] | "
            f"{s['base_cv']*100:.2f}% | {s['head_cv']*100:.2f}% | {s['alloc_change']:+.2f}% | "
            f"{s['base_cpu_util']:.3f}/{s['head_cpu_util']:.3f} | **{s['verdict']}** |"
        )
        lines.append("")
        lines.append(f"- efC{s['efc']} paired ratios: " + ", ".join(f"{r:.4f}x" for r in s["ratios"]))
        lines.append(
            f"- graph deterministic within variant: base={s['base_deterministic']}, head={s['head_deterministic']}"
        )

    if counter_rows:
        lines += [
            "",
            "### Deterministic counter pass",
            "",
            "| efC | Variant | Distance evaluations | Directed edges | Max level | Graph checksum |",
            "| ---: | :--- | ---: | ---: | ---: | ---: |",
        ]
        for row in counter_rows:
            lines.append(
                f"| {row['efConstruction']} | {row['label']} | {integer(row, 'distance_evaluations')} | "
                f"{integer(row, 'directed_edges')} | {integer(row, 'max_observed_level')} | {row.get('graph_checksum', '')} |"
            )

    lines += [
        "",
        "Classification rule:",
        f"- win: paired geometric-mean speedup >= {1+args.min_effect:.3f}x and 95% CI lower bound > 1.0",
        f"- regression: paired geometric-mean speedup <= {1/(1+args.min_effect):.3f}x and 95% CI upper bound < 1.0",
        "- otherwise: inconclusive (do not treat as a rejection)",
        "- results from different hardware fingerprints must be analyzed as separate strata",
    ]

    summary_text = "\n".join(lines) + "\n"
    (out_dir / "summary.md").write_text(summary_text)
    print(summary_text)

    verdicts = [s["verdict"] for s in summaries]
    if args.require_win and any(v != "win" for v in verdicts):
        return 2
    if args.fail_regression and any(v == "regression" or v.startswith("invalid:") for v in verdicts):
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
