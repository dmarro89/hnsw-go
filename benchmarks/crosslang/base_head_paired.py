#!/usr/bin/env python3
import csv
import os
import statistics
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ART = ROOT / '.artifacts'
DATA = ART / 'base-head-data'
DIM = 128
COUNT = 100_000
EFS = (64, 200)
CYCLES = 3
MIN_SPEEDUP = 1.02


def run(cmd, cwd=ROOT, **kwargs):
    return subprocess.run(cmd, cwd=cwd, check=True, text=True, **kwargs)


def prepare_dataset():
    DATA.mkdir(parents=True, exist_ok=True)
    path = DATA / f'vectors-{COUNT}-{DIM}.f32'
    if not path.exists():
        code = (
            "import numpy as np; "
            f"r=np.random.default_rng({20260913 + COUNT}); "
            f"x=r.random(({COUNT},{DIM}),dtype=np.float32); "
            f"x.tofile(r'{path}')"
        )
        run(['python', '-c', code])
    return path


def build(ref, out):
    run(['git', 'worktree', 'add', '--detach', str(out), ref])
    try:
        run(['go', 'build', '-o', str(ART / f'go-runner-{out.name}'), './benchmarks/crosslang/go'], cwd=out)
    finally:
        run(['git', 'worktree', 'remove', '--force', str(out)])


def timed(binary, dataset, efc, label, iteration):
    out = DATA / f'{label}-{efc}-{iteration}.csv'
    env = os.environ.copy()
    env['GOMAXPROCS'] = '1'
    subprocess.run([str(binary), str(dataset), str(COUNT), str(DIM), str(efc), '1', str(out)], cwd=ROOT, env=env, check=True)
    with out.open() as f:
        row = next(csv.DictReader(f))
    return float(row['seconds']), float(row['total_alloc_mib']), float(row['live_heap_delta_mib'])


def main():
    ART.mkdir(exist_ok=True)
    run(['git', 'fetch', '--no-tags', 'origin', 'main'])
    base = os.environ.get('BASE_SHA', 'origin/main')
    head = os.environ.get('HEAD_SHA', 'HEAD')
    base_dir = ART / 'worktree-base'
    head_dir = ART / 'worktree-head'
    build(base, base_dir)
    build(head, head_dir)
    dataset = prepare_dataset()
    base_bin = ART / 'go-runner-worktree-base'
    head_bin = ART / 'go-runner-worktree-head'

    failed = False
    print('| efC | base median | head median | median paired speedup | head alloc MiB | head heap MiB | gate |')
    print('| ---: | ---: | ---: | ---: | ---: | ---: | :--- |')
    for efc in EFS:
        pairs = []
        base_times, head_times, head_alloc, head_heap = [], [], [], []
        # ABBA per cycle: base, head, head, base. Pair corresponding outer/inner runs.
        for cycle in range(CYCLES):
            b1, _, _ = timed(base_bin, dataset, efc, 'base-a', cycle)
            h1, a1, m1 = timed(head_bin, dataset, efc, 'head-b', cycle)
            h2, a2, m2 = timed(head_bin, dataset, efc, 'head-b2', cycle)
            b2, _, _ = timed(base_bin, dataset, efc, 'base-a2', cycle)
            base_times += [b1, b2]
            head_times += [h1, h2]
            head_alloc += [a1, a2]
            head_heap += [m1, m2]
            pairs += [b1 / h1, b2 / h2]
        speedup = statistics.median(pairs)
        ok = speedup >= MIN_SPEEDUP
        failed |= not ok
        print(f'| {efc} | {statistics.median(base_times):.3f}s | {statistics.median(head_times):.3f}s | {speedup:.4f}x | {statistics.median(head_alloc):.2f} | {statistics.median(head_heap):.2f} | {"PASS" if ok else "FAIL"} |')
        print(f'\n- efC={efc} paired ratios: ' + ', '.join(f'{x:.4f}x' for x in pairs))
    print(f'\nPromotion threshold: >= {MIN_SPEEDUP:.2f}x median paired speedup at both efConstruction values.')
    if failed:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
