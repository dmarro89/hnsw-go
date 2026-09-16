#!/usr/bin/env python3
import json, os, statistics, subprocess, sys, tempfile, time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
BASE=os.environ.get('BASE_SHA','8198eed606dd4b0497680d98c2283dc3d3ed23ba')
HEAD=os.environ.get('HEAD_SHA','HEAD')
MIN=float(os.environ.get('MIN_SPEEDUP','1.02'))
EFS=(64,200)
CYCLES=3

def run(ref, ef):
    env=os.environ.copy(); env['GOMAXPROCS']='1'; env['HNSW_EF_CONSTRUCTION']=str(ef)
    cmd=['go','run','./benchmarks/crosslang/go']
    if ref=='HEAD':
        cwd=ROOT
    else:
        wt=Path(tempfile.gettempdir())/f'hnsw-base-{os.getpid()}'
        if not wt.exists(): subprocess.check_call(['git','worktree','add','--detach',str(wt),BASE],cwd=ROOT)
        cwd=wt
    t=time.perf_counter(); out=subprocess.check_output(cmd,cwd=cwd,env=env,text=True); elapsed=time.perf_counter()-t
    # Prefer runner build_seconds when present; wall time is a fallback.
    try:
        data=json.loads(out.strip().splitlines()[-1]); return float(data.get('build_seconds',elapsed))
    except Exception: return elapsed

def main():
    failed=False
    for ef in EFS:
        pairs=[]; base=[]; head=[]
        for _ in range(CYCLES):
            for ref in ('base','HEAD','HEAD','base'):
                v=run(ref,ef)
                (head if ref=='HEAD' else base).append(v)
        for b,h in zip(base,head): pairs.append(b/h)
        med=statistics.median(pairs)
        print(f'efC={ef} base_median={statistics.median(base):.3f}s head_median={statistics.median(head):.3f}s paired_median={med:.4f}x ratios={pairs}')
        if med < MIN: failed=True
    return 1 if failed else 0

if __name__=='__main__': sys.exit(main())
