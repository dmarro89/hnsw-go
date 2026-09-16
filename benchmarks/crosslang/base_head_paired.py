#!/usr/bin/env python3
import csv, os, random, statistics, struct, subprocess, sys, tempfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
BASE=os.environ.get('BASE_SHA','8198eed606dd4b0497680d98c2283dc3d3ed23ba')
MIN=float(os.environ.get('MIN_SPEEDUP','1.02'))
COUNT=100000
DIM=128
EFS=(64,200)
CYCLES=3
TMP=Path(tempfile.gettempdir())/f'hnsw-paired-{os.getpid()}'
TMP.mkdir(parents=True,exist_ok=True)
VECTORS=TMP/'vectors.f32'
BASE_WT=TMP/'base'

def dataset():
    rng=random.Random(424242)
    with VECTORS.open('wb') as f:
        buf=bytearray()
        for _ in range(COUNT*DIM):
            buf += struct.pack('<f', rng.uniform(-1.0,1.0))
            if len(buf)>=1<<20:
                f.write(buf); buf.clear()
        if buf: f.write(buf)

def ensure_base():
    if not BASE_WT.exists():
        subprocess.check_call(['git','worktree','add','--detach',str(BASE_WT),BASE],cwd=ROOT)

def run(ref,ef,seq):
    cwd=ROOT if ref=='head' else BASE_WT
    out=TMP/f'{ref}-{ef}-{seq}.csv'
    env=os.environ.copy(); env['GOMAXPROCS']='1'
    subprocess.check_call(['go','run','./benchmarks/crosslang/go',str(VECTORS),str(COUNT),str(DIM),str(ef),'1',str(out)],cwd=cwd,env=env)
    with out.open() as f:
        row=next(csv.DictReader(f))
    return float(row['seconds']),float(row['total_alloc_mib']),float(row['live_heap_delta_mib'])

def main():
    dataset(); ensure_base(); failed=False
    try:
        for ef in EFS:
            base=[]; head=[]; seq=0; head_mem=[]
            for _ in range(CYCLES):
                for ref in ('base','head','head','base'):
                    seconds,total,live=run(ref,ef,seq); seq+=1
                    (base if ref=='base' else head).append(seconds)
                    if ref=='head': head_mem.append((total,live))
            ratios=[b/h for b,h in zip(base,head)]
            med=statistics.median(ratios)
            print(f'efC={ef} base_median={statistics.median(base):.3f}s head_median={statistics.median(head):.3f}s paired_median={med:.4f}x ratios={[round(x,4) for x in ratios]}')
            print(f'efC={ef} head_total_alloc_mib={statistics.median(x[0] for x in head_mem):.2f} head_live_heap_delta_mib={statistics.median(x[1] for x in head_mem):.2f}')
            if med < MIN: failed=True
    finally:
        subprocess.call(['git','worktree','remove','--force',str(BASE_WT)],cwd=ROOT)
    return 1 if failed else 0

if __name__=='__main__': sys.exit(main())
