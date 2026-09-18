#!/usr/bin/env python3
import csv, os, statistics, subprocess, sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
DATA=ROOT/'.artifacts'/'paired-vectors-100000-128.f32'
EFS=(64,200)
ORDERS=(('base','head','head','base'),('head','base','base','head'))

def run(cmd,cwd=ROOT,env=None):
    return subprocess.run(cmd,cwd=cwd,env=env,check=True,text=True,capture_output=True)

def build(ref,name):
    d=ROOT/'.artifacts'/name
    run(['git','worktree','add','--detach',str(d),ref])
    run(['go','build','-o',str(ROOT/'.artifacts'/f'runner-{name}'),'./benchmarks/crosslang/go'],cwd=d)

def one(name,efc,rep):
    out=ROOT/'.artifacts'/f'{name}-{efc}-{rep}.csv'
    env=os.environ.copy(); env['GOMAXPROCS']='4'
    run([str(ROOT/'.artifacts'/f'runner-{name}'),str(DATA),'100000','128',str(efc),'4',str(out)],env=env)
    with out.open() as f: return next(csv.DictReader(f))

def main():
    (ROOT/'.artifacts').mkdir(exist_ok=True)
    base=os.environ['BASE_SHA']; head=os.environ['HEAD_SHA']
    build(base,'base'); build(head,'head')
    rows=[]
    for efc in EFS:
        rep=0
        for order in ORDERS:
            for name in order:
                rep+=1; r=one(name,efc,rep); r.update(variant=name,rep=str(rep)); rows.append(r)
    print('| efC | metric | base median | head median | speedup / change |')
    print('|---:|:---|---:|---:|---:|')
    failed=False
    for efc in EFS:
        rs=[r for r in rows if int(r['efConstruction'])==efc]
        b=[r for r in rs if r['variant']=='base']; h=[r for r in rs if r['variant']=='head']
        bs=statistics.median(float(r['seconds']) for r in b); hs=statistics.median(float(r['seconds']) for r in h)
        ba=statistics.median(float(r['total_alloc_mib']) for r in b); ha=statistics.median(float(r['total_alloc_mib']) for r in h)
        speed=bs/hs; alloc=(ha/ba-1)*100
        print(f'| {efc} | build seconds | {bs:.3f}s | {hs:.3f}s | {speed:.4f}x |')
        print(f'| {efc} | total alloc | {ba:.2f} MiB | {ha:.2f} MiB | {alloc:+.2f}% |')
        if speed < 1.02: failed=True
    with (ROOT/'.artifacts'/'paired-results.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    if failed:
        print('\nPerformance gate failed: requires >=1.02x median speedup for both efC64 and efC200.',file=sys.stderr); return 1
    return 0
if __name__=='__main__': sys.exit(main())
