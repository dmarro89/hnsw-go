#!/usr/bin/env python3
import csv, os, statistics, subprocess
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]; ART=ROOT/'.artifacts'; DATA=ART/'base-head-data'
DIM=128; COUNT=100_000; EFS=(64,200); CYCLES=3; MIN_SPEEDUP=1.02

def run(cmd,cwd=ROOT,**kw): return subprocess.run(cmd,cwd=cwd,check=True,text=True,**kw)
def dataset():
 DATA.mkdir(parents=True,exist_ok=True); p=DATA/f'vectors-{COUNT}-{DIM}.f32'
 if not p.exists(): np.random.default_rng(20260913+COUNT).random((COUNT,DIM),dtype=np.float32).tofile(p)
 return p
def build(ref,name):
 d=ART/f'wt-{name}'; run(['git','worktree','add','--detach',str(d),ref])
 try: run(['go','build','-o',str(ART/f'runner-{name}'),'./benchmarks/crosslang/go'],cwd=d)
 finally: run(['git','worktree','remove','--force',str(d)])
def timed(binary,data,efc,label,i):
 out=DATA/f'{label}-{efc}-{i}.csv'; env=os.environ.copy(); env['GOMAXPROCS']='1'
 subprocess.run([str(binary),str(data),str(COUNT),str(DIM),str(efc),'1',str(out)],cwd=ROOT,env=env,check=True)
 with out.open() as f: r=next(csv.DictReader(f))
 return float(r['seconds']),float(r['total_alloc_mib']),float(r['live_heap_delta_mib'])
def main():
 ART.mkdir(exist_ok=True); run(['git','fetch','--no-tags','origin','main']); build('origin/main','base'); build('HEAD','head'); data=dataset(); failed=False
 print('| efC | base median | head median | median paired speedup | head alloc MiB | head heap MiB | gate |'); print('| ---: | ---: | ---: | ---: | ---: | ---: | :--- |')
 for efc in EFS:
  pairs=[]; bt=[]; ht=[]; ha=[]; hh=[]
  for i in range(CYCLES):
   b1,_,_=timed(ART/'runner-base',data,efc,'ba',i); h1,a1,m1=timed(ART/'runner-head',data,efc,'hb',i); h2,a2,m2=timed(ART/'runner-head',data,efc,'hb2',i); b2,_,_=timed(ART/'runner-base',data,efc,'ba2',i)
   bt += [b1,b2]; ht += [h1,h2]; ha += [a1,a2]; hh += [m1,m2]; pairs += [b1/h1,b2/h2]
  s=statistics.median(pairs); ok=s>=MIN_SPEEDUP; failed |= not ok
  print(f'| {efc} | {statistics.median(bt):.3f}s | {statistics.median(ht):.3f}s | {s:.4f}x | {statistics.median(ha):.2f} | {statistics.median(hh):.2f} | {"PASS" if ok else "FAIL"} |')
  print(f'\n- efC={efc} paired ratios: '+', '.join(f'{x:.4f}x' for x in pairs))
 print(f'\nPromotion threshold: >= {MIN_SPEEDUP:.2f}x median paired speedup at both efConstruction values.')
 if failed: raise SystemExit(1)
if __name__=='__main__': main()
