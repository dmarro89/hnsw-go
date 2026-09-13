#!/usr/bin/env python3
import csv, os, resource, subprocess, sys, time
from pathlib import Path
import numpy as np

DIM=128
M=16
COUNTS=(10_000,100_000)
EFS=(64,200)
THREADS=(1,4)
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'.artifacts'/'cross-language.csv'
DATA=ROOT/'.artifacts'/'crosslang-data'
DATA.mkdir(parents=True,exist_ok=True)


def dataset(n):
    p=DATA/f'vectors-{n}-{DIM}.f32'
    if not p.exists():
        rng=np.random.default_rng(20260913+n)
        x=rng.random((n,DIM),dtype=np.float32)
        x.tofile(p)
    return p,np.memmap(p,dtype=np.float32,mode='r',shape=(n,DIM))


def rss_mib():
    # Linux ru_maxrss is KiB. It is process-lifetime peak RSS, so each external
    # engine scenario runs in a fresh subprocess and reports its own value.
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024.0


def run_hnswlib(path,n,efc,threads):
    import hnswlib
    x=np.memmap(path,dtype=np.float32,mode='r',shape=(n,DIM))
    idx=hnswlib.Index(space='l2',dim=DIM)
    idx.init_index(max_elements=n,M=M,ef_construction=efc,random_seed=100)
    t=time.perf_counter(); idx.add_items(x,num_threads=threads); sec=time.perf_counter()-t
    return sec,rss_mib()


def run_usearch(path,n,efc,threads):
    from usearch.index import Index
    x=np.memmap(path,dtype=np.float32,mode='r',shape=(n,DIM))
    idx=Index(ndim=DIM,metric='l2sq',dtype='f32',connectivity=M,expansion_add=efc,expansion_search=64)
    keys=np.arange(n,dtype=np.uint64)
    t=time.perf_counter(); idx.add(keys,x,threads=threads); sec=time.perf_counter()-t
    return sec,rss_mib()


def child():
    engine,path,n,efc,threads=sys.argv[2],Path(sys.argv[3]),int(sys.argv[4]),int(sys.argv[5]),int(sys.argv[6])
    fn={'hnswlib':run_hnswlib,'usearch':run_usearch}[engine]
    sec,rss=fn(path,n,efc,threads)
    print(f'{sec},{rss}')


def main():
    rows=[]
    for n in COUNTS:
        path,_=dataset(n)
        for efc in EFS:
            for threads in THREADS:
                goout=DATA/f'go-{n}-{efc}-{threads}.csv'
                subprocess.run(['go','run','./benchmarks/crosslang/go',str(path),str(n),str(DIM),str(efc),str(threads),str(goout)],cwd=ROOT,check=True)
                with goout.open() as f: rows.extend(csv.DictReader(f))
                for engine,language in [('hnswlib','C++'),('usearch','C++')]:
                    cp=subprocess.run([sys.executable,__file__,'child',engine,str(path),str(n),str(efc),str(threads)],capture_output=True,text=True,check=True)
                    sec,rss=map(float,cp.stdout.strip().split(','))
                    rows.append({'engine':engine,'language':language,'vectors':n,'dimensions':DIM,'efConstruction':efc,'threads':threads,'mode':'serial' if threads==1 else 'parallel','seconds':f'{sec:.6f}','vectors_per_second':f'{n/sec:.0f}','total_alloc_mib':'n/a','live_heap_delta_mib':f'{rss:.2f}'})
    OUT.parent.mkdir(parents=True,exist_ok=True)
    fields=['engine','language','vectors','dimensions','efConstruction','threads','mode','seconds','vectors_per_second','total_alloc_mib','live_heap_delta_mib']
    with OUT.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)
    print_table(rows)


def print_table(rows):
    print('| Engine | Lang | N | efC | Threads | Build | vectors/s | Memory* |')
    print('| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: |')
    for r in rows:
        print(f"| {r['engine']} | {r['language']} | {r['vectors']} | {r['efConstruction']} | {r['threads']} | {float(r['seconds']):.3f}s | {r['vectors_per_second']} | {r['live_heap_delta_mib']} MiB |")
    print('\n* hnsw-go reports live Go heap delta; external engines report process peak RSS. These are intentionally labelled separately in the CSV and must not be treated as identical memory metrics.')

if __name__=='__main__':
    if len(sys.argv)>1 and sys.argv[1]=='child': child()
    else: main()
