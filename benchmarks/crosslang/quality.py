#!/usr/bin/env python3
import csv, subprocess, sys
from pathlib import Path
import numpy as np

DIM=128; M=16; K=10; QUERY_COUNT=100
COUNTS=(10_000,100_000); EFC_VALUES=(64,200); THREADS=(1,4); EFS=(32,64,128)
ROOT=Path(__file__).resolve().parents[2]
DATA=ROOT/'.artifacts'/'crosslang-data'
OUT=ROOT/'.artifacts'/'cross-language-quality.csv'

def dataset(n):
 p=DATA/f'vectors-{n}-{DIM}.f32'; return p,np.memmap(p,dtype=np.float32,mode='r',shape=(n,DIM))

def quality_data(n,x):
 qp=DATA/f'queries-{n}-{DIM}.f32'; tp=DATA/f'truth-{n}-{K}.u32'
 if not qp.exists(): np.random.default_rng(20261913+n).random((QUERY_COUNT,DIM),dtype=np.float32).tofile(qp)
 q=np.memmap(qp,dtype=np.float32,mode='r',shape=(QUERY_COUNT,DIM))
 if not tp.exists():
  gt=np.empty((QUERY_COUNT,K),dtype=np.uint32)
  for i,v in enumerate(q):
   d=np.einsum('ij,ij->i',x-v,x-v,optimize=True)
   ids=np.argpartition(d,K-1)[:K]; gt[i]=ids[np.argsort(d[ids])]
  gt.tofile(tp)
 return qp,tp,q,np.memmap(tp,dtype=np.uint32,mode='r',shape=(QUERY_COUNT,K))

def recall(ids,gt):
 return sum(len(set(map(int,ids[i])).intersection(map(int,gt[i]))) for i in range(len(gt)))/(len(gt)*K)

def hnswlib_quality(x,q,gt,efc,threads):
 import hnswlib
 idx=hnswlib.Index(space='l2',dim=DIM); idx.init_index(max_elements=len(x),M=M,ef_construction=efc,random_seed=100); idx.add_items(x,num_threads=threads)
 out={}
 for ef in EFS: idx.set_ef(ef); ids,_=idx.knn_query(q,k=K,num_threads=threads); out[ef]=recall(ids,gt)
 return out

def usearch_quality(x,q,gt,efc,threads):
 from usearch.index import Index
 idx=Index(ndim=DIM,metric='l2sq',dtype='f32',connectivity=M,expansion_add=efc,expansion_search=64); idx.add(np.arange(len(x),dtype=np.uint64),x,threads=threads)
 out={}
 for ef in EFS:
  idx.expansion_search=ef
  matches=idx.search(q,K,threads=threads); out[ef]=recall(matches.keys,gt)
 return out

def go_quality(path,n,efc,threads,qp,tp):
 cp=subprocess.run(['go','run','./benchmarks/crosslang/quality-go',str(path),str(n),str(DIM),str(efc),str(threads),str(qp),str(tp)],cwd=ROOT,capture_output=True,text=True,check=True)
 return {int(a):float(b) for a,b in (line.split(',') for line in cp.stdout.strip().splitlines())}

def main():
 rows=[]
 for n in COUNTS:
  path,x=dataset(n); qp,tp,q,gt=quality_data(n,x)
  for efc in EFC_VALUES:
   for threads in THREADS:
    results=[('hnsw-go','Go',go_quality(path,n,efc,threads,qp,tp)),('hnswlib','C++',hnswlib_quality(x,q,gt,efc,threads)),('usearch','C++',usearch_quality(x,q,gt,efc,threads))]
    for engine,lang,vals in results:
     for ef in EFS: rows.append({'engine':engine,'language':lang,'vectors':n,'dimensions':DIM,'efConstruction':efc,'threads':threads,'efSearch':ef,'recall_at_10':f'{vals[ef]:.4f}'})
 OUT.parent.mkdir(parents=True,exist_ok=True)
 fields=list(rows[0]);
 with OUT.open('w',newline='') as f: w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)
 print('| Engine | Lang | N | efC | Threads | efSearch | Recall@10 |'); print('| :--- | :--- | ---: | ---: | ---: | ---: | ---: |')
 for r in rows: print(f"| {r['engine']} | {r['language']} | {r['vectors']} | {r['efConstruction']} | {r['threads']} | {r['efSearch']} | {r['recall_at_10']} |")

if __name__=='__main__': main()
