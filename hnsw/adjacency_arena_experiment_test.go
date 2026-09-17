package hnsw

import (
 "math/rand/v2"
 "testing"
 "dmarro89.github.com/hnsw-go/structs"
)

func searchLayerArenaAdj(query []float32, entries []int, ef, level int, dst []int, vectors []float32, dim int, h *HNSW, adj *adjacencyReadArena) []int {
 h.visitStamp++; stamp:=h.visitStamp; pool:=h.ensureHeapPool(); candidates:=pool.GetMinHeap(); defer pool.PutMinHeap(candidates); nearest:=pool.GetMaxHeap(); defer pool.PutMaxHeap(nearest)
 for _,id:=range entries { if id<0||id>=len(h.Nodes)||h.markVisitedArena(id,stamp){continue}; d:=h.DistanceFunc(query,arenaVector(vectors,dim,id)); candidates.Push(structs.NewNodeHeap(d,id)); nearest.Push(structs.NewNodeHeap(d,id)); if nearest.Len()>ef{nearest.Pop()} }
 for candidates.Len()>0 { current:=candidates.Pop(); if nearest.Len()>=ef&&current.Dist>nearest.Peek().Dist{break}; for _,id:=range adj.neighbors(current.Id,level){ if id<0||id>=len(h.Nodes)||h.markVisitedArena(id,stamp){continue}; d:=h.DistanceFunc(query,arenaVector(vectors,dim,id)); if nearest.Len()>=ef&&d>=nearest.Peek().Dist{continue}; candidates.Push(structs.NewNodeHeap(d,id)); nearest.Push(structs.NewNodeHeap(d,id)); if nearest.Len()>ef{nearest.Pop()} } }
 n:=nearest.Len(); if cap(dst)<n{dst=make([]int,n)}else{dst=dst[:n]}; for i:=n-1;i>=0;i--{dst[i]=nearest.Pop().Id}; return dst
}

var adjacencySearchSink int
func BenchmarkAdjacencyArenaRealSearch(b *testing.B){
 const n,dim=20000,128
 r:=rand.New(rand.NewPCG(61,62)); vectors:=make([][]float32,n); flat:=make([]float32,n*dim); for i:=range n{v:=flat[i*dim:(i+1)*dim];for j:=range dim{v[j]=r.Float32()};vectors[i]=v}
 h,err:=NewHNSW(Config{M:16,Mmax:16,Mmax0:32,EfConstruction:64,MaxLevel:16,DistanceFunc:EuclideanDistance});if err!=nil{b.Fatal(err)};lr:=rand.New(rand.NewPCG(63,64));h.RandFunc=lr.Float64;h.InsertBatch(vectors);adj:=buildAdjacencyReadArena(h);queries:=vectors[n-256:]
 for _,tc:=range []struct{name string; arena bool}{{"current",false},{"adjacency-arena",true}}{tc:=tc;b.Run(tc.name,func(b *testing.B){b.ReportAllocs();buf:=make([]int,0,64);sum:=0;b.ResetTimer();for i:=0;i<b.N;i++{q:=queries[i%len(queries)];entry:=[]int{h.EntryPoint.ID};var out []int;if tc.arena{out=searchLayerArenaAdj(q,entry,64,0,buf,flat,dim,h,&adj)}else{out=h.searchLayerArena(q,entry,64,0,buf,flat,dim)};sum+=len(out);buf=out[:0]};adjacencySearchSink=sum})}
}
