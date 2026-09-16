package hnsw

import (
 "fmt"
 "math/rand/v2"
 "testing"

 "dmarro89.github.com/hnsw-go/structs"
)

type expansionOutcomeStats struct { distance, accepted, rejected uint64; byLevel map[int][3]uint64 }

func (h *HNSW) searchLayerArenaProfile(query []float32, entries []int, ef, level int, dst []int, arena []float32, dim int, s *expansionOutcomeStats) []int {
 h.visitStamp++; stamp:=h.visitStamp; pool:=h.ensureHeapPool(); candidates:=pool.GetMinHeap(); defer pool.PutMinHeap(candidates); nearest:=pool.GetMaxHeap(); defer pool.PutMaxHeap(nearest)
 for _,id:=range entries { if id<0||id>=len(h.Nodes)||h.markVisitedArena(id,stamp){continue}; d:=h.DistanceFunc(query,arenaVector(arena,dim,id)); candidates.Push(structs.NewNodeHeap(d,id)); nearest.Push(structs.NewNodeHeap(d,id)); if nearest.Len()>ef{nearest.Pop()} }
 if candidates.Len()==0{return dst[:0]}
 for candidates.Len()>0 { current:=candidates.Pop(); if nearest.Len()>=ef&&current.Dist>nearest.Peek().Dist{break}; node:=h.Nodes[current.Id]; if node==nil||level>=len(node.Neighbors){continue}; for _,id:=range node.Neighbors[level]{ if id<0||id>=len(h.Nodes)||h.markVisitedArena(id,stamp){continue}; d:=h.DistanceFunc(query,arenaVector(arena,dim,id)); s.distance++; v:=s.byLevel[level]; v[0]++; if nearest.Len()>=ef&&d>=nearest.Peek().Dist { s.rejected++; v[2]++; s.byLevel[level]=v; continue }; s.accepted++; v[1]++; s.byLevel[level]=v; candidates.Push(structs.NewNodeHeap(d,id)); nearest.Push(structs.NewNodeHeap(d,id)); if nearest.Len()>ef{nearest.Pop()} } }
 n:=nearest.Len(); if cap(dst)<n{dst=make([]int,n)}else{dst=dst[:n]}; for i:=n-1;i>=0;i--{dst[i]=nearest.Pop().Id}; return dst
}

func (h *HNSW) insertLockedArenaProfile(vector []float32,id int,searchBuf []int,arena []float32,dim int,s *expansionOutcomeStats)[]int{
 level:=h.RandomLevel(); newNode:=h.appendNode(id,vector,level); if h.EntryPoint==nil{h.EntryPoint=newNode;return searchBuf}; ep:=h.EntryPoint; entryPoints:=[]int{ep.ID}; L:=ep.Level
 for lc:=L;lc>level;lc--{ep=h.greedySearchLayerArena(vector,ep,lc,arena,dim);entryPoints[0]=ep.ID}
 for lc:=min(L,level);lc>=0;lc--{nearest:=h.searchLayerArenaProfile(vector,entryPoints,h.EfConstruction,lc,searchBuf,arena,dim,s);maxConn:=h.Mmax;if lc==0{maxConn=h.Mmax0};neighbors:=nearest[:min(len(nearest),h.M)];h.updateConnectionsArena(newNode,neighbors,lc,maxConn,arena,dim);if len(nearest)>0{ep=h.Nodes[nearest[0]];entryPoints=nearest;searchBuf=nearest}}
 if level>L{h.EntryPoint=newNode};return searchBuf
}

func TestProfileExpansionThresholdOutcomes(t *testing.T){
 for _,ef:=range []int{64,200}{const n,dim=10000,128;r:=rand.New(rand.NewPCG(0x51,0x99));vectors:=make([][]float32,n);arena:=make([]float32,n*dim);for i:=range vectors{vectors[i]=make([]float32,dim);for j:=range vectors[i]{vectors[i][j]=r.Float32()};copy(arena[i*dim:(i+1)*dim],vectors[i])};h,err:=NewHNSW(Config{M:16,Mmax:32,Mmax0:64,EfConstruction:ef,MaxLevel:16,DistanceFunc:EuclideanDistance});if err!=nil{t.Fatal(err)};lr:=rand.New(rand.NewPCG(0x17,0x23));h.RandFunc=lr.Float64;h.ensureNodeCapacity(n);h.ensureVisitedCapacity(n-1);s:=&expansionOutcomeStats{byLevel:map[int][3]uint64{}};buf:=make([]int,0,ef);for id:=range vectors{buf=h.insertLockedArenaProfile(arena[id*dim:(id+1)*dim],id,buf,arena,dim,s)};fmt.Printf("EXPANSION_OUTCOME ef=%d distance=%d accepted=%d rejected=%d reject_pct=%.2f levels=%v\n",ef,s.distance,s.accepted,s.rejected,100*float64(s.rejected)/float64(s.distance),s.byLevel)}
}
