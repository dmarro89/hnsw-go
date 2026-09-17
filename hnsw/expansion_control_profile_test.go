package hnsw

import (
 "fmt"
 "math/rand/v2"
 "testing"
 "dmarro89.github.com/hnsw-go/structs"
)

type expansionControlStats struct { visits, invalid, visited, distances, rejected, admitted, pops, breaks, evictions uint64 }

func (h *HNSW) searchLayerArenaControlProfile(query []float32, entries []int, ef, level int, dst []int, arena []float32, dim int, s *expansionControlStats) []int {
 h.visitStamp++; stamp:=h.visitStamp; pool:=h.ensureHeapPool(); candidates:=pool.GetMinHeap(); defer pool.PutMinHeap(candidates); nearest:=pool.GetMaxHeap(); defer pool.PutMaxHeap(nearest)
 for _,id:=range entries { if id<0||id>=len(h.Nodes){continue}; if h.markVisitedArena(id,stamp){continue}; d:=h.DistanceFunc(query,arenaVector(arena,dim,id)); candidates.Push(structs.NewNodeHeap(d,id)); nearest.Push(structs.NewNodeHeap(d,id)); if nearest.Len()>ef{nearest.Pop()} }
 if candidates.Len()==0{return dst[:0]}
 for candidates.Len()>0 { current:=candidates.Pop(); s.pops++; if nearest.Len()>=ef&&current.Dist>nearest.Peek().Dist{s.breaks++;break}; node:=h.Nodes[current.Id]; if node==nil||level>=len(node.Neighbors){continue}; for _,id:=range node.Neighbors[level]{ s.visits++; if id<0||id>=len(h.Nodes){s.invalid++;continue}; if h.markVisitedArena(id,stamp){s.visited++;continue}; s.distances++; d:=h.DistanceFunc(query,arenaVector(arena,dim,id)); if nearest.Len()>=ef&&d>=nearest.Peek().Dist{s.rejected++;continue}; s.admitted++; candidates.Push(structs.NewNodeHeap(d,id)); nearest.Push(structs.NewNodeHeap(d,id)); if nearest.Len()>ef{s.evictions++;nearest.Pop()} } }
 n:=nearest.Len(); if cap(dst)<n{dst=make([]int,n)}else{dst=dst[:n]}; for i:=n-1;i>=0;i--{dst[i]=nearest.Pop().Id}; return dst
}

func (h *HNSW) insertLockedArenaControlProfile(vector []float32,id int,searchBuf []int,arena []float32,dim int,s *expansionControlStats)[]int{
 level:=h.RandomLevel(); newNode:=h.appendNode(id,vector,level); if h.EntryPoint==nil{h.EntryPoint=newNode;return searchBuf}; ep:=h.EntryPoint; entryPoints:=[]int{ep.ID}; L:=ep.Level
 for lc:=L;lc>level;lc--{ep=h.greedySearchLayerArena(vector,ep,lc,arena,dim);entryPoints[0]=ep.ID}
 for lc:=min(L,level);lc>=0;lc--{nearest:=h.searchLayerArenaControlProfile(vector,entryPoints,h.EfConstruction,lc,searchBuf,arena,dim,s);maxConn:=h.Mmax;if lc==0{maxConn=h.Mmax0};neighbors:=nearest[:min(len(nearest),h.M)];h.updateConnectionsArena(newNode,neighbors,lc,maxConn,arena,dim);if len(nearest)>0{ep=h.Nodes[nearest[0]];entryPoints=nearest;searchBuf=nearest}}
 if level>L{h.EntryPoint=newNode};return searchBuf
}

func TestProfileExpansionControlFlow(t *testing.T){
 if testing.Short(){t.Skip("evidence-only profile")}
 for _,ef:=range []int{64,200}{const n,dim=10000,128;r:=rand.New(rand.NewPCG(0x51,0x99));arena:=make([]float32,n*dim);for i:=range n{for j:=range dim{arena[i*dim+j]=r.Float32()}};h,err:=NewHNSW(Config{M:16,Mmax:32,Mmax0:64,EfConstruction:ef,MaxLevel:16,DistanceFunc:EuclideanDistance});if err!=nil{t.Fatal(err)};lr:=rand.New(rand.NewPCG(0x17,0x23));h.RandFunc=lr.Float64;h.ensureNodeCapacity(n);h.ensureVisitedCapacity(n-1);s:=&expansionControlStats{};buf:=make([]int,0,ef);for id:=range n{buf=h.insertLockedArenaControlProfile(arena[id*dim:(id+1)*dim],id,buf,arena,dim,s)};fmt.Printf("EXPANSION_CONTROL ef=%d visits=%d invalid=%d visited=%d visited_pct=%.2f distances=%d rejected=%d reject_pct=%.2f admitted=%d pops=%d breaks=%d evictions=%d\n",ef,s.visits,s.invalid,s.visited,100*float64(s.visited)/float64(s.visits),s.distances,s.rejected,100*float64(s.rejected)/float64(s.distances),s.admitted,s.pops,s.breaks,s.evictions)}
}
