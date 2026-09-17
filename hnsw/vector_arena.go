package hnsw

import (
	"math"

	"dmarro89.github.com/hnsw-go/structs"
)

// insertBatchArenaLocked builds an empty index from a uniform-dimension batch.
// Node.Vector aliases the contiguous arena so the index owns a single copy of
// vector data, while construction hot paths address vectors directly by ID.
func (h *HNSW) insertBatchArenaLocked(vectors [][]float32, dim int) {
	arena := make([]float32, len(vectors)*dim)
	for i, vector := range vectors { copy(arena[i*dim:(i+1)*dim], vector) }
	h.ensureNodeCapacity(len(vectors)); h.ensureVisitedCapacity(len(vectors)-1)
	searchBuf := make([]int, 0, h.EfConstruction)
	for id := range vectors { start:=id*dim; searchBuf=h.insertLockedArena(arena[start:start+dim], id, searchBuf, arena, dim) }
}

func (h *HNSW) InsertBatchArenaExperimental(vectors [][]float32) bool { dim,ok:=uniformVectorDimension(vectors); if !ok{return false}; h.mutex.Lock(); defer h.mutex.Unlock(); if len(h.Nodes)!=0{return false}; h.insertBatchArenaLocked(vectors,dim); return true }
func (h *HNSW) InsertBatchBaselineExperimental(vectors [][]float32) { h.mutex.Lock(); defer h.mutex.Unlock(); nextID:=len(h.Nodes); h.ensureNodeCapacity(nextID+len(vectors)); if len(vectors)>0{h.ensureVisitedCapacity(nextID+len(vectors)-1)}; searchBuf:=make([]int,0,h.EfConstruction); for _,vector:=range vectors { searchBuf=h.insertLocked(vector,nextID,searchBuf); nextID++ } }
func uniformVectorDimension(vectors [][]float32)(int,bool){if len(vectors)==0||len(vectors[0])==0{return 0,false}; dim:=len(vectors[0]); for _,v:=range vectors{if len(v)!=dim{return 0,false}}; return dim,true}
func arenaVector(arena []float32,dim,id int)[]float32{start:=id*dim;return arena[start:start+dim]}

func (h *HNSW) insertLockedArena(vector []float32,id int,searchBuf []int,arena []float32,dim int)[]int{level:=h.RandomLevel();newNode:=h.appendNode(id,vector,level);if h.EntryPoint==nil{h.EntryPoint=newNode;return searchBuf};ep:=h.EntryPoint;entryPoints:=[]int{ep.ID};L:=ep.Level;for lc:=L;lc>level;lc--{ep=h.greedySearchLayerArena(vector,ep,lc,arena,dim);entryPoints[0]=ep.ID};for lc:=int(math.Min(float64(L),float64(level)));lc>=0;lc--{nearest:=h.searchLayerArena(vector,entryPoints,h.EfConstruction,lc,searchBuf,arena,dim);maxConn:=h.Mmax;if lc==0{maxConn=h.Mmax0};neighbors:=nearest[:min(len(nearest),h.M)];h.updateConnectionsArena(newNode,neighbors,lc,maxConn,arena,dim);if len(nearest)>0{ep=h.Nodes[nearest[0]];entryPoints=nearest;searchBuf=nearest}};if level>L{h.EntryPoint=newNode};return searchBuf}
func (h *HNSW) greedySearchLayerArena(query []float32,entry *structs.Node,level int,arena []float32,dim int)*structs.Node{current:=entry;best:=h.DistanceFunc(query,arenaVector(arena,dim,current.ID));for{var next *structs.Node;nextDist:=best;if level<len(current.Neighbors){for _,id:=range current.Neighbors[level]{d:=h.DistanceFunc(query,arenaVector(arena,dim,id));if d<nextDist{nextDist,next=d,h.Nodes[id]}}};if next==nil{return current};current,best=next,nextDist}}

func (h *HNSW) searchLayerArena(query []float32,entries []int,ef,level int,dst []int,arena []float32,dim int)[]int{
	h.visitStamp++;stamp:=h.visitStamp;pool:=h.ensureHeapPool();candidates:=pool.GetMinHeap();defer pool.PutMinHeap(candidates);nearest:=pool.GetMaxHeap();defer pool.PutMaxHeap(nearest)
	for _,id:=range entries{if id<0||id>=len(h.Nodes)||h.markVisitedArena(id,stamp){continue};d:=h.DistanceFunc(query,arenaVector(arena,dim,id));item:=structs.NewNodeHeap(d,id);candidates.Push(item);if nearest.Len()<ef{nearest.Push(item)}else if d<nearest.Peek().Dist{nearest.ReplaceTop(item)}}
	if candidates.Len()==0{return dst[:0]}
	for candidates.Len()>0{current:=candidates.Pop();if nearest.Len()>=ef&&current.Dist>nearest.Peek().Dist{break};node:=h.Nodes[current.Id];if node==nil||level>=len(node.Neighbors){continue};for _,id:=range node.Neighbors[level]{if id<0||id>=len(h.Nodes)||h.markVisitedArena(id,stamp){continue};d:=h.DistanceFunc(query,arenaVector(arena,dim,id));if nearest.Len()>=ef&&d>=nearest.Peek().Dist{continue};item:=structs.NewNodeHeap(d,id);candidates.Push(item);if nearest.Len()<ef{nearest.Push(item)}else{nearest.ReplaceTop(item)}}}
	n:=nearest.Len();if cap(dst)<n{dst=make([]int,n)}else{dst=dst[:n]};for i:=n-1;i>=0;i--{dst[i]=nearest.Pop().Id};return dst
}
func (h *HNSW) markVisitedArena(id,stamp int)bool{if id>=len(h.visitedIDs){h.ensureVisitedCapacity(id)};if h.visitedIDs[id]==stamp{return true};h.visitedIDs[id]=stamp;return false}
func (h *HNSW) updateConnectionsArena(q *structs.Node,neighbors []int,level,maxConn int,arena []float32,dim int){q.Neighbors[level]=append(q.Neighbors[level][:0],neighbors...);for _,id:=range neighbors{n:=h.Nodes[id];if level>=len(n.Neighbors){continue};if len(n.Neighbors[level])+1<=maxConn{n.Neighbors[level]=append(n.Neighbors[level],q.ID);continue};selected:=h.selectClosestArena(n.ID,n.Neighbors[level],q.ID,maxConn,arena,dim);n.Neighbors[level]=n.Neighbors[level][:len(selected)];copy(n.Neighbors[level],selected)}}
func (h *HNSW) selectClosestArena(targetID int,existing []int,extraID,limit int,arena []float32,dim int)[]int{ids:=h.scratchCandidatesBuffer(limit);dists:=h.scratchDistancesBuffer(limit);target:=arenaVector(arena,dim,targetID);insert:=func(id int){d:=h.DistanceFunc(target,arenaVector(arena,dim,id));pos:=len(ids);if pos==limit{last:=limit-1;if d>dists[last]||(d==dists[last]&&id>=ids[last]){return};pos=last}else{ids=append(ids,0);dists=append(dists,0)};for pos>0{p:=pos-1;if d>dists[p]||(d==dists[p]&&id>=ids[p]){break};ids[pos],dists[pos]=ids[p],dists[p];pos=p};ids[pos],dists[pos]=id,d};insert(extraID);for _,id:=range existing{insert(id)};return ids}
