package hnsw

import (
	"reflect"

	"dmarro89.github.com/hnsw-go/structs"
)

var euclideanDistancePC = reflect.ValueOf(EuclideanDistance).Pointer()

type searchContext struct {
	visitStamp int
	visitedIDs []int
	heapPool   *structs.HeapPoolManager
}

func newSearchContext(initialVisited int) *searchContext {
	if initialVisited < 1 { initialVisited = 1 }
	return &searchContext{visitedIDs: make([]int, initialVisited), heapPool: structs.NewHeapPoolManager()}
}

func (h *HNSW) acquireSearchContext(initialVisited int) *searchContext {
	if initialVisited < 1 { initialVisited = 1 }
	if pooled := h.searchContextPool.Get(); pooled != nil {
		ctx := pooled.(*searchContext)
		if len(ctx.visitedIDs) < initialVisited {
			ctx.visitedIDs = make([]int, initialVisited)
			ctx.visitStamp = 0
		}
		return ctx
	}
	return newSearchContext(initialVisited)
}

func (h *HNSW) releaseSearchContext(ctx *searchContext) { h.searchContextPool.Put(ctx) }

func (h *HNSW) searchLayerWithEntriesBuffer(query []float32, entryIDs []int, ef, level int, dst []int) []int {
	ctx := searchContext{visitStamp: h.visitStamp, visitedIDs: h.visitedIDs, heapPool: h.ensureHeapPool()}
	results := searchLayerWithEntriesBufferContext(&ctx, h.Nodes, h.DistanceFunc, query, entryIDs, ef, level, dst)
	h.visitStamp = ctx.visitStamp
	h.visitedIDs = ctx.visitedIDs
	return results
}

func searchLayerWithEntriesBufferContext(ctx *searchContext, nodes []*structs.Node, distance func([]float32, []float32) float32, query []float32, entryIDs []int, ef, level int, dst []int) []int {
	ctx.visitStamp++
	visitStamp := ctx.visitStamp
	pool := ctx.heapPool
	if pool == nil { pool = structs.NewHeapPoolManager(); ctx.heapPool = pool }
	visitedIDs := ctx.visitedIDs
	candidates := pool.GetMinHeap()
	defer pool.PutMinHeap(candidates)
	nearest := pool.GetMaxHeap()
	defer pool.PutMaxHeap(nearest)

	for _, entryID := range entryIDs {
		if entryID < 0 || entryID >= len(nodes) { continue }
		if entryID >= len(visitedIDs) {
			newSize := max(entryID*2, entryID+1)
			newVisited := make([]int, newSize)
			copy(newVisited, visitedIDs)
			visitedIDs = newVisited
		}
		if visitedIDs[entryID] == visitStamp { continue }
		visitedIDs[entryID] = visitStamp
		entry := nodes[entryID]
		initialDist := distance(query, entry.Vector)
		candidates.Push(structs.NewNodeHeap(initialDist, entry.ID))
		nearest.Push(structs.NewNodeHeap(initialDist, entry.ID))
		if nearest.Len() > ef { nearest.Pop() }
	}
	if candidates.Len() == 0 { return dst[:0] }

	for candidates.Len() > 0 {
		current := candidates.Pop()
		if nearest.Len() >= ef && current.Dist > nearest.Peek().Dist { break }
		currentNode := nodes[current.Id]
		if currentNode == nil || level >= len(currentNode.Neighbors) { continue }
		for _, neighborID := range currentNode.Neighbors[level] {
			if neighborID < 0 || neighborID >= len(nodes) { continue }
			if neighborID >= len(visitedIDs) {
				newSize := max(neighborID*2, neighborID+1)
				newVisited := make([]int, newSize)
				copy(newVisited, visitedIDs)
				visitedIDs = newVisited
			}
			if visitedIDs[neighborID] == visitStamp { continue }
			visitedIDs[neighborID] = visitStamp
			dist := distance(query, nodes[neighborID].Vector)
			if nearest.Len() >= ef && dist >= nearest.Peek().Dist { continue }
			candidates.Push(structs.NewNodeHeap(dist, neighborID))
			nearest.Push(structs.NewNodeHeap(dist, neighborID))
			if nearest.Len() > ef { nearest.Pop() }
		}
	}
	ctx.visitedIDs = visitedIDs
	nearestLen := nearest.Len()
	if cap(dst) < nearestLen { dst = make([]int, nearestLen) } else { dst = dst[:nearestLen] }
	for i := nearestLen - 1; i >= 0; i-- { dst[i] = nearest.Pop().Id }
	return dst
}

func (h *HNSW) greedySearchLayer(query []float32, entry *structs.Node, level int) *structs.Node {
	return greedySearchLayerNodes(query, entry, level, h.Nodes, h.DistanceFunc)
}

func greedySearchLayerNodes(query []float32, entry *structs.Node, level int, nodes []*structs.Node, distance func([]float32, []float32) float32) *structs.Node {
	currentNode := entry
	bestDist := distance(query, currentNode.Vector)
	for {
		var bestNeighbor *structs.Node
		bestNeighborDist := bestDist
		if level < len(currentNode.Neighbors) {
			for _, neighborID := range currentNode.Neighbors[level] {
				neighbor := nodes[neighborID]
				dist := distance(query, neighbor.Vector)
				if dist < bestNeighborDist { bestNeighborDist = dist; bestNeighbor = neighbor }
			}
		}
		if bestNeighbor == nil { break }
		currentNode = bestNeighbor
		bestDist = bestNeighborDist
	}
	return currentNode
}

func (h *HNSW) KNN_Search(query []float32, K, ef int) []int { return h.SearchInto(query, K, ef, nil) }

func (h *HNSW) SearchInto(query []float32, K, ef int, dst []int) []int {
	if ef < K { ef = K }
	h.mutex.RLock()
	defer h.mutex.RUnlock()
	if h.EntryPoint == nil { return dst[:0] }
	entry := h.EntryPoint
	for lc := entry.Level; lc > 0; lc-- {
		newEntry := h.greedySearchLayer(query, entry, lc)
		if newEntry == nil { break }
		entry = newEntry
	}
	ctx := h.acquireSearchContext(max(len(h.Nodes), ef))
	defer h.releaseSearchContext(ctx)
	var entryIDs [1]int
	entryIDs[0] = entry.ID
	candidates := searchLayerWithEntriesBufferContext(ctx, h.Nodes, h.DistanceFunc, query, entryIDs[:], ef, 0, dst[:0])
	if len(candidates) < K { K = len(candidates) }
	return candidates[:K]
}

func euclideanDistanceWithLimit(a, b []float32, limit float32) (float32, bool) {
	var sum0, sum1, sum2, sum3 float32
	i := 0
	for ; i <= len(a)-4; i += 4 {
		d0 := a[i] - b[i]; d1 := a[i+1] - b[i+1]; d2 := a[i+2] - b[i+2]; d3 := a[i+3] - b[i+3]
		sum0 += d0*d0; sum1 += d1*d1; sum2 += d2*d2; sum3 += d3*d3
		total := sum0+sum1+sum2+sum3
		if total > limit { return total, true }
	}
	var sum float32
	for ; i < len(a); i++ {
		d := a[i]-b[i]; sum += d*d
		total := sum+sum0+sum1+sum2+sum3
		if total > limit { return total, true }
	}
	return sum+sum0+sum1+sum2+sum3, false
}
