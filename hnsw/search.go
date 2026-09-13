package hnsw

import (
	"math"
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
	if initialVisited < 1 {
		initialVisited = 1
	}
	return &searchContext{
		visitedIDs: make([]int, initialVisited),
		heapPool:   structs.NewHeapPoolManager(),
	}
}

func (h *HNSW) acquireSearchContext(initialVisited int) *searchContext {
	if initialVisited < 1 {
		initialVisited = 1
	}
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

func (h *HNSW) releaseSearchContext(ctx *searchContext) {
	h.searchContextPool.Put(ctx)
}

/*
Algorithm 2
SEARCH-LAYER(q, ep, ef, lc)
Input: query element q, enter points ep, number of nearest to q elements to return ef, layer number lc
Output: ef closest neighbors to q

searchLayer performs a beam search at a specific layer of the HNSW graph.
It implements a crucial part of both the insertion and search algorithms,
following the original HNSW paper's approach with some optimizations.

The method uses two priority queues:
- candidates (MinHeap): contains elements to be explored, ordered by distance to query
- nearest (MaxHeap): contains the current ef closest elements found

Parameters:
  - query: the vector we're searching for
  - entry: the entry point node at the current layer
  - ef: size of the dynamic candidate list (controls accuracy vs speed trade-off)
  - level: the current layer in the graph

Returns:
  - The ef closest nodes to the query vector, sorted in ascending order of distance.

Time Complexity: O(ef * log(ef)) average case
Space Complexity: O(ef + N) where N is the number of visited nodes

Note: For ef=1, it automatically switches to a more efficient greedy search strategy.
*/
func (h *HNSW) searchLayerWithEntriesBuffer(query []float32, entryIDs []int, ef, level int, dst []int) []int {
	ctx := searchContext{
		visitStamp: h.visitStamp,
		visitedIDs: h.visitedIDs,
		heapPool:   h.ensureHeapPool(),
	}
	results := searchLayerWithEntriesBufferContext(&ctx, h.Nodes, h.DistanceFunc, query, entryIDs, ef, level, dst)
	h.visitStamp = ctx.visitStamp
	h.visitedIDs = ctx.visitedIDs
	return results
}

func searchLayerWithEntriesBufferContext(ctx *searchContext, nodes []*structs.Node, distance func([]float32, []float32) float32, query []float32, entryIDs []int, ef, level int, dst []int) []int {
	ctx.visitStamp++
	visitStamp := ctx.visitStamp

	pool := ctx.heapPool
	if pool == nil {
		pool = structs.NewHeapPoolManager()
		ctx.heapPool = pool
	}
	visitedIDs := ctx.visitedIDs
	useBoundedDistance := reflect.ValueOf(distance).Pointer() == euclideanDistancePC

	candidates := pool.GetMinHeap()
	defer pool.PutMinHeap(candidates)
	nearest := pool.GetMaxHeap()
	defer pool.PutMaxHeap(nearest)

	for _, entryID := range entryIDs {
		if entryID < 0 || entryID >= len(nodes) {
			continue
		}
		if entryID >= len(visitedIDs) {
			newSize := max(entryID*2, entryID+1)
			newVisited := make([]int, newSize)
			copy(newVisited, visitedIDs)
			visitedIDs = newVisited
		}
		if visitedIDs[entryID] == visitStamp {
			continue
		}
		visitedIDs[entryID] = visitStamp

		entry := nodes[entryID]
		initialDist := distance(query, entry.Vector)
		candidates.Push(structs.NewNodeHeap(initialDist, entry.ID))
		nearest.Push(structs.NewNodeHeap(initialDist, entry.ID))
		if nearest.Len() > ef {
			nearest.Pop()
		}
	}

	if candidates.Len() == 0 {
		return dst[:0]
	}

	var (
		currentDist  float32
		furthestDist = float32(math.MaxFloat32)
		nearestLen   = nearest.Len()
	)

	for candidates.Len() > 0 {
		current := candidates.Pop()
		currentDist = current.Dist
		currentNode := nodes[current.Id]

		if nearestLen >= ef {
			furthest := nearest.Peek()
			furthestDist = furthest.Dist
		} else {
			furthestDist = float32(math.MaxFloat32)
		}

		if currentDist > furthestDist {
			break
		}

		if currentNode == nil || level >= len(currentNode.Neighbors) || len(currentNode.Neighbors[level]) == 0 {
			continue
		}

		for _, neighborID := range currentNode.Neighbors[level] {
			if neighborID >= len(visitedIDs) {
				newSize := max(neighborID*2, neighborID+1)
				newVisited := make([]int, newSize)
				copy(newVisited, visitedIDs)
				visitedIDs = newVisited
			}
			if visitedIDs[neighborID] == visitStamp {
				continue
			}
			visitedIDs[neighborID] = visitStamp

			var dist float32
			if nearestLen < ef || !useBoundedDistance {
				dist = distance(query, nodes[neighborID].Vector)
				if nearestLen >= ef && dist >= furthestDist {
					continue
				}
			} else {
				dist, exceeded := euclideanDistanceWithLimit(query, nodes[neighborID].Vector, furthestDist)
				if exceeded || dist >= furthestDist {
					continue
				}
			}

			candidates.Push(structs.NewNodeHeap(dist, neighborID))
			nearest.Push(structs.NewNodeHeap(dist, neighborID))
			nearestLen++

			if nearestLen > ef {
				nearest.Pop()
				nearestLen--
			}
		}
	}
	ctx.visitedIDs = visitedIDs

	nearestLen = nearest.Len()
	if cap(dst) < nearestLen {
		dst = make([]int, nearestLen)
	} else {
		dst = dst[:nearestLen]
	}

	for i := nearestLen - 1; i >= 0; i-- {
		item := nearest.Pop()
		dst[i] = item.Id
	}

	return dst
}

// greedySearchLayer performs a simple greedy search at a specific layer.
// This is an optimization for ef=1 cases, following a simple hill-climbing approach.
// It's used primarily during the upper layer searches in the HNSW algorithm.
func (h *HNSW) greedySearchLayer(query []float32, entry *structs.Node, level int) *structs.Node {
	return greedySearchLayerNodes(query, entry, level, h.Nodes, h.DistanceFunc)
}

func greedySearchLayerNodes(query []float32, entry *structs.Node, level int, nodes []*structs.Node, distance func([]float32, []float32) float32) *structs.Node {
	currentNode := entry
	bestDist := distance(query, currentNode.Vector)

	for {
		var (
			bestNeighbor     *structs.Node
			bestNeighborDist = bestDist
		)

		if level < len(currentNode.Neighbors) {
			neighbors := currentNode.Neighbors[level]
			for _, neighborID := range neighbors {
				neighbor := nodes[neighborID]
				dist := distance(query, neighbor.Vector)
				if dist < bestNeighborDist {
					bestNeighborDist = dist
					bestNeighbor = neighbor
				}
			}
		}

		if bestNeighbor == nil {
			break
		}

		currentNode = bestNeighbor
		bestDist = bestNeighborDist
	}

	return currentNode
}

// KNN_Search performs a K-nearest neighbor search in the HNSW graph.
// It is safe to call concurrently. The returned slice is newly allocated for
// the caller; use SearchInto to reuse caller-provided result storage.
func (h *HNSW) KNN_Search(query []float32, K, ef int) []int {
	return h.SearchInto(query, K, ef, nil)
}

// SearchInto performs a K-nearest neighbor search while reusing dst for the
// final candidate list. Providing dst with capacity >= ef avoids the result
// allocation on the hot query path. The returned slice aliases dst when its
// capacity is sufficient.
//
// SearchInto is safe to call concurrently as long as each active call owns its
// dst buffer. Passing the same backing array to simultaneous calls is not safe.
func (h *HNSW) SearchInto(query []float32, K, ef int, dst []int) []int {
	if ef < K {
		ef = K
	}

	h.mutex.RLock()
	defer h.mutex.RUnlock()

	if h.EntryPoint == nil {
		return dst[:0]
	}

	entry := h.EntryPoint
	currentLevel := entry.Level

	for lc := currentLevel; lc > 0; lc-- {
		newEntry := h.greedySearchLayer(query, entry, lc)
		if newEntry == nil {
			break
		}
		entry = newEntry
	}

	ctx := h.acquireSearchContext(max(len(h.Nodes), ef))
	defer h.releaseSearchContext(ctx)

	var entryIDs [1]int
	entryIDs[0] = entry.ID
	candidates := searchLayerWithEntriesBufferContext(ctx, h.Nodes, h.DistanceFunc, query, entryIDs[:], ef, 0, dst[:0])

	if len(candidates) < K {
		K = len(candidates)
	}
	return candidates[:K]
}

func euclideanDistanceWithLimit(a, b []float32, limit float32) (float32, bool) {
	var sum0, sum1, sum2, sum3 float32
	i := 0

	for ; i <= len(a)-4; i += 4 {
		d0 := a[i] - b[i]
		d1 := a[i+1] - b[i+1]
		d2 := a[i+2] - b[i+2]
		d3 := a[i+3] - b[i+3]

		sum0 += d0 * d0
		sum1 += d1 * d1
		sum2 += d2 * d2
		sum3 += d3 * d3

		total := sum0 + sum1 + sum2 + sum3
		if total > limit {
			return total, true
		}
	}

	var sum float32
	for ; i < len(a); i++ {
		d := a[i] - b[i]
		sum += d * d
		total := sum + sum0 + sum1 + sum2 + sum3
		if total > limit {
			return total, true
		}
	}

	return sum + sum0 + sum1 + sum2 + sum3, false
}
