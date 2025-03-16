package hnsw

import (
	"container/heap"

	"dmarro89.github.com/hnsw-go/structs"
)

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
  - MaxHeap containing the ef closest elements found

Time Complexity: O(ef * log(ef)) average case
Space Complexity: O(ef + N) where N is the number of visited nodes

Note: For ef=1, it automatically switches to a more efficient greedy search strategy.
*/
func (h *HNSW) searchLayer(query []float32, entry *structs.Node, ef, level int) *structs.MaxHeap {
	//v ← ep  set of visited elements
	visited := h.visitedPool.Get().(map[int]struct{})
	for k := range visited {
		delete(visited, k)
	}

	//C ← ep set of candidates
	candidates := h.heapPool.GetMinHeap()
	// W ← ep dynamic list of found nearest neighbors
	nearest := h.heapPool.GetMaxHeap()

	defer func() {
		h.heapPool.PutMinHeap(candidates)
		h.heapPool.PutMaxHeap(nearest)
	}()

	// Initialize with the entry point
	initialDist := h.DistanceFunc(query, entry.Vector)
	item := structs.EncodeHeapItem(initialDist, entry.ID)
	heap.Push(candidates, item)
	heap.Push(nearest, item)
	visited[entry.ID] = struct{}{}

	var (
		currentDist float32
		currentID   int
	)

	// while │C│ > 0
	for candidates.Len() > 0 {
		// c ← extract nearest element from C to q
		current := heap.Pop(candidates).(uint64)
		currentDist, currentID = structs.DecodeHeapItem(current)
		// f ← get furthest element from W to q
		furthest := nearest.Peek()
		furthestDist, _ := structs.DecodeHeapItem(furthest)

		// if distance(c, q) > distance(f, q)
		// break  -> all elements in W are evaluated
		if currentDist > furthestDist {
			break
		}

		currentNode := h.Nodes[currentID]

		if currentNode == nil || level >= len(currentNode.Neighbors) || len(currentNode.Neighbors[level]) == 0 {
			continue
		}

		neighbors := currentNode.Neighbors[level]
		// for each e ∈ neighbourhood(c) at layer lc
		for _, neighbor := range neighbors {
			// if e ∉ v
			// v ← v ⋃ e
			if _, seen := visited[neighbor.ID]; seen {
				continue
			}
			visited[neighbor.ID] = struct{}{}

			// f ← get furthest element from W to q
			// if distance(e, q) < distance(f, q) or │W│ < ef
			dist := h.DistanceFunc(query, neighbor.Vector)
			if dist < furthestDist || nearest.Len() < ef {
				item := structs.EncodeHeapItem(dist, neighbor.ID)
				// C ← C ⋃ e
				heap.Push(candidates, item)
				// W ← W ⋃ e
				heap.Push(nearest, item)
				// if │W│ > ef
				// remove furthest element from W to q
				if nearest.Len() > ef {
					heap.Pop(nearest)
				}
			}
		}
	}

	return nearest
}

// greedySearchLayer performs a simple greedy search at a specific layer.
// This is an optimization for ef=1 cases, following a simple hill-climbing approach.
// It's used primarily during the upper layer searches in the HNSW algorithm.
func (h *HNSW) greedySearchLayer(query []float32, entry *structs.Node, level int) *structs.Node {
	currentNode := entry
	bestDist := h.DistanceFunc(query, currentNode.Vector)

	for {
		improved := false

		// Check all neighbors at this level
		if level < len(currentNode.Neighbors) {
			for _, neighbor := range currentNode.Neighbors[level] {
				dist := h.DistanceFunc(query, neighbor.Vector)
				if dist < bestDist {
					bestDist = dist
					currentNode = neighbor
					improved = true
					break // Take first improvement
				}
			}
		}

		if !improved {
			break
		}
	}

	return currentNode
}

// KNN_Search performs a K-nearest neighbor search in the HNSW graph.
// This implements Algorithm 5 from the original HNSW paper, using a two-phase search:
// 1. Greedy search through upper layers to find entry point for layer 0
// 2. Beam search at layer 0 to find the K nearest neighbors
//
// Parameters:
//   - query: the target vector to search for
//   - K: number of nearest neighbors to return
//   - ef: size of the dynamic candidate list (controls accuracy vs speed trade-off)
//
// Returns:
//   - Slice of K nearest nodes, sorted by distance to query
//
// Note: ef should be >= K for meaningful results. Larger ef values give better
// accuracy at the cost of slower search times.
func (h *HNSW) KNN_Search(query []float32, K, ef int) []*structs.Node {
	if ef < K {
		ef = K
	}

	h.mutex.RLock()
	defer h.mutex.RUnlock()

	if h.EntryPoint == nil {
		return nil
	}

	// Set the entry point for the search.
	// ep ← get entry point for hnsw
	entry := h.EntryPoint

	// Get the top layer of the entry point.
	// L ← level of ep // top layer for hnsw
	currentLevel := entry.Level

	// Perform greedy search in higher levels (L to 1).
	// for lc ← L … 1
	for lc := currentLevel; lc > 0; lc-- {
		// Perform SEARCH-LAYER(q, ep, ef=1, lc)
		// Greedy search with ef=1 to find the closest element at the current level.
		newEntry := h.greedySearchLayer(query, entry, lc)
		if newEntry == nil {
			break
		}
		// Update the entry point to the nearest element found.
		// ep ← get nearest element from W to q
		entry = newEntry
	}

	// Perform beam search at level 0 with ef size.
	// W ← SEARCH-LAYER(q, ep, ef, lc=0)
	candidates := h.searchLayer(query, entry, ef, 0)

	// Extract the top K nearest elements from W.
	// return K nearest elements from W to q
	results := make([]*structs.Node, 0, K)

	for candidates.Len() > 0 && len(results) < K {
		item := heap.Pop(candidates).(uint64)
		_, itemID := structs.DecodeHeapItem(item)
		results = append(results, h.Nodes[itemID])
	}

	return results
}

// simpleSelectNeighbors selects up to M closest neighbors from the candidates heap.
// This implements the basic neighbor selection strategy from the HNSW paper,
// selecting the M closest elements based on distance.
//
// Note: The input heap is consumed during the process.
func (h *HNSW) simpleSelectNeighbors(candidates *structs.MinHeap, M int) []*structs.Node {
	neighbors := make([]*structs.Node, 0, M)

	// Extract the top M elements from the MinHeap
	for candidates.Len() > 0 && len(neighbors) < M {
		item := heap.Pop(candidates).(uint64)
		_, itemID := structs.DecodeHeapItem(item)

		if itemID > len(h.Nodes) || h.Nodes[itemID] == nil {
			continue
		}

		neighbors = append(neighbors, h.Nodes[itemID])
	}

	return neighbors
}
