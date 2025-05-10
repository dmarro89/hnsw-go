package hnsw

import (
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
  - The ef closest nodes to the query vector, sorted in ascending order of distance.

Time Complexity: O(ef * log(ef)) average case
Space Complexity: O(ef + N) where N is the number of visited nodes

Note: For ef=1, it automatically switches to a more efficient greedy search strategy.
*/
func (h *HNSW) searchLayer(query []float32, entry *structs.Node, ef, level int) []int {
	//v ← ep  set of visited elements
	// Increment the visit stamp for this search
	// This is used to mark nodes as visited and avoid revisiting them
	// in the same search iteration.
	h.visitStamp++

	//C ← ep set of candidates
	candidates := structs.NewMinHeap()
	// W ← ep dynamic list of found nearest neighbors
	nearest := structs.NewMaxHeap()
	defer nearest.Reset()
	defer candidates.Reset()

	// Initialize with the entry point
	initialDist := h.DistanceFunc(query, entry.Vector)

	candidates.Push(structs.NewNodeHeap(initialDist, entry.ID))
	nearest.Push(structs.NewNodeHeap(initialDist, entry.ID))

	// Mark the entry point as visited
	h.markVisited(entry.ID)

	var (
		currentDist  float32
		furthestDist float32
	)

	// while │C│ > 0
	for candidates.Len() > 0 {
		// c ← extract nearest element from C to q
		current := candidates.Pop()
		currentDist = current.Dist
		currentNode := h.Nodes[current.Id]

		// f ← get furthest element from W to q
		if nearest.Len() > 0 {
			furthest := nearest.Peek()
			furthestDist = furthest.Dist
		}

		// if distance(c, q) > distance(f, q)
		// break  -> all elements in W are evaluated
		if currentDist > furthestDist {
			break
		}

		if currentNode == nil || level >= len(currentNode.Neighbors) || len(currentNode.Neighbors[level]) == 0 {
			continue
		}

		// for each e ∈ neighbourhood(c) at layer lc
		for _, neighborID := range currentNode.Neighbors[level] {
			// if e ∉ v
			// v ← v ⋃ e
			if h.markVisited(neighborID) {
				continue
			}

			// f ← get furthest element from W to q
			// if distance(e, q) < distance(f, q) or │W│ < ef
			dist := h.DistanceFunc(query, h.Nodes[neighborID].Vector)
			if dist < furthestDist || nearest.Len() < ef {

				// C ← C ⋃ e
				candidates.Push(structs.NewNodeHeap(dist, neighborID))
				// W ← W ⋃ e
				nearest.Push(structs.NewNodeHeap(dist, neighborID))

				// if │W│ > ef
				// remove furthest element from W to q
				if nearest.Len() > ef {
					nearest.Pop()
				}
			}
		}
	}

	nearestLen := nearest.Len()
	results := make([]int, nearestLen)

	for i := nearestLen - 1; i >= 0; i-- {
		item := nearest.Pop()
		results[i] = item.Id
	}

	return results
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
			for _, neighborID := range currentNode.Neighbors[level] {
				neighbor := h.Nodes[neighborID]
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
func (h *HNSW) KNN_Search(query []float32, K, ef int) []int {
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
	return candidates[:K]
}
