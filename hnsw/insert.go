// insert.go
package hnsw

import (
	"math"

	"dmarro89.github.com/hnsw-go/structs"
)

// Algorithm 1
// INSERT(hnsw, q, M, Mmax, efConstruction, mL)
// The insertion process follows Algorithm 1 of the original HNSW paper and consists
// of two main phases:
// 1. Finding the entry point by descending through layers
// 2. Building connections at each layer from the entry point down
//
// The algorithm maintains the small world properties of the graph by:
// - Randomly selecting the maximum layer for new elements
// - Establishing bidirectional connections at each layer
// - Maintaining a fixed maximum number of connections per node
//
// Time Complexity: O(log N) average case
// Space Complexity: O(M * log N) where M is the max connections per layer
func (h *HNSW) Insert(vector []float32, id int) {
	if len(vector) == 0 {
		panic("vector cannot be empty")
	}

	h.mutex.Lock()
	defer h.mutex.Unlock()

	// l ← ⌊-ln(unif(0..1))∙mL⌋ // new element’s level
	// Generate the level for the new node based on a random distribution.
	level := h.RandomLevel()
	q := structs.NewNode(id, vector, level, h.MaxLevel, h.Mmax)

	// Generate the level for the new node based on a random distribution.
	if h.EntryPoint == nil {
		h.EntryPoint = q
		h.Nodes = append(h.Nodes, q)
		return
	}

	// ep ← get entry point for hnsw
	ep := h.EntryPoint
	// L ← level of ep - top layer for hnsw
	L := ep.Level

	// Add the new node to the list of nodes in the graph
	h.Nodes = append(h.Nodes, q)

	// Phase 1: Descend through layers to find entry point for insertion
	// This phase finds good starting points for the lower layer insertions
	// for lc ← L … l+1
	for lc := L; lc > level; lc-- {
		// W ← SEARCH-LAYER(q, ep, ef=1, lc)
		newEp := h.greedySearchLayer(q.Vector, ep, lc)
		if newEp == nil {
			break
		}

		// Update entry point for next iteration
		ep = newEp
	}

	// Phase 2: Connecting the new node at each layer from the minimum of (L, l) to the base layer (0).
	// for lc ← min(L, l) … 0
	maxLayer := int(math.Min(float64(L), float64(level)))
	for lc := maxLayer; lc >= 0; lc-- {
		// W ← list for the currently found nearest elements
		// W ← SEARCH-LAYER(q, ep, efConstruction, lc)
		nearestNeighbors := h.searchLayer(q.Vector, ep, h.EfConstruction, lc)

		// Ensure that the number of connections does not exceed the allowed limit.
		maxConn := h.Mmax
		if lc == 0 {
			maxConn = h.Mmax0
		}

		// neighbors ← SELECT-NEIGHBORS(q, W, M, lc)
		neighbors := h.simpleSelectNeighbors(nearestNeighbors, maxConn)
		h.updateBidirectionalConnections(q, neighbors, lc, maxConn)

		// ep ← W
		if len(nearestNeighbors) > 0 {
			item := nearestNeighbors[0]
			itemID := item.ID
			ep = h.Nodes[itemID]
		}
	}

	// If the new node's level is higher than the current top level, update the entry point.
	// if l > L
	if level > L {
		h.EntryPoint = q
	}
}

// updateBidirectionalConnections establishes and maintains bidirectional connections
// between a node and its neighbors at a specific level.
//
// The method ensures that:
// 1. The node is connected to its neighbors
// 2. The neighbors are connected back to the node
// 3. No node exceeds its maximum allowed connections
// 4. Connections are optimized to maintain the best possible neighbors
func (h *HNSW) updateBidirectionalConnections(q *structs.Node, neighbors []*structs.Node, level int, maxConn int) {
	// add bidirectional connections from neighbors to q at layer lc
	q.Neighbors[level] = make([]*structs.Node, len(neighbors))
	copy(q.Neighbors[level], neighbors)

	tmpHeap := h.heapPool.GetMinHeap()
	defer h.heapPool.PutMinHeap(tmpHeap)

	// The maximum number of neighbors to consider for optimization
	maxNeighborsCount := maxConn + 1
	// Preallocate the slice for the candidates slice
	candidates := make([]*structs.Node, 0, maxNeighborsCount)

	// for each e ∈ neighbors
	for _, neighbor := range neighbors {
		if level >= len(neighbor.Neighbors) {
			continue
		}

		// append q to the neighborhood of e at layer lc
		neighbor.Neighbors[level] = append(neighbor.Neighbors[level], q)

		// Check if we need to optimize connections
		if len(neighbor.Neighbors[level]) <= maxConn {
			continue
		}

		// Optimize the neighbors' neighborhoods.
		tmpHeap.Reset()
		// Reset the candidates slice
		candidates = candidates[:0]

		// eConn ← neighborhood(neighbor) at layer level
		eConn := neighbor.Neighbors[level]

		for _, n := range eConn {
			dist := h.DistanceFunc(neighbor.Vector, n.Vector)
			tmpHeap.Push(structs.NewNodeHeap(dist, n.ID))
		}

		// Get the top neighbors
		heapSize := tmpHeap.Len()
		for i := 0; i < heapSize; i++ {
			item := tmpHeap.Pop()
			candidates = append(candidates, h.Nodes[item.Id])
		}

		// Shrink the neighborhood if it exceeds the allowed limit.
		// eNewConn ← SELECT-NEIGHBORS(e, eConn, Mmax, lc)
		eNewConn := h.simpleSelectNeighbors(candidates, maxConn)
		neighbor.Neighbors[level] = eNewConn
	}
}
