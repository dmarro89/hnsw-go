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

	newNode := structs.NewNode(id, vector, level, h.MaxLevel, h.Mmax, h.Mmax0)
	// Generate the level for the new node based on a random distribution.
	if h.EntryPoint == nil {
		h.EntryPoint = newNode
		h.Nodes = append(h.Nodes, newNode)
		return
	}

	// ep ← get entry point for hnsw
	ep := h.EntryPoint
	// L ← level of ep - top layer for hnsw
	L := ep.Level

	// Add the new node to the list of nodes in the graph
	h.Nodes = append(h.Nodes, newNode)

	// Phase 1: Descend through layers to find entry point for insertion
	// This phase finds good starting points for the lower layer insertions
	// for lc ← L … l+1
	for lc := L; lc > level; lc-- {
		// W ← SEARCH-LAYER(q, ep, ef=1, lc)
		newEp := h.greedySearchLayer(vector, ep, lc)
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
		nearestNeighbors := h.searchLayer(vector, ep, h.EfConstruction, lc)

		// Ensure that the number of connections does not exceed the allowed limit.
		maxConn := h.Mmax
		if lc == 0 {
			maxConn = h.Mmax0
		}

		// neighbors ← SELECT-NEIGHBORS(q, W, M, lc)
		var neighbors []*structs.Node
		if len(nearestNeighbors) <= maxConn {
			neighbors = nearestNeighbors
		} else {
			neighbors = nearestNeighbors[:maxConn]
		}
		h.updateBidirectionalConnections(newNode, neighbors, lc, maxConn)

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
		h.EntryPoint = newNode
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
	q.Neighbors[level] = q.Neighbors[level][:0]                   // Reset and reuse the slice
	q.Neighbors[level] = append(q.Neighbors[level], neighbors...) // Append neighbors

	// Getting the candidates nodes for the neighbors from the pool
	// and the temporary heap for the optimization process
	candidates := h.nodePool.Get()
	tmpHeap := h.heapPool.GetMinHeap()
	defer h.heapPool.PutMinHeap(tmpHeap)
	defer h.nodePool.Put(candidates)

	// for each e ∈ neighbors
	for _, neighbor := range neighbors {
		if level >= len(neighbor.Neighbors) {
			continue
		}

		// Check if we need to optimize connections
		if len(neighbor.Neighbors[level])+1 <= maxConn {
			currentLen := len(neighbor.Neighbors[level])
			if currentLen < cap(neighbor.Neighbors[level]) {
				// There is enough capacity, so we can reuse the slice
				neighbor.Neighbors[level] = append(neighbor.Neighbors[level], q)
			} else {
				// We need to allocate a new slice with incremented capacity
				newNeighbors := make([]*structs.Node, currentLen+1, currentLen+2)
				copy(newNeighbors, neighbor.Neighbors[level])
				newNeighbors[currentLen] = q
				neighbor.Neighbors[level] = newNeighbors
			}
			continue
		}

		// Optimize the neighbors' neighborhoods.
		// Reset the candidates slice
		candidates = candidates[:0]

		// append q to the list of neighbors
		qDist := h.DistanceFunc(q.Vector, neighbor.Vector)
		nodeHeap := h.nodeHeapPool.Get(qDist, q.ID)
		tmpHeap.Push(nodeHeap)

		// eConn ← neighborhood(neighbor) at layer level
		eConn := neighbor.Neighbors[level]

		for _, n := range eConn {
			dist := h.DistanceFunc(neighbor.Vector, n.Vector)
			nodeHeap := h.nodeHeapPool.Get(dist, n.ID)
			tmpHeap.Push(nodeHeap)
		}

		// Get the top maxConn neighbors
		// Shrink the neighborhood if it exceeds the allowed limit.
		for i := 0; i < maxConn && tmpHeap.Len() > 0; i++ {
			item := tmpHeap.Pop()
			candidates = append(candidates, h.Nodes[item.Id])
			h.nodeHeapPool.Put(item)
		}

		// Clean up the heap
		for tmpHeap.Len() > 0 {
			item := tmpHeap.Pop()
			h.nodeHeapPool.Put(item)
		}

		// eNewConn ← SELECT-NEIGHBORS(e, eConn, Mmax, lc)
		neighbor.Neighbors[level] = neighbor.Neighbors[level][:len(candidates)]
		copy(neighbor.Neighbors[level], candidates)
	}
}
