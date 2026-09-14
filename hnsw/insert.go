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
	h.mutex.Lock()
	defer h.mutex.Unlock()

	h.ensureNodeCapacity(len(h.Nodes) + 1)
	h.insertLocked(vector, id, nil)
}

// InsertBatch appends a batch of vectors using contiguous IDs starting from the
// current node count. Empty indexes with a fixed-dimension batch use a
// contiguous float32 arena for construction locality; incremental batches retain
// the established scalar path to preserve compatibility.
func (h *HNSW) InsertBatch(vectors [][]float32) {
	h.mutex.Lock()
	defer h.mutex.Unlock()

	if len(h.Nodes) == 0 {
		if dim, ok := uniformVectorDimension(vectors); ok {
			h.insertBatchArenaLocked(vectors, dim)
			return
		}
	}

	nextID := len(h.Nodes)
	h.ensureNodeCapacity(nextID + len(vectors))
	if len(vectors) > 0 {
		h.ensureVisitedCapacity(nextID + len(vectors) - 1)
	}
	searchResultsBuf := make([]int, 0, h.EfConstruction)
	for _, vector := range vectors {
		searchResultsBuf = h.insertLocked(vector, nextID, searchResultsBuf)
		nextID++
	}
}

func (h *HNSW) insertLocked(vector []float32, id int, searchResultsBuf []int) []int {
	if len(vector) == 0 {
		panic("vector cannot be empty")
	}

	if id != len(h.Nodes) {
		panic("node id must be contiguous and match insertion order")
	}

	level := h.RandomLevel()

	newNode := h.appendNode(id, vector, level)
	if h.EntryPoint == nil {
		h.EntryPoint = newNode
		return searchResultsBuf
	}

	ep := h.EntryPoint
	entryPoints := []int{ep.ID}
	L := ep.Level

	for lc := L; lc > level; lc-- {
		newEp := h.greedySearchLayer(vector, ep, lc)
		if newEp == nil {
			break
		}

		ep = newEp
		entryPoints = []int{ep.ID}
	}

	maxLayer := int(math.Min(float64(L), float64(level)))
	for lc := maxLayer; lc >= 0; lc-- {
		nearestNeighbors := h.searchLayerWithEntriesBuffer(vector, entryPoints, h.EfConstruction, lc, searchResultsBuf)

		maxConn := h.Mmax
		if lc == 0 {
			maxConn = h.Mmax0
		}

		neighbors := nearestNeighbors[:min(len(nearestNeighbors), h.M)]
		h.updateBidirectionalConnections(newNode, neighbors, lc, maxConn)

		if len(nearestNeighbors) > 0 {
			ep = h.Nodes[nearestNeighbors[0]]
			entryPoints = nearestNeighbors
			searchResultsBuf = nearestNeighbors
		}
	}

	if level > L {
		h.EntryPoint = newNode
	}
	return searchResultsBuf
}

func (h *HNSW) updateBidirectionalConnections(q *structs.Node, neighbors []int, level int, maxConn int) {
	q.Neighbors[level] = q.Neighbors[level][:0]
	q.Neighbors[level] = append(q.Neighbors[level], neighbors...)

	for _, neighborID := range neighbors {
		neighbor := h.Nodes[neighborID]
		if level >= len(neighbor.Neighbors) {
			continue
		}

		if len(neighbor.Neighbors[level])+1 <= maxConn {
			currentLen := len(neighbor.Neighbors[level])
			if currentLen < cap(neighbor.Neighbors[level]) {
				neighbor.Neighbors[level] = append(neighbor.Neighbors[level], q.ID)
			} else {
				newNeighbors := make([]int, currentLen+1, currentLen+2)
				copy(newNeighbors, neighbor.Neighbors[level])
				newNeighbors[currentLen] = q.ID
				neighbor.Neighbors[level] = newNeighbors
			}
			continue
		}

		candidates := h.selectClosestNeighborIDs(neighbor, neighbor.Neighbors[level], q.ID, maxConn)

		neighbor.Neighbors[level] = neighbor.Neighbors[level][:len(candidates)]
		copy(neighbor.Neighbors[level], candidates)
	}
}

func (h *HNSW) selectClosestNeighborIDs(target *structs.Node, existing []int, extraID, limit int) []int {
	selectedIDs := h.scratchCandidatesBuffer(limit)
	selectedDists := h.scratchDistancesBuffer(limit)

	insertCandidate := func(id int, dist float32) {
		insertPos := len(selectedIDs)
		if insertPos == limit {
			last := limit - 1
			if dist > selectedDists[last] || (dist == selectedDists[last] && id >= selectedIDs[last]) {
				return
			}
			insertPos = last
		} else {
			selectedIDs = append(selectedIDs, 0)
			selectedDists = append(selectedDists, 0)
		}

		for insertPos > 0 {
			prev := insertPos - 1
			if dist > selectedDists[prev] || (dist == selectedDists[prev] && id >= selectedIDs[prev]) {
				break
			}
			selectedIDs[insertPos] = selectedIDs[prev]
			selectedDists[insertPos] = selectedDists[prev]
			insertPos = prev
		}

		selectedIDs[insertPos] = id
		selectedDists[insertPos] = dist
	}

	insertCandidate(extraID, h.DistanceFunc(target.Vector, h.Nodes[extraID].Vector))
	for _, id := range existing {
		insertCandidate(id, h.DistanceFunc(target.Vector, h.Nodes[id].Vector))
	}

	return selectedIDs
}

func containsNeighborID(ids []int, target int) bool {
	for _, id := range ids {
		if id == target {
			return true
		}
	}
	return false
}

func removeNeighborID(ids []int, target int) []int {
	write := 0
	for _, id := range ids {
		if id == target {
			continue
		}
		ids[write] = id
		write++
	}
	return ids[:write]
}
