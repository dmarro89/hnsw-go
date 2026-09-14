package hnsw

import (
	"math"

	"dmarro89.github.com/hnsw-go/structs"
)

// InsertBatchArenaExperimental is an end-to-end serial-construction experiment.
// It keeps production Node.Vector/API semantics unchanged, but construction
// distance lookups address a fixed-dimension contiguous arena directly by ID.
// The method exists only to validate the locality hypothesis before changing
// the production HNSW layout.
func (h *HNSW) InsertBatchArenaExperimental(vectors [][]float32) bool {
	if len(vectors) == 0 || len(vectors[0]) == 0 {
		return false
	}
	dim := len(vectors[0])
	for _, vector := range vectors {
		if len(vector) != dim {
			return false
		}
	}

	h.mutex.Lock()
	defer h.mutex.Unlock()
	if len(h.Nodes) != 0 {
		return false
	}

	arena := make([]float32, len(vectors)*dim)
	for i, vector := range vectors {
		copy(arena[i*dim:(i+1)*dim], vector)
	}
	vec := func(id int) []float32 {
		start := id * dim
		return arena[start : start+dim]
	}

	h.ensureNodeCapacity(len(vectors))
	h.ensureVisitedCapacity(len(vectors) - 1)
	searchBuf := make([]int, 0, h.EfConstruction)
	for id, vector := range vectors {
		searchBuf = h.insertLockedArena(vector, id, searchBuf, vec)
	}
	return true
}

func (h *HNSW) insertLockedArena(vector []float32, id int, searchBuf []int, vec func(int) []float32) []int {
	level := h.RandomLevel()
	newNode := h.appendNode(id, vector, level)
	if h.EntryPoint == nil {
		h.EntryPoint = newNode
		return searchBuf
	}

	ep := h.EntryPoint
	entryPoints := []int{ep.ID}
	L := ep.Level
	for lc := L; lc > level; lc-- {
		ep = h.greedySearchLayerArena(vector, ep, lc, vec)
		entryPoints[0] = ep.ID
	}

	for lc := int(math.Min(float64(L), float64(level))); lc >= 0; lc-- {
		nearest := h.searchLayerArena(vector, entryPoints, h.EfConstruction, lc, searchBuf, vec)
		maxConn := h.Mmax
		if lc == 0 {
			maxConn = h.Mmax0
		}
		neighbors := nearest[:min(len(nearest), h.M)]
		h.updateConnectionsArena(newNode, neighbors, lc, maxConn, vec)
		if len(nearest) > 0 {
			ep = h.Nodes[nearest[0]]
			entryPoints = nearest
			searchBuf = nearest
		}
	}
	if level > L {
		h.EntryPoint = newNode
	}
	return searchBuf
}

func (h *HNSW) greedySearchLayerArena(query []float32, entry *structs.Node, level int, vec func(int) []float32) *structs.Node {
	current := entry
	best := h.DistanceFunc(query, vec(current.ID))
	for {
		var next *structs.Node
		nextDist := best
		if level < len(current.Neighbors) {
			for _, id := range current.Neighbors[level] {
				d := h.DistanceFunc(query, vec(id))
				if d < nextDist {
					nextDist, next = d, h.Nodes[id]
				}
			}
		}
		if next == nil {
			return current
		}
		current, best = next, nextDist
	}
}

func (h *HNSW) searchLayerArena(query []float32, entries []int, ef, level int, dst []int, vec func(int) []float32) []int {
	h.visitStamp++
	stamp := h.visitStamp
	candidates := h.ensureHeapPool().GetMinHeap()
	defer h.ensureHeapPool().PutMinHeap(candidates)
	nearest := h.ensureHeapPool().GetMaxHeap()
	defer h.ensureHeapPool().PutMaxHeap(nearest)

	for _, id := range entries {
		if id < 0 || id >= len(h.Nodes) || h.markVisitedArena(id, stamp) {
			continue
		}
		d := h.DistanceFunc(query, vec(id))
		candidates.Push(structs.NewNodeHeap(d, id))
		nearest.Push(structs.NewNodeHeap(d, id))
		if nearest.Len() > ef { nearest.Pop() }
	}
	if candidates.Len() == 0 { return dst[:0] }

	for candidates.Len() > 0 {
		current := candidates.Pop()
		if nearest.Len() >= ef && current.Dist > nearest.Peek().Dist { break }
		node := h.Nodes[current.Id]
		if node == nil || level >= len(node.Neighbors) { continue }
		for _, id := range node.Neighbors[level] {
			if id < 0 || id >= len(h.Nodes) || h.markVisitedArena(id, stamp) { continue }
			d := h.DistanceFunc(query, vec(id))
			if nearest.Len() >= ef && d >= nearest.Peek().Dist { continue }
			candidates.Push(structs.NewNodeHeap(d, id))
			nearest.Push(structs.NewNodeHeap(d, id))
			if nearest.Len() > ef { nearest.Pop() }
		}
	}

	n := nearest.Len()
	if cap(dst) < n { dst = make([]int, n) } else { dst = dst[:n] }
	for i := n-1; i >= 0; i-- { dst[i] = nearest.Pop().Id }
	return dst
}

func (h *HNSW) markVisitedArena(id, stamp int) bool {
	if id >= len(h.visitedIDs) { h.ensureVisitedCapacity(id) }
	if h.visitedIDs[id] == stamp { return true }
	h.visitedIDs[id] = stamp
	return false
}

func (h *HNSW) updateConnectionsArena(q *structs.Node, neighbors []int, level, maxConn int, vec func(int) []float32) {
	q.Neighbors[level] = append(q.Neighbors[level][:0], neighbors...)
	for _, id := range neighbors {
		n := h.Nodes[id]
		if level >= len(n.Neighbors) { continue }
		if len(n.Neighbors[level])+1 <= maxConn {
			n.Neighbors[level] = append(n.Neighbors[level], q.ID)
			continue
		}
		selected := h.selectClosestArena(n.ID, n.Neighbors[level], q.ID, maxConn, vec)
		n.Neighbors[level] = n.Neighbors[level][:len(selected)]
		copy(n.Neighbors[level], selected)
	}
}

func (h *HNSW) selectClosestArena(targetID int, existing []int, extraID, limit int, vec func(int) []float32) []int {
	ids := h.scratchCandidatesBuffer(limit)
	dists := h.scratchDistancesBuffer(limit)
	insert := func(id int) {
		d := h.DistanceFunc(vec(targetID), vec(id))
		pos := len(ids)
		if pos == limit {
			last := limit-1
			if d > dists[last] || (d == dists[last] && id >= ids[last]) { return }
			pos = last
		} else {
			ids = append(ids, 0); dists = append(dists, 0)
		}
		for pos > 0 {
			p := pos-1
			if d > dists[p] || (d == dists[p] && id >= ids[p]) { break }
			ids[pos], dists[pos] = ids[p], dists[p]; pos = p
		}
		ids[pos], dists[pos] = id, d
	}
	insert(extraID)
	for _, id := range existing { insert(id) }
	return ids
}
