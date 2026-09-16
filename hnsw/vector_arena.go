package hnsw

import (
	"math"
	"reflect"

	"dmarro89.github.com/hnsw-go/structs"
)

// insertBatchArenaLocked builds an empty index from a uniform-dimension batch.
// Node.Vector aliases the contiguous arena so the index owns a single copy of
// vector data, while construction hot paths address vectors directly by ID.
func (h *HNSW) insertBatchArenaLocked(vectors [][]float32, dim int) {
	arena := make([]float32, len(vectors)*dim)
	for i, vector := range vectors {
		copy(arena[i*dim:(i+1)*dim], vector)
	}

	// The blocked copy is experimental construction scratch for the default
	// Euclidean metric. It stores four vector lanes per coordinate so one query
	// load feeds four independent candidate distances. Custom metrics retain the
	// existing row-major path unchanged.
	var blocked []float32
	if reflect.ValueOf(h.DistanceFunc).Pointer() == reflect.ValueOf(EuclideanDistance).Pointer() {
		blocked = makeBlockedArena4(vectors, dim)
	}

	h.ensureNodeCapacity(len(vectors))
	h.ensureVisitedCapacity(len(vectors) - 1)
	searchBuf := make([]int, 0, h.EfConstruction)
	for id := range vectors {
		start := id * dim
		vector := arena[start : start+dim]
		searchBuf = h.insertLockedArena(vector, id, searchBuf, arena, blocked, dim)
	}
}

func makeBlockedArena4(vectors [][]float32, dim int) []float32 {
	blocks := (len(vectors) + 3) / 4
	out := make([]float32, blocks*dim*4)
	for id, vector := range vectors {
		base, lane := (id/4)*dim*4, id%4
		for d := 0; d < dim; d++ {
			out[base+d*4+lane] = vector[d]
		}
	}
	return out
}

func distanceArena4(query, blocked []float32, dim int, ids [4]int) [4]float32 {
	var base [4]int
	var lane [4]int
	for j, id := range ids {
		base[j] = (id / 4) * dim * 4
		lane[j] = id % 4
		_ = blocked[base[j]+(dim-1)*4+lane[j]]
	}
	var s0, s1, s2, s3 float32
	for d := 0; d < dim; d++ {
		q := query[d]
		off := d * 4
		x0 := q - blocked[base[0]+off+lane[0]]
		x1 := q - blocked[base[1]+off+lane[1]]
		x2 := q - blocked[base[2]+off+lane[2]]
		x3 := q - blocked[base[3]+off+lane[3]]
		s0 += x0 * x0
		s1 += x1 * x1
		s2 += x2 * x2
		s3 += x3 * x3
	}
	return [4]float32{s0, s1, s2, s3}
}

// InsertBatchArenaExperimental is retained for benchmark compatibility. The
// production InsertBatch path now uses the same implementation when applicable.
func (h *HNSW) InsertBatchArenaExperimental(vectors [][]float32) bool {
	dim, ok := uniformVectorDimension(vectors)
	if !ok {
		return false
	}
	h.mutex.Lock()
	defer h.mutex.Unlock()
	if len(h.Nodes) != 0 {
		return false
	}
	h.insertBatchArenaLocked(vectors, dim)
	return true
}

// InsertBatchBaselineExperimental preserves the pre-arena serial bulk path for
// paired performance comparisons. It is not used by production construction.
func (h *HNSW) InsertBatchBaselineExperimental(vectors [][]float32) {
	h.mutex.Lock()
	defer h.mutex.Unlock()

	nextID := len(h.Nodes)
	h.ensureNodeCapacity(nextID + len(vectors))
	if len(vectors) > 0 {
		h.ensureVisitedCapacity(nextID + len(vectors) - 1)
	}
	searchBuf := make([]int, 0, h.EfConstruction)
	for _, vector := range vectors {
		searchBuf = h.insertLocked(vector, nextID, searchBuf)
		nextID++
	}
}

func uniformVectorDimension(vectors [][]float32) (int, bool) {
	if len(vectors) == 0 || len(vectors[0]) == 0 {
		return 0, false
	}
	dim := len(vectors[0])
	for _, vector := range vectors {
		if len(vector) != dim {
			return 0, false
		}
	}
	return dim, true
}

func arenaVector(arena []float32, dim, id int) []float32 {
	start := id * dim
	return arena[start : start+dim]
}

func (h *HNSW) insertLockedArena(vector []float32, id int, searchBuf []int, arena, blocked []float32, dim int) []int {
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
		ep = h.greedySearchLayerArena(vector, ep, lc, arena, dim)
		entryPoints[0] = ep.ID
	}

	for lc := int(math.Min(float64(L), float64(level))); lc >= 0; lc-- {
		nearest := h.searchLayerArena(vector, entryPoints, h.EfConstruction, lc, searchBuf, arena, blocked, dim)
		maxConn := h.Mmax
		if lc == 0 {
			maxConn = h.Mmax0
		}
		neighbors := nearest[:min(len(nearest), h.M)]
		h.updateConnectionsArena(newNode, neighbors, lc, maxConn, arena, dim)
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

func (h *HNSW) greedySearchLayerArena(query []float32, entry *structs.Node, level int, arena []float32, dim int) *structs.Node {
	current := entry
	best := h.DistanceFunc(query, arenaVector(arena, dim, current.ID))
	for {
		var next *structs.Node
		nextDist := best
		if level < len(current.Neighbors) {
			for _, id := range current.Neighbors[level] {
				d := h.DistanceFunc(query, arenaVector(arena, dim, id))
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

func (h *HNSW) searchLayerArena(query []float32, entries []int, ef, level int, dst []int, arena, blocked []float32, dim int) []int {
	h.visitStamp++
	stamp := h.visitStamp
	pool := h.ensureHeapPool()
	candidates := pool.GetMinHeap()
	defer pool.PutMinHeap(candidates)
	nearest := pool.GetMaxHeap()
	defer pool.PutMaxHeap(nearest)

	for _, id := range entries {
		if id < 0 || id >= len(h.Nodes) || h.markVisitedArena(id, stamp) {
			continue
		}
		d := h.DistanceFunc(query, arenaVector(arena, dim, id))
		candidates.Push(structs.NewNodeHeap(d, id))
		nearest.Push(structs.NewNodeHeap(d, id))
		if nearest.Len() > ef {
			nearest.Pop()
		}
	}
	if candidates.Len() == 0 {
		return dst[:0]
	}

	process := func(id int, d float32) {
		if nearest.Len() >= ef && d >= nearest.Peek().Dist {
			return
		}
		candidates.Push(structs.NewNodeHeap(d, id))
		nearest.Push(structs.NewNodeHeap(d, id))
		if nearest.Len() > ef {
			nearest.Pop()
		}
	}

	for candidates.Len() > 0 {
		current := candidates.Pop()
		if nearest.Len() >= ef && current.Dist > nearest.Peek().Dist {
			break
		}
		node := h.Nodes[current.Id]
		if node == nil || level >= len(node.Neighbors) {
			continue
		}

		if blocked == nil {
			for _, id := range node.Neighbors[level] {
				if id < 0 || id >= len(h.Nodes) || h.markVisitedArena(id, stamp) {
					continue
				}
				process(id, h.DistanceFunc(query, arenaVector(arena, dim, id)))
			}
			continue
		}

		var ids [4]int
		n := 0
		flush := func() {
			if n == 4 {
				dists := distanceArena4(query, blocked, dim, ids)
				for j := 0; j < 4; j++ {
					process(ids[j], dists[j])
				}
			} else {
				for j := 0; j < n; j++ {
					process(ids[j], h.DistanceFunc(query, arenaVector(arena, dim, ids[j])))
				}
			}
			n = 0
		}
		for _, id := range node.Neighbors[level] {
			if id < 0 || id >= len(h.Nodes) || h.markVisitedArena(id, stamp) {
				continue
			}
			ids[n] = id
			n++
			if n == 4 {
				flush()
			}
		}
		if n != 0 {
			flush()
		}
	}

	n := nearest.Len()
	if cap(dst) < n {
		dst = make([]int, n)
	} else {
		dst = dst[:n]
	}
	for i := n - 1; i >= 0; i-- {
		dst[i] = nearest.Pop().Id
	}
	return dst
}

func (h *HNSW) markVisitedArena(id, stamp int) bool {
	if id >= len(h.visitedIDs) {
		h.ensureVisitedCapacity(id)
	}
	if h.visitedIDs[id] == stamp {
		return true
	}
	h.visitedIDs[id] = stamp
	return false
}

func (h *HNSW) updateConnectionsArena(q *structs.Node, neighbors []int, level, maxConn int, arena []float32, dim int) {
	q.Neighbors[level] = append(q.Neighbors[level][:0], neighbors...)
	for _, id := range neighbors {
		n := h.Nodes[id]
		if level >= len(n.Neighbors) {
			continue
		}
		if len(n.Neighbors[level])+1 <= maxConn {
			n.Neighbors[level] = append(n.Neighbors[level], q.ID)
			continue
		}
		selected := h.selectClosestArena(n.ID, n.Neighbors[level], q.ID, maxConn, arena, dim)
		n.Neighbors[level] = n.Neighbors[level][:len(selected)]
		copy(n.Neighbors[level], selected)
	}
}

func (h *HNSW) selectClosestArena(targetID int, existing []int, extraID, limit int, arena []float32, dim int) []int {
	ids := h.scratchCandidatesBuffer(limit)
	dists := h.scratchDistancesBuffer(limit)
	target := arenaVector(arena, dim, targetID)
	insert := func(id int) {
		d := h.DistanceFunc(target, arenaVector(arena, dim, id))
		pos := len(ids)
		if pos == limit {
			last := limit - 1
			if d > dists[last] || (d == dists[last] && id >= ids[last]) {
				return
			}
			pos = last
		} else {
			ids = append(ids, 0)
			dists = append(dists, 0)
		}
		for pos > 0 {
			p := pos - 1
			if d > dists[p] || (d == dists[p] && id >= ids[p]) {
				break
			}
			ids[pos], dists[pos] = ids[p], dists[p]
			pos = p
		}
		ids[pos], dists[pos] = id, d
	}
	insert(extraID)
	for _, id := range existing {
		insert(id)
	}
	return ids
}
