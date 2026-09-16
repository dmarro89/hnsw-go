package hnsw

import "dmarro89.github.com/hnsw-go/structs"

// bulkNeighborArena owns the per-level slice headers and neighbor IDs for an
// empty-index serial bulk build. The levels are generated before insertion in
// exactly the same RNG order as the scalar path.
type bulkNeighborArena struct {
	levels  []int
	headers [][]int
	ids     []int
	header  int
	id      int
}

func newBulkNeighborArena(h *HNSW, count int) *bulkNeighborArena {
	levels := make([]int, count)
	headerCount, idCount := 0, 0
	for i := range levels {
		level := h.RandomLevel()
		levels[i] = level
		headerCount += level + 1
		idCount += h.Mmax0 + level*h.Mmax
	}
	return &bulkNeighborArena{
		levels:  levels,
		headers: make([][]int, headerCount),
		ids:     make([]int, idCount),
	}
}

func (a *bulkNeighborArena) initNode(node *structs.Node, id int, vector []float32, level, mmax, mmax0 int) {
	n := level + 1
	neighbors := a.headers[a.header : a.header+n]
	a.header += n
	for lc := range neighbors {
		capacity := mmax
		if lc == 0 {
			capacity = mmax0
		}
		neighbors[lc] = a.ids[a.id:a.id:a.id+capacity]
		a.id += capacity
	}
	node.ID = id
	node.Vector = vector
	node.Level = level
	node.Neighbors = neighbors
}

func (h *HNSW) appendNodeBulkArena(id int, vector []float32, level int, a *bulkNeighborArena) *structs.Node {
	h.nodeStorage = append(h.nodeStorage, structs.Node{})
	node := &h.nodeStorage[len(h.nodeStorage)-1]
	a.initNode(node, id, vector, level, h.Mmax, h.Mmax0)
	h.Nodes = append(h.Nodes, node)
	return node
}
