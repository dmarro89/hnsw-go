package hnsw

// adjacencyReadArena is a compact read-only mirror of the current graph
// adjacency. It is rebuilt after graph mutations in the experimental path so
// search traversal can measure locality without changing graph semantics.
type adjacencyReadArena struct {
	ids     []int
	offsets [][]int
	lengths [][]int
}

func buildAdjacencyReadArena(h *HNSW) adjacencyReadArena {
	a := adjacencyReadArena{
		offsets: make([][]int, len(h.Nodes)),
		lengths: make([][]int, len(h.Nodes)),
	}
	total := 0
	for i, n := range h.Nodes {
		if n == nil { continue }
		a.offsets[i] = make([]int, len(n.Neighbors))
		a.lengths[i] = make([]int, len(n.Neighbors))
		for l, ns := range n.Neighbors { a.offsets[i][l] = total; a.lengths[i][l] = len(ns); total += len(ns) }
	}
	a.ids = make([]int, total)
	for i, n := range h.Nodes {
		if n == nil { continue }
		for l, ns := range n.Neighbors { copy(a.ids[a.offsets[i][l]:], ns) }
	}
	return a
}

func (a *adjacencyReadArena) neighbors(id, level int) []int {
	if id < 0 || id >= len(a.offsets) || level < 0 || level >= len(a.offsets[id]) { return nil }
	off, n := a.offsets[id][level], a.lengths[id][level]
	return a.ids[off:off+n]
}
