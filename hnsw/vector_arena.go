package hnsw

import "sync"

// This is deliberately experimental storage: keeping it outside HNSW avoids
// committing the public/internal layout to the prototype before the 100k build
// gate proves the end-to-end benefit. Access is under the index lock on build
// paths and sync.Map keeps concurrent read-only SearchInto safe.
type vectorArena struct {
	data []float32
	dim  int
}

var vectorArenas sync.Map // map[*HNSW]vectorArena

func (h *HNSW) prepareVectorArena(vectors [][]float32) bool {
	if len(vectors) == 0 || len(vectors[0]) == 0 {
		return false
	}
	dim := len(vectors[0])
	for _, vector := range vectors {
		if len(vector) != dim {
			return false
		}
	}

	total := len(h.Nodes) + len(vectors)
	data := make([]float32, total*dim)
	for id, node := range h.Nodes {
		if node == nil || len(node.Vector) != dim {
			return false
		}
		copy(data[id*dim:(id+1)*dim], node.Vector)
	}
	base := len(h.Nodes)
	for i, vector := range vectors {
		copy(data[(base+i)*dim:(base+i+1)*dim], vector)
	}
	vectorArenas.Store(h, vectorArena{data: data, dim: dim})
	return true
}

func (h *HNSW) disableVectorArena() {
	vectorArenas.Delete(h)
}

func (h *HNSW) vectorByID(id int) []float32 {
	if value, ok := vectorArenas.Load(h); ok {
		arena := value.(vectorArena)
		start := id * arena.dim
		end := start + arena.dim
		if start >= 0 && end <= len(arena.data) {
			return arena.data[start:end]
		}
	}
	return h.Nodes[id].Vector
}
