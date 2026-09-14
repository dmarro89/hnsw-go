package hnsw

// vectorArena stores a fixed-dimension copy of vectors for direct ID-based
// addressing on the serial bulk-build hot path. Node.Vector remains unchanged
// in this experiment so public behavior and parallel construction can keep the
// existing representation while we measure the locality effect end to end.
type vectorArena struct {
	data []float32
	dim  int
}

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
	h.vectors = vectorArena{data: data, dim: dim}
	return true
}

func (h *HNSW) disableVectorArena() {
	h.vectors = vectorArena{}
}

func (h *HNSW) vectorByID(id int) []float32 {
	if h.vectors.dim > 0 {
		start := id * h.vectors.dim
		end := start + h.vectors.dim
		if start >= 0 && end <= len(h.vectors.data) {
			return h.vectors.data[start:end]
		}
	}
	return h.Nodes[id].Vector
}
