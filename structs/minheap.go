package structs

type NodeHeap struct {
	Dist float32
	Id   int
}

func NewNodeHeap(dist float32, id int) *NodeHeap {
	return &NodeHeap{
		Dist: dist,
		Id:   id,
	}
}

// MinHeap keeps elements in ascending order (smallest on top).
type MinHeap struct {
	nodes []*NodeHeap
}

// NewMinHeap creates a new heap with initial capacity.
func NewMinHeap() *MinHeap {
	return &MinHeap{
		nodes: make([]*NodeHeap, 0, 64),
	}
}

// Len returns the number of elements in the heap.
func (h *MinHeap) Len() int {
	return len(h.nodes)
}

// Push adds a new element and restores the heap property.
func (h *MinHeap) Push(n *NodeHeap) {
	h.nodes = append(h.nodes, n)
	h.siftUp(len(h.nodes) - 1)
}

// Pop removes and returns the element with the minimum value.
func (h *MinHeap) Pop() *NodeHeap {
	if len(h.nodes) == 0 {
		return nil
	}
	min := h.nodes[0]
	lastIndex := len(h.nodes) - 1
	h.nodes[0] = h.nodes[lastIndex]
	h.nodes = h.nodes[:lastIndex]
	h.siftDown(0)
	return min
}

// Peek returns the minimum value without removing it.
func (h *MinHeap) Peek() *NodeHeap {
	if len(h.nodes) == 0 {
		return nil
	}
	return h.nodes[0]
}

// Reset empties the heap while maintaining the underlying capacity.
func (h *MinHeap) Reset() {
	h.nodes = h.nodes[:0]
}

// siftUp restores the heap property by moving up the tree.
func (h *MinHeap) siftUp(i int) {
	for i > 0 {
		parent := (i - 1) / 2
		if h.nodes[i].Dist < h.nodes[parent].Dist {
			h.nodes[i], h.nodes[parent] = h.nodes[parent], h.nodes[i]
			i = parent
		} else {
			break
		}
	}
}

// siftDown restores the heap property by moving down the tree.
func (h *MinHeap) siftDown(i int) {
	n := len(h.nodes)
	for {
		left := 2*i + 1
		right := 2*i + 2
		smallest := i
		if left < n && h.nodes[left].Dist < h.nodes[smallest].Dist {
			smallest = left
		}
		if right < n && h.nodes[right].Dist < h.nodes[smallest].Dist {
			smallest = right
		}
		if smallest == i {
			break
		}
		h.nodes[i], h.nodes[smallest] = h.nodes[smallest], h.nodes[i]
		i = smallest
	}
}
