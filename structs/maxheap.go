package structs

// MaxHeap keeps elements in descending order (largest on top).
type MaxHeap struct {
	nodes []*NodeHeap
}

// NewMaxHeap creates a new max-heap with initial capacity.
func NewMaxHeap() *MaxHeap {
	return &MaxHeap{
		nodes: make([]*NodeHeap, 0, 64),
	}
}

// Len returns the number of elements in the heap.
func (h *MaxHeap) Len() int {
	return len(h.nodes)
}

// Push adds a new element and restores the heap property.
func (h *MaxHeap) Push(n *NodeHeap) {
	h.nodes = append(h.nodes, n)
	h.siftUp(len(h.nodes) - 1)
}

// Pop removes and returns the element with the maximum value.
func (h *MaxHeap) Pop() *NodeHeap {
	if len(h.nodes) == 0 {
		return nil
	}
	max := h.nodes[0]
	lastIndex := len(h.nodes) - 1
	h.nodes[0] = h.nodes[lastIndex]
	h.nodes = h.nodes[:lastIndex]
	h.siftDown(0)
	return max
}

// Peek returns the maximum element without removing it.
func (h *MaxHeap) Peek() *NodeHeap {
	if len(h.nodes) == 0 {
		return nil
	}
	return h.nodes[0]
}

// Reset empties the heap while maintaining the underlying capacity.
func (h *MaxHeap) Reset() {
	h.nodes = h.nodes[:0]
}

// siftUp restores the heap property by moving up the tree.
func (h *MaxHeap) siftUp(i int) {
	for i > 0 {
		parent := (i - 1) / 2
		if h.nodes[i].Dist > h.nodes[parent].Dist {
			h.nodes[i], h.nodes[parent] = h.nodes[parent], h.nodes[i]
			i = parent
		} else {
			break
		}
	}
}

// siftDown restores the heap property by moving down the tree.
func (h *MaxHeap) siftDown(i int) {
	n := len(h.nodes)
	for {
		left := 2*i + 1
		right := 2*i + 2
		largest := i
		if left < n && h.nodes[left].Dist > h.nodes[largest].Dist {
			largest = left
		}
		if right < n && h.nodes[right].Dist > h.nodes[largest].Dist {
			largest = right
		}
		if largest == i {
			break
		}
		h.nodes[i], h.nodes[largest] = h.nodes[largest], h.nodes[i]
		i = largest
	}
}
