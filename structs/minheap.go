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

// MinHeap represents a binary heap where the smallest element is at the root.
// It stores distances and IDs encoded as uint64 values for efficient memory usage
// and comparison operations.
type MinHeap []*NodeHeap

// NewMinHeap creates a new MinHeap with an initial capacity of 64 elements.
func NewMinHeap() *MinHeap {
	return &MinHeap{}
}

// Len returns the number of elements in the heap.
func (h MinHeap) Len() int { return len(h) }

// Less reports whether the element with index i should sort before the element with index j.
// For MinHeap, smaller values have higher priority.
func (h MinHeap) Less(i, j int) bool { return h[i].Dist < h[j].Dist }

// Swap exchanges the elements with indexes i and j.
func (h MinHeap) Swap(i, j int) { h[i], h[j] = h[j], h[i] }

// Push adds x as element Len(). The complexity is O(log n) where n = h.Len().
func (h *MinHeap) Push(x interface{}) {
	*h = append(*h, x.(*NodeHeap))
}

// Pop removes and returns the minimum element (according to Less) from the heap.
// The complexity is O(log n) awhere n = h.Len().
func (h *MinHeap) Pop() interface{} {
	old := *h
	n := len(old)
	x := old[n-1]
	*h = old[0 : n-1]
	return x
}

// Reset clears the heap, maintaining the underlying array capacity.
func (h *MinHeap) Reset() {
	*h = (*h)[:0]
}
