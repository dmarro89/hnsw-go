package structs

import "math"

// MaxHeap represents a binary heap where the largest element is at the root.
// It stores distances and IDs encoded as uint64 values for efficient memory usage
// and comparison operations.
type MaxHeap []uint64

// NewMaxHeap creates a new MaxHeap with an initial capacity of 64 elements.
func NewMaxHeap() *MaxHeap {
	h := MaxHeap(make([]uint64, 0, 64))
	return &h
}

// Len returns the number of elements in the heap.
func (h MaxHeap) Len() int { return len(h) }

// Less reports whether the element with index i should sort before the element with index j.
// For MaxHeap, larger values have higher priority.
func (h MaxHeap) Less(i, j int) bool { return h[i] > h[j] }

// Swap exchanges the elements with indexes i and j.
func (h MaxHeap) Swap(i, j int) { h[i], h[j] = h[j], h[i] }

// Push adds x as element Len(). The complexity is O(log n) where n = h.Len().
func (h *MaxHeap) Push(x interface{}) {
	*h = append(*h, x.(uint64))
}

// Pop removes and returns the maximum element (according to Less) from the heap.
// The complexity is O(log n) where n = h.Len().
func (h *MaxHeap) Pop() interface{} {
	old := *h
	n := len(old)
	x := old[n-1]
	*h = old[0 : n-1]
	return x
}

// Reset clears the heap, maintaining the underlying array capacity.
func (h *MaxHeap) Reset() {
	*h = (*h)[:0]
}

// Peek returns the maximum element without removing it from the heap.
// If the heap is empty, returns MaxUint64.
func (h MaxHeap) Peek() uint64 {
	if len(h) == 0 {
		return math.MaxUint64
	}
	return h[0]
}
