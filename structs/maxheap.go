package structs

// MaxHeap keeps elements in descending order (largest on top).
type MaxHeap struct {
	nodes []NodeHeap
}

func NewMaxHeap() *MaxHeap { return &MaxHeap{nodes: make([]NodeHeap, 0, 64)} }
func (h *MaxHeap) Len() int { return len(h.nodes) }
func (h *MaxHeap) Push(n NodeHeap) { h.nodes = append(h.nodes, n); h.siftUp(len(h.nodes)-1) }
func (h *MaxHeap) Pop() NodeHeap {
	if len(h.nodes)==0 { return NodeHeap{} }
	max := h.nodes[0]
	last := len(h.nodes)-1
	if last == 0 { h.nodes = h.nodes[:0]; return max }
	h.nodes[0] = h.nodes[last]
	h.nodes = h.nodes[:last]
	h.siftDown(0)
	return max
}
func (h *MaxHeap) Peek() *NodeHeap { if len(h.nodes)==0{return nil}; return &h.nodes[0] }
func (h *MaxHeap) Reset() { h.nodes=h.nodes[:0] }

func (h *MaxHeap) siftUp(i int) {
	item := h.nodes[i]
	for i > 0 {
		parent := (i-1)/2
		if item.Dist <= h.nodes[parent].Dist { break }
		h.nodes[i] = h.nodes[parent]
		i = parent
	}
	h.nodes[i] = item
}

func (h *MaxHeap) siftDown(i int) {
	n := len(h.nodes)
	if i >= n { return }
	item := h.nodes[i]
	for {
		left := 2*i+1
		if left >= n { break }
		child := left
		right := left+1
		if right < n && h.nodes[right].Dist > h.nodes[left].Dist { child=right }
		if h.nodes[child].Dist <= item.Dist { break }
		h.nodes[i] = h.nodes[child]
		i = child
	}
	h.nodes[i] = item
}
