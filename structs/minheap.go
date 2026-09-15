package structs

// MinHeap keeps elements in ascending order (smallest on top).
type MinHeap struct { nodes []NodeHeap }
func NewMinHeap() *MinHeap { return &MinHeap{nodes: make([]NodeHeap,0,64)} }
func (h *MinHeap) Len() int { return len(h.nodes) }
func (h *MinHeap) Push(n NodeHeap) { h.nodes=append(h.nodes,n); h.siftUp(len(h.nodes)-1) }
func (h *MinHeap) Pop() NodeHeap {
	if len(h.nodes)==0 { return NodeHeap{} }
	min:=h.nodes[0]
	last:=len(h.nodes)-1
	if last==0 { h.nodes=h.nodes[:0]; return min }
	h.nodes[0]=h.nodes[last]
	h.nodes=h.nodes[:last]
	h.siftDown(0)
	return min
}
func (h *MinHeap) Peek() *NodeHeap { if len(h.nodes)==0{return nil}; return &h.nodes[0] }
func (h *MinHeap) Reset(){ h.nodes=h.nodes[:0] }

func (h *MinHeap) siftUp(i int) {
	item:=h.nodes[i]
	for i>0 {
		parent:=(i-1)/2
		if item.Dist >= h.nodes[parent].Dist { break }
		h.nodes[i]=h.nodes[parent]
		i=parent
	}
	h.nodes[i]=item
}

func (h *MinHeap) siftDown(i int) {
	n:=len(h.nodes)
	if i>=n { return }
	item:=h.nodes[i]
	for {
		left:=2*i+1
		if left>=n { break }
		child:=left
		right:=left+1
		if right<n && h.nodes[right].Dist < h.nodes[left].Dist { child=right }
		if h.nodes[child].Dist >= item.Dist { break }
		h.nodes[i]=h.nodes[child]
		i=child
	}
	h.nodes[i]=item
}
