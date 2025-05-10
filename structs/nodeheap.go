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
