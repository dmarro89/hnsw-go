package structs

import "sync"

type NodeHeap struct {
	Dist float32
	Id   int
}

type NodeHeapPool struct {
	pool sync.Pool
}

func NewNodeHeapPool() *NodeHeapPool {
	return &NodeHeapPool{
		pool: sync.Pool{
			New: func() interface{} {
				return &NodeHeap{}
			},
		},
	}
}

func (p *NodeHeapPool) Get(dist float32, id int) *NodeHeap {
	nh := p.pool.Get().(*NodeHeap)
	nh.Dist = dist
	nh.Id = id
	return nh
}

func (p *NodeHeapPool) Put(nh *NodeHeap) {
	nh.Dist = 0
	nh.Id = -1
	p.pool.Put(nh)
}

func NewNodeHeap(dist float32, id int) *NodeHeap {
	return &NodeHeap{
		Dist: dist,
		Id:   id,
	}
}
