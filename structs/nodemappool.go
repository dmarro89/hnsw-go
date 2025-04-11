package structs

import "sync"

type NodeMapPool struct {
	pool sync.Pool
}

// NewNodeMapPool creates a new NodeMapPool
func NewNodeMapPool() *NodeMapPool {
	return &NodeMapPool{
		pool: sync.Pool{
			New: func() interface{} {
				return make(map[int]*NodeHeap)
			},
		},
	}
}

// Get returns a map from the pool
func (p *NodeMapPool) Get() map[int]*NodeHeap {
	return p.pool.Get().(map[int]*NodeHeap)
}

// Put returns a map to the pool
func (p *NodeMapPool) Put(m map[int]*NodeHeap) {
	// Pulizia della mappa prima di restituirla al pool
	for k := range m {
		delete(m, k)
	}
	p.pool.Put(m)
}
