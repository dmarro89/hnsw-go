package structs

import "sync"

type NodePool struct {
	pool sync.Pool
}

// NewNodePool creates a new NodePool
func NewNodePool(capacity int) *NodePool {
	return &NodePool{
		pool: sync.Pool{
			New: func() interface{} {
				return make([]*Node, 0, capacity)
			},
		},
	}
}

// Get returns a slice from the pool
func (p *NodePool) Get() []*Node {
	return p.pool.Get().([]*Node)
}

// Put returns a slice to the pool
func (p *NodePool) Put(m []*Node) {
	// Pulizia della mappa prima di restituirla al pool
	for i := range m {
		m[i] = nil
	}
	p.pool.Put(m)
}
