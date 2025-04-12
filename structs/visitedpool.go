package structs

import "sync"

type VisitedPool struct {
	pool sync.Pool
}

// NewVisitedPool creates a new VisitedPool
func NewVisitedPool() *VisitedPool {
	return &VisitedPool{
		pool: sync.Pool{
			New: func() interface{} {
				return make(map[int]struct{})
			},
		},
	}
}

// Get returns a map from the pool
func (p *VisitedPool) Get() map[int]struct{} {
	return p.pool.Get().(map[int]struct{})
}

// Put returns a map to the pool
func (p *VisitedPool) Put(m map[int]struct{}) {
	for k := range m {
		delete(m, k)
	}
	p.pool.Put(m)
}
