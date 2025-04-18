package structs

import "sync"

// NodeObjectPool is a pool of Node objects to reduce memory allocations
type NodeObjectPool struct {
	pool sync.Pool
	// Pool for slices of Node pointer slices
	neighborsPool map[int]*sync.Pool
}

// NewNodeObjectPool creates a new Node object pool
func NewNodeObjectPool() *NodeObjectPool {
	return &NodeObjectPool{
		pool: sync.Pool{
			New: func() interface{} {
				return &Node{}
			},
		},
		neighborsPool: make(map[int]*sync.Pool),
	}
}

// Get retrieves a node from the pool or creates a new one
func (p *NodeObjectPool) Get(id int, vector []float32, level, maxLevel, mMax, mMax0 int) *Node {
	node := p.pool.Get().(*Node)

	// Reset and initialize base fields
	node.ID = id
	node.Vector = vector
	node.Level = level

	// Get or create the Neighbors slice
	if node.Neighbors == nil || cap(node.Neighbors) < level+1 {
		// If there's no appropriate slice, create a new one
		node.Neighbors = make([][]*Node, level+1)
	} else {
		// Reuse the existing slice
		node.Neighbors = node.Neighbors[:level+1]
	}

	// Initialize inner slices
	for i := range node.Neighbors {
		// Get or create inner slice using appropriate size
		capacity := mMax
		if i == 0 {
			capacity = mMax0
		}

		if i < len(node.Neighbors) && cap(node.Neighbors[i]) >= capacity {
			// Reuse existing slice
			node.Neighbors[i] = node.Neighbors[i][:0]
		} else {
			// Create new slice
			node.Neighbors[i] = make([]*Node, 0, capacity)
		}
	}

	return node
}

// Put returns a node to the pool
func (p *NodeObjectPool) Put(node *Node) {
	// Clean references to prevent memory leaks
	// NOTE: Vector is not cleaned because it might be shared

	// Clean Neighbors but keep the slices
	for i := range node.Neighbors {
		for j := range node.Neighbors[i] {
			node.Neighbors[i][j] = nil
		}
		node.Neighbors[i] = node.Neighbors[i][:0]
	}

	p.pool.Put(node)
}
