package structs

import (
	"sync"
)

// HeapPoolManager gestisce il pool di heap per l'HNSW
type HeapPoolManager struct {
	minHeapPool sync.Pool
	maxHeapPool sync.Pool
}

// NewHeapPoolManager crea una nuova istanza del pool manager
func NewHeapPoolManager() *HeapPoolManager {
	return &HeapPoolManager{
		minHeapPool: sync.Pool{
			New: func() interface{} {
				return NewMinHeap()
			},
		},
		maxHeapPool: sync.Pool{
			New: func() interface{} {
				return NewMaxHeap()
			},
		},
	}
}

// GetMinHeap ottiene un MinHeap dal pool
func (p *HeapPoolManager) GetMinHeap() *MinHeap {
	heap := p.minHeapPool.Get().(*MinHeap)
	heap.Reset()
	return heap
}

// PutMinHeap rimette un MinHeap nel pool
func (p *HeapPoolManager) PutMinHeap(heap *MinHeap) {
	p.minHeapPool.Put(heap)
}

// GetMaxHeap ottiene un MaxHeap dal pool
func (p *HeapPoolManager) GetMaxHeap() *MaxHeap {
	heap := p.maxHeapPool.Get().(*MaxHeap)
	heap.Reset()
	return heap
}

// PutMaxHeap rimette un MaxHeap nel pool
func (p *HeapPoolManager) PutMaxHeap(heap *MaxHeap) {
	p.maxHeapPool.Put(heap)
}
