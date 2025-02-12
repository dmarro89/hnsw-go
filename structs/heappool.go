package structs

import (
	"sync"
)

type HeapPoolManager struct {
	minHeapPool sync.Pool
	maxHeapPool sync.Pool
}

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

func (p *HeapPoolManager) GetMinHeap() *MinHeap {
	heap := p.minHeapPool.Get().(*MinHeap)
	heap.Reset()
	return heap
}

func (p *HeapPoolManager) PutMinHeap(heap *MinHeap) {
	p.minHeapPool.Put(heap)
}

func (p *HeapPoolManager) GetMaxHeap() *MaxHeap {
	heap := p.maxHeapPool.Get().(*MaxHeap)
	heap.Reset()
	return heap
}

func (p *HeapPoolManager) PutMaxHeap(heap *MaxHeap) {
	p.maxHeapPool.Put(heap)
}
