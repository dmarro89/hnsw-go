package structs

import (
	"container/heap"
	"math"
	"testing"
)

func TestMaxHeap(t *testing.T) {
	tests := []struct {
		name     string
		items    [][2]float32
		expected []float32
	}{
		{
			name: "basic ordering",
			items: [][2]float32{
				{3.0, 1},
				{1.0, 2},
				{2.0, 3},
			},
			expected: []float32{3.0, 2.0, 1.0},
		},
		{
			name: "duplicate distances",
			items: [][2]float32{
				{2.0, 1},
				{2.0, 2},
				{1.0, 3},
			},
			expected: []float32{2.0, 2.0, 1.0},
		},
		{
			name: "negative distances",
			items: [][2]float32{
				{-1.0, 1},
				{-3.0, 2},
				{-2.0, 3},
			},
			expected: []float32{-1.0, -2.0, -3.0},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			h := NewMaxHeap()

			for _, item := range tt.items {
				heap.Push(h, EncodeHeapItem(item[0], int(item[1])))
			}

			if h.Len() != len(tt.items) {
				t.Errorf("heap size = %d, want %d", h.Len(), len(tt.items))
			}

			for i, want := range tt.expected {
				if h.Len() == 0 {
					t.Fatalf("heap empty, but expected more items")
				}
				item := heap.Pop(h).(uint64)
				dist, _ := DecodeHeapItem(item)
				if math.Abs(float64(dist-want)) > 0 {
					t.Errorf("item %d = %f, want %f", i, dist, want)
				}
			}
		})
	}
}

func TestMaxHeapPeek(t *testing.T) {
	tests := []struct {
		name     string
		items    [][2]float32
		expected float32
	}{
		{
			name:     "empty heap",
			items:    [][2]float32{},
			expected: math.Float32frombits(math.MaxUint32),
		},
		{
			name: "single item",
			items: [][2]float32{
				{3.0, 1},
			},
			expected: 3.0,
		},
		{
			name: "multiple items",
			items: [][2]float32{
				{3.0, 1},
				{1.0, 2},
				{2.0, 3},
			},
			expected: 3.0,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			h := NewMaxHeap()

			for _, item := range tt.items {
				heap.Push(h, EncodeHeapItem(item[0], int(item[1])))
			}

			peek := h.Peek()
			if h.Len() > 0 {
				dist, _ := DecodeHeapItem(peek)
				if math.Abs(float64(dist-tt.expected)) > 0 {
					t.Errorf("Peek() = %f, want %f", dist, tt.expected)
				}
			} else {
				if peek != math.MaxUint64 {
					t.Errorf("Peek() on empty heap = %d, want maxUint64", peek)
				}
			}
		})
	}
}
