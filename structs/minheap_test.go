package structs

import (
	"container/heap"
	"math"
	"testing"
)

func TestMinHeap(t *testing.T) {
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
			expected: []float32{1.0, 2.0, 3.0},
		},
		{
			name: "duplicate distances",
			items: [][2]float32{
				{2.0, 1},
				{2.0, 2},
				{1.0, 3},
			},
			expected: []float32{1.0, 2.0, 2.0},
		},
		{
			name: "negative distances",
			items: [][2]float32{
				{-1.0, 1},
				{-3.0, 2},
				{-2.0, 3},
			},
			expected: []float32{-3.0, -2.0, -1.0},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			h := NewMinHeap()

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

func TestHeapItemEncoding(t *testing.T) {
	tests := []struct {
		dist     float32
		id       int
		wantDist float32
		wantID   int
	}{
		{1.5, 42, 1.5, 42},
		{-1.5, 100, -1.5, 100},
		{0.0, 0, 0.0, 0},
		{math.MaxFloat32, 1000000, math.MaxFloat32, 1000000},
	}

	for _, tt := range tests {
		encoded := EncodeHeapItem(tt.dist, tt.id)
		gotDist, gotID := DecodeHeapItem(encoded)

		if math.Abs(float64(gotDist-tt.wantDist)) > 0 {
			t.Errorf("distance = %f, want %f", gotDist, tt.wantDist)
		}
		if gotID != tt.wantID {
			t.Errorf("id = %d, want %d", gotID, tt.wantID)
		}
	}
}
