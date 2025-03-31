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
				heap.Push(h, NewNodeHeap(item[0], int(item[1])))
			}

			if h.Len() != len(tt.items) {
				t.Errorf("heap size = %d, want %d", h.Len(), len(tt.items))
			}

			for i, want := range tt.expected {
				if h.Len() == 0 {
					t.Fatalf("heap empty, but expected more items")
				}
				item := heap.Pop(h).(*NodeHeap)
				if math.Abs(float64(item.Dist-want)) > 0 {
					t.Errorf("item %d = %f, want %f", i, item.Dist, want)
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
				heap.Push(h, NewNodeHeap(item[0], int(item[1])))
			}

			peek := h.Peek()
			if h.Len() > 0 {
				if math.Abs(float64(peek.Dist-tt.expected)) > 0 {
					t.Errorf("Peek() = %f, want %f", peek.Dist, tt.expected)
				}
			} else {
				if peek != nil {
					t.Errorf("Peek() on empty heap should return nil")
				}
			}
		})
	}
}

// TestMaxHeapOrdering verifies that MaxHeap maintains the correct order
// when elements with different distances are inserted
func TestMaxHeapOrdering(t *testing.T) {
	tests := []struct {
		name      string
		items     [][2]float32 // [distance, ID]
		wantOrder []float32    // Expected order of distances after pop
	}{
		{
			name: "distances in ascending order",
			items: [][2]float32{
				{0.1, 1},
				{0.2, 2},
				{0.5, 3},
				{1.0, 4},
				{5.0, 5},
			},
			wantOrder: []float32{5.0, 1.0, 0.5, 0.2, 0.1},
		},
		{
			name: "distances in descending order",
			items: [][2]float32{
				{5.0, 1},
				{1.0, 2},
				{0.5, 3},
				{0.2, 4},
				{0.1, 5},
			},
			wantOrder: []float32{5.0, 1.0, 0.5, 0.2, 0.1},
		},
		{
			name: "distances in random order",
			items: [][2]float32{
				{1.0, 1},
				{0.1, 2},
				{5.0, 3},
				{0.2, 4},
				{0.5, 5},
			},
			wantOrder: []float32{5.0, 1.0, 0.5, 0.2, 0.1},
		},
		{
			name: "equal distances",
			items: [][2]float32{
				{0.5, 1},
				{0.5, 2},
				{0.5, 3},
				{0.5, 4},
			},
			wantOrder: []float32{0.5, 0.5, 0.5, 0.5},
		},
		{
			name: "different IDs with same distance",
			items: [][2]float32{
				{0.5, 1000},
				{0.5, 1},
				{0.5, 9999},
				{0.5, 42},
			},
			wantOrder: []float32{0.5, 0.5, 0.5, 0.5},
		},
		{
			name: "very large distance values",
			items: [][2]float32{
				{999999.0, 1},
				{100000.0, 2},
				{10000.0, 3},
				{1000.0, 4},
			},
			wantOrder: []float32{999999.0, 100000.0, 10000.0, 1000.0},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			h := NewMaxHeap()

			// Insert all elements into the heap
			for _, item := range tt.items {
				encoded := NewNodeHeap(item[0], int(item[1]))
				heap.Push(h, encoded)
			}

			// Verify the correct number of elements
			if h.Len() != len(tt.items) {
				t.Errorf("MaxHeap.Len() = %v, want %v", h.Len(), len(tt.items))
			}

			// Extract elements one by one and verify the order
			gotOrder := make([]float32, 0, len(tt.items))
			gotIDs := make([]int, 0, len(tt.items))

			for h.Len() > 0 {
				item := heap.Pop(h).(*NodeHeap)
				gotOrder = append(gotOrder, item.Dist)
				gotIDs = append(gotIDs, item.Id)
			}

			// Verify that the order of distances matches the expected order
			if len(gotOrder) != len(tt.wantOrder) {
				t.Fatalf("Incorrect number of extracted elements: got %v, want %v", len(gotOrder), len(tt.wantOrder))
			}

			for i, want := range tt.wantOrder {
				if gotOrder[i] != want {
					t.Errorf("MaxHeap extracts distance at position %d = %v, want %v", i, gotOrder[i], want)
				}
			}

			// Print debug information
			t.Logf("Extracted distances: %v", gotOrder)
			t.Logf("Extracted IDs: %v", gotIDs)
		})
	}
}

// TestMaxHeapWithRealEncoding verifies that the encoding/decoding used by HNSW
// preserves the correct ordering of distances in the MaxHeap
func TestMaxHeapWithRealEncoding(t *testing.T) {
	items := [][2]float32{
		{0.1, 10},   // Smaller distance, small ID
		{1.0, 100},  // Medium distance, medium ID
		{5.0, 1000}, // Larger distance, large ID
	}

	h := NewMaxHeap()

	// Insert elements with real uint64 encoding
	for _, item := range items {
		encoded := NewNodeHeap(item[0], int(item[1]))
		t.Logf("Distance %.2f, ID %d -> encoded: %v", item[0], int(item[1]), encoded)
		heap.Push(h, encoded)
	}

	// Expected order is from largest to smallest (by distance)
	expectedDists := []float32{5.0, 1.0, 0.1}
	expectedIDs := []int{1000, 100, 10}

	// Extract and verify
	for i := 0; i < len(expectedDists); i++ {
		if h.Len() == 0 {
			t.Fatalf("MaxHeap empty before extracting all elements")
		}

		item := heap.Pop(h).(*NodeHeap)

		t.Logf("Pop %d: Got distance=%.2f, id=%d", i, item.Dist, item.Id)

		if item.Dist != expectedDists[i] {
			t.Errorf("Pop %d: distance = %v, want %v", i, item.Dist, expectedDists[i])
		}

		if item.Id != expectedIDs[i] {
			t.Errorf("Pop %d: id = %v, want %v", i, item.Id, expectedIDs[i])
		}
	}
}
