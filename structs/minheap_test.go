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

// TestMinHeapOrdering verifies that MinHeap maintains the correct order
// when elements with different distances are inserted
func TestMinHeapOrdering(t *testing.T) {
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
			wantOrder: []float32{0.1, 0.2, 0.5, 1.0, 5.0},
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
			wantOrder: []float32{0.1, 0.2, 0.5, 1.0, 5.0},
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
			wantOrder: []float32{0.1, 0.2, 0.5, 1.0, 5.0},
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
			wantOrder: []float32{1000.0, 10000.0, 100000.0, 999999.0},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			h := NewMinHeap()

			// Insert all elements into the heap
			for _, item := range tt.items {
				encoded := EncodeHeapItem(item[0], int(item[1]))
				heap.Push(h, encoded)
			}

			// Verify the correct number of elements
			if h.Len() != len(tt.items) {
				t.Errorf("MinHeap.Len() = %v, want %v", h.Len(), len(tt.items))
			}

			// Extract elements one by one and verify the order
			gotOrder := make([]float32, 0, len(tt.items))
			gotIDs := make([]int, 0, len(tt.items))

			for h.Len() > 0 {
				item := heap.Pop(h).(uint64)
				dist, id := DecodeHeapItem(item)
				gotOrder = append(gotOrder, dist)
				gotIDs = append(gotIDs, id)
			}

			// Verify that the order of distances matches the expected order
			if len(gotOrder) != len(tt.wantOrder) {
				t.Fatalf("Incorrect number of extracted elements: got %v, want %v", len(gotOrder), len(tt.wantOrder))
			}

			for i, want := range tt.wantOrder {
				if gotOrder[i] != want {
					t.Errorf("MinHeap extracts distance at position %d = %v, want %v", i, gotOrder[i], want)
				}
			}

			// Print debug information
			t.Logf("Extracted distances: %v", gotOrder)
			t.Logf("Extracted IDs: %v", gotIDs)
		})
	}
}

// TestMinHeapWithRealEncoding verifies that the encoding/decoding used by HNSW
// preserves the correct ordering of distances in the MinHeap
func TestMinHeapWithRealEncoding(t *testing.T) {
	items := [][2]float32{
		{5.0, 10},   // Larger distance, small ID
		{1.0, 100},  // Medium distance, medium ID
		{0.1, 1000}, // Smaller distance, large ID
	}

	h := NewMinHeap()

	// Insert elements with real uint64 encoding
	for _, item := range items {
		encoded := EncodeHeapItem(item[0], int(item[1]))
		t.Logf("Distance %.2f, ID %d -> encoded: %d", item[0], int(item[1]), encoded)
		heap.Push(h, encoded)
	}

	// Expected order is from smallest to largest (by distance)
	expectedDists := []float32{0.1, 1.0, 5.0}
	expectedIDs := []int{1000, 100, 10}

	// Extract and verify
	for i := 0; i < len(expectedDists); i++ {
		if h.Len() == 0 {
			t.Fatalf("MinHeap empty before extracting all elements")
		}

		item := heap.Pop(h).(uint64)
		dist, id := DecodeHeapItem(item)

		t.Logf("Pop %d: Got distance=%.2f, id=%d", i, dist, id)

		if dist != expectedDists[i] {
			t.Errorf("Pop %d: distance = %v, want %v", i, dist, expectedDists[i])
		}

		if id != expectedIDs[i] {
			t.Errorf("Pop %d: id = %v, want %v", i, id, expectedIDs[i])
		}
	}
}
