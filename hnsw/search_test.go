package hnsw

import (
	"container/heap"
	"testing"

	"dmarro89.github.com/hnsw-go/structs"
)

// TestKNNSearchEmptyGraph verifies that KNN_Search returns nil when
// searching in an empty graph
func TestKNNSearchEmptyGraph(t *testing.T) {
	config := Config{
		M:              10,
		Mmax:           10,
		Mmax0:          10,
		EfConstruction: 16,
		MaxLevel:       5,
		DistanceFunc:   EuclideanDistance,
	}

	h, err := NewHNSW(config)
	if err != nil {
		t.Fatalf("Failed to create HNSW: %v", err)
	}

	query := []float32{1.0, 2.0}
	results := h.KNN_Search(query, 5, 10)

	if results != nil {
		t.Errorf("Expected nil results for empty graph, got %v", results)
	}
}

// TestKNNSearchSingleElement verifies correct behavior with only one element
func TestKNNSearchSingleElement(t *testing.T) {
	config := Config{
		M:              10,
		Mmax:           10,
		Mmax0:          10,
		EfConstruction: 16,
		MaxLevel:       5,
		DistanceFunc:   EuclideanDistance,
	}

	h, err := NewHNSW(config)
	if err != nil {
		t.Fatalf("Failed to create HNSW: %v", err)
	}

	// Insert one element
	vector := []float32{1.0, 2.0}
	h.Insert(vector, 0)

	// Search for something - should always return the only element
	query := []float32{5.0, 5.0}
	results := h.KNN_Search(query, 1, 1)

	if len(results) != 1 {
		t.Fatalf("Expected 1 result, got %d", len(results))
	}

	if results[0].ID != 0 {
		t.Errorf("Expected node ID 0, got %d", results[0].ID)
	}
}

// TestKNNSearchMultiDimensional verifies search works with higher dimensions
// func TestKNNSearchMultiDimensional(t *testing.T) {
// 	config := Config{
// 		M:              5,
// 		Mmax:           5,
// 		Mmax0:          5,
// 		EfConstruction: 16,
// 		MaxLevel:       3,
// 		DistanceFunc:   EuclideanDistance,
// 	}

// 	h, err := NewHNSW(config)
// 	if err != nil {
// 		t.Fatalf("Failed to create HNSW: %v", err)
// 	}

// 	// 5-dimensional vectors
// 	h.Insert([]float32{0.0, 0.0, 0.0, 0.0, 0.0}, 0)
// 	h.Insert([]float32{1.0, 1.0, 1.0, 1.0, 1.0}, 1)
// 	h.Insert([]float32{2.0, 2.0, 2.0, 2.0, 2.0}, 2)

// 	// Search for closest to {1,1,1,1,1}
// 	results := h.KNN_Search([]float32{1.0, 1.0, 1.0, 1.0, 1.0}, 3, 3)

// 	if len(results) != 3 {
// 		t.Fatalf("Expected 3 results, got %d", len(results))
// 	}

// 	expectedOrder := []int{0, 2, 1} // Ordered by distance to query
// 	for i, id := range expectedOrder {
// 		if results[i].ID != id {
// 			t.Errorf("Expected result %d to be node %d, got %d",
// 				i, id, results[i].ID)
// 		}
// 	}
// }

// TestSearchLayerBasic verifies the basic functionality of searchLayer
// func TestSearchLayerBasic(t *testing.T) {
// 	config := Config{
// 		M:              5,
// 		Mmax:           5,
// 		Mmax0:          5,
// 		EfConstruction: 16,
// 		MaxLevel:       3,
// 		DistanceFunc:   EuclideanDistance,
// 	}

// 	h, err := NewHNSW(config)
// 	if err != nil {
// 		t.Fatalf("Failed to create HNSW: %v", err)
// 	}

// 	// Create a simple graph with known structure
// 	n0 := structs.NewNode(0, []float32{0.0, 0.0}, 0, 3, 5)
// 	n1 := structs.NewNode(1, []float32{1.0, 0.0}, 0, 3, 5)
// 	n2 := structs.NewNode(2, []float32{2.0, 0.0}, 0, 3, 5)

// 	// Connect them at level 0
// 	n0.Neighbors[0] = []*structs.Node{n1}
// 	n1.Neighbors[0] = []*structs.Node{n0, n2}
// 	n2.Neighbors[0] = []*structs.Node{n1}

// 	// Add nodes to graph
// 	h.Nodes = []*structs.Node{n0, n1, n2}
// 	h.EntryPoint = n0

// 	// Search from n0 with ef=2
// 	query := []float32{0.5, 0.0}
// 	nearest := h.searchLayer(query, n0, 2, 0)

// 	// Should find n0 and n1 as the 2 closest
// 	if nearest.Len() != 2 {
// 		t.Fatalf("Expected 2 results, got %d", nearest.Len())
// 	}

// 	// Convert to array to check order
// 	results := make([]int, 0, nearest.Len())
// 	for nearest.Len() > 0 {
// 		item := heap.Pop(nearest).(uint64)
// 		_, id := structs.DecodeHeapItem(item)
// 		results = append(results, id)
// 	}

// 	// Results should be [1, 0] (furthest first due to MaxHeap)
// 	expectedOrder := []int{1, 0}
// 	for i, id := range expectedOrder {
// 		if results[i] != id {
// 			t.Errorf("Expected result %d to be node %d, got %d",
// 				i, id, results[i])
// 		}
// 	}
// }

// TestSimpleSelectNeighbors verifies neighbor selection logic
func TestSimpleSelectNeighbors(t *testing.T) {
	config := Config{
		M:              5,
		Mmax:           5,
		Mmax0:          5,
		EfConstruction: 16,
		MaxLevel:       3,
		DistanceFunc:   EuclideanDistance,
	}

	h, err := NewHNSW(config)
	if err != nil {
		t.Fatalf("Failed to create HNSW: %v", err)
	}

	// Create a graph with 5 nodes
	for i := 0; i < 5; i++ {
		h.Insert([]float32{float32(i), 0.0}, i)
	}

	// Create a minheap with distances to node 2
	candidates := h.heapPool.GetMinHeap()

	// Add items to the heap with specific distances
	items := []struct {
		distance float32
		id       int
	}{
		{2.0, 0}, // 2 units away
		{1.0, 1}, // 1 unit away
		{0.0, 2}, // 0 units away (self)
		{1.0, 3}, // 1 unit away
		{2.0, 4}, // 2 units away
	}

	for _, item := range items {
		heap.Push(candidates, structs.NewNodeHeap(item.distance, item.id))
	}

	// Select top 3 neighbors
	neighbors := h.simpleSelectNeighbors(candidates, 3)

	// Should get the 3 closest: ids 2, 1, 3 (in some order)
	if len(neighbors) != 3 {
		t.Fatalf("Expected 3 neighbors, got %d", len(neighbors))
	}

	// Check all expected IDs are present
	expectedIDs := map[int]bool{1: true, 2: true, 3: true}
	for _, n := range neighbors {
		if !expectedIDs[n.ID] {
			t.Errorf("Unexpected neighbor ID: %d", n.ID)
		}
		delete(expectedIDs, n.ID) // Remove to check duplicates
	}

	if len(expectedIDs) != 0 {
		t.Errorf("Missing some expected neighbors: %v", expectedIDs)
	}
}

// TestSearchWithDifferentEfValues verifies the effect of ef parameter on search quality
func TestSearchWithDifferentEfValues(t *testing.T) {
	config := Config{
		M:              5,
		Mmax:           5,
		Mmax0:          5,
		EfConstruction: 16,
		MaxLevel:       3,
		DistanceFunc:   EuclideanDistance,
	}

	h, err := NewHNSW(config)
	if err != nil {
		t.Fatalf("Failed to create HNSW: %v", err)
	}

	// Insert 20 elements in a 2D grid
	id := 0
	for x := 0; x < 4; x++ {
		for y := 0; y < 5; y++ {
			h.Insert([]float32{float32(x), float32(y)}, id)
			id++
		}
	}

	// Query point
	query := []float32{1.5, 2.5}

	// Search with different ef values
	efValues := []int{1, 3, 10, 20}

	var previousResults []*structs.Node

	for _, ef := range efValues {
		results := h.KNN_Search(query, 5, ef)

		if len(results) != 5 {
			t.Errorf("Expected 5 results with ef=%d, got %d", ef, len(results))
			continue
		}

		// For ef > 1, results should improve or stay the same
		if previousResults != nil {
			lastDist := h.DistanceFunc(query, previousResults[len(previousResults)-1].Vector)
			currentDist := h.DistanceFunc(query, results[len(results)-1].Vector)

			// With higher ef, the furthest neighbor should be the same or closer
			if currentDist > lastDist {
				t.Logf("Warning: ef=%d gave worse results than previous ef", ef)
			}
		}

		previousResults = results
	}
}
