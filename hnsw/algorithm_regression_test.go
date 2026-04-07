package hnsw

import (
	"math"
	"testing"

	"dmarro89.github.com/hnsw-go/structs"
)

func TestNewHNSWRejectsMGreaterThanMmax(t *testing.T) {
	_, err := NewHNSW(Config{
		M:              5,
		Mmax:           4,
		Mmax0:          6,
		EfConstruction: 16,
		MaxLevel:       2,
		DistanceFunc:   EuclideanDistance,
	})
	if err == nil {
		t.Fatal("expected config with M > Mmax to be rejected")
	}
}

func TestNewHNSWRejectsMGreaterThanMmax0(t *testing.T) {
	_, err := NewHNSW(Config{
		M:              5,
		Mmax:           6,
		Mmax0:          4,
		EfConstruction: 16,
		MaxLevel:       2,
		DistanceFunc:   EuclideanDistance,
	})
	if err == nil {
		t.Fatal("expected config with M > Mmax0 to be rejected")
	}
}

func TestRandomLevelFollowsExponentialDistributionWithoutArtificialCap(t *testing.T) {
	h, err := NewHNSW(Config{
		M:              2,
		Mmax:           4,
		Mmax0:          4,
		EfConstruction: 16,
		MaxLevel:       3,
		DistanceFunc:   EuclideanDistance,
	})
	if err != nil {
		t.Fatalf("Failed to create HNSW: %v", err)
	}

	h.RandFunc = func() float64 { return 1e-9 }

	expected := int(-math.Log(1e-9) * h.mL)
	if expected <= h.MaxLevel {
		t.Fatalf("test setup invalid: expected level %d should exceed MaxLevel %d", expected, h.MaxLevel)
	}

	level := h.RandomLevel()
	if level != expected {
		t.Fatalf("Expected uncapped level %d from exponential distribution, got %d", expected, level)
	}
}

func TestKNNSearchGreedyDescentChoosesBestUpperLayerNeighbor(t *testing.T) {
	h, err := NewHNSW(Config{
		M:              2,
		Mmax:           4,
		Mmax0:          4,
		EfConstruction: 16,
		MaxLevel:       2,
		DistanceFunc:   EuclideanDistance,
	})
	if err != nil {
		t.Fatalf("Failed to create HNSW: %v", err)
	}

	n0 := structs.NewNode(0, []float32{100, 0}, 1, h.MaxLevel, h.Mmax, h.Mmax0)
	n1 := structs.NewNode(1, []float32{50, 0}, 1, h.MaxLevel, h.Mmax, h.Mmax0)
	n2 := structs.NewNode(2, []float32{20, 0}, 1, h.MaxLevel, h.Mmax, h.Mmax0)
	n3 := structs.NewNode(3, []float32{0, 0}, 0, h.MaxLevel, h.Mmax, h.Mmax0)

	// At the upper layer, both neighbors improve over the entry point.
	// HNSW greedy descent with ef=1 should move to the best improving neighbor (node 2),
	// not stop at the first improvement in adjacency order (node 1).
	n0.Neighbors[1] = []int{1, 2}

	// At layer 0 only node 2 can reach the true nearest neighbor.
	n2.Neighbors[0] = []int{3}

	h.Nodes = []*structs.Node{n0, n1, n2, n3}
	h.EntryPoint = n0

	results := h.KNN_Search([]float32{0, 0}, 1, 1)
	if len(results) != 1 {
		t.Fatalf("Expected 1 result, got %d", len(results))
	}
	if results[0] != 3 {
		t.Fatalf("Expected search to descend through node 2 and return node 3, got %d", results[0])
	}
}

func TestInsertCarriesMultipleEnterPointsToLowerLayers(t *testing.T) {
	h, err := NewHNSW(Config{
		M:              2,
		Mmax:           4,
		Mmax0:          4,
		EfConstruction: 2,
		MaxLevel:       3,
		DistanceFunc:   EuclideanDistance,
	})
	if err != nil {
		t.Fatalf("Failed to create HNSW: %v", err)
	}

	n0 := structs.NewNode(0, []float32{100, 0}, 2, h.MaxLevel, h.Mmax, h.Mmax0)
	n1 := structs.NewNode(1, []float32{5, 0}, 1, h.MaxLevel, h.Mmax, h.Mmax0)
	n2 := structs.NewNode(2, []float32{10, 0}, 1, h.MaxLevel, h.Mmax, h.Mmax0)
	n3 := structs.NewNode(3, []float32{0, 0}, 0, h.MaxLevel, h.Mmax, h.Mmax0)

	// Layer 2 is trivial; the insertion reaches layer 1 from the entry point.
	// At layer 1 the search should keep both node 1 and node 2 as enter points for layer 0.
	n0.Neighbors[1] = []int{1, 2}

	// Only the second-best layer-1 candidate can reach the true nearest base-layer node.
	n2.Neighbors[0] = []int{3}

	h.Nodes = []*structs.Node{n0, n1, n2, n3}
	h.EntryPoint = n0
	h.RandFunc = func() float64 { return 0.5 } // level 1 when M=2

	h.Insert([]float32{0, 0}, 4)

	inserted := h.Nodes[4]
	found := false
	for _, neighborID := range inserted.Neighbors[0] {
		if neighborID == 3 {
			found = true
			break
		}
	}

	if !found {
		t.Fatalf("Expected insertion to keep multiple enter points and discover node 3 at layer 0, got neighbors %v", inserted.Neighbors[0])
	}
}
