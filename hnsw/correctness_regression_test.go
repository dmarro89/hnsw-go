package hnsw

import (
	"strings"
	"testing"
)

func TestInsertRejectsNonContiguousIDs(t *testing.T) {
	h, err := NewHNSW(Config{
		M:              2,
		Mmax:           4,
		Mmax0:          4,
		EfConstruction: 32,
		MaxLevel:       2,
		DistanceFunc:   EuclideanDistance,
	})
	if err != nil {
		t.Fatalf("Failed to create HNSW: %v", err)
	}

	h.RandFunc = func() float64 { return 0.99 }

	h.Insert([]float32{0, 0}, 0)

	defer func() {
		r := recover()
		if r == nil {
			t.Fatal("expected insert with non-contiguous id to panic")
		}

		msg, ok := r.(string)
		if !ok {
			t.Fatalf("expected panic string, got %T", r)
		}
		if !strings.Contains(msg, "node id must be contiguous") {
			t.Fatalf("unexpected panic message: %q", msg)
		}
	}()

	h.Insert([]float32{1, 0}, 2)
}

func TestInsertLimitsNewNodeConnectionsToM(t *testing.T) {
	h, err := NewHNSW(Config{
		M:              2,
		Mmax:           4,
		Mmax0:          4,
		EfConstruction: 32,
		MaxLevel:       2,
		DistanceFunc:   EuclideanDistance,
	})
	if err != nil {
		t.Fatalf("Failed to create HNSW: %v", err)
	}

	h.RandFunc = func() float64 { return 0.99 }

	for i := 0; i < 6; i++ {
		h.Insert([]float32{float32(i), 0}, i)
	}

	last := h.Nodes[len(h.Nodes)-1]
	if got := len(last.Neighbors[0]); got > h.M {
		t.Fatalf("new node has %d level-0 neighbors, expected at most M=%d", got, h.M)
	}
}

func TestKNNSearchHandlesKGreaterThanNodeCount(t *testing.T) {
	h, err := NewHNSW(Config{
		M:              2,
		Mmax:           4,
		Mmax0:          4,
		EfConstruction: 32,
		MaxLevel:       2,
		DistanceFunc:   EuclideanDistance,
	})
	if err != nil {
		t.Fatalf("Failed to create HNSW: %v", err)
	}

	h.RandFunc = func() float64 { return 0.99 }

	h.Insert([]float32{0, 0}, 0)
	h.Insert([]float32{1, 0}, 1)

	results := h.KNN_Search([]float32{0, 0}, 5, 5)
	if len(results) != 2 {
		t.Fatalf("Expected all available nodes, got %d results", len(results))
	}
}

func TestInsertBatchAssignsContiguousIDs(t *testing.T) {
	h, err := NewHNSW(Config{
		M:              2,
		Mmax:           4,
		Mmax0:          4,
		EfConstruction: 32,
		MaxLevel:       2,
		DistanceFunc:   EuclideanDistance,
	})
	if err != nil {
		t.Fatalf("Failed to create HNSW: %v", err)
	}

	h.RandFunc = func() float64 { return 0.99 }
	h.InsertBatch([][]float32{
		{0, 0},
		{1, 0},
		{2, 0},
	})

	if len(h.Nodes) != 3 {
		t.Fatalf("Expected 3 inserted nodes, got %d", len(h.Nodes))
	}
	for i, node := range h.Nodes {
		if node.ID != i {
			t.Fatalf("Expected node ID %d, got %d", i, node.ID)
		}
	}
}
