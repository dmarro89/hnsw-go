package hnsw

import (
	"math/rand/v2"
	"reflect"
	"testing"
)

func TestBuildParallelMatchesInsertBatchWhenBatchSizeIsOne(t *testing.T) {
	cfg := Config{
		M:              8,
		Mmax:           16,
		Mmax0:          16,
		EfConstruction: 32,
		MaxLevel:       8,
		DistanceFunc:   EuclideanDistance,
	}

	serial, err := NewHNSW(cfg)
	if err != nil {
		t.Fatalf("Failed to create serial HNSW: %v", err)
	}
	parallel, err := NewHNSW(cfg)
	if err != nil {
		t.Fatalf("Failed to create parallel HNSW: %v", err)
	}

	serialRNG := rand.New(rand.NewPCG(42, 42))
	parallelRNG := rand.New(rand.NewPCG(42, 42))
	serial.RandFunc = serialRNG.Float64
	parallel.RandFunc = parallelRNG.Float64

	vectors := benchmarkVectors(128, 16, 99)

	serial.InsertBatch(vectors)
	if err := parallel.BuildParallel(vectors, BulkBuildConfig{
		Workers:        1,
		BatchSize:      1,
		EfConstruction: cfg.EfConstruction,
	}); err != nil {
		t.Fatalf("BuildParallel failed: %v", err)
	}

	if serial.EntryPoint == nil || parallel.EntryPoint == nil {
		t.Fatalf("expected non-nil entry points")
	}
	if len(serial.Nodes) != len(parallel.Nodes) {
		t.Fatalf("node count mismatch: serial=%d parallel=%d", len(serial.Nodes), len(parallel.Nodes))
	}
	if serial.EntryPoint == nil || parallel.EntryPoint == nil {
		t.Fatalf("expected non-nil entry points")
	}
	if serial.EntryPoint.ID != parallel.EntryPoint.ID || serial.EntryPoint.Level != parallel.EntryPoint.Level {
		t.Fatalf("entry point mismatch: serial=(id=%d level=%d) parallel=(id=%d level=%d)",
			serial.EntryPoint.ID, serial.EntryPoint.Level, parallel.EntryPoint.ID, parallel.EntryPoint.Level)
	}

	for i := range serial.Nodes {
		sn := serial.Nodes[i]
		pn := parallel.Nodes[i]
		if sn.ID != pn.ID || sn.Level != pn.Level {
			t.Fatalf("node %d mismatch: serial=(id=%d level=%d) parallel=(id=%d level=%d)",
				i, sn.ID, sn.Level, pn.ID, pn.Level)
		}
		if !reflect.DeepEqual(sn.Vector, pn.Vector) {
			t.Fatalf("vector mismatch at node %d", i)
		}
		if !reflect.DeepEqual(sn.Neighbors, pn.Neighbors) {
			t.Fatalf("neighbors mismatch at node %d: serial=%v parallel=%v", i, sn.Neighbors, pn.Neighbors)
		}
	}
}

func TestBuildParallelPreservesGraphInvariants(t *testing.T) {
	h, err := NewHNSW(Config{
		M:              8,
		Mmax:           16,
		Mmax0:          16,
		EfConstruction: 32,
		MaxLevel:       8,
		DistanceFunc:   EuclideanDistance,
	})
	if err != nil {
		t.Fatalf("Failed to create HNSW: %v", err)
	}

	rng := rand.New(rand.NewPCG(7, 7))
	h.RandFunc = rng.Float64
	vectors := benchmarkVectors(256, 32, 123)

	if err := h.BuildParallel(vectors, BulkBuildConfig{
		Workers:        4,
		BatchSize:      32,
		EfConstruction: 32,
	}); err != nil {
		t.Fatalf("BuildParallel failed: %v", err)
	}

	if len(h.Nodes) != len(vectors) {
		t.Fatalf("expected %d nodes, got %d", len(vectors), len(h.Nodes))
	}

	assertGraphInvariants(t, h)
}

func assertGraphInvariants(t *testing.T, h *HNSW) {
	t.Helper()

	for _, node := range h.Nodes {
		for level := range node.Neighbors {
			maxAllowed := h.Mmax
			if level == 0 {
				maxAllowed = h.Mmax0
			}
			if len(node.Neighbors[level]) > maxAllowed {
				t.Fatalf("node %d level %d has %d neighbors, max allowed %d", node.ID, level, len(node.Neighbors[level]), maxAllowed)
			}
			for _, neighborID := range node.Neighbors[level] {
				if neighborID < 0 || neighborID >= len(h.Nodes) {
					t.Fatalf("node %d level %d has invalid neighbor id %d", node.ID, level, neighborID)
				}
				neighbor := h.Nodes[neighborID]
				if level >= len(neighbor.Neighbors) || !containsInt(neighbor.Neighbors[level], node.ID) {
					t.Fatalf("connection from %d to %d at level %d is not bidirectional", node.ID, neighborID, level)
				}
			}
		}
	}
}

func containsInt(ids []int, target int) bool {
	for _, id := range ids {
		if id == target {
			return true
		}
	}
	return false
}
