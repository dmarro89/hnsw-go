package hnsw

import (
	"math/rand/v2"
	"reflect"
	"testing"
)

func TestBuildParallelIsDeterministicAcrossBatches(t *testing.T) {
	cfg := Config{
		M:              8,
		Mmax:           16,
		Mmax0:          16,
		EfConstruction: 32,
		MaxLevel:       8,
		DistanceFunc:   EuclideanDistance,
	}
	buildCfg := BulkBuildConfig{
		Workers:        4,
		BatchSize:      32,
		EfConstruction: 32,
	}
	vectors := benchmarkVectors(512, 32, 2026)

	build := func() *HNSW {
		index, err := NewHNSW(cfg)
		if err != nil {
			t.Fatalf("NewHNSW failed: %v", err)
		}
		levelRNG := rand.New(rand.NewPCG(77, 77))
		index.RandFunc = levelRNG.Float64
		if err := index.BuildParallel(vectors, buildCfg); err != nil {
			t.Fatalf("BuildParallel failed: %v", err)
		}
		return index
	}

	first := build()
	second := build()

	if first.EntryPoint.ID != second.EntryPoint.ID || first.EntryPoint.Level != second.EntryPoint.Level {
		t.Fatalf("entry point mismatch: first=(%d,%d) second=(%d,%d)", first.EntryPoint.ID, first.EntryPoint.Level, second.EntryPoint.ID, second.EntryPoint.Level)
	}
	if len(first.Nodes) != len(second.Nodes) {
		t.Fatalf("node count mismatch: first=%d second=%d", len(first.Nodes), len(second.Nodes))
	}

	for i := range first.Nodes {
		left := first.Nodes[i]
		right := second.Nodes[i]
		if left.ID != right.ID || left.Level != right.Level {
			t.Fatalf("node %d metadata mismatch", i)
		}
		if !reflect.DeepEqual(left.Neighbors, right.Neighbors) {
			t.Fatalf("node %d neighbor mismatch: first=%v second=%v", i, left.Neighbors, right.Neighbors)
		}
	}
}
