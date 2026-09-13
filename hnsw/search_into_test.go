package hnsw

import (
	"math/rand/v2"
	"reflect"
	"testing"
)

func TestSearchIntoMatchesKNNSearchAndReusesBuffer(t *testing.T) {
	index, err := NewHNSW(Config{
		M:              16,
		Mmax:           32,
		Mmax0:          64,
		EfConstruction: 64,
		MaxLevel:       16,
		DistanceFunc:   EuclideanDistance,
	})
	if err != nil {
		t.Fatalf("NewHNSW failed: %v", err)
	}

	levelRNG := rand.New(rand.NewPCG(8080, 8080))
	index.RandFunc = levelRNG.Float64
	vectors := benchmarkVectors(2_000, 64, 9090)
	index.InsertBatch(vectors)

	const (
		k  = 10
		ef = 64
	)
	query := vectors[1_337]
	expected := index.KNN_Search(query, k, ef)

	buffer := make([]int, 0, ef)
	got := index.SearchInto(query, k, ef, buffer)
	if !reflect.DeepEqual(got, expected) {
		t.Fatalf("SearchInto mismatch: got %v want %v", got, expected)
	}
	if len(got) != k {
		t.Fatalf("expected %d results, got %d", k, len(got))
	}
	if cap(got) != cap(buffer) {
		t.Fatalf("expected SearchInto to reuse buffer capacity %d, got %d", cap(buffer), cap(got))
	}

	// The second query reuses the returned backing array as well.
	second := index.SearchInto(vectors[777], k, ef, got[:0])
	if len(second) != k {
		t.Fatalf("expected %d results on second query, got %d", k, len(second))
	}
}
