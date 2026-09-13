package hnsw

import (
	"math/rand/v2"
	"reflect"
	"sync"
	"testing"
)

func TestConcurrentKNNSearchMatchesSequentialResults(t *testing.T) {
	cfg := Config{
		M:              16,
		Mmax:           32,
		Mmax0:          64,
		EfConstruction: 64,
		MaxLevel:       16,
		DistanceFunc:   EuclideanDistance,
	}
	index, err := NewHNSW(cfg)
	if err != nil {
		t.Fatalf("NewHNSW failed: %v", err)
	}

	levelRNG := rand.New(rand.NewPCG(2026, 2026))
	index.RandFunc = levelRNG.Float64
	vectors := benchmarkVectors(2_000, 64, 99)
	index.InsertBatch(vectors)

	queries := [][]float32{
		vectors[11], vectors[97], vectors[211], vectors[389],
		vectors[601], vectors[877], vectors[1_201], vectors[1_733],
	}
	expected := make([][]int, len(queries))
	for i, query := range queries {
		expected[i] = append([]int(nil), index.KNN_Search(query, 10, 64)...)
	}

	const (
		goroutines = 8
		iterations = 100
	)

	var wg sync.WaitGroup
	errors := make(chan string, goroutines)
	for worker := 0; worker < goroutines; worker++ {
		worker := worker
		wg.Add(1)
		go func() {
			defer wg.Done()
			for iteration := 0; iteration < iterations; iteration++ {
				queryIndex := (worker + iteration) % len(queries)
				got := index.KNN_Search(queries[queryIndex], 10, 64)
				if !reflect.DeepEqual(got, expected[queryIndex]) {
					errors <- "concurrent KNN search returned a result different from the sequential baseline"
					return
				}
			}
		}()
	}

	wg.Wait()
	close(errors)
	for message := range errors {
		t.Error(message)
	}
}
