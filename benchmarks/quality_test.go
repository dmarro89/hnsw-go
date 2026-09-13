package benchmarks

import (
	"math/rand/v2"
	"runtime"
	"sort"
	"testing"
	"time"

	"dmarro89.github.com/hnsw-go/hnsw"
)

const qualityLevelSeed uint64 = 5050

func TestEfConstructionRecallTradeoff(t *testing.T) {
	if testing.Short() {
		t.Skip("Saltando il confronto recall/build time in modalità short")
	}

	const (
		numVectors = 2000
		numQueries = 200
		dimension  = 64
		k          = 10
		efSearch   = 50
	)

	datasetRNG := rand.New(rand.NewPCG(42, 42))
	queryRNG := rand.New(rand.NewPCG(99, 99))
	vectors := generateRandomVectorsWithRNG(numVectors, dimension, datasetRNG)
	queries := generateRandomVectorsWithRNG(numQueries, dimension, queryRNG)

	groundTruth := make([][]int, len(queries))
	for i, query := range queries {
		groundTruth[i] = bruteForceTopK(vectors, query, k)
	}

	efConstructionValues := []int{64, 100, 200}
	for _, efConstruction := range efConstructionValues {
		t.Run("efConstruction", func(t *testing.T) {
			index, buildDuration := buildIndexForQualityTest(t, vectors, efConstruction)

			totalRecall := 0.0
			for i, query := range queries {
				results := index.KNN_Search(query, k, efSearch)
				if len(results) != k {
					t.Fatalf("Expected %d results, got %d", k, len(results))
				}
				totalRecall += recallAtK(results, groundTruth[i], k)
			}

			avgRecall := totalRecall / float64(len(queries))
			t.Logf("efConstruction=%d build=%s recall@%d=%.4f efSearch=%d",
				efConstruction, buildDuration.Round(time.Millisecond), k, avgRecall, efSearch)
		})
	}
}

func TestParallelBuildRecallTradeoff(t *testing.T) {
	if testing.Short() {
		t.Skip("Saltando il confronto recall tra build seriale e parallela in modalità short")
	}

	const (
		numVectors = 3000
		numQueries = 200
		dimension  = 64
		k          = 10
		efSearch   = 50
	)

	datasetRNG := rand.New(rand.NewPCG(123, 123))
	queryRNG := rand.New(rand.NewPCG(321, 321))
	vectors := generateRandomVectorsWithRNG(numVectors, dimension, datasetRNG)
	queries := generateRandomVectorsWithRNG(numQueries, dimension, queryRNG)

	groundTruth := make([][]int, len(queries))
	for i, query := range queries {
		groundTruth[i] = bruteForceTopK(vectors, query, k)
	}

	efConstruction := 64
	serial, serialBuild := buildIndexForQualityTest(t, vectors, efConstruction)
	parallel, parallelBuild := buildParallelIndexForQualityTest(t, vectors, efConstruction)

	serialRecall := averageRecallAtK(serial, queries, groundTruth, k, efSearch)
	parallelRecall := averageRecallAtK(parallel, queries, groundTruth, k, efSearch)
	delta := serialRecall - parallelRecall

	t.Logf("serial build=%s recall@%d=%.4f efSearch=%d", serialBuild.Round(time.Millisecond), k, serialRecall, efSearch)
	t.Logf("parallel build=%s recall@%d=%.4f efSearch=%d", parallelBuild.Round(time.Millisecond), k, parallelRecall, efSearch)
	t.Logf("parallel minus serial recall delta=%.4f", parallelRecall-serialRecall)

	if delta > 0.05 {
		t.Fatalf("parallel recall degraded too much: serial=%.4f parallel=%.4f delta=%.4f", serialRecall, parallelRecall, delta)
	}
}

func buildIndexForQualityTest(t *testing.T, vectors [][]float32, efConstruction int) (*hnsw.HNSW, time.Duration) {
	t.Helper()

	index, err := hnsw.NewHNSW(hnsw.Config{
		M:              16,
		Mmax:           32,
		Mmax0:          64,
		EfConstruction: efConstruction,
		MaxLevel:       16,
		DistanceFunc:   hnsw.EuclideanDistance,
	})
	if err != nil {
		t.Fatalf("Failed to create HNSW: %v", err)
	}

	levelRNG := rand.New(rand.NewPCG(qualityLevelSeed, qualityLevelSeed))
	index.RandFunc = levelRNG.Float64

	start := time.Now()
	index.InsertBatch(vectors)
	return index, time.Since(start)
}

func buildParallelIndexForQualityTest(t *testing.T, vectors [][]float32, efConstruction int) (*hnsw.HNSW, time.Duration) {
	t.Helper()

	index, err := hnsw.NewHNSW(hnsw.Config{
		M:              16,
		Mmax:           32,
		Mmax0:          64,
		EfConstruction: efConstruction,
		MaxLevel:       16,
		DistanceFunc:   hnsw.EuclideanDistance,
	})
	if err != nil {
		t.Fatalf("Failed to create HNSW: %v", err)
	}

	levelRNG := rand.New(rand.NewPCG(qualityLevelSeed, qualityLevelSeed))
	index.RandFunc = levelRNG.Float64

	buildCfg := hnsw.BulkBuildConfig{
		Workers:        runtime.GOMAXPROCS(0),
		BatchSize:      64,
		EfConstruction: efConstruction,
	}

	start := time.Now()
	if err := index.BuildParallel(vectors, buildCfg); err != nil {
		t.Fatalf("BuildParallel failed: %v", err)
	}
	return index, time.Since(start)
}

func bruteForceTopK(vectors [][]float32, query []float32, k int) []int {
	type candidate struct {
		id   int
		dist float32
	}

	candidates := make([]candidate, len(vectors))
	for i, vector := range vectors {
		candidates[i] = candidate{
			id:   i,
			dist: hnsw.EuclideanDistance(vector, query),
		}
	}

	sort.Slice(candidates, func(i, j int) bool {
		if candidates[i].dist == candidates[j].dist {
			return candidates[i].id < candidates[j].id
		}
		return candidates[i].dist < candidates[j].dist
	})

	if k > len(candidates) {
		k = len(candidates)
	}

	results := make([]int, k)
	for i := 0; i < k; i++ {
		results[i] = candidates[i].id
	}
	return results
}

func recallAtK(results, groundTruth []int, k int) float64 {
	truth := make(map[int]struct{}, min(k, len(groundTruth)))
	for i := 0; i < k && i < len(groundTruth); i++ {
		truth[groundTruth[i]] = struct{}{}
	}

	hits := 0
	for i := 0; i < k && i < len(results); i++ {
		if _, ok := truth[results[i]]; ok {
			hits++
		}
	}

	return float64(hits) / float64(k)
}

func averageRecallAtK(index *hnsw.HNSW, queries [][]float32, groundTruth [][]int, k, efSearch int) float64 {
	totalRecall := 0.0
	for i, query := range queries {
		results := index.KNN_Search(query, k, efSearch)
		totalRecall += recallAtK(results, groundTruth[i], k)
	}
	return totalRecall / float64(len(queries))
}
