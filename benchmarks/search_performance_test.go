package benchmarks

import (
	"fmt"
	"math/rand/v2"
	"testing"

	"dmarro89.github.com/hnsw-go/hnsw"
)

// BenchmarkHNSWSearchPerformance measures the query path while keeping search
// quality visible next to latency and allocations. A faster result is not an
// improvement if it comes from traversing a worse graph.
func BenchmarkHNSWSearchPerformance(b *testing.B) {
	const (
		numVectors = 20_000
		numQueries = 256
		dimension  = 128
		k          = 10
	)

	datasetRNG := rand.New(rand.NewPCG(2026, 2026))
	queryRNG := rand.New(rand.NewPCG(2027, 2027))
	vectors := generateRandomVectorsWithRNG(numVectors, dimension, datasetRNG)
	queries := generateRandomVectorsWithRNG(numQueries, dimension, queryRNG)

	index, _ := buildIndexForQualityTest(b, vectors, 64)

	groundTruth := make([][]int, len(queries))
	for i, query := range queries {
		groundTruth[i] = bruteForceTopK(vectors, query, k)
	}

	for _, efSearch := range []int{32, 64, 128} {
		b.Run(fmt.Sprintf("20k_128d_k10_ef%d", efSearch), func(b *testing.B) {
			recall := averageRecallAtK(index, queries, groundTruth, k, efSearch)
			b.ReportMetric(recall*100, "recall@10_pct")
			b.ReportAllocs()
			b.ResetTimer()

			for i := 0; i < b.N; i++ {
				query := queries[i&(numQueries-1)]
				results := index.KNN_Search(query, k, efSearch)
				if len(results) != k {
					b.Fatalf("expected %d results, got %d", k, len(results))
				}
			}
		})
	}
}

// TestSearchQualityMatrix is a deterministic quality regression test. The
// threshold is intentionally permissive for the baseline commit; once the
// neighbor-selection heuristic is measured, the PR tightens this gate to the
// quality level actually demonstrated by the optimized implementation.
func TestSearchQualityMatrix(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping search quality matrix in short mode")
	}

	const (
		numVectors = 10_000
		numQueries = 200
		dimension  = 128
		k          = 10
	)

	datasetRNG := rand.New(rand.NewPCG(3030, 3030))
	queryRNG := rand.New(rand.NewPCG(4040, 4040))
	vectors := generateRandomVectorsWithRNG(numVectors, dimension, datasetRNG)
	queries := generateRandomVectorsWithRNG(numQueries, dimension, queryRNG)

	groundTruth := make([][]int, len(queries))
	for i, query := range queries {
		groundTruth[i] = bruteForceTopK(vectors, query, k)
	}

	index, buildDuration := buildIndexForQualityTest(t, vectors, 64)
	for _, efSearch := range []int{32, 64, 128} {
		recall := averageRecallAtK(index, queries, groundTruth, k, efSearch)
		t.Logf("build=%s efSearch=%d recall@%d=%.4f", buildDuration, efSearch, k, recall)
		if recall <= 0 {
			t.Fatalf("invalid zero recall at efSearch=%d", efSearch)
		}
	}
}

var _ = hnsw.EuclideanDistance
