package benchmarks

import (
	"fmt"
	"runtime"
	"testing"

	"dmarro89.github.com/hnsw-go/hnsw"
)

// BenchmarkParallelBatchSize isolates the batching tradeoff at the runner's
// effective CPU ceiling. Worker count stays fixed so only snapshot staleness,
// proposal/apply barrier frequency, and batching overhead change.
func BenchmarkParallelBatchSize(b *testing.B) {
	const (
		n       = 100000
		dim     = 128
		workers = 4
	)
	vectors := profileVectors(n, dim)
	old := runtime.GOMAXPROCS(workers)
	defer runtime.GOMAXPROCS(old)

	for _, ef := range []int{64, 200} {
		for _, batchSize := range []int{32, 64, 128} {
			ef, batchSize := ef, batchSize
			b.Run(fmt.Sprintf("efc%d/batch%d", ef, batchSize), func(b *testing.B) {
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					idx, err := hnsw.NewHNSW(hnsw.Config{M: 16, Mmax: 16, Mmax0: 32, EfConstruction: ef, MaxLevel: 16, DistanceFunc: hnsw.EuclideanDistance})
					if err != nil { b.Fatal(err) }
					if err := idx.BuildParallel(vectors, hnsw.BulkBuildConfig{Workers: workers, BatchSize: batchSize, EfConstruction: ef}); err != nil { b.Fatal(err) }
				}
			})
		}
	}
}
