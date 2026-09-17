package benchmarks

import (
	"fmt"
	"runtime"
	"testing"

	"dmarro89.github.com/hnsw-go/hnsw"
)

// BenchmarkParallelBuildScaling establishes the multicore scaling curve for the
// existing deterministic proposal/apply bulk builder. Keep the dataset fixed
// across worker counts so speedup and efficiency are directly comparable.
func BenchmarkParallelBuildScaling(b *testing.B) {
	const (
		n   = 100000
		dim = 128
	)
	vectors := profileVectors(n, dim)

	for _, ef := range []int{64, 200} {
		for _, workers := range []int{1, 2, 4, 8} {
			workers := workers
			b.Run(fmt.Sprintf("efc%d/workers%d", ef, workers), func(b *testing.B) {
				old := runtime.GOMAXPROCS(workers)
				defer runtime.GOMAXPROCS(old)
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					idx := hnsw.NewHNSW(16, 200, ef)
					if err := idx.BuildParallel(vectors, hnsw.BulkBuildConfig{
						Workers: workers,
						BatchSize: max(32, workers*16),
						EfConstruction: ef,
					}); err != nil {
						b.Fatal(err)
					}
				}
			})
		}
	}
}
