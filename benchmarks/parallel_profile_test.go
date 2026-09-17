package benchmarks

import (
	"fmt"
	"runtime"
	"testing"

	"dmarro89.github.com/hnsw-go/hnsw"
)

func BenchmarkParallelBuildProfile(b *testing.B) {
	const (
		n   = 100000
		dim = 128
	)
	vectors := profileVectors(n, dim)
	for _, ef := range []int{64, 200} {
		for _, workers := range []int{4, 8} {
			ef, workers := ef, workers
			b.Run(fmt.Sprintf("efc%d/workers%d", ef, workers), func(b *testing.B) {
				old := runtime.GOMAXPROCS(workers)
				defer runtime.GOMAXPROCS(old)
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					idx, err := hnsw.NewHNSW(hnsw.Config{M: 16, Mmax: 16, Mmax0: 32, EfConstruction: ef, MaxLevel: 16, DistanceFunc: hnsw.EuclideanDistance})
					if err != nil { b.Fatal(err) }
					if err := idx.BuildParallel(vectors, hnsw.BulkBuildConfig{Workers: workers, BatchSize: max(32, workers*16), EfConstruction: ef}); err != nil { b.Fatal(err) }
				}
			})
		}
	}
}
