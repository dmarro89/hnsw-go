package benchmarks

import (
	"fmt"
	"math/rand/v2"
	"testing"

	"dmarro89.github.com/hnsw-go/hnsw"
)

func BenchmarkHNSWConstruction(b *testing.B) {
	configs := []struct {
		name      string
		numVecs   int
		dimension int
	}{
		{"small", 100, 128},
		{"medium", 1000, 128},
		{"large", 10000, 128},
	}

	for _, cfg := range configs {
		vectors := generateRandomVectors(cfg.numVecs, cfg.dimension)

		b.Run(fmt.Sprintf("Build_%s_%dv_%dd", cfg.name, cfg.numVecs, cfg.dimension), func(b *testing.B) {
			b.ResetTimer()
			b.ReportAllocs()

			for i := 0; i < b.N; i++ {
				b.StopTimer()
				hnsw, _ := hnsw.NewHNSW(hnsw.Config{
					M:              16,
					Mmax:           8,
					Mmax0:          16,
					EfConstruction: 200,
					MaxLevel:       16,
					DistanceFunc:   hnsw.EuclideanDistance,
				})
				b.StartTimer()

				for j := 0; j < cfg.numVecs; j++ {
					hnsw.Insert(vectors[j], j)
				}

				b.ReportMetric(float64(cfg.numVecs)/b.Elapsed().Seconds(), "vectors/sec")
			}
		})
	}
}

func generateRandomVectors(count, dim int) [][]float32 {
	vectors := make([][]float32, count)
	for i := range vectors {
		vectors[i] = make([]float32, dim)
		for j := range vectors[i] {
			vectors[i][j] = rand.Float32()
		}
	}
	return vectors
}
