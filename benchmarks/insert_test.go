package benchmarks

import (
	"fmt"
	"math/rand/v2"
	"os"
	"runtime"
	"strconv"
	"testing"
	"time"

	"dmarro89.github.com/hnsw-go/hnsw"
)

func BenchmarkHNSWConstruction(b *testing.B) {
	// Usa un seed fisso per generare sempre gli stessi vettori casuali
	// Per disabilitare, impostare la variabile di ambiente HNSW_RAND_SEED=-1
	seedStr := os.Getenv("HNSW_RAND_SEED")
	seedVal := uint64(42) // default seed
	if seedStr != "" {
		if val, err := strconv.ParseUint(seedStr, 10, 64); err == nil {
			seedVal = val
		}
	}

	configs := []struct {
		name      string
		numVecs   int
		dimension int
	}{
		{"small", 1000, 128},
		{"medium", 10000, 128},
		{"large", 100000, 128},
	}
	efConstructionValues := []int{64}

	for cfgIndex, cfg := range configs {
		// Genera i vettori una volta sola per tutti i run con lo stesso seed
		rng := rand.New(rand.NewPCG(seedVal+uint64(cfgIndex), seedVal+uint64(cfgIndex)))
		vectors := generateRandomVectorsWithRNG(cfg.numVecs, cfg.dimension, rng)

		for _, efConstruction := range efConstructionValues {
			b.Run(fmt.Sprintf("Build_%s_%dv_%dd_ef%d", cfg.name, cfg.numVecs, cfg.dimension, efConstruction), func(b *testing.B) {
				b.ReportAllocs()
				b.ResetTimer()

				var totalInsertTime time.Duration
				var totalVectors int64

				for i := 0; i < b.N; i++ {
					b.StopTimer()
					hnsw, _ := hnsw.NewHNSW(hnsw.Config{
						M:              16,
						Mmax:           32,
						Mmax0:          64,
						EfConstruction: efConstruction,
						MaxLevel:       16,
						DistanceFunc:   hnsw.EuclideanDistance,
					})
					b.StartTimer()

					startTime := time.Now()
					hnsw.InsertBatch(vectors)
					elapsed := time.Since(startTime)
					b.StopTimer()

					totalInsertTime += elapsed
					totalVectors += int64(cfg.numVecs)
				}

				if totalInsertTime > 0 {
					b.ReportMetric(float64(totalVectors)/totalInsertTime.Seconds(), "vectors/sec")
				}
			})
		}
	}
}

func BenchmarkHNSWParallelConstruction(b *testing.B) {
	seedStr := os.Getenv("HNSW_RAND_SEED")
	seedVal := uint64(42)
	if seedStr != "" {
		if val, err := strconv.ParseUint(seedStr, 10, 64); err == nil {
			seedVal = val
		}
	}

	configs := []struct {
		name      string
		numVecs   int
		dimension int
	}{
		{"small", 1000, 128},
		{"medium", 10000, 128},
		{"large", 100000, 128},
	}

	workers := runtime.GOMAXPROCS(0)
	if workers < 1 {
		workers = 1
	}

	for cfgIndex, cfg := range configs {
		rng := rand.New(rand.NewPCG(seedVal+uint64(cfgIndex), seedVal+uint64(cfgIndex)))
		vectors := generateRandomVectorsWithRNG(cfg.numVecs, cfg.dimension, rng)

		b.Run(fmt.Sprintf("BuildParallel_%s_%dv_%dd", cfg.name, cfg.numVecs, cfg.dimension), func(b *testing.B) {
			b.ReportAllocs()
			b.ResetTimer()

			var totalInsertTime time.Duration
			var totalVectors int64

			for i := 0; i < b.N; i++ {
				b.StopTimer()
				index, _ := hnsw.NewHNSW(hnsw.Config{
					M:              16,
					Mmax:           32,
					Mmax0:          64,
					EfConstruction: 64,
					MaxLevel:       16,
					DistanceFunc:   hnsw.EuclideanDistance,
				})
				buildCfg := hnsw.BulkBuildConfig{
					Workers:        workers,
					BatchSize:      max(64, workers*16),
					EfConstruction: 64,
				}
				b.StartTimer()

				startTime := time.Now()
				if err := index.BuildParallel(vectors, buildCfg); err != nil {
					b.Fatalf("BuildParallel failed: %v", err)
				}
				elapsed := time.Since(startTime)
				b.StopTimer()

				totalInsertTime += elapsed
				totalVectors += int64(cfg.numVecs)
			}

			if totalInsertTime > 0 {
				b.ReportMetric(float64(totalVectors)/totalInsertTime.Seconds(), "vectors/sec")
			}
		})
	}
}

// Versione modificata per accettare un generatore RNG esplicito
func generateRandomVectorsWithRNG(count, dim int, rng *rand.Rand) [][]float32 {
	vectors := make([][]float32, count)
	for i := range vectors {
		vectors[i] = make([]float32, dim)
		for j := range vectors[i] {
			vectors[i][j] = rng.Float32()
		}
	}
	return vectors
}

// Manteniamo la vecchia funzione per compatibilità
func generateRandomVectors(count, dim int) [][]float32 {
	// Usiamo il generatore globale
	vectors := make([][]float32, count)
	for i := range vectors {
		vectors[i] = make([]float32, dim)
		for j := range vectors[i] {
			vectors[i][j] = rand.Float32()
		}
	}
	return vectors
}
