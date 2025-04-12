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

	// In math/rand/v2, dobbiamo creare un generatore esplicito con seed
	// invece di usare rand.Seed()
	rng := rand.New(rand.NewPCG(seedVal, seedVal))

	// Forza GC prima dell'esecuzione
	runtime.GC()

	configs := []struct {
		name      string
		numVecs   int
		dimension int
	}{
		{"small", 10000, 128},
		{"medium", 100000, 128},
		{"large", 1000000, 128},
	}

	for _, cfg := range configs {
		// Genera i vettori una volta sola per tutti i run con lo stesso seed
		vectors := generateRandomVectorsWithRNG(cfg.numVecs, cfg.dimension, rng)

		b.Run(fmt.Sprintf("Build_%s_%dv_%dd", cfg.name, cfg.numVecs, cfg.dimension), func(b *testing.B) {
			// Riporta informazioni sul sistema
			fmt.Printf("NumCPU: %d, GOMAXPROCS: %d\n", runtime.NumCPU(), runtime.GOMAXPROCS(0))

			b.ResetTimer()
			b.ReportAllocs()

			var totalInsertTime time.Duration
			var totalVectors int

			for i := 0; i < b.N; i++ {
				b.StopTimer()
				hnsw, _ := hnsw.NewHNSW(hnsw.Config{
					M:              16,
					Mmax:           8,
					Mmax0:          16,
					EfConstruction: 100,
					MaxLevel:       16,
					DistanceFunc:   hnsw.EuclideanDistance,
				})
				runtime.GC() // Forza GC prima dell'operazione
				b.StartTimer()

				startTime := time.Now()
				for j := 0; j < cfg.numVecs; j++ {
					hnsw.Insert(vectors[j], j)
				}
				elapsed := time.Since(startTime)
				totalInsertTime += elapsed
				totalVectors += cfg.numVecs

				vectorsPerSecond := float64(cfg.numVecs) / elapsed.Seconds()
				b.ReportMetric(vectorsPerSecond, "vectors/sec")
			}

			// Riporta statistiche globali alla fine di tutti i run
			avgVectorsPerSecond := float64(totalVectors) / totalInsertTime.Seconds()
			fmt.Printf("Average insertion rate: %.2f vectors/sec\n", avgVectorsPerSecond)
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
