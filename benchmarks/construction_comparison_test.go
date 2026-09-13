package benchmarks

import (
	"fmt"
	"math/rand/v2"
	"os"
	"runtime"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"dmarro89.github.com/hnsw-go/hnsw"
)

type constructionMeasurement struct {
	mode       string
	vectors    int
	ef         int
	duration   time.Duration
	throughput float64
	totalAlloc uint64
	peakHeap   uint64
	liveHeap   uint64
	mallocs    uint64
	gcCycles   uint32
}

// TestConstructionComparison is a deliberately small, apples-to-apples matrix
// for comparing serial and parallel index construction. Unlike Benchmark's
// B/op, it also samples HeapAlloc while the build is running so CI reports both
// cumulative allocation traffic and approximate peak live heap.
func TestConstructionComparison(t *testing.T) {
	if os.Getenv("HNSW_CONSTRUCTION_COMPARISON") == "" {
		t.Skip("set HNSW_CONSTRUCTION_COMPARISON=1 to run the construction comparison")
	}

	const dimension = 128
	const seed = uint64(42)
	vectorCounts := []int{10_000, 100_000}
	efValues := []int{32, 64}
	modes := []string{"serial", "parallel"}
	workers := runtime.GOMAXPROCS(0)
	if workers < 1 {
		workers = 1
	}

	measurements := make([]constructionMeasurement, 0, len(vectorCounts)*len(efValues)*len(modes))
	for cfgIndex, count := range vectorCounts {
		rngSeed := seed + uint64(cfgIndex)
		rng := rand.New(rand.NewPCG(rngSeed, rngSeed))
		vectors := generateRandomVectorsWithRNG(count, dimension, rng)

		for _, ef := range efValues {
			for _, mode := range modes {
				name := fmt.Sprintf("%s/%dk/ef%d", mode, count/1000, ef)
				t.Run(name, func(t *testing.T) {
					measurement := measureConstruction(t, vectors, count, ef, mode, workers, seed+10_000+uint64(cfgIndex))
					measurements = append(measurements, measurement)
				})
			}
		}
	}

	var out strings.Builder
	out.WriteString("| Vectors | efConstruction | Mode | Time | Vectors/s | Total allocated | Peak heap delta | Live heap delta | Mallocs | GC |\n")
	out.WriteString("| ---: | ---: | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n")
	for _, m := range measurements {
		fmt.Fprintf(&out, "| %d | %d | %s | %.3fs | %.0f | %.1f MiB | %.1f MiB | %.1f MiB | %d | %d |\n",
			m.vectors, m.ef, m.mode, m.duration.Seconds(), m.throughput,
			bytesToMiB(m.totalAlloc), bytesToMiB(m.peakHeap), bytesToMiB(m.liveHeap), m.mallocs, m.gcCycles)
	}

	t.Log("\n" + out.String())
	if output := os.Getenv("HNSW_CONSTRUCTION_COMPARISON_OUTPUT"); output != "" {
		if err := os.WriteFile(output, []byte(out.String()), 0o644); err != nil {
			t.Fatalf("write comparison output: %v", err)
		}
	}
}

func measureConstruction(t *testing.T, vectors [][]float32, count, ef int, mode string, workers int, levelSeed uint64) constructionMeasurement {
	t.Helper()
	runtime.GC()

	index, err := hnsw.NewHNSW(hnsw.Config{
		M:              16,
		Mmax:           32,
		Mmax0:          64,
		EfConstruction: ef,
		MaxLevel:       16,
		DistanceFunc:   hnsw.EuclideanDistance,
	})
	if err != nil {
		t.Fatalf("NewHNSW: %v", err)
	}
	levelRNG := rand.New(rand.NewPCG(levelSeed, levelSeed))
	index.RandFunc = levelRNG.Float64

	var before runtime.MemStats
	runtime.ReadMemStats(&before)
	var peak atomic.Uint64
	peak.Store(before.HeapAlloc)
	stop := make(chan struct{})
	done := make(chan struct{})
	go func() {
		defer close(done)
		ticker := time.NewTicker(5 * time.Millisecond)
		defer ticker.Stop()
		for {
			select {
			case <-ticker.C:
				var stats runtime.MemStats
				runtime.ReadMemStats(&stats)
				for old := peak.Load(); stats.HeapAlloc > old && !peak.CompareAndSwap(old, stats.HeapAlloc); old = peak.Load() {
				}
			case <-stop:
				return
			}
		}
	}()

	start := time.Now()
	switch mode {
	case "serial":
		index.InsertBatch(vectors)
	case "parallel":
		err = index.BuildParallel(vectors, hnsw.BulkBuildConfig{
			Workers:        workers,
			BatchSize:      max(64, workers*16),
			EfConstruction: ef,
		})
	default:
		t.Fatalf("unknown construction mode %q", mode)
	}
	duration := time.Since(start)
	close(stop)
	<-done

	var after runtime.MemStats
	runtime.ReadMemStats(&after)
	peakHeap := peak.Load()
	if after.HeapAlloc > peakHeap {
		peakHeap = after.HeapAlloc
	}

	return constructionMeasurement{
		mode:       mode,
		vectors:    count,
		ef:         ef,
		duration:   duration,
		throughput: float64(count) / duration.Seconds(),
		totalAlloc: after.TotalAlloc - before.TotalAlloc,
		peakHeap:   subtractFloor(peakHeap, before.HeapAlloc),
		liveHeap:   subtractFloor(after.HeapAlloc, before.HeapAlloc),
		mallocs:    after.Mallocs - before.Mallocs,
		gcCycles:   after.NumGC - before.NumGC,
	}
}

func subtractFloor(value, baseline uint64) uint64 {
	if value <= baseline {
		return 0
	}
	return value - baseline
}

func bytesToMiB(value uint64) float64 {
	return float64(value) / (1024 * 1024)
}
