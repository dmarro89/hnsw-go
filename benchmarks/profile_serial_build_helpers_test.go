package benchmarks

import (
	"fmt"
	"math/rand/v2"
	"os"
	"runtime"
	"runtime/pprof"
	"strconv"
	"testing"
	"time"

	"dmarro89.github.com/hnsw-go/hnsw"
)

func runSerialBuildProfile(t *testing.T) {
	numVectors := profileEnvInt(t, "HNSW_PROFILE_VECTORS", 100_000)
	dimension := profileEnvInt(t, "HNSW_PROFILE_DIM", 128)
	efConstruction := profileEnvInt(t, "HNSW_PROFILE_EF_CONSTRUCTION", 64)
	prefix := os.Getenv("HNSW_PROFILE_PREFIX")
	if prefix == "" {
		prefix = fmt.Sprintf("serial-%dk-%dd-efc%d", numVectors/1000, dimension, efConstruction)
	}

	vectors := profileVectors(numVectors, dimension)
	index, err := hnsw.NewHNSW(hnsw.Config{
		M:              16,
		Mmax:           16,
		Mmax0:          32,
		EfConstruction: efConstruction,
		MaxLevel:       16,
		DistanceFunc:   hnsw.EuclideanDistance,
	})
	if err != nil {
		t.Fatalf("NewHNSW failed: %v", err)
	}
	index.RandFunc = rand.New(rand.NewPCG(5050, 5050)).Float64

	runtime.GC()
	var before runtime.MemStats
	runtime.ReadMemStats(&before)

	cpuPath := prefix + "-cpu.pprof"
	cpuFile, err := os.Create(cpuPath)
	if err != nil {
		t.Fatalf("create CPU profile: %v", err)
	}
	if err := pprof.StartCPUProfile(cpuFile); err != nil {
		_ = cpuFile.Close()
		t.Fatalf("start CPU profile: %v", err)
	}

	start := time.Now()
	index.InsertBatch(vectors)
	elapsed := time.Since(start)
	pprof.StopCPUProfile()
	if err := cpuFile.Close(); err != nil {
		t.Fatalf("close CPU profile: %v", err)
	}

	var after runtime.MemStats
	runtime.ReadMemStats(&after)

	runtime.GC()
	heapPath := prefix + "-heap.pprof"
	heapFile, err := os.Create(heapPath)
	if err != nil {
		t.Fatalf("create heap profile: %v", err)
	}
	if err := pprof.WriteHeapProfile(heapFile); err != nil {
		_ = heapFile.Close()
		t.Fatalf("write heap profile: %v", err)
	}
	if err := heapFile.Close(); err != nil {
		t.Fatalf("close heap profile: %v", err)
	}

	t.Logf(
		"PROFILE_RESULT vectors=%d dim=%d efConstruction=%d elapsed=%s vectors_per_second=%.0f total_alloc_mib=%.2f mallocs=%d gc=%d heap_alloc_delta_mib=%.2f cpu_profile=%s heap_profile=%s",
		numVectors,
		dimension,
		efConstruction,
		elapsed,
		float64(numVectors)/elapsed.Seconds(),
		profileMiB(after.TotalAlloc-before.TotalAlloc),
		after.Mallocs-before.Mallocs,
		after.NumGC-before.NumGC,
		profileMiB(profileUint64Sub(after.HeapAlloc, before.HeapAlloc)),
		cpuPath,
		heapPath,
	)

	runtime.KeepAlive(index)
	runtime.KeepAlive(vectors)
}

func profileEnvInt(t *testing.T, name string, fallback int) int {
	t.Helper()
	value := os.Getenv(name)
	if value == "" {
		return fallback
	}
	parsed, err := strconv.Atoi(value)
	if err != nil || parsed <= 0 {
		t.Fatalf("%s must be a positive integer, got %q", name, value)
	}
	return parsed
}

func profileVectors(count, dimension int) [][]float32 {
	rng := rand.New(rand.NewPCG(3030, 3030))
	vectors := make([][]float32, count)
	for i := range vectors {
		row := make([]float32, dimension)
		for j := range row {
			row[j] = rng.Float32()
		}
		vectors[i] = row
	}
	return vectors
}

func profileMiB(value uint64) float64 {
	return float64(value) / (1024 * 1024)
}

func profileUint64Sub(a, b uint64) uint64 {
	if a <= b {
		return 0
	}
	return a - b
}
