package main

import (
	"encoding/binary"
	"encoding/csv"
	"fmt"
	"math"
	"math/rand/v2"
	"os"
	"runtime"
	"strconv"
	"sync/atomic"
	"syscall"
	"time"

	"dmarro89.github.com/hnsw-go/hnsw"
)

func main() {
	if len(os.Args) != 7 {
		fmt.Fprintln(os.Stderr, "usage: methodology-runner <vectors.f32> <count> <dim> <efConstruction> <threads> <output.csv>")
		os.Exit(2)
	}

	path := os.Args[1]
	count := mustInt(os.Args[2])
	dim := mustInt(os.Args[3])
	efC := mustInt(os.Args[4])
	threads := mustInt(os.Args[5])
	out := os.Args[6]
	instrument := os.Getenv("HNSW_BENCH_INSTRUMENT") == "1"

	vectors, err := readVectors(path, count, dim)
	if err != nil {
		panic(err)
	}

	var distanceEvaluations uint64
	distanceFn := hnsw.EuclideanDistance
	if instrument {
		if threads == 1 {
			distanceFn = func(a, b []float32) float32 {
				distanceEvaluations++
				return hnsw.EuclideanDistance(a, b)
			}
		} else {
			distanceFn = func(a, b []float32) float32 {
				atomic.AddUint64(&distanceEvaluations, 1)
				return hnsw.EuclideanDistance(a, b)
			}
		}
	}

	idx, err := hnsw.NewHNSW(hnsw.Config{
		M:              16,
		Mmax:           16,
		Mmax0:          32,
		EfConstruction: efC,
		MaxLevel:       16,
		DistanceFunc:   distanceFn,
	})
	if err != nil {
		panic(err)
	}
	levelRNG := rand.New(rand.NewPCG(5050, 5050))
	idx.RandFunc = levelRNG.Float64

	runtime.GC()
	var beforeMem runtime.MemStats
	runtime.ReadMemStats(&beforeMem)
	var beforeCPU syscall.Rusage
	if err := syscall.Getrusage(syscall.RUSAGE_SELF, &beforeCPU); err != nil {
		panic(err)
	}

	start := time.Now()
	mode := "serial"
	if threads == 1 {
		idx.InsertBatch(vectors)
	} else {
		mode = "parallel"
		err = idx.BuildParallel(vectors, hnsw.BulkBuildConfig{
			Workers:        threads,
			BatchSize:      max(64, threads*16),
			EfConstruction: efC,
		})
		if err != nil {
			panic(err)
		}
	}
	wall := time.Since(start)

	var afterCPU syscall.Rusage
	if err := syscall.Getrusage(syscall.RUSAGE_SELF, &afterCPU); err != nil {
		panic(err)
	}
	var afterMem runtime.MemStats
	runtime.ReadMemStats(&afterMem)

	directedEdges, maxObservedLevel, graphChecksum, entryPointID := graphMetrics(idx)
	cpuSeconds := rusageSeconds(afterCPU) - rusageSeconds(beforeCPU)
	cpuUtilization := 0.0
	if wall.Seconds() > 0 && threads > 0 {
		cpuUtilization = cpuSeconds / wall.Seconds() / float64(threads)
	}

	f, err := os.Create(out)
	if err != nil {
		panic(err)
	}
	defer f.Close()
	w := csv.NewWriter(f)
	defer w.Flush()

	header := []string{
		"engine", "language", "vectors", "dimensions", "efConstruction", "threads", "mode",
		"seconds", "cpu_seconds", "cpu_utilization", "vectors_per_second",
		"total_alloc_mib", "live_heap_delta_mib", "gc_pause_ms", "num_gc",
		"go_version", "instrumented", "distance_evaluations", "directed_edges",
		"max_observed_level", "entry_point_id", "graph_checksum",
	}
	_ = w.Write(header)
	_ = w.Write([]string{
		"hnsw-go",
		"Go",
		strconv.Itoa(count),
		strconv.Itoa(dim),
		strconv.Itoa(efC),
		strconv.Itoa(threads),
		mode,
		fmt.Sprintf("%.9f", wall.Seconds()),
		fmt.Sprintf("%.9f", cpuSeconds),
		fmt.Sprintf("%.6f", cpuUtilization),
		fmt.Sprintf("%.0f", float64(count)/wall.Seconds()),
		fmt.Sprintf("%.2f", mib(afterMem.TotalAlloc-beforeMem.TotalAlloc)),
		fmt.Sprintf("%.2f", mib(sub(afterMem.HeapAlloc, beforeMem.HeapAlloc))),
		fmt.Sprintf("%.3f", float64(afterMem.PauseTotalNs-beforeMem.PauseTotalNs)/1e6),
		strconv.FormatUint(uint64(afterMem.NumGC-beforeMem.NumGC), 10),
		runtime.Version(),
		strconv.FormatBool(instrument),
		strconv.FormatUint(distanceEvaluations, 10),
		strconv.Itoa(directedEdges),
		strconv.Itoa(maxObservedLevel),
		strconv.Itoa(entryPointID),
		strconv.FormatUint(graphChecksum, 10),
	})
}

func graphMetrics(idx *hnsw.HNSW) (directedEdges, maxObservedLevel int, checksum uint64, entryPointID int) {
	checksum = 1469598103934665603
	const prime uint64 = 1099511628211
	mix := func(v uint64) {
		checksum ^= v
		checksum *= prime
	}

	maxObservedLevel = -1
	entryPointID = -1
	if idx.EntryPoint != nil {
		entryPointID = idx.EntryPoint.ID
	}

	for _, node := range idx.Nodes {
		if node == nil {
			mix(^uint64(0))
			continue
		}
		if node.Level > maxObservedLevel {
			maxObservedLevel = node.Level
		}
		mix(uint64(node.ID) + 1)
		mix(uint64(node.Level) + 1)
		for level, neighbors := range node.Neighbors {
			mix(uint64(level) + 1)
			mix(uint64(len(neighbors)) + 1)
			directedEdges += len(neighbors)
			for _, neighborID := range neighbors {
				mix(uint64(neighborID) + 1)
			}
		}
	}
	return directedEdges, maxObservedLevel, checksum, entryPointID
}

func rusageSeconds(r syscall.Rusage) float64 {
	return timevalSeconds(r.Utime) + timevalSeconds(r.Stime)
}

func timevalSeconds(tv syscall.Timeval) float64 {
	return float64(tv.Sec) + float64(tv.Usec)/1e6
}

func readVectors(path string, count, dim int) ([][]float32, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	want := count * dim * 4
	if len(b) != want {
		return nil, fmt.Errorf("vector file has %d bytes, want %d", len(b), want)
	}

	vectors := make([][]float32, count)
	for i := 0; i < count; i++ {
		row := make([]float32, dim)
		for j := 0; j < dim; j++ {
			u := binary.LittleEndian.Uint32(b[(i*dim+j)*4:])
			row[j] = math.Float32frombits(u)
		}
		vectors[i] = row
	}
	return vectors, nil
}

func mustInt(s string) int {
	v, err := strconv.Atoi(s)
	if err != nil {
		panic(err)
	}
	return v
}

func mib(v uint64) float64 {
	return float64(v) / (1024 * 1024)
}

func sub(a, b uint64) uint64 {
	if a <= b {
		return 0
	}
	return a - b
}
