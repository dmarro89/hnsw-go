//go:build goexperiment.simd && amd64

package main

import (
	"encoding/binary"
	"encoding/csv"
	"fmt"
	"math"
	"math/rand/v2"
	"os"
	"runtime"
	"runtime/pprof"
	"strconv"
	"time"

	"dmarro89.github.com/hnsw-go/hnsw"
	"simd/archsimd"
)

type distanceFn func([]float32, []float32) float32

func main() {
	if len(os.Args) != 9 {
		fmt.Fprintln(os.Stderr, "usage: simd256-go <scalar|simd256> <vectors.f32> <count> <dim> <efConstruction> <queries.f32> <truth.u32> <output.csv>")
		os.Exit(2)
	}

	mode := os.Args[1]
	vectorsPath := os.Args[2]
	count := mustInt(os.Args[3])
	dim := mustInt(os.Args[4])
	efC := mustInt(os.Args[5])
	queriesPath := os.Args[6]
	truthPath := os.Args[7]
	out := os.Args[8]

	var distance distanceFn
	switch mode {
	case "scalar":
		distance = hnsw.EuclideanDistance
	case "simd256":
		distance = distanceSIMD256
	default:
		fmt.Fprintf(os.Stderr, "unknown mode %q\n", mode)
		os.Exit(2)
	}

	var distanceCalls uint64
	if os.Getenv("HNSW_DISTANCE_COUNT") == "1" {
		baseDistance := distance
		distance = func(a, b []float32) float32 {
			distanceCalls++
			return baseDistance(a, b)
		}
	}

	vectors := readF32(vectorsPath, dim)
	if len(vectors) != count {
		panic(fmt.Sprintf("vector count mismatch: got %d want %d", len(vectors), count))
	}
	queries := readF32(queriesPath, dim)
	truth := readU32(truthPath, 10)
	if len(queries) != len(truth) {
		panic("query/truth count mismatch")
	}

	idx, err := hnsw.NewHNSW(hnsw.Config{
		M: 16, Mmax: 16, Mmax0: 32,
		EfConstruction: efC,
		MaxLevel:       16,
		DistanceFunc:   distance,
	})
	if err != nil {
		panic(err)
	}
	idx.RandFunc = rand.New(rand.NewPCG(5050, 5050)).Float64

	var profileFile *os.File
	if profilePath := os.Getenv("HNSW_CPU_PROFILE"); profilePath != "" {
		profileFile, err = os.Create(profilePath)
		if err != nil {
			panic(err)
		}
		if err = pprof.StartCPUProfile(profileFile); err != nil {
			_ = profileFile.Close()
			panic(err)
		}
	}

	runtime.GC()
	var before runtime.MemStats
	runtime.ReadMemStats(&before)
	start := time.Now()
	idx.InsertBatch(vectors)
	build := time.Since(start)
	buildDistanceCalls := distanceCalls
	var after runtime.MemStats
	runtime.ReadMemStats(&after)

	if profileFile != nil {
		pprof.StopCPUProfile()
		if err := profileFile.Close(); err != nil {
			panic(err)
		}
	}

	f, err := os.Create(out)
	if err != nil {
		panic(err)
	}
	defer f.Close()
	w := csv.NewWriter(f)
	defer w.Flush()

	fields := []string{
		"engine", "mode", "vectors", "dimensions", "efConstruction",
		"seconds", "vectors_per_second", "total_alloc_mib", "live_heap_delta_mib",
		"distance_calls", "efSearch", "recall_at_10",
	}
	if err := w.Write(fields); err != nil {
		panic(err)
	}

	engine := "hnsw-go-scalar"
	if mode == "simd256" {
		engine = "hnsw-go-simd256"
	}
	for _, ef := range []int{32, 64, 128} {
		r := recall(idx, queries, truth, ef)
		row := []string{
			engine,
			mode,
			strconv.Itoa(count),
			strconv.Itoa(dim),
			strconv.Itoa(efC),
			fmt.Sprintf("%.6f", build.Seconds()),
			fmt.Sprintf("%.0f", float64(count)/build.Seconds()),
			fmt.Sprintf("%.2f", mib(after.TotalAlloc-before.TotalAlloc)),
			fmt.Sprintf("%.2f", mib(sub(after.HeapAlloc, before.HeapAlloc))),
			strconv.FormatUint(buildDistanceCalls, 10),
			strconv.Itoa(ef),
			fmt.Sprintf("%.4f", r),
		}
		if err := w.Write(row); err != nil {
			panic(err)
		}
	}
}

func distanceSIMD256(a, b []float32) float32 {
	var acc archsimd.Float32x8
	i := 0
	for ; i <= len(a)-8; i += 8 {
		av := archsimd.LoadFloat32x8Slice(a[i:])
		bv := archsimd.LoadFloat32x8Slice(b[i:])
		d := av.Sub(bv)
		acc = d.MulAdd(d, acc)
	}
	var lanes [8]float32
	acc.StoreSlice(lanes[:])
	sum := lanes[0] + lanes[1] + lanes[2] + lanes[3] + lanes[4] + lanes[5] + lanes[6] + lanes[7]
	for ; i < len(a); i++ {
		d := a[i] - b[i]
		sum += d * d
	}
	return sum
}

func recall(idx *hnsw.HNSW, queries [][]float32, truth [][]uint32, ef int) float64 {
	hits := 0
	for i, query := range queries {
		for _, id := range idx.KNN_Search(query, 10, ef) {
			for _, want := range truth[i] {
				if id == int(want) {
					hits++
					break
				}
			}
		}
	}
	return float64(hits) / float64(len(queries)*10)
}

func readF32(path string, width int) [][]float32 {
	b, err := os.ReadFile(path)
	if err != nil {
		panic(err)
	}
	if len(b)%(width*4) != 0 {
		panic("invalid f32 file")
	}
	rows := make([][]float32, len(b)/(width*4))
	for i := range rows {
		rows[i] = make([]float32, width)
		for j := 0; j < width; j++ {
			rows[i][j] = math.Float32frombits(binary.LittleEndian.Uint32(b[(i*width+j)*4:]))
		}
	}
	return rows
}

func readU32(path string, width int) [][]uint32 {
	b, err := os.ReadFile(path)
	if err != nil {
		panic(err)
	}
	if len(b)%(width*4) != 0 {
		panic("invalid u32 file")
	}
	rows := make([][]uint32, len(b)/(width*4))
	for i := range rows {
		rows[i] = make([]uint32, width)
		for j := 0; j < width; j++ {
			rows[i][j] = binary.LittleEndian.Uint32(b[(i*width+j)*4:])
		}
	}
	return rows
}

func mustInt(s string) int {
	v, err := strconv.Atoi(s)
	if err != nil {
		panic(err)
	}
	return v
}

func mib(v uint64) float64 { return float64(v) / (1024 * 1024) }

func sub(a, b uint64) uint64 {
	if a <= b {
		return 0
	}
	return a - b
}
