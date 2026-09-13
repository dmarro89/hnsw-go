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
	"time"

	"dmarro89.github.com/hnsw-go/hnsw"
)

func main() {
	if len(os.Args) != 7 {
		fmt.Fprintln(os.Stderr, "usage: go-runner <vectors.f32> <count> <dim> <efConstruction> <threads> <output.csv>")
		os.Exit(2)
	}
	path := os.Args[1]
	count := mustInt(os.Args[2])
	dim := mustInt(os.Args[3])
	efC := mustInt(os.Args[4])
	threads := mustInt(os.Args[5])
	out := os.Args[6]

	vectors, err := readVectors(path, count, dim)
	if err != nil { panic(err) }
	idx, err := hnsw.NewHNSW(hnsw.Config{M:16, Mmax:32, Mmax0:64, EfConstruction:efC, MaxLevel:16, DistanceFunc:hnsw.EuclideanDistance})
	if err != nil { panic(err) }
	levelRNG := rand.New(rand.NewPCG(5050, 5050))
	idx.RandFunc = levelRNG.Float64

	runtime.GC()
	var before runtime.MemStats
	runtime.ReadMemStats(&before)
	start := time.Now()
	mode := "serial"
	if threads == 1 {
		idx.InsertBatch(vectors)
	} else {
		mode = "parallel"
		err = idx.BuildParallel(vectors, hnsw.BulkBuildConfig{Workers:threads, BatchSize:max(64, threads*16), EfConstruction:efC})
		if err != nil { panic(err) }
	}
	d := time.Since(start)
	var after runtime.MemStats
	runtime.ReadMemStats(&after)

	f, err := os.Create(out)
	if err != nil { panic(err) }
	defer f.Close()
	w := csv.NewWriter(f)
	defer w.Flush()
	_ = w.Write([]string{"engine","language","vectors","dimensions","efConstruction","threads","mode","seconds","vectors_per_second","total_alloc_mib","live_heap_delta_mib"})
	_ = w.Write([]string{"hnsw-go","Go",strconv.Itoa(count),strconv.Itoa(dim),strconv.Itoa(efC),strconv.Itoa(threads),mode,fmt.Sprintf("%.6f",d.Seconds()),fmt.Sprintf("%.0f",float64(count)/d.Seconds()),fmt.Sprintf("%.2f",mib(after.TotalAlloc-before.TotalAlloc)),fmt.Sprintf("%.2f",mib(sub(after.HeapAlloc,before.HeapAlloc)))})
}

func readVectors(path string, count, dim int) ([][]float32, error) {
	b, err := os.ReadFile(path)
	if err != nil { return nil, err }
	want := count*dim*4
	if len(b) != want { return nil, fmt.Errorf("vector file has %d bytes, want %d", len(b), want) }
	v := make([][]float32, count)
	for i:=0;i<count;i++ {
		row := make([]float32, dim)
		for j:=0;j<dim;j++ {
			u := binary.LittleEndian.Uint32(b[(i*dim+j)*4:])
			row[j] = math.Float32frombits(u)
		}
		v[i]=row
	}
	return v,nil
}
func mustInt(s string) int { v,e:=strconv.Atoi(s); if e!=nil { panic(e) }; return v }
func mib(v uint64) float64 { return float64(v)/(1024*1024) }
func sub(a,b uint64) uint64 { if a<=b{return 0}; return a-b }
