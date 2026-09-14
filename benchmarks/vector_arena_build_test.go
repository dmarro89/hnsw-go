package benchmarks

import (
	"fmt"
	"math/rand/v2"
	"testing"

	"dmarro89.github.com/hnsw-go/hnsw"
)

func benchmarkArenaBuild(b *testing.B, count, ef int, arena bool) {
	const dim = 128
	vectors := deterministicBuildVectors(count, dim, uint64(31000+ef))
	name := "baseline"
	if arena {
		name = "production_arena"
	}
	b.Run(fmt.Sprintf("%s/%dk/ef%d", name, count/1000, ef), func(b *testing.B) {
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			cfg := hnsw.Config{M:16, Mmax:16, Mmax0:32, EfConstruction:ef, MaxLevel:16, DistanceFunc:hnsw.EuclideanDistance}
			index, err := hnsw.NewHNSW(cfg)
			if err != nil {
				b.Fatal(err)
			}
			index.RandFunc = rand.New(rand.NewPCG(5050, 5050)).Float64
			if arena {
				index.InsertBatch(vectors)
			} else {
				index.InsertBatchBaselineExperimental(vectors)
			}
		}
	})
}

func BenchmarkVectorArenaBuild20k(b *testing.B) {
	for _, ef := range []int{64, 200} {
		benchmarkArenaBuild(b, 20000, ef, false)
		benchmarkArenaBuild(b, 20000, ef, true)
	}
}

func BenchmarkVectorArenaBuild100k(b *testing.B) {
	for _, ef := range []int{64, 200} {
		benchmarkArenaBuild(b, 100000, ef, false)
		benchmarkArenaBuild(b, 100000, ef, true)
	}
}
