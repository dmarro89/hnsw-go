package benchmarks

import (
	"fmt"
	"math/rand/v2"
	"testing"

	"dmarro89.github.com/hnsw-go/hnsw"
)

func TestParallelBatchSizeRecall(t *testing.T) {
	const n, dim, nq, k = 20000, 128, 500, 10
	vectors := generateRandomVectorsWithRNG(n, dim, rand.New(rand.NewPCG(4242, 4242)))
	queries := generateRandomVectorsWithRNG(nq, dim, rand.New(rand.NewPCG(4343, 4343)))
	truth := make([][]int, nq)
	for i, q := range queries { truth[i] = bruteForceTopK(vectors, q, k) }
	for _, efc := range []int{64, 200} {
		for _, batch := range []int{64, 128} {
			idx, err := hnsw.NewHNSW(hnsw.Config{M:16, Mmax:16, Mmax0:32, EfConstruction:efc, MaxLevel:16, DistanceFunc:hnsw.EuclideanDistance})
			if err != nil { t.Fatal(err) }
			levelRNG := rand.New(rand.NewPCG(5050, 5050)); idx.RandFunc = levelRNG.Float64
			if err := idx.BuildParallel(vectors, hnsw.BulkBuildConfig{Workers:4, BatchSize:batch, EfConstruction:efc}); err != nil { t.Fatal(err) }
			for _, efs := range []int{32,64,128} {
				r := averageRecallAtK(idx, queries, truth, k, efs)
				t.Logf("efC=%d batch=%d efSearch=%d recall@10=%.4f", efc, batch, efs, r)
				if r <= 0 { t.Fatalf("invalid recall: %s", fmt.Sprint(r)) }
			}
		}
	}
}
