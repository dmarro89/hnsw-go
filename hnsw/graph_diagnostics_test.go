package hnsw

import (
	"fmt"
	"math/rand/v2"
	"sort"
	"testing"
)

func TestGraphDiagnostics(t *testing.T) {
	const (
		n          = 10_000
		dim        = 128
		queryCount = 100
		k          = 10
	)

	cfg := DefaultConfig()
	cfg.EfConstruction = 64
	h, err := NewHNSW(cfg)
	if err != nil {
		t.Fatal(err)
	}
	levelRNG := rand.New(rand.NewPCG(5050, 5050))
	h.RandFunc = levelRNG.Float64

	vectors := diagnosticVectors(n, dim, 3030)
	h.InsertBatch(vectors)

	reachable := reachableAtLevel0(h)
	zeroDegree := 0
	degreeSum := 0
	maxDegree := 0
	levelCounts := map[int]int{}
	for _, node := range h.Nodes {
		levelCounts[node.Level]++
		degree := len(node.Neighbors[0])
		degreeSum += degree
		if degree == 0 {
			zeroDegree++
		}
		if degree > maxDegree {
			maxDegree = degree
		}
	}

	levels := make([]int, 0, len(levelCounts))
	for level := range levelCounts {
		levels = append(levels, level)
	}
	sort.Ints(levels)
	levelDistribution := ""
	for _, level := range levels {
		levelDistribution += fmt.Sprintf(" L%d=%d", level, levelCounts[level])
	}

	t.Logf("graph nodes=%d entry=%d entryLevel=%d reachableL0=%d (%.2f%%) zeroDegreeL0=%d avgDegreeL0=%.2f maxDegreeL0=%d levels:%s",
		len(h.Nodes), h.EntryPoint.ID, h.EntryPoint.Level, reachable, 100*float64(reachable)/float64(n), zeroDegree,
		float64(degreeSum)/float64(n), maxDegree, levelDistribution)

	queryRNG := rand.New(rand.NewPCG(4040, 4040))
	var recall32, recall64, recall128 float64
	var reachableGT int
	for qi := 0; qi < queryCount; qi++ {
		query := make([]float32, dim)
		for d := range query {
			query[d] = queryRNG.Float32()
		}
		truth := bruteForceTopK(vectors, query, k)
		for _, id := range truth {
			if reachableAtLevel0From(h, h.EntryPoint.ID, id) {
				reachableGT++
			}
		}
		recall32 += diagnosticRecall(truth, h.KNN_Search(query, k, 32))
		recall64 += diagnosticRecall(truth, h.KNN_Search(query, k, 64))
		recall128 += diagnosticRecall(truth, h.KNN_Search(query, k, 128))
	}

	t.Logf("quality queries=%d k=%d groundTruthReachable=%d/%d (%.2f%%) recall@10 ef32=%.4f ef64=%.4f ef128=%.4f",
		queryCount, k, reachableGT, queryCount*k, 100*float64(reachableGT)/float64(queryCount*k),
		recall32/queryCount, recall64/queryCount, recall128/queryCount)
}

func diagnosticVectors(n, dim int, seed uint64) [][]float32 {
	rng := rand.New(rand.NewPCG(seed, seed))
	vectors := make([][]float32, n)
	for i := range vectors {
		vectors[i] = make([]float32, dim)
		for d := range vectors[i] {
			vectors[i][d] = rng.Float32()
		}
	}
	return vectors
}

func reachableAtLevel0(h *HNSW) int {
	if h.EntryPoint == nil {
		return 0
	}
	seen := make([]bool, len(h.Nodes))
	queue := []int{h.EntryPoint.ID}
	seen[h.EntryPoint.ID] = true
	count := 0
	for len(queue) > 0 {
		id := queue[0]
		queue = queue[1:]
		count++
		for _, next := range h.Nodes[id].Neighbors[0] {
			if !seen[next] {
				seen[next] = true
				queue = append(queue, next)
			}
		}
	}
	return count
}

func reachableAtLevel0From(h *HNSW, start, target int) bool {
	if start == target {
		return true
	}
	seen := make([]bool, len(h.Nodes))
	queue := []int{start}
	seen[start] = true
	for len(queue) > 0 {
		id := queue[0]
		queue = queue[1:]
		for _, next := range h.Nodes[id].Neighbors[0] {
			if next == target {
				return true
			}
			if !seen[next] {
				seen[next] = true
				queue = append(queue, next)
			}
		}
	}
	return false
}

type diagnosticNeighbor struct {
	id   int
	dist float32
}

func bruteForceTopK(vectors [][]float32, query []float32, k int) []int {
	all := make([]diagnosticNeighbor, len(vectors))
	for i, vector := range vectors {
		all[i] = diagnosticNeighbor{id: i, dist: EuclideanDistance(query, vector)}
	}
	sort.Slice(all, func(i, j int) bool {
		if all[i].dist == all[j].dist {
			return all[i].id < all[j].id
		}
		return all[i].dist < all[j].dist
	})
	out := make([]int, min(k, len(all)))
	for i := range out {
		out[i] = all[i].id
	}
	return out
}

func diagnosticRecall(want, got []int) float64 {
	if len(want) == 0 {
		return 1
	}
	set := make(map[int]struct{}, len(want))
	for _, id := range want {
		set[id] = struct{}{}
	}
	matches := 0
	for _, id := range got {
		if _, ok := set[id]; ok {
			matches++
		}
	}
	return float64(matches) / float64(len(want))
}
