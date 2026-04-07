package hnsw

import (
	"fmt"
	"math/rand/v2"
	"testing"

	"dmarro89.github.com/hnsw-go/structs"
)

func benchmarkVectors(count, dim int, seed uint64) [][]float32 {
	rng := rand.New(rand.NewPCG(seed, seed))
	vectors := make([][]float32, count)
	for i := range vectors {
		vectors[i] = make([]float32, dim)
		for j := range vectors[i] {
			vectors[i][j] = rng.Float32()
		}
	}
	return vectors
}

func benchmarkHNSWIndex(tb testing.TB, count, dim int, cfg Config, seed uint64) (*HNSW, [][]float32) {
	tb.Helper()

	h, err := NewHNSW(cfg)
	if err != nil {
		tb.Fatalf("Failed to create HNSW: %v", err)
	}

	vectors := benchmarkVectors(count, dim, seed)
	h.InsertBatch(vectors)
	return h, vectors
}

func BenchmarkGreedySearchLayer(b *testing.B) {
	cfg := Config{
		M:              16,
		Mmax:           32,
		Mmax0:          64,
		EfConstruction: 100,
		MaxLevel:       16,
		DistanceFunc:   EuclideanDistance,
	}

	h, vectors := benchmarkHNSWIndex(b, 5000, 128, cfg, 42)
	query := vectors[len(vectors)-1]
	entry := h.EntryPoint
	level := entry.Level

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = h.greedySearchLayer(query, entry, level)
	}
}

func BenchmarkSearchLayer(b *testing.B) {
	cfg := Config{
		M:              16,
		Mmax:           32,
		Mmax0:          64,
		EfConstruction: 100,
		MaxLevel:       16,
		DistanceFunc:   EuclideanDistance,
	}

	h, vectors := benchmarkHNSWIndex(b, 5000, 128, cfg, 42)
	query := vectors[len(vectors)-1]
	entry := h.EntryPoint

	for _, ef := range []int{16, 64, 128} {
		b.Run(fmt.Sprintf("ef_%d", ef), func(b *testing.B) {
			resultsBuf := make([]int, 0, ef)
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				resultsBuf = h.searchLayerWithEntriesBuffer(query, []int{entry.ID}, ef, 0, resultsBuf)
			}
		})
	}
}

func BenchmarkUpdateBidirectionalConnections(b *testing.B) {
	cfg := Config{
		M:              16,
		Mmax:           32,
		Mmax0:          64,
		EfConstruction: 100,
		MaxLevel:       4,
		DistanceFunc:   EuclideanDistance,
	}

	h, err := NewHNSW(cfg)
	if err != nil {
		b.Fatalf("Failed to create HNSW: %v", err)
	}

	nodeCount := 256
	vectors := benchmarkVectors(nodeCount+1, 128, 99)
	h.Nodes = make([]*structs.Node, 0, nodeCount+1)
	for i := 0; i < nodeCount; i++ {
		node := structs.NewNode(i, vectors[i], 0, h.MaxLevel, h.Mmax, h.Mmax0)
		baseNeighbors := make([]int, 0, h.Mmax0)
		for j := 1; j <= h.Mmax0; j++ {
			baseNeighbors = append(baseNeighbors, (i+j)%nodeCount)
		}
		node.Neighbors[0] = append(node.Neighbors[0], baseNeighbors...)
		h.Nodes = append(h.Nodes, node)
	}

	q := structs.NewNode(nodeCount, vectors[nodeCount], 0, h.MaxLevel, h.Mmax, h.Mmax0)
	h.Nodes = append(h.Nodes, q)

	neighbors := make([]int, h.M)
	for i := range neighbors {
		neighbors[i] = i
	}

	initialNeighborLists := make([][]int, len(neighbors))
	for i, neighborID := range neighbors {
		initialNeighborLists[i] = append([]int(nil), h.Nodes[neighborID].Neighbors[0]...)
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		b.StopTimer()
		q.Neighbors[0] = q.Neighbors[0][:0]
		for idx, neighborID := range neighbors {
			neighbor := h.Nodes[neighborID]
			neighbor.Neighbors[0] = neighbor.Neighbors[0][:0]
			neighbor.Neighbors[0] = append(neighbor.Neighbors[0], initialNeighborLists[idx]...)
		}
		b.StartTimer()

		h.updateBidirectionalConnections(q, neighbors, 0, h.Mmax0)
	}
}
