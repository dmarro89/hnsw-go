package benchmarks

import (
	"math/rand/v2"
	"testing"

	"dmarro89.github.com/hnsw-go/structs"
)

var nodeStorageSink int

func benchmarkLevels(n int) []int {
	r := rand.New(rand.NewPCG(4141, 4242))
	levels := make([]int, n)
	for i := range levels {
		// M=16 level distribution: overwhelmingly level zero, matching production.
		for x := r.Float64(); x < 1.0/16.0; x *= 16.0 {
			levels[i]++
		}
	}
	return levels
}

func BenchmarkNodeNeighborStorage(b *testing.B) {
	const n, mmax, mmax0 = 100000, 32, 64
	levels := benchmarkLevels(n)

	b.Run("production_per_node", func(b *testing.B) {
		b.ReportAllocs()
		for k := 0; k < b.N; k++ {
			nodes := make([]structs.Node, n)
			for i, level := range levels {
				structs.InitNode(&nodes[i], i, nil, level, 16, mmax, mmax0)
			}
			nodeStorageSink = len(nodes[n-1].Neighbors)
		}
	})

	b.Run("single_bulk_arena", func(b *testing.B) {
		b.ReportAllocs()
		totalHeaders, totalIDs := 0, 0
		for _, level := range levels {
			totalHeaders += level + 1
			totalIDs += mmax0 + level*mmax
		}
		b.ResetTimer()
		for k := 0; k < b.N; k++ {
			nodes := make([]structs.Node, n)
			headers := make([][]int, totalHeaders)
			ids := make([]int, totalIDs)
			ho, io := 0, 0
			for i, level := range levels {
				n := &nodes[i]
				n.ID, n.Level = i, level
				n.Neighbors = headers[ho : ho+level+1]
				ho += level + 1
				for lc := range n.Neighbors {
					cap := mmax
					if lc == 0 { cap = mmax0 }
					n.Neighbors[lc] = ids[io:io:io+cap]
					io += cap
				}
			}
			nodeStorageSink = len(nodes[n-1].Neighbors)
		}
	})
}
