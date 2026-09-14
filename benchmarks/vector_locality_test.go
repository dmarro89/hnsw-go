package benchmarks

import (
	"fmt"
	"math/rand/v2"
	"testing"

	"dmarro89.github.com/hnsw-go/hnsw"
)

var vectorLocalitySink float32

type localityNode struct {
	vector []float32
}

func localityFixture(count, dim int, seed uint64) ([]float32, []localityNode, []float32, []int) {
	rng := rand.New(rand.NewPCG(seed, seed^0x9e3779b97f4a7c15))
	query := make([]float32, dim)
	for i := range query {
		query[i] = rng.Float32()
	}

	arena := make([]float32, count*dim)
	nodes := make([]localityNode, count)
	for id := 0; id < count; id++ {
		vector := arena[id*dim : (id+1)*dim]
		for j := range vector {
			vector[j] = rng.Float32()
		}
		nodes[id].vector = vector
	}

	// A deterministic random walk approximates HNSW's non-sequential neighbor
	// visits while keeping the exact same access sequence for every layout.
	ids := make([]int, count*4)
	for i := range ids {
		ids[i] = rng.IntN(count)
	}
	return query, nodes, arena, ids
}

func BenchmarkVectorDataLocality(b *testing.B) {
	for _, dim := range []int{32, 128, 384} {
		const count = 100000
		query, nodes, arena, ids := localityFixture(count, dim, uint64(23000+dim))

		b.Run(fmt.Sprintf("node_slice/%dd", dim), func(b *testing.B) {
			b.ReportAllocs()
			var sum float32
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				id := ids[i%len(ids)]
				sum += hnsw.EuclideanDistance(query, nodes[id].vector)
			}
			vectorLocalitySink = sum
		})

		b.Run(fmt.Sprintf("contiguous_arena/%dd", dim), func(b *testing.B) {
			b.ReportAllocs()
			var sum float32
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				id := ids[i%len(ids)]
				start := id * dim
				sum += hnsw.EuclideanDistance(query, arena[start:start+dim])
			}
			vectorLocalitySink = sum
		})
	}
}

func BenchmarkVectorLookupOnly(b *testing.B) {
	const (
		count = 100000
		dim   = 128
	)
	_, nodes, arena, ids := localityFixture(count, dim, 24128)
	b.Run("node_slice", func(b *testing.B) {
		var sum float32
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			id := ids[i%len(ids)]
			sum += nodes[id].vector[0]
		}
		vectorLocalitySink = sum
	})
	b.Run("contiguous_arena", func(b *testing.B) {
		var sum float32
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			id := ids[i%len(ids)]
			sum += arena[id*dim]
		}
		vectorLocalitySink = sum
	})
}
