package benchmarks

import (
	"fmt"
	"math/rand/v2"
	"testing"

	"dmarro89.github.com/hnsw-go/hnsw"
)

var distanceBatchSink float32

func distanceBatchArena(query, arena []float32, dim int, ids []int, out []float32) {
	for n, id := range ids {
		start := id * dim
		candidate := arena[start : start+dim]
		var s0, s1, s2, s3 float32
		i := 0
		for ; i <= dim-4; i += 4 {
			d0 := query[i] - candidate[i]
			d1 := query[i+1] - candidate[i+1]
			d2 := query[i+2] - candidate[i+2]
			d3 := query[i+3] - candidate[i+3]
			s0 += d0 * d0
			s1 += d1 * d1
			s2 += d2 * d2
			s3 += d3 * d3
		}
		var tail float32
		for ; i < dim; i++ {
			d := query[i] - candidate[i]
			tail += d * d
		}
		out[n] = s0 + s1 + s2 + s3 + tail
	}
}

func TestDistanceBatchArenaAgrees(t *testing.T) {
	const dim = 128
	vectors := deterministicBuildVectors(257, dim, 9101)
	arena := make([]float32, len(vectors)*dim)
	for i, v := range vectors {
		copy(arena[i*dim:(i+1)*dim], v)
	}
	query := vectors[256]
	ids := []int{0, 7, 31, 128, 255}
	out := make([]float32, len(ids))
	distanceBatchArena(query, arena, dim, ids, out)
	for i, id := range ids {
		want := hnsw.EuclideanDistance(query, arena[id*dim:(id+1)*dim])
		if out[i] != want {
			t.Fatalf("id=%d got=%g want=%g", id, out[i], want)
		}
	}
}

func BenchmarkDistanceBatchArena(b *testing.B) {
	const (
		dim = 128
		count = 100000
	)
	vectors := deterministicBuildVectors(count+1, dim, 9201)
	arena := make([]float32, count*dim)
	for i := 0; i < count; i++ {
		copy(arena[i*dim:(i+1)*dim], vectors[i])
	}
	query := vectors[count]
	rng := rand.New(rand.NewPCG(9301, 9302))
	const maxIDs = 32
	ids := make([]int, maxIDs)
	for i := range ids {
		ids[i] = rng.IntN(count)
	}
	out := make([]float32, maxIDs)

	for _, batch := range []int{4, 8, 16, 32} {
		b.Run(fmt.Sprintf("calls/batch_%d", batch), func(b *testing.B) {
			b.ReportAllocs()
			b.SetBytes(int64(batch * dim * 2 * 4))
			var sink float32
			b.ResetTimer()
			for n := 0; n < b.N; n++ {
				for _, id := range ids[:batch] {
					start := id * dim
					sink += hnsw.EuclideanDistance(query, arena[start:start+dim])
				}
			}
			distanceBatchSink = sink
		})

		b.Run(fmt.Sprintf("batched/batch_%d", batch), func(b *testing.B) {
			b.ReportAllocs()
			b.SetBytes(int64(batch * dim * 2 * 4))
			var sink float32
			b.ResetTimer()
			for n := 0; n < b.N; n++ {
				distanceBatchArena(query, arena, dim, ids[:batch], out[:batch])
				sink += out[batch-1]
			}
			distanceBatchSink = sink
		})
	}
}
