package hnsw

import (
	"math/rand"
	"testing"
)

// euclideanDistanceBatch4Arena computes four squared-L2 distances together.
// It deliberately keeps the production row-major arena unchanged: this isolates
// whether reusing each query load across several candidates is valuable before
// considering a more invasive blocked/interleaved arena layout.
func euclideanDistanceBatch4Arena(query, arena []float32, dim int, ids [4]int) [4]float32 {
	base0 := ids[0] * dim
	base1 := ids[1] * dim
	base2 := ids[2] * dim
	base3 := ids[3] * dim
	var s0, s1, s2, s3 float32
	for i, q := range query {
		d0 := q - arena[base0+i]
		d1 := q - arena[base1+i]
		d2 := q - arena[base2+i]
		d3 := q - arena[base3+i]
		s0 += d0 * d0
		s1 += d1 * d1
		s2 += d2 * d2
		s3 += d3 * d3
	}
	return [4]float32{s0, s1, s2, s3}
}

func BenchmarkArenaDistanceBatch4(b *testing.B) {
	const n, dim = 100000, 128
	r := rand.New(rand.NewSource(42))
	arena := make([]float32, n*dim)
	query := make([]float32, dim)
	for i := range arena { arena[i] = r.Float32() }
	for i := range query { query[i] = r.Float32() }
	ids := make([]int, 4096)
	for i := range ids { ids[i] = r.Intn(n) }

	b.Run("scalar-production", func(b *testing.B) {
		var sink float32
		b.ReportAllocs()
		b.SetBytes(int64(len(ids) * dim * 4))
		for k := 0; k < b.N; k++ {
			for _, id := range ids {
				sink += EuclideanDistance(query, arenaVector(arena, dim, id))
			}
		}
		_ = sink
	})

	b.Run("batch4-query-reuse", func(b *testing.B) {
		var sink float32
		b.ReportAllocs()
		b.SetBytes(int64(len(ids) * dim * 4))
		for k := 0; k < b.N; k++ {
			for i := 0; i < len(ids); i += 4 {
				group := [4]int{ids[i], ids[i+1], ids[i+2], ids[i+3]}
				d := euclideanDistanceBatch4Arena(query, arena, dim, group)
				sink += d[0] + d[1] + d[2] + d[3]
			}
		}
		_ = sink
	})
}

func TestEuclideanDistanceBatch4Arena(t *testing.T) {
	const n, dim = 16, 128
	r := rand.New(rand.NewSource(7))
	arena := make([]float32, n*dim)
	query := make([]float32, dim)
	for i := range arena { arena[i] = r.Float32() }
	for i := range query { query[i] = r.Float32() }
	ids := [4]int{1, 5, 9, 15}
	got := euclideanDistanceBatch4Arena(query, arena, dim, ids)
	for i, id := range ids {
		want := EuclideanDistance(query, arenaVector(arena, dim, id))
		// Reassociation changes the final float32 rounding slightly; the
		// experimental kernel must nevertheless remain numerically equivalent.
		delta := got[i] - want
		if delta < 0 { delta = -delta }
		if delta > 1e-4 {
			t.Fatalf("id %d: got %g want %g delta %g", id, got[i], want, delta)
		}
	}
}
