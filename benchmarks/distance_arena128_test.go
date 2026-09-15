package benchmarks

import (
	"math"
	"math/rand/v2"
	"testing"

	"dmarro89.github.com/hnsw-go/hnsw"
)

var arena128Sink float32

// distanceArena128Offset tests the code shape we could expose from the
// production vector arena when dim==128: one arena plus integer offsets,
// rather than constructing candidate []float32 slices at every call site.
func distanceArena128Offset(query, arena []float32, off int) float32 {
	_ = arena[off+127]
	var s0, s1, s2, s3 float32
	for i := 0; i < 128; i += 4 {
		d0 := query[i] - arena[off+i]
		d1 := query[i+1] - arena[off+i+1]
		d2 := query[i+2] - arena[off+i+2]
		d3 := query[i+3] - arena[off+i+3]
		s0 += d0 * d0
		s1 += d1 * d1
		s2 += d2 * d2
		s3 += d3 * d3
	}
	return s0 + s1 + s2 + s3
}

func TestDistanceArena128OffsetAgrees(t *testing.T) {
	const count = 257
	rng := rand.New(rand.NewPCG(1282026, 1282027))
	query := make([]float32, 128)
	arena := make([]float32, count*128)
	for i := range query { query[i] = rng.Float32() }
	for i := range arena { arena[i] = rng.Float32() }
	for id := 0; id < count; id++ {
		want := hnsw.EuclideanDistance(query, arena[id*128:(id+1)*128])
		got := distanceArena128Offset(query, arena, id*128)
		delta := math.Abs(float64(got-want))
		if delta > math.Max(1e-5, math.Abs(float64(want))*2e-6) {
			t.Fatalf("id=%d got=%g want=%g delta=%g", id, got, want, delta)
		}
	}
}

func BenchmarkDistanceArena128Offset(b *testing.B) {
	const count = 1 << 14
	rng := rand.New(rand.NewPCG(8128, 8129))
	query := make([]float32, 128)
	arena := make([]float32, count*128)
	ids := make([]int, 4096)
	for i := range query { query[i] = rng.Float32() }
	for i := range arena { arena[i] = rng.Float32() }
	for i := range ids { ids[i] = rng.IntN(count) }

	b.Run("production_slice", func(b *testing.B) {
		b.ReportAllocs(); b.SetBytes(128*2*4)
		var s float32
		for i := 0; i < b.N; i++ {
			id := ids[i&(len(ids)-1)]; off := id*128
			s = hnsw.EuclideanDistance(query, arena[off:off+128])
		}
		arena128Sink = s
	})
	b.Run("fixed128_offset", func(b *testing.B) {
		b.ReportAllocs(); b.SetBytes(128*2*4)
		var s float32
		for i := 0; i < b.N; i++ {
			id := ids[i&(len(ids)-1)]
			s = distanceArena128Offset(query, arena, id*128)
		}
		arena128Sink = s
	})
}
