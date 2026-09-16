package benchmarks

import (
	"math"
	"math/rand/v2"
	"testing"

	"dmarro89.github.com/hnsw-go/hnsw"
)

var blockedGatherSink [4]float32

func makeGatherBlock4Arena(vectors [][]float32, dim int) []float32 {
	blocks := (len(vectors) + 3) / 4
	out := make([]float32, blocks*dim*4)
	for id, v := range vectors {
		block, lane := id/4, id%4
		base := block * dim * 4
		for d := 0; d < dim; d++ {
			out[base+d*4+lane] = v[d]
		}
	}
	return out
}

// distanceGather4 models the access pattern needed by real graph traversal:
// four arbitrary candidate IDs, usually from four different storage blocks.
// Query coordinates are still reused, but candidate loads are independent.
func distanceGather4(query, blocked []float32, dim int, ids [4]int) [4]float32 {
	var base [4]int
	var lane [4]int
	for j, id := range ids {
		base[j] = (id / 4) * dim * 4
		lane[j] = id % 4
		_ = blocked[base[j]+(dim-1)*4+lane[j]]
	}
	var s0, s1, s2, s3 float32
	for d := 0; d < dim; d++ {
		q := query[d]
		off := d * 4
		x0 := q - blocked[base[0]+off+lane[0]]
		x1 := q - blocked[base[1]+off+lane[1]]
		x2 := q - blocked[base[2]+off+lane[2]]
		x3 := q - blocked[base[3]+off+lane[3]]
		s0 += x0 * x0
		s1 += x1 * x1
		s2 += x2 * x2
		s3 += x3 * x3
	}
	return [4]float32{s0, s1, s2, s3}
}

func TestDistanceGather4Agrees(t *testing.T) {
	const count, dim = 100, 128
	vectors := deterministicBuildVectors(count, dim, 9494)
	blocked := makeGatherBlock4Arena(vectors, dim)
	q := vectors[99]
	groups := [][4]int{{1, 17, 42, 88}, {3, 4, 63, 97}, {12, 35, 66, 91}}
	for _, ids := range groups {
		got := distanceGather4(q, blocked, dim, ids)
		for j, id := range ids {
			want := hnsw.EuclideanDistance(q, vectors[id])
			delta := math.Abs(float64(got[j] - want))
			tol := math.Max(1e-5, math.Abs(float64(want))*2e-6)
			if delta > tol { t.Fatalf("id=%d got=%g want=%g", id, got[j], want) }
		}
	}
}

func BenchmarkDistanceGather4(b *testing.B) {
	const count, dim = 100000, 128
	vectors := deterministicBuildVectors(count, dim, 9595)
	blocked := makeGatherBlock4Arena(vectors, dim)
	q := vectors[count-1]
	rng := rand.New(rand.NewPCG(9696, 9797))
	groups := make([][4]int, 1<<14)
	for i := range groups {
		for j := 0; j < 4; j++ { groups[i][j] = rng.IntN(count-1) }
	}

	b.Run("production_four_random_slices", func(b *testing.B) {
		b.ReportAllocs(); b.SetBytes(int64(dim * 4 * 4))
		var out [4]float32
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			ids := groups[i&(len(groups)-1)]
			out[0] = hnsw.EuclideanDistance(q, vectors[ids[0]])
			out[1] = hnsw.EuclideanDistance(q, vectors[ids[1]])
			out[2] = hnsw.EuclideanDistance(q, vectors[ids[2]])
			out[3] = hnsw.EuclideanDistance(q, vectors[ids[3]])
		}
		blockedGatherSink = out
	})
	b.Run("blocked_four_random_gather", func(b *testing.B) {
		b.ReportAllocs(); b.SetBytes(int64(dim * 4 * 4))
		var out [4]float32
		b.ResetTimer()
		for i := 0; i < b.N; i++ { out = distanceGather4(q, blocked, dim, groups[i&(len(groups)-1)]) }
		blockedGatherSink = out
	})
}
