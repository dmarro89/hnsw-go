package benchmarks

import (
    "math"
    "math/rand/v2"
    "testing"

    "dmarro89.github.com/hnsw-go/hnsw"
)

var blockedDistanceSink [4]float32

// block4 layout is [block][dimension][lane], so one query coordinate is reused
// immediately across four candidates. This is benchmark-only evidence for a
// possible production arena-layout change.
func makeBlock4Arena(vectors [][]float32, dim int) []float32 {
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

func distanceBlock4(query, blocked []float32, dim, block int) [4]float32 {
    base := block * dim * 4
    end := base + dim*4
    _ = blocked[end-1]
    var s0, s1, s2, s3 float32
    p := base
    for d := 0; d < dim; d++ {
        q := query[d]
        x0 := q - blocked[p]
        x1 := q - blocked[p+1]
        x2 := q - blocked[p+2]
        x3 := q - blocked[p+3]
        s0 += x0 * x0
        s1 += x1 * x1
        s2 += x2 * x2
        s3 += x3 * x3
        p += 4
    }
    return [4]float32{s0, s1, s2, s3}
}

func TestDistanceBlock4Agrees(t *testing.T) {
    const count, dim = 20, 128
    vectors := deterministicBuildVectors(count, dim, 9090)
    blocked := makeBlock4Arena(vectors, dim)
    q := vectors[19]
    for block := 0; block < count/4; block++ {
        got := distanceBlock4(q, blocked, dim, block)
        for lane := 0; lane < 4; lane++ {
            want := hnsw.EuclideanDistance(q, vectors[block*4+lane])
            delta := math.Abs(float64(got[lane] - want))
            tol := math.Max(1e-5, math.Abs(float64(want))*2e-6)
            if delta > tol { t.Fatalf("block=%d lane=%d got=%g want=%g", block, lane, got[lane], want) }
        }
    }
}

func BenchmarkDistanceBlock4(b *testing.B) {
    const count, dim = 100000, 128
    vectors := deterministicBuildVectors(count, dim, 9191)
    blocked := makeBlock4Arena(vectors, dim)
    q := vectors[count-1]
    rng := rand.New(rand.NewPCG(9292, 9393))
    blocks := make([]int, 1<<16)
    for i := range blocks { blocks[i] = rng.IntN(count / 4) }

    b.Run("production_four_slices", func(b *testing.B) {
        b.ReportAllocs(); b.SetBytes(int64(dim * 4 * 4))
        var out [4]float32
        b.ResetTimer()
        for i := 0; i < b.N; i++ {
            block := blocks[i&(len(blocks)-1)]
            id := block * 4
            out[0] = hnsw.EuclideanDistance(q, vectors[id])
            out[1] = hnsw.EuclideanDistance(q, vectors[id+1])
            out[2] = hnsw.EuclideanDistance(q, vectors[id+2])
            out[3] = hnsw.EuclideanDistance(q, vectors[id+3])
        }
        blockedDistanceSink = out
    })
    b.Run("blocked_four", func(b *testing.B) {
        b.ReportAllocs(); b.SetBytes(int64(dim * 4 * 4))
        var out [4]float32
        b.ResetTimer()
        for i := 0; i < b.N; i++ { out = distanceBlock4(q, blocked, dim, blocks[i&(len(blocks)-1)]) }
        blockedDistanceSink = out
    })
}
