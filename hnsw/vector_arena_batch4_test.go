package hnsw

import (
	"math"
	"testing"
)

func TestDistanceArena4AgreesWithEuclideanDistance(t *testing.T) {
	const dim = 128
	vectors := make([][]float32, 12)
	for i := range vectors {
		vectors[i] = make([]float32, dim)
		for d := range vectors[i] {
			vectors[i][d] = float32((i+1)*(d+3)%37) / 37
		}
	}
	blocked := makeBlockedArena4(vectors, dim)
	ids := [4]int{1, 4, 7, 10}
	got := distanceArena4(vectors[11], blocked, dim, ids)
	for lane, id := range ids {
		want := EuclideanDistance(vectors[11], vectors[id])
		delta := math.Abs(float64(got[lane] - want))
		tol := math.Max(1e-5, math.Abs(float64(want))*2e-6)
		if delta > tol {
			t.Fatalf("lane=%d id=%d got=%g want=%g", lane, id, got[lane], want)
		}
	}
}
