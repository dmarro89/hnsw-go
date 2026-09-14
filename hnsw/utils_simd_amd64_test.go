//go:build amd64 && goexperiment.simd

package hnsw

import (
	"math"
	"testing"
)

func TestEuclideanDistanceSIMDAgreesWithScalar(t *testing.T) {
	for _, dim := range []int{1, 3, 4, 7, 8, 15, 16, 17, 32, 128, 384, 768, 1536} {
		a := make([]float32, dim)
		b := make([]float32, dim)
		for i := 0; i < dim; i++ {
			a[i] = float32((i*37)%101) / 101
			b[i] = float32((i*53+7)%103) / 103
		}

		want := euclideanDistanceScalarReference(a, b)
		got := EuclideanDistance(a, b)
		delta := math.Abs(float64(got - want))
		tolerance := math.Max(1e-5, math.Abs(float64(want))*5e-6)
		if delta > tolerance {
			t.Fatalf("dim=%d got=%g want=%g delta=%g tolerance=%g", dim, got, want, delta, tolerance)
		}
	}
}

func euclideanDistanceScalarReference(a, b []float32) float32 {
	var sum0, sum1, sum2, sum3 float32
	i := 0
	for ; i <= len(a)-4; i += 4 {
		d0 := a[i] - b[i]
		d1 := a[i+1] - b[i+1]
		d2 := a[i+2] - b[i+2]
		d3 := a[i+3] - b[i+3]
		sum0 += d0 * d0
		sum1 += d1 * d1
		sum2 += d2 * d2
		sum3 += d3 * d3
	}
	var sum float32
	for ; i < len(a); i++ {
		d := a[i] - b[i]
		sum += d * d
	}
	return sum + sum0 + sum1 + sum2 + sum3
}
