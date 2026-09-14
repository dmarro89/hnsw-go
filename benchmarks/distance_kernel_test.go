package benchmarks

import (
	"fmt"
	"math"
	"math/rand/v2"
	"testing"

	"dmarro89.github.com/hnsw-go/hnsw"
)

var distanceKernelSink float32

type distanceKernelFn func([]float32, []float32) float32

func distanceScalar8FourAcc(a, b []float32) float32 {
	var sum0, sum1, sum2, sum3 float32
	i := 0
	for ; i <= len(a)-8; i += 8 {
		d0 := a[i] - b[i]
		d1 := a[i+1] - b[i+1]
		d2 := a[i+2] - b[i+2]
		d3 := a[i+3] - b[i+3]
		d4 := a[i+4] - b[i+4]
		d5 := a[i+5] - b[i+5]
		d6 := a[i+6] - b[i+6]
		d7 := a[i+7] - b[i+7]
		sum0 += d0 * d0
		sum1 += d1 * d1
		sum2 += d2 * d2
		sum3 += d3 * d3
		sum0 += d4 * d4
		sum1 += d5 * d5
		sum2 += d6 * d6
		sum3 += d7 * d7
	}
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
	var tail float32
	for ; i < len(a); i++ {
		d := a[i] - b[i]
		tail += d * d
	}
	return sum0 + sum1 + sum2 + sum3 + tail
}

func distanceScalar16FourAcc(a, b []float32) float32 {
	var sum0, sum1, sum2, sum3 float32
	i := 0
	for ; i <= len(a)-16; i += 16 {
		for j := 0; j < 16; j += 4 {
			d0 := a[i+j] - b[i+j]
			d1 := a[i+j+1] - b[i+j+1]
			d2 := a[i+j+2] - b[i+j+2]
			d3 := a[i+j+3] - b[i+j+3]
			sum0 += d0 * d0
			sum1 += d1 * d1
			sum2 += d2 * d2
			sum3 += d3 * d3
		}
	}
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
	var tail float32
	for ; i < len(a); i++ {
		d := a[i] - b[i]
		tail += d * d
	}
	return sum0 + sum1 + sum2 + sum3 + tail
}

func distanceScalar8EightAcc(a, b []float32) float32 {
	var s0, s1, s2, s3, s4, s5, s6, s7 float32
	i := 0
	for ; i <= len(a)-8; i += 8 {
		d0 := a[i] - b[i]
		d1 := a[i+1] - b[i+1]
		d2 := a[i+2] - b[i+2]
		d3 := a[i+3] - b[i+3]
		d4 := a[i+4] - b[i+4]
		d5 := a[i+5] - b[i+5]
		d6 := a[i+6] - b[i+6]
		d7 := a[i+7] - b[i+7]
		s0 += d0 * d0
		s1 += d1 * d1
		s2 += d2 * d2
		s3 += d3 * d3
		s4 += d4 * d4
		s5 += d5 * d5
		s6 += d6 * d6
		s7 += d7 * d7
	}
	var tail float32
	for ; i < len(a); i++ {
		d := a[i] - b[i]
		tail += d * d
	}
	return s0 + s1 + s2 + s3 + s4 + s5 + s6 + s7 + tail
}

func distanceScalar16EightAcc(a, b []float32) float32 {
	var s0, s1, s2, s3, s4, s5, s6, s7 float32
	i := 0
	for ; i <= len(a)-16; i += 16 {
		for j := 0; j < 16; j += 8 {
			d0 := a[i+j] - b[i+j]
			d1 := a[i+j+1] - b[i+j+1]
			d2 := a[i+j+2] - b[i+j+2]
			d3 := a[i+j+3] - b[i+j+3]
			d4 := a[i+j+4] - b[i+j+4]
			d5 := a[i+j+5] - b[i+j+5]
			d6 := a[i+j+6] - b[i+j+6]
			d7 := a[i+j+7] - b[i+j+7]
			s0 += d0 * d0
			s1 += d1 * d1
			s2 += d2 * d2
			s3 += d3 * d3
			s4 += d4 * d4
			s5 += d5 * d5
			s6 += d6 * d6
			s7 += d7 * d7
		}
	}
	var tail float32
	for ; i < len(a); i++ {
		d := a[i] - b[i]
		tail += d * d
	}
	return s0 + s1 + s2 + s3 + s4 + s5 + s6 + s7 + tail
}

func deterministicDistanceVectors(dim int, seed uint64) ([]float32, []float32) {
	rng := rand.New(rand.NewPCG(seed, seed^0x9e3779b97f4a7c15))
	a := make([]float32, dim)
	b := make([]float32, dim)
	for i := range a {
		a[i] = rng.Float32()
		b[i] = rng.Float32()
	}
	return a, b
}

func TestDistanceKernelVariantsAgree(t *testing.T) {
	kernels := []struct {
		name string
		fn   distanceKernelFn
	}{
		{"scalar8_four_acc", distanceScalar8FourAcc},
		{"scalar16_four_acc", distanceScalar16FourAcc},
		{"scalar8_eight_acc", distanceScalar8EightAcc},
		{"scalar16_eight_acc", distanceScalar16EightAcc},
	}
	for _, dim := range []int{1, 3, 4, 7, 8, 15, 16, 17, 32, 128, 384, 768, 1536} {
		a, b := deterministicDistanceVectors(dim, uint64(dim+2026))
		want := hnsw.EuclideanDistance(a, b)
		for _, kernel := range kernels {
			got := kernel.fn(a, b)
			delta := float64(got - want)
			if delta < 0 {
				delta = -delta
			}
			tol := math.Max(1e-5, math.Abs(float64(want))*2e-6)
			if delta > tol {
				t.Fatalf("dim=%d kernel=%s got=%g want=%g delta=%g tolerance=%g", dim, kernel.name, got, want, delta, tol)
			}
		}
	}
}

func BenchmarkDistanceKernel(b *testing.B) {
	kernels := []struct {
		name string
		fn   distanceKernelFn
	}{
		{"production_scalar4", hnsw.EuclideanDistance},
		{"scalar8_four_acc", distanceScalar8FourAcc},
		{"scalar16_four_acc", distanceScalar16FourAcc},
		{"scalar8_eight_acc", distanceScalar8EightAcc},
		{"scalar16_eight_acc", distanceScalar16EightAcc},
	}
	for _, dim := range []int{32, 128, 384, 768, 1536} {
		a, v := deterministicDistanceVectors(dim, uint64(dim+4040))
		for _, kernel := range kernels {
			b.Run(fmt.Sprintf("%s/%dd", kernel.name, dim), func(b *testing.B) {
				b.ReportAllocs()
				b.SetBytes(int64(dim * 2 * 4))
				var result float32
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					result = kernel.fn(a, v)
				}
				distanceKernelSink = result
			})
		}
	}
}

func benchmarkBuildWithDistance(b *testing.B, name string, distance distanceKernelFn) {
	const (
		count = 20000
		dim   = 128
	)
	vectors := benchmarkVectors(count, dim, 6060)
	b.Run(name, func(b *testing.B) {
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			cfg := hnsw.Config{M: 16, Mmax: 16, Mmax0: 32, EfConstruction: 64, MaxLevel: 16, DistanceFunc: distance}
			index, err := hnsw.NewHNSW(cfg)
			if err != nil {
				b.Fatal(err)
			}
			index.RandFunc = rand.New(rand.NewPCG(5050, 5050)).Float64
			index.InsertBatch(vectors)
		}
	})
}

func BenchmarkDistanceKernelBuildImpact(b *testing.B) {
	benchmarkBuildWithDistance(b, "production_scalar4", hnsw.EuclideanDistance)
	benchmarkBuildWithDistance(b, "scalar8_four_acc", distanceScalar8FourAcc)
	benchmarkBuildWithDistance(b, "scalar16_four_acc", distanceScalar16FourAcc)
	benchmarkBuildWithDistance(b, "scalar8_eight_acc", distanceScalar8EightAcc)
	benchmarkBuildWithDistance(b, "scalar16_eight_acc", distanceScalar16EightAcc)
}
