//go:build goexperiment.simd && amd64

package benchmarks

import (
	"fmt"
	"math"
	"math/rand/v2"
	"testing"

	"dmarro89.github.com/hnsw-go/hnsw"
	"simd/archsimd"
)

func distanceSIMD128(a, b []float32) float32 {
	var acc archsimd.Float32x4
	i := 0
	for ; i <= len(a)-4; i += 4 {
		av := archsimd.LoadFloat32x4Slice(a[i:])
		bv := archsimd.LoadFloat32x4Slice(b[i:])
		d := av.Sub(bv)
		acc = d.MulAdd(d, acc)
	}
	var lanes [4]float32
	acc.StoreSlice(lanes[:])
	sum := lanes[0] + lanes[1] + lanes[2] + lanes[3]
	for ; i < len(a); i++ {
		d := a[i] - b[i]
		sum += d * d
	}
	return sum
}

func distanceSIMD256(a, b []float32) float32 {
	var acc archsimd.Float32x8
	i := 0
	for ; i <= len(a)-8; i += 8 {
		av := archsimd.LoadFloat32x8Slice(a[i:])
		bv := archsimd.LoadFloat32x8Slice(b[i:])
		d := av.Sub(bv)
		acc = d.MulAdd(d, acc)
	}
	var lanes [8]float32
	acc.StoreSlice(lanes[:])
	sum := lanes[0] + lanes[1] + lanes[2] + lanes[3] + lanes[4] + lanes[5] + lanes[6] + lanes[7]
	for ; i < len(a); i++ {
		d := a[i] - b[i]
		sum += d * d
	}
	return sum
}

func TestDistanceKernelSIMDAgrees(t *testing.T) {
	for _, dim := range []int{1, 3, 4, 7, 8, 15, 16, 17, 32, 128, 384, 768, 1536} {
		a, b := deterministicDistanceVectors(dim, uint64(dim+7070))
		want := hnsw.EuclideanDistance(a, b)
		for _, tc := range []struct {
			name string
			fn   distanceKernelFn
		}{
			{"simd128_fma", distanceSIMD128},
			{"simd256_fma", distanceSIMD256},
		} {
			got := tc.fn(a, b)
			delta := math.Abs(float64(got - want))
			tol := math.Max(1e-5, math.Abs(float64(want))*5e-6)
			if delta > tol {
				t.Fatalf("dim=%d kernel=%s got=%g want=%g delta=%g tolerance=%g", dim, tc.name, got, want, delta, tol)
			}
		}
	}
}

func BenchmarkDistanceKernelSIMD(b *testing.B) {
	kernels := []struct {
		name string
		fn   distanceKernelFn
	}{
		{"production_scalar4", hnsw.EuclideanDistance},
		{"simd128_fma", distanceSIMD128},
		{"simd256_fma", distanceSIMD256},
	}
	for _, dim := range []int{32, 128, 384, 768, 1536} {
		a, v := deterministicDistanceVectors(dim, uint64(dim+8080))
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

//go:noinline
func invokeDistanceIndirect(fn distanceKernelFn, a, b []float32) float32 {
	return fn(a, b)
}

func BenchmarkDistanceKernelSIMDRandomAccess(b *testing.B) {
	const (
		count     = 100_000
		dim       = 128
		indexRing = 1 << 16
	)
	vectors := deterministicBuildVectors(count, dim, 9090)
	query, _ := deterministicDistanceVectors(dim, 9191)
	indices := make([]int, indexRing)
	rng := rand.New(rand.NewPCG(9292, 9393))
	for i := range indices {
		indices[i] = rng.IntN(count)
	}

	kernels := []struct {
		name string
		fn   distanceKernelFn
	}{
		{"production_scalar4", hnsw.EuclideanDistance},
		{"simd256_fma", distanceSIMD256},
	}
	for _, kernel := range kernels {
		b.Run(kernel.name, func(b *testing.B) {
			b.ReportAllocs()
			b.SetBytes(dim * 2 * 4)
			var sum float32
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				candidate := vectors[indices[i&(indexRing-1)]]
				sum += invokeDistanceIndirect(kernel.fn, query, candidate)
			}
			distanceKernelSink = sum
		})
	}
}

func BenchmarkDistanceKernelSIMDBuildImpact(b *testing.B) {
	benchmarkBuildWithDistance(b, "production_scalar4", hnsw.EuclideanDistance)
	benchmarkBuildWithDistance(b, "simd128_fma", distanceSIMD128)
	benchmarkBuildWithDistance(b, "simd256_fma", distanceSIMD256)
}
