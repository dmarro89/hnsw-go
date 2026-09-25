//go:build goexperiment.simd

package benchmarks

import (
	"fmt"
	"math"
	"math/rand/v2"
	"testing"

	"dmarro89.github.com/hnsw-go/hnsw"
	"simd"
)

func distancePortableSIMD(a, b []float32) float32 {
	var acc simd.Float32s
	i := 0
	width := acc.Len()
	for ; i+width <= len(a); i += width {
		av := simd.LoadFloat32s(a[i : i+width])
		bv := simd.LoadFloat32s(b[i : i+width])
		d := av.Sub(bv)
		acc = d.MulAdd(d, acc)
	}
	if i < len(a) {
		av, _ := simd.LoadFloat32sPart(a[i:])
		bv, _ := simd.LoadFloat32sPart(b[i:])
		d := av.Sub(bv)
		acc = d.MulAdd(d, acc)
	}

	var lanes [64]float32
	if width > len(lanes) {
		panic("portable SIMD width exceeds reduction scratch")
	}
	acc.Store(lanes[:width])
	var sum float32
	for _, v := range lanes[:width] {
		sum += v
	}
	return sum
}

func TestDistanceKernelPortableSIMDAgrees(t *testing.T) {
	for _, dim := range []int{1, 3, 4, 7, 8, 15, 16, 17, 32, 128, 384, 768, 1536} {
		a, b := deterministicDistanceVectors(dim, uint64(dim+10070))
		want := hnsw.EuclideanDistance(a, b)
		got := distancePortableSIMD(a, b)
		delta := math.Abs(float64(got - want))
		tol := math.Max(1e-5, math.Abs(float64(want))*5e-6)
		if delta > tol {
			t.Fatalf("dim=%d portable_simd got=%g want=%g delta=%g tolerance=%g", dim, got, want, delta, tol)
		}
	}
}

func BenchmarkDistanceKernelPortableSIMD(b *testing.B) {
	kernels := []struct {
		name string
		fn   distanceKernelFn
	}{
		{"production_scalar4", hnsw.EuclideanDistance},
		{"portable_simd", distancePortableSIMD},
	}
	for _, dim := range []int{32, 128, 384, 768, 1536} {
		a, v := deterministicDistanceVectors(dim, uint64(dim+11080))
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
func invokePortableDistanceIndirect(fn distanceKernelFn, a, b []float32) float32 {
	return fn(a, b)
}

func BenchmarkDistanceKernelPortableSIMDRandomAccess(b *testing.B) {
	const (
		count     = 100_000
		dim       = 128
		indexRing = 1 << 16
	)
	vectors := deterministicBuildVectors(count, dim, 12090)
	query, _ := deterministicDistanceVectors(dim, 12191)
	indices := make([]int, indexRing)
	rng := rand.New(rand.NewPCG(12292, 12393))
	for i := range indices {
		indices[i] = rng.IntN(count)
	}

	kernels := []struct {
		name string
		fn   distanceKernelFn
	}{
		{"production_scalar4", hnsw.EuclideanDistance},
		{"portable_simd", distancePortableSIMD},
	}
	for _, kernel := range kernels {
		b.Run(kernel.name, func(b *testing.B) {
			b.ReportAllocs()
			b.SetBytes(dim * 2 * 4)
			var sum float32
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				candidate := vectors[indices[i&(indexRing-1)]]
				sum += invokePortableDistanceIndirect(kernel.fn, query, candidate)
			}
			distanceKernelSink = sum
		})
	}
}

func BenchmarkDistanceKernelPortableSIMDBuildImpact(b *testing.B) {
	benchmarkBuildWithDistance(b, "production_scalar4", hnsw.EuclideanDistance)
	benchmarkBuildWithDistance(b, "portable_simd", distancePortableSIMD)
}
