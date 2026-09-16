package hnsw

import (
	"math/rand/v2"
	"testing"
)

var thresholdBenchSink float32

func thresholdBenchData(n, dim int, acceptEvery int) ([][]float32, [][]float32, []float32) {
	r := rand.New(rand.NewPCG(9090, 9090))
	a := make([][]float32, n)
	b := make([][]float32, n)
	limits := make([]float32, n)
	for i := range n {
		a[i] = make([]float32, dim)
		b[i] = make([]float32, dim)
		for j := range dim {
			a[i][j] = r.Float32()
			b[i][j] = r.Float32()
		}
		d := EuclideanDistance(a[i], b[i])
		if i%acceptEvery == 0 {
			limits[i] = d * 1.05
		} else {
			// Representative rejection threshold. A second benchmark below uses
			// a tighter 0.8 ratio to bound sensitivity to how late rejection occurs.
			limits[i] = d * 0.5
		}
	}
	return a, b, limits
}

func benchmarkThresholdDistance(b *testing.B, rejectRatio float32) {
	const n = 4096
	a, vectors, limits := thresholdBenchData(n, 128, 5) // 80% rejected
	if rejectRatio != 0.5 {
		for i := range limits {
			d := EuclideanDistance(a[i], vectors[i])
			if i%5 != 0 {
				limits[i] = d * rejectRatio
			}
	}
	}
	b.ReportAllocs()
	b.ResetTimer()
	var sink float32
	for i := 0; i < b.N; i++ {
		idx := i & (n - 1)
		d, _ := euclideanDistanceWithLimit(a[idx], vectors[idx], limits[idx])
		sink += d
	}
	thresholdBenchSink = sink
}

func BenchmarkEuclideanThresholdEarlyAbandon50(b *testing.B) { benchmarkThresholdDistance(b, 0.5) }
func BenchmarkEuclideanThresholdEarlyAbandon80(b *testing.B) { benchmarkThresholdDistance(b, 0.8) }

func BenchmarkEuclideanThresholdFullDistance(b *testing.B) {
	const n = 4096
	a, vectors, _ := thresholdBenchData(n, 128, 5)
	b.ReportAllocs()
	b.ResetTimer()
	var sink float32
	for i := 0; i < b.N; i++ {
		idx := i & (n - 1)
		sink += EuclideanDistance(a[idx], vectors[idx])
	}
	thresholdBenchSink = sink
}
