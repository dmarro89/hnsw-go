//go:build amd64 && goexperiment.simd

package hnsw

import "simd/archsimd"

func euclideanDistance(a, b []float32) float32 {
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
