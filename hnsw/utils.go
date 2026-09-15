package hnsw

func EuclideanDistance(a, b []float32) float32 {
	var sum0, sum1, sum2, sum3 float32
	var sum4, sum5, sum6, sum7 float32
	i := 0

	// Eight independent accumulators expose more instruction-level parallelism
	// to the scalar amd64 backend while preserving the existing squared-L2
	// semantics and deterministic accumulation order within each lane.
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
		sum4 += d4 * d4
		sum5 += d5 * d5
		sum6 += d6 * d6
		sum7 += d7 * d7
	}

	var sum float32
	for ; i < len(a); i++ {
		d := a[i] - b[i]
		sum += d * d
	}

	return sum + sum0 + sum1 + sum2 + sum3 + sum4 + sum5 + sum6 + sum7
}
