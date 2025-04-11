package hnsw

func EuclideanDistance(a, b []float32) float32 {
	var sum0, sum1, sum2, sum3 float32
	i := 0

	// Vectorization for 4 elements at a time
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

	// Remaining elements
	var sum float32
	for ; i < len(a); i++ {
		d := a[i] - b[i]
		sum += d * d
	}

	return sum + sum0 + sum1 + sum2 + sum3
}
