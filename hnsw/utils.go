package hnsw

func EuclideanDistance(a, b []float32) float32 {
	var sum0, sum1 float32
	i := 0

	for ; i <= len(a)-2; i += 2 {
		d0 := a[i] - b[i]
		d1 := a[i+1] - b[i+1]

		sum0 += d0 * d0
		sum1 += d1 * d1
	}

	var sum float32
	for ; i < len(a); i++ {
		d := a[i] - b[i]
		sum += d * d
	}

	return sum + sum0 + sum1
}
