package hnsw

// EuclideanDistance calculates the Euclidean distance between two vectors.
func EuclideanDistance(vec1, vec2 []float32) float32 {
	sum := float32(0.0)
	for i := range vec1 {
		diff := vec1[i] - vec2[i]
		sum += diff * diff
	}
	return sum
}
