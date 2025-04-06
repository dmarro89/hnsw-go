package structs

import (
	"math"
)

// L2Norm calculates the Euclidean (L2) norm of a vector.
// This optimized implementation avoids unnecessary allocations and uses
// specialized math functions for better performance.
//
// Parameters:
//   - vector: slice of float32 values representing the vector
//
// Returns:
//   - float32: the L2 norm of the vector
func L2Norm(vector []float32) float32 {
	if len(vector) == 0 {
		return 0
	}

	// For very small vectors, unrolled calculation can be faster
	switch len(vector) {
	case 1:
		return float32(math.Abs(float64(vector[0])))
	case 2:
		return float32(math.Hypot(float64(vector[0]), float64(vector[1])))
	}

	// For larger vectors, use squared sum method with compiler optimizations
	var sumSquared float32

	// Process 4 elements at a time when possible (SIMD optimization friendly)
	i := 0
	for i <= len(vector)-4 {
		sumSquared += vector[i]*vector[i] +
			vector[i+1]*vector[i+1] +
			vector[i+2]*vector[i+2] +
			vector[i+3]*vector[i+3]
		i += 4
	}

	// Handle remaining elements
	for ; i < len(vector); i++ {
		sumSquared += vector[i] * vector[i]
	}

	// Use faster specialized square root when possible
	return float32(math.Sqrt(float64(sumSquared)))
}

// NormalizeVector transforms a vector to have unit length (L2 norm = 1)
// while preserving its direction.
//
// Parameters:
//   - vector: slice of float32 values representing the vector
//
// Returns:
//   - []float32: a new normalized vector
//   - error: if normalization is not possible (zero vector)
func NormalizeVector(vector []float32) []float32 {
	norm := L2Norm(vector)

	if norm == 0 {
		return vector
	}

	normalized := make([]float32, len(vector))
	for i, val := range vector {
		normalized[i] = val / norm
	}

	return normalized
}