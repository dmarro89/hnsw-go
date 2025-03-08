package hnsw

import (
	"math"
	"testing"
)

func TestEuclideanDistance(t *testing.T) {
	tests := []struct {
		name     string
		vec1     []float32
		vec2     []float32
		expected float32
	}{
		{
			name:     "Same vectors",
			vec1:     []float32{1.0, 2.0, 3.0},
			vec2:     []float32{1.0, 2.0, 3.0},
			expected: 0.0,
		},
		{
			name:     "Different vectors",
			vec1:     []float32{1.0, 2.0, 3.0},
			vec2:     []float32{4.0, 5.0, 6.0},
			expected: 27.0, // (4-1)^2 + (5-2)^2 + (6-3)^2 = 9 + 9 + 9 = 27
		},
		{
			name:     "Negative values",
			vec1:     []float32{-1.0, -2.0, -3.0},
			vec2:     []float32{1.0, 2.0, 3.0},
			expected: 56.0, // (1-(-1))^2 + (2-(-2))^2 + (3-(-3))^2 = 4 + 16 + 36 = 56
		},
		{
			name:     "Zero vector",
			vec1:     []float32{0.0, 0.0, 0.0},
			vec2:     []float32{1.0, 1.0, 1.0},
			expected: 3.0, // (1-0)^2 + (1-0)^2 + (1-0)^2 = 3
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := EuclideanDistance(tt.vec1, tt.vec2)
			if math.Abs(float64(result-tt.expected)) > 1e-6 {
				t.Errorf("EuclideanDistance() = %v, expected %v", result, tt.expected)
			}
		})
	}
}

func BenchmarkEuclideanDistance(b *testing.B) {
	vec1 := make([]float32, 128)
	vec2 := make([]float32, 128)
	for i := 0; i < len(vec1); i++ {
		vec1[i] = float32(i) / 128.0
		vec2[i] = float32(i*i) / 128.0
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		EuclideanDistance(vec1, vec2)
	}
}
