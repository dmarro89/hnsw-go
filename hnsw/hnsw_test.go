package hnsw

import (
	"errors"
	"fmt"
	"math/rand/v2"
	"testing"
)

func TestDefaultConfig(t *testing.T) {
	cfg := DefaultConfig()

	if cfg.M != 16 {
		t.Errorf("Expected M to be 16, got %d", cfg.M)
	}
	if cfg.Mmax != 32 {
		t.Errorf("Expected Mmax to be 32, got %d", cfg.Mmax)
	}
	if cfg.Mmax0 != 64 {
		t.Errorf("Expected Mmax0 to be 64, got %d", cfg.Mmax0)
	}
	if cfg.EfConstruction != 200 {
		t.Errorf("Expected EfConstruction to be 200, got %d", cfg.EfConstruction)
	}
	if cfg.MaxLevel != 16 {
		t.Errorf("Expected MaxLevel to be 16, got %d", cfg.MaxLevel)
	}
	if cfg.DistanceFunc == nil {
		t.Error("Expected DistanceFunc to be non-nil")
	}
}

func TestValidateConfig(t *testing.T) {
	tests := []struct {
		cfg      Config
		expected error
	}{
		{Config{M: 0, Mmax: 32, Mmax0: 64, EfConstruction: 200, MaxLevel: 16, DistanceFunc: EuclideanDistance}, errors.New("M must be positive")},
		{Config{M: 16, Mmax: 0, Mmax0: 64, EfConstruction: 200, MaxLevel: 16, DistanceFunc: EuclideanDistance}, errors.New("Mmax must be positive")},
		{Config{M: 16, Mmax: 32, Mmax0: 0, EfConstruction: 200, MaxLevel: 16, DistanceFunc: EuclideanDistance}, errors.New("Mmax0 must be positive")},
		{Config{M: 16, Mmax: 32, Mmax0: 64, EfConstruction: 0, MaxLevel: 16, DistanceFunc: EuclideanDistance}, errors.New("EfConstruction must be positive")},
		{Config{M: 16, Mmax: 32, Mmax0: 64, EfConstruction: 200, MaxLevel: 0, DistanceFunc: EuclideanDistance}, errors.New("MaxLevel must be positive")},
		{Config{M: 16, Mmax: 32, Mmax0: 64, EfConstruction: 200, MaxLevel: 16, DistanceFunc: nil}, errors.New("DistanceFunc must be provided")},
		{Config{M: 16, Mmax: 32, Mmax0: 64, EfConstruction: 200, MaxLevel: 16, DistanceFunc: EuclideanDistance}, nil},
	}

	for _, test := range tests {
		err := validateConfig(test.cfg)
		if err != nil && err.Error() != test.expected.Error() {
			t.Errorf("Expected error %v, got %v", test.expected, err)
		}
		if err == nil && test.expected != nil {
			t.Errorf("Expected error %v, got nil", test.expected)
		}
	}
}

func TestNewHNSW(t *testing.T) {
	cfg := DefaultConfig()
	hnsw, err := NewHNSW(cfg)
	if err != nil {
		t.Fatalf("Expected no error, got %v", err)
	}

	if hnsw.M != cfg.M {
		t.Errorf("Expected M to be %d, got %d", cfg.M, hnsw.M)
	}
	if hnsw.Mmax != cfg.Mmax {
		t.Errorf("Expected Mmax to be %d, got %d", cfg.Mmax, hnsw.Mmax)
	}
	if hnsw.Mmax0 != cfg.Mmax0 {
		t.Errorf("Expected Mmax0 to be %d, got %d", cfg.Mmax0, hnsw.Mmax0)
	}
	if hnsw.EfConstruction != cfg.EfConstruction {
		t.Errorf("Expected EfConstruction to be %d, got %d", cfg.EfConstruction, hnsw.EfConstruction)
	}
	if hnsw.MaxLevel != cfg.MaxLevel {
		t.Errorf("Expected MaxLevel to be %d, got %d", cfg.MaxLevel, hnsw.MaxLevel)
	}
	if hnsw.DistanceFunc == nil {
		t.Error("Expected DistanceFunc to be non-nil")
	}
}

func TestRandomLevel(t *testing.T) {
	cfg := DefaultConfig()
	h, err := NewHNSW(cfg)
	if err != nil {
		t.Fatalf("Failed to create HNSW: %v", err)
	}

	// Test with fixed random values
	testCases := []struct {
		randValue float64
		want      int
	}{
		{1.0, 0},   // -ln(1.0) * 0.3612 = 0
		{0.5, 0},   // -ln(0.5) * 0.3612 ≈ 0.25 → 0
		{0.1, 0},   // -ln(0.1) * 0.3612 ≈ 0.83 → 2
		{0.01, 1},  // -ln(0.01) * 0.3612 ≈ 1.66 → 1
		{0.001, 2}, // -ln(0.001) * 0.3612 ≈ 2.49 → 2
	}

	for _, tc := range testCases {
		t.Run(fmt.Sprintf("rand=%.3f", tc.randValue), func(t *testing.T) {
			h.RandFunc = func() float64 { return tc.randValue }
			got := h.RandomLevel()
			if got != tc.want {
				t.Errorf("RandomLevel() = %v, want %v", got, tc.want)
			}
			if got > h.MaxLevel {
				t.Errorf("RandomLevel() = %v, exceeded MaxLevel = %v", got, h.MaxLevel)
			}
		})
	}

	// Test distribution properties
	t.Run("distribution", func(t *testing.T) {
		h.RandFunc = rand.Float64 // Reset to random
		levels := make([]int, h.MaxLevel+1)
		n := 10000

		for i := 0; i < n; i++ {
			level := h.RandomLevel()
			if level < 0 || level > h.MaxLevel {
				t.Errorf("RandomLevel() = %v, want between 0 and %v", level, h.MaxLevel)
			}
			levels[level]++
		}

		// Verify exponential decay property
		for i := 1; i < len(levels); i++ {
			if levels[i] > levels[i-1] {
				t.Errorf("Level %d has more nodes (%d) than level %d (%d), violating exponential decay",
					i, levels[i], i-1, levels[i-1])
			}
		}
	})
}
