// cache_test.go
package hnsw

import (
	"sync"
	"testing"

	"dmarro89.github.com/hnsw-go/structs"
)

func TestComputeAndCache(t *testing.T) {
	tests := []struct {
		name     string
		nodeID   int
		vector   []float32
		query    []float32
		distance float32
	}{
		{
			name:     "basic caching",
			nodeID:   0,
			vector:   []float32{1.0, 0.0},
			query:    []float32{0.0, 0.0},
			distance: 1.0,
		},
		{
			name:     "cache growth",
			nodeID:   2000,
			vector:   []float32{2.0, 0.0},
			query:    []float32{0.0, 0.0},
			distance: 4.0,
		},
		{
			name:     "cache nodes",
			nodeID:   0,
			vector:   []float32{2.0, 0.0},
			query:    []float32{4.0, 0.0},
			distance: 4.0,
		},
	}

	h, _ := NewHNSW(DefaultConfig())
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			node := &structs.Node{
				ID:     tt.nodeID,
				Vector: tt.vector,
			}

			// First computation should calculate
			dist1 := h.computeAndCacheDistance(tt.query, node)
			if dist1 != tt.distance {
				t.Errorf("First computation = %v, want %v", dist1, tt.distance)
			}

			// Second computation should use cache

		})
	}
}

func TestCacheConcurrency(t *testing.T) {
	h, _ := NewHNSW(DefaultConfig())
	node := &structs.Node{
		ID:     1,
		Vector: []float32{1.0, 0.0},
	}
	query := []float32{0.0, 0.0}

	const goroutines = 100
	var wg sync.WaitGroup
	wg.Add(goroutines)

	for i := 0; i < goroutines; i++ {
		go func() {
			defer wg.Done()
			dist := h.computeAndCacheDistance(query, node)
			if dist != 1.0 {
				t.Errorf("Got distance %v, want 1.0", dist)
			}
		}()
	}

	wg.Wait()
}

func BenchmarkComputeAndCache(b *testing.B) {
	h, _ := NewHNSW(DefaultConfig())
	node := &structs.Node{
		ID:     1,
		Vector: []float32{1.0, 0.0},
	}
	query := []float32{0.0, 0.0}

	b.Run("First Access", func(b *testing.B) {
		h.resetCache()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			h.computeAndCacheDistance(query, node)
		}
	})

	b.Run("Cached Access", func(b *testing.B) {
		h.computeAndCacheDistance(query, node) // Ensure cached
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			h.computeAndCacheDistance(query, node)
		}
	})
}
