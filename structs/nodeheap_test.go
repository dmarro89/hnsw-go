package structs

import (
	"math"
	"sync"
	"testing"
)

func TestNewNodeHeap(t *testing.T) {
	tests := []struct {
		name     string
		distance float32
		id       int
	}{
		{"positive values", 1.5, 42},
		{"zero distance", 0.0, 100},
		{"negative distance", -3.14, 200},
		{"large values", 9999.99, 1000000},
		{"extreme value", math.MaxFloat32, math.MaxInt32},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			node := NewNodeHeap(tt.distance, tt.id)

			if node == nil {
				t.Fatal("NewNodeHeap returned nil")
			}

			if node.Dist != tt.distance {
				t.Errorf("Distance = %v, want %v", node.Dist, tt.distance)
			}

			if node.Id != tt.id {
				t.Errorf("ID = %v, want %v", node.Id, tt.id)
			}
		})
	}
}

func TestNodeHeapPool(t *testing.T) {
	t.Run("Create pool", func(t *testing.T) {
		pool := NewNodeHeapPool()
		if pool == nil {
			t.Fatal("NewNodeHeapPool returned nil")
		}
	})

	t.Run("Get creates new instance", func(t *testing.T) {
		pool := NewNodeHeapPool()
		node := pool.Get(1.5, 42)

		if node == nil {
			t.Fatal("Get returned nil")
		}

		if node.Dist != 1.5 {
			t.Errorf("Distance = %v, want 1.5", node.Dist)
		}

		if node.Id != 42 {
			t.Errorf("ID = %v, want 42", node.Id)
		}
	})

	t.Run("Put and Get reuse instances", func(t *testing.T) {
		pool := NewNodeHeapPool()

		// Create and put back a node
		node1 := pool.Get(1.5, 42)
		pool.Put(node1)

		// Get a node again - should be the same instance but with updated values
		node2 := pool.Get(3.0, 100)

		if node2.Dist != 3.0 {
			t.Errorf("Distance = %v, want 3.0", node2.Dist)
		}

		if node2.Id != 100 {
			t.Errorf("ID = %v, want 100", node2.Id)
		}
	})
}

func TestNodeHeapPoolConcurrent(t *testing.T) {
	pool := NewNodeHeapPool()
	concurrency := 10
	operations := 1000

	var wg sync.WaitGroup
	wg.Add(concurrency)

	for i := 0; i < concurrency; i++ {
		go func(workerID int) {
			defer wg.Done()

			for j := 0; j < operations; j++ {
				// Create unique distance and ID based on worker and operation
				dist := float32(workerID) + float32(j)*0.01
				id := workerID*operations + j

				// Get a node from the pool
				node := pool.Get(dist, id)

				// Verify it has the right values
				if node.Dist != dist || node.Id != id {
					t.Errorf("Worker %d, operation %d: incorrect values: got (%v,%v), want (%v,%v)",
						workerID, j, node.Dist, node.Id, dist, id)
				}

				// Return it to the pool
				pool.Put(node)
			}
		}(i)
	}

	wg.Wait()
}

func BenchmarkNodeHeap(b *testing.B) {
	b.Run("Direct creation", func(b *testing.B) {
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			node := NewNodeHeap(float32(i), i)
			_ = node // Prevent optimization
		}
	})

	b.Run("Pool creation", func(b *testing.B) {
		pool := NewNodeHeapPool()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			node := pool.Get(float32(i), i)
			pool.Put(node)
		}
	})
}
