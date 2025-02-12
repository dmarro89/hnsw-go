package structs

import (
	"testing"
)

func TestHeapPoolManager_MinHeap(t *testing.T) {
	manager := NewHeapPoolManager()

	t.Run("Get returns initialized MinHeap", func(t *testing.T) {
		h := manager.GetMinHeap()
		if h == nil {
			t.Error("GetMinHeap returned nil")
		}
		if h != nil && h.Len() != 0 {
			t.Errorf("New heap should be empty, got length %d", h.Len())
		}
	})

	t.Run("Get returns clean heap after Put", func(t *testing.T) {
		// Get a heap and add some items
		h1 := manager.GetMinHeap()
		h1.Push(uint64(1))
		h1.Push(uint64(2))

		if h1.Len() != 2 {
			t.Errorf("Expected length 2, got %d", h1.Len())
		}

		// Put it back and get a new one
		manager.PutMinHeap(h1)
		h2 := manager.GetMinHeap()

		if h2.Len() != 0 {
			t.Errorf("Recycled heap should be empty, got length %d", h2.Len())
		}
	})

	t.Run("Multiple Get/Put operations", func(t *testing.T) {
		heaps := make([]*MinHeap, 5)

		// Get multiple heaps
		for i := range heaps {
			heaps[i] = manager.GetMinHeap()
			heaps[i].Push(uint64(i))
		}

		// Put them back
		for _, h := range heaps {
			manager.PutMinHeap(h)
		}

		// Get them again and verify they're clean
		for i := 0; i < 5; i++ {
			h := manager.GetMinHeap()
			if h.Len() != 0 {
				t.Errorf("Recycled heap should be empty, got length %d", h.Len())
			}
		}
	})
}

func TestHeapPoolManager_MaxHeap(t *testing.T) {
	manager := NewHeapPoolManager()

	t.Run("Get returns initialized MaxHeap", func(t *testing.T) {
		h := manager.GetMaxHeap()
		if h == nil {
			t.Error("GetMaxHeap returned nil")
		}
		if h != nil && h.Len() != 0 {
			t.Errorf("New heap should be empty, got length %d", h.Len())
		}
	})

	t.Run("Get returns clean heap after Put", func(t *testing.T) {
		// Get a heap and add some items
		h1 := manager.GetMaxHeap()
		h1.Push(uint64(1))
		h1.Push(uint64(2))

		if h1.Len() != 2 {
			t.Errorf("Expected length 2, got %d", h1.Len())
		}

		// Put it back and get a new one
		manager.PutMaxHeap(h1)
		h2 := manager.GetMaxHeap()

		if h2.Len() != 0 {
			t.Errorf("Recycled heap should be empty, got length %d", h2.Len())
		}
	})

	t.Run("Multiple Get/Put operations", func(t *testing.T) {
		heaps := make([]*MaxHeap, 5)

		// Get multiple heaps
		for i := range heaps {
			heaps[i] = manager.GetMaxHeap()
			heaps[i].Push(uint64(i))
		}

		// Put them back
		for _, h := range heaps {
			manager.PutMaxHeap(h)
		}

		// Get them again and verify they're clean
		for i := 0; i < 5; i++ {
			h := manager.GetMaxHeap()
			if h.Len() != 0 {
				t.Errorf("Recycled heap should be empty, got length %d", h.Len())
			}
		}
	})
}

func TestHeapPoolManager_Concurrent(t *testing.T) {
	manager := NewHeapPoolManager()
	const numGoroutines = 10
	const numOperations = 100

	t.Run("Concurrent Min Heap operations", func(t *testing.T) {
		done := make(chan bool)

		for i := 0; i < numGoroutines; i++ {
			go func() {
				for j := 0; j < numOperations; j++ {
					h := manager.GetMinHeap()
					h.Push(uint64(j))
					manager.PutMinHeap(h)
				}
				done <- true
			}()
		}

		// Wait for all goroutines to complete
		for i := 0; i < numGoroutines; i++ {
			<-done
		}
	})

	t.Run("Concurrent Max Heap operations", func(t *testing.T) {
		done := make(chan bool)

		for i := 0; i < numGoroutines; i++ {
			go func() {
				for j := 0; j < numOperations; j++ {
					h := manager.GetMaxHeap()
					h.Push(uint64(j))
					manager.PutMaxHeap(h)
				}
				done <- true
			}()
		}

		// Wait for all goroutines to complete
		for i := 0; i < numGoroutines; i++ {
			<-done
		}
	})
}
