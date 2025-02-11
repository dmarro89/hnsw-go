package hnsw

import (
	"sync"

	"dmarro89.github.com/hnsw-go/structs"
)

// distanceCache manages caching of distance calculations between vectors.
// It uses a pre-allocated slice to store distances and grows exponentially when needed.
type distanceCache struct {
	cache []float32
	mutex sync.RWMutex
}

// newDistanceCache creates a new distance cache with an initial capacity.
func newDistanceCache(initialCapacity int) *distanceCache {
	cache := make([]float32, initialCapacity)
	for i := range cache {
		cache[i] = -1 // Initialize with invalid distance
	}
	return &distanceCache{
		cache: cache,
	}
}

// computeAndCache calculates and caches the distance between a query vector and a node.
// If the distance is already cached, returns the cached value.
// The cache grows exponentially if needed to accommodate new node IDs.
func (h *HNSW) computeAndCacheDistance(query []float32, node *structs.Node) float32 {
	cache := h.globalDistanceCache.cache

	if node.ID < len(cache) {
		dist := cache[node.ID]
		if dist >= 0 {
			return dist
		}
	}

	dist := h.DistanceFunc(query, node.Vector)

	if node.ID >= len(cache) {
		newLen := len(cache)
		if newLen == 0 {
			newLen = 1
		}
		for newLen <= node.ID {
			newLen *= 2
		}
		newCache := make([]float32, newLen)
		copy(newCache, cache)
		for i := len(cache); i < newLen; i++ {
			newCache[i] = -1.0
		}
		h.globalDistanceCache.cache = newCache
		cache = newCache
	}
	if cache[node.ID] < 0 {
		cache[node.ID] = dist
	}

	return dist
}

// resetCache clears all cached distances.
func (h *HNSW) resetCache() {
	h.globalDistanceCache.mutex.Lock()
	defer h.globalDistanceCache.mutex.Unlock()

	for i := range h.globalDistanceCache.cache {
		h.globalDistanceCache.cache[i] = -1
	}
}
