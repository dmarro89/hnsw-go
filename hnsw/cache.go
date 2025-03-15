package hnsw

import (
	"sync"

	"dmarro89.github.com/hnsw-go/structs"
)

// distanceCache provides efficient caching of distance calculations between nodes.
// It uses a hierarchical map structure where each node has its own cache of
// distances to other nodes.
type distanceCache struct {
	// nodeDistances maps each node ID to its own distance cache
	// nodeID -> (otherNodeID -> distance)
	nodeDistances map[int]map[int]float32

	// mutex protects concurrent access to the cache
	mutex sync.RWMutex
}

// newDistanceCache creates a new distance cache with the specified initial capacity
func newDistanceCache() *distanceCache {
	return &distanceCache{
		nodeDistances: make(map[int]map[int]float32),
	}
}

// get retrieves a cached distance between two nodes if available
func (dc *distanceCache) get(id1, id2 int) (float32, bool) {
	dc.mutex.RLock()
	defer dc.mutex.RUnlock()

	// Try to find distance from id1 -> id2
	if nodeCache, exists := dc.nodeDistances[id1]; exists {
		if dist, found := nodeCache[id2]; found {
			return dist, true
		}
	}

	// If not found, try the reverse direction (since distance is symmetric)
	if nodeCache, exists := dc.nodeDistances[id2]; exists {
		if dist, found := nodeCache[id1]; found {
			return dist, true
		}
	}

	return 0, false
}

// set stores a distance between two nodes in the cache
func (dc *distanceCache) set(id1, id2 int, distance float32) {
	dc.mutex.Lock()
	defer dc.mutex.Unlock()

	// Get or create the cache for the first node
	nodeCache, exists := dc.nodeDistances[id1]
	if !exists {
		nodeCache = make(map[int]float32)
		dc.nodeDistances[id1] = nodeCache
	}

	// Store the distance
	nodeCache[id2] = distance
}

// clear empties the entire cache
func (dc *distanceCache) clear() {
	dc.mutex.Lock()
	defer dc.mutex.Unlock()

	dc.nodeDistances = make(map[int]map[int]float32)
}

// computeAndCacheDistance calculates and caches the distance between vectors
func (h *HNSW) computeAndCacheDistance(v1 []float32, n2 *structs.Node) float32 {
	// First: try to identify if v1 belongs to a node in our index (for caching)
	var sourceID int = -1

	// Check if v1 is from a node we know about
	for _, node := range h.Nodes {
		if node != nil && &node.Vector[0] == &v1[0] { // Compare memory addresses for efficiency
			sourceID = node.ID
			break
		}
	}

	// If this is a query/search vector (not belonging to any node),
	// or we're calculating distance to the same node, just compute without caching
	if sourceID == -1 || sourceID == n2.ID {
		return h.DistanceFunc(v1, n2.Vector)
	}

	// Try to retrieve from cache
	if dist, found := h.globalDistanceCache.get(sourceID, n2.ID); found {
		return dist
	}

	// Calculate and cache the distance
	dist := h.DistanceFunc(v1, n2.Vector)
	h.globalDistanceCache.set(sourceID, n2.ID, dist)
	return dist
}

// resetCache clears all cached distances.
func (h *HNSW) resetCache() {
	h.globalDistanceCache.clear()
}
