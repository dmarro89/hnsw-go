package hnsw

import (
	"errors"
	"math"
	"math/rand/v2"
	"sync"

	"dmarro89.github.com/hnsw-go/structs"
)

// HNSW (Hierarchical Navigable Small World) represents a graph-based index
// for approximate nearest neighbor search. It organizes nodes in a hierarchical
// structure where each level is a navigable small world graph.
type HNSW struct {
	// Nodes contains all vectors in the index
	Nodes []*structs.Node

	// nodeStorage keeps node structs contiguous in memory during bulk builds.
	nodeStorage []structs.Node

	// RandFunc provides random values for level generation
	RandFunc func() float64

	// M is the number of established connections on index construction
	M int

	// Mmax is the maximum number of connections per layer for layers > 0
	Mmax int

	// Mmax0 is the maximum number of connections for layer 0
	Mmax0 int

	// mL is the normalization factor for level generation (1/ln(M))
	mL float64

	// EfConstruction controls the quality of index construction
	// Higher values provide better quality at the cost of longer construction time
	EfConstruction int

	// DistanceFunc calculates the distance between two vectors
	DistanceFunc func([]float32, []float32) float32

	// MaxLevel is retained for configuration compatibility
	// The actual graph height follows the random level distribution
	MaxLevel int

	// EntryPoint is the highest-level node in the graph
	EntryPoint *structs.Node

	// mutex is used to synchronize access and write to the HNSW index
	mutex sync.RWMutex

	// Versioning counter for faster visited node check during construction.
	visitStamp int

	// Pre-allocated array for tracking visited nodes during construction.
	visitedIDs []int

	// heapPool reuses temporary heaps during construction.
	heapPool *structs.HeapPoolManager

	// searchContextPool provides isolated reusable scratch state for concurrent
	// read-only searches. Search contexts must never be shared by active queries.
	searchContextPool sync.Pool

	// scratchCandidates reuses temporary storage when pruning neighbors.
	scratchCandidates []int

	// scratchDistances reuses temporary distance storage when pruning neighbors.
	scratchDistances []float32
}

// Config holds the configuration parameters for HNSW construction
type Config struct {
	// M is the number of established connections
	M int

	// Mmax is the maximum number of connections per layer (layers > 0)
	Mmax int

	// Mmax0 is the maximum number of connections for layer 0
	Mmax0 int

	// EfConstruction controls construction quality vs time trade-off
	EfConstruction int

	// MaxLevel is retained for configuration compatibility
	MaxLevel int

	// DistanceFunc is the distance function to use
	DistanceFunc func([]float32, []float32) float32
}

// DefaultConfig returns a Config with recommended default values
func DefaultConfig() Config {
	return Config{
		M:              16,
		Mmax:           32,
		Mmax0:          64,
		EfConstruction: 64,
		MaxLevel:       16,
		DistanceFunc:   EuclideanDistance,
	}
}

// NewHNSW creates a new HNSW index with the specified configuration.
// Returns an error if the configuration is invalid.
func NewHNSW(cfg Config) (*HNSW, error) {
	if err := validateConfig(cfg); err != nil {
		return nil, err
	}

	h := &HNSW{
		M:                 cfg.M,
		Mmax:              cfg.Mmax,
		Mmax0:             cfg.Mmax0,
		mL:                1 / math.Log(float64(cfg.M)),
		EfConstruction:    cfg.EfConstruction,
		MaxLevel:          cfg.MaxLevel,
		DistanceFunc:      cfg.DistanceFunc,
		RandFunc:          rand.Float64,
		visitStamp:        0,
		visitedIDs:        make([]int, cfg.EfConstruction),
		heapPool:          structs.NewHeapPoolManager(),
		scratchCandidates: make([]int, 0, max(cfg.Mmax, cfg.Mmax0)),
		scratchDistances:  make([]float32, 0, max(cfg.Mmax, cfg.Mmax0)),
	}

	return h, nil
}

func validateConfig(cfg Config) error {
	if cfg.M <= 0 {
		return errors.New("m must be positive")
	}
	if cfg.Mmax <= 0 {
		return errors.New("mmax must be positive")
	}
	if cfg.Mmax0 <= 0 {
		return errors.New("Mmax0 must be positive")
	}
	if cfg.EfConstruction <= 0 {
		return errors.New("EfConstruction must be positive")
	}
	if cfg.MaxLevel <= 0 {
		return errors.New("MaxLevel must be positive")
	}
	if cfg.M > cfg.Mmax {
		return errors.New("M must be less than or equal to Mmax")
	}
	if cfg.M > cfg.Mmax0 {
		return errors.New("M must be less than or equal to Mmax0")
	}
	if cfg.DistanceFunc == nil {
		return errors.New("DistanceFunc must be provided")
	}
	return nil
}

// The integer level 𝑙 is randomly selected with an exponentially decaying probability distribution, normalized by a parameter 𝑚𝐿.
// This process ensures that the probability of being in higher levels decreases exponentially.
// The formula used to generate the level 𝑙 is:
// l=−ln(unif(0,1))⋅mL
// where:
// - ln is the natural logarithm
// - unif(0,1) represents a random value uniformly distributed between 0 and 1
// - 𝑚𝐿 is a normalization factor that controls the hierarchy of the graph
func (h *HNSW) RandomLevel() int {
	// Generate a random value between 0 and 1
	randValue := h.RandFunc()

	// Calculate the level using the formula
	level := int(-math.Log(randValue) * h.mL)

	return level
}

// markVisited marks a node as visited during the search process
func (h *HNSW) markVisited(id int) bool {
	// Exppand the visitedIDs slice if necessary
	if id >= len(h.visitedIDs) {
		newSize := id * 2
		newVisited := make([]int, newSize)
		copy(newVisited, h.visitedIDs)
		h.visitedIDs = newVisited
	}

	if h.visitedIDs[id] == h.visitStamp {
		return true
	}

	h.visitedIDs[id] = h.visitStamp
	return false
}

func (h *HNSW) ensureHeapPool() *structs.HeapPoolManager {
	if h.heapPool == nil {
		h.heapPool = structs.NewHeapPoolManager()
	}
	return h.heapPool
}

func (h *HNSW) scratchCandidatesBuffer(size int) []int {
	if cap(h.scratchCandidates) < size {
		h.scratchCandidates = make([]int, 0, size)
	}
	return h.scratchCandidates[:0]
}

func (h *HNSW) scratchDistancesBuffer(size int) []float32 {
	if cap(h.scratchDistances) < size {
		h.scratchDistances = make([]float32, 0, size)
	}
	return h.scratchDistances[:0]
}

func (h *HNSW) appendNode(id int, vector []float32, level int) *structs.Node {
	h.nodeStorage = append(h.nodeStorage, structs.Node{})
	node := &h.nodeStorage[len(h.nodeStorage)-1]
	structs.InitNode(node, id, vector, level, h.MaxLevel, h.Mmax, h.Mmax0)
	h.Nodes = append(h.Nodes, node)
	return node
}

func (h *HNSW) ensureNodeCapacity(total int) {
	if cap(h.Nodes) >= total && cap(h.nodeStorage) >= total {
		return
	}

	currentLen := len(h.Nodes)
	newStorage := make([]structs.Node, currentLen, total)

	if len(h.nodeStorage) == currentLen {
		copy(newStorage, h.nodeStorage)
	} else {
		for i, node := range h.Nodes {
			if node != nil {
				newStorage[i] = *node
			}
		}
	}

	newNodes := make([]*structs.Node, currentLen, total)
	for i := range newStorage {
		newNodes[i] = &newStorage[i]
	}

	h.nodeStorage = newStorage
	h.Nodes = newNodes
	if h.EntryPoint != nil && h.EntryPoint.ID >= 0 && h.EntryPoint.ID < len(h.nodeStorage) {
		h.EntryPoint = &h.nodeStorage[h.EntryPoint.ID]
	}
}

func (h *HNSW) ensureVisitedCapacity(maxID int) {
	if maxID < len(h.visitedIDs) {
		return
	}
	newSize := max(maxID+1, len(h.visitedIDs)*2)
	newVisited := make([]int, newSize)
	copy(newVisited, h.visitedIDs)
	h.visitedIDs = newVisited
}
