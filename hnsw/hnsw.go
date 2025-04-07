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

	// MaxLevel is the highest level in the graph
	MaxLevel int

	// EntryPoint is the highest-level node in the graph
	EntryPoint *structs.Node

	// heapPool manages heap objects for reuse
	heapPool *structs.HeapPoolManager

	// nodeHeapPool manages node heap objects for reuse
	nodeHeapPool *structs.NodeHeapPool

	mutex sync.RWMutex
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

	// MaxLevel is the maximum level in the graph
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
		EfConstruction: 200,
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
		M:              cfg.M,
		Mmax:           cfg.Mmax,
		Mmax0:          cfg.Mmax0,
		mL:             1 / math.Log(float64(cfg.M)),
		EfConstruction: cfg.EfConstruction,
		MaxLevel:       cfg.MaxLevel,
		DistanceFunc:   cfg.DistanceFunc,
		RandFunc:       rand.Float64,
		heapPool:       structs.NewHeapPoolManager(),
		nodeHeapPool:   structs.NewNodeHeapPool(),
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

	// Cap the level at the maximum allowed level
	if level > h.MaxLevel {
		level = h.MaxLevel
	}
	return level
}
