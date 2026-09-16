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

	// bulkNeighborHeaders and bulkNeighborIDs back per-node neighbor slices for
	// empty-index serial bulk builds. They are sized exactly after levels are
	// generated, removing two heap allocations from every node initialization.
	bulkNeighborHeaders [][]int
	bulkNeighborIDs []int

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
	M int
	Mmax int
	Mmax0 int
	EfConstruction int
	MaxLevel int
	DistanceFunc func([]float32, []float32) float32
}

func DefaultConfig() Config {
	return Config{M:16, Mmax:32, Mmax0:64, EfConstruction:64, MaxLevel:16, DistanceFunc:EuclideanDistance}
}

func NewHNSW(cfg Config) (*HNSW, error) {
	if err := validateConfig(cfg); err != nil { return nil, err }
	return &HNSW{M:cfg.M,Mmax:cfg.Mmax,Mmax0:cfg.Mmax0,EfConstruction:cfg.EfConstruction,MaxLevel:cfg.MaxLevel,DistanceFunc:cfg.DistanceFunc,mL:1/math.Log(float64(cfg.M)),RandFunc:rand.Float64}, nil
}

func validateConfig(cfg Config) error {
	if cfg.M <= 1 { return errors.New("M must be greater than 1") }
	if cfg.Mmax < cfg.M || cfg.Mmax0 < cfg.M { return errors.New("Mmax and Mmax0 must be >= M") }
	if cfg.EfConstruction <= 0 { return errors.New("EfConstruction must be positive") }
	if cfg.MaxLevel <= 0 { return errors.New("MaxLevel must be positive") }
	if cfg.DistanceFunc == nil { return errors.New("DistanceFunc must not be nil") }
	return nil
}

func (h *HNSW) RandomLevel() int {
	r := h.RandFunc()
	if r <= 0 { r = math.SmallestNonzeroFloat64 }
	return int(-math.Log(r) * h.mL)
}

func (h *HNSW) Insert(vector []float32) {
	h.mutex.Lock(); defer h.mutex.Unlock()
	h.insertLocked(vector, len(h.Nodes), nil)
}

func (h *HNSW) InsertBatch(vectors [][]float32) {
	if len(vectors)==0 { return }
	if dim,ok:=uniformVectorDimension(vectors); ok {
		h.mutex.Lock(); defer h.mutex.Unlock()
		if len(h.Nodes)==0 { h.insertBatchArenaLocked(vectors,dim); return }
	}
	h.mutex.Lock(); defer h.mutex.Unlock()
	nextID:=len(h.Nodes); h.ensureNodeCapacity(nextID+len(vectors)); h.ensureVisitedCapacity(nextID+len(vectors)-1)
	buf:=make([]int,0,h.EfConstruction)
	for _,v:=range vectors { buf=h.insertLocked(v,nextID,buf); nextID++ }
}

func (h *HNSW) markVisited(id int) bool {
	if id>=len(h.visitedIDs) { h.ensureVisitedCapacity(id) }
	if h.visitedIDs[id]==h.visitStamp { return true }
	h.visitedIDs[id]=h.visitStamp; return false
}
func (h *HNSW) ensureHeapPool()*structs.HeapPoolManager { if h.heapPool==nil { h.heapPool=structs.NewHeapPoolManager() }; return h.heapPool }
func (h *HNSW) scratchCandidatesBuffer(size int)[]int { if cap(h.scratchCandidates)<size { h.scratchCandidates=make([]int,0,size) }; return h.scratchCandidates[:0] }
func (h *HNSW) scratchDistancesBuffer(size int)[]float32 { if cap(h.scratchDistances)<size { h.scratchDistances=make([]float32,0,size) }; return h.scratchDistances[:0] }
func (h *HNSW) appendNode(id int, vector []float32, level int)*structs.Node { h.nodeStorage=append(h.nodeStorage,structs.Node{}); n:=&h.nodeStorage[len(h.nodeStorage)-1]; structs.InitNode(n,id,vector,level,h.MaxLevel,h.Mmax,h.Mmax0); h.Nodes=append(h.Nodes,n); return n }
func (h *HNSW) ensureNodeCapacity(total int) {
	if cap(h.Nodes)>=total && cap(h.nodeStorage)>=total{return}; cur:=len(h.Nodes); ns:=make([]structs.Node,cur,total)
	if len(h.nodeStorage)==cur { copy(ns,h.nodeStorage) } else { for i,n:=range h.Nodes { if n!=nil { ns[i]=*n } } }
	np:=make([]*structs.Node,cur,total); for i:=range ns { np[i]=&ns[i] }; h.nodeStorage=ns; h.Nodes=np
	if h.EntryPoint!=nil && h.EntryPoint.ID>=0 && h.EntryPoint.ID<len(h.nodeStorage){h.EntryPoint=&h.nodeStorage[h.EntryPoint.ID]}
}
func (h *HNSW) ensureVisitedCapacity(maxID int){if maxID<len(h.visitedIDs){return}; n:=max(maxID+1,len(h.visitedIDs)*2); v:=make([]int,n); copy(v,h.visitedIDs); h.visitedIDs=v}
