package hnsw

import (
	"errors"
	"math"
	"math/rand/v2"
	"sync"

	"dmarro89.github.com/hnsw-go/structs"
)

type HNSW struct {
	Nodes []*structs.Node
	nodeStorage []structs.Node
	bulkNeighborHeaders [][]int
	bulkNeighborIDs []int
	RandFunc func() float64
	M int
	Mmax int
	Mmax0 int
	mL float64
	EfConstruction int
	DistanceFunc func([]float32, []float32) float32
	MaxLevel int
	EntryPoint *structs.Node
	mutex sync.RWMutex
	visitStamp int
	visitedIDs []int
	heapPool *structs.HeapPoolManager
	searchContextPool sync.Pool
	scratchCandidates []int
	scratchDistances []float32
}

type Config struct { M int; Mmax int; Mmax0 int; EfConstruction int; MaxLevel int; DistanceFunc func([]float32, []float32) float32 }
func DefaultConfig() Config { return Config{M:16,Mmax:32,Mmax0:64,EfConstruction:64,MaxLevel:16,DistanceFunc:EuclideanDistance} }
func NewHNSW(cfg Config)(*HNSW,error){
	if err:=validateConfig(cfg);err!=nil{return nil,err}
	h:=&HNSW{M:cfg.M,Mmax:cfg.Mmax,Mmax0:cfg.Mmax0,mL:1/math.Log(float64(cfg.M)),EfConstruction:cfg.EfConstruction,MaxLevel:cfg.MaxLevel,DistanceFunc:cfg.DistanceFunc,RandFunc:rand.Float64,visitedIDs:make([]int,cfg.EfConstruction),heapPool:structs.NewHeapPoolManager(),scratchCandidates:make([]int,0,max(cfg.Mmax,cfg.Mmax0)),scratchDistances:make([]float32,0,max(cfg.Mmax,cfg.Mmax0))}
	return h,nil
}
func validateConfig(cfg Config) error {
	if cfg.M<=0{return errors.New("m must be positive")}; if cfg.Mmax<=0{return errors.New("mmax must be positive")}; if cfg.Mmax0<=0{return errors.New("Mmax0 must be positive")}; if cfg.EfConstruction<=0{return errors.New("EfConstruction must be positive")}; if cfg.MaxLevel<=0{return errors.New("MaxLevel must be positive")}; if cfg.M>cfg.Mmax{return errors.New("M must be less than or equal to Mmax")}; if cfg.M>cfg.Mmax0{return errors.New("M must be less than or equal to Mmax0")}; if cfg.DistanceFunc==nil{return errors.New("DistanceFunc must be provided")}; return nil
}
func (h *HNSW) RandomLevel() int { return int(-math.Log(h.RandFunc())*h.mL) }
func (h *HNSW) markVisited(id int) bool { if id>=len(h.visitedIDs){newSize:=id*2;newVisited:=make([]int,newSize);copy(newVisited,h.visitedIDs);h.visitedIDs=newVisited};if h.visitedIDs[id]==h.visitStamp{return true};h.visitedIDs[id]=h.visitStamp;return false }
func (h *HNSW) ensureHeapPool()*structs.HeapPoolManager{if h.heapPool==nil{h.heapPool=structs.NewHeapPoolManager()};return h.heapPool}
func (h *HNSW) scratchCandidatesBuffer(size int)[]int{if cap(h.scratchCandidates)<size{h.scratchCandidates=make([]int,0,size)};return h.scratchCandidates[:0]}
func (h *HNSW) scratchDistancesBuffer(size int)[]float32{if cap(h.scratchDistances)<size{h.scratchDistances=make([]float32,0,size)};return h.scratchDistances[:0]}
func (h *HNSW) appendNode(id int,vector []float32,level int)*structs.Node{h.nodeStorage=append(h.nodeStorage,structs.Node{});node:=&h.nodeStorage[len(h.nodeStorage)-1];structs.InitNode(node,id,vector,level,h.MaxLevel,h.Mmax,h.Mmax0);h.Nodes=append(h.Nodes,node);return node}
func (h *HNSW) ensureNodeCapacity(total int){if cap(h.Nodes)>=total&&cap(h.nodeStorage)>=total{return};currentLen:=len(h.Nodes);newStorage:=make([]structs.Node,currentLen,total);if len(h.nodeStorage)==currentLen{copy(newStorage,h.nodeStorage)}else{for i,node:=range h.Nodes{if node!=nil{newStorage[i]=*node}}};newNodes:=make([]*structs.Node,currentLen,total);for i:=range newStorage{newNodes[i]=&newStorage[i]};h.nodeStorage=newStorage;h.Nodes=newNodes;if h.EntryPoint!=nil&&h.EntryPoint.ID>=0&&h.EntryPoint.ID<len(h.nodeStorage){h.EntryPoint=&h.nodeStorage[h.EntryPoint.ID]}}
func (h *HNSW) ensureVisitedCapacity(maxID int){if maxID<len(h.visitedIDs){return};newSize:=max(maxID+1,len(h.visitedIDs)*2);newVisited:=make([]int,newSize);copy(newVisited,h.visitedIDs);h.visitedIDs=newVisited}
