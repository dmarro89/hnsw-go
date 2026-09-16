package hnsw

import (
	"math/rand/v2"
	"reflect"
	"testing"
)

func TestBulkNeighborArenaPreservesBuildTopology(t *testing.T) {
	vectors:=make([][]float32,256); r:=rand.New(rand.NewPCG(101,202))
	for i:=range vectors{vectors[i]=make([]float32,32);for j:=range vectors[i]{vectors[i][j]=r.Float32()}}
	cfg:=Config{M:16,Mmax:32,Mmax0:64,EfConstruction:64,MaxLevel:16,DistanceFunc:EuclideanDistance}
	base,_:=NewHNSW(cfg); head,_:=NewHNSW(cfg)
	baseR:=rand.New(rand.NewPCG(303,404));headR:=rand.New(rand.NewPCG(303,404));base.RandFunc=baseR.Float64;head.RandFunc=headR.Float64
	base.InsertBatchBaselineExperimental(vectors); if !head.InsertBatchArenaExperimental(vectors){t.Fatal("arena build rejected")}
	if len(base.Nodes)!=len(head.Nodes){t.Fatalf("node count %d != %d",len(base.Nodes),len(head.Nodes))}
	for i:=range base.Nodes{a,b:=base.Nodes[i],head.Nodes[i];if a.Level!=b.Level{t.Fatalf("node %d level %d != %d",i,a.Level,b.Level)};if !reflect.DeepEqual(a.Neighbors,b.Neighbors){t.Fatalf("node %d neighbors differ",i)}}
	if base.EntryPoint.ID!=head.EntryPoint.ID{t.Fatalf("entry point %d != %d",base.EntryPoint.ID,head.EntryPoint.ID)}
}
