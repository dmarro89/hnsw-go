package hnsw

import (
 "fmt"
 "math/rand/v2"
 "runtime"
 "strings"
 "sync/atomic"
 "testing"
)

var expansionProfile struct { calls, expansion, accepted, rejected atomic.Uint64 }

func profiledExpansionDistance(a,b []float32) float32 {
 d:=EuclideanDistance(a,b); expansionProfile.calls.Add(1)
 var pcs [8]uintptr; n:=runtime.Callers(2,pcs[:]); frames:=runtime.CallersFrames(pcs[:n])
 for { f,more:=frames.Next(); if strings.Contains(f.Function,"searchLayerArena") { expansionProfile.expansion.Add(1); break }; if !more { break } }
 return d
}

// TestProfileSearchExpansionOutcomes measures how much search-layer distance work
// is useful. It deliberately instruments only an evidence workload; production is unchanged.
func TestProfileSearchExpansionOutcomes(t *testing.T) {
 for _,ef:=range []int{64,200} {
  expansionProfile.calls.Store(0); expansionProfile.expansion.Store(0)
  const n,dim=10000,128
  r:=rand.New(rand.NewPCG(0x51,0x99)); vectors:=make([][]float32,n)
  for i:=range vectors { vectors[i]=make([]float32,dim); for j:=range vectors[i] { vectors[i][j]=r.Float32() } }
  h,err:=NewHNSW(Config{M:16,Mmax:32,Mmax0:64,EfConstruction:ef,MaxLevel:16,DistanceFunc:profiledExpansionDistance}); if err!=nil { t.Fatal(err) }
  lr:=rand.New(rand.NewPCG(0x17,0x23)); h.RandFunc=lr.Float64
  h.InsertBatch(vectors)
  fmt.Printf("EXPANSION_PROFILE ef=%d total_distance=%d search_layer_distance=%d\n",ef,expansionProfile.calls.Load(),expansionProfile.expansion.Load())
 }
}
