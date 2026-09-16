package benchmarks

import (
    "fmt"
    "math/rand/v2"
    "runtime"
    "strings"
    "sync/atomic"
    "testing"

    hnsw "dmarro89.github.com/hnsw-go/hnsw"
)

// Evidence-only: split searchLayerArena distance calls between entry-point
// initialization and graph-neighbor expansion. If entry recomputation is
// material, the next experiment can carry distances between layers without
// changing graph traversal semantics.
func TestSearchEntryDistanceRecompute(t *testing.T) {
    for _, ef := range []int{64, 200} {
        var entry, expand, other atomic.Uint64
        cfg := hnsw.Config{M:16, Mmax:32, Mmax0:64, EfConstruction:ef, MaxLevel:16}
        cfg.DistanceFunc = func(a,b []float32) float32 {
            pcs := make([]uintptr, 8)
            n := runtime.Callers(2, pcs)
            frames := runtime.CallersFrames(pcs[:n])
            classified := false
            for {
                f, more := frames.Next()
                if strings.Contains(f.Function, "searchLayerArena") {
                    // line attribution distinguishes the entry loop from expansion.
                    // Current production vector_arena.go: entry distance is before
                    // the candidate loop; expansion distance is later.
                    if f.Line < 165 { entry.Add(1) } else { expand.Add(1) }
                    classified = true
                    break
                }
                if !more { break }
            }
            if !classified { other.Add(1) }
            return hnsw.EuclideanDistance(a,b)
        }
        idx,err:=hnsw.NewHNSW(cfg); if err!=nil{t.Fatal(err)}
        lr:=rand.New(rand.NewPCG(11,22)); idx.RandFunc=lr.Float64
        r:=rand.New(rand.NewPCG(33,44)); vectors:=make([][]float32,100000)
        for i:=range vectors { v:=make([]float32,128); for j:=range v {v[j]=r.Float32()}; vectors[i]=v }
        idx.InsertBatch(vectors)
        e,x,o:=entry.Load(),expand.Load(),other.Load()
        fmt.Printf("SEARCH_CALL_SPLIT efC=%d entry=%d expand=%d search_total=%d entry_pct=%.2f other=%d\n",ef,e,x,e+x,100*float64(e)/float64(e+x),o)
    }
}
