package hnsw

import (
    "fmt"
    "math/rand/v2"
    "runtime"
    "strings"
    "testing"
)

// TestProfileConstructionDistanceCallSites is an evidence-only construction
// profile. runtime.Caller intentionally makes it unsuitable as a timing test;
// the output measures exact distance-call counts by production call site.
func TestProfileConstructionDistanceCallSites(t *testing.T) {
    if testing.Short() { t.Skip("profiling test") }
    const count, dim = 10000, 128
    vectors := make([][]float32, count)
    r := rand.New(rand.NewPCG(424242, 777))
    for i := range vectors {
        vectors[i] = make([]float32, dim)
        for j := range vectors[i] { vectors[i][j] = r.Float32() }
    }
    for _, efc := range []int{64, 200} {
        cfg := DefaultConfig(); cfg.EfConstruction = efc
        counts := map[string]uint64{}
        cfg.DistanceFunc = func(a,b []float32) float32 {
            pc,_,_,ok := runtime.Caller(1)
            name := "unknown"
            if ok { name = runtime.FuncForPC(pc).Name() }
            switch {
            case strings.Contains(name,"greedySearchLayerArena"): counts["greedy"]++
            case strings.Contains(name,"searchLayerArena"): counts["search"]++
            case strings.Contains(name,"selectClosestArena"): counts["prune"]++
            default: counts["other"]++
            }
            return EuclideanDistance(a,b)
        }
        idx,err:=NewHNSW(cfg); if err!=nil{t.Fatal(err)}
        lr:=rand.New(rand.NewPCG(12345,67890)); idx.RandFunc=lr.Float64
        if !idx.InsertBatchArenaExperimental(vectors){t.Fatal("arena build rejected")}
        total:=counts["greedy"]+counts["search"]+counts["prune"]+counts["other"]
        fmt.Printf("DISTANCE_CALLS efC=%d total=%d search=%d(%.2f%%) prune=%d(%.2f%%) greedy=%d(%.2f%%) other=%d(%.2f%%)\n",efc,total,counts["search"],100*float64(counts["search"])/float64(total),counts["prune"],100*float64(counts["prune"])/float64(total),counts["greedy"],100*float64(counts["greedy"])/float64(total),counts["other"],100*float64(counts["other"])/float64(total))
    }
}
