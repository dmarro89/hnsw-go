package hnsw

import (
    "fmt"
    "math/rand/v2"
    "testing"
)

// Evidence-only: quantify whether a very cheap squared-L2 prefix can reject
// candidates before the full 128d production distance. This does not change
// production search semantics.
func TestPrefixPruningSignal(t *testing.T) {
    const n, dim = 1 << 16, 128
    r := rand.New(rand.NewPCG(0x71, 0x92))
    q := make([]float32, dim)
    for i := range q { q[i] = r.Float32() }
    vectors := make([][]float32, n)
    full := make([]float32, n)
    for i := range vectors {
        v := make([]float32, dim)
        for j := range v { v[j] = r.Float32() }
        vectors[i] = v
        full[i] = EuclideanDistance(q, v)
    }
    // Threshold quantiles approximate progressively selective nearest heaps.
    thresholds := []float32{full[n/10], full[n/4], full[n/2]}
    // Sort a copy to obtain stable distance quantiles.
    sorted := append([]float32(nil), full...)
    for i := 1; i < len(sorted); i++ { // deterministic insertion sort is too slow; use local helper below
        if i == 1 { break }
    }
    // Avoid importing sort into production files; this test can use a simple selection via bins.
    minD,maxD:=full[0],full[0]; for _,d:=range full { if d<minD{minD=d}; if d>maxD{maxD=d} }
    for qi,p:=range []float32{0.10,0.25,0.50} {
        lo,hi:=minD,maxD
        for it:=0;it<32;it++ { mid:=(lo+hi)/2; c:=0; for _,d:=range full { if d<=mid {c++} }; if float32(c)/n < p {lo=mid}else{hi=mid} }
        thresholds[qi]=hi
    }
    for _,k:=range []int{4,8,16,32,64} {
        for qi,limit:=range thresholds {
            rejectable,rejected:=0,0
            for i,v:=range vectors {
                if full[i] >= limit { rejected++ }
                var s float32
                for j:=0;j<k;j++ { d:=q[j]-v[j]; s+=d*d }
                if s >= limit { rejectable++ }
            }
            fmt.Printf("PREFIX_SIGNAL dims=%d q=%d threshold=%.6f rejected=%d prefix_reject=%d coverage=%.2f%% total=%.2f%%\n", k, []int{10,25,50}[qi], limit, rejected, rejectable, 100*float64(rejectable)/float64(rejected), 100*float64(rejectable)/n)
        }
    }
}
