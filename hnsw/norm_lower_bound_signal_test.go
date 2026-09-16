package hnsw

import (
    "fmt"
    "math"
    "math/rand/v2"
    "testing"
)

// Evidence-only: test the exact reverse-triangle lower bound
// ||q-v|| >= abs(||q||-||v||). Candidate norms could be precomputed once,
// making the traversal check a subtract/multiply before the full 128d L2.
func TestNormLowerBoundSignal(t *testing.T) {
    const n, dim = 1 << 16, 128
    r := rand.New(rand.NewPCG(0x71, 0x92))
    q := make([]float32, dim)
    var q2 float32
    for i := range q { q[i] = r.Float32(); q2 += q[i]*q[i] }
    qnorm := float32(math.Sqrt(float64(q2)))
    full := make([]float32, n)
    norms := make([]float32, n)
    for i := 0; i < n; i++ {
        v := make([]float32, dim)
        var v2 float32
        for j := range v { v[j] = r.Float32(); v2 += v[j]*v[j] }
        norms[i] = float32(math.Sqrt(float64(v2)))
        full[i] = EuclideanDistance(q, v)
    }
    minD,maxD:=full[0],full[0]
    for _,d:=range full { if d<minD{minD=d}; if d>maxD{maxD=d} }
    for qi,p:=range []float32{0.10,0.25,0.50} {
        lo,hi:=minD,maxD
        for it:=0;it<32;it++ { mid:=(lo+hi)/2; c:=0; for _,d:=range full { if d<=mid {c++} }; if float32(c)/n < p {lo=mid}else{hi=mid} }
        limit:=hi
        rejected,pruned:=0,0
        for i,d:=range full {
            if d >= limit { rejected++ }
            delta:=qnorm-norms[i]
            if delta*delta >= limit { pruned++ }
        }
        fmt.Printf("NORM_SIGNAL q=%d threshold=%.6f rejected=%d norm_reject=%d coverage=%.4f%% total=%.4f%%\n", []int{10,25,50}[qi], limit, rejected, pruned, 100*float64(pruned)/float64(rejected), 100*float64(pruned)/n)
    }
}
