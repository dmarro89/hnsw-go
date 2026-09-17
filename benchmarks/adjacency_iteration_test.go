package benchmarks

import (
	"math/rand"
	"testing"
)

var adjacencySink int

// BenchmarkAdjacencyIteration compares the current per-node [][]int-style
// storage shape with a flat offset arena for the exact operation dominating
// search expansion before distance evaluation: iterating neighbor IDs.
func BenchmarkAdjacencyIteration(b *testing.B) {
	const nodes = 100000
	const degree = 32
	r := rand.New(rand.NewSource(1))
	perNode := make([][]int, nodes)
	flat := make([]int, nodes*degree)
	for i := 0; i < nodes; i++ {
		perNode[i] = make([]int, degree)
		for j := 0; j < degree; j++ {
			v := r.Intn(nodes)
			perNode[i][j] = v
			flat[i*degree+j] = v
		}
	}
	ids := make([]int, 1<<16)
	for i := range ids { ids[i] = r.Intn(nodes) }
	b.Run("per-node-slices", func(b *testing.B) {
		b.ReportAllocs(); sum := 0
		for n:=0;n<b.N;n++ { for _,id := range ids { for _,v := range perNode[id] { sum += v } } }
		adjacencySink = sum
	})
	b.Run("flat-offset-arena", func(b *testing.B) {
		b.ReportAllocs(); sum := 0
		for n:=0;n<b.N;n++ { for _,id := range ids { off:=id*degree; ns:=flat[off:off+degree]; for _,v := range ns { sum += v } } }
		adjacencySink = sum
	})
}
