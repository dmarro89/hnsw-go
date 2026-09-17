package hnsw

import "testing"

var visitedFastPathSink bool

// BenchmarkVisitedArenaFastPath isolates the per-neighbor visited-stamp cost.
// Production arena bulk builds pre-size visitedIDs for the whole batch, so the
// capacity-growth branch in markVisitedArena is redundant on that path. Each
// ID is deliberately presented twice per stamp to exercise both first-visit
// and already-visited paths, and the result is consumed to keep the comparison
// representative of the traversal branch.
func BenchmarkVisitedArenaFastPath(b *testing.B) {
	const n = 100000
	ids := make([]int, n*2)
	for i := 0; i < n; i++ {
		ids[2*i] = i
		ids[2*i+1] = i
	}

	b.Run("current", func(b *testing.B) {
		h := &HNSW{visitedIDs: make([]int, n)}
		stamp := 1
		seenCount := 0
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			id := ids[i%len(ids)]
			if h.markVisitedArena(id, stamp) {
				seenCount++
			}
			if i%len(ids) == len(ids)-1 {
				stamp++
			}
		}
		visitedFastPathSink = seenCount != 0
	})

	b.Run("presized-direct", func(b *testing.B) {
		visited := make([]int, n)
		stamp := 1
		seenCount := 0
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			id := ids[i%len(ids)]
			seen := visited[id] == stamp
			if !seen {
				visited[id] = stamp
			} else {
				seenCount++
			}
			if i%len(ids) == len(ids)-1 {
				stamp++
			}
		}
		visitedFastPathSink = seenCount != 0
	})
}
