package hnsw

import "testing"

// BenchmarkVisitedArenaFastPath isolates the per-neighbor visited-stamp cost.
// Production arena bulk builds pre-size visitedIDs for the whole batch, so the
// capacity-growth branch in markVisitedArena is redundant on that path.
func BenchmarkVisitedArenaFastPath(b *testing.B) {
	const n = 100000
	ids := make([]int, n)
	for i := range ids {
		ids[i] = i
	}

	b.Run("current", func(b *testing.B) {
		h := &HNSW{visitedIDs: make([]int, n)}
		stamp := 1
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			id := ids[i%n]
			_ = h.markVisitedArena(id, stamp)
			if id == n-1 {
				stamp++
			}
		}
	})

	b.Run("presized-direct", func(b *testing.B) {
		visited := make([]int, n)
		stamp := 1
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			id := ids[i%n]
			seen := visited[id] == stamp
			if !seen {
				visited[id] = stamp
			}
			if id == n-1 {
				stamp++
			}
		}
	})
}
