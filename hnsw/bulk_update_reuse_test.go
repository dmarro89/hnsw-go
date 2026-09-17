package hnsw

import "testing"

var bulkUpdateSink int

// BenchmarkPreviousNeighborScratch isolates the dominant allocation identified
// by #57: copying the previous neighbor list before pruning. The production
// candidate would reuse one scratch buffer across the serial apply phase.
func BenchmarkPreviousNeighborScratch(b *testing.B) {
 const degree=32
 src:=make([]int,degree);for i:=range src{src[i]=i}
 b.Run("allocate-copy",func(b *testing.B){b.ReportAllocs();sum:=0;for i:=0;i<b.N;i++{previous:=append([]int(nil),src...);sum+=previous[i%degree]};bulkUpdateSink=sum})
 b.Run("reuse-copy",func(b *testing.B){b.ReportAllocs();scratch:=make([]int,degree);sum:=0;b.ResetTimer();for i:=0;i<b.N;i++{previous:=scratch[:degree];copy(previous,src);sum+=previous[i%degree]};bulkUpdateSink=sum})
}
