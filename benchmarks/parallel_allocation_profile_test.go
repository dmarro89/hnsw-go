package benchmarks

import (
 "fmt"
 "runtime"
 "testing"
 "dmarro89.github.com/hnsw-go/hnsw"
)

func BenchmarkParallelAllocationProfile(b *testing.B){
 const n,dim,workers=25000,128,4
 vectors:=profileVectors(n,dim)
 old:=runtime.GOMAXPROCS(workers);defer runtime.GOMAXPROCS(old)
 for _,ef:=range []int{64,200}{ef:=ef;b.Run(fmt.Sprintf("efc%d",ef),func(b *testing.B){b.ReportAllocs();for i:=0;i<b.N;i++{idx,err:=hnsw.NewHNSW(hnsw.Config{M:16,Mmax:16,Mmax0:32,EfConstruction:ef,MaxLevel:16,DistanceFunc:hnsw.EuclideanDistance});if err!=nil{b.Fatal(err)};if err:=idx.BuildParallel(vectors,hnsw.BulkBuildConfig{Workers:workers,BatchSize:64,EfConstruction:ef});err!=nil{b.Fatal(err)}}})}
}
