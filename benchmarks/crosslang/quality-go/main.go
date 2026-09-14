package main

import (
 "encoding/binary"
 "fmt"
 "math"
 "math/rand/v2"
 "os"
 "strconv"
 "dmarro89.github.com/hnsw-go/hnsw"
)

func main() {
 if len(os.Args)!=8 { fmt.Fprintln(os.Stderr,"usage: quality-go <vectors> <n> <dim> <efc> <threads> <queries> <truth>"); os.Exit(2) }
 n:=atoi(os.Args[2]); dim:=atoi(os.Args[3]); efc:=atoi(os.Args[4]); threads:=atoi(os.Args[5])
 vectors:=readF32(os.Args[1],dim); if len(vectors)!=n { panic("vector count mismatch") }
 queries:=readF32(os.Args[6],dim); truth:=readU32(os.Args[7],10)
 idx,err:=hnsw.NewHNSW(hnsw.Config{M:16,Mmax:16,Mmax0:32,EfConstruction:efc,MaxLevel:16,DistanceFunc:hnsw.EuclideanDistance}); if err!=nil { panic(err) }
 idx.RandFunc=rand.New(rand.NewPCG(5050,5050)).Float64
 if threads==1 { idx.InsertBatch(vectors) } else if err=idx.BuildParallel(vectors,hnsw.BulkBuildConfig{Workers:threads,BatchSize:max(64,threads*16),EfConstruction:efc}); err!=nil { panic(err) }
 for _,ef:=range []int{32,64,128} { fmt.Printf("%d,%.4f\n",ef,recall(idx,queries,truth,ef)) }
}

func recall(idx *hnsw.HNSW, qs [][]float32, gt [][]uint32, ef int) float64 {
 hits:=0
 for i,q:=range qs { for _,id:=range idx.KNN_Search(q,10,ef) { for _,want:=range gt[i] { if id==int(want) { hits++; break } } } }
 return float64(hits)/float64(len(qs)*10)
}
func readF32(path string,w int) [][]float32 { b,e:=os.ReadFile(path); if e!=nil { panic(e) }; if len(b)%(w*4)!=0 { panic("invalid f32 file") }; rows:=make([][]float32,len(b)/(w*4)); for i:=range rows { rows[i]=make([]float32,w); for j:=0;j<w;j++ { rows[i][j]=math.Float32frombits(binary.LittleEndian.Uint32(b[(i*w+j)*4:])) } }; return rows }
func readU32(path string,w int) [][]uint32 { b,e:=os.ReadFile(path); if e!=nil { panic(e) }; if len(b)%(w*4)!=0 { panic("invalid u32 file") }; rows:=make([][]uint32,len(b)/(w*4)); for i:=range rows { rows[i]=make([]uint32,w); for j:=0;j<w;j++ { rows[i][j]=binary.LittleEndian.Uint32(b[(i*w+j)*4:]) } }; return rows }
func atoi(s string) int { v,e:=strconv.Atoi(s); if e!=nil { panic(e) }; return v }
