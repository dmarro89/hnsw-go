# hnsw-go

Go implementation of HNSW (Hierarchical Navigable Small World) for approximate nearest neighbor search.

## Status

The project currently includes:

- a serial HNSW implementation with corrected `SEARCH-LAYER` semantics
- a tuned serial bulk builder via `InsertBatch`
- a dedicated parallel bulk builder via `BuildParallel`
- concurrent-safe query scratch state
- a zero-allocation `SearchInto` result path when the caller provides enough capacity
- deterministic correctness, quality, construction, and search benchmarks in CI
- a cross-language benchmark harness for hnsw-go, hnswlib, and USearch

`EfConstruction=64` remains the default construction setting. The project quality tests explicitly measure the construction/search trade-off rather than assuming a higher `efConstruction` is always better.

## Quick Start

### Create an index

```go
package main

import "dmarro89.github.com/hnsw-go/hnsw"

func main() {
	cfg := hnsw.DefaultConfig()

	index, err := hnsw.NewHNSW(cfg)
	if err != nil {
		panic(err)
	}

	_ = index
}
```

### Serial bulk build

Use `InsertBatch` when you want the most conservative construction path and lower temporary memory pressure.

```go
vectors := [][]float32{
	{0.1, 0.2, 0.3},
	{0.3, 0.1, 0.7},
	{0.9, 0.4, 0.2},
}

index, _ := hnsw.NewHNSW(hnsw.DefaultConfig())
index.InsertBatch(vectors)
```

### Parallel bulk build

Use `BuildParallel` when build latency matters more than temporary memory usage.

```go
vectors := loadVectors()

index, _ := hnsw.NewHNSW(hnsw.DefaultConfig())

buildCfg := hnsw.BulkBuildConfig{
	Workers:        8,
	BatchSize:      64,
	EfConstruction: 64,
}

if err := index.BuildParallel(vectors, buildCfg); err != nil {
	panic(err)
}
```

`BuildParallel` parallelizes the expensive search/proposal phase and keeps graph mutation serial.

### Reuse result storage

`SearchInto` can avoid the result allocation on the hot query path when `dst` has capacity for at least `ef` entries.

```go
dst := make([]int, 0, 64)
results := index.SearchInto(query, 10, 64, dst)
```

Each concurrent call must own its destination backing array.

## Parallel Builder Design

`BuildParallel` is not a naive concurrent version of `Insert`.

The implemented design is:

1. snapshot the current graph for a batch
2. compute insertion proposals in parallel
3. merge proposals serially into the mutable graph

HNSW insertion mutates bidirectional neighbor lists and may prune existing connections. The builder therefore parallelizes the expensive proposal work while keeping graph mutation controlled and deterministic.

## Performance And Quality

The numbers below are reference points from GitHub Actions after restoring correct HNSW search-layer semantics. They are not universal performance guarantees: CPU model, Go version, dimensionality, graph parameters, dataset distribution, and `ef` values all matter.

Current CI uses Linux/amd64, `GOMAXPROCS=4`, 128-dimensional `float32` vectors, and `EfConstruction=64` for the standard construction benchmark.

### Construction baseline

#### Serial `InsertBatch`

| Dataset | Time | Throughput | Allocated/op | Allocs/op |
| --- | ---: | ---: | ---: | ---: |
| 1,000 x 128d | 55.1 ms | 18,155 vectors/sec | 665,576 B | 3,825 |
| 10,000 x 128d | 1.79 s | 5,571 vectors/sec | 6,603,376 B | 45,584 |
| 100,000 x 128d | 73.58 s | 1,359 vectors/sec | 65,925,792 B | 491,939 |

#### Parallel `BuildParallel`

| Dataset | Time | Throughput | Allocated/op | Allocs/op |
| --- | ---: | ---: | ---: | ---: |
| 1,000 x 128d | 25.7 ms | 38,933 vectors/sec | 1,848,440 B | 5,761 |
| 10,000 x 128d | 855 ms | 11,695 vectors/sec | 29,759,832 B | 81,761 |
| 100,000 x 128d | 24.88 s | 4,019 vectors/sec | 396,029,288 B | 1,022,744 |

The allocation columns above are cumulative allocation traffic reported by Go benchmarks, not peak resident memory.

A separate apples-to-apples serial/parallel CI comparison also tracks sampled peak heap, live heap, malloc count, and GC count. On the latest 100k/128d/efConstruction=64 comparison, the serial build completed in about 51.6 s and the parallel build in about 23.5 s. Treat benchmark-to-benchmark timing differences as methodology and runner noise; compare like with like.

### Search baseline

The deterministic search benchmark uses 20k vectors, 128 dimensions, `k=10`, and the corrected search implementation.

| efSearch | Typical latency | Recall@10 | B/op | Allocs/op |
| ---: | ---: | ---: | ---: | ---: |
| 32 | 144-151 us/query | 0.6320 | 256 | 1 |
| 64 | 247-260 us/query | 0.7875 | 512 | 1 |
| 128 | 441-453 us/query | 0.9195 | 1,024 | 1 |

`SearchInto` at `efSearch=64` measured about 250-268 us/query while retaining **0 B/op and 0 allocs/op** for the result path.

The most important quality invariant is monotonic behavior: increasing `efSearch` should improve recall. CI keeps deterministic recall floors as a blocking regression gate.

### Construction quality diagnostics

On the deterministic construction-quality test with `efSearch=50`:

- `efConstruction=64`: `recall@10 = 0.9795`
- `efConstruction=100`: `recall@10 = 0.9780`
- `efConstruction=200`: `recall@10 = 0.9775`
- serial builder: `recall@10 = 0.9550`
- parallel builder: `recall@10 = 0.9485`
- parallel minus serial: `-0.0065`

These figures come from a different test corpus than the search-quality matrix above, so raw recall values should not be compared across the two tests. They are intended as regression and trade-off diagnostics.

## Cross-Language Benchmark

The repository includes a reproducible CI benchmark against:

- **hnsw-go** (Go)
- **hnswlib** (C++ core through Python bindings)
- **USearch** (C++ core through Python bindings)

The comparison currently uses:

- 10k and 100k vectors
- 128 `float32` dimensions
- the same deterministic raw vector files for every engine
- `M=16`
- `efConstruction=64` and `200`
- 1 and 4 construction threads
- `k=10`
- `efSearch=32`, `64`, and `128`
- exact brute-force ground truth for recall@10

For the cross-language harness, hnsw-go uses `M=16`, `Mmax=16`, and `Mmax0=32` so its layer connection budget is aligned as closely as the exposed APIs allow with the standard `M=16` topology used by hnswlib. This is intentionally different from older internal hnsw-go benchmark configurations that used larger `Mmax` values.

Cross-language results must be read as **quality plus performance**, not performance alone. A faster query with materially lower recall is not an equivalent result.

Memory columns also require care: hnsw-go reports Go heap deltas while the external benchmark subprocesses report process peak RSS. Those are useful diagnostics, but they are not equivalent measurements and should not be used for a direct memory-efficiency ranking.

Construction concurrency semantics also differ between implementations. The benchmark therefore measures real implementation-level wall-clock behavior under common inputs and high-level parameters; it does not claim that every engine performs an identical insertion schedule internally.

## Useful Commands

Run tests:

```bash
go test ./...
```

Run the race detector on the core packages:

```bash
go test -race ./hnsw ./structs
```

Run construction benchmarks:

```bash
go test ./benchmarks -run '^$' -bench 'BenchmarkHNSW(Parallel)?Construction' -benchmem
```

Run search benchmarks:

```bash
go test ./benchmarks -run '^$' -bench '^BenchmarkHNSWSearchPerformance$' -benchmem
```

Run deterministic quality gates:

```bash
go test ./benchmarks -run 'Test(SearchQualityMatrix|EfConstructionRecallTradeoff|ParallelBuildRecallTradeoff)$' -v
```

CI publishes benchmark summaries in GitHub Actions and uploads the raw benchmark artifacts for later comparison.

## Notes

- IDs are expected to be contiguous and match insertion order
- `KNN_Search` safely handles `K > number of indexed nodes`
- `BuildParallel` is intended for offline construction, not interactive concurrent insertion
- performance optimizations are accepted only when correctness and recall remain validated
