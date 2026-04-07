# hnsw-go

Go implementation of HNSW (Hierarchical Navigable Small World) for approximate nearest neighbor search.

## Status

The project now includes:

- a corrected serial HNSW implementation aligned with the core algorithm
- a tuned serial bulk builder via `InsertBatch`
- a dedicated parallel bulk builder via `BuildParallel`
- benchmarks and regression tests for both correctness and performance

Default construction settings currently use `EfConstruction=64`, which gave the best build-time tradeoff in this codebase without a measurable recall loss in the project benchmarks.

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

Use `InsertBatch` when you want the most balanced path in terms of time, memory, and behavior closest to the classic online insertion flow.

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

Use `BuildParallel` when build time matters more than temporary memory usage.

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

`BuildParallel` parallelizes the expensive search/proposal phase and keeps the graph-merge phase deterministic and serial.

## When To Use What

### `InsertBatch`

Choose `InsertBatch` when:

- memory pressure matters
- you want the simpler and more conservative builder
- you want behavior closest to the classic insertion process

### `BuildParallel`

Choose `BuildParallel` when:

- build latency is the main problem
- you can afford substantially higher temporary memory usage during construction
- you are building the index offline, not inserting interactively one vector at a time

## Parallel Builder Design

`BuildParallel` is not a naive concurrent version of `Insert`.

The implemented design is:

1. snapshot the current graph for a batch
2. compute insertion proposals in parallel
3. merge proposals serially into the mutable graph

This is important because HNSW insertion mutates bidirectional neighbor lists and may prune existing connections. Running plain concurrent inserts on the same graph would introduce heavy lock contention and make graph quality hard to control.

The current builder therefore optimizes the part that is actually expensive:

- greedy descent on upper layers
- `searchLayer` at lower layers
- candidate selection for new nodes

and keeps the dangerous part controlled:

- graph mutation
- pruning
- bidirectional consistency

## Performance Wrap-Up

The benchmark below uses the current optimized implementation with `EfConstruction=64`.

Command:

```bash
go test ./benchmarks -run '^$' -bench 'BenchmarkHNSW(Parallel)?Construction/(Build_(small|medium|large)_.*_ef64|BuildParallel_(small|medium|large)_.*)' -benchmem -benchtime=1x
```

### Serial `InsertBatch`

| Dataset | Time | Throughput | Memory | Allocs |
| --- | ---: | ---: | ---: | ---: |
| 1,000 x 128d | 82.6 ms | 12,109 vectors/sec | 671 KB/op | 4,815 |
| 10,000 x 128d | 2.20 s | 4,549 vectors/sec | 6.60 MB/op | 47,353 |
| 100,000 x 128d | 42.70 s | 2,342 vectors/sec | 65.87 MB/op | 487,963 |

### Parallel `BuildParallel`

| Dataset | Time | Throughput | Memory | Allocs |
| --- | ---: | ---: | ---: | ---: |
| 1,000 x 128d | 27.4 ms | 36,563 vectors/sec | 2.10 MB/op | 5,878 |
| 10,000 x 128d | 1.71 s | 5,854 vectors/sec | 57.60 MB/op | 82,987 |
| 100,000 x 128d | 19.82 s | 5,045 vectors/sec | 2.87 GB/op | 913,547 |

### Reading The Numbers

- `InsertBatch` is the better balanced builder
- `BuildParallel` is the faster builder
- the parallel path gives a clear win on build time
- the parallel path also uses much more temporary memory, especially on large builds

So the practical guidance is:

- use `InsertBatch` if you want the safer default
- use `BuildParallel` if you are building large indices offline and want to minimize wall-clock build time

## Quality Check

The project also includes a direct recall comparison between the serial and parallel builders.

Command:

```bash
go test ./benchmarks -run TestParallelBuildRecallTradeoff -v
```

Observed result on the current benchmark dataset:

- serial build: `468ms`, `recall@10 = 0.0915`
- parallel build: `146ms`, `recall@10 = 0.0945`

This comparison is useful as a relative check:

- the parallel builder was much faster
- recall was not worse in that benchmark

It should still be validated on your real dataset before making `BuildParallel` the default production path.

## Notes

- IDs are expected to be contiguous and match insertion order
- `KNN_Search` safely handles `K > number of indexed nodes`
- `BuildParallel` is intended for offline construction, not interactive concurrent insertion

## Useful Commands

CI runs the construction benchmarks automatically on `main`, on manual dispatch, and on a weekly schedule.
Results are published in two places:

- GitHub Actions job summary
- uploaded artifact `hnsw-benchmarks-<run_id>`

Run tests:

```bash
go test ./...
```

Run insertion benchmarks:

```bash
go test ./benchmarks -run '^$' -bench BenchmarkHNSWConstruction -benchmem
```

Run parallel builder benchmarks:

```bash
go test ./benchmarks -run '^$' -bench BenchmarkHNSWParallelConstruction -benchmem
```
