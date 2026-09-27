# Benchmark methodology v2

Performance work in this repository must separate algorithmic signal from hosted-runner noise. Absolute wall-clock values from different GitHub Actions runs are not directly comparable: hosted runners may use different CPU vendors, models, frequencies, and scheduling conditions.

This document defines the default methodology for optimization work.

## 1. Compare variants on the same machine

Performance claims must come from base/head comparisons executed in the same job on the same runner. Do not compare an absolute number from one workflow run with an absolute number from another workflow run.

Every benchmark artifact must record a hardware fingerprint containing at least CPU vendor, model name, family, model, and stepping. Results from different fingerprints are separate strata. Intel and AMD results must not be pooled into one median.

A result that changes sign across hardware strata is hardware-dependent, not automatically a false result.

## 2. Pin benchmark processes to CPUs

`GOMAXPROCS` is not CPU affinity. The benchmark harness pins each process with `taskset` and prefers CPUs from distinct physical cores when more than one worker is used.

Pinning is required for decisive CI runs. The selected CPU set is stored in `metadata.json`.

## 3. Warm up before measuring

Each variant gets at least one warm-up run per scenario before timed samples. Warm-up runs are not included in the statistics.

Warm-up exists to reduce first-run effects from page faults, executable/code cache population, lazy runtime initialization, and frequency ramping.

## 4. Use adjacent temporal pairs

Do not calculate the primary speedup as `median(base) / median(head)` from two independent sample sets.

The v2 harness executes adjacent pairs and alternates order:

- pair 0: base -> head
- pair 1: head -> base
- pair 2: base -> head
- pair 3: head -> base
- ...

For every pair it calculates:

`speedup_i = base_seconds_i / head_seconds_i`

The primary estimate is the geometric mean of the paired ratios. A value above 1.0 means head is faster.

This reduces bias from time-varying runner conditions because each comparison uses measurements close together in time.

## 5. Report uncertainty, not only a threshold

For each scenario the harness reports:

- base and head median wall-clock time;
- each paired speedup ratio;
- geometric-mean paired speedup;
- 95% bootstrap confidence interval;
- coefficient of variation for base and head;
- allocation change;
- CPU utilization;
- graph determinism within each variant.

The default minimum practically interesting effect is 2%.

Classification:

- **win**: estimate >= 1.02x and the 95% CI lower bound is > 1.0;
- **regression**: estimate <= 1/1.02x and the 95% CI upper bound is < 1.0;
- **inconclusive**: everything else.

`inconclusive` is not a rejection. Small positive changes may be accumulated or re-tested with more pairs or on a controlled/self-hosted runner.

## 6. Minimum repetition

For a final 100k x 128d construction decision:

- use at least 6 adjacent pairs for expected effects below 5%;
- 4 pairs may be used for a strong preliminary signal above 5%;
- efConstruction 64 and 200 are both required;
- serial tests use one pinned CPU;
- parallel tests pin the requested worker count to distinct physical cores when possible.

A first pass only slightly above the acceptance threshold requires an independent confirmation. Confirmation results must be interpreted by hardware fingerprint rather than pooled blindly with the first run.

## 7. Separate timed runs from deterministic counters

Wall-clock time answers whether a change is faster on a given machine. Deterministic counters answer whether the algorithm is doing less or different work.

The methodology runner supports a separate instrumented pass (`HNSW_BENCH_INSTRUMENT=1`) and reports:

- distance evaluations;
- directed graph edges;
- maximum observed HNSW level;
- entry point;
- graph checksum.

Instrumentation is intentionally separate from timed samples so counting overhead cannot distort the performance result.

For algorithmic changes, use the counter pass whenever practical. If a change reduces deterministic work while wall-clock remains inconclusive, do not discard the change solely because a noisy hosted runner cannot resolve the effect. Investigate the gap between work reduction and elapsed time.

For changes that should not alter the algorithm (for example compiler/toolchain comparisons), matching graph checksums and distance counts are strong evidence that the same work was executed.

## 8. Detect graph nondeterminism

The timed methodology runner emits a graph checksum after construction. If repeated runs of the same variant produce different checksums, the performance comparison is marked invalid because the variants are not repeatedly executing the same graph construction outcome.

A base checksum does not have to equal a head checksum for an intentional algorithmic change, but each variant should normally be stable internally.

## 9. Keep quality gates independent from speed gates

A performance result is not mergeable by itself. Construction optimizations must still pass:

- correctness tests;
- Recall@10 / quality checks;
- the cross-language Go/hnswlib/USearch matrix where applicable;
- allocation and memory review;
- topology/counter review when the algorithm changes.

Do not trade a measurable recall regression for a small construction speedup unless the change explicitly introduces a documented quality/performance trade-off.

## 10. Microbenchmarks are evidence, not the final verdict

A microbenchmark is useful to decide whether an idea deserves a 100k HNSW run. It does not prove end-to-end benefit.

Conversely, a positive microbenchmark that disappears end-to-end is not necessarily "wrong": Amdahl's law may make the global effect smaller than runner noise. Profile share and deterministic work counters should be used to determine whether this is expected.

## 11. Hosted-runner limitations

GitHub-hosted runners are appropriate for:

- large effects;
- paired same-run comparisons;
- regression detection;
- collecting evidence across multiple CPU models.

They are weak at resolving sub-2% effects reliably. For optimizations in that range, prefer more pairs or a stable self-hosted machine before making a final negative claim.

Never rerun a genuine performance failure merely to obtain a green status. A rerun is justified when it is explicitly collecting another hardware stratum or testing repeatability.

## 12. Harness usage

Build one benchmark runner from each source variant and run:

```bash
python benchmarks/methodology/paired_hnsw.py \
  --base-bin .artifacts/base-runner \
  --head-bin .artifacts/head-runner \
  --data .artifacts/vectors-100000-128.f32 \
  --count 100000 \
  --dim 128 \
  --efc 64,200 \
  --threads 1 \
  --pairs 6 \
  --warmups 1 \
  --counter-pass \
  --require-pinning \
  --out-dir .artifacts/methodology
```

Use `--require-win` only for a decisive optimization gate. The default mode records `win`, `regression`, or `inconclusive` without converting an inconclusive result into a CI failure.

Artifacts:

- `summary.md` - human-readable decision report;
- `raw.csv` - every timed sample and pair order;
- `metadata.json` - hardware and runner fingerprint;
- `counters.csv` - optional deterministic counter pass.

## 13. Reusable GitHub Actions workflow

`.github/workflows/paired-hnsw-performance.yml` is the standard CI entry point for decisive comparisons. It can be invoked manually with `workflow_dispatch` or called from an experiment workflow with `workflow_call`.

It accepts base/head refs plus dataset size, dimensions, efConstruction values, worker count, pair count, warm-ups, counter-pass selection, and whether a statistical win is mandatory.

The workflow deliberately copies the same methodology runner source into both git worktrees before compiling. This allows comparisons against historical refs created before methodology v2 existed while keeping the measured HNSW implementation at the exact requested base/head commits.

For exploratory experiments, leave `require_win=false` and inspect the classification. Set `require_win=true` only for the final promotion gate after the experiment has already shown a credible signal.
