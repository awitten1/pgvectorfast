# Detailed Benchmark Results

The HNSW comparison uses pgvectorfast 0.2.1, upstream pgvector 0.8.7,
Chroma 1.5.9, and Qdrant 1.20.1 on the same SIFT10M (128D) and DEEP10M (96D)
vectors. The IVFFlat and SQL operator results below are from earlier runs.

The README figures use the latest Chroma rerun: 855.1 / 856.7 QPS at 16 GiB RAM
and 14.2 / 13.6 QPS at 2 GiB RAM with 16 GiB of NVMe swap (SIFT10M / DEEP10M).
The other systems' plotted measurements used no swap. The original no-swap
tables are retained below, followed by the Chroma rerun's full results.

## HNSW setup

- Hardware: Intel Core i9-13900K, 62 GiB host RAM, Samsung 980 PRO NVMe.
- Each service has one pinned server CPU, one client on a separate CPU, and
  one outstanding query. Both PostgreSQL variants use PostgreSQL 18.7.
- Indexes use float32 vectors, L2 distance, `M=16`, and `ef_construction=64`,
  without quantization. The PostgreSQL variants share an identical heap/index
  snapshot. Chroma and Qdrant build their own HNSW graphs from the same vectors.
- Total service memory, including file cache, is limited to **16 GiB** in memory
  and **2 GiB** for the larger-than-memory runs, with swap disabled. PostgreSQL
  uses 14 GiB and 1536 MiB of shared buffers, respectively, plus `io_uring` and
  direct data I/O. Files from the preceding service are evicted from its cache
  before starting the next cgroup.
- A fixed random permutation (NumPy seed `20261115`) splits each dataset's
  10,000 queries: 4,000 for calibration, 6,000 disjoint queries for in-memory
  timings, and the first 1,000 of those for larger-than-memory timings. The
  first 2,000 calibration queries warm each engine. Every engine uses the same
  order and split. The smallest `ef_search` reaching calibration Recall@10 of
  at least 0.9336 for SIFT10M or 0.8712 for DEEP10M is selected. The tables
  report actual timed-query recall.
- Results are medians of three repeats; chart error bars show their full range.
  Latency includes the native client API
  call: prepared binary psycopg queries for PostgreSQL, gRPC for Qdrant, and
  HTTP for Chroma. QPS is `1000 / mean_latency_ms`; recall checks and result
  hashing are outside the timer. Index builds are paused during timings.
- Larger-than-memory results require physical NVMe reads during the timed
  queries. The raw results include cgroup memory and I/O counters and Calico
  mapping checks.

See the [runner and reproduction instructions](bench/hnsw/README.md) and
[raw results](bench/hnsw/results) for calibration trials, all timing repeats,
image identities, and environment details.

## HNSW summary

- At 16 GiB per service, pgvectorfast delivers **5.32x** the throughput of upstream pgvector on SIFT10M and **5.88x** on DEEP10M.
- At 2 GiB per service, pgvectorfast delivers **5.70x** the throughput of upstream pgvector on SIFT10M and **5.91x** on DEEP10M.

## Search calibration

| Dataset | System | ef_search | Calibration Recall@10 |
| --- | --- | ---: | ---: |
| SIFT10M | pgvector | 40 | 0.93388 |
| SIFT10M | pgvectorfast | 40 | 0.93388 |
| SIFT10M | Chroma | 61 | 0.93415 |
| SIFT10M | Qdrant | 35 | 0.93458 |
| DEEP10M | pgvector | 39 | 0.87320 |
| DEEP10M | pgvectorfast | 39 | 0.87315 |
| DEEP10M | Chroma | 59 | 0.87275 |
| DEEP10M | Qdrant | 34 | 0.87262 |

## In-Memory HNSW (16 GiB per Service)

| Dataset | System | ef_search | Recall@10 | QPS | Mean (ms) | P95 (ms) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| SIFT10M | pgvector | 40 | 0.9334 | 927.0 | 1.079 | 1.460 |
| SIFT10M | pgvectorfast | 40 | 0.9334 | 4,933.1 | 0.203 | 0.250 |
| SIFT10M | Chroma | 61 | 0.9335 | 847.4 | 1.180 | 1.296 |
| SIFT10M | Qdrant | 35 | 0.9335 | 1,668.5 | 0.599 | 0.675 |
| DEEP10M | pgvector | 39 | 0.8655 | 836.1 | 1.196 | 1.550 |
| DEEP10M | pgvectorfast | 39 | 0.8656 | 4,913.3 | 0.204 | 0.245 |
| DEEP10M | Chroma | 59 | 0.8664 | 862.9 | 1.159 | 1.282 |
| DEEP10M | Qdrant | 34 | 0.8673 | 1,659.1 | 0.603 | 0.668 |

## Larger-Than-Memory HNSW (2 GiB per Service)

| Dataset | System | ef_search | Recall@10 | QPS | Mean (ms) | P95 (ms) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| SIFT10M | pgvector | 40 | 0.9356 | 11.4 | 87.668 | 129.250 |
| SIFT10M | pgvectorfast | 40 | 0.9355 | 65.0 | 15.393 | 19.753 |
| SIFT10M | Chroma | 61 | — | OOM at 2 GiB | — | — |
| SIFT10M | Qdrant | 35 | 0.9372 | 21.2 | 47.240 | 74.524 |
| DEEP10M | pgvector | 39 | 0.8753 | 10.4 | 96.219 | 127.511 |
| DEEP10M | pgvectorfast | 39 | 0.8753 | 61.4 | 16.294 | 19.903 |
| DEEP10M | Chroma | 59 | — | OOM at 2 GiB | — | — |
| DEEP10M | Qdrant | 34 | 0.8779 | 17.9 | 55.917 | 74.860 |

With OS swap disabled, Chroma's process was OOM-killed while loading the 10M-vector HNSW index under the 2 GiB limit on SIFT10M and DEEP10M. Throughput and recall are unavailable for those runs.

## Chroma with OS Paging

Chroma 1.5.9 was rerun on the same persisted 10M-vector indexes with a **2 GiB
RAM limit and 16 GiB of NVMe-backed swap**. The indexes loaded under the 2 GiB
limit. Search settings remained `ef_search=61` for SIFT10M and `59` for DEEP10M.
Each paging run used the same 2,000 warm-up queries and 1,000 held-out timing
queries as the original larger-than-memory runs, with three repeats. The 16 GiB
in-memory references were also repeated with 6,000 timing queries per repeat.

| Dataset | RAM cap | Swap allowance | ef_search | Recall@10 | QPS | Mean / P95 (ms) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| SIFT10M | 16 GiB | 0 GiB | 61 | 0.9335 | 855.1 | 1.169 / 1.288 |
| SIFT10M | 2 GiB | 16 GiB | 61 | 0.9339 | 14.2 | 70.431 / 108.953 |
| DEEP10M | 16 GiB | 0 GiB | 59 | 0.8664 | 856.7 | 1.167 / 1.283 |
| DEEP10M | 2 GiB | 16 GiB | 59 | 0.8749 | 13.6 | 73.312 / 95.622 |

These runs permit OS swap; the four-system comparison above disables it. The
paging cases ended with 7.22 GiB of swap in use for SIFT10M and 6.00 GiB for
DEEP10M. Timed queries performed 10.15 GiB and 10.45 GiB of physical reads,
respectively, and both runs completed without OOM events. The temporary swap
area was removed afterward, with the original swap configuration preserved.

[Raw paging results and environment](bench/hnsw/results/chroma-paging) include
actual cgroup limits, swap usage, page-fault counts, I/O counters, and all three
repeats. See the [paging command](bench/hnsw/README.md#run) to reproduce the run.

## IVFFlat Vector Search (`shared_buffers = 32 GB`)

Even without HNSW-specific optimistic read or graph prefetch paths, replacing PostgreSQL's shared buffer lookup hash table reduces query latency for IVFFlat scans:

| Dataset | Dimensions | Recall | Stock `pgvector` Median (ms) | `pgvectorfast` Median (ms) | Latency Reduction |
| --- | --- | --- | --- | --- | --- |
| `DBPedia OpenAI-1M` | 1536 | `Recall@3 = 0.91` | 8.62 | **6.27** | **-27.3%** |
| `GIST1M` | 960 | `Recall@3 = 0.87` | 18.63 | **15.37** | **-17.5%** |

## Analytical Scans & Joins (`shared_buffers = 128 GB`)

Because `CALL calico.activate()` upgrades PostgreSQL's shared buffer pool for all relations in the instance, standard SQL analytical queries also benefit across row widths from `128 B` to `6144 B` (`1.03x`–`1.17x` at `128 B`, scaling up to `2.99x` at `6144 B`):

| PostgreSQL Operator | Speedup vs. Stock PostgreSQL (`6144 B` Rows) |
| --- | --- |
| Sequential Scan | **2.99x** |
| Hash Join | **2.34x** |
| Merge Join | **1.88x** |
| Nested Loop Join | **1.78x** |
| Indexed Nested Loop Join | **1.52x** |
