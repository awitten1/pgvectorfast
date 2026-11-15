# Detailed Benchmark Results

All benchmarks below compare **`pgvectorfast`** against **`pgvector` on stock PostgreSQL** at matching index parameters and recall. See [README.md](README.md) for the summary charts.

## Summary

- **In-memory HNSW (`shared_buffers = 32 GB`)**: Over 10M vectors (`k = 10`), `pgvectorfast` delivers **3.84x** higher throughput on `SIFT10M` (`5,263` vs `1,370` QPS at `Recall@10 = 0.93`) and **3.95x** on `DEEP10M` (`4,762` vs `1,205` QPS at `Recall@10 = 0.87`) compared to `pgvector` on stock PostgreSQL.
- **Larger-than-memory HNSW (`shared_buffers = 2 GB`)**: When the index exceeds `shared_buffers` by ~5x, `pgvectorfast` achieves **5.83x** higher throughput on `SIFT10M` (`67.8` vs `11.6` QPS) and **6.57x** on `DEEP10M` (`62.1` vs `9.5` QPS). On unmodified PostgreSQL 18 (`io_method = io_uring`, `debug_io_direct = 'data'`, `shared_buffers = 2048MB`, `ef_search = 40`, `Recall@10 = 0.9359`, 1,000 `SIFT10M` queries), `pgvectorfast` reaches **64.30 QPS** (`15.55 ms` mean, `23.00 ms` p95) versus **13.78 QPS** (`72.55 ms` mean, `113.26 ms` p95) on stock PostgreSQL (**4.67x** speedup with identical recall and query results).
- **IVFFlat (`shared_buffers = 32 GB`)**: Lower per-page lookup overhead reduces median query latency by **27.3%** on `DBPedia OpenAI-1M` (`6.27 ms` vs `8.62 ms`, `Recall@3 = 0.91`) and **17.5%** on `GIST1M` (`15.37 ms` vs `18.63 ms`, `Recall@3 = 0.87`).
- **Analytical scans & joins (`shared_buffers = 128 GB`)**: Across tuple widths from `128 B` to `6144 B`, lightweight buffer-pool lookups accelerate general PostgreSQL query operators as well—reaching **2.99x** on Sequential Scan, **2.34x** on Hash Join, **1.88x** on Merge Join, **1.78x** on Nested Loop Join, and **1.52x** on Indexed Nested Loop Join at `6144 B`.

---

## In-Memory HNSW Vector Search (`shared_buffers = 32 GB`)

Measured on 10,000 queries (`k = 10`) over 10M vectors (`SIFT10M` 128D at `Recall@10 = 0.93` and `DEEP10M` 96D at `Recall@10 = 0.87`) where the HNSW index fits in memory:

| Configuration | SIFT10M Mean / Median / P95 (ms) | SIFT10M Speedup | DEEP10M Mean / Median / P95 (ms) | DEEP10M Speedup |
| --- | --- | --- | --- | --- |
| Stock `pgvector` (PostgreSQL Hashtable Baseline) | 0.73 / 0.76 / 1.05 | 1.00x | 0.86 / 0.88 / 1.14 | 1.00x |
| Calico (Array Translation) | 0.47 / 0.49 / 0.67 | 1.52x | 0.54 / 0.54 / 0.71 | 1.32x |
| Calico + Optimistic Reads | 0.30 / 0.31 / 0.41 | 2.50x | 0.32 / 0.31 / 0.43 | 2.77x |
| **`pgvectorfast` (Calico + Optimistic Reads + Group Prefetch, Default)** | **0.19 / 0.20 / 0.26** | **3.84x** | **0.20 / 0.20 / 0.26** | **3.95x** |

## Larger-Than-Memory HNSW Vector Search (`shared_buffers = 2 GB`)

When the HNSW index exceeds `shared_buffers` by ~5x, I/O latency dominates and batched group prefetch provides up to **6.57x** higher throughput:

| Configuration | SIFT10M Mean / Median / P95 (ms) | SIFT10M Speedup | DEEP10M Mean / Median / P95 (ms) | DEEP10M Speedup |
| --- | --- | --- | --- | --- |
| Stock `pgvector` (PostgreSQL Hashtable Baseline) | 85.93 / 91.07 / 134.40 | 1.00x | 105.81 / 106.98 / 142.28 | 1.00x |
| **`pgvectorfast` (Calico + Optimistic Reads + Group Prefetch, Default)** | **14.74 / 15.16 / 20.64** | **5.83x** | **16.11 / 15.73 / 21.03** | **6.57x** |

Running the `pgvectorfast` extension on unmodified PostgreSQL 18 (`io_method = io_uring`, `debug_io_direct = 'data'`, `shared_buffers = 2048MB`, `ef_search = 40`, `Recall@10 = 0.9359`, 1,000 `SIFT10M` queries) achieves **64.30 QPS** (`15.55 ms` mean, `23.00 ms` p95) compared to **13.78 QPS** (`72.55 ms` mean, `113.26 ms` p95) on stock PostgreSQL—a **4.67x** speedup with identical recall and query results.

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
