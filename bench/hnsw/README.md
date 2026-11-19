# HNSW comparison

Compare pgvectorfast, upstream pgvector, single-node Chroma, and Qdrant on
SIFT10M and DEEP10M using native client APIs and matched Recall@10.

## Inputs and setup

Use Linux x86-64 with Docker and enough free NVMe storage for the indexes.
Install the Python dependencies in a virtual environment:

```sh
python3 -m venv /tmp/hnsw-venv
/tmp/hnsw-venv/bin/pip install -r bench/hnsw/requirements.txt
docker pull chromadb/chroma:1.5.9
docker pull qdrant/qdrant:v1.20.1
docker build -t pgvectorfast-021-test:trixie .
docker build -f bench/hnsw/Dockerfile.pgvector -t pgvector-upstream-bench:0.8.7 .
```

`--data-root` contains:

- `SIFT10M/SIFT10Mfeatures.mat`: HDF5 `fea` matrix; the first 10M rows are
  vectors and the next 10,000 rows are queries. Values are converted to float32.
- `SIFT10M/ground_truth_indices.npy`: zero-based exact top-10 neighbor IDs.
- `deep10m_data/base.10M.fbin` and `query.public.10K.fbin`: two little-endian
  uint32 header fields (rows, dimensions), followed by float32 vectors.
- `deep10m_data/groundtruth.ibin`: the same header format, followed by int32 IDs.

Prepare PostgreSQL snapshots once using a PostgreSQL 18 installation with
pgvectorfast installed. The loader uses binary COPY and creates an HNSW index
with `m=16, ef_construction=64`:

```sh
/tmp/hnsw-venv/bin/python bench/hnsw/create_fixture.py \
  --dataset sift10m --data-root /path/to/datasets \
  --pg-bin /path/to/postgres18/bin \
  --data-dir /path/to/datasets/perf_work/sift10m_template
```

Repeat for `deep10m` and `perf_work/deep10m_template`. Existing clean snapshots
with the same table/index names and one-based vector IDs can be reused. Both
PostgreSQL variants use the same snapshot; the upstream image provides a library
alias for its existing `$libdir/vectorfast` catalog references.

## Run

Run each store and dataset separately. `--work-dir` must be on actual NVMe for
timings. Imports and recall calibration may use other disks.

```sh
/tmp/hnsw-venv/bin/python bench/hnsw/crossstore.py \
  --store chroma --dataset sift10m --data-root /path/to/datasets \
  --work-dir /nvme/hnsw --output /path/to/results
```

Stores are `pgvector`, `pgvectorfast`, `chroma`, and `qdrant`. Datasets are
`sift10m` and `deep10m`. Use `--action build`, `calibrate`, or `measure` to run
phases separately. Once a PostgreSQL fixture has been copied, reuse it with
`calibrate` and `measure` when switching between the two PostgreSQL variants.
Select available cores with `--build-cpus`, `--server-cpu`, and `--client-cpu`.
In import metadata, `build_seconds` covers the last invocation, including pauses.

For an OS-paging run, provision swap on the host and select `paging`:

```sh
/tmp/hnsw-venv/bin/python bench/hnsw/crossstore.py \
  --store chroma --dataset sift10m --data-root /path/to/datasets \
  --work-dir /nvme/hnsw --output /path/to/paging-results \
  --action measure --mode paging --swap-gib 16
```

This caps resident memory, including file cache, at 2 GiB and allows an additional
16 GiB of swap. Docker's [combined memory-and-swap limit](https://docs.docker.com/engine/containers/resource_constraints/#--memory-swap-details)
is therefore 18 GiB.
The runner records the actual cgroup limits, swap usage, memory pressure, and I/O
counters. Host swap is provisioned separately; the runner does not enable swap or
change the host's swappiness. Paging results use a separate `*-paging.json` file.

## Method

- Index configuration: float32 L2, HNSW `M=16`, `ef_construction=64`, no quantization.
- Query indices are shuffled once with NumPy's default generator and seed
  `20261115`; every engine uses the same order. The split files record the
  original indices. Search effort is calibrated on the first 4,000 queries,
  choosing the smallest `ef_search` that reaches Recall@10 of at least
  0.9336 (SIFT10M) or 0.8712 (DEEP10M).
- The first 2,000 shuffled queries warm the engine. Timings use the remaining
  6,000 queries in memory, or the first 1,000 of those for the larger-than-memory
  case, with three repeats. Calibration and timing queries are disjoint.
- All server threads share one pinned CPU. The client uses another CPU and
  issues one outstanding query. PostgreSQL uses prepared binary parameters,
  Qdrant uses gRPC, and Chroma uses HTTP. Latencies include client API calls;
  recall calculation and result hashing are outside the timer. QPS is
  `1000 / mean_latency_ms` for those sequential API calls.
- The default `inmem` and `oom` modes have service-wide memory limits of 16 GiB
  and 2 GiB, including file cache, with swap disabled. The explicit `paging` mode
  permits the configured swap allowance. PostgreSQL uses 14 GiB and 1536 MiB of shared buffers,
  respectively, and `io_uring` with direct data I/O.
- Before each service starts, its stopped data files are advised out of the
  file cache so cache charged to an earlier cgroup cannot bypass the new cap.
  The runner records cgroup memory and physical read counters and requires
  at least 64 MiB of physical reads during larger-than-memory timings.
- Chroma is restarted between search-parameter trials. Its complete graph is
  checkpointed before calibration, so reloading does not change the graph.
- Time engines sequentially, with other index construction stopped or paused.
  A build can be gated by creating `STORE-DATASET.pause` in the results folder.
  When `STORE-DATASET.paused.json` appears, no ingestion RPC is outstanding;
  freeze that container with `docker pause`. Unfreeze it before removing the flag.

A memory-limit failure is recorded explicitly, with container OOM state, rather
than assigned a throughput. The summary checks the vector count, repeat count,
physical I/O for larger-than-memory runs, and recall spread. It allows at most
0.1 percentage point of recall variation across repeats and 1 percentage point
across engines for the same dataset and memory budget.

```sh
/tmp/hnsw-venv/bin/python bench/hnsw/summarize.py \
  --results /path/to/results --assets assets
```

This exports the two HNSW charts, a CSV summary, and Markdown result tables.
When `chroma-paging/` is present in the results directory, the charts use its
latest Chroma in-memory and paging measurements, with the swap settings labeled
on the larger-than-memory figure. The CSV and tables retain the original
four-system no-swap results.

The README figures share fonts, colors, and layout helpers in
`bench/plot_style.py`. Use the saved measurements in `bench/figure_data.json`
to regenerate the filtered-search, IVFFlat, and analytical-query figures.

```sh
/tmp/hnsw-venv/bin/python bench/plot_other_figures.py --assets assets
```

The figures use Noto Sans from GitHub's Markdown font stack. Install it before
rendering (`fonts-noto-core` on Debian/Ubuntu).
