# Filtered vector search: pgvector and pgvectorfast

The recorded results cover ArXiv at 16 GiB and 2 GiB RAM. See the
[results and interpretation](../../BENCHMARKS.md#filtered-vector-search-arxiv).

This runner compares upstream pgvector 0.8.7 and pgvectorfast using the
published filtered queries and ground truth.

- [ArXiv](https://github.com/qdrant/ann-filtering-benchmark-datasets):
  2,138,591 vectors, 384 dimensions, cosine distance; category, submitter, and
  update-date predicates.
- [Big ANN YFCC-10M](https://github.com/harsha-simhadri/big-ann-benchmarks/blob/main/neurips23/README.md):
  10,000,000 vectors, 192 dimensions, L2 distance; conjunctions of one or two tags.
  Use the actual `base.10M.u8bin` dataset, not the dummy or unfiltered variants.

Install `bench/hnsw/requirements.txt` in a Python environment. Use the same
PostgreSQL Docker images described in `bench/hnsw/README.md`. Extract the ArXiv
archive into `DATA/arxiv/`; put the five YFCC files into `DATA/yfcc10m/`:

```sh
mkdir -p DATA/arxiv DATA/yfcc10m
curl -fL https://storage.googleapis.com/ann-filtered-benchmark/datasets/arxiv.tar.gz \
  -o DATA/arxiv.tar.gz
tar -xzf DATA/arxiv.tar.gz -C DATA/arxiv \
  vectors.npy payloads.jsonl tests.jsonl filters.json
for name in base.10M.u8bin query.public.100K.u8bin base.metadata.10M.spmat \
  query.metadata.public.100K.spmat GT.public.ibin; do
  curl -fL "https://dl.fbaipublicfiles.com/billion-scale-ann-benchmarks/yfcc100M/$name" \
    -o "DATA/yfcc10m/$name"
done
```

Check YFCC's selected reference neighbors against their tag predicates and
published distances before building:

```sh
python bench/filtered/check_yfcc.py --data-root DATA --output yfcc-input-check.json
```

Build one fixture per dataset on real NVMe storage:

```sh
python bench/filtered/build.py --dataset arxiv --data-root DATA \
  --work-dir /nvme/filtered/postgres --output RESULTS
```

The builder imports zero-based IDs, float32 vectors, and the metadata needed for
all published query predicates. It creates HNSW with `m=16, ef_construction=64`,
GIN indexes on array metadata, and B-tree indexes on ArXiv submitter/date fields.
Both implementations query exactly the same persisted heap and HNSW graph.
Stock runs load the upstream pgvector library with no shared preload; Calico
is activated only in pgvectorfast runs.
Index construction is outside the query timings.

For each dataset and implementation:

```sh
python bench/filtered/run.py --dataset arxiv --store pgvector --action validate \
  --data-root DATA --work-dir /nvme/filtered/postgres --output RESULTS
python bench/filtered/run.py --dataset arxiv --store pgvector --action calibrate \
  --data-root DATA --work-dir /nvme/filtered/postgres --output RESULTS
python bench/filtered/run.py --dataset arxiv --store pgvector --action measure \
  --data-root DATA --work-dir /nvme/filtered/postgres --output RESULTS --ram-gib 16
python bench/filtered/run.py --dataset arxiv --store pgvector --action measure \
  --data-root DATA --work-dir /nvme/filtered/postgres --output RESULTS --ram-gib 2
```

Repeat with `--store pgvectorfast`. For the optimized ArXiv runs in the README,
build a pgvectorfast image from commit `42993e12` and select it with `--image`
when running `calibrate` and `measure`. Use `--dataset yfcc10m --ram-gib 32` for its
larger memory budget, and `--ram-gib 2` for its constrained-memory run. Host swap
must be available: every measured configuration permits an additional 16 GiB
of swap, identically for both implementations. The runner records actual cgroup
RAM/swap limits and usage. It does not provision or modify host swap itself.

## Measurement protocol

- The primary `--plan auto` workload lets PostgreSQL choose between filtered
  HNSW, metadata-index scans followed by exact ranking, and sequential scans.
  `--plan hnsw` is available for a separate diagnostic; it discourages sequential,
  bitmap, and sort paths. Query plans are recorded, so confirm the actual path.
- HNSW uses `relaxed_order` iterative scans with an outer materialized-CTE sort
  for ordered results. It retrieves up to 50 filtered candidates, then returns
  the nearest 10 after sorting. This compensates for relaxed scan ordering;
  both implementations execute the same SQL. Exact reference checks request 10
  directly without using HNSW.
  Custom query plans account for each query's actual predicate and vector.
- A fixed seed (`20261117`) selects 100 calibration and 300 disjoint timing
  queries per filter type: three ArXiv types and one/two-tag YFCC types. The
  published benchmark populations and selected query IDs are retained.
  The 2 GiB runs use the first 100 selected timing queries per filter type,
  preserving their shuffled order, and warm with the first 30 calibration
  queries per type. Both implementations use the identical subset.
- Calibration searches increasing `ef_search` and `max_scan_tuples` budgets.
  Each filter type must reach mean filtered Recall@10 >= 0.95 and return 10
  results for >= 99% of calibration queries. Timing uses the selected settings
  without tuning on held-out queries.
- Reference validation runs exact PostgreSQL searches for three calibration
  queries per filter type and checks agreement with the published neighbors.
- Query execution uses one pinned server CPU, a separate pinned client CPU, and
  one outstanding request. Timings include planning, filtering, sorting, and
  binary psycopg result retrieval. Recall calculations and result diagnostics
  are outside the timer.
- Larger memory runs prewarm all benchmark relations. Every run then warms with
  calibration queries and measures three repeats of the same held-out queries.
  Result JSON preserves individual latencies, IDs, recall, and result counts,
  plus per-filter aggregates and the distribution of actual query plans.
- The 2 GiB limit includes PostgreSQL and file cache; shared buffers are 1536 MiB.
  The 16/32 GiB cases use 14/28 GiB shared buffers. PostgreSQL uses direct data I/O
  and `io_uring`. Before each restart, only the stopped fixture's files are
  advised out of cache so memory accounting belongs to the new cgroup.

Results must be compared at similar held-out recall and completeness. Report
which execution strategies PostgreSQL actually used; a filtered query may use
an exact metadata-index scan instead of HNSW.

Validate and summarize the ArXiv measurements saved in the output directory.

```sh
python bench/filtered/summarize.py --results RESULTS --datasets arxiv
```

For example, an ArXiv category query uses the same SQL in both implementations:

```sql
SET hnsw.iterative_scan = relaxed_order;

WITH candidates AS MATERIALIZED (
    SELECT id, embedding <=> $1 AS distance
    FROM items
    WHERE labels @> $2::text[]
    ORDER BY distance
    LIMIT 50
)
SELECT id, distance
FROM candidates
ORDER BY distance + 0.0
LIMIT 10;
```

This follows pgvector's [iterative scan and outer-sort pattern](../../PGVECTOR.md#iterative-index-scans),
with candidate overfetching to improve filtered recall. YFCC uses `<->` and an
integer-array containment predicate instead.
