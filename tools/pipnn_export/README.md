# pipnn_export

Builds an HNSW index whose graph comes from [Ginger](https://github.com/awitten1/ginger)'s
PiPNN construction instead of from HNSW insertion.

1. `pipnn_export` reads the rows that `CREATE INDEX` on `--table`(`--column`)
   will index: every row whose column is not NULL, with its heap TID. It reads
   them with binary COPY, in one REPEATABLE READ snapshot. It then builds a
   PiPNN graph with Ginger's
   recommended configuration, as `RecommendGingerConfig` sets it: up to 64
   neighbours per point, two passes, robust pruning at `--alpha`. It writes
   the graph to a file, keyed by heap TID.
2. `CREATE INDEX` with `hnsw.build_graph_file` set loads that graph as the
   index's layer 0, with every element on that layer, and starts searches at
   Ginger's entry point.

The result is a single-layer PiPNN graph stored in an `hnsw` index, not an
HNSW graph: there are no upper layers, and every search starts from the same
entry point, as Ginger's own search does. Search and vacuum work unchanged.

Ginger's reordering and graph compression are turned off. They only relabel
and encode the graph for Ginger's own storage.

## Build

Ginger is a submodule. Its first configure fetches its dependencies (abseil,
protobuf, highway, oneTBB, KaMinPar, ...), which needs network and takes a
while. Ginger compiles with `-march=native`, so build on the machine that runs
the tool. CMake 3.24 or later and a C++20 compiler are needed.

```sh
git submodule update --init third_party/ginger
cmake -S tools/pipnn_export -B tools/pipnn_export/build -G Ninja \
      -DCMAKE_BUILD_TYPE=Release -DPG_CONFIG=$(which pg_config)
cmake --build tools/pipnn_export/build --target pipnn_export
```

## Use

```sh
tools/pipnn_export/build/pipnn_export \
  --conninfo "host=localhost dbname=postgres user=postgres" \
  --table items --column embedding \
  --out /path/items.graph
```

```sql
SET hnsw.build_graph_file = '/path/items.graph';  -- superuser only; read by the server
CREATE INDEX ON items USING hnsw (embedding vector_l2_ops) WITH (m = 32);
```

- **Table and column:** name the same table and column as the `CREATE
  INDEX`. `--table` is a SQL name: it may be schema-qualified (`app.items`),
  is resolved with the connection's `search_path`, and needs double quotes
  around a part with capitals or spaces (`--table 'app."My Items"'`).
  `--column` is the column's exact name, without quotes
  (`--column 'Emb Col'`). The tool refuses a column that is not a pgvector
  `vector`, and a partitioned table: export and index each partition instead.
  Partial and expression indexes aren't supported.
- **`m`:** layer 0 holds `2 * m` neighbours, so `m = 32` fits PiPNN's 64.
  `CREATE INDEX` refuses a graph with more neighbours per node than that.
- **Matching rows:** each graph node is matched to its row by heap TID. The
  import assumes the table is not written to or rewritten between the export
  and `CREATE INDEX`; plain `VACUUM` is fine. Most changes make `CREATE INDEX`
  fail and name the first node it can't match, but not all: export again after
  any change. A table that has had updates should be rewritten (`VACUUM FULL`
  or `CLUSTER`) before the export, so each row's `ctid` is the TID the index
  build sees.
- **Build:** the build is serial and streams. It writes element tuples during
  the heap scan, then fills in the neighbour tuples from the file. It uses
  about 42 bytes of memory per row (about 1.1 GB for 27M rows), whatever
  `maintenance_work_mem` is set to.
- **Later inserts:** rows inserted after the import go through normal HNSW
  insertion. Each gets a random level, so upper layers grow from inserted rows
  only, and the first such row becomes the entry point. Their neighbour lists
  are HNSW-pruned. The index stays correct, but its graph drifts from PiPNN's
  as rows are added; rebuild from a fresh export to restore it.
- **Vectors and metric:** only `vector` columns and L2 distance are supported.
  Ginger builds with squared L2, which ranks the same as cosine on unit
  vectors.

Options: `--alpha` (default 1.08), `--max-degree` and `--passes` (default:
Ginger's recommended values).

## Graph file

The file is little-endian:
- a 40-byte header: magic `PIPNNGR1`, version, maximum degree, node count,
  entry node, dimensions;
- an 8-byte heap TID per node;
- per node, a `uint32` degree followed by `max_degree` `uint32` neighbour
  numbers, with unused slots set to `0xFFFFFFFF`.

Node `i` is the `i`-th row read.
