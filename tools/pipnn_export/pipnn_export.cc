// pipnn_export: builds a PiPNN graph over vectors SELECTed out of Postgres and
// writes it, keyed by heap TID, for CREATE INDEX with hnsw.build_graph_file
// (see README.md in this directory). Links Ginger from third_party/ginger.
//
//   pipnn_export --conninfo "host=10.10.1.2 dbname=postgres user=postgres" \
//                --query "SELECT ctid, embedding FROM items_hn" \
//                --out /mydata/items_hn.pipnn
//
// The query must return two columns: the row's ctid and a pgvector `vector`.
// Rows are read with COPY (query) TO STDOUT (FORMAT binary) inside one
// REPEATABLE READ transaction, after a count(*) in the same snapshot, so the
// vectors fit one exact allocation.
//
// The graph is built by Ginger's own builder with the recommended config
// (RecommendGingerConfig: PiPNN, max degree 64, two passes, robust pruning at
// --alpha), so it is the graph a Ginger index of the same data would search.
// Reordering and graph compression are turned off: they only relabel and
// encode the graph for Ginger's storage. --max-degree and --passes override
// the recommended values.
//
// Output (little-endian):
//   char[8]  magic "PIPNNGR1"
//   uint32   version (1)
//   uint32   max_degree
//   uint64   num_nodes
//   uint64   entry_node        Ginger's search entry point
//   uint32   dimensions
//   uint32   reserved (0)
//   num_nodes x { uint32 block; uint16 offset; uint16 pad }   heap TID per node
//   num_nodes x { uint32 degree; uint32 neighbor[max_degree] } unused slots
//                                                             are 0xFFFFFFFF
// Node i is the i-th row the query returned.

#include <arpa/inet.h>
#include <libpq-fe.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <variant>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "ginger/ginger.h"

namespace {

using clk = std::chrono::steady_clock;

[[noreturn]] void Die(const std::string& msg) {
  std::fprintf(stderr, "error: %s\n", msg.c_str());
  std::exit(1);
}

double Seconds(clk::time_point t0) {
  return std::chrono::duration<double>(clk::now() - t0).count();
}

struct Options {
  std::string conninfo;
  std::string query;
  std::string out;
  float alpha = 1.08f;
  int max_degree = 0;  // 0: the recommended value
  int passes = 0;      // 0: the recommended value
};

struct Tid {
  uint32_t block;
  uint16_t offset;
  uint16_t pad;
};
static_assert(sizeof(Tid) == 8);

// Reads big-endian values from a COPY BINARY stream held in `buf`.
class Reader {
 public:
  explicit Reader(const std::string& buf) : buf_(buf) {}
  bool Has(size_t n) const { return pos_ + n <= buf_.size(); }
  size_t pos() const { return pos_; }
  void Seek(size_t p) { pos_ = p; }
  int16_t I16() {
    uint16_t v;
    std::memcpy(&v, buf_.data() + pos_, 2);
    pos_ += 2;
    return static_cast<int16_t>(ntohs(v));
  }
  int32_t I32() {
    uint32_t v;
    std::memcpy(&v, buf_.data() + pos_, 4);
    pos_ += 4;
    return static_cast<int32_t>(ntohl(v));
  }
  uint32_t U32() { return static_cast<uint32_t>(I32()); }
  uint16_t U16() { return static_cast<uint16_t>(I16()); }
  float F32() {
    const uint32_t bits = U32();
    float f;
    std::memcpy(&f, &bits, 4);
    return f;
  }
  void Skip(size_t n) { pos_ += n; }

 private:
  const std::string& buf_;
  size_t pos_ = 0;
};

PGresult* Exec(PGconn* conn, const std::string& sql, ExecStatusType want) {
  PGresult* res = PQexec(conn, sql.c_str());
  if (PQresultStatus(res) != want) {
    const std::string err = PQerrorMessage(conn);
    PQclear(res);
    Die(sql + ": " + err);
  }
  return res;
}

// Reads every row of `query` into `tids` and `vectors`; returns the dimensions.
int ReadRows(PGconn* conn, const std::string& query, std::vector<Tid>& tids,
             std::vector<float>& vectors) {
  PQclear(Exec(conn, "BEGIN ISOLATION LEVEL REPEATABLE READ READ ONLY",
               PGRES_COMMAND_OK));
  PGresult* res = Exec(conn, "SELECT count(*) FROM (" + query + ") q",
                       PGRES_TUPLES_OK);
  const int64_t expected = std::atoll(PQgetvalue(res, 0, 0));
  PQclear(res);
  if (expected <= 0) Die("the query returned no rows");
  if (expected > INT32_MAX) Die("more rows than Ginger node ids can address");
  std::fprintf(stderr, "reading %lld rows\n", static_cast<long long>(expected));

  PQclear(Exec(conn, "COPY (" + query + ") TO STDOUT (FORMAT binary)",
               PGRES_COPY_OUT));
  tids.reserve(expected);

  std::string buf;   // unparsed bytes
  bool header_done = false, trailer = false;
  int dim = -1;
  int64_t rows = 0;
  const auto t0 = clk::now();
  for (;;) {
    char* chunk = nullptr;
    const int n = PQgetCopyData(conn, &chunk, 0);
    if (n == -1) break;  // COPY done
    if (n == -2) Die(std::string("COPY failed: ") + PQerrorMessage(conn));
    buf.append(chunk, n);
    PQfreemem(chunk);

    Reader r(buf);
    if (!header_done) {
      if (!r.Has(19)) continue;
      if (std::memcmp(buf.data(), "PGCOPY\n\377\r\n\0", 11) != 0) Die("not a COPY BINARY stream");
      r.Skip(11);
      r.Skip(4);  // flags
      const int32_t ext = r.I32();
      if (!r.Has(ext)) continue;
      r.Skip(ext);
      header_done = true;
    }
    for (;;) {
      const size_t start = r.pos();
      if (!r.Has(2)) break;
      const int16_t fields = r.I16();
      if (fields == -1) {
        trailer = true;
        break;
      }
      if (fields != 2) Die("the query must return exactly two columns (ctid, vector)");
      // ctid
      if (!r.Has(4)) { r.Seek(start); break; }
      const int32_t tlen = r.I32();
      if (tlen != 6) Die("the first column must be a ctid");
      if (!r.Has(6 + 4)) { r.Seek(start); break; }
      Tid tid{r.U32(), r.U16(), 0};
      // vector
      const int32_t vlen = r.I32();
      if (vlen < 4) Die("the second column must be a non-null pgvector vector");
      if (!r.Has(vlen)) { r.Seek(start); break; }
      const int d = r.I16();
      r.Skip(2);  // unused
      if (vlen != 4 + 4 * d) Die("the second column is not a pgvector vector");
      if (dim < 0) {
        dim = d;
        vectors.assign(static_cast<size_t>(expected) * dim, 0.0f);
      } else if (d != dim) {
        Die("rows have different dimensions");
      }
      if (rows >= expected) Die("the query returned more rows than counted");
      float* dst = vectors.data() + static_cast<size_t>(rows) * dim;
      for (int i = 0; i < d; i++) dst[i] = r.F32();
      tids.push_back(tid);
      rows++;
      if (rows % 1000000 == 0)
        std::fprintf(stderr, "  %lld rows, %.0f s\n", static_cast<long long>(rows), Seconds(t0));
    }
    buf.erase(0, r.pos());
  }
  PGresult* end = PQgetResult(conn);
  if (PQresultStatus(end) != PGRES_COMMAND_OK) Die(std::string("COPY failed: ") + PQerrorMessage(conn));
  PQclear(end);
  PQclear(Exec(conn, "COMMIT", PGRES_COMMAND_OK));
  if (!trailer) Die("COPY stream ended without its trailer");
  if (rows != expected) Die("the query returned fewer rows than counted");
  std::fprintf(stderr, "read %lld rows of %d dimensions in %.0f s\n",
               static_cast<long long>(rows), dim, Seconds(t0));
  return dim;
}

void Write(FILE* f, const void* p, size_t n, const std::string& path) {
  if (std::fwrite(p, 1, n, f) != n) Die("cannot write " + path);
}

}  // namespace

int main(int argc, char** argv) {
  Options o;
  for (int i = 1; i < argc; i++) {
    const std::string a = argv[i];
    auto next = [&]() -> std::string {
      if (i + 1 >= argc) Die(a + " needs a value");
      return argv[++i];
    };
    if (a == "--conninfo") o.conninfo = next();
    else if (a == "--query") o.query = next();
    else if (a == "--out") o.out = next();
    else if (a == "--alpha") o.alpha = std::stof(next());
    else if (a == "--max-degree") o.max_degree = std::stoi(next());
    else if (a == "--passes") o.passes = std::stoi(next());
    else Die("unknown argument " + a);
  }
  if (o.query.empty() || o.out.empty()) Die("--query and --out are required (and usually --conninfo)");

  PGconn* conn = PQconnectdb(o.conninfo.c_str());
  if (PQstatus(conn) != CONNECTION_OK) Die(std::string("cannot connect: ") + PQerrorMessage(conn));
  std::vector<Tid> tids;
  std::vector<float> vectors;
  const int dim = ReadRows(conn, o.query, tids, vectors);
  PQfinish(conn);
  const size_t n = tids.size();

  absl::StatusOr<ginger::GingerSearcherConfig> rc =
      ginger::RecommendGingerConfig(dim, ginger::SQUARED_L2, o.alpha);
  if (!rc.ok()) Die("RecommendGingerConfig: " + std::string(rc.status().message()));
  ginger::GingerSearcherConfig config = *std::move(rc);
  config.set_reordering_algorithm(ginger::REORDERING_ALGORITHM_NONE);
  config.set_use_graph_compression(false);
  if (o.max_degree > 0) config.mutable_pipnn_config()->set_max_degree(o.max_degree);
  if (o.passes > 0) config.mutable_pipnn_config()->set_num_passes(o.passes);

  absl::StatusOr<std::shared_ptr<const ginger::SpanDataset>> ds =
      ginger::SpanDataset::Wrap(absl::MakeConstSpan(vectors), static_cast<size_t>(dim));
  if (!ds.ok()) Die("SpanDataset::Wrap: " + std::string(ds.status().message()));

  std::fprintf(stderr, "building the PiPNN graph (max degree %d, %d passes, alpha %.2f)\n",
               config.pipnn_config().max_degree(), config.pipnn_config().num_passes(), o.alpha);
  const auto t0 = clk::now();
  absl::StatusOr<std::unique_ptr<ginger::GingerSearcher>> s =
      ginger::GingerSearcherBuilder(config).AddDataset(*ds).Build();
  if (!s.ok()) Die("Build: " + std::string(s.status().message()));
  std::fprintf(stderr, "built in %.0f s\n", Seconds(t0));

  const ginger::GingerSearcher& searcher = **s;
  if (!searcher.remapping().empty()) Die("the graph was reordered; expected node i = row i");
  const auto* flat = std::get_if<std::unique_ptr<const ginger::FlatGraph>>(&searcher.index());
  if (flat == nullptr || *flat == nullptr) Die("the built index is not a flat graph");
  const ginger::FlatGraph& g = **flat;
  if (static_cast<size_t>(g.NumNodes()) != n) Die("the graph has a different number of nodes than rows");

  const uint32_t max_degree = static_cast<uint32_t>(g.MaxDegree());
  const uint64_t num_nodes = n;
  const uint64_t entry = static_cast<uint64_t>(searcher.entry_point());
  const uint32_t version = 1, dims = static_cast<uint32_t>(dim), reserved = 0;

  const std::string tmp = o.out + ".tmp";
  FILE* f = std::fopen(tmp.c_str(), "wb");
  if (f == nullptr) Die("cannot create " + tmp);
  Write(f, "PIPNNGR1", 8, tmp);
  Write(f, &version, 4, tmp);
  Write(f, &max_degree, 4, tmp);
  Write(f, &num_nodes, 8, tmp);
  Write(f, &entry, 8, tmp);
  Write(f, &dims, 4, tmp);
  Write(f, &reserved, 4, tmp);
  Write(f, tids.data(), tids.size() * sizeof(Tid), tmp);
  std::vector<uint32_t> row(max_degree + 1);
  uint64_t edges = 0;
  for (size_t u = 0; u < n; u++) {
    const absl::Span<const ginger::GingerNodeId> nb = g.Neighbors(static_cast<ginger::GingerNodeId>(u));
    row[0] = static_cast<uint32_t>(nb.size());
    for (uint32_t j = 0; j < max_degree; j++)
      row[1 + j] = j < nb.size() ? static_cast<uint32_t>(nb[j]) : 0xFFFFFFFFu;
    edges += nb.size();
    Write(f, row.data(), row.size() * 4, tmp);
  }
  if (std::fclose(f) != 0) Die("cannot write " + tmp);
  if (std::rename(tmp.c_str(), o.out.c_str()) != 0) Die("cannot rename " + tmp + " to " + o.out);
  std::fprintf(stderr, "wrote %s: %llu nodes, max degree %u, mean degree %.1f, entry node %llu\n",
               o.out.c_str(), static_cast<unsigned long long>(num_nodes), max_degree,
               static_cast<double>(edges) / static_cast<double>(n),
               static_cast<unsigned long long>(entry));
  return 0;
}
