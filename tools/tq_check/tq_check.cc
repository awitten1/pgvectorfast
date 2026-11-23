// Checks the C TurboQuant codec (src/hnswtq.c) against Ginger's TurboQuant on
// random vectors, for several dimensions, seeds, extra bits and both metrics:
//
// - random signs and the rotation matrix generated from the seed;
// - rotated vectors, 4-bit codes, extra bits, norms and scaling factors;
// - int8 and float queries;
// - 4-bit and 4+k bit distances, against Ginger's scalar reference scorers.
//
// Downstream of the rotation matrix, the C codec is given Ginger's matrix, so
// differences come from the code under test, not from the matrix. Then it
// reports the recall of 4-bit search with 4+k reranking on random data, as a
// sanity check of the estimates. Exits nonzero if a check fails.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <utility>
#include <vector>

#include "absl/types/span.h"
#include "hwy/base.h"
#include "ginger/turboquant/codec/codec.h"
#include "ginger/turboquant/datapoint/datapoint.h"
#include "ginger/turboquant/layout/dataset_soa.h"
#include "ginger/turboquant/query/query.h"
#include "ginger/turboquant/rotation/rotation.h"
#include "ginger/turboquant/scoring/scoring_scalar_reference.h"

extern "C" {
#include "hnswtq.h"
}

namespace {

int failures = 0;

void Check(bool ok, const char* what, int dim, uint32_t seed, int k, const char* detail) {
  if (!ok) {
    ++failures;
    std::printf("FAIL d=%d seed=%u k=%d: %s (%s)\n", dim, seed, k, what, detail);
  }
}

float RelDiff(float a, float b) {
  const float scale = std::max({std::fabs(a), std::fabs(b), 1e-30f});
  return std::fabs(a - b) / scale;
}

struct Codec {
  std::vector<char> codec_mem;
  std::vector<float> scratch;
  HnswTqCodec* codec;

  Codec(int dim, uint32_t seed, const float* rotation)
      : codec_mem(HnswTqCodecSize(dim)), scratch(HnswTqScratchSize(dim) / sizeof(float)) {
    codec = reinterpret_cast<HnswTqCodec*>(codec_mem.data());
    HnswTqInitCodec(codec, dim, seed, rotation);
  }
};

struct Point {
  std::vector<uint8_t> codes;
  std::vector<uint8_t> extra;
  HnswTqPointInfo info;
};

struct Query {
  std::vector<double> mem;  // 8-byte aligned
  HnswTqQuery* query;

  explicit Query(int dim) : mem((HnswTqQuerySize(dim) + 7) / 8) {
    query = HnswTqInitQuery(mem.data(), dim);
  }
};

std::vector<std::vector<float>> RandomVectors(int n, int dim, std::mt19937& gen) {
  std::normal_distribution<float> normal;
  std::uniform_real_distribution<float> uniform(-1.0f, 1.0f);
  std::vector<std::vector<float>> vectors(n, std::vector<float>(dim));
  for (int i = 0; i < n; ++i) {
    // Varying norms, and some vectors with a common offset (not centered)
    const float sigma = std::exp(3.0f * uniform(gen));
    const float offset = (i % 3 == 0) ? 0.5f * uniform(gen) : 0.0f;
    for (int j = 0; j < dim; ++j) vectors[i][j] = offset + sigma * normal(gen);
  }
  return vectors;
}

// Signs and rotation matrix generated from the seed
void CheckRotationMatrix(int dim, uint32_t seed) {
  ginger::HadamardAndMatrixRotation rotation(dim, seed);
  Codec ours(dim, seed, nullptr);
  const int r = ours.codec->subspaces;
  char detail[128];

  int sign_mismatches = 0;
  for (int i = 0; i < dim; ++i) sign_mismatches += ours.codec->signs[i] != rotation.signs()[i];
  std::snprintf(detail, sizeof(detail), "%d of %d differ", sign_mismatches, dim);
  Check(sign_mismatches == 0, "signs", dim, seed, -1, detail);

  const Eigen::MatrixXf& expected = rotation.GetRotationMatrix();
  Check(expected.rows() == r, "subspaces", dim, seed, -1, "r differs");
  float max_diff = 0.0f;
  for (int j = 0; j < r; ++j)
    for (int i = 0; i < r; ++i)
      max_diff = std::max(max_diff, std::fabs(ours.codec->rotation[j * r + i] - expected(i, j)));
  std::snprintf(detail, sizeof(detail), "max |diff| %.3g", max_diff);
  Check(max_diff <= 1e-5f, "rotation matrix", dim, seed, -1, detail);
  std::printf("d=%-5d seed=%-3u r=%-2d s=%-5d rotation matrix max |diff| %.3g\n", dim, seed, r,
              dim / r, max_diff);
}

// Everything downstream of the rotation matrix, for one metric
void CheckCodec(int dim, uint32_t seed, ginger::TurboQuantDistanceMetric metric) {
  const bool l2 = metric == ginger::TurboQuantDistanceMetric::kSquaredL2Distance;
  const HnswTqMetric tq_metric = l2 ? HNSW_TQ_METRIC_L2 : HNSW_TQ_METRIC_DOT;
  ginger::TurboQuantCodec ginger_codec(dim, seed, metric);
  ginger::HadamardAndMatrixRotation rotation(dim, seed);
  const Eigen::MatrixXf& matrix = rotation.GetRotationMatrix();
  std::vector<float> matrix_col_major(matrix.data(), matrix.data() + matrix.size());
  Codec ours(dim, seed, matrix_col_major.data());
  char detail[160];

  std::mt19937 gen(dim * 7919u + seed);
  std::vector<std::vector<float>> points = RandomVectors(200, dim, gen);
  std::fill(points[1].begin(), points[1].end(), 0.0f);  // a zero vector
  std::vector<std::vector<float>> queries = RandomVectors(20, dim, gen);

  // Rotation of each point
  float max_rot_diff = 0.0f;
  for (const auto& x : points) {
    float sq = 0.0f;
    for (float v : x) sq += v * v;
    if (sq == 0.0f) continue;
    const float norm = std::sqrt(sq);
    std::vector<float> expected(dim), actual(dim);
    if (!rotation.FlipRandomSignsNormalizeAndRotate(x, norm, absl::MakeSpan(expected)).ok()) {
      Check(false, "Ginger rotation", dim, seed, -1, "error");
      continue;
    }
    const float inv_norm = 1.0f / norm;
    for (int i = 0; i < dim; ++i) actual[i] = x[i] * ours.codec->signs[i] * inv_norm;
    HnswTqRotate(ours.codec, actual.data(), ours.scratch.data());
    for (int i = 0; i < dim; ++i) max_rot_diff = std::max(max_rot_diff, std::fabs(actual[i] - expected[i]));
  }
  const int r = ours.codec->subspaces;
  std::snprintf(detail, sizeof(detail), "max |diff| %.3g", max_rot_diff);
  // r = 1 and 3 round as Ginger does, except that Ginger's scalar tail for
  // blocks narrower than a vector (s < 16) fuses differently; other r use an
  // Eigen matrix product whose summation order differs
  const bool exact_rotation = r <= 3 && dim / r >= 16;
  Check(exact_rotation ? max_rot_diff == 0.0f : max_rot_diff <= 1e-4f, "rotation", dim, seed, -1, detail);

  // Queries
  std::vector<Query> our_queries;
  std::vector<ginger::TurboQuantQuery> ginger_queries;
  for (const auto& q : queries) {
    auto gq = ginger::TurboQuantQuery::Create(ginger_codec, q, /*enable_higher_precision_scoring=*/true);
    if (!gq.ok()) {
      Check(false, "Ginger query", dim, seed, -1, "error");
      return;
    }
    our_queries.emplace_back(dim);
    HnswTqQuery* oq = our_queries.back().query;
    Check(HnswTqEncodeQuery(ours.codec, q.data(), oq, ours.scratch.data()), "encode query", dim, seed, -1, "false");

    Check(gq->num_bytes_per_query() == static_cast<size_t>(2 * oq->half), "query layout", dim, seed, -1,
          "q8 size differs");
    int q8_mismatches = 0, q8_max_diff = 0;
    for (int i = 0; i < 2 * oq->half; ++i) {
      const int diff = std::abs(oq->q8[i] - gq->query_data()[i]);
      q8_mismatches += diff != 0;
      q8_max_diff = std::max(q8_max_diff, diff);
    }
    std::snprintf(detail, sizeof(detail), "%d of %d differ, max |diff| %d", q8_mismatches, dim, q8_max_diff);
    // A coordinate can round the other way if its norm differs in the last bit
    Check(q8_max_diff <= 1 && q8_mismatches <= dim / 500 + 1, "int8 query", dim, seed, -1, detail);
    Check(q8_mismatches > 0 || oq->byteSum == gq->query_byte_sum(), "query byte sum", dim, seed, -1, "differs");
    std::snprintf(detail, sizeof(detail), "scale %.3g, sq norm %.3g", RelDiff(oq->scale, gq->norm_scaling_factor()),
                  RelDiff(oq->sqNorm, gq->unquantized_squared_norm()));
    Check(RelDiff(oq->scale, gq->norm_scaling_factor()) <= 1e-5f &&
              RelDiff(oq->sqNorm, gq->unquantized_squared_norm()) <= 1e-5f,
          "query norms", dim, seed, -1, detail);
    float max_qf_diff = 0.0f;
    const float qnorm = std::sqrt(oq->sqNorm);
    for (int i = 0; i < dim; ++i)
      max_qf_diff = std::max(max_qf_diff, std::fabs(oq->qf[i] - gq->float_query_data()[i]) / qnorm);
    std::snprintf(detail, sizeof(detail), "max |diff| / ||q|| %.3g", max_qf_diff);
    Check(max_qf_diff <= 1e-6f, "float query", dim, seed, -1, detail);
    ginger_queries.push_back(*std::move(gq));
  }

  for (int k : {0, 1, 2, 3, 4, 6, 8}) {
    // Points. ||x|| is summed in a different order than Eigen's, so a
    // coordinate near a boundary can rarely code differently; codes, extra
    // bits and scaling factors are compared point by point, and scaling
    // factors only where the codes they depend on match.
    std::vector<Point> our_points(points.size());
    long code_mismatches = 0, extra_mismatches = 0;
    float max_scale_diff = 0.0f;
    for (size_t p = 0; p < points.size(); ++p) {
      Point& op = our_points[p];
      op.codes.resize(HnswTqCodeBytes(dim));
      op.extra.resize(HnswTqExtraBytes(dim, k));
      Check(HnswTqEncodePoint(ours.codec, points[p].data(), k, op.codes.data(), op.extra.data(), &op.info,
                              ours.scratch.data()),
            "encode point", dim, seed, k, "false");

      auto gp = ginger::EncodeDatapoint(ginger_codec, points[p], 4 + k, /*has_3bit_prefix=*/false);
      if (!gp.ok()) {
        Check(false, "Ginger encode", dim, seed, k, "error");
        return;
      }
      const ginger::TurboQuantBasePointView view = gp->view();
      Check(view.quantized_data.size() == op.codes.size() && view.extra_precision_data.size() == op.extra.size(),
            "code sizes", dim, seed, k, "differ");
      long point_code_mismatches = 0, point_extra_mismatches = 0;
      for (int d = 0; d < dim; ++d) {
        const int shift = (d % 2) * 4;
        point_code_mismatches += ((op.codes[d / 2] >> shift) & 0xF) != ((view.quantized_data[d / 2] >> shift) & 0xF);
      }
      // (With an odd dimension the last high nibble is padding, which Ginger
      // fills with 8 for k = 0 and 0 otherwise; it scores against a zero.)
      for (size_t i = 0; i < op.extra.size(); ++i)
        point_extra_mismatches += op.extra[i] != view.extra_precision_data[i];
      code_mismatches += point_code_mismatches;
      extra_mismatches += point_extra_mismatches;

      max_scale_diff = std::max(max_scale_diff, RelDiff(op.info.sqNorm, view.unquantized_squared_norm));
      if (point_code_mismatches == 0)
        max_scale_diff = std::max(max_scale_diff, RelDiff(op.info.scale4, view.norm_scaling_factor_4bit));
      if (point_code_mismatches == 0 && point_extra_mismatches == 0)
        max_scale_diff = std::max(max_scale_diff, RelDiff(op.info.scale4k, view.norm_scaling_factor_4plusk));
    }
    const long total_dims = static_cast<long>(points.size()) * dim;
    // Allow a few boundary flips; more where the rotation itself differs (r > 3)
    const long allowed = (r <= 3 ? total_dims / 10000 : total_dims / 1000) + 2;
    std::snprintf(detail, sizeof(detail), "%ld of %ld dimensions differ", code_mismatches, total_dims);
    Check(code_mismatches <= allowed, "4-bit codes", dim, seed, k, detail);
    std::snprintf(detail, sizeof(detail), "%ld bytes differ", extra_mismatches);
    Check(extra_mismatches <= allowed, "extra bits", dim, seed, k, detail);
    std::snprintf(detail, sizeof(detail), "max relative diff %.3g", max_scale_diff);
    Check(max_scale_diff <= 1e-5f, "norms and scaling factors", dim, seed, k, detail);

    // Distances, against Ginger's scalar reference scorers on its own encoding
    std::vector<absl::Span<const float>> spans(points.begin(), points.end());
    ginger::TurboQuantDatasetOptions options;
    options.quantization_bits = 4 + k;
    auto db = ginger::TurboQuantDatasetSoA::Create(ginger_codec, spans, options);
    if (!db.ok()) {
      Check(false, "Ginger dataset", dim, seed, k, std::string(db.status().message()).c_str());
      return;
    }
    std::vector<int32_t> indices(points.size());
    for (size_t i = 0; i < indices.size(); ++i) indices[i] = static_cast<int32_t>(i);

    float max_d4_diff = 0.0f, max_d4k_diff = 0.0f;
    for (size_t qi = 0; qi < queries.size(); ++qi) {
      std::vector<float> expected4(points.size()), expected4k(points.size());
      ginger::TurboQuantScoringInt8ScalarReference(*db, ginger_queries[qi], absl::MakeSpan(expected4), indices);
      if (k > 0)
        ginger::TurboQuantScoringExtraPrecisionScalarReference(*db, ginger_queries[qi], absl::MakeSpan(expected4k),
                                                               indices);
      const HnswTqQuery* oq = our_queries[qi].query;
      for (size_t p = 0; p < points.size(); ++p) {
        const Point& op = our_points[p];
        // Relative to the size of the terms, since an L2 distance cancels them
        const float scale = l2 ? op.info.sqNorm + oq->sqNorm : std::sqrt(op.info.sqNorm * oq->sqNorm);
        if (scale == 0.0f) continue;
        // For inner product, Ginger's dataset keeps the 4-bit scaling factor
        // as a float16 (NormLayout::kScalingOnly); HNSW keeps a float.
        HnswTqPointInfo info4 = op.info;
        if (!l2) info4.scale4 = hwy::F32FromF16(hwy::F16FromF32(info4.scale4));
        const float d4 = HnswTqDistance4(oq, op.codes.data(), &info4, tq_metric);
        max_d4_diff = std::max(max_d4_diff, std::fabs(d4 - expected4[p]) / scale);
        if (k > 0) {
          const float d4k = HnswTqDistance4k(oq, op.codes.data(), op.extra.data(), k, &op.info, tq_metric);
          max_d4k_diff = std::max(max_d4k_diff, std::fabs(d4k - expected4k[p]) / scale);
        }
      }
    }
    std::snprintf(detail, sizeof(detail), "4-bit %.3g, 4+k %.3g (max |diff| / scale)", max_d4_diff, max_d4k_diff);
    Check(max_d4_diff <= 1e-4f && max_d4k_diff <= 1e-4f, "distances", dim, seed, k, detail);
    if (seed == 0 && l2)
      std::printf("d=%-5d k=%d codes differ %ld/%ld, extra bytes differ %ld, scales %.2g, distances %.2g / %.2g\n",
                  dim, k, code_mismatches, total_dims, extra_mismatches, max_scale_diff, max_d4_diff, max_d4k_diff);
  }

  // A zero query: every L2 distance is ||x||^2, every inner product 0
  Query zero(dim);
  std::vector<float> zeros(dim, 0.0f);
  Check(HnswTqEncodeQuery(ours.codec, zeros.data(), zero.query, ours.scratch.data()), "zero query", dim, seed, -1,
        "false");
  Point p0;
  p0.codes.resize(HnswTqCodeBytes(dim));
  p0.extra.resize(HnswTqExtraBytes(dim, 2));
  HnswTqEncodePoint(ours.codec, points[0].data(), 2, p0.codes.data(), p0.extra.data(), &p0.info, ours.scratch.data());
  const float d4 = HnswTqDistance4(zero.query, p0.codes.data(), &p0.info, tq_metric);
  const float d4k = HnswTqDistance4k(zero.query, p0.codes.data(), p0.extra.data(), 2, &p0.info, tq_metric);
  const float expected = l2 ? p0.info.sqNorm : 0.0f;
  Check(d4 == expected && d4k == expected, "zero query distances", dim, seed, -1, "not ||x||^2 or 0");
}

// Recall@10 of 4-bit search (top `candidates` by 4-bit estimate), reranked by
// the 4+k estimate, against exact squared L2, on random Gaussian data
void ReportRecall(int dim, int k, int candidates) {
  const int n = 5000, nq = 50, topk = 10;
  std::mt19937 gen(12345u + dim);
  std::normal_distribution<float> normal;
  std::vector<std::vector<float>> points(n, std::vector<float>(dim));
  std::vector<std::vector<float>> queries(nq, std::vector<float>(dim));
  for (auto& v : points)
    for (float& x : v) x = normal(gen);
  for (auto& v : queries)
    for (float& x : v) x = normal(gen);

  Codec ours(dim, HNSW_TQ_DEFAULT_SEED, nullptr);
  std::vector<Point> encoded(n);
  for (int i = 0; i < n; ++i) {
    encoded[i].codes.resize(HnswTqCodeBytes(dim));
    encoded[i].extra.resize(HnswTqExtraBytes(dim, k));
    HnswTqEncodePoint(ours.codec, points[i].data(), k, encoded[i].codes.data(), encoded[i].extra.data(),
                      &encoded[i].info, ours.scratch.data());
  }

  double recall4 = 0.0, recall_rerank = 0.0;
  for (const auto& q : queries) {
    Query oq(dim);
    HnswTqEncodeQuery(ours.codec, q.data(), oq.query, ours.scratch.data());
    std::vector<std::pair<float, int>> exact(n), est4(n);
    for (int i = 0; i < n; ++i) {
      float d = 0.0f;
      for (int j = 0; j < dim; ++j) d += (points[i][j] - q[j]) * (points[i][j] - q[j]);
      exact[i] = {d, i};
      est4[i] = {HnswTqDistance4(oq.query, encoded[i].codes.data(), &encoded[i].info, HNSW_TQ_METRIC_L2), i};
    }
    std::partial_sort(exact.begin(), exact.begin() + topk, exact.end());
    std::partial_sort(est4.begin(), est4.begin() + candidates, est4.end());
    std::vector<std::pair<float, int>> rerank(candidates);
    for (int c = 0; c < candidates; ++c) {
      const Point& p = encoded[est4[c].second];
      rerank[c] = {HnswTqDistance4k(oq.query, p.codes.data(), p.extra.data(), k, &p.info, HNSW_TQ_METRIC_L2),
                   est4[c].second};
    }
    std::partial_sort(rerank.begin(), rerank.begin() + topk, rerank.end());
    for (int a = 0; a < topk; ++a) {
      for (int b = 0; b < topk; ++b) {
        recall4 += est4[b].second == exact[a].second;
        recall_rerank += rerank[b].second == exact[a].second;
      }
    }
  }
  std::printf("d=%-5d k=%d recall@10 on %d random points: 4-bit %.3f, top %d reranked by 4+%d bits %.3f\n", dim, k,
              n, recall4 / (nq * topk), candidates, k, recall_rerank / (nq * topk));
}

}  // namespace

int main() {
  const int dims[] = {3, 7, 8, 16, 30, 62, 96, 100, 128, 200, 384, 768, 960, 1024, 1536};
  for (int dim : dims) {
    for (uint32_t seed : {0u, 42u}) {
      CheckRotationMatrix(dim, seed);
      CheckCodec(dim, seed, ginger::TurboQuantDistanceMetric::kSquaredL2Distance);
      CheckCodec(dim, seed, ginger::TurboQuantDistanceMetric::kDotProductDistance);
    }
  }

  for (int dim : {128, 768})
    for (int k : {0, 2, 4, 8}) ReportRecall(dim, k, 100);

  std::printf(failures == 0 ? "OK\n" : "%d checks failed\n", failures);
  return failures == 0 ? 0 : 1;
}
