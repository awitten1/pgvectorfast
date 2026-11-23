/*
 * TurboQuant codec for HNSW, ported from Ginger's ginger/turboquant
 * (third_party/ginger; TurboQuant: arXiv 2504.19874).
 *
 * A vector x is encoded as follows:
 *
 * 1. Flip the sign of each coordinate (random signs from the seed), divide by
 *    ||x||, and rotate: apply an unnormalized Walsh-Hadamard transform to each
 *    of r blocks of s = d / r coordinates (s a power of 2, r odd), then mix the
 *    blocks with an r x r orthonormal matrix scaled by sqrt(r). The rotated
 *    coordinates are close to N(0, 1).
 * 2. Code each rotated coordinate with 4 bits: a sign and one of 8 magnitude
 *    buckets of a fixed Lloyd-Max codebook (hnswtq_codebooks.h). Optionally,
 *    k extra bits pick one of 2^k sub-centroids within the bucket, for
 *    reranking.
 * 3. Store ||x||^2 and the norm scaling factors ||x|| / ||codes|| (with
 *    Ginger's debiasing), one for 4 bits and one for 4+k bits.
 *
 * A query is rotated the same way. For 4-bit scoring it is quantized to int8;
 * the inner product is then an integer dot product of the query with the
 * int8 centroids of the codes, times both scaling factors. For 4+k bit
 * scoring the rotated query stays in float.
 *
 * Everything follows Ginger: the random signs, the rotation matrix (Ginger's
 * Mersenne Twister, polar method and Eigen Householder QR), the codebooks,
 * the code and query layouts, and the distance formulas of its scalar
 * reference scorers. tools/tq_check compares the two. Not ported: TurboSlice,
 * the 3-bit prefix scaling storage, PCA, and Ginger's batched (SoA) scorers.
 *
 * Unlike Ginger, a zero query is accepted (all of its distances are ||x||^2
 * for L2 and 0 for inner product), since SQL can order by distance to it.
 *
 * This file uses no PostgreSQL APIs, so tools/tq_check can link it directly.
 * Compile it without -ffast-math or FP contraction (see the Makefile): the
 * rotation must round as Ginger's does for the codes to match.
 */
#ifndef HNSWTQ_H
#define HNSWTQ_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

/* Largest r (number of blocks) supported; the matrix is r * r floats */
#define HNSW_TQ_MAX_SUBSPACES 31

/* The query's even and odd halves are each padded to a multiple of this */
#define HNSW_TQ_QUERY_PADDING 64

/* Ginger's default seed (TurboQuantConfig.seed) */
#define HNSW_TQ_DEFAULT_SEED 0

typedef enum HnswTqMetric
{
	HNSW_TQ_METRIC_L2,			/* squared L2 distance */
	HNSW_TQ_METRIC_DOT			/* negative inner product */
} HnswTqMetric;

typedef struct HnswTqCodec
{
	int			dim;
	int			subspaces;		/* r: odd */
	int			subdim;			/* s = dim / r: a power of 2 */
	uint32_t	seed;
	/* sqrt(r) * Q, column-major: rotation[j * r + i] is row i, column j */
	float		rotation[HNSW_TQ_MAX_SUBSPACES * HNSW_TQ_MAX_SUBSPACES];
	float		signs[];		/* dim entries, +1 or -1 */
} HnswTqCodec;

/* Per-vector values stored next to the codes */
typedef struct HnswTqPointInfo
{
	float		sqNorm;			/* ||x||^2 */
	float		scale4;			/* 4-bit norm scaling factor */
	float		scale4k;		/* 4+k bit norm scaling factor (scale4 if k = 0) */
} HnswTqPointInfo;

typedef struct HnswTqQuery
{
	int			dim;
	int			half;			/* offset of the odd dimensions in q8 */
	float		scale;			/* ||q|| / ||q8|| */
	float		sqNorm;			/* ||q||^2 */
	int32_t		byteSum;		/* sum of q8 */
	int8_t	   *q8;				/* 2 * half: even dimensions, then odd, zero padded */
	float	   *qf;				/* dim: the rotated query, with norm ||q|| */
} HnswTqQuery;

/* Sizes */
extern int	HnswTqSubspaces(int dim);
extern bool HnswTqValidExtraBits(int extraBits);
extern size_t HnswTqCodecSize(int dim);
extern size_t HnswTqCodeBytes(int dim);
extern size_t HnswTqExtraBytes(int dim, int extraBits);
extern size_t HnswTqQuerySize(int dim);
extern size_t HnswTqScratchSize(int dim);

/* Codec */
extern void HnswTqGenerateRotation(int subspaces, uint32_t seed, float *rotation);
extern bool HnswTqInitCodec(HnswTqCodec * codec, int dim, uint32_t seed, const float *rotation);
extern void HnswTqRotate(const HnswTqCodec * codec, float *x, float *scratch);

/* Encoding */
extern bool HnswTqEncodePoint(const HnswTqCodec * codec, const float *x, int extraBits,
							  uint8_t *codes, uint8_t *extra, HnswTqPointInfo * info, float *scratch);
extern HnswTqQuery *HnswTqInitQuery(void *mem, int dim);
extern bool HnswTqEncodeQuery(const HnswTqCodec * codec, const float *q, HnswTqQuery * query, float *scratch);

/* Distances (squared L2, or negative inner product) */
extern int32_t HnswTqDot4(const HnswTqQuery * query, const uint8_t *codes);
extern float HnswTqDistance4(const HnswTqQuery * query, const uint8_t *codes,
							 const HnswTqPointInfo * info, HnswTqMetric metric);
extern float HnswTqDistance4k(const HnswTqQuery * query, const uint8_t *codes, const uint8_t *extra,
							  int extraBits, const HnswTqPointInfo * info, HnswTqMetric metric);

#endif							/* HNSWTQ_H */
