/*
 * TurboQuant codec for HNSW, ported from Ginger (see hnswtq.h).
 *
 * Each function names the Ginger function it follows. Floating-point
 * expressions keep Ginger's order of operations, and use fmaf() where Ginger
 * uses a fused multiply-add, so that this file, compiled without contraction,
 * rounds as Ginger does.
 */
#include "hnswtq.h"

#include <float.h>
#include <math.h>
#include <string.h>

#if defined(__AVX2__)
#include <immintrin.h>
#endif

#include "hnswtq_codebooks.h"

/* debiasing_4bit() without the 3-bit prefix (encode_datapoint.cc) */
#define HNSW_TQ_DEBIAS_4BIT 1.0047848f

/* kDebiasing4PlusKBit (encode_datapoint.cc), indexed by k - 1 */
static const float hnsw_tq_debias_4plusk[8] = {
	1.0012649f, 1.0003301f, 1.0000854f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f
};

/* kValueCap (encode_query.cc): rotated query coordinates are clamped to it */
#define HNSW_TQ_QUERY_VALUE_CAP 3.91724f

/*
 * Mersenne Twister (std::mt19937), for Ginger's signs and rotation matrix
 */
typedef struct HnswTqMt19937
{
	uint32_t	mt[624];
	int			index;
}			HnswTqMt19937;

static void
MtSeed(HnswTqMt19937 * gen, uint32_t seed)
{
	gen->mt[0] = seed;
	for (int i = 1; i < 624; i++)
		gen->mt[i] = 1812433253U * (gen->mt[i - 1] ^ (gen->mt[i - 1] >> 30)) + (uint32_t) i;
	gen->index = 624;
}

static uint32_t
MtNext(HnswTqMt19937 * gen)
{
	uint32_t	y;

	if (gen->index >= 624)
	{
		for (int i = 0; i < 624; i++)
		{
			y = (gen->mt[i] & 0x80000000U) | (gen->mt[(i + 1) % 624] & 0x7FFFFFFFU);
			gen->mt[i] = gen->mt[(i + 397) % 624] ^ (y >> 1) ^ ((y & 1U) ? 0x9908B0DFU : 0U);
		}
		gen->index = 0;
	}

	y = gen->mt[gen->index++];
	y ^= y >> 11;
	y ^= (y << 7) & 0x9D2C5680U;
	y ^= (y << 15) & 0xEFC60000U;
	y ^= y >> 18;
	return y;
}

/*
 * Standard normal samples by the polar method
 * (portable_random::NormalDistribution)
 */
typedef struct HnswTqNormal
{
	bool		hasCached;
	float		cached;
}			HnswTqNormal;

static float
NormalUniform(HnswTqMt19937 * gen)
{
	const float canonical = (float) MtNext(gen) / 4294967296.0f;

	return 2.0f * canonical + -1.0f;
}

static float
NormalNext(HnswTqNormal * normal, HnswTqMt19937 * gen)
{
	float		u;
	float		v;
	float		s;
	float		factor;

	if (normal->hasCached)
	{
		normal->hasCached = false;
		return normal->cached;
	}

	do
	{
		float		uu;
		float		vv;

		u = NormalUniform(gen);
		v = NormalUniform(gen);
		uu = u * u;
		vv = v * v;
		s = uu + vv;
	} while (s > 1 || s == 0);

	factor = sqrtf(-2 * logf(s) / s);
	normal->cached = v * factor;
	normal->hasCached = true;
	return u * factor;
}

/*
 * Number of blocks r: the dimension with all factors of 2 removed
 * (HadamardAndMatrixRotation constructor)
 */
int
HnswTqSubspaces(int dim)
{
	int			r = dim;

	while (r > 1 && r % 2 == 0)
		r /= 2;
	return r;
}

/*
 * Supported extra bits per dimension (ValidateTurboQuantExtraBits)
 */
bool
HnswTqValidExtraBits(int extraBits)
{
	return extraBits == 0 || extraBits == 1 || extraBits == 2 || extraBits == 3 ||
		extraBits == 4 || extraBits == 6 || extraBits == 8;
}

size_t
HnswTqCodecSize(int dim)
{
	return offsetof(HnswTqCodec, signs) + sizeof(float) * (size_t) dim;
}

/*
 * Bytes of 4-bit codes: two dimensions per byte, the even one in the low
 * nibble
 */
size_t
HnswTqCodeBytes(int dim)
{
	return ((size_t) dim + 1) / 2;
}

/*
 * Bytes of extra bits, packed contiguously, plus a padding byte so every
 * value can be read as two bytes (GetExtraPrecisionBytes)
 */
size_t
HnswTqExtraBytes(int dim, int extraBits)
{
	if (extraBits == 0)
		return 0;
	return ((size_t) dim * extraBits + 7) / 8 + 1;
}

static int
QueryHalf(int dim)
{
	return (int) ((HnswTqCodeBytes(dim) + HNSW_TQ_QUERY_PADDING - 1) / HNSW_TQ_QUERY_PADDING * HNSW_TQ_QUERY_PADDING);
}

static size_t
QueryHeaderSize(void)
{
	/* Keep q8 aligned for vector loads */
	return (sizeof(HnswTqQuery) + 63) / 64 * 64;
}

size_t
HnswTqQuerySize(int dim)
{
	return QueryHeaderSize() + 2 * (size_t) QueryHalf(dim) + sizeof(float) * (size_t) dim;
}

/*
 * Scratch space for HnswTqEncodePoint() and HnswTqEncodeQuery(): the rotated
 * vector, and space for the rotation
 */
size_t
HnswTqScratchSize(int dim)
{
	return 2 * sizeof(float) * (size_t) dim;
}

/*
 * Apply the Householder reflector H = I - tau * [1 essential] [1 essential]^T
 * from the left to rows [row0, n) and columns [col0, n) of the column-major
 * n x n matrix a (MatrixBase::applyHouseholderOnTheLeft)
 */
static void
ApplyHouseholderOnTheLeft(float *a, int n, int row0, int col0, const float *essential, float tau)
{
	int			rows = n - row0;

	if (rows == 1)
	{
		for (int c = col0; c < n; c++)
			a[c * n + row0] *= 1.0f - tau;
		return;
	}

	if (tau == 0.0f)
		return;

	for (int c = col0; c < n; c++)
	{
		float	   *col = &a[c * n];
		float		tmp = 0.0f;

		/* tmp = essential^T * bottom + row 0 */
		for (int i = 1; i < rows; i++)
			tmp += essential[i - 1] * col[row0 + i];
		tmp += col[row0];

		col[row0] -= tau * tmp;
		for (int i = 1; i < rows; i++)
			col[row0 + i] -= tau * essential[i - 1] * tmp;
	}
}

/*
 * Generate the r x r mixing matrix sqrt(r) * Q, where Q comes from the
 * Householder QR decomposition of a matrix of standard normal samples
 * (GenerateRandomOrthonormalMatrix, with Eigen's HouseholderQR and
 * householderQ(); r is below Eigen's block sizes, so its unblocked paths)
 *
 * Writes r * r floats, column-major. The result can differ from Ginger's in
 * the last bits (logf() and Eigen's vectorized reductions), so an index
 * stores the matrix rather than regenerating it.
 */
void
HnswTqGenerateRotation(int r, uint32_t seed, float *rotation)
{
	HnswTqMt19937 gen;
	HnswTqNormal normal = {false, 0.0f};
	float		a[HNSW_TQ_MAX_SUBSPACES * HNSW_TQ_MAX_SUBSPACES];
	float		tau[HNSW_TQ_MAX_SUBSPACES];
	float		scale;

	/* Samples fill the matrix row by row */
	MtSeed(&gen, seed);
	for (int i = 0; i < r; i++)
		for (int j = 0; j < r; j++)
			a[j * r + i] = NormalNext(&normal, &gen);

	/* QR: column k's reflector is kept below the diagonal (householder_qr_inplace_unblocked) */
	for (int k = 0; k < r; k++)
	{
		float	   *col = &a[k * r];
		float		tailSqNorm = 0.0f;
		float		c0 = col[k];
		float		beta;

		/* MatrixBase::makeHouseholder */
		for (int i = k + 1; i < r; i++)
			tailSqNorm += col[i] * col[i];

		if (tailSqNorm <= FLT_MIN)
		{
			tau[k] = 0.0f;
			beta = c0;
			for (int i = k + 1; i < r; i++)
				col[i] = 0.0f;
		}
		else
		{
			beta = sqrtf(c0 * c0 + tailSqNorm);
			if (c0 >= 0.0f)
				beta = -beta;
			for (int i = k + 1; i < r; i++)
				col[i] = col[i] / (c0 - beta);
			tau[k] = (beta - c0) / beta;
		}
		col[k] = beta;

		ApplyHouseholderOnTheLeft(a, r, k, k + 1, &col[k + 1], tau[k]);
	}

	/* Q = H_0 H_1 ... H_{r-1}, applied to the identity from the last (HouseholderSequence::evalTo) */
	for (int i = 0; i < r * r; i++)
		rotation[i] = 0.0f;
	for (int i = 0; i < r; i++)
		rotation[i * r + i] = 1.0f;
	for (int k = r - 1; k >= 0; k--)
		ApplyHouseholderOnTheLeft(rotation, r, k, k, &a[k * r + k + 1], tau[k]);

	/* std::sqrt(int) is a double, converted to float by Eigen */
	scale = (float) sqrt((double) r);
	for (int i = 0; i < r * r; i++)
		rotation[i] = scale * rotation[i];
}

/*
 * Initialize a codec. 'rotation' is the r x r mixing matrix, or NULL to
 * generate it from the seed. Returns false if the dimension is not positive
 * or needs more than HNSW_TQ_MAX_SUBSPACES blocks.
 */
bool
HnswTqInitCodec(HnswTqCodec * codec, int dim, uint32_t seed, const float *rotation)
{
	HnswTqMt19937 gen;
	int			r;

	if (dim <= 0)
		return false;
	r = HnswTqSubspaces(dim);
	if (r > HNSW_TQ_MAX_SUBSPACES)
		return false;

	codec->dim = dim;
	codec->subspaces = r;
	codec->subdim = dim / r;
	codec->seed = seed;

	memset(codec->rotation, 0, sizeof(codec->rotation));
	if (rotation != NULL)
		memcpy(codec->rotation, rotation, sizeof(float) * r * r);
	else
		HnswTqGenerateRotation(r, seed, codec->rotation);

	/* A fresh generator with the same seed (UniformInt(gen, 2) is one bit) */
	MtSeed(&gen, seed);
	for (int i = 0; i < dim; i++)
		codec->signs[i] = 2.0f * (float) (MtNext(&gen) & 1U) - 1;
	return true;
}

/*
 * Unnormalized Walsh-Hadamard transform of n (a power of 2) floats in place.
 * Ginger fuses stages in registers, but applies them in this order, with one
 * addition or subtraction per output per stage, so the results are the same
 * (WalshHadamardImpl).
 */
static void
WalshHadamard(float *x, int n)
{
	for (int len = 1; len < n; len *= 2)
	{
		for (int i = 0; i < n; i += 2 * len)
		{
			for (int j = i; j < i + len; j++)
			{
				const float a = x[j];
				const float b = x[j + len];

				x[j] = a + b;
				x[j + len] = a - b;
			}
		}
	}
}

/*
 * Rotate x in place: Walsh-Hadamard transform of each block, then mix the
 * blocks, viewing x as a column-major s x r matrix X: X := X * R
 * (HadamardAndMatrixRotation::Rotate). 'scratch' holds dim floats.
 */
void
HnswTqRotate(const HnswTqCodec * codec, float *x, float *scratch)
{
	const int	r = codec->subspaces;
	const int	s = codec->subdim;
	const float *rot = codec->rotation;

	if (s > 1)
	{
		for (int b = 0; b < r; b++)
			WalshHadamard(x + b * s, s);
	}

	if (r == 1)
		return;

	if (r == 3)
	{
		/* SubspaceMixR3Impl: elementwise, so it can work in place */
		float	   *x0 = x;
		float	   *x1 = x + s;
		float	   *x2 = x + 2 * s;

		for (int i = 0; i < s; i++)
		{
			const float a0 = x0[i];
			const float a1 = x1[i];
			const float a2 = x2[i];

			x0[i] = fmaf(a2, rot[2], fmaf(a1, rot[1], a0 * rot[0]));
			x1[i] = fmaf(a2, rot[5], fmaf(a1, rot[4], a0 * rot[3]));
			x2[i] = fmaf(a2, rot[8], fmaf(a1, rot[7], a0 * rot[6]));
		}
		return;
	}

	/* Ginger uses an Eigen matrix product here; its summation order differs */
	for (int j = 0; j < r; j++)
	{
		for (int i = 0; i < s; i++)
		{
			float		sum = x[i] * rot[j * r];

			for (int k = 1; k < r; k++)
				sum = fmaf(x[k * s + i], rot[j * r + k], sum);
			scratch[j * s + i] = sum;
		}
	}
	memcpy(x, scratch, sizeof(float) * r * s);
}

/*
 * out = Rotate(signs * x / norm) (FlipRandomSignsNormalizeAndRotate)
 */
static void
FlipSignsNormalizeAndRotate(const HnswTqCodec * codec, const float *x, float norm, float *out, float *scratch)
{
	const float invNorm = 1.0f / norm;

	/* FlipSignsAndScaleImpl */
	for (int i = 0; i < codec->dim; i++)
		out[i] = x[i] * codec->signs[i] * invNorm;

	HnswTqRotate(codec, out, scratch);
}

/*
 * ||x||^2. Ginger sums in float with Eigen's vectorized order; this sums in
 * double, so the two can differ in the last bit.
 */
static float
SquaredNorm(const float *x, int dim)
{
	double		sum = 0.0;

	for (int i = 0; i < dim; i++)
		sum += (double) x[i] * (double) x[i];
	return (float) sum;
}

/*
 * The 4-bit code of a rotated coordinate: bit 3 is the sign (set unless
 * positive), bits 0-2 the bucket, the number of boundaries at or below |x|
 * (FourBitEncoding, FindBucket)
 */
static inline uint8_t
FourBitCode(float x)
{
	const float a = fabsf(x);
	uint8_t		bucket = 7;

	while (bucket > 0 && a < hnsw_tq_boundaries[bucket - 1])
		bucket--;
	return bucket | (x > 0 ? 0 : 8);
}

/*
 * The 4+k bit codebook of a bucket: 2^k positive sub-centroids and 2^k - 1
 * boundaries
 */
static void
SubCodebook(int extraBits, int bucket, const float **centroids, const float **boundaries)
{
	switch (extraBits)
	{
		case 1:
			*centroids = hnsw_tq_centroids_4plus1[bucket];
			*boundaries = hnsw_tq_boundaries_4plus1[bucket];
			break;
		case 2:
			*centroids = hnsw_tq_centroids_4plus2[bucket];
			*boundaries = hnsw_tq_boundaries_4plus2[bucket];
			break;
		case 3:
			*centroids = hnsw_tq_centroids_4plus3[bucket];
			*boundaries = hnsw_tq_boundaries_4plus3[bucket];
			break;
		case 4:
			*centroids = hnsw_tq_centroids_4plus4[bucket];
			*boundaries = hnsw_tq_boundaries_4plus4[bucket];
			break;
		case 6:
			*centroids = hnsw_tq_centroids_4plus6[bucket];
			*boundaries = hnsw_tq_boundaries_4plus6[bucket];
			break;
		default:
			*centroids = hnsw_tq_centroids_4plus8[bucket];
			*boundaries = hnsw_tq_boundaries_4plus8[bucket];
			break;
	}
}

/*
 * Index of the first boundary above a, like std::upper_bound
 */
static inline int
UpperBound(const float *boundaries, int n, float a)
{
	int			lo = 0;
	int			hi = n;

	while (lo < hi)
	{
		int			mid = (lo + hi) / 2;

		if (a < boundaries[mid])
			hi = mid;
		else
			lo = mid + 1;
	}
	return lo;
}

/*
 * Encode x: HnswTqCodeBytes(dim) bytes of 4-bit codes, and, if extraBits > 0,
 * HnswTqExtraBytes(dim, extraBits) bytes of extra bits
 * (EncodeDatapointStructured, FourBitEncodeImpl, EncodeDatapoint4PlusKBits)
 *
 * Returns false if ||x|| is not finite. 'scratch' holds
 * HnswTqScratchSize(dim) bytes.
 */
bool
HnswTqEncodePoint(const HnswTqCodec * codec, const float *x, int extraBits,
				  uint8_t *codes, uint8_t *extra, HnswTqPointInfo * info, float *scratch)
{
	const int	dim = codec->dim;
	const size_t extraBytes = HnswTqExtraBytes(dim, extraBits);
	float	   *rotated = scratch;
	float		sqNorm = SquaredNorm(x, dim);
	float		norm;
	float		sq4 = 0.0f;
	float		sq4k = 0.0f;
	int			bit = 0;

	if (!isfinite(sqNorm))
		return false;

	memset(codes, 0, HnswTqCodeBytes(dim));
	if (extraBytes > 0)
		memset(extra, 0, extraBytes);

	if (sqNorm == 0.0f)
	{
		info->sqNorm = 0.0f;
		info->scale4 = 0.0f;
		info->scale4k = 0.0f;
		return true;
	}

	norm = sqrtf(sqNorm);
	FlipSignsNormalizeAndRotate(codec, x, norm, rotated, scratch + dim);

	for (int i = 0; i < dim; i++)
	{
		const uint8_t code = FourBitCode(rotated[i]);
		const int	bucket = code & 7;

		codes[i / 2] |= (i % 2 == 0) ? code : (uint8_t) (code << 4);
		sq4 += hnsw_tq_squared_centroids[bucket];

		if (extraBits > 0)
		{
			const float *centroids;
			const float *boundaries;
			int			sub;
			uint16_t	val;

			SubCodebook(extraBits, bucket, &centroids, &boundaries);
			sub = UpperBound(boundaries, (1 << extraBits) - 1, fabsf(rotated[i]));

			/* LSB first; the padding byte makes the second write safe */
			val = (uint16_t) (sub << (bit % 8));
			extra[bit / 8] |= (uint8_t) val;
			extra[bit / 8 + 1] |= (uint8_t) (val >> 8);
			bit += extraBits;

			sq4k += centroids[sub] * centroids[sub];
		}
	}

	info->sqNorm = sqNorm;
	info->scale4 = (norm / sqrtf(sq4)) * HNSW_TQ_DEBIAS_4BIT;
	info->scale4k = extraBits > 0 ? (norm / sqrtf(sq4k)) * hnsw_tq_debias_4plusk[extraBits - 1] : info->scale4;
	return true;
}

/*
 * Lay out a query in 'mem', which holds HnswTqQuerySize(dim) bytes, aligned
 * to 8 bytes or more
 */
HnswTqQuery *
HnswTqInitQuery(void *mem, int dim)
{
	HnswTqQuery *query = (HnswTqQuery *) mem;
	const int	half = QueryHalf(dim);

	memset(mem, 0, HnswTqQuerySize(dim));
	query->dim = dim;
	query->half = half;
	query->q8 = (int8_t *) ((char *) mem + QueryHeaderSize());
	query->qf = (float *) (query->q8 + 2 * half);
	return query;
}

/*
 * Encode a query: the int8 query for 4-bit scoring, and the float query for
 * 4+k bit scoring (TurboQuantQuery::Create with higher-precision scoring,
 * PrepareQueryForQuantization, EncodeQuery)
 *
 * Returns false if ||q|| is not finite. 'scratch' holds
 * HnswTqScratchSize(dim) bytes.
 */
bool
HnswTqEncodeQuery(const HnswTqCodec * codec, const float *q, HnswTqQuery * query, float *scratch)
{
	const int	dim = codec->dim;
	const int	half = query->half;
	float	   *rotated = scratch;
	float		sqNorm = SquaredNorm(q, dim);
	float		norm;
	float		floatScale;
	float		maxValue = 0.0f;
	float		scaleFactor;
	int			quantNorm = 0;
	int32_t		byteSum = 0;

	if (!isfinite(sqNorm))
		return false;

	memset(query->q8, 0, 2 * (size_t) half);

	/* Ginger rejects a zero query; here every term with it is zero */
	if (sqNorm == 0.0f)
	{
		memset(query->qf, 0, sizeof(float) * dim);
		query->scale = 0.0f;
		query->sqNorm = 0.0f;
		query->byteSum = 0;
		return true;
	}

	norm = sqrtf(sqNorm);
	FlipSignsNormalizeAndRotate(codec, q, norm, rotated, scratch + dim);

	/* The float query, scaled back to ||q|| */
	floatScale = norm / sqrtf((float) dim);
	for (int i = 0; i < dim; i++)
		query->qf[i] = rotated[i] * floatScale;

	for (int i = 0; i < dim; i++)
	{
		if (rotated[i] < -HNSW_TQ_QUERY_VALUE_CAP)
			rotated[i] = -HNSW_TQ_QUERY_VALUE_CAP;
		else if (rotated[i] > HNSW_TQ_QUERY_VALUE_CAP)
			rotated[i] = HNSW_TQ_QUERY_VALUE_CAP;
		if (fabsf(rotated[i]) > maxValue)
			maxValue = fabsf(rotated[i]);
	}
	scaleFactor = 127.0f / maxValue;

	/* Even dimensions first, odd dimensions from 'half' */
	for (int i = 0; i < dim; i++)
	{
		float		v = roundf(rotated[i] * scaleFactor);
		int8_t		snapped;

		if (v < -127.0f)
			v = -127.0f;
		else if (v > 127.0f)
			v = 127.0f;
		snapped = (int8_t) v;
		query->q8[i / 2 + (i % 2) * half] = snapped;
		quantNorm += snapped * snapped;
		byteSum += snapped;
	}

	query->scale = norm / sqrtf((float) quantNorm);
	query->sqNorm = sqNorm;
	query->byteSum = byteSum;
	return true;
}

/*
 * Integer dot product of the int8 query with the int8 centroids of the codes
 * (the loop of TurboQuantScoringInt8ScalarReference)
 */
static int32_t
Dot4Scalar(const HnswTqQuery * query, const uint8_t *codes, int start)
{
	const int8_t *q8 = query->q8;
	const int	half = query->half;
	const int	dim = query->dim;
	int32_t		dot = 0;

	for (int j = start; j < dim / 2; j++)
	{
		dot += (int32_t) hnsw_tq_centroids_int8[codes[j] & 0xF] * q8[j];
		dot += (int32_t) hnsw_tq_centroids_int8[codes[j] >> 4] * q8[half + j];
	}
	if (dim % 2 != 0)
		dot += (int32_t) hnsw_tq_centroids_int8[codes[dim / 2] & 0xF] * q8[dim / 2];
	return dot;
}

#if defined(__AVX2__)
/*
 * The same dot product, 64 dimensions at a time. Shuffles turn codes into
 * signed centroids c. maddubs needs an unsigned operand, so it multiplies |c|
 * by q * sign(c): with |c| and |q| at most 127, its pairwise int16 sums
 * cannot saturate. The result equals Dot4Scalar() exactly.
 */
static int32_t
Dot4Avx2(const HnswTqQuery * query, const uint8_t *codes)
{
	const int8_t *q8 = query->q8;
	const int	half = query->half;
	const int	fullBytes = query->dim / 2;
	const __m256i table = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i *) hnsw_tq_centroids_int8));
	const __m256i nibble = _mm256_set1_epi8(0x0F);
	const __m256i ones = _mm256_set1_epi16(1);
	__m256i		acc = _mm256_setzero_si256();
	__m128i		sum128;
	int			j = 0;

	for (; j + 32 <= fullBytes; j += 32)
	{
		const __m256i packed = _mm256_loadu_si256((const __m256i *) (codes + j));
		const __m256i ceven = _mm256_shuffle_epi8(table, _mm256_and_si256(packed, nibble));
		const __m256i codd = _mm256_shuffle_epi8(table, _mm256_and_si256(_mm256_srli_epi16(packed, 4), nibble));
		const __m256i qeven = _mm256_loadu_si256((const __m256i *) (q8 + j));
		const __m256i qodd = _mm256_loadu_si256((const __m256i *) (q8 + half + j));
		const __m256i peven = _mm256_maddubs_epi16(_mm256_abs_epi8(ceven), _mm256_sign_epi8(qeven, ceven));
		const __m256i podd = _mm256_maddubs_epi16(_mm256_abs_epi8(codd), _mm256_sign_epi8(qodd, codd));

		acc = _mm256_add_epi32(acc, _mm256_madd_epi16(peven, ones));
		acc = _mm256_add_epi32(acc, _mm256_madd_epi16(podd, ones));
	}

	sum128 = _mm_add_epi32(_mm256_castsi256_si128(acc), _mm256_extracti128_si256(acc, 1));
	sum128 = _mm_add_epi32(sum128, _mm_shuffle_epi32(sum128, _MM_SHUFFLE(1, 0, 3, 2)));
	sum128 = _mm_add_epi32(sum128, _mm_shuffle_epi32(sum128, _MM_SHUFFLE(2, 3, 0, 1)));

	return _mm_cvtsi128_si32(sum128) + Dot4Scalar(query, codes, j);
}
#endif

int32_t
HnswTqDot4(const HnswTqQuery * query, const uint8_t *codes)
{
#if defined(__AVX2__)
	return Dot4Avx2(query, codes);
#else
	return Dot4Scalar(query, codes, 0);
#endif
}

/*
 * 4-bit distance estimate (TurboQuantScoringInt8ScalarReference)
 */
float
HnswTqDistance4(const HnswTqQuery * query, const uint8_t *codes, const HnswTqPointInfo * info, HnswTqMetric metric)
{
	const float negDot = -(float) HnswTqDot4(query, codes) * info->scale4 * query->scale;

	if (metric == HNSW_TQ_METRIC_L2)
		return info->sqNorm + 2 * negDot + query->sqNorm;
	return negDot;
}

/*
 * 4+k bit distance estimate, from the float query and the signed 4+k bit
 * centroids (TurboQuantScoringExtraPrecisionScalarReference)
 */
float
HnswTqDistance4k(const HnswTqQuery * query, const uint8_t *codes, const uint8_t *extra,
				 int extraBits, const HnswTqPointInfo * info, HnswTqMetric metric)
{
	const int	dim = query->dim;
	const float *qf = query->qf;
	const int	numSub = 1 << extraBits;
	const uint16_t mask = (uint16_t) (numSub - 1);
	const float *centroids;
	const float *boundaries;
	float		sum = 0.0f;
	float		score;
	int			bit = 0;

	if (extraBits == 0)
		return HnswTqDistance4(query, codes, info, metric);

	/* Bucket b's sub-centroids are centroids[b * numSub, (b + 1) * numSub) */
	SubCodebook(extraBits, 0, &centroids, &boundaries);

	for (int d = 0; d < dim; d++)
	{
		const uint8_t byte = codes[d / 2];
		const uint8_t code = (d % 2 == 0) ? (byte & 0xF) : (byte >> 4);
		uint16_t	twoBytes;
		int			sub;
		float		c;

		/* k bits at bit offset 'bit', which may cross a byte boundary */
		memcpy(&twoBytes, extra + bit / 8, sizeof(twoBytes));
		sub = (twoBytes >> (bit % 8)) & mask;
		bit += extraBits;

		c = centroids[(code & 7) * numSub + sub];
		sum += qf[d] * ((code & 8) ? -c : c);
	}

	score = sum * info->scale4k;
	if (metric == HNSW_TQ_METRIC_L2)
		return info->sqNorm + query->sqNorm - 2 * score;
	return -score;
}
