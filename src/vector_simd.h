#ifndef VECTOR_SIMD_H
#define VECTOR_SIMD_H

#include "halfvec.h"
#include "vector.h"

#if (defined(__GNUC__) || defined(__clang__)) && defined(__AVX__) && defined(__FMA__)
#define VECTOR_NATIVE_AVX_FMA 1
#endif

#if defined(VECTOR_NATIVE_AVX_FMA) || defined(USE_DISPATCH)
#include <immintrin.h>
#endif

#if !defined(VECTOR_NATIVE_AVX_FMA) && defined(USE_DISPATCH)
extern float (*VectorL2SquaredDistancePtr) (int dim, const float *ax, const float *bx);
extern float (*VectorInnerProductPtr) (int dim, const float *ax, const float *bx);
extern double (*VectorCosineSimilarityPtr) (int dim, const float *ax, const float *bx);
#endif

extern void VectorInit(void);

#define VECTOR_L2_SQUARED_AVX_FMA_IMPL(dim, ax, bx) \
	do { \
		int			i = 0; \
		float		distance; \
		__m256		sum0 = _mm256_setzero_ps(); \
		__m256		sum1 = _mm256_setzero_ps(); \
		__m256		sum2 = _mm256_setzero_ps(); \
		__m256		sum3 = _mm256_setzero_ps(); \
		__m128		hi; \
		__m128		lo; \
		for (; i + 32 <= (dim); i += 32) \
		{ \
			__m256		d0 = _mm256_sub_ps(_mm256_loadu_ps((ax) + i), _mm256_loadu_ps((bx) + i)); \
			__m256		d1 = _mm256_sub_ps(_mm256_loadu_ps((ax) + i + 8), _mm256_loadu_ps((bx) + i + 8)); \
			__m256		d2 = _mm256_sub_ps(_mm256_loadu_ps((ax) + i + 16), _mm256_loadu_ps((bx) + i + 16)); \
			__m256		d3 = _mm256_sub_ps(_mm256_loadu_ps((ax) + i + 24), _mm256_loadu_ps((bx) + i + 24)); \
			sum0 = _mm256_fmadd_ps(d0, d0, sum0); \
			sum1 = _mm256_fmadd_ps(d1, d1, sum1); \
			sum2 = _mm256_fmadd_ps(d2, d2, sum2); \
			sum3 = _mm256_fmadd_ps(d3, d3, sum3); \
		} \
		for (; i + 8 <= (dim); i += 8) \
		{ \
			__m256		d = _mm256_sub_ps(_mm256_loadu_ps((ax) + i), _mm256_loadu_ps((bx) + i)); \
			sum0 = _mm256_fmadd_ps(d, d, sum0); \
		} \
		sum0 = _mm256_add_ps(_mm256_add_ps(sum0, sum1), _mm256_add_ps(sum2, sum3)); \
		hi = _mm256_extractf128_ps(sum0, 1); \
		lo = _mm256_castps256_ps128(sum0); \
		lo = _mm_add_ps(lo, hi); \
		hi = _mm_movehl_ps(lo, lo); \
		lo = _mm_add_ps(lo, hi); \
		lo = _mm_add_ss(lo, _mm_shuffle_ps(lo, lo, 0x55)); \
		distance = _mm_cvtss_f32(lo); \
		for (; i < (dim); i++) \
		{ \
			float		diff = (ax)[i] - (bx)[i]; \
			distance += diff * diff; \
		} \
		return distance; \
	} while (0)

#define VECTOR_INNER_PRODUCT_AVX_FMA_IMPL(dim, ax, bx) \
	do { \
		int			i = 0; \
		float		distance; \
		__m256		sum0 = _mm256_setzero_ps(); \
		__m256		sum1 = _mm256_setzero_ps(); \
		__m256		sum2 = _mm256_setzero_ps(); \
		__m256		sum3 = _mm256_setzero_ps(); \
		__m128		hi; \
		__m128		lo; \
		for (; i + 32 <= (dim); i += 32) \
		{ \
			sum0 = _mm256_fmadd_ps(_mm256_loadu_ps((ax) + i), _mm256_loadu_ps((bx) + i), sum0); \
			sum1 = _mm256_fmadd_ps(_mm256_loadu_ps((ax) + i + 8), _mm256_loadu_ps((bx) + i + 8), sum1); \
			sum2 = _mm256_fmadd_ps(_mm256_loadu_ps((ax) + i + 16), _mm256_loadu_ps((bx) + i + 16), sum2); \
			sum3 = _mm256_fmadd_ps(_mm256_loadu_ps((ax) + i + 24), _mm256_loadu_ps((bx) + i + 24), sum3); \
		} \
		for (; i + 8 <= (dim); i += 8) \
			sum0 = _mm256_fmadd_ps(_mm256_loadu_ps((ax) + i), _mm256_loadu_ps((bx) + i), sum0); \
		sum0 = _mm256_add_ps(_mm256_add_ps(sum0, sum1), _mm256_add_ps(sum2, sum3)); \
		hi = _mm256_extractf128_ps(sum0, 1); \
		lo = _mm256_castps256_ps128(sum0); \
		lo = _mm_add_ps(lo, hi); \
		hi = _mm_movehl_ps(lo, lo); \
		lo = _mm_add_ps(lo, hi); \
		lo = _mm_add_ss(lo, _mm_shuffle_ps(lo, lo, 0x55)); \
		distance = _mm_cvtss_f32(lo); \
		for (; i < (dim); i++) \
			distance += (ax)[i] * (bx)[i]; \
		return distance; \
	} while (0)

#define VECTOR_COSINE_AVX_FMA_IMPL(dim, ax, bx) \
	do { \
		int			i = 0; \
		float		similarity; \
		float		norma; \
		float		normb; \
		__m256		sim0 = _mm256_setzero_ps(); \
		__m256		sim1 = _mm256_setzero_ps(); \
		__m256		na0 = _mm256_setzero_ps(); \
		__m256		na1 = _mm256_setzero_ps(); \
		__m256		nb0 = _mm256_setzero_ps(); \
		__m256		nb1 = _mm256_setzero_ps(); \
		__m128		lo; \
		__m128		hi; \
		for (; i + 16 <= (dim); i += 16) \
		{ \
			__m256		a0 = _mm256_loadu_ps((ax) + i); \
			__m256		b0 = _mm256_loadu_ps((bx) + i); \
			__m256		a1 = _mm256_loadu_ps((ax) + i + 8); \
			__m256		b1 = _mm256_loadu_ps((bx) + i + 8); \
			sim0 = _mm256_fmadd_ps(a0, b0, sim0); \
			sim1 = _mm256_fmadd_ps(a1, b1, sim1); \
			na0 = _mm256_fmadd_ps(a0, a0, na0); \
			na1 = _mm256_fmadd_ps(a1, a1, na1); \
			nb0 = _mm256_fmadd_ps(b0, b0, nb0); \
			nb1 = _mm256_fmadd_ps(b1, b1, nb1); \
		} \
		for (; i + 8 <= (dim); i += 8) \
		{ \
			__m256		a0 = _mm256_loadu_ps((ax) + i); \
			__m256		b0 = _mm256_loadu_ps((bx) + i); \
			sim0 = _mm256_fmadd_ps(a0, b0, sim0); \
			na0 = _mm256_fmadd_ps(a0, a0, na0); \
			nb0 = _mm256_fmadd_ps(b0, b0, nb0); \
		} \
		sim0 = _mm256_add_ps(sim0, sim1); \
		na0 = _mm256_add_ps(na0, na1); \
		nb0 = _mm256_add_ps(nb0, nb1); \
		hi = _mm256_extractf128_ps(sim0, 1); lo = _mm256_castps256_ps128(sim0); lo = _mm_add_ps(lo, hi); hi = _mm_movehl_ps(lo, lo); lo = _mm_add_ps(lo, hi); lo = _mm_add_ss(lo, _mm_shuffle_ps(lo, lo, 0x55)); similarity = _mm_cvtss_f32(lo); \
		hi = _mm256_extractf128_ps(na0, 1); lo = _mm256_castps256_ps128(na0); lo = _mm_add_ps(lo, hi); hi = _mm_movehl_ps(lo, lo); lo = _mm_add_ps(lo, hi); lo = _mm_add_ss(lo, _mm_shuffle_ps(lo, lo, 0x55)); norma = _mm_cvtss_f32(lo); \
		hi = _mm256_extractf128_ps(nb0, 1); lo = _mm256_castps256_ps128(nb0); lo = _mm_add_ps(lo, hi); hi = _mm_movehl_ps(lo, lo); lo = _mm_add_ps(lo, hi); lo = _mm_add_ss(lo, _mm_shuffle_ps(lo, lo, 0x55)); normb = _mm_cvtss_f32(lo); \
		for (; i < (dim); i++) \
		{ \
			similarity += (ax)[i] * (bx)[i]; \
			norma += (ax)[i] * (ax)[i]; \
			normb += (bx)[i] * (bx)[i]; \
		} \
		return (double) similarity / sqrt((double) norma * (double) normb); \
	} while (0)

static inline float
VectorL2SquaredDistanceScalar(int dim, const float *ax, const float *bx)
{
	float		s0 = 0.0f;
	float		s1 = 0.0f;
	float		s2 = 0.0f;
	float		s3 = 0.0f;
	int			i = 0;

	/* Auto-vectorized with 4 independent accumulators to hide FMA latency */
	for (; i + 4 <= dim; i += 4)
	{
		float		d0 = ax[i] - bx[i];
		float		d1 = ax[i + 1] - bx[i + 1];
		float		d2 = ax[i + 2] - bx[i + 2];
		float		d3 = ax[i + 3] - bx[i + 3];

		s0 += d0 * d0;
		s1 += d1 * d1;
		s2 += d2 * d2;
		s3 += d3 * d3;
	}
	for (; i < dim; i++)
	{
		float		d = ax[i] - bx[i];

		s0 += d * d;
	}
	return (s0 + s1) + (s2 + s3);
}

static inline float
VectorL2SquaredDistance(int dim, const float *ax, const float *bx)
{
#ifdef VECTOR_NATIVE_AVX_FMA
	VECTOR_L2_SQUARED_AVX_FMA_IMPL(dim, ax, bx);
#elif defined(USE_DISPATCH)
	return VectorL2SquaredDistancePtr(dim, ax, bx);
#else
	return VectorL2SquaredDistanceScalar(dim, ax, bx);
#endif
}

static inline float
VectorInnerProduct(int dim, const float *ax, const float *bx)
{
#ifdef VECTOR_NATIVE_AVX_FMA
	VECTOR_INNER_PRODUCT_AVX_FMA_IMPL(dim, ax, bx);
#elif defined(USE_DISPATCH)
	return VectorInnerProductPtr(dim, ax, bx);
#else
	float		s0 = 0.0f;
	float		s1 = 0.0f;
	float		s2 = 0.0f;
	float		s3 = 0.0f;
	int			i = 0;

	for (; i + 4 <= dim; i += 4)
	{
		s0 += ax[i] * bx[i];
		s1 += ax[i + 1] * bx[i + 1];
		s2 += ax[i + 2] * bx[i + 2];
		s3 += ax[i + 3] * bx[i + 3];
	}
	for (; i < dim; i++)
		s0 += ax[i] * bx[i];

	return (s0 + s1) + (s2 + s3);
#endif
}

static inline double
VectorCosineSimilarity(int dim, const float *ax, const float *bx)
{
#ifdef VECTOR_NATIVE_AVX_FMA
	VECTOR_COSINE_AVX_FMA_IMPL(dim, ax, bx);
#elif defined(USE_DISPATCH)
	return VectorCosineSimilarityPtr(dim, ax, bx);
#else
	float		similarity = 0.0f;
	float		norma = 0.0f;
	float		normb = 0.0f;

	for (int i = 0; i < dim; i++)
	{
		similarity += ax[i] * bx[i];
		norma += ax[i] * ax[i];
		normb += bx[i] * bx[i];
	}

	return (double) similarity / sqrt((double) norma * (double) normb);
#endif
}

#endif
