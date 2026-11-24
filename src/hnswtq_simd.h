/*
 * Fast kernels for the experimental TurboQuant codes (hnswtq.c).
 *
 * Self-contained (stdint and the codebook tables only) so that
 * bench/cloudlab/tq_kernel_opt_bench.c can include and time exactly this code.
 *
 * TqNavDotFast: inner product of a rotated query with a 4-bit code. The
 *   compiler turns the scalar loop into AVX2 gathers, which are slow on
 *   Haswell. Here the 8 positive centroids live in one ymm register and each
 *   nibble selects its centroid with vpermps (which reads only the low 3 bits
 *   of each index); bit 3 of the nibble becomes the float sign bit.
 *
 * TqRefineDotFast: inner product of a rotated query with a nested (4+k)-bit
 *   code. The generic loop re-derived k-bit offsets and chose the table for
 *   every coordinate. For k = 8 (ntq12) the refinement index is a byte; for
 *   k = 6 (ntq10) four indices pack into three bytes. The k-specific loops
 *   look up 8 x 2^k tables (L1-resident), with four accumulators and no
 *   gathers.
 *
 * Results differ from the scalar loops only in float summation order.
 */
#ifndef HNSWTQ_SIMD_H
#define HNSWTQ_SIMD_H

#include <stdint.h>
#include <string.h>

#if defined(__AVX2__) && defined(__FMA__)
#include <immintrin.h>
#define HNSW_TQ_AVX2 1
#endif

/* Sign of nibble n applied to magnitude c, without a branch */
static inline float
TqSigned(float c, unsigned nib)
{
	uint32_t	bits;

	memcpy(&bits, &c, sizeof(bits));
	bits ^= (uint32_t) (nib & 8) << 28;
	memcpy(&c, &bits, sizeof(c));
	return c;
}

#ifdef HNSW_TQ_AVX2
static inline float
TqHsum256(__m256 v)
{
	__m128		lo = _mm256_castps256_ps128(v);
	__m128		hi = _mm256_extractf128_ps(v, 1);

	lo = _mm_add_ps(lo, hi);
	lo = _mm_add_ps(lo, _mm_movehl_ps(lo, lo));
	lo = _mm_add_ss(lo, _mm_movehdup_ps(lo));
	return _mm_cvtss_f32(lo);
}

/* 8 nibbles (coordinates j..j+7) from 4 code bytes, one per 32-bit lane */
static inline __m256i
TqNibbles8(const uint8_t *code4)
{
	uint32_t	w;

	memcpy(&w, code4, sizeof(w));
	return _mm256_srlv_epi32(_mm256_set1_epi32((int) w),
							 _mm256_setr_epi32(0, 4, 8, 12, 16, 20, 24, 28));
}

/* Signed centroids for 8 nibbles: vpermps uses bits 0..2, bit 3 is the sign */
static inline __m256
TqCentroids8(__m256 mag, __m256i nib)
{
	__m256		v = _mm256_permutevar8x32_ps(mag, nib);
	__m256i		sign = _mm256_slli_epi32(_mm256_and_si256(nib, _mm256_set1_epi32(8)), 28);

	return _mm256_xor_ps(v, _mm256_castsi256_ps(sign));
}
#endif

/*
 * Inner product between a rotated query and a 4-bit code. centroids are the 8
 * positive codebook centroids; lut the 16 signed ones (used for the tail).
 */
static inline float
TqNavDotFast(const float *qr, const uint8_t *code, int pdim,
			 const float *centroids, const float *lut)
{
	float		acc = 0;
	int			i = 0;

#ifdef HNSW_TQ_AVX2
	{
		const __m256 mag = _mm256_loadu_ps(centroids);
		__m256		acc0 = _mm256_setzero_ps();
		__m256		acc1 = _mm256_setzero_ps();

		for (; i + 16 <= pdim; i += 16)
		{
			__m256		v0 = TqCentroids8(mag, TqNibbles8(code + i / 2));
			__m256		v1 = TqCentroids8(mag, TqNibbles8(code + i / 2 + 4));

			acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(qr + i), v0, acc0);
			acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(qr + i + 8), v1, acc1);
		}
		for (; i + 8 <= pdim; i += 8)
			acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(qr + i),
								   TqCentroids8(mag, TqNibbles8(code + i / 2)), acc0);
		acc = TqHsum256(_mm256_add_ps(acc0, acc1));
	}
#endif
	for (; i < pdim; i++)
		acc += qr[i] * lut[(code[i >> 1] >> ((i & 1) << 2)) & 0x0F];

	return acc;
}

/*
 * Inner product between a rotated query and a nested (4+k)-bit code, for
 * k = 8 or 6. cells is the [8][2^k] table of positive refined centroids.
 */
static inline float
TqRefineDotFast(const float *qr, const uint8_t *code, const uint8_t *ref,
				int pdim, int k, const float *cells)
{
	float		a0 = 0,
				a1 = 0,
				a2 = 0,
				a3 = 0;
	int			i = 0;

	if (k == 8)
	{
		for (; i + 4 <= pdim; i += 4)
		{
			unsigned	b0 = code[i >> 1];
			unsigned	b1 = code[(i >> 1) + 1];
			unsigned	n0 = b0 & 15,
						n1 = b0 >> 4,
						n2 = b1 & 15,
						n3 = b1 >> 4;

			a0 += qr[i] * TqSigned(cells[((n0 & 7) << 8) | ref[i]], n0);
			a1 += qr[i + 1] * TqSigned(cells[((n1 & 7) << 8) | ref[i + 1]], n1);
			a2 += qr[i + 2] * TqSigned(cells[((n2 & 7) << 8) | ref[i + 2]], n2);
			a3 += qr[i + 3] * TqSigned(cells[((n3 & 7) << 8) | ref[i + 3]], n3);
		}
		for (; i < pdim; i++)
		{
			unsigned	n = (code[i >> 1] >> ((i & 1) << 2)) & 15;

			a0 += qr[i] * TqSigned(cells[((n & 7) << 8) | ref[i]], n);
		}
	}
	else
	{
		/* k = 6: indices 4j..4j+3 are the 24 bits of ref[3j..3j+2] */
		for (; i + 4 <= pdim; i += 4)
		{
			const uint8_t *r = ref + (i / 4) * 3;
			uint32_t	w = (uint32_t) r[0] | ((uint32_t) r[1] << 8) | ((uint32_t) r[2] << 16);
			unsigned	b0 = code[i >> 1];
			unsigned	b1 = code[(i >> 1) + 1];
			unsigned	n0 = b0 & 15,
						n1 = b0 >> 4,
						n2 = b1 & 15,
						n3 = b1 >> 4;

			a0 += qr[i] * TqSigned(cells[((n0 & 7) << 6) | (w & 63)], n0);
			a1 += qr[i + 1] * TqSigned(cells[((n1 & 7) << 6) | ((w >> 6) & 63)], n1);
			a2 += qr[i + 2] * TqSigned(cells[((n2 & 7) << 6) | ((w >> 12) & 63)], n2);
			a3 += qr[i + 3] * TqSigned(cells[((n3 & 7) << 6) | ((w >> 18) & 63)], n3);
		}
		for (; i < pdim; i++)
		{
			unsigned	n = (code[i >> 1] >> ((i & 1) << 2)) & 15;
			uint32_t	off = (uint32_t) i * 6;
			uint32_t	w = (uint32_t) ref[off >> 3] | ((uint32_t) ref[(off >> 3) + 1] << 8);

			a0 += qr[i] * TqSigned(cells[((n & 7) << 6) | ((w >> (off & 7)) & 63)], n);
		}
	}

	return (a0 + a1) + (a2 + a3);
}

#endif							/* HNSWTQ_SIMD_H */
