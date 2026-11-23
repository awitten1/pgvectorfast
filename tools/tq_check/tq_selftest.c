/*
 * Self-test of src/hnswtq.c that needs no Ginger, for any architecture:
 *
 * - HnswTqDot4() (the AVX2 kernel when built with -mavx2) against a plain
 *   scalar dot product, on random codes and queries, and on the extreme
 *   values where an int16 sum could saturate;
 * - a checksum of codes, extra bits and distances on fixed random inputs, to
 *   compare builds: the output should be identical on every architecture.
 *
 *   cc -O2 -ffp-contract=off -Isrc src/hnswtq.c tools/tq_check/tq_selftest.c -lm -o tq_selftest
 *   cc -O2 -ffp-contract=off -mavx2 -mfma -Isrc ...   (x86-64, AVX2 kernel)
 */
#include <inttypes.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "hnswtq.h"

static const int8_t centroids[16] = {6, 18, 31, 44, 58, 75, 96, 127, -6, -18, -31, -44, -58, -75, -96, -127};

static int	failures = 0;

/* xorshift64*, for inputs that are the same everywhere */
static uint64_t rng_state = 88172645463325252ULL;

static uint64_t
NextRandom(void)
{
	rng_state ^= rng_state >> 12;
	rng_state ^= rng_state << 25;
	rng_state ^= rng_state >> 27;
	return rng_state * 2685821657736338717ULL;
}

/* Approximately standard normal, from 12 uniforms */
static float
RandomNormal(void)
{
	double		sum = 0.0;

	for (int i = 0; i < 12; i++)
		sum += (double) (NextRandom() >> 11) / 9007199254740992.0;
	return (float) (sum - 6.0);
}

static int32_t
ReferenceDot(const HnswTqQuery *q, const uint8_t *codes)
{
	int32_t		dot = 0;

	for (int d = 0; d < q->dim; d++)
	{
		const int	code = (d % 2 == 0) ? (codes[d / 2] & 0xF) : (codes[d / 2] >> 4);

		dot += centroids[code] * q->q8[d / 2 + (d % 2) * q->half];
	}
	return dot;
}

static void
CheckDot(const HnswTqQuery *q, const uint8_t *codes, const char *what)
{
	const int32_t expected = ReferenceDot(q, codes);
	const int32_t actual = HnswTqDot4(q, codes);

	if (expected != actual)
	{
		failures++;
		printf("FAIL d=%d %s: dot %" PRId32 ", expected %" PRId32 "\n", q->dim, what, actual, expected);
	}
}

/* FNV-1a over bytes */
static uint64_t
Hash(uint64_t h, const void *data, size_t len)
{
	const unsigned char *p = data;

	for (size_t i = 0; i < len; i++)
		h = (h ^ p[i]) * 1099511628211ULL;
	return h;
}

int
main(void)
{
	static const int dims[] = {1, 2, 3, 7, 16, 31, 62, 64, 96, 100, 120, 124, 128, 200, 384, 768, 960, 1024, 1536, 1920};
	static const int unsupported[] = {0, 33, 63, 2000};
	uint64_t	checksum = 14695981039346656037ULL;

	/* Dimensions needing more than HNSW_TQ_MAX_SUBSPACES blocks are rejected */
	for (size_t i = 0; i < sizeof(unsupported) / sizeof(unsupported[0]); i++)
	{
		HnswTqCodec *codec = malloc(HnswTqCodecSize(2000));

		if (HnswTqInitCodec(codec, unsupported[i], HNSW_TQ_DEFAULT_SEED, NULL))
		{
			failures++;
			printf("FAIL d=%d: accepted\n", unsupported[i]);
		}
		free(codec);
	}

	for (size_t di = 0; di < sizeof(dims) / sizeof(dims[0]); di++)
	{
		const int	dim = dims[di];
		const size_t codeBytes = HnswTqCodeBytes(dim);
		HnswTqCodec *codec = malloc(HnswTqCodecSize(dim));
		float	   *scratch = malloc(HnswTqScratchSize(dim));
		double	   *qmem = malloc(HnswTqQuerySize(dim) + 8);
		HnswTqQuery *q = HnswTqInitQuery(qmem, dim);
		uint8_t    *codes = malloc(codeBytes);
		uint8_t    *extra = malloc(HnswTqExtraBytes(dim, 8));
		float	   *x = malloc(sizeof(float) * dim);

		if (!HnswTqInitCodec(codec, dim, HNSW_TQ_DEFAULT_SEED, NULL))
		{
			failures++;
			printf("FAIL d=%d: unsupported\n", dim);
			continue;
		}

		/* Random queries against random code bytes */
		for (int trial = 0; trial < 50; trial++)
		{
			for (int i = 0; i < dim; i++)
				x[i] = RandomNormal() * (1 + trial);
			HnswTqEncodeQuery(codec, x, q, scratch);
			for (size_t i = 0; i < codeBytes; i++)
				codes[i] = (uint8_t) NextRandom();
			if (dim % 2 != 0)
				codes[codeBytes - 1] &= 0x0F;
			CheckDot(q, codes, "random");
		}

		/* Extremes: |q8| = 127 against centroids of +-127, same and opposite signs */
		for (int sign = 0; sign < 4; sign++)
		{
			for (int i = 0; i < 2 * q->half; i++)
				q->q8[i] = 0;
			for (int d = 0; d < dim; d++)
				q->q8[d / 2 + (d % 2) * q->half] = (sign & 1) ? -127 : 127;
			memset(codes, (sign & 2) ? 0xFF : 0x77, codeBytes);
			if (dim % 2 != 0)
				codes[codeBytes - 1] &= 0x0F;
			CheckDot(q, codes, "extreme");
		}

		/* Checksum of the codec's outputs on fixed inputs */
		checksum = Hash(checksum, codec->rotation, sizeof(float) * codec->subspaces * codec->subspaces);
		checksum = Hash(checksum, codec->signs, sizeof(float) * dim);
		for (int i = 0; i < dim; i++)
			x[i] = RandomNormal();
		HnswTqEncodeQuery(codec, x, q, scratch);
		checksum = Hash(checksum, q->q8, 2 * (size_t) q->half);
		checksum = Hash(checksum, q->qf, sizeof(float) * dim);
		for (int k = 0; k <= 8; k++)
		{
			HnswTqPointInfo info;
			float		d4;
			float		d4k;

			if (!HnswTqValidExtraBits(k))
				continue;
			for (int i = 0; i < dim; i++)
				x[i] = RandomNormal() + 0.25f;
			HnswTqEncodePoint(codec, x, k, codes, extra, &info, scratch);
			d4 = HnswTqDistance4(q, codes, &info, HNSW_TQ_METRIC_L2);
			d4k = HnswTqDistance4k(q, codes, extra, k, &info, HNSW_TQ_METRIC_DOT);
			checksum = Hash(checksum, codes, codeBytes);
			checksum = Hash(checksum, extra, HnswTqExtraBytes(dim, k));
			checksum = Hash(checksum, &info, sizeof(info));
			checksum = Hash(checksum, &d4, sizeof(d4));
			checksum = Hash(checksum, &d4k, sizeof(d4k));
		}

		free(codec);
		free(scratch);
		free(qmem);
		free(codes);
		free(extra);
		free(x);
	}

#if defined(__AVX2__)
	printf("kernel: AVX2\n");
#else
	printf("kernel: scalar\n");
#endif
	printf("checksum: %016" PRIx64 "\n", checksum);
	printf(failures == 0 ? "OK\n" : "%d checks failed\n", failures);
	return failures == 0 ? 0 : 1;
}
