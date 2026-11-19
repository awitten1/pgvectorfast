/* Candidate storage allocation and radix sorting for iterative search. */
#include "postgres.h"

#include "hnswqueue.h"

HnswIterativeState *
HnswInitIterativeState(int ef)
{
	HnswIterativeState *state;

	state = (HnswIterativeState *) palloc0(sizeof(*state));
	state->pending_capacity = state->candidate_capacity = Max(ef, 256);
	state->pending = palloc_array_checked(HnswIterativeKey, state->pending_capacity);
	state->candidates = palloc_array_checked(HnswIterativeKey, state->candidate_capacity);
	state->result_capacity = ef;
	state->results = palloc_array_checked(HnswIterativeKey, ef);
	state->chunk_capacity = 16;
	state->chunks = palloc_array_checked(HnswIterativeNode *, state->chunk_capacity);
	state->chunks[0] = palloc_array_checked(HnswIterativeNode, HNSW_ITER_CHUNK_SIZE);
	state->chunk_count = 1;
	state->extra_capacity = 64;
	state->extra_htids = palloc_array_checked(ItemPointerData, state->extra_capacity);

	return state;
}

static pg_attribute_always_inline uint32
HnswFloatDescRadixKey(float d)
{
	uint32		u;

	memcpy(&u, &d, sizeof(uint32));
	if (unlikely((u & 0x7fffffffU) > 0x7f800000U))
		return 0U;
	return ((int32) u < 0) ? u : (u ^ 0x7fffffffU);
}

pg_noinline void
HnswSortPendingDesc(HnswIterativeState *state)
{
	const int	n = state->pending_count;
	uint32		hist[4][256];
	HnswIterativeKey *tmp;
	HnswIterativeKey *src;
	HnswIterativeKey *dst;

	if (n <= 1)
	{
		state->pending_sorted = true;
		return;
	}

	memset(hist, 0, sizeof(hist));
	src = state->pending;
	for (int i = 0; i < n; i++)
	{
		uint32		k = HnswFloatDescRadixKey(src[i].distance);

		hist[0][k & 0xffU]++;
		hist[1][(k >> 8) & 0xffU]++;
		hist[2][(k >> 16) & 0xffU]++;
		hist[3][k >> 24]++;
	}

	tmp = (HnswIterativeKey *) palloc(mul_size(sizeof(HnswIterativeKey), (Size) n));
	dst = tmp;

	for (int pass = 0; pass < 4; pass++)
	{
		uint32	   *h = hist[pass];
		uint32		sum = 0;
		bool		trivial = false;
		const int	shift = pass * 8;

		for (int b = 0; b < 256; b++)
		{
			uint32		c = h[b];

			if (c == (uint32) n)
			{
				trivial = true;
				break;
			}
			h[b] = sum;
			sum += c;
		}
		if (trivial)
			continue;

		for (int i = 0; i < n; i++)
		{
			uint32		k = HnswFloatDescRadixKey(src[i].distance);
			uint32		b = (k >> shift) & 0xffU;

			dst[h[b]++] = src[i];
		}
		{
			HnswIterativeKey *swap = src;

			src = dst;
			dst = swap;
		}
	}

	if (src != state->pending)
		memcpy(state->pending, src, mul_size(sizeof(HnswIterativeKey), (Size) n));
	pfree(tmp);
	state->pending_sorted = true;
}
