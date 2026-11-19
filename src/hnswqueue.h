/* Candidate heaps and the iterative node arena. */
#ifndef HNSWQUEUE_H
#define HNSWQUEUE_H

#include "hnsw.h"
#include "utils/memutils.h"

#define HNSW_ITER_CHUNK_BITS 10
#define HNSW_ITER_CHUNK_SIZE (1U << HNSW_ITER_CHUNK_BITS)
#define HNSW_ITER_CHUNK_MASK (HNSW_ITER_CHUNK_SIZE - 1U)
#define HNSW_ITER_HEAP_PREFETCH_MAX 64
#define HNSW_ITER_DRAIN_BATCH 256

/*
 * Compact value candidate for fast contiguous binary-heap search.
 * Eliminates pairingheap pointers, per-candidate HnswElement allocations,
 * and eager heaptids[10] copies during traversal.
 */
typedef struct HnswFastCand
{
	double		distance;
	BlockNumber blkno;
	BlockNumber neighborPage;
	OffsetNumber offno;
	OffsetNumber neighborOffno;
	uint8		level;
	uint8		version;
}			HnswFastCand;

static pg_attribute_always_inline void
HnswFastHeapSiftUpMin(HnswFastCand *heap, int idx)
{
	HnswFastCand item = heap[idx];

	while (idx > 0)
	{
		int			parent = (idx - 1) >> 1;

		if (item.distance >= heap[parent].distance)
			break;
		heap[idx] = heap[parent];
		idx = parent;
	}
	heap[idx] = item;
}

static pg_attribute_always_inline void
HnswFastHeapSiftDownMin(HnswFastCand *heap, int n, int idx)
{
	HnswFastCand item = heap[idx];
	int			half = n >> 1;

	while (idx < half)
	{
		int			child = (idx << 1) + 1;
		int			right = child + 1;

		if (right < n && heap[right].distance < heap[child].distance)
			child = right;
		if (item.distance <= heap[child].distance)
			break;
		heap[idx] = heap[child];
		idx = child;
	}
	heap[idx] = item;
}

static pg_attribute_always_inline void
HnswFastHeapSiftUpMax(HnswFastResultCand *heap, int idx)
{
	HnswFastResultCand item = heap[idx];

	while (idx > 0)
	{
		int			parent = (idx - 1) >> 1;

		if (item.distance <= heap[parent].distance)
			break;
		heap[idx] = heap[parent];
		idx = parent;
	}
	heap[idx] = item;
}

static pg_attribute_always_inline void
HnswFastHeapSiftDownMax(HnswFastResultCand *heap, int n, int idx)
{
	HnswFastResultCand item = heap[idx];
	int			half = n >> 1;

	while (idx < half)
	{
		int			child = (idx << 1) + 1;
		int			right = child + 1;

		if (right < n && heap[right].distance > heap[child].distance)
			child = right;
		if (item.distance >= heap[child].distance)
			break;
		heap[idx] = heap[child];
		idx = child;
	}
	heap[idx] = item;
}

static pg_attribute_always_inline void
HnswFastPushCand(const HnswFastCand *cand,
				 HnswFastCand **c_heap_ptr, int *clen, int *ccap, bool *c_on_stack)
{
	if (unlikely(*clen == *ccap))
	{
		int			newcap = (*ccap) * 2;

		if (*c_on_stack)
		{
			HnswFastCand *nc = palloc_array_checked(HnswFastCand, newcap);

			memcpy(nc, *c_heap_ptr, sizeof(HnswFastCand) * (Size) (*clen));
			*c_heap_ptr = nc;
			*c_on_stack = false;
		}
		else
			*c_heap_ptr = (HnswFastCand *) repalloc(*c_heap_ptr,
													sizeof(HnswFastCand) * (Size) newcap);
		*ccap = newcap;
	}

	(*c_heap_ptr)[*clen] = *cand;
	HnswFastHeapSiftUpMin(*c_heap_ptr, (*clen)++);
}

static pg_attribute_always_inline void
HnswFastCommitCand(const HnswFastCand *cand, int ef, int lc,
				   HnswFastCand **c_heap_ptr, int *clen, int *ccap, bool *c_on_stack,
				   HnswFastResultCand *w_heap, int *wlen)
{
	if (unlikely(cand->level < lc))
		return;

	if (*wlen >= ef && cand->distance >= w_heap[0].distance)
		return;

	HnswFastPushCand(cand, c_heap_ptr, clen, ccap, c_on_stack);

	if (*wlen < ef)
	{
		int			pos = (*wlen)++;

		w_heap[pos].distance = cand->distance;
		w_heap[pos].blkno = cand->blkno;
		w_heap[pos].offno = cand->offno;
		w_heap[pos]._pad = 0;
		HnswFastHeapSiftUpMax(w_heap, pos);
	}
	else
	{
		w_heap[0].distance = cand->distance;
		w_heap[0].blkno = cand->blkno;
		w_heap[0].offno = cand->offno;
		w_heap[0]._pad = 0;
		HnswFastHeapSiftDownMax(w_heap, *wlen, 0);
	}
}

typedef struct HnswIterativeKey
{
	float		distance;
	uint32		id;
}			HnswIterativeKey;

typedef struct HnswIterativeNode
{
	BlockNumber neighborPage;
	uint32		htid_meta;		/* (heap_count << 24) | extra_off */
	OffsetNumber neighborOffno;
	ItemPointerData first_htid;
	uint8		level;
	uint8		version;
}			HnswIterativeNode;

struct HnswIterativeState
{
	HnswIterativeKey *pending;
	HnswIterativeKey *candidates;
	HnswIterativeKey *results;
	int			pending_count;
	int			pending_capacity;
	int			candidate_capacity;
	int			result_count;
	int			result_capacity;
	int			result_idx;
	int			pref_start;
	int			pref_end;
	uint8		cur_rem_htids;
	bool		oom_mode;
	uint8		oom_hit_streak;
	bool		pending_sorted;
	uint8		drain_batches;
	HnswIterativeNode **chunks;
	uint32		node_count;
	uint32		chunk_count;
	uint32		chunk_capacity;
	ItemPointerData *extra_htids;
	uint32		extra_count;
	uint32		extra_capacity;
	uint32		pref_htid_meta[HNSW_ITER_HEAP_PREFETCH_MAX];
	ItemPointerData pref_first_htids[HNSW_ITER_HEAP_PREFETCH_MAX];
	OffsetNumber pref_offs[HNSW_ITER_HEAP_PREFETCH_MAX];
	CalicoOptimisticSlot pref_slots[HNSW_ITER_HEAP_PREFETCH_MAX];
};

static inline const HnswIterativeNode *
HnswIterativeGetNode(const HnswIterativeState *state, uint32 id)
{
	return &state->chunks[id >> HNSW_ITER_CHUNK_BITS][id & HNSW_ITER_CHUNK_MASK];
}

static inline bool
HnswIterativeEmpty(const HnswIterativeState *state)
{
	return state->pending_count == 0;
}

/*
 * Compact 8-byte key queues and chunked 20-byte node arena live in the scan
 * temporary context and retain discarded candidates across iterative batches.
 * Heap TIDs are copied during the validated vector read so returning rows
 * never rereads index tuples or allocates per-candidate search objects.
 */
StaticAssertDecl(sizeof(HnswIterativeKey) == 8, "HnswIterativeKey must be 8 bytes");
StaticAssertDecl(sizeof(HnswIterativeNode) == 20, "HnswIterativeNode must be 20 bytes");

static pg_attribute_always_inline HnswIterativeKey
HnswIterativeRememberRaw(HnswIterativeState *state,
						 BlockNumber neighborPage, OffsetNumber neighborOffno,
						 uint8 level, uint8 version,
						 const ItemPointerData *htids, uint8 n_htids,
						 double distance)
{
	uint32		id = state->node_count;
	uint32		chunk_idx = id >> HNSW_ITER_CHUNK_BITS;
	HnswIterativeNode *node;
	HnswIterativeKey key;

	if (unlikely(chunk_idx >= state->chunk_count))
	{
		if (unlikely(state->chunk_count == state->chunk_capacity))
		{
			uint32		new_cap = state->chunk_capacity * 2U;

			state->chunks = (HnswIterativeNode **) repalloc(state->chunks,
															mul_size(sizeof(HnswIterativeNode *), new_cap));
			state->chunk_capacity = new_cap;
		}
		state->chunks[state->chunk_count++] =
			(HnswIterativeNode *) palloc_array_checked(HnswIterativeNode, HNSW_ITER_CHUNK_SIZE);
	}

	node = &state->chunks[chunk_idx][id & HNSW_ITER_CHUNK_MASK];
	node->neighborPage = neighborPage;
	node->neighborOffno = neighborOffno;
	node->level = level;
	node->version = version;

	if (likely(n_htids <= 1))
	{
		node->htid_meta = ((uint32) n_htids << 24);
		if (likely(n_htids == 1))
			node->first_htid = htids[0];
		else
			ItemPointerSetInvalid(&node->first_htid);
	}
	else
	{
		uint32		extra_needed = (uint32) n_htids - 1U;
		uint32		extra_off = state->extra_count;

		if (unlikely(extra_off + extra_needed > state->extra_capacity))
		{
			Size		new_cap = Max((Size) state->extra_capacity * 2,
									  (Size) extra_off + extra_needed);

			state->extra_htids = (ItemPointerData *) repalloc(state->extra_htids,
															  mul_size(sizeof(ItemPointerData), new_cap));
			state->extra_capacity = (uint32) new_cap;
		}
		if (unlikely(extra_off > 0x00ffffffU))
			elog(ERROR, "hnsw iterative heap TID arena overflow");

		node->first_htid = htids[0];
		memcpy(state->extra_htids + extra_off, htids + 1,
			   sizeof(ItemPointerData) * extra_needed);
		node->htid_meta = ((uint32) n_htids << 24) | extra_off;
		state->extra_count = extra_off + extra_needed;
	}

	state->node_count = id + 1U;
	key.distance = (float) distance;
	key.id = id;
	return key;
}

static inline HnswIterativeKey
HnswIterativeRememberElement(HnswIterativeState *state, HnswElement el, double distance)
{
	return HnswIterativeRememberRaw(state,
									el->neighborPage, el->neighborOffno,
									el->level, el->version,
									el->heaptids, (uint8) el->heaptidsLength,
									distance);
}

static inline void
HnswIterativePushMin4(HnswIterativeKey key, HnswIterativeKey **heap,
					  int *count, int *capacity)
{
	int			pos;
	HnswIterativeKey *h;

	if (unlikely(*count == *capacity))
	{
		Size		new_capacity = (*capacity >= 256) ? (Size) (*capacity) * 4 : (Size) (*capacity) * 2;

		*heap = (HnswIterativeKey *) repalloc(*heap, mul_size(sizeof(HnswIterativeKey), new_capacity));
		*capacity = (int) new_capacity;
	}
	h = *heap;
	pos = (*count)++;
	while (pos > 0)
	{
		int			parent = (pos - 1) >> 2;

		if (key.distance >= h[parent].distance)
			break;
		h[pos] = h[parent];
		pos = parent;
	}
	h[pos] = key;
}

static inline void
HnswIterativeSiftDownMin4(HnswIterativeKey *heap, int count, int pos)
{
	HnswIterativeKey key = heap[pos];

	for (;;)
	{
		int			first = (pos << 2) + 1;
		int			best;
		float		best_d;

		if (first >= count)
			break;

		if (first + 3 < count)
		{
			float		d0 = heap[first].distance;
			float		d1 = heap[first + 1].distance;
			float		d2 = heap[first + 2].distance;
			float		d3 = heap[first + 3].distance;
			int			b0 = __builtin_isgreater(d0, d1) ? (first + 1) : first;
			float		m0 = __builtin_isgreater(d0, d1) ? d1 : d0;
			int			b1 = __builtin_isgreater(d2, d3) ? (first + 3) : (first + 2);
			float		m1 = __builtin_isgreater(d2, d3) ? d3 : d2;

			best = __builtin_isgreater(m0, m1) ? b1 : b0;
			best_d = __builtin_isgreater(m0, m1) ? m1 : m0;
		}
		else
		{
			best = first;
			best_d = heap[first].distance;
			for (int c = first + 1; c < count; c++)
			{
				if (__builtin_isgreater(best_d, heap[c].distance))
				{
					best = c;
					best_d = heap[c].distance;
				}
			}
		}

		if (!__builtin_isgreater(key.distance, best_d))
			break;
		if (count > 5 && best <= (count - 6) >> 4)
		{
			int			gc = (best << 4) + 5;

			__builtin_prefetch(&heap[gc], 0, 3);
			if (gc + 8 < count)
				__builtin_prefetch(&heap[gc + 8], 0, 3);
		}
		heap[pos] = heap[best];
		pos = best;
	}
	heap[pos] = key;
}

static inline HnswIterativeKey
HnswIterativePop(HnswIterativeState *state)
{
	if (unlikely(state->pending_sorted))
		return state->pending[--state->pending_count];
	{
		HnswIterativeKey key = state->pending[0];

		state->pending[0] = state->pending[--state->pending_count];
		if (state->pending_count > 0)
			HnswIterativeSiftDownMin4(state->pending, state->pending_count, 0);
		return key;
	}
}

static inline void
HnswIterativeSiftDownMax4(HnswIterativeKey *heap, int count, int pos)
{
	HnswIterativeKey key = heap[pos];

	for (;;)
	{
		int			first = (pos << 2) + 1;
		int			best;
		float		best_d;

		if (first >= count)
			break;

		if (first + 3 < count)
		{
			float		d0 = heap[first].distance;
			float		d1 = heap[first + 1].distance;
			float		d2 = heap[first + 2].distance;
			float		d3 = heap[first + 3].distance;
			int			b0 = __builtin_isgreater(d1, d0) ? (first + 1) : first;
			float		m0 = __builtin_isgreater(d1, d0) ? d1 : d0;
			int			b1 = __builtin_isgreater(d3, d2) ? (first + 3) : (first + 2);
			float		m1 = __builtin_isgreater(d3, d2) ? d3 : d2;

			best = __builtin_isgreater(m1, m0) ? b1 : b0;
			best_d = __builtin_isgreater(m1, m0) ? m1 : m0;
		}
		else
		{
			best = first;
			best_d = heap[first].distance;
			for (int c = first + 1; c < count; c++)
			{
				if (__builtin_isgreater(heap[c].distance, best_d))
				{
					best = c;
					best_d = heap[c].distance;
				}
			}
		}

		if (!__builtin_isgreater(best_d, key.distance))
			break;
		if ((best << 2) + 1 < count)
			__builtin_prefetch(&heap[(best << 2) + 1], 0, 3);
		heap[pos] = heap[best];
		pos = best;
	}
	heap[pos] = key;
}

static inline void
HnswIterativePushMax4(HnswIterativeKey key, HnswIterativeKey *heap, int *count)
{
	int			pos = (*count)++;

	while (pos > 0)
	{
		int			parent = (pos - 1) >> 2;

		if (key.distance <= heap[parent].distance)
			break;
		heap[pos] = heap[parent];
		pos = parent;
	}
	heap[pos] = key;
}

static inline bool
HnswIterativeCommit(HnswIterativeState *state, HnswIterativeKey key,
					int ef, int *candidate_count, int *result_count)
{
	if (*result_count == ef && key.distance >= state->results[0].distance)
	{
		HnswIterativePushMin4(key, &state->pending, &state->pending_count,
							  &state->pending_capacity);
		return false;
	}
	HnswIterativePushMin4(key, &state->candidates, candidate_count,
						  &state->candidate_capacity);
	if (*result_count < ef)
		HnswIterativePushMax4(key, state->results, result_count);
	else
	{
		HnswIterativePushMin4(state->results[0], &state->pending,
							  &state->pending_count, &state->pending_capacity);
		state->results[0] = key;
		HnswIterativeSiftDownMax4(state->results, *result_count, 0);
	}
	return true;
}

extern __attribute__((visibility("hidden"))) HnswIterativeState *
HnswInitIterativeState(int ef);

extern __attribute__((visibility("hidden"))) pg_noinline void
HnswSortPendingDesc(HnswIterativeState *state);

#endif							/* HNSWQUEUE_H */
