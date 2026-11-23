/* Shared HNSW tuple scoring and neighbor I/O. */
#ifndef HNSWIO_H
#define HNSWIO_H

#include "hnswprefetch.h"
#include "hnswvisited.h"
#include "utils/rel.h"
#include "vector_simd.h"

extern __attribute__((visibility("hidden"))) double
HnswGetDistance(Datum a, Datum b, HnswSupport *support);

extern __attribute__((visibility("hidden"))) pg_noinline void
HnswLoadElementImpl(BlockNumber blkno, OffsetNumber offno, double *distance, HnswQuery * q, Relation index, HnswSupport * support, bool loadVec, double *maxDistance, HnswElement * element);

extern __attribute__((visibility("hidden"))) pg_noinline bool
HnswLoadNeighborTidsAndHashSlow(BlockNumber neighborPage, OffsetNumber neighborOffno,
								uint8 level, uint8 version, Relation index,
								HnswSupport * support, int m, int lm, int lc,
								uint32 mask, uint64 *slots,
								uint64 *tids48, uint32 *buckets, int *n_tids_out);

extern __attribute__((visibility("hidden"))) pg_noinline bool
HnswLoadUnvisitedMissingOrPinSlow(Relation index, HnswSupport * support,
								  const BlockNumber *batch_blocknums, int count,
								  const HnswUnvisited * unvisited,
								  CalicoOptimisticSlot *slots_out,
								  Size tuple_prefetch_bytes,
								  CalicoAsyncBatch **batch_out_ptr);

extern __attribute__((visibility("hidden"))) pg_noinline void
HnswLoadUnvisitedRaw(BlockNumber neighborPage, OffsetNumber neighborOffno,
					 uint8 level, uint8 version,
					 HnswUnvisited * unvisited, int *unvisitedLength,
					 visited_hash * v, Relation index, HnswSupport * support,
					 int m, int lm, int lc,
					 CalicoOptimisticSlot *slots_out, Size tuple_prefetch_bytes,
					 Size min_valid_bytes, HnswElementTuple *etups_out,
					 uint16 *etup_limits_out, CalicoAsyncBatch **batch_out_ptr);

/* Read a line pointer once and bound every speculative tuple access to this
 * frame. Validation can reject raced data; it cannot undo an invalid pointer
 * dereference that has already happened. */
static inline void *
HnswOptimisticItem(Page page, OffsetNumber offno, Size minimum)
{
    ItemIdData item;
    unsigned offset, length;
    if (offno < FirstOffsetNumber ||
        offno > (BLCKSZ - SizeOfPageHeaderData) / sizeof(ItemIdData))
        return NULL;
    item = *(volatile ItemIdData *) PageGetItemId(page, offno);
    offset = ItemIdGetOffset(&item);
    length = ItemIdGetLength(&item);
    if (!ItemIdIsNormal(&item) || offset < SizeOfPageHeaderData ||
        length < minimum || offset + length > BLCKSZ)
        return NULL;
    return page + offset;
}

/*
 * Calculate a supported vector distance optimistically - assumes 'a' is stable but 'b' may be modified
 * Directly computes distance without copying by validating varlena header first
 */
static pg_attribute_always_inline bool
HnswGetDistanceOptimisticPrefetchImpl(HnswSupport *support, const Vector *a_vec,
									   const Vector *b_ptr, Size vec_size,
									   bool use_avx_fma, const float *pref_bx,
									   double *distance_out)
{
	uint32		vl_len = __atomic_load_n((const uint32 *) &b_ptr->vl_len_, __ATOMIC_RELAXED);
	int16		b_dim = __atomic_load_n((const int16 *) &b_ptr->dim, __ATOMIC_RELAXED);

	/*
	 * A tq4 search: b is a stored element value. Check its marker,
	 * dimensions and length before reading its codes; the caller validates
	 * the frame afterwards.
	 */
	if (unlikely(a_vec->dim == HNSW_TQ_QUERY_MARKER))
	{
		const HnswTqQueryValue *qv = (const HnswTqQueryValue *) a_vec;
		const HnswTqElementValue *ev = (const HnswTqElementValue *) b_ptr;
		uint16		ev_dim = __atomic_load_n(&ev->dim, __ATOMIC_RELAXED);

		(void) use_avx_fma;
		(void) pref_bx;
		if (unlikely(b_dim != HNSW_TQ_ELEMENT_MARKER || ev_dim != qv->dim ||
					 (vl_len >> 2) != (uint32) vec_size || (vl_len & 0x03) != 0))
			return false;

		*distance_out = HnswTqQueryDistance(qv, ev);
		return true;
	}

	if (unlikely(b_dim != a_vec->dim || b_dim <= 0 ||
				 (vl_len >> 2) != (uint32) vec_size || (vl_len & 0x03) != 0))
		return false;

	Assert(support->vectorDistance != HNSW_VECTOR_DISTANCE_NONE);
#if !defined(VECTOR_NATIVE_AVX_FMA) && defined(USE_DISPATCH)
	if (use_avx_fma)
	{
		if (pref_bx != NULL)
		{
			if (support->vectorDistance == HNSW_VECTOR_DISTANCE_L2)
				*distance_out = (double) VectorL2SquaredDistanceAvxFmaPrefetchInline(b_dim, a_vec->x, b_ptr->x, pref_bx);
			else
				*distance_out = (double) -VectorInnerProductAvxFmaPrefetchInline(b_dim, a_vec->x, b_ptr->x, pref_bx);
		}
		else
		{
			if (support->vectorDistance == HNSW_VECTOR_DISTANCE_L2)
				*distance_out = (double) VectorL2SquaredDistanceAvxFmaInline(b_dim, a_vec->x, b_ptr->x);
			else
				*distance_out = (double) -VectorInnerProductAvxFmaInline(b_dim, a_vec->x, b_ptr->x);
		}
		return true;
	}
#elif defined(VECTOR_NATIVE_AVX_FMA)
	(void) use_avx_fma;
	if (pref_bx != NULL)
	{
		if (support->vectorDistance == HNSW_VECTOR_DISTANCE_L2)
			*distance_out = (double) VectorL2SquaredDistanceAvxFmaPrefetchInline(b_dim, a_vec->x, b_ptr->x, pref_bx);
		else
			*distance_out = (double) -VectorInnerProductAvxFmaPrefetchInline(b_dim, a_vec->x, b_ptr->x, pref_bx);
		return true;
	}
#else
	(void) use_avx_fma;
	(void) pref_bx;
#endif
	if (support->vectorDistance == HNSW_VECTOR_DISTANCE_L2)
		*distance_out = (double) VectorL2SquaredDistance(b_dim, a_vec->x, b_ptr->x);
	else
		*distance_out = (double) -VectorInnerProduct(b_dim, a_vec->x, b_ptr->x);
	return true;
}

static pg_attribute_always_inline bool
HnswGetDistanceOptimisticImpl(HnswSupport *support, const Vector *a_vec, const Vector *b_ptr, Size vec_size, bool use_avx_fma, double *distance_out)
{
	return HnswGetDistanceOptimisticPrefetchImpl(support, a_vec, b_ptr, vec_size,
												use_avx_fma, NULL, distance_out);
}

static pg_attribute_always_inline bool
HnswCollectUnvisitedFromNeighbor(BlockNumber neighborPage, OffsetNumber neighborOffno,
								 uint8 level, uint8 version,
								 const CalicoOptimisticSlot *pre_slot,
								 HnswUnvisited *unvisited_out, int *unvisited_cnt_out,
								 visited_hash *v, Relation index, HnswSupport *support,
								 int m, int lm, int lc,
								 bool collect_batch_meta,
								 BlockNumber *batch_blocknums,
								 uint16 *linp_offs)
{
	uint64		tids48[HNSW_MAX_M * 2];
	uint32		buckets[HNSW_MAX_M * 2];
	int			n_tids = 0;
	int			ulen = 0;
	HnswFastVisited *fv = &v->tids;
	uint64	   *slots;
	uint32		mask;
	uint64		gen_prefix;
	uint16		gen;
	const bool	opt_active = (calico_enable_optimistic_reads &&
							  calico_is_active() && !RelationUsesLocalBuffers(index));

	*unvisited_cnt_out = 0;

	if (unlikely(fv->count + lm > fv->grow_at))
		hnsw_fast_visited_grow(fv, (uint32) lm);

	slots = fv->slots;
	mask = fv->mask;
	gen = fv->gen;
	gen_prefix = (uint64) gen << 48;

	/*
	 * Fast optimistic neighbor load + single-pass TID hash & prefetch:
	 * read ntup->indextids directly into registers, prefetch visited slots,
	 * and validate translation entry before mutating fv->slots.
	 */
	if (likely(opt_active && (pre_slot != NULL || support != NULL)))
	{
		uint64		opt_version;
		uint32		buf_id;
		void	   *entry_ptr;
		bool		have_frame;

		if (pre_slot != NULL)
		{
			have_frame = pre_slot->opt_valid && pre_slot->entry_ptr != NULL;
			opt_version = pre_slot->version;
			buf_id = pre_slot->frame_id;
			entry_ptr = (void *) pre_slot->entry_ptr;
		}
		else
		{
			have_frame = calico_optimistic_lookup_handle_frame(&support->read_handle,
															   neighborPage,
															   &opt_version,
															   &buf_id,
															   &entry_ptr);
		}

		if (likely(have_frame))
		{
			if (likely(level >= lc && m >= 0 && m <= HNSW_MAX_M &&
					   lm >= 0 && lm <= HNSW_MAX_M * 2))
			{
				Page		page = CalicoFramePage(buf_id);
				Size		start = (Size) (level - lc) * (Size) m;
				Size		req_size = offsetof(HnswNeighborTupleData, indextids) +
					sizeof(ItemPointerData) * (start + (Size) lm);
				HnswNeighborTuple ntup = HnswOptimisticItem(page, neighborOffno, req_size);

				if (likely(ntup != NULL &&
						   ntup->version == version &&
						   ntup->count == (level + 2) * m))
				{
					const ItemPointerData *src_tids = &ntup->indextids[start];
					int			cnt = 0;

					for (int i = 0; i < lm; i++)
					{
						ItemPointerData tid;
						uint64		tid48;
						uint32		pos;

						memcpy(&tid, &src_tids[i], sizeof(ItemPointerData));
						if (!ItemPointerIsValid(&tid))
							break;

						tid48 = ((uint64) BlockIdGetBlockNumber(&tid.ip_blkid) << 16) |
							(uint64) tid.ip_posid;
						pos = hnsw_hash_tid48(tid48) & mask;
						tids48[cnt] = tid48;
						buckets[cnt] = pos;
						__builtin_prefetch(&slots[pos], 0, 3);
						cnt++;
					}

					if (likely(calico_optimistic_validate(entry_ptr, opt_version)))
					{
						n_tids = cnt;
						goto have_tids;
					}
				}
			}
			calico_optimistic_neighbor_fallbacks++;
		}
		else
			calico_optimistic_neighbor_fallbacks++;
	}

	if (!HnswLoadNeighborTidsAndHashSlow(neighborPage, neighborOffno, level, version,
										 index, support, m, lm, lc, mask, slots,
										 tids48, buckets, &n_tids))
		return false;

have_tids:
	for (int i = 0; i < n_tids; i++)
	{
		uint64		tid48 = tids48[i];
		uint64		target = gen_prefix | tid48;
		uint32		pos = buckets[i];
		bool		found = false;

		for (;;)
		{
			uint64		cur = slots[pos];

			if (cur == target)
			{
				found = true;
				break;
			}
			if ((uint16) (cur >> 48) != gen)
			{
				slots[pos] = target;
				fv->count++;
				break;
			}
			pos = (pos + 1) & mask;
		}

		if (!found)
		{
			BlockNumber blkno = (BlockNumber) (tid48 >> 16);
			OffsetNumber offno = (OffsetNumber) (tid48 & 0xffffU);

			ItemPointerSet(&unvisited_out[ulen].indextid,
						   blkno,
						   offno);

			if (collect_batch_meta)
			{
				batch_blocknums[ulen] = blkno;
				linp_offs[ulen] = likely(offno >= FirstOffsetNumber &&
										 offno <= (BLCKSZ - SizeOfPageHeaderData) / sizeof(ItemIdData))
					? (uint16) (offsetof(PageHeaderData, pd_linp) + (offno - 1) * sizeof(ItemIdData))
					: 0;
				if (likely(support != NULL))
					calico_optimistic_prefetch_block(&support->read_handle, blkno);
			}

			ulen++;
		}
	}

	*unvisited_cnt_out = ulen;
	return true;
}

static inline void
HnswInitOptDistanceParams(HnswQuery *q, Relation index, HnswSupport *support,
						  bool *canOptDistanceOut, const Vector **queryVecOut,
						  Size *optVecSizeOut, Size *optPrefetchBytesOut)
{
	bool		canOptDistance = false;
	const Vector *queryVec = NULL;
	Size		optVecSize = 0;
	Size		optPrefetchBytes = 0;

	if (calico_enable_optimistic_reads &&
		support != NULL &&
		calico_is_active() && !RelationUsesLocalBuffers(index))
	{
		if (support->vectorDistance != HNSW_VECTOR_DISTANCE_NONE)
		{
			if (DatumGetPointer(q->value) == NULL)
			{
				canOptDistance = true;
				optPrefetchBytes = offsetof(HnswElementTupleData, data);
			}
			else
			{
				queryVec = (const Vector *) DatumGetPointer(q->value);
				if (likely(HnswQueryValueSize(queryVec) > 0))
				{
					optVecSize = HnswQueryValueSize(queryVec);
					if (likely(optVecSize <= BLCKSZ - offsetof(HnswElementTupleData, data)))
					{
						canOptDistance = true;
						optPrefetchBytes = Min(offsetof(HnswElementTupleData, data) + optVecSize,
											   (Size) BLCKSZ);
					}
				}
			}
		}
		else if (hnsw_enable_prefetch)
		{
			optPrefetchBytes = sizeof(HnswElementTupleData) + 64;
		}
	}

	*canOptDistanceOut = canOptDistance;
	*queryVecOut = queryVec;
	*optVecSizeOut = optVecSize;
	*optPrefetchBytesOut = optPrefetchBytes;
}

#endif							/* HNSWIO_H */
