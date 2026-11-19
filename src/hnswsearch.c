/* Compact candidate HNSW search for on-disk indexes. */
#include "postgres.h"

#include "hnswio.h"
#include "hnswsearch.h"
#include "lib/pairingheap.h"
#include "nodes/pg_list.h"
#include "storage/bufmgr.h"

static pg_attribute_always_inline bool
HnswLoadElementOptimisticSlotFastImpl(HnswSupport *support, BlockNumber blkno, OffsetNumber offno,
									  const CalicoOptimisticSlot *slot,
									  HnswElementTuple cached_etup,
									  const char *buf_blocks,
									  const Vector *a_vec, Size vec_size,
									  double maxDistance, bool alwaysAdd,
									  bool use_avx_fma,
									  HnswFastCand *cand_out, bool *accepted)
{
	Page		page;
	HnswElementTuple etup;
	double		dist;
	bool		should_add;
	uint8		level = 0;
	uint8		version = 0;
	ItemPointerData nb;

	ItemPointerSetInvalid(&nb);

	if (likely(cached_etup != NULL))
	{
		etup = cached_etup;
	}
	else
	{
		Size		min_bytes;

		if (unlikely(slot == NULL || !slot->opt_valid || slot->entry_ptr == NULL))
			return false;

		page = CalicoFramePageBase(buf_blocks, slot->frame_id);
		min_bytes = likely(a_vec != NULL)
			? (offsetof(HnswElementTupleData, data) + vec_size)
			: offsetof(HnswElementTupleData, data);
		etup = HnswOptimisticItem(page, offno, min_bytes);
		if (unlikely(etup == NULL))
			return false;
	}

	if (unlikely(etup->deleted))
		return false;

	if (likely(a_vec != NULL))
	{
		if (unlikely(!HnswGetDistanceOptimisticImpl(support, a_vec, &etup->data, vec_size, use_avx_fma, &dist)))
			return false;
	}
	else
		dist = 0;

	should_add = (alwaysAdd || dist < maxDistance);
	if (should_add)
	{
		level = etup->level;
		version = etup->version;
		nb = etup->neighbortid;
	}

	if (unlikely(!calico_optimistic_validate(slot->entry_ptr, slot->version)))
		return false;

	if (should_add)
	{
		cand_out->distance = dist;
		cand_out->blkno = blkno;
		cand_out->neighborPage = ItemPointerGetBlockNumber(&nb);
		cand_out->offno = offno;
		cand_out->neighborOffno = ItemPointerGetOffsetNumber(&nb);
		cand_out->level = level;
		cand_out->version = version;
		*accepted = true;
	}
	else
		*accepted = false;

	return true;
}

static pg_attribute_always_inline bool
HnswLoadElementOptimisticSlotFast(HnswSupport *support, BlockNumber blkno, OffsetNumber offno,
								  const CalicoOptimisticSlot *slot,
								  HnswElementTuple cached_etup,
								  const Vector *a_vec, Size vec_size,
								  double maxDistance, bool alwaysAdd,
								  HnswFastCand *cand_out, bool *accepted)
{
	return HnswLoadElementOptimisticSlotFastImpl(support, blkno, offno, slot, cached_etup,
												 BufferBlocks, a_vec, vec_size,
												 maxDistance, alwaysAdd, false,
												 cand_out, accepted);
}

static pg_noinline bool
HnswLoadFastCandSlow(BlockNumber blkno, OffsetNumber offno,
					 HnswQuery * q, Relation index, HnswSupport * support,
					 bool canOptDistance, const Vector *a_vec, Size vec_size,
					 double maxDistance, bool alwaysAdd,
					 HnswFastCand *cand_out)
{
	Buffer		buf;
	Page		page;
	HnswElementTuple etup;
	double		dist;

	if (canOptDistance)
	{
		bool		found = false;
		uint64		version = 0;
		uint32		buf_id = 0;
		void	   *entry_ptr = NULL;

		found = calico_optimistic_lookup_handle_frame(&support->read_handle, blkno,
													  &version, &buf_id, &entry_ptr);
		if (found)
		{
			CalicoOptimisticSlot retry_slot;
			bool		accepted = false;

			retry_slot.entry_ptr = (CalicoTranslationEntry *) entry_ptr;
			retry_slot.version = version;
			retry_slot.frame_id = buf_id;
			retry_slot.resident = true;
			retry_slot.opt_valid = true;
			if (HnswLoadElementOptimisticSlotFast(support, blkno, offno, &retry_slot, NULL,
												  a_vec, vec_size,
												  maxDistance, alwaysAdd,
												  cand_out, &accepted))
				return accepted;
		}
		calico_optimistic_element_fallbacks++;
	}

	buf = ReadBuffer(index, blkno);
	LockBuffer(buf, BUFFER_LOCK_SHARE);
	page = BufferGetPage(buf);

	etup = (HnswElementTuple) PageGetItem(page, PageGetItemId(page, offno));
	Assert(HnswIsElementTuple(etup));

	if (unlikely(etup->deleted))
		elog(ERROR, "cannot load deleted element");

	if (DatumGetPointer(q->value) == NULL)
		dist = 0;
	else
		dist = HnswGetDistance(q->value, PointerGetDatum(&etup->data), support);

	if (alwaysAdd || dist < maxDistance)
	{
		cand_out->distance = dist;
		cand_out->blkno = blkno;
		cand_out->neighborPage = ItemPointerGetBlockNumber(&etup->neighbortid);
		cand_out->offno = offno;
		cand_out->neighborOffno = ItemPointerGetOffsetNumber(&etup->neighbortid);
		cand_out->level = etup->level;
		cand_out->version = etup->version;
		UnlockReleaseBuffer(buf);
		return true;
	}

	UnlockReleaseBuffer(buf);
	return false;
}

static pg_noinline void
HnswEvalAsyncUnvisitedBatchFast(HnswQuery * q, int ef, int lc,
								Relation index, HnswSupport * support,
								bool canOptDistance, const Vector *queryVec,
								Size optVecSize, Size optPrefetchBytes,
								HnswUnvisited * unvisited, int unvisitedLength,
								CalicoOptimisticSlot *unvisited_slots,
								CalicoAsyncBatch *async_batch,
								HnswFastCand **c_heap_ptr, int *clen, int *ccap, bool *c_on_stack,
								HnswFastResultCand *w_heap, int *wlen)
{
	bool		processed[HNSW_MAX_M * 2];
	int			res_idxs[HNSW_MAX_M * 2];
	int			n_res = 0;
	int			rem_idxs[HNSW_MAX_M * 2];
	BlockNumber rem_blks[HNSW_MAX_M * 2];
	CalicoOptimisticSlot rem_slots[HNSW_MAX_M * 2];
	int			n_rem = 0;

	for (int i = 0; i < unvisitedLength; i++)
	{
		processed[i] = false;
		if (unvisited_slots[i].resident && unvisited_slots[i].opt_valid)
			res_idxs[n_res++] = i;
	}

	/* Phase 1: evaluate resident neighbors concurrently with io_uring NVMe reads */
	for (int r = 0; r < n_res; r++)
	{
		int			idx = res_idxs[r];
		ItemPointer indextid = &unvisited[idx].indextid;
		BlockNumber blkno = ItemPointerGetBlockNumber(indextid);
		OffsetNumber offno = ItemPointerGetOffsetNumber(indextid);
		HnswFastCand cand;
		bool		accepted = false;
		bool		alwaysAdd = (*wlen < ef);
		double		maxDist = alwaysAdd ? 0.0 : w_heap[0].distance;

		if (hnsw_enable_prefetch && r + HNSW_PREFETCH_AHEAD < n_res)
		{
			int			p_idx = res_idxs[r + HNSW_PREFETCH_AHEAD];

			HnswPrefetchSlotTuple(&unvisited_slots[p_idx],
								  ItemPointerGetOffsetNumber(&unvisited[p_idx].indextid),
								  optPrefetchBytes);
		}

		if (canOptDistance &&
			HnswLoadElementOptimisticSlotFast(support, blkno, offno, &unvisited_slots[idx], NULL,
											  queryVec, optVecSize,
											  maxDist, alwaysAdd,
											  &cand, &accepted))
		{
			processed[idx] = true;
			if (accepted)
				HnswFastCommitCand(&cand, ef, lc,
								   c_heap_ptr, clen, ccap, c_on_stack,
								   w_heap, wlen);
		}
	}

	/* Phase 2: wait for async reads, mark frames ready & release pins */
	CalicoAsyncBatchFinish(async_batch);

	for (int i = 0; i < unvisitedLength; i++)
	{
		if (!processed[i])
		{
			rem_idxs[n_rem] = i;
			rem_blks[n_rem] = ItemPointerGetBlockNumber(&unvisited[i].indextid);
			n_rem++;
		}
	}

	if (n_rem > 0)
	{
		if (support != NULL)
		{
			calico_optimistic_lookup_batch_slots(&support->read_handle,
												 rem_blks, n_rem, rem_slots);
			for (int k = 0; k < n_rem; k++)
				unvisited_slots[rem_idxs[k]] = rem_slots[k];
		}

		if (hnsw_enable_prefetch)
		{
			int			prime = Min(n_rem, HNSW_PREFETCH_AHEAD);

			for (int k = 0; k < prime; k++)
			{
				int			p_idx = rem_idxs[k];

				HnswPrefetchSlotTuple(&unvisited_slots[p_idx],
									  ItemPointerGetOffsetNumber(&unvisited[p_idx].indextid),
									  optPrefetchBytes);
			}
		}

		for (int k = 0; k < n_rem; k++)
		{
			int			idx = rem_idxs[k];
			BlockNumber blkno = rem_blks[k];
			OffsetNumber offno = ItemPointerGetOffsetNumber(&unvisited[idx].indextid);
			HnswFastCand cand;
			bool		accepted = false;
			bool		alwaysAdd = (*wlen < ef);
			double		maxDist = alwaysAdd ? 0.0 : w_heap[0].distance;

			if (hnsw_enable_prefetch && k + HNSW_PREFETCH_AHEAD < n_rem)
			{
				int			p_idx = rem_idxs[k + HNSW_PREFETCH_AHEAD];

				HnswPrefetchSlotTuple(&unvisited_slots[p_idx],
									  ItemPointerGetOffsetNumber(&unvisited[p_idx].indextid),
									  optPrefetchBytes);
			}

			if (canOptDistance &&
				HnswLoadElementOptimisticSlotFast(support, blkno, offno, &unvisited_slots[idx], NULL,
												  queryVec, optVecSize,
												  maxDist, alwaysAdd,
												  &cand, &accepted))
			{
				if (accepted)
					HnswFastCommitCand(&cand, ef, lc,
									   c_heap_ptr, clen, ccap, c_on_stack,
									   w_heap, wlen);
			}
			else if (HnswLoadFastCandSlow(blkno, offno, q, index, support,
										  canOptDistance, queryVec, optVecSize,
										  maxDist, alwaysAdd, &cand))
			{
				HnswFastCommitCand(&cand, ef, lc,
								   c_heap_ptr, clen, ccap, c_on_stack,
								   w_heap, wlen);
			}
		}
	}
}

static pg_attribute_always_inline void
HnswScoreDiskFastResidentBody(HnswQuery *q, int ef, int lc, int m, int lm,
							  Relation index, HnswSupport *support,
							  const HnswUnvisited *unvisited, int unvisitedLength,
							  const CalicoOptimisticSlot *unvisited_slots,
							  HnswElementTuple *cand_etups, uint16 *cand_limits,
							  bool canOptDistance, const Vector *queryVec,
							  Size optVecSize, Size optPrefetchBytes, Size minValidBytes,
							  const char *buf_blocks, CalicoReadHandle *read_handle,
							  bool doStepPrefetch, bool doPipe,
							  BlockNumber p0_page, OffsetNumber p0_off,
							  uint8 p0_level, int p0_stage, int p0_at_i, uint32 p0_buf_id,
							  HnswFastCand **c_heap_ptr, int *clen_ptr,
							  int *ccap_ptr, bool *c_on_stack_ptr,
							  HnswFastResultCand *w_heap, int *wlen_ptr,
							  bool force_avx_fma)
{
	for (int i = 0; i < unvisitedLength; i++)
	{
		const ItemPointerData *indextid = &unvisited[i].indextid;
		BlockNumber blkno = ItemPointerGetBlockNumber(indextid);
		OffsetNumber offno = ItemPointerGetOffsetNumber(indextid);
		HnswFastCand cand;
		bool		accepted = false;
		bool		alwaysAdd = (*wlen_ptr < ef);
		double		maxDist = alwaysAdd ? 0.0 : w_heap[0].distance;

		if (doStepPrefetch)
			HnswSteppedPrefetchAdvance(i, unvisitedLength, unvisited, unvisited_slots,
									   cand_etups, cand_limits,
									   optPrefetchBytes, minValidBytes, buf_blocks);

		if (p0_stage == 1 && (i - p0_at_i) >= 3)
		{
			uint64		opt_ver;
			void	   *entry_ptr;

			if (likely(calico_optimistic_lookup_handle_frame(read_handle, p0_page,
															 &opt_ver, &p0_buf_id, &entry_ptr)) &&
				likely(p0_off >= FirstOffsetNumber &&
					   p0_off <= (BLCKSZ - SizeOfPageHeaderData) / sizeof(ItemIdData)))
			{
				__builtin_prefetch(PageGetItemId(CalicoFramePageBase(buf_blocks, p0_buf_id), p0_off), 0, 3);
				p0_stage = 2;
				p0_at_i = i;
			}
			else
				p0_stage = -1;
		}
		else if (p0_stage == 2 && (i - p0_at_i) >= 4)
		{
			HnswPrefetchNextNeighborTuple(buf_blocks, p0_buf_id, p0_off, p0_level, m, lm);
			p0_stage = 3;
		}

		if (canOptDistance &&
			HnswLoadElementOptimisticSlotFastImpl(support, blkno, offno, &unvisited_slots[i],
												  cand_etups[i], buf_blocks,
												  queryVec, optVecSize,
												  maxDist, alwaysAdd,
												  force_avx_fma,
												  &cand, &accepted))
		{
			if (accepted)
			{
				HnswFastCommitCand(&cand, ef, lc,
								   c_heap_ptr, clen_ptr, ccap_ptr, c_on_stack_ptr,
								   w_heap, wlen_ptr);
				if (doPipe &&
					((*c_heap_ptr)[0].neighborPage != p0_page ||
					 (*c_heap_ptr)[0].neighborOffno != p0_off))
				{
					p0_page = (*c_heap_ptr)[0].neighborPage;
					p0_off = (*c_heap_ptr)[0].neighborOffno;
					p0_level = (*c_heap_ptr)[0].level;
					calico_optimistic_prefetch_block(read_handle, p0_page);
					p0_stage = 1;
					p0_at_i = i;
				}
			}
		}
		else if (HnswLoadFastCandSlow(blkno, offno, q, index, support,
									  canOptDistance, queryVec, optVecSize,
									  maxDist, alwaysAdd, &cand))
		{
			HnswFastCommitCand(&cand, ef, lc,
							   c_heap_ptr, clen_ptr, ccap_ptr, c_on_stack_ptr,
							   w_heap, wlen_ptr);
			if (doPipe &&
				((*c_heap_ptr)[0].neighborPage != p0_page ||
				 (*c_heap_ptr)[0].neighborOffno != p0_off))
			{
				p0_page = (*c_heap_ptr)[0].neighborPage;
				p0_off = (*c_heap_ptr)[0].neighborOffno;
				p0_level = (*c_heap_ptr)[0].level;
				calico_optimistic_prefetch_block(read_handle, p0_page);
				p0_stage = 1;
				p0_at_i = i;
			}
		}
	}

	if (p0_stage == 2)
		HnswPrefetchNextNeighborTuple(buf_blocks, p0_buf_id, p0_off, p0_level, m, lm);
	else if (p0_stage == 1 && (unvisitedLength - p0_at_i) >= 2)
	{
		uint64		opt_ver;
		void	   *entry_ptr;

		if (likely(calico_optimistic_lookup_handle_frame(read_handle, p0_page,
														 &opt_ver, &p0_buf_id, &entry_ptr)) &&
			likely(p0_off >= FirstOffsetNumber &&
				   p0_off <= (BLCKSZ - SizeOfPageHeaderData) / sizeof(ItemIdData)))
			__builtin_prefetch(PageGetItemId(CalicoFramePageBase(buf_blocks, p0_buf_id), p0_off), 0, 3);
	}
}

#if !defined(VECTOR_NATIVE_AVX_FMA) && defined(USE_DISPATCH)
static pg_noinline VECTOR_TARGET_AVX_FMA void
HnswScoreDiskFastResidentAvxFma(HnswQuery *q, int ef, int lc, int m, int lm,
								Relation index, HnswSupport *support,
								const HnswUnvisited *unvisited, int unvisitedLength,
								const CalicoOptimisticSlot *unvisited_slots,
								HnswElementTuple *cand_etups, uint16 *cand_limits,
								const Vector *queryVec, Size optVecSize,
								Size optPrefetchBytes, Size minValidBytes,
								const char *buf_blocks, CalicoReadHandle *read_handle,
								bool doStepPrefetch, bool doPipe,
								BlockNumber p0_page, OffsetNumber p0_off,
								uint8 p0_level, int p0_stage, int p0_at_i, uint32 p0_buf_id,
								HnswFastCand **c_heap_ptr, int *clen_ptr,
								int *ccap_ptr, bool *c_on_stack_ptr,
								HnswFastResultCand *w_heap, int *wlen_ptr)
{
	HnswScoreDiskFastResidentBody(q, ef, lc, m, lm, index, support,
								  unvisited, unvisitedLength, unvisited_slots,
								  cand_etups, cand_limits, true, queryVec,
								  optVecSize, optPrefetchBytes, minValidBytes,
								  buf_blocks, read_handle, doStepPrefetch, doPipe,
								  p0_page, p0_off, p0_level, p0_stage, p0_at_i, p0_buf_id,
								  c_heap_ptr, clen_ptr, ccap_ptr, c_on_stack_ptr,
								  w_heap, wlen_ptr, true);
}
#endif

static pg_noinline List *
HnswSearchLayerDiskFastCore(char *base, HnswQuery * q, List *ep,
							const HnswFastCand *single_ep,
							int ef, int lc,
							Relation index, HnswSupport * support, int m,
							visited_hash * v, bool initVisited, int64 *tuples,
							bool have_opt_distance, bool canOptDistance,
							const Vector *queryVec, Size optVecSize,
							Size optPrefetchBytes,
							HnswFastResultCand *inline_buf, int inline_cap,
							HnswFastResultCand **fast_out, int *fast_count_out)
{
	List	   *w = NIL;
	HnswFastCand c_stack[256];
	HnswFastResultCand w_stack[64];
	HnswFastCand *c_heap = c_stack;
	HnswFastResultCand *w_heap;
	int			clen = 0;
	int			ccap = lengthof(c_stack);
	bool		c_on_stack = true;
	int			wlen = 0;
	bool		w_on_stack;
	visited_hash vh;
	bool		use_local_v = (v == NULL);
	int			lm = HnswGetLayerM(m, lc);
	HnswUnvisited stack_unvisited[64];
	CalicoOptimisticSlot stack_slots[64];
	HnswElementTuple stack_etups[64];
	uint16		stack_limits[64];
	HnswUnvisited *unvisited;
	CalicoOptimisticSlot *unvisited_slots;
	HnswElementTuple *cand_etups;
	uint16	   *cand_limits;
	int			unvisitedLength;
	uint64		stack_visited_buf[128];
	CalicoAsyncBatch *async_batch = NULL;
	const bool	doPrefetch = hnsw_enable_prefetch;
	const bool	optActive = (calico_enable_optimistic_reads &&
							 support != NULL &&
							 calico_is_active() &&
							 !RelationUsesLocalBuffers(index));
	CalicoReadHandle *const read_handle = optActive ? &support->read_handle : NULL;
	const char *const buf_blocks = BufferBlocks;
	Size		rawPrefetchBytes;
	Size		minValidBytes;
	bool		doStepPrefetch;
	const bool	doPipe = (doPrefetch && optActive && lc == 0);

	if (likely(ef <= lengthof(w_stack)))
	{
		w_heap = w_stack;
		w_on_stack = true;
	}
	else
	{
		w_heap = palloc_array_checked(HnswFastResultCand, ef);
		w_on_stack = false;
	}

	if (likely(lm <= lengthof(stack_unvisited)))
	{
		unvisited = stack_unvisited;
		unvisited_slots = stack_slots;
		cand_etups = stack_etups;
		cand_limits = stack_limits;
	}
	else
	{
		unvisited = palloc_array_checked(HnswUnvisited, lm);
		unvisited_slots = palloc_array_checked(CalicoOptimisticSlot, lm);
		cand_etups = palloc_array_checked(HnswElementTuple, lm);
		cand_limits = palloc_array_checked(uint16, lm);
	}

	if (!have_opt_distance)
		HnswInitOptDistanceParams(q, index, support, &canOptDistance,
								  &queryVec, &optVecSize, &optPrefetchBytes);

	rawPrefetchBytes = doPrefetch ? optPrefetchBytes : 0;
	minValidBytes = offsetof(HnswElementTupleData, data) + optVecSize;
	doStepPrefetch = (doPrefetch && canOptDistance && optPrefetchBytes > 0);

	if (use_local_v)
	{
		memset(&vh, 0, sizeof(vh));
		v = &vh;
		initVisited = true;
	}

	if (initVisited)
		hnsw_fast_visited_init(&v->tids, ef, m,
							  use_local_v ? stack_visited_buf : NULL,
							  use_local_v ? lengthof(stack_visited_buf) : 0);

	if (likely(single_ep != NULL))
	{
		if (initVisited)
		{
			(void) hnsw_fast_visited_insert(&v->tids,
											single_ep->blkno,
											single_ep->offno);
			if (tuples != NULL)
				(*tuples)++;
		}

		c_heap[0] = *single_ep;
		clen = 1;
		w_heap[0].distance = single_ep->distance;
		w_heap[0].blkno = single_ep->blkno;
		w_heap[0].offno = single_ep->offno;
		w_heap[0]._pad = 0;
		wlen = 1;
	}
	else
	{
		ListCell   *lc2;

		foreach(lc2, ep)
		{
			HnswSearchCandidate *sc = (HnswSearchCandidate *) lfirst(lc2);
			HnswElement ep_el = HnswPtrAccess(base, sc->element);
			HnswFastCand cand;

			if (initVisited)
			{
				(void) hnsw_fast_visited_insert(&v->tids, ep_el->blkno, ep_el->offno);
				if (tuples != NULL)
					(*tuples)++;
			}

			cand.distance = sc->distance;
			cand.blkno = ep_el->blkno;
			cand.neighborPage = ep_el->neighborPage;
			cand.offno = ep_el->offno;
			cand.neighborOffno = ep_el->neighborOffno;
			cand.level = ep_el->level;
			cand.version = ep_el->version;

			if (unlikely(clen == ccap))
			{
				int			newcap = ccap * 2;

				if (c_on_stack)
				{
					HnswFastCand *nc = palloc_array_checked(HnswFastCand, newcap);

					memcpy(nc, c_heap, sizeof(HnswFastCand) * (Size) clen);
					c_heap = nc;
					c_on_stack = false;
				}
				else
					c_heap = (HnswFastCand *) repalloc(c_heap,
													   sizeof(HnswFastCand) * (Size) newcap);
				ccap = newcap;
			}
			c_heap[clen] = cand;
			HnswFastHeapSiftUpMin(c_heap, clen++);

			if (wlen < ef)
			{
				int			pos = wlen++;

				w_heap[pos].distance = cand.distance;
				w_heap[pos].blkno = cand.blkno;
				w_heap[pos].offno = cand.offno;
				w_heap[pos]._pad = 0;
				HnswFastHeapSiftUpMax(w_heap, pos);
			}
			else if (cand.distance < w_heap[0].distance)
			{
				w_heap[0].distance = cand.distance;
				w_heap[0].blkno = cand.blkno;
				w_heap[0].offno = cand.offno;
				w_heap[0]._pad = 0;
				HnswFastHeapSiftDownMax(w_heap, wlen, 0);
			}
		}
	}

	while (clen > 0)
	{
		HnswFastCand c = c_heap[0];
		BlockNumber p0_page = InvalidBlockNumber;
		OffsetNumber p0_off = InvalidOffsetNumber;
		uint8		p0_level = 0;
		int			p0_stage = 0;
		int			p0_at_i = -2;
		uint32		p0_buf_id = 0;

		if (wlen > 0 && c.distance > w_heap[0].distance)
			break;

		c_heap[0] = c_heap[--clen];
		if (clen > 0)
			HnswFastHeapSiftDownMin(c_heap, clen, 0);

		if (doPipe && clen > 0 &&
			(wlen == 0 || c_heap[0].distance <= w_heap[0].distance))
		{
			p0_page = c_heap[0].neighborPage;
			p0_off = c_heap[0].neighborOffno;
			p0_level = c_heap[0].level;
			calico_optimistic_prefetch_block(read_handle, p0_page);
			p0_stage = 1;
		}

		HnswLoadUnvisitedRaw(c.neighborPage, c.neighborOffno, c.level, c.version,
							 unvisited, &unvisitedLength, v, index, support,
							 m, lm, lc, unvisited_slots,
							 rawPrefetchBytes, minValidBytes,
							 canOptDistance ? cand_etups : NULL,
							 canOptDistance ? cand_limits : NULL,
							 &async_batch);

		if (tuples != NULL)
			(*tuples) += unvisitedLength;

		if (likely(async_batch == NULL || !async_batch->active))
		{
#if !defined(VECTOR_NATIVE_AVX_FMA) && defined(USE_DISPATCH)
			if (likely(canOptDistance && queryVec != NULL && vector_has_avx_fma))
			{
				HnswScoreDiskFastResidentAvxFma(q, ef, lc, m, lm, index, support,
												unvisited, unvisitedLength, unvisited_slots,
												cand_etups, cand_limits,
												queryVec, optVecSize,
												optPrefetchBytes, minValidBytes,
												buf_blocks, read_handle, doStepPrefetch, doPipe,
												p0_page, p0_off, p0_level,
												p0_stage, p0_at_i, p0_buf_id,
												&c_heap, &clen, &ccap, &c_on_stack,
												w_heap, &wlen);
			}
			else
#endif
			{
				HnswScoreDiskFastResidentBody(q, ef, lc, m, lm, index, support,
											  unvisited, unvisitedLength, unvisited_slots,
											  cand_etups, cand_limits,
											  canOptDistance, queryVec, optVecSize,
											  optPrefetchBytes, minValidBytes,
											  buf_blocks, read_handle, doStepPrefetch, doPipe,
											  p0_page, p0_off, p0_level,
											  p0_stage, p0_at_i, p0_buf_id,
											  &c_heap, &clen, &ccap, &c_on_stack,
											  w_heap, &wlen, false);
			}
		}
		else
		{
			HnswEvalAsyncUnvisitedBatchFast(q, ef, lc, index, support,
											canOptDistance, queryVec, optVecSize,
											optPrefetchBytes,
											unvisited, unvisitedLength,
											unvisited_slots, async_batch,
											&c_heap, &clen, &ccap, &c_on_stack,
											w_heap, &wlen);
		}
	}

	if (use_local_v && vh.tids.slots != NULL && !vh.tids.on_stack)
	{
		pfree(vh.tids.slots);
		vh.tids.slots = NULL;
	}

	if (unlikely(async_batch != NULL))
		pfree(async_batch);

	for (int rem = wlen - 1; rem > 0; rem--)
	{
		HnswFastResultCand top = w_heap[0];

		w_heap[0] = w_heap[rem];
		w_heap[rem] = top;
		HnswFastHeapSiftDownMax(w_heap, rem, 0);
	}

	if (!c_on_stack)
		pfree(c_heap);
	if (unlikely(lm > lengthof(stack_unvisited)))
	{
		pfree(unvisited);
		pfree(unvisited_slots);
		pfree(cand_etups);
		pfree(cand_limits);
	}

	if (fast_out != NULL)
	{
		*fast_count_out = wlen;
		if (wlen == 0)
		{
			*fast_out = NULL;
			if (!w_on_stack)
				pfree(w_heap);
		}
		else if (w_on_stack)
		{
			HnswFastResultCand *out;

			if (inline_buf != NULL && wlen <= inline_cap)
				out = inline_buf;
			else
				out = palloc_array_checked(HnswFastResultCand, wlen);

			memcpy(out, w_heap, sizeof(HnswFastResultCand) * (Size) wlen);
			*fast_out = out;
		}
		else
		{
			*fast_out = w_heap;
		}
		return NIL;
	}

	w = HnswMaterializeFastList(base, q, index, support, w_heap, wlen);
	if (!w_on_stack)
		pfree(w_heap);

	return w;
}

List *
HnswSearchLayerDiskFast(char *base, HnswQuery * q, List *ep, int ef, int lc,
						Relation index, HnswSupport * support, int m,
						visited_hash * v, bool initVisited, int64 *tuples,
						HnswFastResultCand *inline_buf, int inline_cap,
						HnswFastResultCand **fast_out, int *fast_count_out)
{
	return HnswSearchLayerDiskFastCore(base, q, ep, NULL, ef, lc, index, support, m,
									   v, initVisited, tuples,
									   false, false, NULL, 0, 0,
									   inline_buf, inline_cap, fast_out, fast_count_out);
}

bool
HnswSearchLayerFastCands(char *base, HnswQuery * q, List *ep, int ef, int lc,
						 Relation index, HnswSupport * support, int m,
						 visited_hash * v, int64 *tuples,
						 HnswFastResultCand *inline_buf, int inline_cap,
						 HnswFastResultCand **fast_out, int *fast_count_out)
{
	if (unlikely(index == NULL || ef <= 0))
		return false;

	HnswSearchLayerDiskFast(base, q, ep, ef, lc, index, support, m, v, true, tuples,
							inline_buf, inline_cap, fast_out, fast_count_out);
	return true;
}

pg_noinline void
HnswSearchLayerUpperDiskFast(HnswQuery * q, HnswElement entryPoint,
							 Relation index, HnswSupport * support, int m,
							 bool canOptDistance, const Vector *queryVec,
							 Size optVecSize, Size optPrefetchBytes,
							 HnswFastCand *best_out)
{
	static uint64 hnsw_upper_visited_slots[256];
	static uint16 hnsw_upper_visited_gen = 0;
	HnswFastCand best;
	bool		ep_loaded = false;
	Size		minValidBytes = offsetof(HnswElementTupleData, data) + optVecSize;

	if (likely(canOptDistance))
	{
		uint64		opt_ver = 0;
		uint32		buf_id = 0;
		void	   *entry_ptr = NULL;

		if (likely(calico_optimistic_lookup_handle_frame(&support->read_handle,
														 entryPoint->blkno,
														 &opt_ver, &buf_id, &entry_ptr)))
		{
			CalicoOptimisticSlot ep_slot;
			HnswElementTuple etup;
			bool		ep_acc = false;

			ep_slot.entry_ptr = entry_ptr;
			ep_slot.version = opt_ver;
			ep_slot.frame_id = buf_id;
			ep_slot.resident = true;
			ep_slot.opt_valid = true;
			etup = HnswPrefetchSlotTupleEx(&ep_slot, entryPoint->offno,
										   optPrefetchBytes, minValidBytes);
			if (likely(HnswLoadElementOptimisticSlotFast(support, entryPoint->blkno,
														 entryPoint->offno,
														 &ep_slot, etup,
														 queryVec, optVecSize,
														 0.0, true,
														 &best, &ep_acc) && ep_acc))
				ep_loaded = true;
		}
	}

	if (unlikely(!ep_loaded))
	{
		if (unlikely(!HnswLoadFastCandSlow(entryPoint->blkno, entryPoint->offno,
										   q, index, support,
										   canOptDistance, queryVec, optVecSize,
										   0.0, true, &best)))
			elog(ERROR, "failed to load entry point element");
	}

	if (likely(entryPoint->level >= 1))
	{
		visited_hash upper_vh;
		uint64		upper_visited_buf[128];
		HnswUnvisited stack_unvisited[64];
		CalicoOptimisticSlot stack_slots[64];
		HnswElementTuple stack_etups[64];
		uint16		stack_limits[64];
		HnswUnvisited *unvisited;
		CalicoOptimisticSlot *unvisited_slots;
		HnswElementTuple *cand_etups;
		uint16	   *cand_limits;
		int			lm_max = HnswGetLayerM(m, 1);
		CalicoAsyncBatch *async_batch = NULL;
		const char *const buf_blocks = BufferBlocks;
		const bool	doPrefetch = hnsw_enable_prefetch;
		const Size	rawPrefetchBytes = doPrefetch ? optPrefetchBytes : 0;
		const bool	doStepPrefetch = (doPrefetch && canOptDistance && optPrefetchBytes > 0);
		const bool	use_static_visited = (canOptDistance && m <= 48);

		if (likely(lm_max <= lengthof(stack_unvisited)))
		{
			unvisited = stack_unvisited;
			unvisited_slots = stack_slots;
			cand_etups = stack_etups;
			cand_limits = stack_limits;
		}
		else
		{
			unvisited = palloc_array_checked(HnswUnvisited, lm_max);
			unvisited_slots = palloc_array_checked(CalicoOptimisticSlot, lm_max);
			cand_etups = palloc_array_checked(HnswElementTuple, lm_max);
			cand_limits = palloc_array_checked(uint16, lm_max);
		}

		upper_vh.tids.slots = NULL;
		upper_vh.tids.mask = 0;
		upper_vh.tids.count = 0;
		upper_vh.tids.grow_at = 0;
		upper_vh.tids.gen = 0;
		upper_vh.tids.on_stack = true;
		upper_vh.tids.alloc_ctx = CurrentMemoryContext;

		if (likely(use_static_visited))
		{
			upper_vh.tids.slots = hnsw_upper_visited_slots;
			upper_vh.tids.mask = lengthof(hnsw_upper_visited_slots) - 1;
			upper_vh.tids.grow_at = (lengthof(hnsw_upper_visited_slots) * 3) / 4;
		}

		for (int lc = entryPoint->level; lc >= 1; lc--)
		{
			int			lm = HnswGetLayerM(m, lc);
			bool		improved;

			if (likely(upper_vh.tids.slots == hnsw_upper_visited_slots))
			{
				hnsw_upper_visited_gen++;
				if (unlikely(hnsw_upper_visited_gen == 0))
				{
					memset(hnsw_upper_visited_slots, 0, sizeof(hnsw_upper_visited_slots));
					hnsw_upper_visited_gen = 1;
				}
				upper_vh.tids.gen = hnsw_upper_visited_gen;
				upper_vh.tids.count = 0;
			}
			else
			{
				hnsw_fast_visited_init(&upper_vh.tids, 1, m,
									   upper_visited_buf, lengthof(upper_visited_buf));
			}
			(void) hnsw_fast_visited_insert(&upper_vh.tids, best.blkno, best.offno);

			do
			{
				HnswFastCand c = best;
				int			unvisitedLength;

				improved = false;
				HnswLoadUnvisitedRaw(c.neighborPage, c.neighborOffno, c.level, c.version,
									 unvisited, &unvisitedLength, &upper_vh, index, support,
									 m, lm, lc, unvisited_slots,
									 rawPrefetchBytes,
									 minValidBytes,
									 canOptDistance ? cand_etups : NULL,
									 canOptDistance ? cand_limits : NULL,
									 &async_batch);

				if (likely(async_batch == NULL || !async_batch->active))
				{
					for (int i = 0; i < unvisitedLength; i++)
					{
						ItemPointer indextid = &unvisited[i].indextid;
						BlockNumber blkno = ItemPointerGetBlockNumber(indextid);
						OffsetNumber offno = ItemPointerGetOffsetNumber(indextid);
						HnswFastCand cand;
						bool		accepted = false;

						if (doStepPrefetch)
							HnswSteppedPrefetchAdvance(i, unvisitedLength, unvisited,
													   unvisited_slots, cand_etups,
													   cand_limits, optPrefetchBytes,
													   minValidBytes, buf_blocks);

						if (canOptDistance &&
							HnswLoadElementOptimisticSlotFast(support, blkno, offno, &unvisited_slots[i],
															  cand_etups[i],
															  queryVec, optVecSize,
															  best.distance, false,
															  &cand, &accepted))
						{
							if (accepted && likely(cand.level >= lc))
							{
								best = cand;
								improved = true;
							}
						}
						else if (HnswLoadFastCandSlow(blkno, offno, q, index, support,
													  canOptDistance, queryVec, optVecSize,
													  best.distance, false, &cand))
						{
							if (likely(cand.level >= lc))
							{
								best = cand;
								improved = true;
							}
						}
					}
				}
				else
				{
					HnswFastResultCand w_one;
					HnswFastCand tmp_c_stack[16];
					HnswFastCand *tmp_c_heap = tmp_c_stack;
					int			tmp_clen = 0;
					int			tmp_ccap = lengthof(tmp_c_stack);
					bool		tmp_c_on_stack = true;
					int			wlen = 1;

					w_one.distance = best.distance;
					w_one.blkno = best.blkno;
					w_one.offno = best.offno;
					w_one._pad = 0;

					HnswEvalAsyncUnvisitedBatchFast(q, 1, lc, index, support,
													canOptDistance, queryVec, optVecSize,
													optPrefetchBytes,
													unvisited, unvisitedLength,
													unvisited_slots, async_batch,
													&tmp_c_heap, &tmp_clen, &tmp_ccap, &tmp_c_on_stack,
													&w_one, &wlen);

					if (w_one.blkno != best.blkno ||
						w_one.offno != best.offno ||
						w_one.distance != best.distance)
					{
						bool		synced = false;

						for (int k = 0; k < tmp_clen; k++)
						{
							if (tmp_c_heap[k].blkno == w_one.blkno &&
								tmp_c_heap[k].offno == w_one.offno &&
								tmp_c_heap[k].distance == w_one.distance)
							{
								best = tmp_c_heap[k];
								synced = true;
								break;
							}
						}
						if (unlikely(!synced))
						{
							if (unlikely(!HnswLoadFastCandSlow(w_one.blkno, w_one.offno,
															   q, index, support,
															   canOptDistance, queryVec, optVecSize,
															   0.0, true, &best)))
								elog(ERROR, "failed to load upper layer candidate");
						}
						improved = true;
					}

					if (!tmp_c_on_stack)
						pfree(tmp_c_heap);
				}
			} while (improved);
		}

		if (upper_vh.tids.slots != NULL && !upper_vh.tids.on_stack)
			pfree(upper_vh.tids.slots);
		if (unlikely(async_batch != NULL))
			pfree(async_batch);
		if (unlikely(lm_max > lengthof(stack_unvisited)))
		{
			pfree(unvisited);
			pfree(unvisited_slots);
			pfree(cand_etups);
			pfree(cand_limits);
		}
	}

	*best_out = best;
}

bool
HnswSearchLayerFromEntryFastCands(char *base, HnswQuery * q, HnswElement entryPoint,
								  int ef, Relation index, HnswSupport * support, int m,
								  visited_hash * v, int64 *tuples,
								  HnswFastResultCand *inline_buf, int inline_cap,
								  HnswFastResultCand **fast_out, int *fast_count_out)
{
	bool		canOptDistance = false;
	const Vector *queryVec = NULL;
	Size		optVecSize = 0;
	Size		optPrefetchBytes = 0;
	HnswFastCand best;

	if (unlikely(index == NULL || ef <= 0 || entryPoint == NULL))
		return false;

	HnswInitOptDistanceParams(q, index, support, &canOptDistance, &queryVec,
							  &optVecSize, &optPrefetchBytes);

	HnswSearchLayerUpperDiskFast(q, entryPoint, index, support, m,
								 canOptDistance, queryVec, optVecSize, optPrefetchBytes,
								 &best);

	HnswSearchLayerDiskFastCore(base, q, NIL, &best, ef, 0, index, support, m,
								v, true, tuples,
								true, canOptDistance, queryVec, optVecSize, optPrefetchBytes,
								inline_buf, inline_cap, fast_out, fast_count_out);
	return true;
}
