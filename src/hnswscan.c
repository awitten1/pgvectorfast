#include "postgres.h"

#include <limits.h>

#include "access/genam.h"
#include "access/relscan.h"
#include "hnsw.h"
#include "lib/pairingheap.h"
#include "miscadmin.h"
#include "nodes/pg_list.h"
#include "pgstat.h"
#include "storage/lmgr.h"
#include "utils/float.h"
#include "utils/memutils.h"
#include "utils/relcache.h"
#include "utils/snapmgr.h"

#if PG_VERSION_NUM >= 160000
#include "varatt.h"
#endif

/*
 * Algorithm 5 from paper
 */
static List *
GetScanItems(IndexScanDesc scan, Datum value)
{
	HnswScanOpaque so = (HnswScanOpaque) scan->opaque;
	Relation	index = scan->indexRelation;
	HnswSupport *support = &so->support;
	List	   *ep;
	List	   *w;
	int			m;
	int			dimensions;
	HnswElement entryPoint;
	char	   *base = NULL;
	HnswQuery  *q = &so->q;

	/* Get m, dimensions, and entry point */
	HnswGetMetaPageInfoWithHandle(index, &support->read_handle, &m, &dimensions, &entryPoint);

	/* Check dimensions match index */
	if (DatumGetPointer(value) != NULL)
		HnswCheckDim(dimensions, so->typeInfo, support->collation, value);

	q->value = value;
	so->m = m;

	if (entryPoint == NULL)
		return NIL;

	if (hnsw_iterative_scan == HNSW_ITERATIVE_SCAN_OFF &&
		HnswSearchLayerFromEntryFastCands(base, q, entryPoint, hnsw_ef_search,
										  index, support, m,
										  &so->v, &so->tuples,
										  so->fast_inline, lengthof(so->fast_inline),
										  &so->fast_cands, &so->fast_count))
	{
		if (so->fast_count > 0)
		{
			int			n_win = Min(so->fast_count, HNSW_HEAP_PREFETCH_MAX);

			HnswLoadFastWindow(index, support, so->fast_cands, 0, n_win,
							   false, so->fast_win_tids, so->fast_win_len);
			so->fast_loaded_start = 0;
			so->fast_loaded_end = n_win;
			so->fast_idx = 0;
		}
		return NIL;
	}

	if (hnsw_iterative_scan != HNSW_ITERATIVE_SCAN_OFF &&
		support->vectorDistance != HNSW_VECTOR_DISTANCE_NONE)
	{
		HnswSearchIterativeFromEntry(q, entryPoint, hnsw_ef_search, index, support,
									 m, &so->v, &so->tuples, &so->iterative);
		return NIL;
	}

	ep = list_make1(HnswEntryCandidate(base, entryPoint, q, index, support, false));

	for (int lc = entryPoint->level; lc >= 1; lc--)
	{
		w = HnswSearchLayer(base, q, ep, 1, lc, index, support, m, false, NULL, NULL, NULL, true, NULL);
		ep = w;
	}

	if (hnsw_iterative_scan == HNSW_ITERATIVE_SCAN_OFF &&
		HnswSearchLayerFastCands(base, q, ep, hnsw_ef_search, 0, index, support, m,
								 &so->v, &so->tuples,
								 so->fast_inline, lengthof(so->fast_inline),
								 &so->fast_cands, &so->fast_count))
	{
		if (so->fast_count > 0)
		{
			int			n_win = Min(so->fast_count, HNSW_HEAP_PREFETCH_MAX);

			HnswLoadFastWindow(index, support, so->fast_cands, 0, n_win,
							   false, so->fast_win_tids, so->fast_win_len);
			so->fast_loaded_start = 0;
			so->fast_loaded_end = n_win;
			so->fast_idx = 0;
		}
		return NIL;
	}

	return HnswSearchLayer(base, q, ep, hnsw_ef_search, 0, index, support, m, false, NULL, &so->v, hnsw_iterative_scan != HNSW_ITERATIVE_SCAN_OFF ? &so->discarded : NULL, true, &so->tuples);
}

/*
 * Resume scan at ground level with discarded candidates
 */
static List *
ResumeScanItems(IndexScanDesc scan)
{
	HnswScanOpaque so = (HnswScanOpaque) scan->opaque;
	Relation	index = scan->indexRelation;
	List	   *ep = NIL;
	char	   *base = NULL;
	int			batch_size = hnsw_ef_search;

	if (pairingheap_is_empty(so->discarded))
		return NIL;

	/* Get next batch of candidates */
	for (int i = 0; i < batch_size; i++)
	{
		HnswSearchCandidate *sc;

		if (pairingheap_is_empty(so->discarded))
			break;

		sc = HnswGetSearchCandidate(w_node, pairingheap_remove_first(so->discarded));

		ep = lappend(ep, sc);
	}

	return HnswSearchLayer(base, &so->q, ep, batch_size, 0, index, &so->support, so->m, false, NULL, &so->v, &so->discarded, false, &so->tuples);
}

/*
 * Get scan value
 */
static Datum
GetScanValue(IndexScanDesc scan)
{
	HnswScanOpaque so = (HnswScanOpaque) scan->opaque;
	Datum		value;

	if (scan->orderByData->sk_flags & SK_ISNULL)
		value = PointerGetDatum(NULL);
	else
	{
		value = scan->orderByData->sk_argument;

		/* Value should not be compressed or toasted */
		Assert(!VARATT_IS_COMPRESSED(DatumGetPointer(value)));
		Assert(!VARATT_IS_EXTENDED(DatumGetPointer(value)));

		/* Normalize if needed */
		if (so->support.normprocinfo != NULL)
			value = HnswNormValue(so->typeInfo, so->support.collation, value);
	}

	return value;
}

#if defined(HNSW_MEMORY)
/*
 * Show memory usage
 */
static void
ShowMemoryUsage(HnswScanOpaque so)
{
	elog(INFO, "memory: %zu KB, tuples: " INT64_FORMAT, MemoryContextMemAllocated(so->tmpCtx, false) / 1024, so->tuples);
}
#endif

static inline void
HnswPrefetchHeapTuplePayloadSlot(const char *buf_blocks,
								 const CalicoOptimisticSlot *slot,
								 OffsetNumber offno)
{
	Page		hpage;
	ItemIdData	item;
	uint16		toff;
	uint16		tlen;
	const char *p;

	if (!slot->opt_valid)
		return;
	if (unlikely(offno < FirstOffsetNumber ||
				 offno > (BLCKSZ - SizeOfPageHeaderData) / sizeof(ItemIdData)))
		return;

	hpage = (Page) (buf_blocks + ((Size) slot->frame_id) * BLCKSZ);
	item = *(volatile ItemIdData *) PageGetItemId(hpage, offno);
	if (!ItemIdIsNormal(&item))
		return;

	toff = ItemIdGetOffset(&item);
	tlen = ItemIdGetLength(&item);
	if (unlikely(toff < SizeOfPageHeaderData || tlen == 0 || toff + tlen > BLCKSZ))
		return;

	p = (const char *) hpage + toff;
	if (tlen <= 448)
	{
		uintptr_t	cl = (uintptr_t) p & ~(uintptr_t) 63;
		uintptr_t	last_cl = (uintptr_t) (p + tlen - 1) & ~(uintptr_t) 63;

		for (; cl <= last_cl; cl += 64)
			__builtin_prefetch((const void *) cl, 0, 3);
	}
	else
	{
		uintptr_t	cl = (uintptr_t) p & ~(uintptr_t) 63;
		uintptr_t	head_last_cl = (uintptr_t) (p + 192 - 1) & ~(uintptr_t) 63;
		uintptr_t	tail_cl = (uintptr_t) (p + tlen - 256) & ~(uintptr_t) 63;
		uintptr_t	tail_last_cl = (uintptr_t) (p + tlen - 1) & ~(uintptr_t) 63;

		for (; cl <= head_last_cl; cl += 64)
			__builtin_prefetch((const void *) cl, 0, 3);
		if (tail_cl <= head_last_cl)
			tail_cl = head_last_cl + 64;
		for (; tail_cl <= tail_last_cl; tail_cl += 64)
			__builtin_prefetch((const void *) tail_cl, 0, 3);
	}
}

static inline void
HnswPrefetchHeapSlotMeta(const char *buf_blocks,
						 const CalicoOptimisticSlot *slot,
						 OffsetNumber offno)
{
	Page		hpage;

	if (!slot->resident)
		return;

	if (slot->entry_ptr != NULL)
		__builtin_prefetch(slot->entry_ptr, 0, 3);
	__builtin_prefetch(GetBufferDescriptor(slot->frame_id), 1, 3);
	if (!slot->opt_valid)
		return;

	hpage = (Page) (buf_blocks + ((Size) slot->frame_id) * BLCKSZ);
	__builtin_prefetch(hpage, 0, 3);
	if (likely(offno >= FirstOffsetNumber &&
			   offno <= (BLCKSZ - SizeOfPageHeaderData) / sizeof(ItemIdData)) &&
		offsetof(PageHeaderData, pd_linp) + (offno - 1) * sizeof(ItemIdData) >= 64)
	{
		__builtin_prefetch(PageGetItemId(hpage, offno), 0, 3);
	}
}

static bool
HnswReadMissingHeapBlocksAsync(Relation heapRel, HnswScanOpaque so,
								 const BlockNumber *blks,
								 const CalicoOptimisticSlot *slots, int n)
{
	BlockNumber missing[HNSW_ITER_HEAP_PREFETCH_MAX];
	int			n_missing = 0;
	bool		refreshed = false;

	for (int j = 0; j < n; j++)
	{
		if (!slots[j].resident)
		{
			BlockNumber blk = blks[j];

			if (so->heap_nblocks == InvalidBlockNumber)
				so->heap_nblocks = RelationGetNumberOfBlocks(heapRel);
			if (blk >= so->heap_nblocks && !refreshed)
			{
				so->heap_nblocks = RelationGetNumberOfBlocks(heapRel);
				refreshed = true;
			}
			if (blk < so->heap_nblocks)
				missing[n_missing++] = blk;
		}
	}

	if (n_missing > 0)
	{
		CalicoReadBuffersAsync(RelationGetSmgr(heapRel),
							   heapRel->rd_rel->relpersistence,
							   MAIN_FORKNUM,
							   missing,
							   n_missing);
		return true;
	}

	return false;
}

static void
HnswPrefetchIterativeHeapWindow(IndexScanDesc scan, HnswScanOpaque so)
{
	Relation	heapRel = scan->heapRelation;
	HnswIterativeState *state = so->iterative;
	BlockNumber blks[HNSW_ITER_HEAP_PREFETCH_MAX];
	CalicoOptimisticSlot tmp_slots[HNSW_ITER_HEAP_PREFETCH_MAX];
	uint8		map_idx[HNSW_ITER_HEAP_PREFETCH_MAX];
	int			idx;
	int			end;
	int			cnt;
	int			max_win;
	int			n = 0;
	bool		all_resident;

	if (state == NULL || state->result_idx >= state->result_count)
		return;

	idx = state->result_idx;
	max_win = (so->heap_io_window > 0) ? (int) so->heap_io_window : HNSW_ITER_HEAP_PREFETCH_MAX;
	end = Min(state->result_count, idx + max_win);
	cnt = end - idx;
	state->pref_start = idx;
	state->pref_end = end;

	for (int k = 0; k < cnt; k++)
	{
		const HnswIterativeNode *node;

		if (k + 8 < cnt)
			__builtin_prefetch(HnswIterativeGetNode(state, state->results[idx + k + 8].id), 0, 0);
		node = HnswIterativeGetNode(state, state->results[idx + k].id);
		state->pref_htid_meta[k] = node->htid_meta;
		state->pref_first_htids[k] = node->first_htid;
	}

	if (!calico_enable_optimistic_reads || !hnsw_enable_prefetch ||
		heapRel == NULL || RelationUsesLocalBuffers(heapRel) || !calico_is_active())
	{
		for (int k = 0; k < cnt; k++)
		{
			state->pref_slots[k].resident = false;
			state->pref_slots[k].opt_valid = false;
			state->pref_offs[k] = InvalidOffsetNumber;
		}
		return;
	}

	if (!so->heap_handle_inited)
	{
		calico_read_handle_init(&so->heap_handle, &heapRel->rd_locator, MAIN_FORKNUM);
		so->heap_handle_inited = true;
	}

	for (int k = 0; k < cnt; k++)
	{
		uint32		htid_meta = state->pref_htid_meta[k];
		uint8		n_ht = (uint8) (htid_meta >> 24);

		state->pref_slots[k].resident = false;
		state->pref_slots[k].opt_valid = false;
		state->pref_offs[k] = InvalidOffsetNumber;
		if (likely(n_ht > 0))
		{
			const ItemPointerData *tid = (n_ht == 1)
				? &state->pref_first_htids[k]
				: &state->extra_htids[(htid_meta & 0x00ffffffU) + n_ht - 2U];

			if (ItemPointerIsValid(tid))
			{
				BlockNumber blk = ItemPointerGetBlockNumber(tid);

				calico_optimistic_prefetch_block(&so->heap_handle, blk);
				blks[n] = blk;
				state->pref_offs[k] = ItemPointerGetOffsetNumber(tid);
				map_idx[n] = (uint8) k;
				n++;
			}
		}
	}

	if (n <= 0)
		return;

	all_resident = calico_optimistic_lookup_batch_slots_noprefetch(&so->heap_handle, blks, n, tmp_slots);
	if (unlikely(!all_resident))
	{
		if (so->heap_io_window == 0)
		{
			so->heap_io_window = 8;
			if (cnt > 8)
			{
				cnt = 8;
				state->pref_end = idx + cnt;
				while (n > 0 && map_idx[n - 1] >= cnt)
					n--;
			}
		}
		if (HnswReadMissingHeapBlocksAsync(heapRel, so, blks, tmp_slots, n))
			calico_optimistic_lookup_batch_slots_noprefetch(&so->heap_handle, blks, n, tmp_slots);
		so->heap_io_window = (uint8) Min((int) so->heap_io_window * 2,
										   HNSW_ITER_HEAP_PREFETCH_MAX);
	}
	for (int j = 0; j < n; j++)
		state->pref_slots[map_idx[j]] = tmp_slots[j];

	{
		const char *const buf_blocks = BufferBlocks;
		int			meta_prime = Min(cnt, 4);
		int			pay_prime = Min(cnt, 2);

		for (int k = 0; k < meta_prime; k++)
			HnswPrefetchHeapSlotMeta(buf_blocks, &state->pref_slots[k], state->pref_offs[k]);
		for (int k = 0; k < pay_prime; k++)
			HnswPrefetchHeapTuplePayloadSlot(buf_blocks, &state->pref_slots[k], state->pref_offs[k]);
	}
}

static void
HnswPrefetchHeapResults(IndexScanDesc scan, HnswScanOpaque so)
{
	Relation	heapRel = scan->heapRelation;
	char	   *base = NULL;
	int			wlen;
	int			npref;
	BlockNumber blks[HNSW_HEAP_PREFETCH_MAX];
	int			n = 0;
	int			k;

	if (so->iterative != NULL)
	{
		HnswPrefetchIterativeHeapWindow(scan, so);
		return;
	}

	if (!calico_enable_optimistic_reads || !hnsw_enable_prefetch ||
		heapRel == NULL || RelationUsesLocalBuffers(heapRel) || !calico_is_active())
		return;

	if (!so->heap_handle_inited)
	{
		calico_read_handle_init(&so->heap_handle, &heapRel->rd_locator, MAIN_FORKNUM);
		so->heap_handle_inited = true;
	}

	if (so->fast_cands != NULL)
	{
		int			win_cnt = so->fast_loaded_end - so->fast_loaded_start;

		npref = Min(win_cnt, HNSW_HEAP_PREFETCH_MAX);
		for (k = 0; k < npref; k++)
		{
			uint8		len = so->fast_win_len[k];

			if (len > 0)
			{
				ItemPointer tid = &so->fast_win_tids[k][len - 1];

				if (ItemPointerIsValid(tid))
				{
					BlockNumber blk = ItemPointerGetBlockNumber(tid);

					calico_optimistic_prefetch_block(&so->heap_handle, blk);
					blks[n] = blk;
					so->heap_offs[n] = ItemPointerGetOffsetNumber(tid);
					n++;
				}
			}
		}
	}
	else
	{
		wlen = list_length(so->w);
		if (wlen <= 0)
			return;

		npref = Min(wlen, HNSW_HEAP_PREFETCH_MAX);
		for (k = 0; k < npref; k++)
		{
			HnswSearchCandidate *cand = (HnswSearchCandidate *) list_nth(so->w, wlen - 1 - k);
			HnswElement el = HnswPtrAccess(base, cand->element);

			if (el->heaptidsLength > 0)
			{
				ItemPointer tid = &el->heaptids[el->heaptidsLength - 1];

				if (ItemPointerIsValid(tid))
				{
					BlockNumber blk = ItemPointerGetBlockNumber(tid);

					calico_optimistic_prefetch_block(&so->heap_handle, blk);
					blks[n] = blk;
					so->heap_offs[n] = ItemPointerGetOffsetNumber(tid);
					n++;
				}
			}
		}
	}

	if (n <= 0)
		return;

	{
		bool		all_resident;

		all_resident = calico_optimistic_lookup_batch_slots_noprefetch(&so->heap_handle,
																	   blks, n, so->heap_slots);
		if (unlikely(!all_resident) &&
			HnswReadMissingHeapBlocksAsync(heapRel, so, blks, so->heap_slots, n))
			calico_optimistic_lookup_batch_slots_noprefetch(&so->heap_handle,
															blks, n, so->heap_slots);
	}
	so->heap_pref_count = (uint8) n;
	so->heap_pref_idx = 0;
	{
		const char *const buf_blocks = BufferBlocks;
		int			meta_prime = Min(n, 4);
		int			pay_prime = Min(n, 2);

		for (k = 0; k < meta_prime; k++)
			HnswPrefetchHeapSlotMeta(buf_blocks, &so->heap_slots[k], so->heap_offs[k]);
		for (k = 0; k < pay_prime; k++)
			HnswPrefetchHeapTuplePayloadSlot(buf_blocks, &so->heap_slots[k], so->heap_offs[k]);
	}
}

/*
 * Prepare for an index scan
 */
IndexScanDesc
hnswbeginscan(Relation index, int nkeys, int norderbys)
{
	IndexScanDesc scan;
	HnswScanOpaque so;
	double		maxMemory;

	scan = RelationGetIndexScan(index, nkeys, norderbys);

	so = (HnswScanOpaque) palloc(sizeof(HnswScanOpaqueData));
	memset(so, 0, offsetof(HnswScanOpaqueData, fast_inline));
	so->heap_nblocks = InvalidBlockNumber;
	so->typeInfo = HnswGetTypeInfo(index);

	/* Set support functions */
	HnswInitSupport(&so->support, index);

	/*
	 * Use a lower max allocation size than default to allow scanning more
	 * tuples for iterative search before exceeding work_mem
	 */
	so->tmpCtx = AllocSetContextCreate(CurrentMemoryContext,
									   "Hnsw scan temporary context",
									   0, 8 * 1024, 256 * 1024);

	HnswAcquireScanVisited(&so->v.tids, CurrentMemoryContext, so->tmpCtx,
						   &so->visited_cb, &so->visited_cb_registered);

	/* Calculate max memory */
	/* Add 256 extra bytes to fill last block when close */
	maxMemory = (double) work_mem * hnsw_scan_mem_multiplier * 1024.0 + 256;
	so->maxMemory = Min(maxMemory, (double) (SIZE_MAX / 2));

	scan->opaque = so;

	return scan;
}

/*
 * Start or restart an index scan
 */
void
hnswrescan(IndexScanDesc scan, ScanKey keys, int nkeys, ScanKey orderbys, int norderbys)
{
	HnswScanOpaque so = (HnswScanOpaque) scan->opaque;

	so->first = true;
	so->heap_nblocks = InvalidBlockNumber;
	so->heap_pref_count = 0;
	so->heap_pref_idx = 0;
	so->heap_io_window = 0;
	so->fast_cands = NULL;
	so->fast_count = 0;
	so->fast_idx = 0;
	so->fast_loaded_start = 0;
	so->fast_loaded_end = 0;
	/* v's visited slots persist across rescans; discarded is allocated in tmpCtx */
	so->v.tids.count = 0;
	so->discarded = NULL;
	so->iterative = NULL;
	so->tuples = 0;
	so->previousDistance = -get_float8_infinity();
	MemoryContextReset(so->tmpCtx);
	so->visited_cb_registered = false;
	HnswAcquireScanVisited(&so->v.tids, CurrentMemoryContext, so->tmpCtx,
						   &so->visited_cb, &so->visited_cb_registered);

	if (keys && scan->numberOfKeys > 0)
		memmove(scan->keyData, keys, scan->numberOfKeys * sizeof(ScanKeyData));

	if (orderbys && scan->numberOfOrderBys > 0)
		memmove(scan->orderByData, orderbys, scan->numberOfOrderBys * sizeof(ScanKeyData));
}

/*
 * Fetch the next tuple in the given scan
 */
bool
hnswgettuple(IndexScanDesc scan, ScanDirection dir)
{
	HnswScanOpaque so = (HnswScanOpaque) scan->opaque;
	MemoryContext oldCtx = MemoryContextSwitchTo(so->tmpCtx);

	/*
	 * Index can be used to scan backward, but Postgres doesn't support
	 * backward scan on operators
	 */
	Assert(ScanDirectionIsForward(dir));

	if (so->first)
	{
		Datum		value;

		/* Count index scan for stats */
		pgstat_count_index_scan(scan->indexRelation);
#if PG_VERSION_NUM >= 180000
		if (scan->instrument)
			scan->instrument->nsearches++;
#endif

		/* Safety check */
		if (scan->orderByData == NULL)
			elog(ERROR, "cannot scan hnsw index without order");

		/* Requires MVCC-compliant snapshot as not able to maintain a pin */
		/* https://www.postgresql.org/docs/current/index-locking.html */
		if (!IsMVCCSnapshot(scan->xs_snapshot))
			elog(ERROR, "non-MVCC snapshots are not supported with hnsw");

		/* Get scan value */
		value = GetScanValue(scan);

		/*
		 * Get a shared lock. This allows vacuum to ensure no in-flight scans
		 * before marking tuples as deleted.
		 */
		LockPage(scan->indexRelation, HNSW_SCAN_LOCK, ShareLock);

		so->w = GetScanItems(scan, value);

		/* Release shared lock */
		UnlockPage(scan->indexRelation, HNSW_SCAN_LOCK, ShareLock);

		HnswPrefetchHeapResults(scan, so);
		so->first = false;

#if defined(HNSW_MEMORY)
		ShowMemoryUsage(so);
#endif
	}

	for (;;)
	{
		char	   *base = NULL;
		HnswSearchCandidate *sc;
		HnswElement element;
		const ItemPointerData *heaptid;

		if (so->fast_cands != NULL)
		{
			int			wslot;

			if (so->fast_idx >= so->fast_loaded_end)
			{
				int			rem = so->fast_count - so->fast_idx;
				int			next_cnt;

				if (rem <= 0)
					break;

				next_cnt = Min(HNSW_HEAP_PREFETCH_MAX, rem);
				LockPage(scan->indexRelation, HNSW_SCAN_LOCK, ShareLock);
				HnswLoadFastWindow(scan->indexRelation, &so->support,
								   so->fast_cands, so->fast_idx, next_cnt, true,
								   so->fast_win_tids, so->fast_win_len);
				UnlockPage(scan->indexRelation, HNSW_SCAN_LOCK, ShareLock);
				so->fast_loaded_start = so->fast_idx;
				so->fast_loaded_end = so->fast_idx + next_cnt;
				so->heap_pref_count = 0;
				so->heap_pref_idx = 0;
				HnswPrefetchHeapResults(scan, so);
			}

			wslot = so->fast_idx - so->fast_loaded_start;
			if (so->fast_win_len[wslot] == 0)
			{
				so->fast_idx++;
				continue;
			}

			heaptid = &so->fast_win_tids[wslot][--so->fast_win_len[wslot]];
			if (so->fast_win_len[wslot] == 0)
				so->fast_idx++;
		}
		else if (so->iterative != NULL)
		{
			HnswIterativeState *state = so->iterative;
			uint32		htid_meta;
			bool		was_first = false;
			int			wslot;

			if (state->result_idx >= state->result_count)
			{
				bool		expand;

				if (HnswIterativeEmpty(state))
					break;

				expand = so->tuples < hnsw_max_scan_tuples &&
					MemoryContextMemAllocated(so->tmpCtx, false) <= so->maxMemory;
				if (expand)
					LockPage(scan->indexRelation, HNSW_SCAN_LOCK, ShareLock);
				HnswSearchIterativeBatch(&so->q,
										 expand ? hnsw_ef_search : HNSW_ITER_DRAIN_BATCH,
										 scan->indexRelation, &so->support,
										 so->m, &so->v, &so->tuples, state, expand);
				if (expand)
					UnlockPage(scan->indexRelation, HNSW_SCAN_LOCK, ShareLock);

				state->pref_start = 0;
				state->pref_end = 0;
				state->cur_rem_htids = 0;
				if (state->result_count == 0)
					break;
				HnswPrefetchIterativeHeapWindow(scan, so);
			}
			else if (state->result_idx >= state->pref_end)
			{
				HnswPrefetchIterativeHeapWindow(scan, so);
			}

			wslot = state->result_idx - state->pref_start;
			Assert(wslot >= 0 && state->result_idx < state->pref_end);
			htid_meta = state->pref_htid_meta[wslot];
			if (state->cur_rem_htids == 0)
			{
				uint8		n_htids = (uint8) (htid_meta >> 24);

				if (n_htids == 0)
				{
					state->result_idx++;
					continue;
				}
				if (hnsw_iterative_scan == HNSW_ITERATIVE_SCAN_STRICT)
				{
					double		dist = (double) state->results[state->result_idx].distance;

					if (dist < so->previousDistance)
					{
						state->result_idx++;
						continue;
					}
					so->previousDistance = dist;
				}
				state->cur_rem_htids = n_htids;
				was_first = true;
			}

			if (state->cur_rem_htids > 1)
			{
				uint32		extra_off = htid_meta & 0x00ffffffU;

				heaptid = &state->extra_htids[extra_off + state->cur_rem_htids - 2U];
				state->cur_rem_htids--;
			}
			else
			{
				heaptid = &state->pref_first_htids[wslot];
				state->cur_rem_htids = 0;
				state->result_idx++;
			}

			MemoryContextSwitchTo(oldCtx);

			scan->xs_heaptid = *heaptid;
			scan->xs_recheck = false;
			scan->xs_recheckorderby = false;
			if (was_first)
			{
				int			wlen = state->pref_end - state->pref_start;
				const char *const buf_blocks = BufferBlocks;

				if (wslot + 4 < wlen)
					HnswPrefetchHeapSlotMeta(buf_blocks,
											 &state->pref_slots[wslot + 4],
											 state->pref_offs[wslot + 4]);
				if (wslot + 2 < wlen)
					HnswPrefetchHeapTuplePayloadSlot(buf_blocks,
													 &state->pref_slots[wslot + 2],
													 state->pref_offs[wslot + 2]);
			}
			return true;
		}
		else
		{
			if (list_length(so->w) == 0)
			{
				if (hnsw_iterative_scan == HNSW_ITERATIVE_SCAN_OFF)
					break;

				/* Empty index */
				if (so->discarded == NULL)
					break;

				/* Reached max number of tuples or memory limit */
				if (so->tuples >= hnsw_max_scan_tuples || MemoryContextMemAllocated(so->tmpCtx, false) > so->maxMemory)
				{
					if (pairingheap_is_empty(so->discarded))
						break;

					/* Return remaining tuples */
					so->w = lappend(so->w, HnswGetSearchCandidate(w_node, pairingheap_remove_first(so->discarded)));
				}
				else
				{
					/*
					 * Locking ensures when neighbors are read, the elements they
					 * reference will not be deleted (and replaced) during the
					 * iteration.
					 *
					 * Elements loaded into memory on previous iterations may have
					 * been deleted (and replaced), so when reading neighbors, the
					 * element version must be checked.
					 */
					LockPage(scan->indexRelation, HNSW_SCAN_LOCK, ShareLock);

					so->w = ResumeScanItems(scan);

					UnlockPage(scan->indexRelation, HNSW_SCAN_LOCK, ShareLock);

#if defined(HNSW_MEMORY)
					ShowMemoryUsage(so);
#endif
				}
				so->heap_pref_count = 0;
				so->heap_pref_idx = 0;
				HnswPrefetchHeapResults(scan, so);

				if (list_length(so->w) == 0)
					break;
			}

			sc = llast(so->w);
			element = HnswPtrAccess(base, sc->element);

			/* Move to next element if no valid heap TIDs */
			if (element->heaptidsLength == 0)
			{
				so->w = list_delete_last(so->w);

				/* Mark memory as free for next iteration */
				if (hnsw_iterative_scan != HNSW_ITERATIVE_SCAN_OFF)
				{
					pfree(element);
					pfree(sc);
				}

				continue;
			}

			heaptid = &element->heaptids[--element->heaptidsLength];

			if (hnsw_iterative_scan == HNSW_ITERATIVE_SCAN_STRICT)
			{
				if (sc->distance < so->previousDistance)
					continue;

				so->previousDistance = sc->distance;
			}
		}

		MemoryContextSwitchTo(oldCtx);

		scan->xs_heaptid = *heaptid;
		scan->xs_recheck = false;
		scan->xs_recheckorderby = false;
		if (so->heap_pref_idx < so->heap_pref_count)
		{
			uint8		cur = so->heap_pref_idx++;
			const char *const buf_blocks = BufferBlocks;

			if (cur + 4 < so->heap_pref_count)
				HnswPrefetchHeapSlotMeta(buf_blocks, &so->heap_slots[cur + 4], so->heap_offs[cur + 4]);
			if (cur + 2 < so->heap_pref_count)
				HnswPrefetchHeapTuplePayloadSlot(buf_blocks, &so->heap_slots[cur + 2], so->heap_offs[cur + 2]);
		}
		return true;
	}

	MemoryContextSwitchTo(oldCtx);
	return false;
}

/*
 * End a scan and release resources
 */
void
hnswendscan(IndexScanDesc scan)
{
	HnswScanOpaque so = (HnswScanOpaque) scan->opaque;

	MemoryContextDelete(so->tmpCtx);
	HnswReleaseScanVisited(&so->v.tids);

	pfree(so);
	scan->opaque = NULL;
}
