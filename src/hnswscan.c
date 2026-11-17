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

	if (hnsw_iterative_scan != HNSW_ITERATIVE_SCAN_OFF &&
		support->vectorDistance != HNSW_VECTOR_DISTANCE_NONE)
		return HnswSearchIterative(q, ep, hnsw_ef_search, index, support, m,
								  &so->v, &so->tuples, &so->iterative, true);

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
HnswPrefetchHeapTuplePayloadSlot(const CalicoOptimisticSlot *slot, OffsetNumber offno)
{
	Page		hpage;
	ItemId		lpp;
	uint16		toff;
	uint16		tlen;

	if (!slot->opt_valid)
		return;
	if (unlikely(offno < FirstOffsetNumber ||
				 offno > (BLCKSZ - SizeOfPageHeaderData) / sizeof(ItemIdData)))
		return;

	hpage = CalicoFramePage(slot->frame_id);
	lpp = PageGetItemId(hpage, offno);
	if (!ItemIdIsNormal(lpp))
		return;

	toff = ItemIdGetOffset(lpp);
	tlen = ItemIdGetLength(lpp);
	if (unlikely(toff < SizeOfPageHeaderData || tlen == 0 || toff + tlen > BLCKSZ))
		return;

	{
		const char *p = (const char *) hpage + toff;
		Size		limit = Min((Size) tlen, (Size) 1024);
		Size		off;

		for (off = 0; off < limit; off += 64)
			__builtin_prefetch(p + off, 0, 3);
		if (limit > 0 &&
			((uintptr_t) (p + limit - 1) >> 6) != ((uintptr_t) (p + off - 64) >> 6))
			__builtin_prefetch(p + limit - 1, 0, 3);
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

	calico_optimistic_lookup_batch_slots_noprefetch(&so->heap_handle, blks, n, so->heap_slots);
	so->heap_pref_count = (uint8) n;
	so->heap_pref_idx = 0;
	for (k = 0; k < n; k++)
	{
		if (so->heap_slots[k].opt_valid &&
			likely(so->heap_offs[k] >= FirstOffsetNumber &&
				   so->heap_offs[k] <= (BLCKSZ - SizeOfPageHeaderData) / sizeof(ItemIdData)))
		{
			Page		hpage = CalicoFramePage(so->heap_slots[k].frame_id);

			__builtin_prefetch(PageGetItemId(hpage, so->heap_offs[k]), 0, 3);
		}
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
	so->heap_pref_count = 0;
	so->heap_pref_idx = 0;
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
		ItemPointer heaptid;

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
		else
		{
			if (list_length(so->w) == 0)
			{
				if (hnsw_iterative_scan == HNSW_ITERATIVE_SCAN_OFF)
					break;

				if (so->iterative != NULL)
				{
					bool expand;

					if (HnswIterativeEmpty(so->iterative))
						break;
					expand = so->tuples < hnsw_max_scan_tuples &&
						MemoryContextMemAllocated(so->tmpCtx, false) <= so->maxMemory;
					if (expand)
						LockPage(scan->indexRelation, HNSW_SCAN_LOCK, ShareLock);
					so->w = HnswSearchIterative(&so->q, NIL, expand ? hnsw_ef_search : 1,
						scan->indexRelation, &so->support, so->m, &so->v,
						&so->tuples, &so->iterative, expand);
					if (expand)
						UnlockPage(scan->indexRelation, HNSW_SCAN_LOCK, ShareLock);
				}
				else
				{
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

			if (cur > 0)
			{
				if (cur == 1)
					HnswPrefetchHeapTuplePayloadSlot(&so->heap_slots[1], so->heap_offs[1]);
				if (cur + 1 < so->heap_pref_count)
					HnswPrefetchHeapTuplePayloadSlot(&so->heap_slots[cur + 1], so->heap_offs[cur + 1]);
			}
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
