/* Calico batch I/O and result-window loading for HNSW. */
#include "postgres.h"

#include "hnswio.h"
#include "storage/bufmgr.h"
#include "storage/smgr.h"

pg_noinline bool
HnswLoadUnvisitedMissingOrPinSlow(Relation index, HnswSupport * support,
								  const BlockNumber *batch_blocknums, int count,
								  const HnswUnvisited * unvisited,
								  CalicoOptimisticSlot *slots_out,
								  Size tuple_prefetch_bytes,
								  CalicoAsyncBatch **batch_out_ptr)
{
	BlockNumber missing_blocks[HNSW_MAX_M * 2];
	int			missing_count = 0;
	int			res_indices[HNSW_MAX_M * 2];
	int			res_count = 0;

	if (support == NULL)
	{
		uint32		batch_frameids[HNSW_MAX_M * 2];
		bool		batch_found[HNSW_MAX_M * 2];

		CalicoLookupAndOptimisticPinBatch(RelationGetSmgr(index),
										  MAIN_FORKNUM,
										  batch_blocknums,
										  count,
										  batch_frameids,
										  batch_found);
		for (int j = 0; j < count; j++)
		{
			slots_out[j].entry_ptr = NULL;
			slots_out[j].version = 0;
			slots_out[j].frame_id = batch_frameids[j];
			slots_out[j].resident = batch_found[j];
			slots_out[j].opt_valid = false;
		}
	}

	for (int j = 0; j < count; j++)
	{
		if (slots_out[j].resident)
		{
			Page		prefetch_page = BufferGetPage(slots_out[j].frame_id + 1);
			OffsetNumber offno = ItemPointerGetOffsetNumber(&unvisited[j].indextid);

			res_indices[res_count++] = j;
			if (likely(offno >= FirstOffsetNumber &&
					   offno <= (BLCKSZ - SizeOfPageHeaderData) / sizeof(ItemIdData)))
				__builtin_prefetch(PageGetItemId(prefetch_page, offno), 0, 3);
		}
		else
			missing_blocks[missing_count++] = batch_blocknums[j];
	}

	if (missing_count > 0)
	{
		if (batch_out_ptr != NULL)
		{
			CalicoAsyncBatch *batch;

			if (*batch_out_ptr == NULL)
				*batch_out_ptr = (CalicoAsyncBatch *) palloc(sizeof(CalicoAsyncBatch));
			batch = *batch_out_ptr;
			batch->active = false;
			batch->num_ranges = 0;

			CalicoAsyncBatchStart(batch,
								  RelationGetSmgr(index),
								  index->rd_rel->relpersistence,
								  MAIN_FORKNUM,
								  missing_blocks,
								  missing_count);

			if (batch->active)
			{
				/*
				 * I/O is in flight; prime prefetch ONLY for resident neighbors
				 * so Phase 1 can evaluate them immediately in parallel.
				 */
				if (tuple_prefetch_bytes > 0 && res_count > 0)
				{
					int			prime = Min(res_count, HNSW_PREFETCH_AHEAD);

					for (int k = 0; k < prime; k++)
					{
						int			idx = res_indices[k];

						HnswPrefetchSlotTuple(&slots_out[idx],
											  ItemPointerGetOffsetNumber(&unvisited[idx].indextid),
											  tuple_prefetch_bytes);
					}
				}
				return true;
			}
		}
		else
		{
			CalicoReadBuffersAsync(RelationGetSmgr(index),
								   index->rd_rel->relpersistence,
								   MAIN_FORKNUM,
								   missing_blocks,
								   missing_count);
		}

		/* Refresh slots for newly fetched blocks so search uses fast inline path + prefetch */
		if (support != NULL)
		{
			for (int j = 0; j < count; j++)
			{
				if (!slots_out[j].resident)
				{
					bool		found = false;
					uint64		version = 0;
					uint32		buf_id = 0;
					void	   *entry_ptr = NULL;

					found = calico_optimistic_lookup_handle_frame(&support->read_handle,
																  batch_blocknums[j],
																  &version,
																  &buf_id,
																  &entry_ptr);
					if (found)
					{
						slots_out[j].entry_ptr = (CalicoTranslationEntry *) entry_ptr;
						slots_out[j].version = version;
						slots_out[j].frame_id = buf_id;
						slots_out[j].resident = true;
						slots_out[j].opt_valid = true;
					}
				}
			}
		}
	}

	return false;
}

void
HnswLoadFastWindow(Relation index, HnswSupport * support,
				   const HnswFastResultCand *cands, int start_idx, int count,
				   bool allow_deleted,
				   ItemPointerData win_tids[][HNSW_HEAPTIDS], uint8 *win_len)
{
	bool		use_opt_mat;
	CalicoOptimisticSlot mat_slots[HNSW_HEAP_PREFETCH_MAX];

	if (count <= 0)
		return;

	Assert(count <= HNSW_HEAP_PREFETCH_MAX);

	use_opt_mat = (calico_enable_optimistic_reads &&
				   support != NULL &&
				   calico_is_active() &&
				   !RelationUsesLocalBuffers(index));

	if (use_opt_mat)
	{
		BlockNumber miss_blks[HNSW_HEAP_PREFETCH_MAX];
		uint8		miss_idx[HNSW_HEAP_PREFETCH_MAX];
		int			n_miss = 0;

		for (int k = 0; k < count; k++)
			calico_optimistic_prefetch_block(&support->read_handle,
											 cands[start_idx + k].blkno);

		for (int k = 0; k < count; k++)
		{
			BlockNumber blk = cands[start_idx + k].blkno;
			OffsetNumber off = cands[start_idx + k].offno;
			uint64		version;
			uint32		frame_id;
			void	   *entry_ptr;
			bool		off_ok = (off >= FirstOffsetNumber &&
								  off <= (BLCKSZ - SizeOfPageHeaderData) / sizeof(ItemIdData));
			bool		found = (off_ok &&
								 calico_optimistic_lookup_handle_frame(&support->read_handle,
																	   blk, &version,
																	   &frame_id, &entry_ptr));

			mat_slots[k].opt_valid = found;
			mat_slots[k].frame_id = frame_id;
			mat_slots[k].version = version;
			mat_slots[k].entry_ptr = (CalicoTranslationEntry *) entry_ptr;
			if (unlikely(off_ok && !found))
			{
				miss_blks[n_miss] = blk;
				miss_idx[n_miss] = (uint8) k;
				n_miss++;
			}
		}

		if (unlikely(n_miss > 0))
		{
			CalicoReadBuffersAsync(RelationGetSmgr(index),
								   index->rd_rel->relpersistence,
								   MAIN_FORKNUM,
								   miss_blks,
								   n_miss);
			for (int j = 0; j < n_miss; j++)
			{
				int			k = miss_idx[j];
				uint64		version;
				uint32		frame_id;
				void	   *entry_ptr;

				if (calico_optimistic_lookup_handle_frame(&support->read_handle,
														  miss_blks[j], &version,
														  &frame_id, &entry_ptr))
				{
					mat_slots[k].opt_valid = true;
					mat_slots[k].frame_id = frame_id;
					mat_slots[k].version = version;
					mat_slots[k].entry_ptr = (CalicoTranslationEntry *) entry_ptr;
				}
			}
		}

		for (int k = 0; k < count; k++)
		{
			if (likely(mat_slots[k].opt_valid))
			{
				uint32		linp_off = offsetof(PageHeaderData, pd_linp) +
					(cands[start_idx + k].offno - 1) * sizeof(ItemIdData);

				__builtin_prefetch((const char *) CalicoFramePage(mat_slots[k].frame_id) + linp_off, 0, 3);
			}
		}
	}

	for (int k = 0; k < count; k++)
	{
		const HnswFastResultCand *cand = &cands[start_idx + k];
		bool		loaded = false;
		uint8		tlen = 0;

		if (use_opt_mat && mat_slots[k].opt_valid)
		{
			Page		page = CalicoFramePage(mat_slots[k].frame_id);
			HnswElementTuple etup = HnswOptimisticItem(page, cand->offno,
													   offsetof(HnswElementTupleData, data));

			if (likely(etup != NULL && HnswIsElementTuple(etup)))
			{
				bool		del = etup->deleted;
				int			max_ht = 0;

				if (!del)
				{
					for (int i = 0; i < HNSW_HEAPTIDS; i++)
					{
						ItemPointerData htid;

						memcpy(&htid, &etup->heaptids[i], sizeof(ItemPointerData));
						if (!ItemPointerIsValid(&htid))
							break;
						win_tids[k][max_ht++] = htid;
					}
					tlen = (uint8) max_ht;
				}
				else if (!allow_deleted)
					goto need_fallback;

				if (likely(calico_optimistic_validate(mat_slots[k].entry_ptr,
													  mat_slots[k].version)))
				{
					win_len[k] = tlen;
					loaded = true;
				}
			}
		}

need_fallback:
		if (unlikely(!loaded))
		{
			Buffer		buf;
			Page		page;
			HnswElementTuple etup;

			if (use_opt_mat)
				calico_optimistic_element_fallbacks++;

			buf = ReadBuffer(index, cand->blkno);
			LockBuffer(buf, BUFFER_LOCK_SHARE);
			page = BufferGetPage(buf);
			etup = (HnswElementTuple) PageGetItem(page, PageGetItemId(page, cand->offno));
			Assert(HnswIsElementTuple(etup));

			if (unlikely(etup->deleted))
			{
				if (!allow_deleted)
				{
					UnlockReleaseBuffer(buf);
					elog(ERROR, "cannot load deleted element");
				}
				win_len[k] = 0;
			}
			else
			{
				int			max_ht = 0;

				for (int i = 0; i < HNSW_HEAPTIDS; i++)
				{
					if (!ItemPointerIsValid(&etup->heaptids[i]))
						break;
					win_tids[k][max_ht++] = etup->heaptids[i];
				}
				win_len[k] = (uint8) max_ht;
			}

			UnlockReleaseBuffer(buf);
		}
	}
}
