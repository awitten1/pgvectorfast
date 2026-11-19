/* Iterative HNSW search, including resident scoring and asynchronous traversal. */
#include "postgres.h"

#include "hnswio.h"
#include "hnswsearch.h"
#include "storage/bufmgr.h"

#define HNSW_ITER_OOM_WAVE_MAX 4
#define HNSW_ITER_OOM_UNVISITED_CAP 256

typedef struct HnswPipeCand
{
	CalicoTranslationEntry *entry_ptr;
	uint64		opt_ver;
	uint32		id;
	BlockNumber page;
	OffsetNumber off;
	uint8		level;
	int8		stage;
	int16		at_i;
	uint32		buf_id;
}			HnswPipeCand;

#define HNSW_ITUP_PREF_L1 0
#define HNSW_ITUP_PREF_L2 1
#define HNSW_ITUP_PREF_HDR 2

static pg_attribute_always_inline HnswElementTuple
HnswResolveIterativeTuple(const char *buf_blocks, const CalicoOptimisticSlot *slot,
						  OffsetNumber offno, Size tuple_bytes,
						  Size min_valid_bytes, int pref_mode)
{
	Page		page;
	ItemIdData	item;
	unsigned	offset;
	unsigned	length;
	const char *p;
	Size		limit;
	uintptr_t	cl;
	uintptr_t	last_cl;

	if (unlikely(!slot->opt_valid || slot->entry_ptr == NULL))
		return NULL;
	if (unlikely(offno < FirstOffsetNumber ||
				 offno > (BLCKSZ - SizeOfPageHeaderData) / sizeof(ItemIdData)))
		return NULL;

	page = CalicoFramePageBase(buf_blocks, slot->frame_id);
	item = *(volatile ItemIdData *) PageGetItemId(page, offno);
	if (unlikely(!ItemIdIsNormal(&item)))
		return NULL;

	offset = ItemIdGetOffset(&item);
	length = ItemIdGetLength(&item);
	if (unlikely(offset < SizeOfPageHeaderData ||
				 length < sizeof(HnswElementTupleData) ||
				 length < min_valid_bytes ||
				 offset + length > BLCKSZ))
		return NULL;

	limit = Min(tuple_bytes, (Size) length);
	if (unlikely(limit == 0))
		return NULL;

	p = (const char *) page + offset;
	cl = (uintptr_t) p & ~(uintptr_t) 63;

	if (pref_mode == HNSW_ITUP_PREF_L1)
	{
		last_cl = (uintptr_t) (p + limit - 1) & ~(uintptr_t) 63;
		for (; cl <= last_cl; cl += 64)
			__builtin_prefetch((const void *) cl, 0, 3);
	}
	else if (pref_mode == HNSW_ITUP_PREF_L2)
	{
		last_cl = (uintptr_t) (p + limit - 1) & ~(uintptr_t) 63;
		for (; cl <= last_cl; cl += 64)
			__builtin_prefetch((const void *) cl, 0, 2);
	}
	else
	{
		Size		hdr_bytes = Min(limit, (Size) 192);
		uintptr_t	end_addr = (uintptr_t) (p + limit - 1);

		last_cl = (uintptr_t) (p + hdr_bytes - 1) & ~(uintptr_t) 63;
		for (; cl <= last_cl; cl += 64)
			__builtin_prefetch((const void *) cl, 0, 3);
		if (unlikely((((uintptr_t) p ^ end_addr) & ~(uintptr_t) 4095) != 0))
			__builtin_prefetch((const void *) (end_addr & ~(uintptr_t) 4095), 0, 2);
	}

	return (HnswElementTuple) p;
}

static pg_attribute_always_inline void
HnswAdvancePipeCand(HnswPipeCand *pc, int i, int m, int lm,
					const char *buf_blocks, CalicoReadHandle *read_handle)
{
	if (pc->stage == 1 && (i - pc->at_i) >= 2)
	{
		uint64		opt_ver;
		void	   *entry_ptr;

		if (likely(calico_optimistic_lookup_handle_frame(read_handle, pc->page,
														 &opt_ver, &pc->buf_id, &entry_ptr)) &&
			likely(pc->off >= FirstOffsetNumber &&
				   pc->off <= (BLCKSZ - SizeOfPageHeaderData) / sizeof(ItemIdData)))
		{
			__builtin_prefetch(PageGetItemId(CalicoFramePageBase(buf_blocks, pc->buf_id), pc->off), 0, 3);
			pc->entry_ptr = (CalicoTranslationEntry *) entry_ptr;
			pc->opt_ver = opt_ver;
			pc->stage = 2;
			pc->at_i = (int16) i;
		}
		else
			pc->stage = -1;
	}
	else if (pc->stage == 2 && (i - pc->at_i) >= 2)
	{
		HnswPrefetchNextNeighborTuple(buf_blocks, pc->buf_id, pc->off, pc->level, m, lm);
		pc->stage = 3;
	}
}

static pg_attribute_always_inline void
HnswFinishPipeCand(HnswPipeCand *pc, int count, int m, int lm,
				   const char *buf_blocks, CalicoReadHandle *read_handle)
{
	if (pc->stage == 2)
	{
		HnswPrefetchNextNeighborTuple(buf_blocks, pc->buf_id, pc->off, pc->level, m, lm);
		pc->stage = 3;
	}
	else if (pc->stage == 1 && (count - pc->at_i) >= 2)
	{
		uint64		opt_ver;
		void	   *entry_ptr;

		if (likely(calico_optimistic_lookup_handle_frame(read_handle, pc->page,
														 &opt_ver, &pc->buf_id, &entry_ptr)) &&
			likely(pc->off >= FirstOffsetNumber &&
				   pc->off <= (BLCKSZ - SizeOfPageHeaderData) / sizeof(ItemIdData)))
		{
			__builtin_prefetch(PageGetItemId(CalicoFramePageBase(buf_blocks, pc->buf_id), pc->off), 0, 3);
			pc->entry_ptr = (CalicoTranslationEntry *) entry_ptr;
			pc->opt_ver = opt_ver;
			pc->stage = 2;
		}
	}
}

static pg_attribute_always_inline bool
HnswLoadIterativeKeyOptimisticSlotImpl(HnswSupport *support, OffsetNumber offno,
									   const CalicoOptimisticSlot *slot,
									   HnswElementTuple cached_etup,
									   const float *pref_bx,
									   const char *buf_blocks,
									   const Vector *a_vec, Size vec_size,
									   double *dist_out, HnswIterativeState *state,
									   bool use_avx_fma,
									   HnswIterativeKey *key_out)
{
	Page		page;
	HnswElementTuple etup;
	double		dist;
	uint8		level;
	uint8		version;
	ItemPointerData nb;
	ItemPointerData htids[HNSW_HEAPTIDS];
	uint8		n_ht = 0;

	if (likely(cached_etup != NULL))
	{
		etup = cached_etup;
	}
	else
	{
		if (unlikely(slot == NULL || !slot->opt_valid || slot->entry_ptr == NULL))
			return false;

		page = CalicoFramePageBase(buf_blocks, slot->frame_id);
		etup = HnswOptimisticItem(page, offno,
								  likely(a_vec != NULL)
								  ? (offsetof(HnswElementTupleData, data) + vec_size)
								  : offsetof(HnswElementTupleData, data));
		if (unlikely(etup == NULL))
			return false;
	}

	if (unlikely(etup->deleted))
		return false;

	if (likely(a_vec != NULL))
	{
		if (unlikely(!HnswGetDistanceOptimisticPrefetchImpl(support, a_vec, &etup->data,
														   vec_size, use_avx_fma, pref_bx,
														   &dist)))
			return false;
	}
	else
		dist = 0;

	level = etup->level;
	version = etup->version;
	nb = etup->neighbortid;
	for (int i = 0; i < HNSW_HEAPTIDS; i++)
	{
		ItemPointerData htid;

		memcpy(&htid, &etup->heaptids[i], sizeof(ItemPointerData));
		if (!ItemPointerIsValid(&htid))
			break;
		htids[n_ht++] = htid;
	}

	if (unlikely(!calico_optimistic_validate(slot->entry_ptr, slot->version)))
		return false;

	if (dist_out != NULL)
		dist = *dist_out;

	*key_out = HnswIterativeRememberRaw(state,
										ItemPointerGetBlockNumber(&nb),
										ItemPointerGetOffsetNumber(&nb),
										level, version, htids, n_ht, dist);
	return true;
}

#if !defined(VECTOR_NATIVE_AVX_FMA) && defined(USE_DISPATCH)
static pg_noinline VECTOR_TARGET_AVX_FMA bool
HnswLoadIterativeKeyOptimisticSlotAvxFma(HnswSupport *support, OffsetNumber offno,
										 const CalicoOptimisticSlot *slot,
										 const char *buf_blocks,
										 const Vector *a_vec, Size vec_size,
										 HnswIterativeState *state,
										 HnswIterativeKey *key_out)
{
	return HnswLoadIterativeKeyOptimisticSlotImpl(support, offno, slot,
												  NULL, NULL, buf_blocks,
												  a_vec, vec_size, NULL, state,
												  true, key_out);
}
#endif

static pg_attribute_always_inline bool
HnswLoadIterativeKeyOptimisticSlotFast(HnswSupport *support, OffsetNumber offno,
									   const CalicoOptimisticSlot *slot,
									   const char *buf_blocks,
									   const Vector *a_vec, Size vec_size,
									   HnswIterativeState *state,
									   HnswIterativeKey *key_out)
{
#if !defined(VECTOR_NATIVE_AVX_FMA) && defined(USE_DISPATCH)
	if (likely(a_vec != NULL && vector_has_avx_fma))
		return HnswLoadIterativeKeyOptimisticSlotAvxFma(support, offno, slot,
														buf_blocks, a_vec, vec_size,
														state, key_out);
#endif
	return HnswLoadIterativeKeyOptimisticSlotImpl(support, offno, slot,
												  NULL, NULL, buf_blocks,
												  a_vec, vec_size, NULL, state,
												  false, key_out);
}

static pg_attribute_always_inline void
HnswScoreIterativeResidentBody(HnswQuery *q, int ef, int m, int lm,
							   Relation index, HnswSupport *support,
							   HnswIterativeState *state,
							   const HnswUnvisited *unvisited, int count,
							   const CalicoOptimisticSlot *slots,
							   HnswElementTuple *cand_etups,
							   bool optimistic,
							   const Vector *query, Size vec_size,
							   Size prefetch, Size minimum,
							   const char *buf_blocks,
							   bool doStepPrefetch, bool doPipe,
							   HnswPipeCand *pipe,
							   int *candidate_count, int *result_count,
							   bool force_avx_fma)
{
	CalicoReadHandle *const read_handle = optimistic ? &support->read_handle : NULL;
	HnswPipeCand p0 = pipe[0];
	HnswPipeCand p1 = pipe[1];

	for (int i = 0; i < count; i++)
	{
		HnswIterativeKey next;
		BlockNumber block = ItemPointerGetBlockNumber(&unvisited[i].indextid);
		OffsetNumber offset = ItemPointerGetOffsetNumber(&unvisited[i].indextid);
		HnswElementTuple cached_etup;
		const float *pref_bx = NULL;

		if (doStepPrefetch)
		{
			if (i + 8 < count && likely(slots[i + 8].opt_valid))
			{
				OffsetNumber off8 = ItemPointerGetOffsetNumber(&unvisited[i + 8].indextid);

				if (likely(off8 >= FirstOffsetNumber &&
						   off8 <= (BLCKSZ - SizeOfPageHeaderData) / sizeof(ItemIdData)))
				{
					Page		page8 = CalicoFramePageBase(buf_blocks, slots[i + 8].frame_id);

					__builtin_prefetch(PageGetItemId(page8, off8), 0, 3);
				}
			}
			if (i + 4 < count)
				cand_etups[i + 4] = HnswResolveIterativeTuple(buf_blocks, &slots[i + 4],
															  ItemPointerGetOffsetNumber(&unvisited[i + 4].indextid),
															  prefetch, minimum,
															  HNSW_ITUP_PREF_HDR);
			if (likely(query != NULL) && i + 2 < count && likely(cand_etups[i + 2] != NULL))
				pref_bx = cand_etups[i + 2]->data.x;
		}

		if (doPipe)
		{
			HnswAdvancePipeCand(&p0, i, m, lm, buf_blocks, read_handle);
			HnswAdvancePipeCand(&p1, i, m, lm, buf_blocks, read_handle);
		}

		cached_etup = (cand_etups != NULL) ? cand_etups[i] : NULL;
		if (!optimistic ||
			!HnswLoadIterativeKeyOptimisticSlotImpl(support, offset, &slots[i],
													cached_etup, pref_bx, buf_blocks,
													query, vec_size, NULL, state,
													force_avx_fma, &next))
		{
			HnswElementData element_data;
			HnswElement element = &element_data;
			double		distance;

			HnswLoadElementImpl(block, offset, &distance, q, index, support,
								false, NULL, &element);
			next = HnswIterativeRememberElement(state, element, distance);
		}

		if (HnswIterativeCommit(state, next, ef, candidate_count, result_count) &&
			doPipe && state->candidates[0].id != p0.id)
		{
			uint32		new_id = state->candidates[0].id;

			if (new_id == p1.id)
			{
				HnswPipeCand tmp = p0;

				p0 = p1;
				p1 = tmp;
			}
			else
			{
				const HnswIterativeNode *p0_node;

				p1 = p0;
				p0.id = new_id;
				p0_node = HnswIterativeGetNode(state, new_id);
				__builtin_prefetch(p0_node, 0, 3);
				p0.page = p0_node->neighborPage;
				p0.off = p0_node->neighborOffno;
				p0.level = p0_node->level;
				calico_optimistic_prefetch_block(read_handle, p0.page);
				p0.stage = 1;
				p0.at_i = (int16) i;
			}
		}
	}

	if (doPipe)
	{
		HnswFinishPipeCand(&p0, count, m, lm, buf_blocks, read_handle);
		HnswFinishPipeCand(&p1, count, m, lm, buf_blocks, read_handle);
		pipe[0] = p0;
		pipe[1] = p1;
	}
}

#if !defined(VECTOR_NATIVE_AVX_FMA) && defined(USE_DISPATCH)
static pg_noinline VECTOR_TARGET_AVX_FMA void
HnswScoreIterativeResidentAvxFma(HnswQuery *q, int ef, int m, int lm,
								 Relation index, HnswSupport *support,
								 HnswIterativeState *state,
								 const HnswUnvisited *unvisited, int count,
								 const CalicoOptimisticSlot *slots,
								 HnswElementTuple *cand_etups,
								 const Vector *query, Size vec_size,
								 Size prefetch, Size minimum,
								 const char *buf_blocks,
								 bool doStepPrefetch, bool doPipe,
								 HnswPipeCand *pipe,
								 int *candidate_count, int *result_count)
{
	HnswScoreIterativeResidentBody(q, ef, m, lm, index, support, state,
								   unvisited, count, slots, cand_etups,
								   true, query, vec_size, prefetch, minimum,
								   buf_blocks, doStepPrefetch, doPipe,
								   pipe, candidate_count, result_count, true);
}
#endif

static pg_noinline void
HnswScoreIterativeAsyncWave(HnswQuery *q, int ef, Relation index,
							HnswSupport *support, HnswIterativeState *state,
							const HnswUnvisited *unvisited,
							CalicoOptimisticSlot *slots, int count,
							bool optimistic, const Vector *query,
							Size vec_size, Size prefetch,
							const char *buf_blocks, CalicoAsyncBatch *batch,
							int *candidate_count, int *result_count)
{
	bool		processed[HNSW_ITER_OOM_UNVISITED_CAP];
	int			res_idxs[HNSW_ITER_OOM_UNVISITED_CAP];
	int			rem_idxs[HNSW_ITER_OOM_UNVISITED_CAP];
	BlockNumber rem_blks[HNSW_ITER_OOM_UNVISITED_CAP];
	CalicoOptimisticSlot rem_slots[HNSW_ITER_OOM_UNVISITED_CAP];
	int			n_res = 0;
	int			n_rem = 0;
	const bool	doPrefetch = (hnsw_enable_prefetch && prefetch > 0);

	Assert(count <= HNSW_ITER_OOM_UNVISITED_CAP);

	if (optimistic)
	{
		for (int i = 0; i < count; i++)
		{
			processed[i] = false;
			if (slots[i].resident && slots[i].opt_valid)
				res_idxs[n_res++] = i;
		}

		if (doPrefetch && n_res > 0)
		{
			int			prime = Min(n_res, HNSW_PREFETCH_AHEAD);

			for (int r = 0; r < prime; r++)
			{
				int			p_idx = res_idxs[r];

				HnswPrefetchSlotTuple(&slots[p_idx],
									  ItemPointerGetOffsetNumber(&unvisited[p_idx].indextid),
									  prefetch);
			}
		}

		for (int r = 0; r < n_res; r++)
		{
			int			idx = res_idxs[r];
			OffsetNumber offset = ItemPointerGetOffsetNumber(&unvisited[idx].indextid);
			HnswIterativeKey next;

			if (doPrefetch && r + HNSW_PREFETCH_AHEAD < n_res)
			{
				int			p_idx = res_idxs[r + HNSW_PREFETCH_AHEAD];

				HnswPrefetchSlotTuple(&slots[p_idx],
									  ItemPointerGetOffsetNumber(&unvisited[p_idx].indextid),
									  prefetch);
			}

			if (HnswLoadIterativeKeyOptimisticSlotFast(support, offset, &slots[idx],
													   buf_blocks, query, vec_size,
													   state, &next))
			{
				processed[idx] = true;
				HnswIterativeCommit(state, next, ef, candidate_count, result_count);
			}
		}
	}
	else
	{
		for (int i = 0; i < count; i++)
			processed[i] = false;
	}

	if (batch != NULL && batch->active)
		CalicoAsyncBatchFinish(batch);

	for (int i = 0; i < count; i++)
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
		if (optimistic && support != NULL)
		{
			calico_optimistic_lookup_batch_slots(&support->read_handle,
												 rem_blks, n_rem, rem_slots);
			for (int k = 0; k < n_rem; k++)
				slots[rem_idxs[k]] = rem_slots[k];

			if (doPrefetch)
			{
				int			prime = Min(n_rem, HNSW_PREFETCH_AHEAD);

				for (int k = 0; k < prime; k++)
				{
					int			p_idx = rem_idxs[k];

					HnswPrefetchSlotTuple(&slots[p_idx],
										  ItemPointerGetOffsetNumber(&unvisited[p_idx].indextid),
										  prefetch);
				}
			}
		}

		for (int k = 0; k < n_rem; k++)
		{
			int			idx = rem_idxs[k];
			BlockNumber block = rem_blks[k];
			OffsetNumber offset = ItemPointerGetOffsetNumber(&unvisited[idx].indextid);
			HnswIterativeKey next;

			if (optimistic && doPrefetch && k + HNSW_PREFETCH_AHEAD < n_rem)
			{
				int			p_idx = rem_idxs[k + HNSW_PREFETCH_AHEAD];

				HnswPrefetchSlotTuple(&slots[p_idx],
									  ItemPointerGetOffsetNumber(&unvisited[p_idx].indextid),
									  prefetch);
			}

			if (!optimistic ||
				!HnswLoadIterativeKeyOptimisticSlotFast(support, offset, &slots[idx],
														buf_blocks, query, vec_size,
														state, &next))
			{
				HnswElementData element_data;
				HnswElement element = &element_data;
				double		distance;

				HnswLoadElementImpl(block, offset, &distance, q, index, support,
									false, NULL, &element);
				next = HnswIterativeRememberElement(state, element, distance);
			}
			HnswIterativeCommit(state, next, ef, candidate_count, result_count);
		}
	}
}

static pg_noinline void
HnswExpandIterativeOomWave(HnswQuery *q, int ef, Relation index,
						   HnswSupport *support, int m, int lm,
						   visited_hash *v, int64 *tuples,
						   HnswIterativeState *state, bool optimistic,
						   const Vector *query, Size vec_size, Size prefetch,
						   const char *buf_blocks, CalicoAsyncBatch **batch_ptr,
						   int *candidate_count, int *result_count)
{
	int			max_wave = Max(1, Min(HNSW_ITER_OOM_WAVE_MAX,
									  HNSW_ITER_OOM_UNVISITED_CAP / Max(lm, 1)));
	const HnswIterativeNode *nodes[HNSW_ITER_OOM_WAVE_MAX];
	CalicoOptimisticSlot nb_slots[HNSW_ITER_OOM_WAVE_MAX];
	int			miss_nb_idxs[HNSW_ITER_OOM_WAVE_MAX];
	BlockNumber miss_nb_blks[HNSW_ITER_OOM_WAVE_MAX];
	int			n_wave = 0;
	int			n_miss_nb = 0;
	HnswUnvisited wave_unvisited[HNSW_ITER_OOM_UNVISITED_CAP];
	CalicoOptimisticSlot wave_slots[HNSW_ITER_OOM_UNVISITED_CAP];
	BlockNumber wave_blks[HNSW_ITER_OOM_UNVISITED_CAP];
	BlockNumber miss_el_blks[HNSW_ITER_OOM_UNVISITED_CAP];
	int			total_u = 0;
	int			n_miss_el = 0;

	Assert(lm <= HNSW_ITER_OOM_UNVISITED_CAP);

	while (n_wave < max_wave && *candidate_count > 0)
	{
		HnswIterativeKey cand = state->candidates[0];

		if (*result_count > 0 && cand.distance > state->results[0].distance)
			break;

		state->candidates[0] = state->candidates[--(*candidate_count)];
		if (*candidate_count > 0)
			HnswIterativeSiftDownMin4(state->candidates, *candidate_count, 0);
		nodes[n_wave++] = HnswIterativeGetNode(state, cand.id);
	}

	if (n_wave == 0)
		return;

	if (optimistic)
	{
		for (int w = 0; w < n_wave; w++)
			calico_optimistic_prefetch_block(&support->read_handle, nodes[w]->neighborPage);

		for (int w = 0; w < n_wave; w++)
		{
			uint64		opt_ver = 0;
			uint32		buf_id = 0;
			void	   *entry_ptr = NULL;
			BlockNumber nb_blk = nodes[w]->neighborPage;

			if (calico_optimistic_lookup_handle_frame(&support->read_handle, nb_blk,
													  &opt_ver, &buf_id, &entry_ptr))
			{
				nb_slots[w].entry_ptr = (CalicoTranslationEntry *) entry_ptr;
				nb_slots[w].version = opt_ver;
				nb_slots[w].frame_id = buf_id;
				nb_slots[w].resident = true;
				nb_slots[w].opt_valid = true;
			}
			else
			{
				nb_slots[w].entry_ptr = NULL;
				nb_slots[w].version = 0;
				nb_slots[w].frame_id = 0;
				nb_slots[w].resident = false;
				nb_slots[w].opt_valid = false;
				miss_nb_idxs[n_miss_nb] = w;
				miss_nb_blks[n_miss_nb] = nb_blk;
				n_miss_nb++;
			}
		}

		if (n_miss_nb > 0)
		{
			CalicoReadBuffersAsync(RelationGetSmgr(index),
								   index->rd_rel->relpersistence,
								   MAIN_FORKNUM,
								   miss_nb_blks,
								   n_miss_nb);
			for (int j = 0; j < n_miss_nb; j++)
			{
				int			w = miss_nb_idxs[j];
				uint64		opt_ver = 0;
				uint32		buf_id = 0;
				void	   *entry_ptr = NULL;

				if (calico_optimistic_lookup_handle_frame(&support->read_handle,
														  miss_nb_blks[j],
														  &opt_ver, &buf_id, &entry_ptr))
				{
					nb_slots[w].entry_ptr = (CalicoTranslationEntry *) entry_ptr;
					nb_slots[w].version = opt_ver;
					nb_slots[w].frame_id = buf_id;
					nb_slots[w].resident = true;
					nb_slots[w].opt_valid = true;
				}
			}
		}
	}

	for (int w = 0; w < n_wave; w++)
	{
		int			u_cnt = 0;

		if (HnswCollectUnvisitedFromNeighbor(nodes[w]->neighborPage, nodes[w]->neighborOffno,
											 nodes[w]->level, nodes[w]->version,
											 optimistic ? &nb_slots[w] : NULL,
											 &wave_unvisited[total_u], &u_cnt,
											 v, index, support, m, lm, 0,
											 false, NULL, NULL))
		{
			*tuples += u_cnt;
			total_u += u_cnt;
		}
	}

	if (total_u > 0)
	{
		if (optimistic)
		{
			for (int i = 0; i < total_u; i++)
			{
				wave_blks[i] = ItemPointerGetBlockNumber(&wave_unvisited[i].indextid);
				calico_optimistic_prefetch_block(&support->read_handle, wave_blks[i]);
			}
			calico_optimistic_lookup_batch_slots_noprefetch(&support->read_handle,
															wave_blks, total_u, wave_slots);
		}
		else
		{
			for (int i = 0; i < total_u; i++)
			{
				wave_blks[i] = ItemPointerGetBlockNumber(&wave_unvisited[i].indextid);
				wave_slots[i].entry_ptr = NULL;
				wave_slots[i].version = 0;
				wave_slots[i].frame_id = 0;
				wave_slots[i].resident = false;
				wave_slots[i].opt_valid = false;
			}
		}

		for (int i = 0; i < total_u; i++)
		{
			if (!wave_slots[i].resident)
				miss_el_blks[n_miss_el++] = wave_blks[i];
		}

		if (n_miss_nb == 0 && n_miss_el == 0)
		{
			if (++state->oom_hit_streak >= 2)
				state->oom_mode = false;
		}
		else
		{
			state->oom_hit_streak = 0;
			if (n_miss_el > 0 && calico_is_active() && !RelationUsesLocalBuffers(index))
			{
				if (*batch_ptr == NULL)
					*batch_ptr = (CalicoAsyncBatch *) palloc(sizeof(CalicoAsyncBatch));
				(*batch_ptr)->active = false;
				(*batch_ptr)->num_ranges = 0;
				CalicoAsyncBatchStart(*batch_ptr,
									  RelationGetSmgr(index),
									  index->rd_rel->relpersistence,
									  MAIN_FORKNUM,
									  miss_el_blks,
									  n_miss_el);
			}
		}

		HnswScoreIterativeAsyncWave(q, ef, index, support, state,
									wave_unvisited, wave_slots, total_u,
									optimistic, query, vec_size, prefetch,
									buf_blocks, *batch_ptr,
									candidate_count, result_count);
	}
	else if (n_miss_nb == 0)
	{
		if (++state->oom_hit_streak >= 2)
			state->oom_mode = false;
	}
	else
		state->oom_hit_streak = 0;
}

static pg_attribute_always_inline bool
HnswPrimeIterativeUnvisited(Relation index, HnswSupport *support,
							const HnswUnvisited *unvisited, int count,
							const BlockNumber *batch_blocknums,
							const uint16 *linp_offs,
							bool useBatchOpt,
							CalicoOptimisticSlot *slots_out,
							HnswElementTuple *cand_etups,
							Size prefetch, Size minimum,
							const char *buf_blocks,
							CalicoAsyncBatch **batch_out_ptr)
{
	if (likely(useBatchOpt))
	{
		bool		all_resident;

		all_resident = calico_optimistic_lookup_batch_slots_noprefetch(&support->read_handle,
																		batch_blocknums,
																		count,
																		slots_out);
		if (unlikely(!all_resident))
		{
			if (cand_etups != NULL)
				memset(cand_etups, 0, sizeof(HnswElementTuple) * count);
			if (HnswLoadUnvisitedMissingOrPinSlow(index, support, batch_blocknums,
												  count, unvisited, slots_out,
												  prefetch, batch_out_ptr))
				return true;
		}
		else if (likely(prefetch > 0 && cand_etups != NULL))
		{
			int			linp_prime = Min(count, 8);

			for (int j = 0; j < linp_prime; j++)
			{
				if (likely(slots_out[j].resident && linp_offs[j] != 0))
					__builtin_prefetch((const char *) CalicoFramePageBase(buf_blocks, slots_out[j].frame_id) + linp_offs[j], 0, 3);
			}
			cand_etups[0] = HnswResolveIterativeTuple(buf_blocks, &slots_out[0],
													  ItemPointerGetOffsetNumber(&unvisited[0].indextid),
													  prefetch, minimum, HNSW_ITUP_PREF_L1);
			if (count > 1)
				cand_etups[1] = HnswResolveIterativeTuple(buf_blocks, &slots_out[1],
														  ItemPointerGetOffsetNumber(&unvisited[1].indextid),
														  prefetch, minimum, HNSW_ITUP_PREF_L2);
			for (int j = 2; j < Min(count, 4); j++)
				cand_etups[j] = HnswResolveIterativeTuple(buf_blocks, &slots_out[j],
														  ItemPointerGetOffsetNumber(&unvisited[j].indextid),
														  prefetch, minimum, HNSW_ITUP_PREF_HDR);
			for (int j = 4; j < count; j++)
				cand_etups[j] = NULL;
		}
		else if (cand_etups != NULL)
		{
			memset(cand_etups, 0, sizeof(HnswElementTuple) * count);
		}
	}
	else
	{
		for (int j = 0; j < count; j++)
		{
			slots_out[j].resident = false;
			slots_out[j].opt_valid = false;
		}
		if (cand_etups != NULL)
			memset(cand_etups, 0, sizeof(HnswElementTuple) * count);
	}

	return false;
}

static void
HnswSearchIterativeCore(HnswQuery *q, int ef, Relation index,
						HnswSupport *support, int m, visited_hash *v,
						int64 *tuples, HnswIterativeState *state,
						bool expand, bool optimistic,
						const Vector *query, Size vec_size, Size prefetch)
{
	int			candidate_count = 0;
	int			result_count = 0;
	int			lm = HnswGetLayerM(m, 0);
	HnswUnvisited unvisited[HNSW_MAX_M * 2];
	BlockNumber batch_blocknums[HNSW_MAX_M * 2];
	uint16		linp_offs[HNSW_MAX_M * 2];
	CalicoOptimisticSlot slots[HNSW_MAX_M * 2];
	HnswElementTuple cand_etups[HNSW_MAX_M * 2];
	CalicoAsyncBatch *batch = NULL;
	Size		minimum = offsetof(HnswElementTupleData, data) + vec_size;
	const char *const buf_blocks = BufferBlocks;
	const bool	doPrefetch = hnsw_enable_prefetch;
	const bool	useBatchOpt = (optimistic && doPrefetch);
	const bool	doStepPrefetch = (useBatchOpt && prefetch > 0);
	const bool	doPipe = useBatchOpt;

	Assert(ef > 0 && lm <= lengthof(unvisited));

	/* A resumed cursor may have batch prefetch disabled between fetches. */
	if (!useBatchOpt)
	{
		state->oom_mode = false;
		state->oom_hit_streak = 0;
	}

	if (ef > state->result_capacity)
	{
		state->results = (HnswIterativeKey *) repalloc(state->results,
													   mul_size(sizeof(HnswIterativeKey), ef));
		state->result_capacity = ef;
	}

	if (!expand)
	{
		int			drain;

		if (!state->pending_sorted)
		{
			if (state->drain_batches >= 1 && state->pending_count > 4096)
				HnswSortPendingDesc(state);
			else if (state->drain_batches < 255)
				state->drain_batches++;
		}

		drain = Min(ef, state->pending_count);
		if (state->pending_sorted)
		{
			for (int i = 0; i < drain; i++)
				state->results[result_count++] = state->pending[--state->pending_count];
		}
		else
		{
			for (int i = 0; i < drain; i++)
				state->results[result_count++] = HnswIterativePop(state);
		}
		state->result_count = result_count;
		state->result_idx = 0;
		state->cur_rem_htids = 0;
		return;
	}

	state->drain_batches = 0;
	if (unlikely(state->pending_sorted))
	{
		for (int l = 0, r = state->pending_count - 1; l < r; l++, r--)
		{
			HnswIterativeKey swap = state->pending[l];

			state->pending[l] = state->pending[r];
			state->pending[r] = swap;
		}
		state->pending_sorted = false;
	}

	for (int i = 0; i < ef && state->pending_count > 0; i++)
	{
		HnswIterativeKey key = HnswIterativePop(state);

		HnswIterativePushMin4(key, &state->candidates, &candidate_count,
							  &state->candidate_capacity);
		HnswIterativePushMax4(key, state->results, &result_count);
	}

	{
		HnswPipeCand pipe[2];

		pipe[0].entry_ptr = NULL;
		pipe[0].opt_ver = 0;
		pipe[0].id = UINT32_MAX;
		pipe[0].page = InvalidBlockNumber;
		pipe[0].off = InvalidOffsetNumber;
		pipe[0].level = 0;
		pipe[0].stage = 0;
		pipe[0].at_i = -2;
		pipe[0].buf_id = 0;
		pipe[1] = pipe[0];

		while (candidate_count > 0)
		{
			int			count = 0;
			bool		asynchronous;
			uint64		nb_fb_before;

			CHECK_FOR_INTERRUPTS();
			if (state->candidates[0].distance > state->results[0].distance)
				break;
			if (unlikely(state->oom_mode))
			{
				pipe[0].id = UINT32_MAX;
				pipe[0].stage = 0;
				pipe[0].entry_ptr = NULL;
				pipe[1] = pipe[0];
				HnswExpandIterativeOomWave(q, ef, index, support, m, lm, v, tuples,
										   state, optimistic, query, vec_size, prefetch,
										   buf_blocks, &batch,
										   &candidate_count, &result_count);
				continue;
			}

			if (unlikely(batch != NULL))
			{
				batch->active = false;
				batch->num_ranges = 0;
			}

			nb_fb_before = calico_optimistic_neighbor_fallbacks;
			{
				int			n_popped = 0;

				while (candidate_count > 0)
				{
					HnswIterativeKey pop_key = state->candidates[0];
					const HnswIterativeNode *pop_node;
					int			u_cnt = 0;
					CalicoOptimisticSlot pre_slot_buf;
					const CalicoOptimisticSlot *pre_slot = NULL;

					if (n_popped > 0)
					{
						if (count > 0 &&
							(n_popped >= 2 || ef < 32 || count >= 8 ||
							 calico_optimistic_neighbor_fallbacks != nb_fb_before))
							break;
						if (count + lm > (int) lengthof(unvisited))
							break;
						if (result_count == ef && pop_key.distance > state->results[0].distance)
							break;
					}

					if (count == 0 && n_popped > 0)
						CHECK_FOR_INTERRUPTS();

					if (doPipe)
					{
						const HnswPipeCand *hit = NULL;

						if (pop_key.id == pipe[0].id && pipe[0].stage >= 2)
							hit = &pipe[0];
						else if (pop_key.id == pipe[1].id && pipe[1].stage >= 2)
							hit = &pipe[1];

						if (hit != NULL && likely(hit->entry_ptr != NULL))
						{
							pre_slot_buf.entry_ptr = hit->entry_ptr;
							pre_slot_buf.version = hit->opt_ver;
							pre_slot_buf.frame_id = hit->buf_id;
							pre_slot_buf.resident = true;
							pre_slot_buf.opt_valid = true;
							pre_slot = &pre_slot_buf;
						}
					}

					state->candidates[0] = state->candidates[--candidate_count];
					if (candidate_count > 0)
						HnswIterativeSiftDownMin4(state->candidates, candidate_count, 0);
					pop_node = HnswIterativeGetNode(state, pop_key.id);
					if (HnswCollectUnvisitedFromNeighbor(pop_node->neighborPage,
														 pop_node->neighborOffno,
														 pop_node->level,
														 pop_node->version,
														 pre_slot,
														 &unvisited[count], &u_cnt,
														 v, index, support, m, lm, 0,
														 useBatchOpt,
														 &batch_blocknums[count],
														 &linp_offs[count]))
					{
						(*tuples) += u_cnt;
						count += u_cnt;
					}
					n_popped++;
				}
			}

			if (unlikely(count == 0))
			{
				if (useBatchOpt && unlikely(calico_optimistic_neighbor_fallbacks != nb_fb_before))
				{
					state->oom_mode = true;
					state->oom_hit_streak = 0;
				}
				continue;
			}

			{
				HnswPipeCand prev_p0 = pipe[0];
				HnswPipeCand prev_p1 = pipe[1];

				pipe[0].entry_ptr = NULL;
				pipe[0].opt_ver = 0;
				pipe[0].id = UINT32_MAX;
				pipe[0].page = InvalidBlockNumber;
				pipe[0].off = InvalidOffsetNumber;
				pipe[0].level = 0;
				pipe[0].stage = 0;
				pipe[0].at_i = -2;
				pipe[0].buf_id = 0;
				pipe[1] = pipe[0];

				if (doPipe && candidate_count > 0)
				{
					if (result_count < ef ||
						state->candidates[0].distance <= state->results[0].distance)
					{
						uint32		c0_id = state->candidates[0].id;

						if (c0_id == prev_p0.id && prev_p0.stage >= 2)
						{
							pipe[0] = prev_p0;
							pipe[0].at_i = -2;
						}
						else if (c0_id == prev_p1.id && prev_p1.stage >= 2)
						{
							pipe[0] = prev_p1;
							pipe[0].at_i = -2;
						}
						else
						{
							const HnswIterativeNode *p0_node = HnswIterativeGetNode(state, c0_id);

							pipe[0].id = c0_id;
							__builtin_prefetch(p0_node, 0, 3);
							pipe[0].page = p0_node->neighborPage;
							pipe[0].off = p0_node->neighborOffno;
							pipe[0].level = p0_node->level;
							calico_optimistic_prefetch_block(&support->read_handle, pipe[0].page);
							pipe[0].stage = 1;
							pipe[0].at_i = -2;
						}

						if (ef >= 32 && candidate_count > 1)
						{
							int			best_h = 1;
							float		best_d = state->candidates[1].distance;
							int			chk_end = Min(candidate_count, 5);

							for (int h = 2; h < chk_end; h++)
							{
								float		d = state->candidates[h].distance;

								if (d < best_d)
								{
									best_d = d;
									best_h = h;
								}
							}
							if (result_count < ef || best_d <= state->results[0].distance)
							{
								uint32		c1_id = state->candidates[best_h].id;

								if (c1_id == prev_p0.id && prev_p0.stage >= 2)
								{
									pipe[1] = prev_p0;
									pipe[1].at_i = -1;
								}
								else if (c1_id == prev_p1.id && prev_p1.stage >= 2)
								{
									pipe[1] = prev_p1;
									pipe[1].at_i = -1;
								}
								else
								{
									const HnswIterativeNode *p1_node = HnswIterativeGetNode(state, c1_id);

									pipe[1].id = c1_id;
									__builtin_prefetch(p1_node, 0, 3);
									pipe[1].page = p1_node->neighborPage;
									pipe[1].off = p1_node->neighborOffno;
									pipe[1].level = p1_node->level;
									calico_optimistic_prefetch_block(&support->read_handle, pipe[1].page);
									pipe[1].stage = 1;
									pipe[1].at_i = -1;
								}
							}
						}
					}
					for (int h = 1; h < Min(candidate_count, 5); h++)
					{
						uint32		hn_id = state->candidates[h].id;

						if (hn_id != pipe[1].id)
						{
							const HnswIterativeNode *hn = HnswIterativeGetNode(state, hn_id);

							__builtin_prefetch(hn, 0, 2);
							calico_optimistic_prefetch_block(&support->read_handle, hn->neighborPage);
						}
					}
				}
			}

			asynchronous = HnswPrimeIterativeUnvisited(index, support, unvisited, count,
													   batch_blocknums, linp_offs,
													   useBatchOpt, slots,
													   useBatchOpt ? cand_etups : NULL,
													   prefetch, minimum, buf_blocks, &batch);
			if (useBatchOpt &&
				unlikely(asynchronous || calico_optimistic_neighbor_fallbacks != nb_fb_before))
			{
				state->oom_mode = true;
				state->oom_hit_streak = 0;
			}

			if (likely(!asynchronous))
			{
#if !defined(VECTOR_NATIVE_AVX_FMA) && defined(USE_DISPATCH)
				if (likely(optimistic && query != NULL && vector_has_avx_fma))
				{
					HnswScoreIterativeResidentAvxFma(q, ef, m, lm, index, support, state,
													 unvisited, count, slots,
													 useBatchOpt ? cand_etups : NULL,
													 query, vec_size, prefetch, minimum,
													 buf_blocks, doStepPrefetch, doPipe,
													 pipe, &candidate_count, &result_count);
				}
				else
#endif
				{
					HnswScoreIterativeResidentBody(q, ef, m, lm, index, support, state,
												   unvisited, count, slots,
												   useBatchOpt ? cand_etups : NULL,
												   optimistic,
												   query, vec_size, prefetch, minimum,
												   buf_blocks, doStepPrefetch, doPipe,
												   pipe, &candidate_count, &result_count, false);
				}
			}
			else
			{
				HnswScoreIterativeAsyncWave(q, ef, index, support, state,
											unvisited, slots, count,
											optimistic, query, vec_size, prefetch,
											buf_blocks, batch,
											&candidate_count, &result_count);
			}
		}
	}

	if (batch != NULL)
		pfree(batch);

	/* Arrange max-heap results in ascending distance order (closest first). */
	for (int rem = result_count - 1; rem > 0; rem--)
	{
		HnswIterativeKey top = state->results[0];

		state->results[0] = state->results[rem];
		state->results[rem] = top;
		HnswIterativeSiftDownMax4(state->results, rem, 0);
	}
	state->result_count = result_count;
	state->result_idx = 0;
	state->cur_rem_htids = 0;
}

void
HnswSearchIterativeFromEntry(HnswQuery *q, HnswElement entryPoint, int ef,
							 Relation index, HnswSupport *support, int m,
							 visited_hash *v, int64 *tuples,
							 HnswIterativeState **state_ptr)
{
	HnswIterativeState *state;
	bool		canOptDistance;
	const Vector *queryVec;
	Size		optVecSize;
	Size		optPrefetchBytes;
	HnswFastCand best;
	HnswIterativeKey ep_key;
	bool		ep_loaded = false;
	Size		prefetch;

	Assert(support->vectorDistance != HNSW_VECTOR_DISTANCE_NONE);
	Assert(ef > 0 && entryPoint != NULL);

	HnswInitOptDistanceParams(q, index, support, &canOptDistance, &queryVec,
							  &optVecSize, &optPrefetchBytes);
	prefetch = hnsw_enable_prefetch ? optPrefetchBytes : 0;

	HnswSearchLayerUpperDiskFast(q, entryPoint, index, support, m,
								 canOptDistance, queryVec, optVecSize, optPrefetchBytes,
								 &best);

	state = HnswInitIterativeState(ef);
	*state_ptr = state;

	hnsw_fast_visited_init(&v->tids, ef, m, NULL, 0);
	(void) hnsw_fast_visited_insert(&v->tids, best.blkno, best.offno);
	(*tuples)++;

	if (canOptDistance)
	{
		uint64		opt_ver = 0;
		uint32		buf_id = 0;
		void	   *entry_ptr = NULL;

		if (likely(calico_optimistic_lookup_handle_frame(&support->read_handle,
														 best.blkno,
														 &opt_ver, &buf_id, &entry_ptr)))
		{
			CalicoOptimisticSlot ep_slot;
			double		best_dist = best.distance;

			ep_slot.entry_ptr = (CalicoTranslationEntry *) entry_ptr;
			ep_slot.version = opt_ver;
			ep_slot.frame_id = buf_id;
			ep_slot.resident = true;
			ep_slot.opt_valid = true;

			ep_loaded = HnswLoadIterativeKeyOptimisticSlotImpl(support, best.offno,
															   &ep_slot, NULL, NULL, BufferBlocks,
															   NULL, 0,
															   &best_dist, state, false, &ep_key);
		}
		if (unlikely(!ep_loaded))
			calico_optimistic_element_fallbacks++;
	}

	if (unlikely(!ep_loaded))
	{
		HnswElementData ep_data;
		HnswElement ep_el = &ep_data;

		HnswLoadElementImpl(best.blkno, best.offno, NULL, q, index, support,
							false, NULL, &ep_el);
		ep_key = HnswIterativeRememberElement(state, ep_el, best.distance);
	}

	HnswIterativePushMin4(ep_key, &state->pending, &state->pending_count,
						  &state->pending_capacity);

	HnswSearchIterativeCore(q, ef, index, support, m, v, tuples, state, true,
							calico_enable_optimistic_reads && calico_is_active() &&
							!RelationUsesLocalBuffers(index),
							queryVec, optVecSize, prefetch);
}

void
HnswSearchIterativeBatch(HnswQuery *q, int ef, Relation index,
						 HnswSupport *support, int m, visited_hash *v,
						 int64 *tuples, HnswIterativeState *state, bool expand)
{
	const Vector *query = (const Vector *) DatumGetPointer(q->value);
	Size		vec_size = query != NULL ? VECTOR_SIZE(query->dim) : 0;
	Size		minimum = offsetof(HnswElementTupleData, data) + vec_size;
	Size		prefetch = hnsw_enable_prefetch ? Min(minimum, (Size) BLCKSZ) : 0;
	bool		optimistic = calico_enable_optimistic_reads && calico_is_active() &&
		!RelationUsesLocalBuffers(index);

	Assert(support->vectorDistance != HNSW_VECTOR_DISTANCE_NONE);
	HnswSearchIterativeCore(q, ef, index, support, m, v, tuples, state, expand,
							optimistic, query, vec_size, prefetch);
}
