/* Staged prefetch of index line pointers, tuple headers, and vector payloads. */
#ifndef HNSWPREFETCH_H
#define HNSWPREFETCH_H

#include "hnsw.h"
#include "storage/buf_internals.h"

#define CalicoFramePageBase(base, frame_id) \
	((Page) ((base) + ((Size) (frame_id)) * BLCKSZ))

#define HNSW_PREFETCH_AHEAD 3
#define HNSW_PREFETCH_STEP_LINP 8
#define HNSW_PREFETCH_STEP_STAGEA 4
#define HNSW_PREFETCH_STEP_STAGEB 2
#define HNSW_PREFETCH_STEP_STAGEC 1

/*
 * Prefetch all relevant cachelines of a neighbor's tuple (header + vector)
 * within page bounds from a resolved optimistic slot.
 */
static pg_attribute_always_inline HnswElementTuple
HnswPrefetchSlotTupleEx(const CalicoOptimisticSlot *slot, OffsetNumber offno,
						Size tuple_bytes, Size min_valid_bytes)
{
	Page		page;
	ItemIdData	item;
	unsigned	offset;
	unsigned	length;
	const char *p;
	Size		limit;

	if (unlikely(!slot->opt_valid || slot->entry_ptr == NULL))
		return NULL;
	if (unlikely(offno < FirstOffsetNumber ||
				 offno > (BLCKSZ - SizeOfPageHeaderData) / sizeof(ItemIdData)))
		return NULL;

	page = CalicoFramePage(slot->frame_id);
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
	p = (const char *) page + offset;
	/* Exact branchless cache-line sweep: covers [p, p + limit - 1] once per 64B line. */
	if (likely(limit > 0))
	{
		uintptr_t	cl = (uintptr_t) p & ~(uintptr_t) 63;
		uintptr_t	last_cl = (uintptr_t) (p + limit - 1) & ~(uintptr_t) 63;

		for (; cl + 192 <= last_cl; cl += 256)
		{
			__builtin_prefetch((const void *) cl, 0, 3);
			__builtin_prefetch((const void *) (cl + 64), 0, 3);
			__builtin_prefetch((const void *) (cl + 128), 0, 3);
			__builtin_prefetch((const void *) (cl + 192), 0, 3);
		}
		for (; cl <= last_cl; cl += 64)
			__builtin_prefetch((const void *) cl, 0, 3);
	}

	return (HnswElementTuple) p;
}

static inline void
HnswPrefetchSlotTuple(const CalicoOptimisticSlot *slot, OffsetNumber offno, Size tuple_bytes)
{
	(void) HnswPrefetchSlotTupleEx(slot, offno, tuple_bytes, 0);
}

static pg_attribute_always_inline HnswElementTuple
HnswPrefetchTupleStageA(const char *buf_blocks, const CalicoOptimisticSlot *slot,
						OffsetNumber offno, Size tuple_bytes,
						Size min_valid_bytes, uint16 *limit_out)
{
	Page		page;
	ItemIdData	item;
	unsigned	offset;
	unsigned	length;
	const char *p;
	Size		limit;
	Size		first_a;
	uintptr_t	cl;
	uintptr_t	last_cl;

	*limit_out = 0;
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
	*limit_out = (uint16) limit;
	first_a = Min(limit, (Size) 256);
	cl = (uintptr_t) p & ~(uintptr_t) 63;
	last_cl = (uintptr_t) (p + first_a - 1) & ~(uintptr_t) 63;
	for (; cl <= last_cl; cl += 64)
		__builtin_prefetch((const void *) cl, 0, 3);

	return (HnswElementTuple) p;
}

static pg_attribute_always_inline void
HnswPrefetchTupleStageB(const HnswElementTuple etup, uint16 limit)
{
	const char *p;
	Size		mid;
	uintptr_t	cl;
	uintptr_t	last_cl;

	if (etup == NULL || limit <= 256)
		return;

	p = (const char *) etup;
	mid = (limit <= 1024) ? (Size) limit : ((((Size) limit + 256) >> 1) & ~(Size) 63);
	cl = ((uintptr_t) (p + 255) & ~(uintptr_t) 63) + 64;
	last_cl = (uintptr_t) (p + mid - 1) & ~(uintptr_t) 63;
	if (likely(limit <= 2048))
	{
		for (; cl <= last_cl; cl += 64)
			__builtin_prefetch((const void *) cl, 0, 3);
	}
	else
	{
		for (; cl <= last_cl; cl += 64)
			__builtin_prefetch((const void *) cl, 0, 2);
	}
}

static pg_attribute_always_inline void
HnswPrefetchTupleStageC(const HnswElementTuple etup, uint16 limit)
{
	const char *p;
	Size		mid;
	uintptr_t	cl;
	uintptr_t	last_cl;

	if (etup == NULL || limit <= 1024)
		return;

	p = (const char *) etup;
	mid = (((Size) limit + 256) >> 1) & ~(Size) 63;
	cl = ((uintptr_t) (p + mid - 1) & ~(uintptr_t) 63) + 64;
	last_cl = (uintptr_t) (p + limit - 1) & ~(uintptr_t) 63;
	if (likely(limit <= 2048))
	{
		for (; cl <= last_cl; cl += 64)
			__builtin_prefetch((const void *) cl, 0, 3);
	}
	else
	{
		for (; cl <= last_cl; cl += 64)
			__builtin_prefetch((const void *) cl, 0, 2);
	}
}

static pg_attribute_always_inline void
HnswPrefetchNextNeighborTuple(const char *buf_blocks, uint32 frame_id,
							  OffsetNumber neighborOffno,
							  uint8 level, int m, int lm)
{
	Page		page;
	ItemIdData	item;
	unsigned	offset;
	unsigned	length;
	Size		tid_rel;
	const char *p;
	const char *end;
	uintptr_t	hdr_cl;
	uintptr_t	cl;
	uintptr_t	last_cl;

	if (unlikely(neighborOffno < FirstOffsetNumber ||
				 neighborOffno > (BLCKSZ - SizeOfPageHeaderData) / sizeof(ItemIdData)))
		return;

	page = CalicoFramePageBase(buf_blocks, frame_id);
	item = *(volatile ItemIdData *) PageGetItemId(page, neighborOffno);
	if (unlikely(!ItemIdIsNormal(&item)))
		return;

	offset = ItemIdGetOffset(&item);
	length = ItemIdGetLength(&item);
	if (unlikely(offset < SizeOfPageHeaderData ||
				 length < offsetof(HnswNeighborTupleData, indextids) ||
				 offset + length > BLCKSZ))
		return;

	p = (const char *) page + offset;
	hdr_cl = (uintptr_t) p & ~(uintptr_t) 63;
	__builtin_prefetch((const void *) hdr_cl, 0, 3);

	tid_rel = offsetof(HnswNeighborTupleData, indextids) +
		(Size) level * (Size) m * sizeof(ItemPointerData);
	if (unlikely(tid_rel >= length))
		return;

	end = p + Min((Size) length, tid_rel + (Size) lm * sizeof(ItemPointerData));
	cl = (uintptr_t) (p + tid_rel) & ~(uintptr_t) 63;
	if (cl == hdr_cl)
		cl += 64;
	last_cl = (uintptr_t) (end - 1) & ~(uintptr_t) 63;

	for (; cl <= last_cl; cl += 64)
		__builtin_prefetch((const void *) cl, 0, 3);
}

static pg_attribute_always_inline void
HnswSteppedPrefetchAdvance(int i, int count, const HnswUnvisited *unvisited,
						   const CalicoOptimisticSlot *slots,
						   HnswElementTuple *etups, uint16 *limits,
						   Size tuple_bytes, Size min_valid_bytes,
						   const char *buf_blocks)
{
	if (i + HNSW_PREFETCH_STEP_LINP < count)
	{
		int			idx8 = i + HNSW_PREFETCH_STEP_LINP;

		if (likely(slots[idx8].opt_valid))
		{
			OffsetNumber off8 = ItemPointerGetOffsetNumber(&unvisited[idx8].indextid);

			if (likely(off8 >= FirstOffsetNumber &&
					   off8 <= (BLCKSZ - SizeOfPageHeaderData) / sizeof(ItemIdData)))
			{
				Page		page8 = CalicoFramePageBase(buf_blocks, slots[idx8].frame_id);

				__builtin_prefetch(PageGetItemId(page8, off8), 0, 3);
			}
		}
	}
	if (i + HNSW_PREFETCH_STEP_STAGEA < count)
	{
		int			idx4 = i + HNSW_PREFETCH_STEP_STAGEA;

		etups[idx4] = HnswPrefetchTupleStageA(buf_blocks, &slots[idx4],
											  ItemPointerGetOffsetNumber(&unvisited[idx4].indextid),
											  tuple_bytes, min_valid_bytes, &limits[idx4]);
	}
	if (i + HNSW_PREFETCH_STEP_STAGEB < count)
	{
		int			idx2 = i + HNSW_PREFETCH_STEP_STAGEB;

		HnswPrefetchTupleStageB(etups[idx2], limits[idx2]);
	}
	if (i + HNSW_PREFETCH_STEP_STAGEC < count)
	{
		int			idx1 = i + HNSW_PREFETCH_STEP_STAGEC;

		HnswPrefetchTupleStageC(etups[idx1], limits[idx1]);
	}
}

#endif							/* HNSWPREFETCH_H */
