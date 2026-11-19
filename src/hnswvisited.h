/* Visited sets for HNSW traversal. */
#ifndef HNSWVISITED_H
#define HNSWVISITED_H

#include "postgres.h"

#include "storage/itemptr.h"
#include "utils/memutils.h"

typedef struct HnswFastVisited
{
	uint64	   *slots;
	uint32		mask;			/* capacity - 1 (power of 2), 0 if uninitialized */
	uint32		count;			/* entries inserted in current generation */
	uint32		grow_at;		/* resize threshold */
	MemoryContext alloc_ctx;	/* context owning heap slots */
	uint16		gen;			/* active 16-bit generation tag (1..65535) */
	bool		on_stack;		/* true if slots points to stack buffer */
}			HnswFastVisited;

void		HnswAcquireScanVisited(HnswFastVisited *fv, MemoryContext scanCtx, MemoryContext tmpCtx, MemoryContextCallback *cb);
void		HnswReleaseScanVisited(HnswFastVisited *fv);

extern __attribute__((visibility("hidden"))) pg_noinline void
hnsw_fast_visited_grow(HnswFastVisited *fv, uint32 extra);
extern __attribute__((visibility("hidden"))) void
hnsw_fast_visited_init(HnswFastVisited *fv, int ef, int m,
					   uint64 *stack_buf, uint32 stack_cap);

static inline uint32
hnsw_hash_tid48(uint64 tid48)
{
	uint64		h = tid48 ^ (tid48 >> 25);

	h *= UINT64CONST(0xbf58476d1ce4e5b9);
	return (uint32) (h ^ (h >> 31));
}

static inline bool
hnsw_fast_visited_insert(HnswFastVisited *fv, BlockNumber blkno, OffsetNumber offno)
{
	uint64		tid48 = ((uint64) blkno << 16) | (uint64) offno;
	uint64		target = ((uint64) fv->gen << 48) | tid48;
	uint64	   *slots;
	uint32		mask;
	uint32		pos;

	if (unlikely(fv->count >= fv->grow_at))
		hnsw_fast_visited_grow(fv, 1);

	slots = fv->slots;
	mask = fv->mask;
	pos = hnsw_hash_tid48(tid48) & mask;

	for (;;)
	{
		uint64		cur = slots[pos];

		if (cur == target)
			return true;
		if ((uint16) (cur >> 48) != fv->gen)
		{
			slots[pos] = target;
			fv->count++;
			return false;
		}
		pos = (pos + 1) & mask;
	}
}

#endif							/* HNSWVISITED_H */
