/* Generation-stamped HNSW visited table and backend cache. */
#include "postgres.h"

#include "hnswvisited.h"
#include "utils/memdebug.h"
#include "utils/memutils.h"

#define HNSW_BACKEND_VISITED_MAX_CAP (524288U)	/* 4 MB retained per backend */
#define HNSW_VISITED_GROW_AT(cap) (((cap) * 25U) >> 5)

typedef struct HnswBackendVisitedCache
{
	uint64	   *slots;
	uint32		cap;
	uint16		gen;
	HnswFastVisited *owner;
}			HnswBackendVisitedCache;

static HnswBackendVisitedCache hnsw_backend_visited;

static void
HnswReleaseBackendVisitedCb(void *arg)
{
	HnswFastVisited *fv = arg;

	/* The scan may have moved to a private table since it borrowed the cache. */
	if (hnsw_backend_visited.owner == fv)
	{
		hnsw_backend_visited.owner = NULL;
		fv->slots = NULL;
	}
}

void
HnswAcquireScanVisited(HnswFastVisited *fv, MemoryContext scanCtx, MemoryContext tmpCtx,
					   MemoryContextCallback *cb)
{
	if (fv->alloc_ctx == NULL)
		fv->alloc_ctx = scanCtx;
	fv->count = 0;

	/* A scan with a private table keeps it across rescans. */
	if (fv->slots != NULL || hnsw_backend_visited.owner != NULL)
		return;

	hnsw_backend_visited.owner = fv;
	fv->on_stack = false;
	fv->slots = hnsw_backend_visited.slots;
	fv->mask = hnsw_backend_visited.cap > 0 ? hnsw_backend_visited.cap - 1 : 0;
	fv->grow_at = HNSW_VISITED_GROW_AT(hnsw_backend_visited.cap);
	fv->gen = hnsw_backend_visited.gen;

	cb->func = HnswReleaseBackendVisitedCb;
	cb->arg = fv;
	MemoryContextRegisterResetCallback(tmpCtx, cb);
}

void
HnswReleaseScanVisited(HnswFastVisited *fv)
{
	if (hnsw_backend_visited.owner == fv)
		HnswReleaseBackendVisitedCb(fv);
	else if (fv->slots != NULL && !fv->on_stack)
	{
		pfree(fv->slots);
		fv->slots = NULL;
	}
	fv->mask = 0;
	fv->count = 0;
	fv->grow_at = 0;
}

static inline uint32
hnsw_next_pow2_u32(uint32 v)
{
	uint32		cap = 128;

	while (cap < v && cap < (1U << 24))
		cap <<= 1;
	return cap;
}

pg_noinline void
hnsw_fast_visited_grow(HnswFastVisited *fv, uint32 extra)
{
	uint32		old_cap = fv->mask + 1;
	uint32		new_cap = (old_cap >= 16384U && old_cap < 262144U) ? (old_cap << 2) : (old_cap << 1);
	uint64	   *new_slots;
	uint64	   *old_slots = fv->slots;
	uint32		new_mask;
	uint16		gen = fv->gen;
	bool		was_backend_cache = hnsw_backend_visited.owner == fv;
	bool		keep_backend_cache;
	MemoryContext ctx;

	while (HNSW_VISITED_GROW_AT(new_cap) < fv->count + extra && new_cap < (1U << 26))
		new_cap <<= 1;
	keep_backend_cache = was_backend_cache && new_cap <= HNSW_BACKEND_VISITED_MAX_CAP;

	/*
	 * Fast in-place promotion within the retained 4 MB backend buffer via
	 * single-pass eviction rehash under a new generation tag.
	 */
	if (was_backend_cache &&
		hnsw_backend_visited.slots != NULL &&
		old_slots == hnsw_backend_visited.slots &&
		new_cap <= hnsw_backend_visited.cap &&
		likely(hnsw_backend_visited.gen < UINT16_MAX))
	{
		uint16		next_gen = (uint16) (hnsw_backend_visited.gen + 1U);
		uint64		new_gen_prefix = (uint64) next_gen << 48;

		hnsw_backend_visited.gen = next_gen;
		new_mask = new_cap - 1;

		for (uint32 i = 0; i < old_cap; i++)
		{
			uint64		val = old_slots[i];

			if ((uint16) (val >> 48) != gen)
				continue;

			old_slots[i] = 0;
			val &= UINT64CONST(0x0000ffffffffffff);
			for (;;)
			{
				uint32		pos = hnsw_hash_tid48(val) & new_mask;
				uint64		displaced;

				while ((uint16) (old_slots[pos] >> 48) == next_gen)
					pos = (pos + 1) & new_mask;

				displaced = old_slots[pos];
				old_slots[pos] = new_gen_prefix | val;
				if (pos > i && pos < old_cap && (uint16) (displaced >> 48) == gen)
				{
					val = displaced & UINT64CONST(0x0000ffffffffffff);
					continue;
				}
				break;
			}
		}

		fv->mask = new_mask;
		fv->grow_at = HNSW_VISITED_GROW_AT(new_cap);
		fv->gen = next_gen;
		return;
	}

	if (keep_backend_cache)
		ctx = TopMemoryContext;
	else
	{
		ctx = fv->alloc_ctx != NULL ? fv->alloc_ctx : CurrentMemoryContext;
		fv->alloc_ctx = ctx;
	}

	new_slots = (uint64 *) MemoryContextAllocZero(ctx, (Size) new_cap * sizeof(uint64));
	new_mask = new_cap - 1;

	for (uint32 i = 0; i < old_cap; i++)
	{
		uint64		val = old_slots[i];

		if ((uint16) (val >> 48) == gen)
		{
			uint64		tid48 = val & UINT64CONST(0x0000ffffffffffff);
			uint32		pos = hnsw_hash_tid48(tid48) & new_mask;

			while (new_slots[pos] != 0)
				pos = (pos + 1) & new_mask;
			new_slots[pos] = val;
		}
	}

	fv->slots = new_slots;
	fv->mask = new_mask;
	fv->grow_at = HNSW_VISITED_GROW_AT(new_cap);
	fv->gen = gen;

	if (keep_backend_cache)
	{
		hnsw_backend_visited.slots = new_slots;
		hnsw_backend_visited.cap = new_cap;
		hnsw_backend_visited.gen = gen;
		fv->on_stack = false;
		pfree(old_slots);
	}
	else if (was_backend_cache)
		hnsw_backend_visited.owner = NULL;
	else if (!fv->on_stack)
		pfree(old_slots);
	else
		fv->on_stack = false;
}

void
hnsw_fast_visited_init(HnswFastVisited *fv, int ef, int m,
					   uint64 *stack_buf, uint32 stack_cap)
{
	uint32		min_cap = (uint32) Max((Size) ef * m * 4, (Size) 128);
	uint32		desired = hnsw_next_pow2_u32(min_cap);

	if (hnsw_backend_visited.owner == fv && hnsw_backend_visited.slots != NULL)
	{
		if (desired <= hnsw_backend_visited.cap)
		{
			uint16		next_gen;

			if (unlikely(hnsw_backend_visited.gen >= 0xfff0U))
			{
				memset(hnsw_backend_visited.slots, 0,
					   (Size) hnsw_backend_visited.cap * sizeof(uint64));
				next_gen = 1;
			}
			else
				next_gen = (uint16) (hnsw_backend_visited.gen + 1U);

			hnsw_backend_visited.gen = next_gen;
			fv->slots = hnsw_backend_visited.slots;
			fv->mask = desired - 1;
			fv->grow_at = HNSW_VISITED_GROW_AT(desired);
			fv->gen = next_gen;
			fv->count = 0;
			fv->on_stack = false;
			return;
		}
		if (desired <= HNSW_BACKEND_VISITED_MAX_CAP)
		{
			uint64	   *new_slots = (uint64 *) MemoryContextAllocZero(TopMemoryContext,
																	(Size) desired * sizeof(uint64));
			uint64	   *old_slots = hnsw_backend_visited.slots;

			hnsw_backend_visited.slots = new_slots;
			hnsw_backend_visited.cap = desired;
			hnsw_backend_visited.gen = 1;
			fv->slots = new_slots;
			fv->mask = desired - 1;
			fv->grow_at = HNSW_VISITED_GROW_AT(desired);
			fv->gen = 1;
			fv->count = 0;
			fv->on_stack = false;
			pfree(old_slots);
			return;
		}
		HnswReleaseBackendVisitedCb(fv);
	}
	else if (fv->slots != NULL)
	{
		uint32		cur_cap = fv->mask + 1;

		if (cur_cap >= desired)
		{
			fv->gen++;
			if (unlikely(fv->gen == 0))
			{
				memset(fv->slots, 0, (Size) cur_cap * sizeof(uint64));
				fv->gen = 1;
			}
			fv->count = 0;
			return;
		}

		if (!fv->on_stack)
			pfree(fv->slots);

		fv->slots = NULL;
	}

	if (stack_buf != NULL && stack_cap >= desired)
	{
		HnswReleaseBackendVisitedCb(fv);
		memset(stack_buf, 0, (Size) desired * sizeof(uint64));
		fv->slots = stack_buf;
		fv->mask = desired - 1;
		fv->grow_at = HNSW_VISITED_GROW_AT(desired);
		fv->gen = 1;
		fv->count = 0;
		fv->on_stack = true;
		if (fv->alloc_ctx == NULL)
			fv->alloc_ctx = CurrentMemoryContext;
		return;
	}

	if (hnsw_backend_visited.owner == fv)
	{
		if (desired <= HNSW_BACKEND_VISITED_MAX_CAP)
		{
			uint64	   *new_slots = (uint64 *) MemoryContextAllocZero(TopMemoryContext,
																	(Size) desired * sizeof(uint64));

			hnsw_backend_visited.slots = new_slots;
			hnsw_backend_visited.cap = desired;
			hnsw_backend_visited.gen = 1;
			fv->slots = new_slots;
			fv->mask = desired - 1;
			fv->grow_at = HNSW_VISITED_GROW_AT(desired);
			fv->gen = 1;
			fv->count = 0;
			fv->on_stack = false;
			return;
		}
		HnswReleaseBackendVisitedCb(fv);
	}

	{
		MemoryContext ctx = fv->alloc_ctx != NULL ? fv->alloc_ctx : CurrentMemoryContext;

		fv->alloc_ctx = ctx;
		fv->slots = (uint64 *) MemoryContextAllocZero(ctx, (Size) desired * sizeof(uint64));
		fv->mask = desired - 1;
		fv->grow_at = HNSW_VISITED_GROW_AT(desired);
		fv->gen = 1;
		fv->count = 0;
		fv->on_stack = false;
	}
}
