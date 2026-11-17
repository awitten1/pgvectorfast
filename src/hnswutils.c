#include "postgres.h"
#include "calico_internal.h"
#include <math.h>

#include "access/genam.h"
#include "access/generic_xlog.h"
#include "common/hashfn.h"
#include "fmgr.h"
extern Datum vector_l2_squared_distance(PG_FUNCTION_ARGS);
extern Datum vector_negative_inner_product(PG_FUNCTION_ARGS);
#include "hnsw.h"
#include "lib/pairingheap.h"
#include "nodes/pg_list.h"
#include "port/atomics.h"
#include "sparsevec.h"
#include "storage/bufmgr.h"
#include "storage/buf_internals.h"
#include "storage/smgr.h"
#include "utils/datum.h"
#include "utils/fmgrprotos.h"
#include "utils/memdebug.h"
#include "utils/memutils.h"
#include "utils/rel.h"
#include "vector.h"
#include "vector_simd.h"

#if PG_VERSION_NUM >= 160000
#include "varatt.h"
#endif

#if PG_VERSION_NUM >= 160000
#include "varatt.h"
#endif

#if PG_VERSION_NUM < 170000
static inline uint64
murmurhash64(uint64 data)
{
	uint64		h = data;

	h ^= h >> 33;
	h *= 0xff51afd7ed558ccd;
	h ^= h >> 33;
	h *= 0xc4ceb9fe1a85ec53;
	h ^= h >> 33;

	return h;
}
#endif

/* TID hash table */
static inline uint32
hash_tid(ItemPointerData tid)
{
	return murmurhash64(((uint64) BlockIdGetBlockNumber(&tid.ip_blkid) << 16) |
						tid.ip_posid);
}

static inline bool
tid_equal(ItemPointerData a, ItemPointerData b)
{
	return BlockIdGetBlockNumber(&a.ip_blkid) == BlockIdGetBlockNumber(&b.ip_blkid) &&
		a.ip_posid == b.ip_posid;
}

#define SH_PREFIX		tidhash
#define SH_ELEMENT_TYPE	TidHashEntry
#define SH_KEY_TYPE		ItemPointerData
#define	SH_KEY			tid
#define SH_HASH_KEY(tb, key)	hash_tid(key)
#define SH_EQUAL(tb, a, b)		tid_equal(a, b)
#define	SH_SCOPE		extern
#define SH_DEFINE
#include "lib/simplehash.h"

/* Pointer hash table */
static uint32
hash_pointer(uintptr_t ptr)
{
#if SIZEOF_VOID_P == 8
	return murmurhash64((uint64) ptr);
#else
	return murmurhash32((uint32) ptr);
#endif
}

#define SH_PREFIX		pointerhash
#define SH_ELEMENT_TYPE	PointerHashEntry
#define SH_KEY_TYPE		uintptr_t
#define	SH_KEY			ptr
#define SH_HASH_KEY(tb, key)	hash_pointer(key)
#define SH_EQUAL(tb, a, b)		(a == b)
#define	SH_SCOPE		extern
#define SH_DEFINE
#include "lib/simplehash.h"

/* Offset hash table */
static uint32
hash_offset(Size offset)
{
#if SIZEOF_SIZE_T == 8
	return murmurhash64((uint64) offset);
#else
	return murmurhash32((uint32) offset);
#endif
}

#define SH_PREFIX		offsethash
#define SH_ELEMENT_TYPE	OffsetHashEntry
#define SH_KEY_TYPE		Size
#define	SH_KEY			offset
#define SH_HASH_KEY(tb, key)	hash_offset(key)
#define SH_EQUAL(tb, a, b)		(a == b)
#define	SH_SCOPE		extern
#define SH_DEFINE
#include "lib/simplehash.h"

/*
 * Get the max number of connections in an upper layer for each element in the index
 */
int
HnswGetM(Relation index)
{
	HnswOptions *opts = (HnswOptions *) index->rd_options;

	if (opts)
		return opts->m;

	return HNSW_DEFAULT_M;
}

/*
 * Get the size of the dynamic candidate list in the index
 */
int
HnswGetEfConstruction(Relation index)
{
	HnswOptions *opts = (HnswOptions *) index->rd_options;

	if (opts)
		return opts->efConstruction;

	return HNSW_DEFAULT_EF_CONSTRUCTION;
}

/*
 * Get proc
 */
FmgrInfo *
HnswOptionalProcInfo(Relation index, uint16 procnum)
{
	if (!OidIsValid(index_getprocid(index, 1, procnum)))
		return NULL;

	return index_getprocinfo(index, 1, procnum);
}

/*
 * Init support functions
 */
void
HnswInitSupport(HnswSupport * support, Relation index)
{
	support->procinfo = index_getprocinfo(index, 1, HNSW_DISTANCE_PROC);
	support->vectorDistance = support->procinfo->fn_addr == vector_l2_squared_distance
		? HNSW_VECTOR_DISTANCE_L2
		: (support->procinfo->fn_addr == vector_negative_inner_product
		   ? HNSW_VECTOR_DISTANCE_INNER_PRODUCT : HNSW_VECTOR_DISTANCE_NONE);
	support->collation = index->rd_indcollation[0];
	support->normprocinfo = HnswOptionalProcInfo(index, HNSW_NORM_PROC);
	calico_read_handle_init(&support->read_handle, &index->rd_locator, MAIN_FORKNUM);
}

/*
 * Normalize value
 */
Datum
HnswNormValue(const HnswTypeInfo * typeInfo, Oid collation, Datum value)
{
	return DirectFunctionCall1Coll(typeInfo->normalize, collation, value);
}

/*
 * Check if non-zero norm
 */
bool
HnswCheckNorm(HnswSupport * support, Datum value)
{
	return DatumGetFloat8(FunctionCall1Coll(support->normprocinfo, support->collation, value)) > 0;
}

/*
 * New buffer
 */
Buffer
HnswNewBuffer(Relation index, ForkNumber forkNum)
{
	Buffer		buf = ReadBufferExtended(index, forkNum, P_NEW, RBM_NORMAL, NULL);

	LockBuffer(buf, BUFFER_LOCK_EXCLUSIVE);
	return buf;
}

/*
 * Init page
 */
void
HnswInitPage(Buffer buf, Page page)
{
	PageInit(page, BufferGetPageSize(buf), sizeof(HnswPageOpaqueData));
	HnswPageGetOpaque(page)->nextblkno = InvalidBlockNumber;
	HnswPageGetOpaque(page)->page_id = HNSW_PAGE_ID;
}

/*
 * Allocate a neighbor array
 */
HnswNeighborArray *
HnswInitNeighborArray(int lm, HnswAllocator * allocator)
{
	HnswNeighborArray *a = HnswAlloc(allocator, HNSW_NEIGHBOR_ARRAY_SIZE(lm));

	a->length = 0;
	a->closerSet = false;
	return a;
}

/*
 * Allocate neighbors
 */
void
HnswInitNeighbors(char *base, HnswElement element, int m, HnswAllocator * allocator)
{
	int			level = element->level;
	HnswNeighborArrayPtr *neighborList = (HnswNeighborArrayPtr *) HnswAlloc(allocator, mul_size(sizeof(HnswNeighborArrayPtr), add_size(level, 1)));

	HnswPtrStore(base, element->neighbors, neighborList);

	for (int lc = 0; lc <= level; lc++)
		HnswPtrStore(base, neighborList[lc], HnswInitNeighborArray(HnswGetLayerM(m, lc), allocator));
}

/*
 * Allocate memory from the allocator
 */
void *
HnswAlloc(HnswAllocator * allocator, Size size)
{
	if (allocator)
		return (*(allocator)->alloc) (size, (allocator)->state);

	return palloc(size);
}

/*
 * Allocate an element
 */
HnswElement
HnswInitElement(char *base, ItemPointer heaptid, int m, double ml, int maxLevel, HnswAllocator * allocator)
{
	HnswElement element = HnswAlloc(allocator, sizeof(HnswElementData));

	double		uniform = RandomDouble();
	int			level = uniform == 0.0 ? maxLevel : (int) (-log(uniform) * ml);

	/* Cap level */
	if (level > maxLevel)
		level = maxLevel;

	element->heaptidsLength = 0;
	HnswAddHeapTid(element, heaptid);

	element->level = level;
	element->deleted = 0;
	/* Start at one to make it easier to find issues */
	element->version = 1;

	HnswInitNeighbors(base, element, m, allocator);

	HnswPtrStore(base, element->value, (char *) NULL);

	return element;
}

/*
 * Add a heap TID to an element
 */
void
HnswAddHeapTid(HnswElement element, ItemPointer heaptid)
{
	element->heaptids[element->heaptidsLength++] = *heaptid;
}

/*
 * Allocate an element from block and offset numbers
 */
HnswElement
HnswInitElementFromBlock(BlockNumber blkno, OffsetNumber offno)
{
	HnswElement element = palloc_object(HnswElementData);
	char	   *base = NULL;

	element->blkno = blkno;
	element->offno = offno;
	HnswPtrStore(base, element->neighbors, (HnswNeighborArrayPtr *) NULL);
	HnswPtrStore(base, element->value, (char *) NULL);
	return element;
}

/*
 * Get the metapage info
 */
void
HnswGetMetaPageInfoWithHandle(Relation index, CalicoReadHandle *handle, int *m, int *dimensions, HnswElement * entryPoint)
{
	Buffer		buf;
	Page		page;
	HnswMetaPage metap;

	if (calico_enable_optimistic_reads && calico_is_active() &&
		!RelationUsesLocalBuffers(index))
	{
		uint64		version = 0;
		uint32		buf_id = 0;
		void	   *entry_ptr = NULL;
		bool		found;

		found = (handle != NULL) ?
			calico_optimistic_lookup_handle_frame(handle, HNSW_METAPAGE_BLKNO,
												 &version, &buf_id, &entry_ptr) :
			calico_optimistic_lookup_frame(RelationGetSmgr(index), MAIN_FORKNUM,
										   HNSW_METAPAGE_BLKNO,
										   &version, &buf_id, &entry_ptr);

		if (likely(found))
		{
			Page		opt_page = CalicoFramePage(buf_id);
			HnswMetaPage opt_metap = HnswPageGetMeta(opt_page);
			uint32		magic = opt_metap->magicNumber;
			int			mp_m = opt_metap->m;
			int			mp_dims = opt_metap->dimensions;
			BlockNumber ep_blk = opt_metap->entryBlkno;
			OffsetNumber ep_off = opt_metap->entryOffno;
			uint16		ep_level = opt_metap->entryLevel;

			if (likely(calico_optimistic_validate(entry_ptr, version) &&
					   magic == HNSW_MAGIC_NUMBER))
			{
				if (m != NULL)
					*m = mp_m;
				if (dimensions != NULL)
					*dimensions = mp_dims;
				if (entryPoint != NULL)
				{
					if (BlockNumberIsValid(ep_blk))
					{
						*entryPoint = HnswInitElementFromBlock(ep_blk, ep_off);
						(*entryPoint)->level = ep_level;
					}
					else
						*entryPoint = NULL;
				}
				return;
			}
		}
	}

	buf = ReadBuffer(index, HNSW_METAPAGE_BLKNO);
	LockBuffer(buf, BUFFER_LOCK_SHARE);
	page = BufferGetPage(buf);
	metap = HnswPageGetMeta(page);

	if (unlikely(metap->magicNumber != HNSW_MAGIC_NUMBER))
		elog(ERROR, "hnsw index is not valid");

	if (m != NULL)
		*m = metap->m;

	if (dimensions != NULL)
		*dimensions = metap->dimensions;

	if (entryPoint != NULL)
	{
		if (BlockNumberIsValid(metap->entryBlkno))
		{
			*entryPoint = HnswInitElementFromBlock(metap->entryBlkno, metap->entryOffno);
			(*entryPoint)->level = metap->entryLevel;
		}
		else
			*entryPoint = NULL;
	}

	UnlockReleaseBuffer(buf);
}

void
HnswGetMetaPageInfo(Relation index, int *m, int *dimensions, HnswElement * entryPoint)
{
	HnswGetMetaPageInfoWithHandle(index, NULL, m, dimensions, entryPoint);
}

/*
 * Get the entry point
 */
HnswElement
HnswGetEntryPoint(Relation index)
{
	HnswElement entryPoint;

	HnswGetMetaPageInfo(index, NULL, NULL, &entryPoint);

	return entryPoint;
}

/*
 * Update the metapage info
 */
static void
HnswUpdateMetaPageInfo(Page page, int updateEntry, HnswElement entryPoint, BlockNumber insertPage)
{
	HnswMetaPage metap = HnswPageGetMeta(page);

	if (updateEntry)
	{
		if (entryPoint == NULL)
		{
			metap->entryBlkno = InvalidBlockNumber;
			metap->entryOffno = InvalidOffsetNumber;
			metap->entryLevel = -1;
		}
		else if (entryPoint->level > metap->entryLevel || updateEntry == HNSW_UPDATE_ENTRY_ALWAYS)
		{
			metap->entryBlkno = entryPoint->blkno;
			metap->entryOffno = entryPoint->offno;
			metap->entryLevel = entryPoint->level;
		}
	}

	if (BlockNumberIsValid(insertPage))
		metap->insertPage = insertPage;
}

/*
 * Update the metapage
 */
void
HnswUpdateMetaPage(Relation index, int updateEntry, HnswElement entryPoint, BlockNumber insertPage, ForkNumber forkNum, bool building)
{
	Buffer		buf;
	Page		page;
	GenericXLogState *state;

	buf = ReadBufferExtended(index, forkNum, HNSW_METAPAGE_BLKNO, RBM_NORMAL, NULL);
	LockBuffer(buf, BUFFER_LOCK_EXCLUSIVE);
	if (building)
	{
		state = NULL;
		page = BufferGetPage(buf);
	}
	else
	{
		state = GenericXLogStart(index);
		page = GenericXLogRegisterBuffer(state, buf, 0);
	}

	HnswUpdateMetaPageInfo(page, updateEntry, entryPoint, insertPage);

	if (building)
		MarkBufferDirty(buf);
	else
		GenericXLogFinish(state);
	UnlockReleaseBuffer(buf);
}

/*
 * Form index value
 */
bool
HnswFormIndexValue(Datum *out, Datum *values, bool *isnull, const HnswTypeInfo * typeInfo, HnswSupport * support)
{
	/* Detoast once for all calls */
	Datum		value = PointerGetDatum(PG_DETOAST_DATUM(values[0]));

	/* Check value */
	if (typeInfo->checkValue != NULL)
		typeInfo->checkValue(DatumGetPointer(value));

	/* Normalize if needed */
	if (support->normprocinfo != NULL)
	{
		if (!HnswCheckNorm(support, value))
			return false;

		value = HnswNormValue(typeInfo, support->collation, value);
	}

	*out = value;

	return true;
}

/*
 * Set element tuple, except for neighbor info
 */
void
HnswSetElementTuple(char *base, HnswElementTuple etup, HnswElement element)
{
	Pointer		valuePtr = HnswPtrAccess(base, element->value);

	etup->type = HNSW_ELEMENT_TUPLE_TYPE;
	etup->level = element->level;
	etup->deleted = 0;
	etup->version = element->version;
	for (int i = 0; i < HNSW_HEAPTIDS; i++)
	{
		if (i < element->heaptidsLength)
			etup->heaptids[i] = element->heaptids[i];
		else
			ItemPointerSetInvalid(&etup->heaptids[i]);
	}
	memcpy(&etup->data, valuePtr, VARSIZE_ANY(valuePtr));
}

/*
 * Set neighbor tuple
 */
void
HnswSetNeighborTuple(char *base, HnswNeighborTuple ntup, HnswElement e, int m)
{
	int			idx = 0;

	ntup->type = HNSW_NEIGHBOR_TUPLE_TYPE;

	for (int lc = e->level; lc >= 0; lc--)
	{
		HnswNeighborArray *neighbors = HnswGetNeighbors(base, e, lc);
		int			lm = HnswGetLayerM(m, lc);

		for (int i = 0; i < lm; i++)
		{
			ItemPointer indextid = &ntup->indextids[idx++];

			if (i < neighbors->length)
			{
				HnswCandidate *hc = &neighbors->items[i];
				HnswElement hce = HnswPtrAccess(base, hc->element);

				ItemPointerSet(indextid, hce->blkno, hce->offno);
			}
			else
				ItemPointerSetInvalid(indextid);
		}
	}

	ntup->count = idx;
	ntup->version = e->version;
}

/*
 * Load an element from a tuple
 */
void
HnswLoadElementFromTuple(HnswElement element, HnswElementTuple etup, bool loadHeaptids, bool loadVec)
{
	element->level = etup->level;
	element->deleted = etup->deleted;
	element->version = etup->version;
	element->neighborPage = ItemPointerGetBlockNumber(&etup->neighbortid);
	element->neighborOffno = ItemPointerGetOffsetNumber(&etup->neighbortid);
	element->heaptidsLength = 0;

	if (loadHeaptids)
	{
		for (int i = 0; i < HNSW_HEAPTIDS; i++)
		{
			/* Can stop at first invalid */
			if (!ItemPointerIsValid(&etup->heaptids[i]))
				break;

			HnswAddHeapTid(element, &etup->heaptids[i]);
		}
	}

	if (loadVec)
	{
		char	   *base = NULL;
		Datum		value = datumCopy(PointerGetDatum(&etup->data), false, -1);

		HnswPtrStore(base, element->value, (char *) DatumGetPointer(value));
	}
}

/*
 * Calculate the distance between values
 */
static inline double
HnswGetDistance(Datum a, Datum b, HnswSupport * support)
{
	return DatumGetFloat8(FunctionCall2Coll(support->procinfo, support->collation, a, b));
}

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
static inline bool
HnswGetDistanceOptimistic(HnswSupport *support, const Vector *a_vec, const Vector *b_ptr, Size vec_size, double *distance_out)
{
	uint32		vl_len = __atomic_load_n((const uint32 *) &b_ptr->vl_len_, __ATOMIC_RELAXED);
	int16		b_dim = __atomic_load_n((const int16 *) &b_ptr->dim, __ATOMIC_RELAXED);

	if (unlikely(b_dim != a_vec->dim || b_dim <= 0 ||
				 (vl_len >> 2) != (uint32) vec_size || (vl_len & 0x03) != 0))
		return false;

	Assert(support->vectorDistance != HNSW_VECTOR_DISTANCE_NONE);
	if (support->vectorDistance == HNSW_VECTOR_DISTANCE_L2)
		*distance_out = (double) VectorL2SquaredDistance(b_dim, a_vec->x, b_ptr->x);
	else
		*distance_out = (double) -VectorInnerProduct(b_dim, a_vec->x, b_ptr->x);
	return true;
}

/*
 * Load element data optimistically from tuple
 * This is like HnswLoadElementFromTuple but makes defensive copies for optimistic reads
 */
static inline void
HnswLoadElementFromTupleOptimistic(HnswElement element, HnswElementTuple etup, 
								   bool loadHeaptids, bool loadVec, Page page)
{
	/* Read basic fields - these are small and atomic on most architectures */
	element->level = etup->level;
	element->deleted = etup->deleted;
	element->version = etup->version;
	element->neighborPage = ItemPointerGetBlockNumber(&etup->neighbortid);
	element->neighborOffno = ItemPointerGetOffsetNumber(&etup->neighbortid);
	element->heaptidsLength = 0;
	
	if (loadHeaptids)
	{
		for (int i = 0; i < HNSW_HEAPTIDS; i++)
		{
			/* Can stop at first invalid */
			if (!ItemPointerIsValid(&etup->heaptids[i]))
				break;

			HnswAddHeapTid(element, &etup->heaptids[i]);
		}
	}
	
	if (loadVec)
	{
		char	   *base = NULL;
		Vector	   *vec_ptr = (Vector *) &etup->data;
		Size		varlena_len;
		Vector	   *vec_copy;
		char	   *page_start;
		char	   *page_end;
		char	   *vec_end;
		
		/* Read varlena header with memory barrier */
		pg_read_barrier();
		varlena_len = VARSIZE_ANY(vec_ptr);
		
		/* Calculate page bounds */
		page_start = (char *) page;
		page_end = page_start + BLCKSZ;
		vec_end = ((char *) vec_ptr) + varlena_len;
		
		/* Validate varlena length is within page bounds */
		if (varlena_len < VARHDRSZ || 
		    varlena_len > BLCKSZ ||
		    ((char *) vec_ptr) < page_start ||
		    ((char *) vec_ptr) >= page_end ||
		    vec_end > page_end)
		{
			/* Invalid varlena header - will be caught by validation */
			HnswPtrStore(base, element->value, (Pointer) NULL);
			return;
		}
		
		/* Make a proper copy of the entire varlena datum */
		vec_copy = (Vector *) palloc(varlena_len);
		memcpy(vec_copy, vec_ptr, varlena_len);
		
		HnswPtrStore(base, element->value, (Pointer) vec_copy);
	}
}

/*
 * Try loading and/or scoring an element optimistically from a resolved slot.
 * Precondition: optimistic vector distance path is enabled and !loadVec.
 */
static pg_attribute_always_inline bool
HnswLoadElementOptimisticSlot(HnswSupport *support, BlockNumber blkno, OffsetNumber offno,
							  const CalicoOptimisticSlot *slot,
							  double *distance, const Vector *a_vec,
							  Size vec_size, double *maxDistance,
							  HnswElement *element)
{
	Page		page;
	HnswElementTuple etup;
	bool		was_element_null;
	bool		should_load;
	bool		validation_passed;

	if (unlikely(slot == NULL || !slot->opt_valid || slot->entry_ptr == NULL))
		return false;

	page = BufferGetPage(slot->frame_id + 1);
	was_element_null = (*element == NULL);

	if (distance != NULL && a_vec != NULL)
	{
		etup = HnswOptimisticItem(page, offno,
								  offsetof(HnswElementTupleData, data) + vec_size);
		if (unlikely(etup == NULL || etup->deleted))
			return false;
		if (unlikely(!HnswGetDistanceOptimistic(support, a_vec, &etup->data, vec_size, distance)))
			return false;
	}
	else
	{
		etup = HnswOptimisticItem(page, offno, offsetof(HnswElementTupleData, data));
		if (unlikely(etup == NULL || etup->deleted))
			return false;
		if (distance != NULL)
			*distance = 0;
	}

	should_load = (distance == NULL || maxDistance == NULL || *distance < *maxDistance);
	if (should_load)
	{
		if (*element == NULL)
			*element = HnswInitElementFromBlock(blkno, offno);
		HnswLoadElementFromTuple(*element, etup, true, false);
	}

	validation_passed = calico_optimistic_validate(slot->entry_ptr, slot->version);
	if (likely(validation_passed))
		return true;

	if (should_load && was_element_null)
		*element = NULL;

	return false;
}

/*
 * Load an element and optionally get its distance from q
 */
static pg_noinline void
HnswLoadElementImpl(BlockNumber blkno, OffsetNumber offno, double *distance, HnswQuery * q, Relation index, HnswSupport * support, bool loadVec, double *maxDistance, HnswElement * element)
{
	Buffer		buf;
	Page		page;
	HnswElementTuple etup;

	/*
	 * Try optimistic read path if enabled and buffer manager supports it.
	 * This avoids taking buffer locks for read-only access.
	 */
	if (calico_enable_optimistic_reads && !loadVec &&
	    support->vectorDistance != HNSW_VECTOR_DISTANCE_NONE &&
	    calico_is_active() && !RelationUsesLocalBuffers(index))
	{
		const Vector *a_vec = (const Vector *) DatumGetPointer(q->value);
		Size		vec_size = 0;
		bool		found = false;
		uint64		version = 0;
		uint32		buf_id = 0;
		void	   *entry_ptr = NULL;

		if (distance != NULL && a_vec != NULL)
		{
			if (unlikely(a_vec->dim <= 0))
				goto optimistic_element_failed;
			vec_size = VECTOR_SIZE(a_vec->dim);
			if (unlikely(vec_size > BLCKSZ - offsetof(HnswElementTupleData, data)))
				goto optimistic_element_failed;
		}

		/* Reuse this operation's translation region; validate every read. */
		found = calico_optimistic_lookup_handle_frame(
			&support->read_handle,
			blkno,
			&version,
			&buf_id,
			&entry_ptr
		);

		if (found)
		{
			CalicoOptimisticSlot slot;

			slot.entry_ptr = (CalicoTranslationEntry *) entry_ptr;
			slot.version = version;
			slot.frame_id = buf_id;
			slot.resident = true;
			slot.opt_valid = true;

			if (HnswLoadElementOptimisticSlot(support, blkno, offno, &slot,
											  distance, a_vec, vec_size,
											  maxDistance, element))
				return;
		}
optimistic_element_failed:
		calico_optimistic_element_fallbacks++;
	}
	
	/*
	 * Traditional locked path:
	 * Either optimistic read is disabled, not supported, or validation failed.
	 */
	/* Read vector */
	buf = ReadBuffer(index, blkno);
	LockBuffer(buf, BUFFER_LOCK_SHARE);
	page = BufferGetPage(buf);

	etup = (HnswElementTuple) PageGetItem(page, PageGetItemId(page, offno));

	Assert(HnswIsElementTuple(etup));

	if (unlikely(etup->deleted))
		elog(ERROR, "cannot load deleted element");

	/* Calculate distance */
	if (distance != NULL)
	{
		if (DatumGetPointer(q->value) == NULL)
			*distance = 0;
		else
			*distance = HnswGetDistance(q->value, PointerGetDatum(&etup->data), support);
	}

	/* Load element */
	if (distance == NULL || maxDistance == NULL || *distance < *maxDistance)
	{
		if (*element == NULL)
			*element = HnswInitElementFromBlock(blkno, offno);

		HnswLoadElementFromTuple(*element, etup, true, loadVec);
	}

	UnlockReleaseBuffer(buf);
}

/*
 * Load an element and optionally get its distance from q
 */
void
HnswLoadElement(HnswElement element, double *distance, HnswQuery * q, Relation index, HnswSupport * support, bool loadVec, double *maxDistance)
{
	/* No cached relation_id for this public wrapper */
	HnswLoadElementImpl(element->blkno, element->offno, distance, q, index, support, loadVec, maxDistance, &element);
}

/*
 * Get the distance for an element
 */
static double
GetElementDistance(char *base, HnswElement element, HnswQuery * q, HnswSupport * support)
{
	Datum		value = HnswGetValue(base, element);

	return HnswGetDistance(q->value, value, support);
}

/*
 * Allocate a search candidate
 */
static HnswSearchCandidate *
HnswInitSearchCandidate(char *base, HnswElement element, double distance)
{
	HnswSearchCandidate *sc = palloc_object(HnswSearchCandidate);

	HnswPtrStore(base, sc->element, element);
	sc->distance = distance;
	return sc;
}

/*
 * Create a candidate for the entry point
 */
HnswSearchCandidate *
HnswEntryCandidate(char *base, HnswElement entryPoint, HnswQuery * q, Relation index, HnswSupport * support, bool loadVec)
{
	bool		inMemory = index == NULL;
	double		distance;

	if (inMemory)
		distance = GetElementDistance(base, entryPoint, q, support);
	else
		HnswLoadElement(entryPoint, &distance, q, index, support, loadVec, NULL);

	return HnswInitSearchCandidate(base, entryPoint, distance);
}

/*
 * Compare candidate distances
 */
static int
CompareNearestCandidates(const pairingheap_node *a, const pairingheap_node *b, void *arg)
{
	if (HnswGetSearchCandidateConst(c_node, a)->distance < HnswGetSearchCandidateConst(c_node, b)->distance)
		return 1;

	if (HnswGetSearchCandidateConst(c_node, a)->distance > HnswGetSearchCandidateConst(c_node, b)->distance)
		return -1;

	return 0;
}

/*
 * Compare discarded candidate distances
 */
static int
CompareNearestDiscardedCandidates(const pairingheap_node *a, const pairingheap_node *b, void *arg)
{
	if (HnswGetSearchCandidateConst(w_node, a)->distance < HnswGetSearchCandidateConst(w_node, b)->distance)
		return 1;

	if (HnswGetSearchCandidateConst(w_node, a)->distance > HnswGetSearchCandidateConst(w_node, b)->distance)
		return -1;

	return 0;
}

/*
 * Compare candidate distances
 */
static int
CompareFurthestCandidates(const pairingheap_node *a, const pairingheap_node *b, void *arg)
{
	if (HnswGetSearchCandidateConst(w_node, a)->distance < HnswGetSearchCandidateConst(w_node, b)->distance)
		return -1;

	if (HnswGetSearchCandidateConst(w_node, a)->distance > HnswGetSearchCandidateConst(w_node, b)->distance)
		return 1;

	return 0;
}

#define HNSW_BACKEND_VISITED_MAX_CAP (131072U)	/* 1 MB retained per backend */

typedef struct HnswBackendVisitedCache
{
	uint64	   *slots;
	uint32		cap;
	uint16		gen;
	bool		in_use;
}			HnswBackendVisitedCache;

static HnswBackendVisitedCache hnsw_backend_visited = {NULL, 0, 0, false};

static void
HnswReleaseBackendVisitedCb(void *arg)
{
	hnsw_backend_visited.in_use = false;
}

void
HnswAcquireScanVisited(HnswFastVisited *fv, MemoryContext scanCtx, MemoryContext tmpCtx,
					   MemoryContextCallback *cb, bool *cbRegistered)
{
	fv->alloc_ctx = scanCtx;
	fv->count = 0;

	/*
	 * Nested/concurrent scans retain their own allocation across rescans
	 * instead of clobbering it.
	 */
	if (fv->slots != NULL && !fv->is_backend_cache)
		return;

	if (!hnsw_backend_visited.in_use)
	{
		hnsw_backend_visited.in_use = true;
		fv->is_backend_cache = true;
		fv->on_stack = false;
		fv->slots = hnsw_backend_visited.slots;
		fv->mask = hnsw_backend_visited.cap > 0 ? hnsw_backend_visited.cap - 1 : 0;
		fv->grow_at = (hnsw_backend_visited.cap * 3) / 4;
		fv->gen = hnsw_backend_visited.gen;

		if (!*cbRegistered)
		{
			cb->func = HnswReleaseBackendVisitedCb;
			cb->arg = NULL;
			MemoryContextRegisterResetCallback(tmpCtx, cb);
			*cbRegistered = true;
		}
	}
	else if (fv->is_backend_cache)
	{
		fv->is_backend_cache = false;
		fv->slots = NULL;
		fv->mask = 0;
		fv->grow_at = 0;
		fv->gen = 0;
	}
}

void
HnswReleaseScanVisited(HnswFastVisited *fv)
{
	if (fv->is_backend_cache)
	{
		hnsw_backend_visited.in_use = false;
		fv->slots = NULL;
		fv->is_backend_cache = false;
	}
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

static pg_noinline void
hnsw_fast_visited_grow(HnswFastVisited *fv, uint32 extra)
{
	uint32		old_cap = fv->mask + 1;
	uint32		new_cap = old_cap << 1;
	uint64	   *new_slots;
	uint64	   *old_slots = fv->slots;
	uint32		new_mask;
	uint16		gen = fv->gen;
	bool		was_backend_cache = fv->is_backend_cache;
	MemoryContext ctx;

	while ((new_cap * 3) / 4 < fv->count + extra && new_cap < (1U << 26))
		new_cap <<= 1;

	if (fv->is_backend_cache && new_cap <= HNSW_BACKEND_VISITED_MAX_CAP)
		ctx = TopMemoryContext;
	else
	{
		if (fv->is_backend_cache)
		{
			fv->is_backend_cache = false;
			hnsw_backend_visited.in_use = false;
		}
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
	fv->grow_at = (new_cap * 3) / 4;
	fv->gen = gen;

	if (fv->is_backend_cache)
	{
		hnsw_backend_visited.slots = new_slots;
		hnsw_backend_visited.cap = new_cap;
		hnsw_backend_visited.gen = gen;
		fv->on_stack = false;
		pfree(old_slots);
	}
	else if (!was_backend_cache && !fv->on_stack)
		pfree(old_slots);
	else
		fv->on_stack = false;
}

static void
hnsw_fast_visited_init(HnswFastVisited *fv, int ef, int m,
					   uint64 *stack_buf, uint32 stack_cap)
{
	uint32		min_cap = (uint32) Max((Size) ef * m * 4, (Size) 128);
	uint32		desired = hnsw_next_pow2_u32(min_cap);

	if (fv->slots != NULL)
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
			if (fv->is_backend_cache)
				hnsw_backend_visited.gen = fv->gen;
			return;
		}

		if (fv->is_backend_cache)
		{
			if (desired <= HNSW_BACKEND_VISITED_MAX_CAP)
			{
				uint64	   *new_slots = (uint64 *) MemoryContextAllocZero(TopMemoryContext,
																		(Size) desired * sizeof(uint64));
				uint64	   *old_slots = fv->slots;

				hnsw_backend_visited.slots = new_slots;
				hnsw_backend_visited.cap = desired;
				hnsw_backend_visited.gen = 1;
				fv->slots = new_slots;
				fv->mask = desired - 1;
				fv->grow_at = (desired * 3) / 4;
				fv->gen = 1;
				fv->count = 0;
				fv->on_stack = false;
				pfree(old_slots);
				return;
			}
			fv->is_backend_cache = false;
			hnsw_backend_visited.in_use = false;
		}
		else if (!fv->on_stack)
			pfree(fv->slots);

		fv->slots = NULL;
	}

	if (stack_buf != NULL && stack_cap >= desired)
	{
		memset(stack_buf, 0, (Size) desired * sizeof(uint64));
		fv->slots = stack_buf;
		fv->mask = desired - 1;
		fv->grow_at = (desired * 3) / 4;
		fv->gen = 1;
		fv->count = 0;
		fv->on_stack = true;
		fv->is_backend_cache = false;
		if (fv->alloc_ctx == NULL)
			fv->alloc_ctx = CurrentMemoryContext;
		return;
	}

	if (fv->is_backend_cache)
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
			fv->grow_at = (desired * 3) / 4;
			fv->gen = 1;
			fv->count = 0;
			fv->on_stack = false;
			return;
		}
		fv->is_backend_cache = false;
		hnsw_backend_visited.in_use = false;
	}

	{
		MemoryContext ctx = fv->alloc_ctx != NULL ? fv->alloc_ctx : CurrentMemoryContext;

		fv->alloc_ctx = ctx;
		fv->slots = (uint64 *) MemoryContextAllocZero(ctx, (Size) desired * sizeof(uint64));
		fv->mask = desired - 1;
		fv->grow_at = (desired * 3) / 4;
		fv->gen = 1;
		fv->count = 0;
		fv->on_stack = false;
	}
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

/*
 * Init visited
 */
static inline void
InitVisited(char *base, visited_hash * v, bool inMemory, int ef, int m,
			uint64 *stack_buf, uint32 stack_cap)
{
	if (!inMemory)
		hnsw_fast_visited_init(&v->tids, ef, m, stack_buf, stack_cap);
	else if (base != NULL)
		v->offsets = offsethash_create(CurrentMemoryContext, ef * m * 2, NULL);
	else
		v->pointers = pointerhash_create(CurrentMemoryContext, ef * m * 2, NULL);
}

/*
 * Add to visited
 */
static inline void
AddToVisited(char *base, visited_hash * v, HnswElementPtr elementPtr, bool inMemory, bool *found)
{
	if (!inMemory)
	{
		HnswElement element = HnswPtrAccess(base, elementPtr);

		*found = hnsw_fast_visited_insert(&v->tids, element->blkno, element->offno);
	}
	else if (base != NULL)
	{
		HnswElement element = HnswPtrAccess(base, elementPtr);

		offsethash_insert_hash(v->offsets, HnswPtrOffset(elementPtr), element->hash, found);
	}
	else
	{
		HnswElement element = HnswPtrAccess(base, elementPtr);

		pointerhash_insert_hash(v->pointers, (uintptr_t) HnswPtrPointer(elementPtr), element->hash, found);
	}
}

/*
 * Count element towards ef
 */
static inline bool
CountElement(HnswElement skipElement, HnswElement e)
{
	if (skipElement == NULL)
		return true;

	/* Ensure does not access heaptidsLength during in-memory build */
	pg_memory_barrier();

	/* Keep scan-build happy on Mac x86-64 */
	Assert(e);

	return e->heaptidsLength != 0;
}

/*
 * Load unvisited neighbors from memory
 */
static void
HnswLoadUnvisitedFromMemory(char *base, HnswElement element, HnswUnvisited * unvisited, int *unvisitedLength, visited_hash * v, int lc, HnswNeighborArray * localNeighborhood, Size neighborhoodSize)
{
	/* Get the neighborhood at layer lc */
	HnswNeighborArray *neighborhood = HnswGetNeighbors(base, element, lc);

	/* Copy neighborhood to local memory */
	LWLockAcquire(&element->lock, LW_SHARED);
	memcpy(localNeighborhood, neighborhood, neighborhoodSize);
	LWLockRelease(&element->lock);

	*unvisitedLength = 0;

	for (int i = 0; i < localNeighborhood->length; i++)
	{
		HnswCandidate *hc = &localNeighborhood->items[i];
		bool		found;

		AddToVisited(base, v, hc->element, true, &found);

		if (!found)
			unvisited[(*unvisitedLength)++].element = HnswPtrAccess(base, hc->element);
	}
}

/*
 * Load neighbor index TIDs from explicit tuple metadata fields
 */
static inline bool
HnswLoadNeighborTidsRaw(BlockNumber neighborPage, OffsetNumber neighborOffno,
						uint8 level, uint8 version,
						ItemPointerData *indextids, Relation index,
						HnswSupport * support, int m, int lm, int lc)
{
	Buffer		buf;
	Page		page;
	HnswNeighborTuple ntup;
	Size		start;
	bool		found;
	uint64		opt_version;
	uint32		buf_id;
	void	   *entry_ptr;

	/*
	 * Try optimistic read path if enabled and buffer manager supports it.
	 * This avoids taking buffer locks for read-only access to neighbor data.
	 */
	if (calico_enable_optimistic_reads &&
	    calico_is_active() && !RelationUsesLocalBuffers(index))
	{
		if (support != NULL)
			found = calico_optimistic_lookup_handle_frame(&support->read_handle,
														  neighborPage,
														  &opt_version,
														  &buf_id,
														  &entry_ptr);
		else
			found = calico_optimistic_lookup_frame(RelationGetSmgr(index),
												   MAIN_FORKNUM,
												   neighborPage,
												   &opt_version,
												   &buf_id,
												   &entry_ptr);

		/* Try optimistic pin without locking, get frameId and entry_ptr directly */
		if (found)
		{
			Size		req_size;
			Size		copy_bytes;

			/* Read neighbor tuple data optimistically */
			page = BufferGetPage(buf_id + 1);
			if (unlikely(level < lc || m < 0 || m > HNSW_MAX_M ||
						 lm < 0 || lm > HNSW_MAX_M * 2))
				goto optimistic_neighbor_failed;
			start = (Size) (level - lc) * (Size) m;
			copy_bytes = sizeof(ItemPointerData) * (Size) lm;
			req_size = offsetof(HnswNeighborTupleData, indextids) +
				sizeof(ItemPointerData) * start + copy_bytes;
			ntup = HnswOptimisticItem(page, neighborOffno, req_size);
			if (ntup == NULL)
				goto optimistic_neighbor_failed;

			/*
			 * Check version consistency:
			 * Ensure the neighbor tuple has not been deleted or replaced
			 */
			if (ntup->version != version || ntup->count != (level + 2) * m)
			{
				/* Data validation failed - fall through to locked path */
			}
			else
			{
				/* Copy neighbor TIDs optimistically without PLT memcpy */
				for (int k = 0; k < lm; k++)
					indextids[k] = ntup->indextids[start + k];

				/* Validate the optimistic read using cached entry_ptr - fast path */
				if (calico_optimistic_validate(entry_ptr, opt_version))
				{
					/* Validation successful - data is consistent */
					return true;
				}
			}
		}
optimistic_neighbor_failed:
		calico_optimistic_neighbor_fallbacks++;
	}

	/*
	 * Traditional locked path:
	 * Either optimistic read is disabled, not supported, or validation failed.
	 */
	buf = ReadBuffer(index, neighborPage);
	LockBuffer(buf, BUFFER_LOCK_SHARE);
	page = BufferGetPage(buf);

	ntup = (HnswNeighborTuple) PageGetItem(page, PageGetItemId(page, neighborOffno));

	/*
	 * Ensure the neighbor tuple has not been deleted or replaced between
	 * index scan iterations
	 */
	if (ntup->version != version || ntup->count != (level + 2) * m)
	{
		UnlockReleaseBuffer(buf);
		return false;
	}

	/* Copy to minimize lock time */
	start = mul_size(level - lc, m);
	memcpy(indextids, ntup->indextids + start, mul_size(sizeof(ItemPointerData), lm));

	UnlockReleaseBuffer(buf);
	return true;
}

/*
 * Load neighbor index TIDs
 */
bool
HnswLoadNeighborTids(HnswElement element, ItemPointerData *indextids, Relation index, HnswSupport * support, int m, int lm, int lc)
{
	return HnswLoadNeighborTidsRaw(element->neighborPage, element->neighborOffno,
								   element->level, element->version,
								   indextids, index, support, m, lm, lc);
}

#define HNSW_PREFETCH_AHEAD 3

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

static pg_noinline bool
HnswLoadNeighborTidsAndHashSlow(BlockNumber neighborPage, OffsetNumber neighborOffno,
								uint8 level, uint8 version, Relation index,
								HnswSupport * support, int m, int lm, int lc,
								uint32 mask, uint64 *slots,
								uint64 *tids48, uint32 *buckets, int *n_tids_out)
{
	ItemPointerData indextids[HNSW_MAX_M * 2];
	int			n_tids = 0;

	if (!HnswLoadNeighborTidsRaw(neighborPage, neighborOffno, level, version,
								 indextids, index, support, m, lm, lc))
		return false;

	for (int i = 0; i < lm; i++)
	{
		ItemPointer indextid = &indextids[i];
		uint64		tid48;
		uint32		pos;

		if (!ItemPointerIsValid(indextid))
			break;

		tid48 = ((uint64) BlockIdGetBlockNumber(&indextid->ip_blkid) << 16) |
			(uint64) indextid->ip_posid;
		pos = hnsw_hash_tid48(tid48) & mask;
		tids48[n_tids] = tid48;
		buckets[n_tids] = pos;
		__builtin_prefetch(&slots[pos], 0, 3);
		n_tids++;
	}

	*n_tids_out = n_tids;
	return true;
}

static pg_noinline bool
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

/*
 * Load unvisited neighbors from disk.
 * Populates slots_out (aligned 1-to-1 with unvisited[]) for downstream optimistic access and prefetching.
 * When batch_out_ptr != NULL and missing blocks exist, starts async AIO reads without
 * waiting so caller can evaluate resident neighbors concurrently.
 */
static pg_noinline void
HnswLoadUnvisitedRaw(BlockNumber neighborPage, OffsetNumber neighborOffno,
					 uint8 level, uint8 version,
					 HnswUnvisited * unvisited, int *unvisitedLength,
					 visited_hash * v, Relation index, HnswSupport * support,
					 int m, int lm, int lc,
					 CalicoOptimisticSlot *slots_out, Size tuple_prefetch_bytes,
					 Size min_valid_bytes, HnswElementTuple *etups_out,
					 CalicoAsyncBatch **batch_out_ptr)
{
	bool		use_prefetch = false;
	BlockNumber batch_blocknums[HNSW_MAX_M * 2];
	uint16		linp_offs[HNSW_MAX_M * 2];
	uint64		tids48[HNSW_MAX_M * 2];
	uint32		buckets[HNSW_MAX_M * 2];
	int			n_tids = 0;
	int			ulen = 0;
	HnswFastVisited *fv = &v->tids;
	uint64	   *slots;
	uint32		mask;
	uint64		gen_prefix;
	uint16		gen;

	*unvisitedLength = 0;
	if (unlikely(batch_out_ptr != NULL && *batch_out_ptr != NULL))
	{
		(*batch_out_ptr)->active = false;
		(*batch_out_ptr)->num_ranges = 0;
	}

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
	if (likely(calico_enable_optimistic_reads && support != NULL &&
			   calico_is_active() && !RelationUsesLocalBuffers(index)))
	{
		uint64		opt_version;
		uint32		buf_id;
		void	   *entry_ptr;

		if (likely(calico_optimistic_lookup_handle_frame(&support->read_handle,
														 neighborPage,
														 &opt_version,
														 &buf_id,
														 &entry_ptr)))
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
		return;

have_tids:
	if (calico_enable_optimistic_reads &&
		hnsw_enable_prefetch &&
		calico_is_active() && !RelationUsesLocalBuffers(index))
	{
		use_prefetch = true;
	}

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

			ItemPointerSet(&unvisited[ulen].indextid,
						   blkno,
						   offno);

			if (use_prefetch)
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

	*unvisitedLength = ulen;
	if (likely(use_prefetch && ulen > 0))
	{
		int			count = ulen;

		if (likely(support != NULL))
		{
			bool		all_resident = calico_optimistic_lookup_batch_slots_noprefetch_linp(&support->read_handle,
																						batch_blocknums,
																						count,
																						slots_out,
																						linp_offs);

			if (unlikely(!all_resident))
			{
				if (etups_out != NULL)
					memset(etups_out, 0, sizeof(HnswElementTuple) * count);
				if (HnswLoadUnvisitedMissingOrPinSlow(index, support, batch_blocknums,
													  count, unvisited, slots_out,
													  tuple_prefetch_bytes, batch_out_ptr))
					return;
			}
		}
		else
		{
			if (etups_out != NULL)
				memset(etups_out, 0, sizeof(HnswElementTuple) * count);
			if (HnswLoadUnvisitedMissingOrPinSlow(index, NULL, batch_blocknums,
												  count, unvisited, slots_out,
												  tuple_prefetch_bytes, batch_out_ptr))
				return;
		}

		if (tuple_prefetch_bytes > 0)
		{
			int			prime = Min(count, HNSW_PREFETCH_AHEAD);

			if (likely(etups_out != NULL))
			{
				for (int j = 0; j < prime; j++)
					etups_out[j] = HnswPrefetchSlotTupleEx(&slots_out[j],
														   ItemPointerGetOffsetNumber(&unvisited[j].indextid),
														   tuple_prefetch_bytes,
														   min_valid_bytes);
			}
			else
			{
				for (int j = 0; j < prime; j++)
					HnswPrefetchSlotTuple(&slots_out[j],
										  ItemPointerGetOffsetNumber(&unvisited[j].indextid),
										  tuple_prefetch_bytes);
			}
		}
		else if (unlikely(etups_out != NULL))
		{
			memset(etups_out, 0, sizeof(HnswElementTuple) * count);
		}
	}
	else
	{
		for (int j = 0; j < ulen; j++)
		{
			slots_out[j].resident = false;
			slots_out[j].opt_valid = false;
		}
		if (unlikely(etups_out != NULL && ulen > 0))
			memset(etups_out, 0, sizeof(HnswElementTuple) * ulen);
	}
}

static void
HnswLoadUnvisitedFromDisk(HnswElement element, HnswUnvisited * unvisited, int *unvisitedLength,
						  visited_hash * v, Relation index, HnswSupport * support, int m, int lm, int lc,
						  CalicoOptimisticSlot *slots_out, Size tuple_prefetch_bytes,
						  CalicoAsyncBatch **batch_out_ptr)
{
	HnswLoadUnvisitedRaw(element->neighborPage, element->neighborOffno,
						 element->level, element->version,
						 unvisited, unvisitedLength, v, index, support,
						 m, lm, lc, slots_out, tuple_prefetch_bytes, 0, NULL,
						 batch_out_ptr);
}

static pg_attribute_always_inline void
HnswCommitCandidate(char *base, HnswElement eElement, double eDistance,
					HnswElement skipElement, int ef, int lc,
					int *wlen, pairingheap *C, pairingheap *W,
					pairingheap **discarded, HnswSearchCandidate **f_ptr)
{
	HnswSearchCandidate *f = *f_ptr;
	HnswSearchCandidate *e;
	bool		alwaysAdd = (*wlen < ef);

	if (!(eDistance < f->distance || alwaysAdd))
	{
		if (discarded != NULL)
		{
			e = HnswInitSearchCandidate(base, eElement, eDistance);
			pairingheap_add(*discarded, &e->w_node);
		}
		return;
	}

	/* Make robust to issues */
	if (eElement->level < lc)
		return;

	e = HnswInitSearchCandidate(base, eElement, eDistance);
	pairingheap_add(C, &e->c_node);
	pairingheap_add(W, &e->w_node);

	/*
	 * Do not count elements being deleted towards ef when vacuuming. It would
	 * be ideal to do this for inserts as well, but this could affect insert
	 * performance.
	 */
	if (CountElement(skipElement, eElement))
	{
		(*wlen)++;

		/* No need to decrement wlen */
		if (*wlen > ef)
		{
			HnswSearchCandidate *d = HnswGetSearchCandidate(w_node, pairingheap_remove_first(W));

			if (discarded != NULL)
				pairingheap_add(*discarded, &d->w_node);
		}
	}

	*f_ptr = HnswGetSearchCandidate(w_node, pairingheap_first(W));
}

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

static pg_attribute_always_inline bool
HnswLoadElementOptimisticSlotFast(HnswSupport *support, BlockNumber blkno, OffsetNumber offno,
								  const CalicoOptimisticSlot *slot,
								  HnswElementTuple cached_etup,
								  const Vector *a_vec, Size vec_size,
								  double maxDistance, bool alwaysAdd,
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

		page = CalicoFramePage(slot->frame_id);
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
		if (unlikely(!HnswGetDistanceOptimistic(support, a_vec, &etup->data, vec_size, &dist)))
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

static pg_noinline List *
HnswMaterializeFastList(char *base, HnswQuery * q, Relation index,
					   HnswSupport * support,
					   const HnswFastResultCand *w_heap, int wlen)
{
	List	   *w = NIL;
	bool		use_opt_mat = (calico_enable_optimistic_reads &&
							   support != NULL &&
							   calico_is_active() &&
							   !RelationUsesLocalBuffers(index));

	for (int k = 0; k < wlen; k++)
	{
		const HnswFastResultCand *cand = &w_heap[wlen - 1 - k];
		HnswElement el = HnswInitElementFromBlock(cand->blkno, cand->offno);
		bool		loaded = false;

		if (use_opt_mat &&
			cand->offno >= FirstOffsetNumber &&
			cand->offno <= (BLCKSZ - SizeOfPageHeaderData) / sizeof(ItemIdData))
		{
			uint64		version;
			uint32		frame_id;
			void	   *entry_ptr;

			if (calico_optimistic_lookup_handle_frame(&support->read_handle,
													 cand->blkno, &version,
													 &frame_id, &entry_ptr))
			{
				Page		page = CalicoFramePage(frame_id);
				HnswElementTuple etup = HnswOptimisticItem(page, cand->offno,
														   offsetof(HnswElementTupleData, data));

				if (likely(etup != NULL && HnswIsElementTuple(etup) && !etup->deleted))
				{
					HnswLoadElementFromTupleOptimistic(el, etup, true, false, page);
					if (likely(calico_optimistic_validate(entry_ptr, version)))
						loaded = true;
				}
			}
		}

		if (unlikely(!loaded))
		{
			el->heaptidsLength = 0;
			HnswLoadElementImpl(cand->blkno, cand->offno, NULL, q, index,
								support, false, NULL, &el);
		}

		w = lappend(w, HnswInitSearchCandidate(base, el, cand->distance));
	}

	return w;
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
			bool		ok = (off >= FirstOffsetNumber &&
							  off <= (BLCKSZ - SizeOfPageHeaderData) / sizeof(ItemIdData) &&
							  calico_optimistic_lookup_handle_frame(&support->read_handle,
																	blk, &version,
																	&frame_id, &entry_ptr));

			mat_slots[k].opt_valid = ok;
			mat_slots[k].frame_id = frame_id;
			mat_slots[k].version = version;
			mat_slots[k].entry_ptr = (CalicoTranslationEntry *) entry_ptr;
			if (likely(ok))
			{
				uint32		linp_off = offsetof(PageHeaderData, pd_linp) +
					(off - 1) * sizeof(ItemIdData);

				__builtin_prefetch((const char *) CalicoFramePage(frame_id) + linp_off, 0, 3);
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

static pg_attribute_always_inline void
HnswPrefetchNextNeighborTuple(uint32 frame_id, OffsetNumber neighborOffno,
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

	page = CalicoFramePage(frame_id);
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
	HnswUnvisited *unvisited;
	CalicoOptimisticSlot *unvisited_slots;
	HnswElementTuple *cand_etups;
	int			unvisitedLength;
	uint64		stack_visited_buf[128];
	CalicoAsyncBatch *async_batch = NULL;
	const bool	doPrefetch = hnsw_enable_prefetch;
	const bool	optActive = (calico_enable_optimistic_reads &&
							 support != NULL &&
							 calico_is_active() &&
							 !RelationUsesLocalBuffers(index));
	CalicoReadHandle *const read_handle = optActive ? &support->read_handle : NULL;
	Size		rawPrefetchBytes;
	Size		minValidBytes;
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
	}
	else
	{
		unvisited = palloc_array_checked(HnswUnvisited, lm);
		unvisited_slots = palloc_array_checked(CalicoOptimisticSlot, lm);
		cand_etups = palloc_array_checked(HnswElementTuple, lm);
	}

	if (!have_opt_distance)
	{
		canOptDistance = false;
		queryVec = NULL;
		optVecSize = 0;
		optPrefetchBytes = 0;

		if (optActive &&
			support->vectorDistance != HNSW_VECTOR_DISTANCE_NONE)
		{
			if (DatumGetPointer(q->value) == NULL)
			{
				canOptDistance = true;
				optPrefetchBytes = offsetof(HnswElementTupleData, data);
			}
			else
			{
				queryVec = (const Vector *) DatumGetPointer(q->value);
				if (likely(queryVec->dim > 0))
				{
					optVecSize = VECTOR_SIZE(queryVec->dim);
					if (likely(optVecSize <= BLCKSZ - offsetof(HnswElementTupleData, data)))
					{
						canOptDistance = true;
						optPrefetchBytes = Min(offsetof(HnswElementTupleData, data) + optVecSize,
											   (Size) 1024);
					}
				}
			}
		}
		else if (optActive && doPrefetch)
		{
			optPrefetchBytes = sizeof(HnswElementTupleData) + 64;
		}
	}

	rawPrefetchBytes = doPrefetch ? optPrefetchBytes : 0;
	minValidBytes = offsetof(HnswElementTupleData, data) + optVecSize;

	if (use_local_v)
	{
		memset(&vh, 0, sizeof(vh));
		v = &vh;
		initVisited = true;
	}

	if (initVisited)
		InitVisited(base, v, false, ef, m,
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
			bool		found;

			if (initVisited)
			{
				AddToVisited(base, v, sc->element, false, &found);
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
							 &async_batch);

		if (tuples != NULL)
			(*tuples) += unvisitedLength;

		if (likely(async_batch == NULL || !async_batch->active))
		{
			for (int i = 0; i < unvisitedLength; i++)
			{
				ItemPointer indextid = &unvisited[i].indextid;
				BlockNumber blkno = ItemPointerGetBlockNumber(indextid);
				OffsetNumber offno = ItemPointerGetOffsetNumber(indextid);
				HnswFastCand cand;
				bool		accepted = false;
				bool		alwaysAdd = (wlen < ef);
				double		maxDist = alwaysAdd ? 0.0 : w_heap[0].distance;

				if (doPrefetch && i + HNSW_PREFETCH_AHEAD < unvisitedLength)
				{
					cand_etups[i + HNSW_PREFETCH_AHEAD] =
						HnswPrefetchSlotTupleEx(&unvisited_slots[i + HNSW_PREFETCH_AHEAD],
												ItemPointerGetOffsetNumber(&unvisited[i + HNSW_PREFETCH_AHEAD].indextid),
												optPrefetchBytes, minValidBytes);
				}

				if (p0_stage == 1 && (i - p0_at_i) >= 3)
				{
					uint64		opt_ver;
					void	   *entry_ptr;

					if (likely(calico_optimistic_lookup_handle_frame(read_handle, p0_page,
																	 &opt_ver, &p0_buf_id, &entry_ptr)) &&
						likely(p0_off >= FirstOffsetNumber &&
							   p0_off <= (BLCKSZ - SizeOfPageHeaderData) / sizeof(ItemIdData)))
					{
						__builtin_prefetch(PageGetItemId(CalicoFramePage(p0_buf_id), p0_off), 0, 3);
						p0_stage = 2;
						p0_at_i = i;
					}
					else
						p0_stage = -1;
				}
				else if (p0_stage == 2 && (i - p0_at_i) >= 4)
				{
					HnswPrefetchNextNeighborTuple(p0_buf_id, p0_off, p0_level, m, lm);
					p0_stage = 3;
				}

				if (canOptDistance &&
					HnswLoadElementOptimisticSlotFast(support, blkno, offno, &unvisited_slots[i],
													  cand_etups[i],
													  queryVec, optVecSize,
													  maxDist, alwaysAdd,
													  &cand, &accepted))
				{
					if (accepted)
					{
						HnswFastCommitCand(&cand, ef, lc,
										   &c_heap, &clen, &ccap, &c_on_stack,
										   w_heap, &wlen);
						if (doPipe &&
							(c_heap[0].neighborPage != p0_page ||
							 c_heap[0].neighborOffno != p0_off))
						{
							p0_page = c_heap[0].neighborPage;
							p0_off = c_heap[0].neighborOffno;
							p0_level = c_heap[0].level;
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
									   &c_heap, &clen, &ccap, &c_on_stack,
									   w_heap, &wlen);
					if (doPipe &&
						(c_heap[0].neighborPage != p0_page ||
						 c_heap[0].neighborOffno != p0_off))
					{
						p0_page = c_heap[0].neighborPage;
						p0_off = c_heap[0].neighborOffno;
						p0_level = c_heap[0].level;
						calico_optimistic_prefetch_block(read_handle, p0_page);
						p0_stage = 1;
						p0_at_i = i;
					}
				}
			}

			if (p0_stage == 2)
				HnswPrefetchNextNeighborTuple(p0_buf_id, p0_off, p0_level, m, lm);
			else if (p0_stage == 1 && (unvisitedLength - p0_at_i) >= 2)
			{
				uint64		opt_ver;
				void	   *entry_ptr;

				if (likely(calico_optimistic_lookup_handle_frame(read_handle, p0_page,
																 &opt_ver, &p0_buf_id, &entry_ptr)) &&
					likely(p0_off >= FirstOffsetNumber &&
						   p0_off <= (BLCKSZ - SizeOfPageHeaderData) / sizeof(ItemIdData)))
					__builtin_prefetch(PageGetItemId(CalicoFramePage(p0_buf_id), p0_off), 0, 3);
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

static List *
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

static pg_noinline void
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
		HnswUnvisited *unvisited;
		CalicoOptimisticSlot *unvisited_slots;
		HnswElementTuple *cand_etups;
		int			lm_max = HnswGetLayerM(m, 1);
		CalicoAsyncBatch *async_batch = NULL;
		const bool	doPrefetch = hnsw_enable_prefetch;
		const Size	rawPrefetchBytes = doPrefetch ? optPrefetchBytes : 0;
		const bool	use_static_visited = (canOptDistance && m <= 48);

		if (likely(lm_max <= lengthof(stack_unvisited)))
		{
			unvisited = stack_unvisited;
			unvisited_slots = stack_slots;
			cand_etups = stack_etups;
		}
		else
		{
			unvisited = palloc_array_checked(HnswUnvisited, lm_max);
			unvisited_slots = palloc_array_checked(CalicoOptimisticSlot, lm_max);
			cand_etups = palloc_array_checked(HnswElementTuple, lm_max);
		}

		upper_vh.tids.slots = NULL;
		upper_vh.tids.mask = 0;
		upper_vh.tids.count = 0;
		upper_vh.tids.grow_at = 0;
		upper_vh.tids.gen = 0;
		upper_vh.tids.on_stack = true;
		upper_vh.tids.is_backend_cache = false;
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

						if (doPrefetch && i + HNSW_PREFETCH_AHEAD < unvisitedLength)
						{
							cand_etups[i + HNSW_PREFETCH_AHEAD] =
								HnswPrefetchSlotTupleEx(&unvisited_slots[i + HNSW_PREFETCH_AHEAD],
														ItemPointerGetOffsetNumber(&unvisited[i + HNSW_PREFETCH_AHEAD].indextid),
														optPrefetchBytes, minValidBytes);
						}

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
				if (likely(queryVec->dim > 0))
				{
					optVecSize = VECTOR_SIZE(queryVec->dim);
					if (likely(optVecSize <= BLCKSZ - offsetof(HnswElementTupleData, data)))
					{
						canOptDistance = true;
						optPrefetchBytes = Min(offsetof(HnswElementTupleData, data) + optVecSize,
											   (Size) 1024);
					}
				}
			}
		}
		else if (hnsw_enable_prefetch)
		{
			optPrefetchBytes = sizeof(HnswElementTupleData) + 64;
		}
	}

	HnswSearchLayerUpperDiskFast(q, entryPoint, index, support, m,
								 canOptDistance, queryVec, optVecSize, optPrefetchBytes,
								 &best);

	HnswSearchLayerDiskFastCore(base, q, NIL, &best, ef, 0, index, support, m,
								v, true, tuples,
								true, canOptDistance, queryVec, optVecSize, optPrefetchBytes,
								inline_buf, inline_cap, fast_out, fast_count_out);
	return true;
}

/*
 * Cold out-of-line helper for async batch I/O evaluation in HnswSearchLayer.
 * Keeps Phase 1/2 stack scratch arrays and duplicate loops out of the hot
 * in-memory search loop so instruction cache and inlining stay compact.
 */
static pg_noinline void
HnswEvalAsyncUnvisitedBatch(char *base, HnswQuery * q, int ef, int lc,
							Relation index, HnswSupport * support,
							HnswElement skipElement, pairingheap **discarded,
							bool canOptDistance, const Vector *queryVec,
							Size optVecSize, Size optPrefetchBytes,
							HnswUnvisited * unvisited, int unvisitedLength,
							CalicoOptimisticSlot *unvisited_slots,
							CalicoAsyncBatch *async_batch,
							int *wlen, pairingheap *C, pairingheap *W,
							HnswSearchCandidate **f_ptr)
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
		HnswElement eElement = NULL;
		double		eDistance;
		bool		alwaysAdd = (*wlen < ef);
		double	   *maxDistPtr = alwaysAdd || discarded != NULL ? NULL : &(*f_ptr)->distance;

		if (hnsw_enable_prefetch && r + HNSW_PREFETCH_AHEAD < n_res)
		{
			int			p_idx = res_idxs[r + HNSW_PREFETCH_AHEAD];

			HnswPrefetchSlotTuple(&unvisited_slots[p_idx],
								  ItemPointerGetOffsetNumber(&unvisited[p_idx].indextid),
								  optPrefetchBytes);
		}

		if (canOptDistance &&
			HnswLoadElementOptimisticSlot(support, blkno, offno, &unvisited_slots[idx],
										   &eDistance, queryVec, optVecSize,
										   maxDistPtr, &eElement))
		{
			processed[idx] = true;
			if (eElement != NULL)
				HnswCommitCandidate(base, eElement, eDistance, skipElement,
									ef, lc, wlen, C, W, discarded, f_ptr);
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
			HnswElement eElement = NULL;
			double		eDistance;
			bool		alwaysAdd = (*wlen < ef);
			double	   *maxDistPtr = alwaysAdd || discarded != NULL ? NULL : &(*f_ptr)->distance;

			if (hnsw_enable_prefetch && k + HNSW_PREFETCH_AHEAD < n_rem)
			{
				int			p_idx = rem_idxs[k + HNSW_PREFETCH_AHEAD];

				HnswPrefetchSlotTuple(&unvisited_slots[p_idx],
									  ItemPointerGetOffsetNumber(&unvisited[p_idx].indextid),
									  optPrefetchBytes);
			}

			if (!canOptDistance ||
				!HnswLoadElementOptimisticSlot(support, blkno, offno, &unvisited_slots[idx],
											   &eDistance, queryVec, optVecSize,
											   maxDistPtr, &eElement))
			{
				HnswLoadElementImpl(blkno, offno, &eDistance, q, index, support,
									false, maxDistPtr, &eElement);
			}

			if (eElement != NULL)
				HnswCommitCandidate(base, eElement, eDistance, skipElement,
									ef, lc, wlen, C, W, discarded, f_ptr);
		}
	}
}

/*
 * Algorithm 2 from paper
 */
List *
HnswSearchLayer(char *base, HnswQuery * q, List *ep, int ef, int lc, Relation index, HnswSupport * support, int m, bool inserting, HnswElement skipElement, visited_hash * v, pairingheap **discarded, bool initVisited, int64 *tuples)
{
	List	   *w = NIL;
	pairingheap *C;
	pairingheap *W;
	int			wlen = 0;
	visited_hash vh;
	ListCell   *lc2;
	HnswNeighborArray *localNeighborhood = NULL;
	Size		neighborhoodSize = 0;
	int			lm;
	HnswUnvisited stack_unvisited[64];
	CalicoOptimisticSlot stack_slots[64];
	HnswUnvisited *unvisited;
	CalicoOptimisticSlot *unvisited_slots;
	int			unvisitedLength;
	bool		inMemory = index == NULL;
	bool		canOptDistance = false;
	const Vector *queryVec = NULL;
	Size		optVecSize = 0;
	Size		optPrefetchBytes = 0;
	uint64		stack_visited_buf[128];
	bool		use_local_v = (v == NULL);
	CalicoAsyncBatch *async_batch = NULL;

	if (likely(!inMemory && !inserting && skipElement == NULL &&
			   discarded == NULL && ef > 0))
		return HnswSearchLayerDiskFast(base, q, ep, ef, lc, index, support, m,
									   v, initVisited, tuples,
									   NULL, 0, NULL, NULL);

	C = pairingheap_allocate(CompareNearestCandidates, NULL);
	W = pairingheap_allocate(CompareFurthestCandidates, NULL);
	lm = HnswGetLayerM(m, lc);

	if (likely(lm <= lengthof(stack_unvisited)))
	{
		unvisited = stack_unvisited;
		unvisited_slots = stack_slots;
	}
	else
	{
		unvisited = palloc_array_checked(HnswUnvisited, lm);
		unvisited_slots = palloc_array_checked(CalicoOptimisticSlot, lm);
	}

	if (!inMemory && !inserting && calico_enable_optimistic_reads &&
		support != NULL &&
		support->vectorDistance != HNSW_VECTOR_DISTANCE_NONE &&
		calico_is_active() && !RelationUsesLocalBuffers(index))
	{
		if (DatumGetPointer(q->value) == NULL)
		{
			canOptDistance = true;
			optPrefetchBytes = offsetof(HnswElementTupleData, data);
		}
		else
		{
			queryVec = (const Vector *) DatumGetPointer(q->value);
			if (likely(queryVec->dim > 0))
			{
				optVecSize = VECTOR_SIZE(queryVec->dim);
				if (likely(optVecSize <= BLCKSZ - offsetof(HnswElementTupleData, data)))
				{
					canOptDistance = true;
					optPrefetchBytes = Min(offsetof(HnswElementTupleData, data) + optVecSize,
										   (Size) 1024);
				}
			}
		}
	}
	else if (!inMemory && calico_enable_optimistic_reads && hnsw_enable_prefetch &&
			 calico_is_active() && !RelationUsesLocalBuffers(index))
	{
		optPrefetchBytes = sizeof(HnswElementTupleData) + 64;
	}

	if (use_local_v)
	{
		memset(&vh, 0, sizeof(vh));
		v = &vh;
		initVisited = true;
	}

	if (initVisited)
	{
		InitVisited(base, v, inMemory, ef, m,
					(use_local_v && !inMemory) ? stack_visited_buf : NULL,
					(use_local_v && !inMemory) ? lengthof(stack_visited_buf) : 0);

		if (discarded != NULL)
			*discarded = pairingheap_allocate(CompareNearestDiscardedCandidates, NULL);
	}

	/* Create local memory for neighborhood if needed */
	if (inMemory)
	{
		neighborhoodSize = HNSW_NEIGHBOR_ARRAY_SIZE(lm);
		localNeighborhood = palloc(neighborhoodSize);
	}

	/* Add entry points to v, C, and W */
	foreach(lc2, ep)
	{
		HnswSearchCandidate *sc = (HnswSearchCandidate *) lfirst(lc2);
		bool		found;

		if (initVisited)
		{
			AddToVisited(base, v, sc->element, inMemory, &found);

			/* OK to count elements instead of tuples */
			if (tuples != NULL)
				(*tuples)++;
		}

		pairingheap_add(C, &sc->c_node);
		pairingheap_add(W, &sc->w_node);

		/*
		 * Do not count elements being deleted towards ef when vacuuming. It
		 * would be ideal to do this for inserts as well, but this could
		 * affect insert performance.
		 */
		if (CountElement(skipElement, HnswPtrAccess(base, sc->element)))
			wlen++;
	}

	while (!pairingheap_is_empty(C))
	{
		HnswSearchCandidate *c = HnswGetSearchCandidate(c_node, pairingheap_remove_first(C));
		HnswSearchCandidate *f = HnswGetSearchCandidate(w_node, pairingheap_first(W));
		HnswElement cElement;

		if (c->distance > f->distance)
			break;

		cElement = HnswPtrAccess(base, c->element);
		if (unlikely(async_batch != NULL))
		{
			async_batch->active = false;
			async_batch->num_ranges = 0;
		}

		if (inMemory)
			HnswLoadUnvisitedFromMemory(base, cElement, unvisited, &unvisitedLength, v, lc, localNeighborhood, neighborhoodSize);
		else
			HnswLoadUnvisitedFromDisk(cElement, unvisited, &unvisitedLength, v, index, support, m, lm, lc,
									  unvisited_slots, hnsw_enable_prefetch ? optPrefetchBytes : 0,
									  !inserting ? &async_batch : NULL);

		/* OK to count elements instead of tuples */
		if (tuples != NULL)
			(*tuples) += unvisitedLength;

		f = HnswGetSearchCandidate(w_node, pairingheap_first(W));

		if (likely(inMemory || async_batch == NULL || !async_batch->active))
		{
			for (int i = 0; i < unvisitedLength; i++)
			{
				HnswElement eElement;
				double		eDistance;
				bool		alwaysAdd = wlen < ef;

				/* Prefetch full tuple + vector several steps ahead while evaluating earlier items */
				if (!inMemory && hnsw_enable_prefetch && i + HNSW_PREFETCH_AHEAD < unvisitedLength)
				{
					HnswPrefetchSlotTuple(&unvisited_slots[i + HNSW_PREFETCH_AHEAD],
										  ItemPointerGetOffsetNumber(&unvisited[i + HNSW_PREFETCH_AHEAD].indextid),
										  optPrefetchBytes);
				}

				if (inMemory)
				{
					eElement = unvisited[i].element;
					eDistance = GetElementDistance(base, eElement, q, support);
				}
				else
				{
					ItemPointer indextid = &unvisited[i].indextid;
					BlockNumber blkno = ItemPointerGetBlockNumber(indextid);
					OffsetNumber offno = ItemPointerGetOffsetNumber(indextid);
					double	   *maxDistPtr = alwaysAdd || discarded != NULL ? NULL : &f->distance;

					/* Avoid any allocations if not adding */
					eElement = NULL;
					if (!canOptDistance ||
						!HnswLoadElementOptimisticSlot(support, blkno, offno, &unvisited_slots[i],
													   &eDistance, queryVec, optVecSize,
													   maxDistPtr, &eElement))
					{
						HnswLoadElementImpl(blkno, offno, &eDistance, q, index, support,
											inserting, maxDistPtr, &eElement);
					}

					if (eElement == NULL)
						continue;
				}

				HnswCommitCandidate(base, eElement, eDistance, skipElement,
									ef, lc, &wlen, C, W, discarded, &f);
			}
		}
		else
		{
			HnswEvalAsyncUnvisitedBatch(base, q, ef, lc, index, support,
										skipElement, discarded, canOptDistance,
										queryVec, optVecSize, optPrefetchBytes,
										unvisited, unvisitedLength,
										unvisited_slots, async_batch,
										&wlen, C, W, &f);
		}
	}

	if (unlikely(async_batch != NULL))
		pfree(async_batch);

	if (use_local_v && !inMemory && vh.tids.slots != NULL && !vh.tids.on_stack)
	{
		pfree(vh.tids.slots);
		vh.tids.slots = NULL;
	}

	/* Add each element of W to w */
	while (!pairingheap_is_empty(W))
	{
		HnswSearchCandidate *sc = HnswGetSearchCandidate(w_node, pairingheap_remove_first(W));

		w = lappend(w, sc);
	}

	return w;
}

/*
 * Compare candidate distances with pointer tie-breaker
 */
static int
CompareCandidateDistances(const ListCell *a, const ListCell *b)
{
	HnswCandidate *hca = lfirst(a);
	HnswCandidate *hcb = lfirst(b);

	if (hca->distance < hcb->distance)
		return 1;

	if (hca->distance > hcb->distance)
		return -1;

	if (HnswPtrPointer(hca->element) < HnswPtrPointer(hcb->element))
		return 1;

	if (HnswPtrPointer(hca->element) > HnswPtrPointer(hcb->element))
		return -1;

	return 0;
}

/*
 * Compare candidate distances with offset tie-breaker
 */
static int
CompareCandidateDistancesOffset(const ListCell *a, const ListCell *b)
{
	HnswCandidate *hca = lfirst(a);
	HnswCandidate *hcb = lfirst(b);

	if (hca->distance < hcb->distance)
		return 1;

	if (hca->distance > hcb->distance)
		return -1;

	if (HnswPtrOffset(hca->element) < HnswPtrOffset(hcb->element))
		return 1;

	if (HnswPtrOffset(hca->element) > HnswPtrOffset(hcb->element))
		return -1;

	return 0;
}

/*
 * Check if an element is closer to q than any element from R
 */
static bool
CheckElementCloser(char *base, HnswCandidate * e, List *r, HnswSupport * support)
{
	HnswElement eElement = HnswPtrAccess(base, e->element);
	Datum		eValue = HnswGetValue(base, eElement);
	ListCell   *lc2;

	foreach(lc2, r)
	{
		HnswCandidate *ri = lfirst(lc2);
		HnswElement riElement = HnswPtrAccess(base, ri->element);
		Datum		riValue = HnswGetValue(base, riElement);
		float		distance = HnswGetDistance(eValue, riValue, support);

		if (distance <= e->distance)
			return false;
	}

	return true;
}

/*
 * Algorithm 4 from paper
 */
static List *
SelectNeighbors(char *base, List *c, int lm, HnswSupport * support, bool *closerSet, HnswCandidate * newCandidate, HnswCandidate * *pruned, bool sortCandidates)
{
	List	   *r = NIL;
	List	   *w = list_copy(c);
	HnswCandidate **wd;
	int			wdlen = 0;
	int			wdoff = 0;
	bool		mustCalculate = !(*closerSet);
	List	   *added = NIL;
	bool		removedAny = false;

	if (list_length(w) <= lm)
		return w;

	wd = palloc_array_checked(HnswCandidate *, list_length(w));

	/* Ensure order of candidates is deterministic for closer caching */
	if (sortCandidates)
	{
		if (base == NULL)
			list_sort(w, CompareCandidateDistances);
		else
			list_sort(w, CompareCandidateDistancesOffset);
	}

	while (list_length(w) > 0 && list_length(r) < lm)
	{
		/* Assumes w is already ordered desc */
		HnswCandidate *e = llast(w);

		w = list_delete_last(w);

		/* Use previous state of r and wd to skip work when possible */
		if (mustCalculate)
			e->closer = CheckElementCloser(base, e, r, support);
		else if (list_length(added) > 0)
		{
			/* Keep Valgrind happy for in-memory, parallel builds */
			if (base != NULL)
				VALGRIND_MAKE_MEM_DEFINED(&e->closer, 1);

			/*
			 * If the current candidate was closer, we only need to compare it
			 * with the other candidates that we have added.
			 */
			if (e->closer)
			{
				e->closer = CheckElementCloser(base, e, added, support);

				if (!e->closer)
					removedAny = true;
			}
			else
			{
				/*
				 * If we have removed any candidates from closer, a candidate
				 * that was not closer earlier might now be.
				 */
				if (removedAny)
				{
					e->closer = CheckElementCloser(base, e, r, support);
					if (e->closer)
						added = lappend(added, e);
				}
			}
		}
		else if (e == newCandidate)
		{
			e->closer = CheckElementCloser(base, e, r, support);
			if (e->closer)
				added = lappend(added, e);
		}

		/* Keep Valgrind happy for in-memory, parallel builds */
		if (base != NULL)
			VALGRIND_MAKE_MEM_DEFINED(&e->closer, 1);

		if (e->closer)
			r = lappend(r, e);
		else
			wd[wdlen++] = e;
	}

	/* Cached value can only be used in future if sorted deterministically */
	*closerSet = sortCandidates;

	/* Keep pruned connections */
	while (wdoff < wdlen && list_length(r) < lm)
		r = lappend(r, wd[wdoff++]);

	/* Return pruned for update connections */
	if (pruned != NULL)
	{
		if (wdoff < wdlen)
			*pruned = wd[wdoff];
		else
			*pruned = linitial(w);
	}

	return r;
}

/*
 * Add connections
 */
static void
AddConnections(char *base, HnswElement element, List *neighbors, int lc)
{
	ListCell   *lc2;
	HnswNeighborArray *a = HnswGetNeighbors(base, element, lc);

	foreach(lc2, neighbors)
		a->items[a->length++] = *((HnswCandidate *) lfirst(lc2));
}

/*
 * Update connections
 */
void
HnswUpdateConnection(char *base, HnswNeighborArray * neighbors, HnswElement newElement, float distance, int lm, int *updateIdx, Relation index, HnswSupport * support)
{
	HnswCandidate newHc;

	HnswPtrStore(base, newHc.element, newElement);
	newHc.distance = distance;

	if (neighbors->length < lm)
	{
		neighbors->items[neighbors->length++] = newHc;

		/* Track update */
		if (updateIdx != NULL)
			*updateIdx = -2;
	}
	else
	{
		/* Shrink connections */
		List	   *c = NIL;
		HnswCandidate *pruned = NULL;

		/* Add candidates */
		for (int i = 0; i < neighbors->length; i++)
			c = lappend(c, &neighbors->items[i]);
		c = lappend(c, &newHc);

		SelectNeighbors(base, c, lm, support, &neighbors->closerSet, &newHc, &pruned, true);

		/* Should not happen */
		if (pruned == NULL)
			return;

		/* Find and replace the pruned element */
		for (int i = 0; i < neighbors->length; i++)
		{
			if (HnswPtrEqual(base, neighbors->items[i].element, pruned->element))
			{
				neighbors->items[i] = newHc;

				/* Track update */
				if (updateIdx != NULL)
					*updateIdx = i;

				break;
			}
		}
	}
}

/*
 * Remove elements being deleted or skipped
 */
static List *
RemoveElements(char *base, List *w, HnswElement skipElement)
{
	ListCell   *lc2;
	List	   *w2 = NIL;

	/* Ensure does not access heaptidsLength during in-memory build */
	pg_memory_barrier();

	foreach(lc2, w)
	{
		HnswCandidate *hc = (HnswCandidate *) lfirst(lc2);
		HnswElement hce = HnswPtrAccess(base, hc->element);

		/* Skip self for vacuuming update */
		if (skipElement != NULL && hce->blkno == skipElement->blkno && hce->offno == skipElement->offno)
			continue;

		if (hce->heaptidsLength != 0)
			w2 = lappend(w2, hc);
	}

	return w2;
}

/*
 * Precompute hash
 */
static void
PrecomputeHash(char *base, HnswElement element)
{
	HnswElementPtr ptr;

	HnswPtrStore(base, ptr, element);

	if (base == NULL)
		element->hash = hash_pointer((uintptr_t) HnswPtrPointer(ptr));
	else
		element->hash = hash_offset(HnswPtrOffset(ptr));
}

/*
 * Algorithm 1 from paper
 */
void
HnswFindElementNeighbors(char *base, HnswElement element, HnswElement entryPoint, Relation index, HnswSupport * support, int m, int efConstruction, bool existing)
{
	List	   *ep;
	List	   *w;
	int			level = element->level;
	int			entryLevel;
	HnswQuery	q;
	HnswElement skipElement = existing ? element : NULL;
	bool		inMemory = index == NULL;

	q.value = HnswGetValue(base, element);

	/* Precompute hash */
	if (inMemory)
		PrecomputeHash(base, element);

	/* No neighbors if no entry point */
	if (entryPoint == NULL)
		return;

	/* Get entry point and level */
	ep = list_make1(HnswEntryCandidate(base, entryPoint, &q, index, support, true));
	entryLevel = entryPoint->level;

	/* 1st phase: greedy search to insert level */
	for (int lc = entryLevel; lc >= level + 1; lc--)
	{
		w = HnswSearchLayer(base, &q, ep, 1, lc, index, support, m, true, skipElement, NULL, NULL, true, NULL);
		ep = w;
	}

	if (level > entryLevel)
		level = entryLevel;

	/* Add one for existing element */
	if (existing)
		efConstruction++;

	/* 2nd phase */
	for (int lc = level; lc >= 0; lc--)
	{
		int			lm = HnswGetLayerM(m, lc);
		List	   *neighbors;
		List	   *lw = NIL;
		ListCell   *lc2;

		w = HnswSearchLayer(base, &q, ep, efConstruction, lc, index, support, m, true, skipElement, NULL, NULL, true, NULL);

		/* Convert search candidates to candidates */
		foreach(lc2, w)
		{
			HnswSearchCandidate *sc = lfirst(lc2);
			HnswCandidate *hc = palloc_object(HnswCandidate);

			hc->element = sc->element;
			hc->distance = sc->distance;

			lw = lappend(lw, hc);
		}

		/* Elements being deleted or skipped can help with search */
		/* but should be removed before selecting neighbors */
		if (!inMemory)
			lw = RemoveElements(base, lw, skipElement);

		/*
		 * Candidates are sorted, but not deterministically. Could set
		 * sortCandidates to true for in-memory builds to enable closer
		 * caching, but there does not seem to be a difference in performance.
		 */
		neighbors = SelectNeighbors(base, lw, lm, support, &HnswGetNeighbors(base, element, lc)->closerSet, NULL, NULL, false);

		AddConnections(base, element, neighbors, lc);

		ep = w;
	}
}

/*
 * Check dimensions match index
 */
void
HnswCheckDim(int expected, const HnswTypeInfo * typeInfo, Oid collation, Datum value)
{
	int32		dim = DatumGetInt32(DirectFunctionCall1Coll(typeInfo->dimensions, collation, value));

	if (dim != expected)
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("expected %d dimensions, not %d", expected, dim)));
}

PGDLLEXPORT Datum vector_dims(PG_FUNCTION_ARGS);
PGDLLEXPORT Datum halfvec_vector_dims(PG_FUNCTION_ARGS);
PGDLLEXPORT Datum sparsevec_vector_dims(PG_FUNCTION_ARGS);

PGDLLEXPORT Datum l2_normalize(PG_FUNCTION_ARGS);
PGDLLEXPORT Datum halfvec_l2_normalize(PG_FUNCTION_ARGS);
PGDLLEXPORT Datum sparsevec_l2_normalize(PG_FUNCTION_ARGS);

static void
SparsevecCheckValue(Pointer v)
{
	SparseVector *vec = (SparseVector *) v;

	if (vec->nnz > HNSW_MAX_NNZ)
		ereport(ERROR,
				(errcode(ERRCODE_PROGRAM_LIMIT_EXCEEDED),
				 errmsg("sparsevec cannot have more than %d non-zero elements for hnsw index", HNSW_MAX_NNZ)));
}

/*
 * Get type info
 */
const		HnswTypeInfo *
HnswGetTypeInfo(Relation index)
{
	FmgrInfo   *procinfo = HnswOptionalProcInfo(index, HNSW_TYPE_INFO_PROC);

	if (procinfo == NULL)
	{
		static const HnswTypeInfo typeInfo = {
			.maxDimensions = HNSW_MAX_DIM,
			.dimensions = vector_dims,
			.normalize = l2_normalize,
			.checkValue = NULL
		};

		return (&typeInfo);
	}
	else
		return (const HnswTypeInfo *) DatumGetPointer(FunctionCall0Coll(procinfo, InvalidOid));
}

FUNCTION_PREFIX PG_FUNCTION_INFO_V1(hnsw_halfvec_support);
Datum
hnsw_halfvec_support(PG_FUNCTION_ARGS)
{
	static const HnswTypeInfo typeInfo = {
		.maxDimensions = HNSW_MAX_DIM * 2,
		.dimensions = halfvec_vector_dims,
		.normalize = halfvec_l2_normalize,
		.checkValue = NULL
	};

	PG_RETURN_POINTER(&typeInfo);
}

FUNCTION_PREFIX PG_FUNCTION_INFO_V1(hnsw_bit_support);
Datum
hnsw_bit_support(PG_FUNCTION_ARGS)
{
	static const HnswTypeInfo typeInfo = {
		.maxDimensions = HNSW_MAX_DIM * 32,
		.dimensions = bitlength,
		.normalize = NULL,
		.checkValue = NULL
	};

	PG_RETURN_POINTER(&typeInfo);
}

FUNCTION_PREFIX PG_FUNCTION_INFO_V1(hnsw_sparsevec_support);
Datum
hnsw_sparsevec_support(PG_FUNCTION_ARGS)
{
	static const HnswTypeInfo typeInfo = {
		.maxDimensions = SPARSEVEC_MAX_DIM,
		.dimensions = sparsevec_vector_dims,
		.normalize = sparsevec_l2_normalize,
		.checkValue = SparsevecCheckValue
	};

	PG_RETURN_POINTER(&typeInfo);
}
