#ifndef HNSW_H
#define HNSW_H

#include "postgres.h"
#include "calico_internal.h"
#include "storage/lwlock.h"
#include "storage/spin.h"
#include "storage/condition_variable.h"
#include "storage/bufpage.h"
#include "varatt.h"

#include <math.h>

#include "access/genam.h"
#include "access/parallel.h"
#include "lib/pairingheap.h"
#include "nodes/execnodes.h"
#include "port.h"				/* for random() */
#include "storage/bufpage.h"
#include "storage/condition_variable.h"
#include "storage/lwlock.h"
#include "storage/s_lock.h"
#include "utils/relptr.h"
#include "utils/sampling.h"
#include "vector.h"

#if PG_VERSION_NUM < 190000
#include "storage/shmem.h"		/* for add_size()/mul_size() in some versions */
#endif

#ifdef HNSW_BENCH
#include "portability/instr_time.h"
#endif

#if PG_VERSION_NUM >= 190000
typedef Pointer Item;
#endif

#define HNSW_MAX_DIM 2000
#define HNSW_MAX_NNZ 1000

/* Support functions */
#define HNSW_DISTANCE_PROC 1
#define HNSW_NORM_PROC 2
#define HNSW_TYPE_INFO_PROC 3

#define HNSW_VERSION	1
#define HNSW_MAGIC_NUMBER 0xA953A953
#define HNSW_PAGE_ID	0xFF90

/* Preserved page numbers */
#define HNSW_METAPAGE_BLKNO	0
#define HNSW_HEAD_BLKNO		1	/* first element page */

/* Must correspond to page numbers since page lock is used */
#define HNSW_UPDATE_LOCK 	0
#define HNSW_SCAN_LOCK		1

/* HNSW parameters */
#define HNSW_DEFAULT_M	16
#define HNSW_MIN_M	2
#define HNSW_MAX_M		100
#define HNSW_DEFAULT_EF_CONSTRUCTION	64
#define HNSW_MIN_EF_CONSTRUCTION	4
#define HNSW_MAX_EF_CONSTRUCTION		1000
#define HNSW_DEFAULT_EF_SEARCH	40
#define HNSW_MIN_EF_SEARCH		1
#define HNSW_MAX_EF_SEARCH		1000

/* Tuple types */
#define HNSW_ELEMENT_TUPLE_TYPE  1
#define HNSW_NEIGHBOR_TUPLE_TYPE 2

/* Make graph robust against non-HOT updates */
#define HNSW_HEAPTIDS 10

#define HNSW_UPDATE_ENTRY_GREATER 1
#define HNSW_UPDATE_ENTRY_ALWAYS 2

/* Build phases */
/* PROGRESS_CREATEIDX_SUBPHASE_INITIALIZE is 1 */
#define PROGRESS_HNSW_PHASE_LOAD		2

#define HNSW_MAX_SIZE (BLCKSZ - MAXALIGN(SizeOfPageHeaderData) - MAXALIGN(sizeof(HnswPageOpaqueData)) - sizeof(ItemIdData))
#define HNSW_TUPLE_ALLOC_SIZE BLCKSZ

#define HNSW_ELEMENT_TUPLE_SIZE(size)	MAXALIGN(add_size(offsetof(HnswElementTupleData, data), size))
#define HNSW_NEIGHBOR_TUPLE_SIZE(level, m)	MAXALIGN(add_size(offsetof(HnswNeighborTupleData, indextids), mul_size(sizeof(ItemPointerData), mul_size(add_size(level, 2), m))))

#define HNSW_NEIGHBOR_ARRAY_SIZE(lm)	add_size(offsetof(HnswNeighborArray, items), mul_size(sizeof(HnswCandidate), lm))

#define HnswPageGetOpaque(page)	((HnswPageOpaque) PageGetSpecialPointer(page))
#define HnswPageGetMeta(page)	((HnswMetaPageData *) PageGetContents(page))

#ifdef HNSW_BENCH
#define HnswBench(name, code) \
	do { \
		instr_time	start; \
		instr_time	duration; \
		INSTR_TIME_SET_CURRENT(start); \
		(code); \
		INSTR_TIME_SET_CURRENT(duration); \
		INSTR_TIME_SUBTRACT(duration, start); \
		elog(INFO, "%s: %.3f ms", name, INSTR_TIME_GET_MILLISEC(duration)); \
	} while (0)
#else
#define HnswBench(name, code) (code)
#endif

#if PG_VERSION_NUM >= 150000
#define RandomDouble() pg_prng_double(&pg_global_prng_state)
#define SeedRandom(seed) pg_prng_seed(&pg_global_prng_state, seed)
#else
#define RandomDouble() (((double) random()) / MAX_RANDOM_VALUE)
#define SeedRandom(seed) srandom(seed)
#endif

#if PG_VERSION_NUM < 140006
#define palloc_object(type) ((type *) palloc(sizeof(type)))
#define palloc0_object(type) ((type *) palloc0(sizeof(type)))
#endif

#if PG_VERSION_NUM >= 190000
#define palloc_array_checked(type, count) ((type *) palloc_array(type, count))
#else
#define palloc_array_checked(type, count) ((type *) palloc(mul_size(sizeof(type), count)))
#endif

#define HnswIsElementTuple(tup) ((tup)->type == HNSW_ELEMENT_TUPLE_TYPE)
#define HnswIsNeighborTuple(tup) ((tup)->type == HNSW_NEIGHBOR_TUPLE_TYPE)

/* 2 * M connections for ground layer */
#define HnswGetLayerM(m, layer) (layer == 0 ? (m) * 2 : (m))

/* Optimal ML from paper */
#define HnswGetMl(m) (1 / log(m))

/* Ensure fits on page and in uint8 */
#define HnswGetMaxLevel(m) Min(((BLCKSZ - MAXALIGN(SizeOfPageHeaderData) - MAXALIGN(sizeof(HnswPageOpaqueData)) - offsetof(HnswNeighborTupleData, indextids) - sizeof(ItemIdData)) / (sizeof(ItemPointerData)) / (m)) - 2, 63)

#define HnswGetSearchCandidate(membername, ptr) pairingheap_container(HnswSearchCandidate, membername, ptr)
#define HnswGetSearchCandidateConst(membername, ptr) pairingheap_const_container(HnswSearchCandidate, membername, ptr)

#define HnswGetValue(base, element) PointerGetDatum(HnswPtrAccess(base, (element)->value))

#if PG_VERSION_NUM < 140005
#define relptr_offset(rp) ((rp).relptr_off - 1)
#endif

/* Pointer macros */
#define HnswPtrAccess(base, hp) ((base) == NULL ? (hp).ptr : relptr_access(base, (hp).relptr))
#define HnswPtrStore(base, hp, value) ((base) == NULL ? (void) ((hp).ptr = (value)) : (void) relptr_store(base, (hp).relptr, (__typeof__((hp).relptr.relptr_type)) (value)))
#define HnswPtrIsNull(base, hp) ((base) == NULL ? (hp).ptr == NULL : relptr_is_null((hp).relptr))
#define HnswPtrEqual(base, hp1, hp2) ((base) == NULL ? (hp1).ptr == (hp2).ptr : relptr_offset((hp1).relptr) == relptr_offset((hp2).relptr))

/* For code paths dedicated to each type */
#define HnswPtrPointer(hp) (hp).ptr
#define HnswPtrOffset(hp) relptr_offset((hp).relptr)

/* Variables */
extern int	hnsw_ef_search;
extern int	hnsw_iterative_scan;
extern int	hnsw_max_scan_tuples;
extern double hnsw_scan_mem_multiplier;
extern bool	hnsw_enable_prefetch;
extern int	hnsw_lock_tranche_id;

typedef enum HnswIterativeScanMode
{
	HNSW_ITERATIVE_SCAN_OFF,
	HNSW_ITERATIVE_SCAN_RELAXED,
	HNSW_ITERATIVE_SCAN_STRICT
}			HnswIterativeScanMode;

typedef struct HnswElementData HnswElementData;
typedef struct HnswNeighborArray HnswNeighborArray;

#define HnswPtrDeclare(type, relptrtype, ptrtype) \
	relptr_declare(type, relptrtype); \
	typedef union { type *ptr; relptrtype relptr; } ptrtype

/* Pointers that can be absolute or relative */
/* Use char for DatumPtr so works with Pointer */
HnswPtrDeclare(HnswElementData, HnswElementRelptr, HnswElementPtr);
HnswPtrDeclare(HnswNeighborArray, HnswNeighborArrayRelptr, HnswNeighborArrayPtr);
HnswPtrDeclare(HnswNeighborArrayPtr, HnswNeighborsRelptr, HnswNeighborsPtr);
HnswPtrDeclare(char, DatumRelptr, DatumPtr);

struct HnswElementData
{
	HnswElementPtr next;
	ItemPointerData heaptids[HNSW_HEAPTIDS];
	uint8		heaptidsLength;
	uint8		level;
	uint8		deleted;
	uint8		version;
	uint32		hash;
	HnswNeighborsPtr neighbors;
	BlockNumber blkno;
	OffsetNumber offno;
	OffsetNumber neighborOffno;
	BlockNumber neighborPage;
	DatumPtr	value;
	LWLock		lock;
};

typedef HnswElementData * HnswElement;

typedef struct HnswCandidate
{
	HnswElementPtr element;
	float		distance;
	bool		closer;
}			HnswCandidate;

struct HnswNeighborArray
{
	int			length;
	bool		closerSet;
	HnswCandidate items[FLEXIBLE_ARRAY_MEMBER];
};

typedef struct HnswSearchCandidate
{
	pairingheap_node c_node;
	pairingheap_node w_node;
	HnswElementPtr element;
	double		distance;
}			HnswSearchCandidate;

/* HNSW index options */
typedef struct HnswOptions
{
	int32		vl_len_;		/* varlena header (do not touch directly!) */
	int			m;				/* number of connections */
	int			efConstruction; /* size of dynamic candidate list */
}			HnswOptions;

typedef struct HnswGraph
{
	/* Graph state */
	slock_t		lock;
	HnswElementPtr head;
	double		indtuples;

	/* Entry state */
	LWLock		entryLock;
	LWLock		entryWaitLock;
	HnswElementPtr entryPoint;

	/* Allocations state */
	LWLock		allocatorLock;
	Size		memoryUsed;
	Size		memoryTotal;

	/* Flushed state */
	LWLock		flushLock;
	bool		flushed;
}			HnswGraph;

typedef struct HnswShared
{
	/* Immutable state */
	Oid			heaprelid;
	Oid			indexrelid;
	bool		isconcurrent;

	/* Worker progress */
	ConditionVariable workersdonecv;

	/* Mutex for mutable state */
	slock_t		mutex;

	/* Mutable state */
	int			nparticipantsdone;
	double		reltuples;
	HnswGraph	graphData;
}			HnswShared;

#define ParallelTableScanFromHnswShared(shared) \
	(ParallelTableScanDesc) ((char *) (shared) + BUFFERALIGN(sizeof(HnswShared)))

typedef struct HnswLeader
{
	ParallelContext *pcxt;
	int			nparticipanttuplesorts;
	HnswShared *hnswshared;
	Snapshot	snapshot;
	char	   *hnswarea;
}			HnswLeader;

typedef struct HnswAllocator
{
	void	   *(*alloc) (Size size, void *state);
	void	   *state;
}			HnswAllocator;

typedef struct HnswTypeInfo
{
	int			maxDimensions;
	Datum		(*dimensions) (PG_FUNCTION_ARGS);
	Datum		(*normalize) (PG_FUNCTION_ARGS);
	void		(*checkValue) (Pointer v);
}			HnswTypeInfo;

typedef enum HnswVectorDistance
{
	HNSW_VECTOR_DISTANCE_NONE,
	HNSW_VECTOR_DISTANCE_L2,
	HNSW_VECTOR_DISTANCE_INNER_PRODUCT
} HnswVectorDistance;

typedef struct HnswSupport
{
	FmgrInfo   *procinfo;
	FmgrInfo   *normprocinfo;
	Oid			collation;
	CalicoReadHandle read_handle;
	HnswVectorDistance vectorDistance;
}			HnswSupport;

typedef struct HnswQuery
{
	Datum		value;
}			HnswQuery;

typedef struct HnswBuildState
{
	/* Info */
	Relation	heap;
	Relation	index;
	IndexInfo  *indexInfo;
	ForkNumber	forkNum;
	const		HnswTypeInfo *typeInfo;

	/* Settings */
	int			dimensions;
	int			m;
	int			efConstruction;

	/* Statistics */
	double		indtuples;
	double		reltuples;

	/* Support functions */
	HnswSupport support;

	/* Variables */
	HnswGraph	graphData;
	HnswGraph  *graph;
	double		ml;
	int			maxLevel;

	/* Memory */
	MemoryContext graphCtx;
	MemoryContext tmpCtx;
	HnswAllocator allocator;

	/* Parallel builds */
	HnswLeader *hnswleader;
	HnswShared *hnswshared;
	char	   *hnswarea;
}			HnswBuildState;

typedef struct HnswMetaPageData
{
	uint32		magicNumber;
	uint32		version;
	uint32		dimensions;
	uint16		m;
	uint16		efConstruction;
	BlockNumber entryBlkno;
	OffsetNumber entryOffno;
	int16		entryLevel;
	BlockNumber insertPage;
}			HnswMetaPageData;

typedef HnswMetaPageData * HnswMetaPage;

typedef struct HnswPageOpaqueData
{
	BlockNumber nextblkno;
	uint16		unused;
	uint16		page_id;		/* for identification of HNSW indexes */
}			HnswPageOpaqueData;

typedef HnswPageOpaqueData * HnswPageOpaque;

typedef struct HnswElementTupleData
{
	uint8		type;
	uint8		level;
	uint8		deleted;
	uint8		version;
	ItemPointerData heaptids[HNSW_HEAPTIDS];
	ItemPointerData neighbortid;
	uint16		unused;
	Vector		data;
}			HnswElementTupleData;

typedef HnswElementTupleData * HnswElementTuple;

typedef struct HnswNeighborTupleData
{
	uint8		type;
	uint8		version;
	uint16		count;
	ItemPointerData indextids[FLEXIBLE_ARRAY_MEMBER];
}			HnswNeighborTupleData;

typedef HnswNeighborTupleData * HnswNeighborTuple;

typedef struct HnswFastVisited
{
	uint64	   *slots;
	uint32		mask;			/* capacity - 1 (power of 2), 0 if uninitialized */
	uint32		count;			/* entries inserted in current generation */
	uint32		grow_at;		/* resize threshold */
	MemoryContext alloc_ctx;	/* context owning heap slots */
	uint16		gen;			/* active 16-bit generation tag (1..65535) */
	bool		on_stack;		/* true if slots points to stack buffer */
	bool		is_backend_cache;	/* true if slots points to backend cache */
}			HnswFastVisited;

static inline uint32
hnsw_hash_tid48(uint64 tid48)
{
	uint64		h = tid48 ^ (tid48 >> 25);

	h *= UINT64CONST(0xbf58476d1ce4e5b9);
	return (uint32) (h ^ (h >> 31));
}

typedef union
{
	struct pointerhash_hash *pointers;
	struct offsethash_hash *offsets;
	HnswFastVisited tids;
}			visited_hash;

typedef union
{
	HnswElement element;
	ItemPointerData indextid;
}			HnswUnvisited;

#define HNSW_HEAP_PREFETCH_MAX 12
#define HNSW_ITER_CHUNK_BITS 10
#define HNSW_ITER_CHUNK_SIZE (1U << HNSW_ITER_CHUNK_BITS)
#define HNSW_ITER_CHUNK_MASK (HNSW_ITER_CHUNK_SIZE - 1U)
#define HNSW_ITER_HEAP_PREFETCH_MAX 64
#define HNSW_ITER_DRAIN_BATCH 64
#define HNSW_ITER_OOM_WAVE_MAX 4
#define HNSW_ITER_OOM_UNVISITED_CAP 256

typedef struct HnswFastResultCand
{
	double		distance;
	BlockNumber blkno;
	OffsetNumber offno;
	uint16		_pad;
}			HnswFastResultCand;

typedef struct HnswIterativeKey
{
	float		distance;
	uint32		id;
}			HnswIterativeKey;

typedef struct HnswIterativeNode
{
	BlockNumber blkno;
	BlockNumber neighborPage;
	uint32		htid_meta;		/* (heap_count << 24) | extra_off */
	OffsetNumber offno;
	OffsetNumber neighborOffno;
	ItemPointerData first_htid;
	uint8		level;
	uint8		version;
}			HnswIterativeNode;

typedef struct HnswIterativeState
{
	HnswIterativeKey *pending;
	HnswIterativeKey *candidates;
	HnswIterativeKey *results;
	int			pending_count;
	int			pending_capacity;
	int			candidate_capacity;
	int			result_count;
	int			result_capacity;
	int			result_idx;
	int			pref_start;
	int			pref_end;
	uint8		cur_rem_htids;
	bool		oom_mode;
	uint8		oom_hit_streak;
	HnswIterativeNode **chunks;
	uint32		node_count;
	uint32		chunk_count;
	uint32		chunk_capacity;
	ItemPointerData *extra_htids;
	uint32		extra_count;
	uint32		extra_capacity;
	OffsetNumber pref_offs[HNSW_ITER_HEAP_PREFETCH_MAX];
	CalicoOptimisticSlot pref_slots[HNSW_ITER_HEAP_PREFETCH_MAX];
}			HnswIterativeState;

static inline const HnswIterativeNode *
HnswIterativeGetNode(const HnswIterativeState *state, uint32 id)
{
	return &state->chunks[id >> HNSW_ITER_CHUNK_BITS][id & HNSW_ITER_CHUNK_MASK];
}

static inline bool
HnswIterativeEmpty(const HnswIterativeState *state)
{
	return state->pending_count == 0;
}

typedef struct HnswScanOpaqueData
{
	const		HnswTypeInfo *typeInfo;
	bool		first;
	List	   *w;
	visited_hash v;
	bool		visited_cb_registered;
	MemoryContextCallback visited_cb;
	pairingheap *discarded;
	HnswIterativeState *iterative;
	HnswQuery	q;
	int			m;
	int64		tuples;
	double		previousDistance;
	Size		maxMemory;
	MemoryContext tmpCtx;

	/* Support functions */
	HnswSupport support;
	CalicoReadHandle heap_handle;
	bool		heap_handle_inited;
	BlockNumber heap_nblocks;
	uint8		heap_pref_count;
	uint8		heap_pref_idx;
	uint8		heap_io_window;
	OffsetNumber heap_offs[HNSW_HEAP_PREFETCH_MAX];
	CalicoOptimisticSlot heap_slots[HNSW_HEAP_PREFETCH_MAX];
	HnswFastResultCand *fast_cands;
	int			fast_count;
	int			fast_idx;
	int			fast_loaded_start;
	int			fast_loaded_end;
	HnswFastResultCand fast_inline[64];
	ItemPointerData fast_win_tids[HNSW_HEAP_PREFETCH_MAX][HNSW_HEAPTIDS];
	uint8		fast_win_len[HNSW_HEAP_PREFETCH_MAX];
}			HnswScanOpaqueData;

typedef HnswScanOpaqueData * HnswScanOpaque;

typedef struct HnswVacuumState
{
	/* Info */
	Relation	index;
	IndexBulkDeleteResult *stats;
	IndexBulkDeleteCallback callback;
	void	   *callback_state;

	/* Settings */
	int			m;
	int			efConstruction;

	/* Support functions */
	HnswSupport support;

	/* Variables */
	struct tidhash_hash *deleting;
	BufferAccessStrategy bas;
	HnswNeighborTuple ntup;
	HnswElementData highestPoint;
	HnswElementData fallbackPoint;

	/* Memory */
	MemoryContext tmpCtx;
}			HnswVacuumState;

/* Methods */
int			HnswGetM(Relation index);
int			HnswGetEfConstruction(Relation index);
FmgrInfo   *HnswOptionalProcInfo(Relation index, uint16 procnum);
void		HnswInitSupport(HnswSupport * support, Relation index);
void		HnswSearchIterativeFromEntry(HnswQuery * q, HnswElement entryPoint, int ef, Relation index, HnswSupport * support, int m, visited_hash * v, int64 *tuples, HnswIterativeState **state_ptr);
void		HnswSearchIterativeBatch(HnswQuery * q, int ef, Relation index, HnswSupport * support, int m, visited_hash * v, int64 *tuples, HnswIterativeState *state, bool expand);
Datum		HnswNormValue(const HnswTypeInfo * typeInfo, Oid collation, Datum value);
bool		HnswCheckNorm(HnswSupport * support, Datum value);
Buffer		HnswNewBuffer(Relation index, ForkNumber forkNum);
void		HnswInitPage(Buffer buf, Page page);
void		HnswInit(void);
void		HnswAcquireScanVisited(HnswFastVisited *fv, MemoryContext scanCtx, MemoryContext tmpCtx, MemoryContextCallback *cb, bool *cbRegistered);
void		HnswReleaseScanVisited(HnswFastVisited *fv);
bool		HnswSearchLayerFastCands(char *base, HnswQuery * q, List *ep, int ef, int lc, Relation index, HnswSupport * support, int m, visited_hash * v, int64 *tuples, HnswFastResultCand *inline_buf, int inline_cap, HnswFastResultCand **fast_out, int *fast_count_out);
bool		HnswSearchLayerFromEntryFastCands(char *base, HnswQuery * q, HnswElement entryPoint, int ef, Relation index, HnswSupport * support, int m, visited_hash * v, int64 *tuples, HnswFastResultCand *inline_buf, int inline_cap, HnswFastResultCand **fast_out, int *fast_count_out);
void		HnswLoadFastWindow(Relation index, HnswSupport * support, const HnswFastResultCand *cands, int start_idx, int count, bool allow_deleted, ItemPointerData win_tids[][HNSW_HEAPTIDS], uint8 *win_len);
List	   *HnswSearchLayer(char *base, HnswQuery * q, List *ep, int ef, int lc, Relation index, HnswSupport * support, int m, bool inserting, HnswElement skipElement, visited_hash * v, pairingheap **discarded, bool initVisited, int64 *tuples);
HnswElement HnswGetEntryPoint(Relation index);
void		HnswGetMetaPageInfo(Relation index, int *m, int *dimensions, HnswElement * entryPoint);
void		HnswGetMetaPageInfoWithHandle(Relation index, CalicoReadHandle *handle, int *m, int *dimensions, HnswElement * entryPoint);
void	   *HnswAlloc(HnswAllocator * allocator, Size size);
HnswElement HnswInitElement(char *base, ItemPointer tid, int m, double ml, int maxLevel, HnswAllocator * alloc);
HnswElement HnswInitElementFromBlock(BlockNumber blkno, OffsetNumber offno);
void		HnswFindElementNeighbors(char *base, HnswElement element, HnswElement entryPoint, Relation index, HnswSupport * support, int m, int efConstruction, bool existing);
HnswSearchCandidate *HnswEntryCandidate(char *base, HnswElement entryPoint, HnswQuery * q, Relation index, HnswSupport * support, bool loadVec);
void		HnswUpdateMetaPage(Relation index, int updateEntry, HnswElement entryPoint, BlockNumber insertPage, ForkNumber forkNum, bool building);
void		HnswSetNeighborTuple(char *base, HnswNeighborTuple ntup, HnswElement e, int m);
void		HnswAddHeapTid(HnswElement element, ItemPointer heaptid);
HnswNeighborArray *HnswInitNeighborArray(int lm, HnswAllocator * allocator);
void		HnswInitNeighbors(char *base, HnswElement element, int m, HnswAllocator * alloc);
bool		HnswInsertTupleOnDisk(Relation index, const HnswTypeInfo * typeInfo, HnswSupport * support, Datum value, ItemPointer heaptid, bool building);
void		HnswUpdateNeighborsOnDisk(Relation index, HnswSupport * support, HnswElement e, int m, bool building);
void		HnswLoadElementFromTuple(HnswElement element, HnswElementTuple etup, bool loadHeaptids, bool loadVec);
void		HnswLoadElement(HnswElement element, double *distance, HnswQuery * q, Relation index, HnswSupport * support, bool loadVec, double *maxDistance);
bool		HnswFormIndexValue(Datum *out, Datum *values, bool *isnull, const HnswTypeInfo * typeInfo, HnswSupport * support);
void		HnswSetElementTuple(char *base, HnswElementTuple etup, HnswElement element);
void		HnswUpdateConnection(char *base, HnswNeighborArray * neighbors, HnswElement newElement, float distance, int lm, int *updateIdx, Relation index, HnswSupport * support);
bool		HnswLoadNeighborTids(HnswElement element, ItemPointerData *indextids, Relation index, HnswSupport * support, int m, int lm, int lc);
void		HnswInitLockTranche(void);
void		HnswCheckDim(int expected, const HnswTypeInfo * typeInfo, Oid collation, Datum value);
const		HnswTypeInfo *HnswGetTypeInfo(Relation index);
PGDLLEXPORT void HnswParallelBuildMain(dsm_segment *seg, shm_toc *toc);

/* Index access methods */
IndexBuildResult *hnswbuild(Relation heap, Relation index, IndexInfo *indexInfo);
void		hnswbuildempty(Relation index);
bool		hnswinsert(Relation index, Datum *values, bool *isnull, ItemPointer heap_tid, Relation heap, IndexUniqueCheck checkUnique
#if PG_VERSION_NUM >= 140000
					   ,bool indexUnchanged
#endif
					   ,IndexInfo *indexInfo
);
IndexBulkDeleteResult *hnswbulkdelete(IndexVacuumInfo *info, IndexBulkDeleteResult *stats, IndexBulkDeleteCallback callback, void *callback_state);
IndexBulkDeleteResult *hnswvacuumcleanup(IndexVacuumInfo *info, IndexBulkDeleteResult *stats);
IndexScanDesc hnswbeginscan(Relation index, int nkeys, int norderbys);
void		hnswrescan(IndexScanDesc scan, ScanKey keys, int nkeys, ScanKey orderbys, int norderbys);
bool		hnswgettuple(IndexScanDesc scan, ScanDirection dir);
void		hnswendscan(IndexScanDesc scan);

static inline HnswNeighborArray *
HnswGetNeighbors(char *base, HnswElement element, int lc)
{
	HnswNeighborArrayPtr *neighborList = HnswPtrAccess(base, element->neighbors);

	Assert(element->level >= lc);

	return HnswPtrAccess(base, neighborList[lc]);
}

/* Hash tables */
typedef struct TidHashEntry
{
	ItemPointerData tid;
	char		status;
}			TidHashEntry;

#define SH_PREFIX tidhash
#define SH_ELEMENT_TYPE TidHashEntry
#define SH_KEY_TYPE ItemPointerData
#define SH_SCOPE extern
#define SH_DECLARE
#include "lib/simplehash.h"

typedef struct PointerHashEntry
{
	uintptr_t	ptr;
	char		status;
}			PointerHashEntry;

#define SH_PREFIX pointerhash
#define SH_ELEMENT_TYPE PointerHashEntry
#define SH_KEY_TYPE uintptr_t
#define SH_SCOPE extern
#define SH_DECLARE
#include "lib/simplehash.h"

typedef struct OffsetHashEntry
{
	Size		offset;
	char		status;
}			OffsetHashEntry;

#define SH_PREFIX offsethash
#define SH_ELEMENT_TYPE OffsetHashEntry
#define SH_KEY_TYPE Size
#define SH_SCOPE extern
#define SH_DECLARE
#include "lib/simplehash.h"

#endif
