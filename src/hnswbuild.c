/*
 * The HNSW build happens in two phases:
 *
 * 1. In-memory phase
 *
 * In this first phase, the graph is held completely in memory. When the graph
 * is fully built, or we run out of memory reserved for the build (determined
 * by maintenance_work_mem), we materialize the graph to disk (see
 * FlushPages()), and switch to the on-disk phase.
 *
 * In a parallel build, a large contiguous chunk of shared memory is allocated
 * to hold the graph. Each worker process has its own HnswBuildState struct in
 * private memory, which contains information that doesn't change throughout
 * the build, and pointers to the shared structs in shared memory. The shared
 * memory area is mapped to a different address in each worker process, and
 * 'HnswBuildState.hnswarea' points to the beginning of the shared area in the
 * worker process's address space. All pointers used in the graph are
 * "relative pointers", stored as an offset from 'hnswarea'.
 *
 * Each element is protected by an LWLock. It must be held when reading or
 * modifying the element's neighbors or 'heaptids'.
 *
 * In a non-parallel build, the graph is held in backend-private memory. All
 * the elements are allocated in a dedicated memory context, 'graphCtx', and
 * the pointers used in the graph are regular pointers.
 *
 * 2. On-disk phase
 *
 * In the on-disk phase, the index is built by inserting each vector to the
 * index one by one, just like on INSERT. The only difference is that we don't
 * WAL-log the individual inserts. If the graph fit completely in memory and
 * was fully built in the in-memory phase, the on-disk phase is skipped.
 *
 * After we have finished building the graph, we perform one more scan through
 * the index and write all the pages to the WAL.
 *
 * Graph import (hnsw.build_graph_file)
 *
 * Instead of building the graph by insertion, the index can take it from a
 * file written by tools/pipnn_export (a PiPNN graph from Ginger), streaming
 * (BuildGraphFromFile): no in-memory graph, about 42 bytes of memory per row.
 * Every element goes on layer 0, and layer-0 tuples have a fixed size, so:
 *
 * 1. The heap scan writes each row's element tuple, and a placeholder for its
 *    neighbor tuple, in scan order, with the same page layout as
 *    CreateGraphPages(), and remembers the row's heap TID and where its two
 *    tuples went.
 * 2. Graph nodes are matched to rows by heap TID (both sorted).
 * 3. The graph file's neighbor lists are streamed in, and each placeholder is
 *    overwritten with the neighbors' element TIDs, as WriteNeighborTuples()
 *    does.
 *
 * The build is serial, and fails if the table and the file disagree.
 */
#include "postgres.h"

#include <limits.h>

#include "access/genam.h"
#include "access/parallel.h"
#include "access/relscan.h"
#include "access/table.h"
#include "access/tableam.h"
#include "access/tupdesc.h"
#include "access/xact.h"
#include "access/xloginsert.h"
#include "catalog/index.h"
#include "catalog/pg_type_d.h"
#include "commands/progress.h"
#include "hnsw.h"
#include "miscadmin.h"
#include "nodes/execnodes.h"
#include "optimizer/optimizer.h"
#include "storage/bufmgr.h"
#include "storage/condition_variable.h"
#include "tcop/tcopprot.h"
#include "utils/datum.h"
#include "utils/memutils.h"
#include "utils/rel.h"
#include "utils/snapmgr.h"

#if PG_VERSION_NUM >= 160000
#include "varatt.h"
#endif

#if PG_VERSION_NUM >= 140000
#include "utils/backend_progress.h"
#else
#include "pgstat.h"
#endif

#if PG_VERSION_NUM >= 140000
#include "utils/backend_status.h"
#include "utils/wait_event.h"
#endif

#define PARALLEL_KEY_HNSW_SHARED		UINT64CONST(0xA000000000000001)
#define PARALLEL_KEY_HNSW_AREA			UINT64CONST(0xA000000000000002)
#define PARALLEL_KEY_QUERY_TEXT			UINT64CONST(0xA000000000000003)

#define HNSW_MAX_GRAPH_MEMORY (SIZE_MAX / 2)

/*
 * Create the metapage
 */
static void
CreateMetaPage(HnswBuildState * buildstate)
{
	Relation	index = buildstate->index;
	ForkNumber	forkNum = buildstate->forkNum;
	Buffer		buf;
	Page		page;
	HnswMetaPage metap;

	buf = HnswNewBuffer(index, forkNum);
	page = BufferGetPage(buf);
	HnswInitPage(buf, page);

	/* Set metapage data */
	metap = HnswPageGetMeta(page);
	metap->magicNumber = HNSW_MAGIC_NUMBER;
	metap->version = HNSW_VERSION;
	metap->dimensions = buildstate->dimensions;
	metap->m = buildstate->m;
	metap->efConstruction = buildstate->efConstruction;
	metap->entryBlkno = InvalidBlockNumber;
	metap->entryOffno = InvalidOffsetNumber;
	metap->entryLevel = -1;
	metap->insertPage = InvalidBlockNumber;
	((PageHeader) page)->pd_lower =
		((char *) metap + sizeof(HnswMetaPageData)) - (char *) page;

	MarkBufferDirty(buf);
	UnlockReleaseBuffer(buf);
}

/*
 * Add a new page
 */
static void
HnswBuildAppendPage(Relation index, Buffer *buf, Page *page, ForkNumber forkNum)
{
	/* Add a new page */
	Buffer		newbuf = HnswNewBuffer(index, forkNum);

	/* Update previous page */
	HnswPageGetOpaque(*page)->nextblkno = BufferGetBlockNumber(newbuf);

	/* Commit */
	MarkBufferDirty(*buf);
	UnlockReleaseBuffer(*buf);

	/* Can take a while, so ensure we can interrupt */
	/* Needs to be called when no buffer locks are held */
	LockBuffer(newbuf, BUFFER_LOCK_UNLOCK);
	CHECK_FOR_INTERRUPTS();
	LockBuffer(newbuf, BUFFER_LOCK_EXCLUSIVE);

	/* Prepare new page */
	*buf = newbuf;
	*page = BufferGetPage(*buf);
	HnswInitPage(*buf, *page);
}

/*
 * Create graph pages
 */
static void
CreateGraphPages(HnswBuildState * buildstate)
{
	Relation	index = buildstate->index;
	ForkNumber	forkNum = buildstate->forkNum;
	Size		maxSize;
	HnswElementTuple etup;
	HnswNeighborTuple ntup;
	BlockNumber insertPage;
	HnswElement entryPoint;
	Buffer		buf;
	Page		page;
	HnswElementPtr iter = buildstate->graph->head;
	char	   *base = buildstate->hnswarea;

	/* Calculate sizes */
	maxSize = HNSW_MAX_SIZE;

	/* Allocate once */
	etup = palloc0(HNSW_TUPLE_ALLOC_SIZE);
	ntup = palloc0(HNSW_TUPLE_ALLOC_SIZE);

	/* Prepare first page */
	buf = HnswNewBuffer(index, forkNum);
	page = BufferGetPage(buf);
	HnswInitPage(buf, page);

	while (!HnswPtrIsNull(base, iter))
	{
		HnswElement element = HnswPtrAccess(base, iter);
		Size		etupSize;
		Size		ntupSize;
		Size		combinedSize;
		Pointer		valuePtr = HnswPtrAccess(base, element->value);

		/* Update iterator */
		iter = element->next;

		/* Zero memory for each element */
		MemSet(etup, 0, HNSW_TUPLE_ALLOC_SIZE);

		/* Calculate sizes */
		etupSize = HNSW_ELEMENT_TUPLE_SIZE(VARSIZE_ANY(valuePtr));
		ntupSize = HNSW_NEIGHBOR_TUPLE_SIZE(element->level, buildstate->m);
		combinedSize = etupSize + ntupSize + sizeof(ItemIdData);

		/* Initial size check */
		if (etupSize > HNSW_TUPLE_ALLOC_SIZE)
			ereport(ERROR,
					(errcode(ERRCODE_PROGRAM_LIMIT_EXCEEDED),
					 errmsg("index tuple too large")));

		HnswSetElementTuple(base, etup, element);

		/* Keep element and neighbors on the same page if possible */
		if (PageGetFreeSpace(page) < etupSize || (combinedSize <= maxSize && PageGetFreeSpace(page) < combinedSize))
			HnswBuildAppendPage(index, &buf, &page, forkNum);

		/* Calculate offsets */
		element->blkno = BufferGetBlockNumber(buf);
		element->offno = OffsetNumberNext(PageGetMaxOffsetNumber(page));
		if (combinedSize <= maxSize)
		{
			element->neighborPage = element->blkno;
			element->neighborOffno = OffsetNumberNext(element->offno);
		}
		else
		{
			element->neighborPage = element->blkno + 1;
			element->neighborOffno = FirstOffsetNumber;
		}

		ItemPointerSet(&etup->neighbortid, element->neighborPage, element->neighborOffno);

		/* Add element */
		if (PageAddItem(page, (Item) etup, etupSize, InvalidOffsetNumber, false, false) != element->offno)
			elog(ERROR, "failed to add index item to \"%s\"", RelationGetRelationName(index));

		/* Add new page if needed */
		if (PageGetFreeSpace(page) < ntupSize)
			HnswBuildAppendPage(index, &buf, &page, forkNum);

		/* Add placeholder for neighbors */
		if (PageAddItem(page, (Item) ntup, ntupSize, InvalidOffsetNumber, false, false) != element->neighborOffno)
			elog(ERROR, "failed to add index item to \"%s\"", RelationGetRelationName(index));
	}

	insertPage = BufferGetBlockNumber(buf);

	/* Commit */
	MarkBufferDirty(buf);
	UnlockReleaseBuffer(buf);

	entryPoint = HnswPtrAccess(base, buildstate->graph->entryPoint);
	HnswUpdateMetaPage(index, HNSW_UPDATE_ENTRY_ALWAYS, entryPoint, insertPage, forkNum, true);

	pfree(etup);
	pfree(ntup);
}

/*
 * Write neighbor tuples
 */
static void
WriteNeighborTuples(HnswBuildState * buildstate)
{
	Relation	index = buildstate->index;
	ForkNumber	forkNum = buildstate->forkNum;
	int			m = buildstate->m;
	HnswElementPtr iter = buildstate->graph->head;
	char	   *base = buildstate->hnswarea;
	HnswNeighborTuple ntup;

	/* Allocate once */
	ntup = palloc0(HNSW_TUPLE_ALLOC_SIZE);

	while (!HnswPtrIsNull(base, iter))
	{
		HnswElement element = HnswPtrAccess(base, iter);
		Buffer		buf;
		Page		page;
		Size		ntupSize = HNSW_NEIGHBOR_TUPLE_SIZE(element->level, m);

		/* Update iterator */
		iter = element->next;

		/* Zero memory for each element */
		MemSet(ntup, 0, HNSW_TUPLE_ALLOC_SIZE);

		/* Can take a while, so ensure we can interrupt */
		/* Needs to be called when no buffer locks are held */
		CHECK_FOR_INTERRUPTS();

		buf = ReadBufferExtended(index, forkNum, element->neighborPage, RBM_NORMAL, NULL);
		LockBuffer(buf, BUFFER_LOCK_EXCLUSIVE);
		page = BufferGetPage(buf);

		HnswSetNeighborTuple(base, ntup, element, m);

		if (!PageIndexTupleOverwrite(page, element->neighborOffno, (Item) ntup, ntupSize))
			elog(ERROR, "failed to add index item to \"%s\"", RelationGetRelationName(index));

		/* Commit */
		MarkBufferDirty(buf);
		UnlockReleaseBuffer(buf);
	}

	pfree(ntup);
}

/*
 * Flush pages
 */
static void
FlushPages(HnswBuildState * buildstate)
{
#ifdef HNSW_MEMORY
	elog(INFO, "memory: %zu MB", buildstate->graph->memoryUsed / (1024 * 1024));
#endif

	CreateMetaPage(buildstate);
	CreateGraphPages(buildstate);
	WriteNeighborTuples(buildstate);

	buildstate->graph->flushed = true;
	MemoryContextReset(buildstate->graphCtx);
}

/*
 * Add a heap TID to an existing element
 */
static bool
AddDuplicateInMemory(HnswElement element, HnswElement dup)
{
	LWLockAcquire(&dup->lock, LW_EXCLUSIVE);

	if (dup->heaptidsLength == HNSW_HEAPTIDS)
	{
		LWLockRelease(&dup->lock);
		return false;
	}

	HnswAddHeapTid(dup, &element->heaptids[0]);

	LWLockRelease(&dup->lock);

	return true;
}

/*
 * Find duplicate element
 */
static bool
FindDuplicateInMemory(char *base, HnswElement element)
{
	HnswNeighborArray *neighbors = HnswGetNeighbors(base, element, 0);
	Datum		value = HnswGetValue(base, element);

	for (int i = 0; i < neighbors->length; i++)
	{
		HnswCandidate *neighbor = &neighbors->items[i];
		HnswElement neighborElement = HnswPtrAccess(base, neighbor->element);
		Datum		neighborValue = HnswGetValue(base, neighborElement);

		/* Exit early since ordered by distance */
		if (!datumIsEqual(value, neighborValue, false, -1))
			return false;

		/* Check for space */
		if (AddDuplicateInMemory(element, neighborElement))
			return true;
	}

	return false;
}

/*
 * Add to element list
 */
static void
AddElementInMemory(char *base, HnswGraph * graph, HnswElement element)
{
	SpinLockAcquire(&graph->lock);
	element->next = graph->head;
	HnswPtrStore(base, graph->head, element);
	SpinLockRelease(&graph->lock);
}

/*
 * Update neighbors
 */
static void
UpdateNeighborsInMemory(char *base, HnswSupport * support, HnswElement e, int m)
{
	for (int lc = e->level; lc >= 0; lc--)
	{
		int			lm = HnswGetLayerM(m, lc);
		Size		neighborsSize = HNSW_NEIGHBOR_ARRAY_SIZE(lm);
		HnswNeighborArray *neighbors = palloc(neighborsSize);

		/* Copy neighbors to local memory */
		LWLockAcquire(&e->lock, LW_SHARED);
		memcpy(neighbors, HnswGetNeighbors(base, e, lc), neighborsSize);
		LWLockRelease(&e->lock);

		for (int i = 0; i < neighbors->length; i++)
		{
			HnswCandidate *hc = &neighbors->items[i];
			HnswElement neighborElement = HnswPtrAccess(base, hc->element);

			/* Keep scan-build happy on Mac x86-64 */
			Assert(neighborElement);

			LWLockAcquire(&neighborElement->lock, LW_EXCLUSIVE);
			HnswUpdateConnection(base, HnswGetNeighbors(base, neighborElement, lc), e, hc->distance, lm, NULL, NULL, support);
			LWLockRelease(&neighborElement->lock);
		}
	}
}

/*
 * Update graph in memory
 */
static void
UpdateGraphInMemory(HnswSupport * support, HnswElement element, int m, HnswElement entryPoint, HnswBuildState * buildstate)
{
	HnswGraph  *graph = buildstate->graph;
	char	   *base = buildstate->hnswarea;

	/* Look for duplicate */
	if (FindDuplicateInMemory(base, element))
		return;

	/* Add element */
	AddElementInMemory(base, graph, element);

	/* Update neighbors */
	UpdateNeighborsInMemory(base, support, element, m);

	/* Update entry point if needed (already have lock) */
	if (entryPoint == NULL || element->level > entryPoint->level)
		HnswPtrStore(base, graph->entryPoint, element);
}

/*
 * Insert tuple in memory
 */
static void
InsertTupleInMemory(HnswBuildState * buildstate, HnswElement element)
{
	HnswGraph  *graph = buildstate->graph;
	HnswSupport *support = &buildstate->support;
	HnswElement entryPoint;
	LWLock	   *entryLock = &graph->entryLock;
	LWLock	   *entryWaitLock = &graph->entryWaitLock;
	int			efConstruction = buildstate->efConstruction;
	int			m = buildstate->m;
	char	   *base = buildstate->hnswarea;

	/* Wait if another process needs exclusive lock on entry lock */
	LWLockAcquire(entryWaitLock, LW_EXCLUSIVE);
	LWLockRelease(entryWaitLock);

	/* Get entry point */
	LWLockAcquire(entryLock, LW_SHARED);
	entryPoint = HnswPtrAccess(base, graph->entryPoint);

	/* Prevent concurrent inserts when likely updating entry point */
	if (entryPoint == NULL || element->level > entryPoint->level)
	{
		/* Release shared lock */
		LWLockRelease(entryLock);

		/* Tell other processes to wait and get exclusive lock */
		LWLockAcquire(entryWaitLock, LW_EXCLUSIVE);
		LWLockAcquire(entryLock, LW_EXCLUSIVE);
		LWLockRelease(entryWaitLock);

		/* Get latest entry point after lock is acquired */
		entryPoint = HnswPtrAccess(base, graph->entryPoint);
	}

	/* Find neighbors for element */
	HnswFindElementNeighbors(base, element, entryPoint, NULL, support, m, efConstruction, false);

	/* Update graph in memory */
	UpdateGraphInMemory(support, element, m, entryPoint, buildstate);

	/* Release entry lock */
	LWLockRelease(entryLock);
}

/*
 * Insert tuple
 */
static bool
InsertTuple(Relation index, Datum *values, bool *isnull, ItemPointer heaptid, HnswBuildState * buildstate)
{
	HnswGraph  *graph = buildstate->graph;
	HnswElement element;
	HnswAllocator *allocator = &buildstate->allocator;
	HnswSupport *support = &buildstate->support;
	Size		valueSize;
	Pointer		valuePtr;
	LWLock	   *flushLock = &graph->flushLock;
	char	   *base = buildstate->hnswarea;
	Datum		value;
	Size		memoryMargin;

	/* Form index value */
	if (!HnswFormIndexValue(&value, values, isnull, buildstate->typeInfo, support))
		return false;

	/* Check dimensions match index */
	HnswCheckDim(buildstate->dimensions, buildstate->typeInfo, support->collation, value);

	/* Get datum size */
	valueSize = VARSIZE_ANY(DatumGetPointer(value));

	/* In a parallel build, add a margin so allocations never fail */
	memoryMargin = base == NULL ? 0 : 1024 * 1024;

	/* Ensure graph not flushed when inserting */
	LWLockAcquire(flushLock, LW_SHARED);

	/* Are we in the on-disk phase? */
	if (graph->flushed)
	{
		LWLockRelease(flushLock);

		return HnswInsertTupleOnDisk(index, buildstate->typeInfo, support, value, heaptid, true);
	}

	/*
	 * In a parallel build, the HnswElement is allocated from the shared
	 * memory area, so we need to coordinate with other processes.
	 */
	LWLockAcquire(&graph->allocatorLock, LW_EXCLUSIVE);

	/*
	 * Check that we have enough memory available for the new element now that
	 * we have the allocator lock, and flush pages if needed.
	 */
	if (add_size(graph->memoryUsed, memoryMargin) >= graph->memoryTotal)
	{
		LWLockRelease(&graph->allocatorLock);

		LWLockRelease(flushLock);
		LWLockAcquire(flushLock, LW_EXCLUSIVE);

		if (!graph->flushed)
		{
			ereport(NOTICE,
					(errmsg("hnsw graph no longer fits into maintenance_work_mem after " INT64_FORMAT " tuples", (int64) graph->indtuples),
					 errdetail("Building will take significantly more time."),
					 errhint("Increase maintenance_work_mem to speed up builds.")));

			FlushPages(buildstate);
		}

		LWLockRelease(flushLock);

		return HnswInsertTupleOnDisk(index, buildstate->typeInfo, support, value, heaptid, true);
	}

	/* Ok, we can proceed to allocate the element */
	element = HnswInitElement(base, heaptid, buildstate->m, buildstate->ml, buildstate->maxLevel, allocator);
	valuePtr = HnswAlloc(allocator, valueSize);

	/*
	 * We have now allocated the space needed for the element, so we don't
	 * need the allocator lock anymore. Release it and initialize the rest of
	 * the element.
	 */
	LWLockRelease(&graph->allocatorLock);

	/* Copy the datum */
	memcpy(valuePtr, DatumGetPointer(value), valueSize);
	HnswPtrStore(base, element->value, (char *) valuePtr);

	/* Create a lock for the element */
	LWLockInitialize(&element->lock, hnsw_lock_tranche_id);

	/* Insert tuple */
	InsertTupleInMemory(buildstate, element);

	/* Release flush lock */
	LWLockRelease(flushLock);

	return true;
}

/*
 * Callback for table_index_build_scan
 */
static void
BuildCallback(Relation index, ItemPointer tid, Datum *values,
			  bool *isnull, bool tupleIsAlive, void *state)
{
	HnswBuildState *buildstate = (HnswBuildState *) state;
	HnswGraph  *graph = buildstate->graph;
	MemoryContext oldCtx;

	/* Skip nulls */
	if (isnull[0])
		return;

	/* Use memory context */
	oldCtx = MemoryContextSwitchTo(buildstate->tmpCtx);

	/* Insert tuple */
	if (InsertTuple(index, values, isnull, tid, buildstate))
	{
		/* Update progress */
		SpinLockAcquire(&graph->lock);
		pgstat_progress_update_param(PROGRESS_CREATEIDX_TUPLES_DONE, ++graph->indtuples);
		SpinLockRelease(&graph->lock);
	}

	/* Reset memory context */
	MemoryContextSwitchTo(oldCtx);
	MemoryContextReset(buildstate->tmpCtx);
}

/*
 * Initialize the graph
 */
static void
InitGraph(HnswGraph * graph, char *base, Size memoryTotal)
{
	/* Initialize the lock tranche if needed */
	HnswInitLockTranche();

	HnswPtrStore(base, graph->head, (HnswElement) NULL);
	HnswPtrStore(base, graph->entryPoint, (HnswElement) NULL);
	graph->memoryUsed = 0;
	graph->memoryTotal = Min(memoryTotal, HNSW_MAX_GRAPH_MEMORY);
	graph->flushed = false;
	graph->indtuples = 0;
	SpinLockInit(&graph->lock);
	LWLockInitialize(&graph->entryLock, hnsw_lock_tranche_id);
	LWLockInitialize(&graph->entryWaitLock, hnsw_lock_tranche_id);
	LWLockInitialize(&graph->allocatorLock, hnsw_lock_tranche_id);
	LWLockInitialize(&graph->flushLock, hnsw_lock_tranche_id);
}

/*
 * Initialize an allocator
 */
static void
InitAllocator(HnswAllocator * allocator, void *(*alloc) (Size size, void *state), void *state)
{
	allocator->alloc = alloc;
	allocator->state = state;
}

/*
 * Memory context allocator
 */
static void *
HnswMemoryContextAlloc(Size size, void *state)
{
	HnswBuildState *buildstate = (HnswBuildState *) state;
	void	   *chunk = MemoryContextAlloc(buildstate->graphCtx, size);

	buildstate->graphData.memoryUsed = MemoryContextMemAllocated(buildstate->graphCtx, false);

	return chunk;
}

/*
 * Shared memory allocator
 */
static void *
HnswSharedMemoryAlloc(Size size, void *state)
{
	HnswBuildState *buildstate = (HnswBuildState *) state;
	Size		alignedSize = MAXALIGN(size);
	Size		newMemoryUsed;
	void	   *chunk;

	if (alignedSize > 1024 * 1024)
		elog(ERROR, "hnsw allocation too large");

	newMemoryUsed = add_size(buildstate->graph->memoryUsed, alignedSize);
	if (newMemoryUsed > buildstate->graph->memoryTotal)
		elog(ERROR, "hnsw allocator out of memory");

	chunk = buildstate->hnswarea + buildstate->graph->memoryUsed;
	buildstate->graph->memoryUsed = newMemoryUsed;
	return chunk;
}

/*
 * Initialize the build state
 */
static void
InitBuildState(HnswBuildState * buildstate, Relation heap, Relation index, IndexInfo *indexInfo, ForkNumber forkNum)
{
	buildstate->heap = heap;
	buildstate->index = index;
	buildstate->indexInfo = indexInfo;
	buildstate->forkNum = forkNum;
	buildstate->typeInfo = HnswGetTypeInfo(index);

	buildstate->m = HnswGetM(index);
	buildstate->efConstruction = HnswGetEfConstruction(index);
	buildstate->dimensions = TupleDescAttr(index->rd_att, 0)->atttypmod;

	/* Disallow varbit since require fixed dimensions */
	if (TupleDescAttr(index->rd_att, 0)->atttypid == VARBITOID)
		ereport(ERROR,
				(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
				 errmsg("type not supported for hnsw index")));

	/* Require column to have dimensions to be indexed */
	if (buildstate->dimensions < 0)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("column does not have dimensions")));

	if (buildstate->dimensions > buildstate->typeInfo->maxDimensions)
		ereport(ERROR,
				(errcode(ERRCODE_PROGRAM_LIMIT_EXCEEDED),
				 errmsg("column cannot have more than %d dimensions for hnsw index", buildstate->typeInfo->maxDimensions)));

	if (buildstate->efConstruction / 2 < buildstate->m)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("ef_construction must be greater than or equal to 2 * m")));

	buildstate->reltuples = 0;
	buildstate->indtuples = 0;

	/* Get support functions */
	HnswInitSupport(&buildstate->support, index);

	InitGraph(&buildstate->graphData, NULL, mul_size(maintenance_work_mem, 1024));
	buildstate->graph = &buildstate->graphData;
	buildstate->ml = HnswGetMl(buildstate->m);
	buildstate->maxLevel = HnswGetMaxLevel(buildstate->m);

	/* Take the graph from hnsw.build_graph_file (not for the init fork) */
	buildstate->importGraph = heap != NULL && hnsw_build_graph_file != NULL && hnsw_build_graph_file[0] != '\0';

	buildstate->graphCtx = GenerationContextCreate(CurrentMemoryContext,
												   "Hnsw build graph context",
#if PG_VERSION_NUM >= 150000
												   1024 * 1024, 1024 * 1024,
#endif
												   1024 * 1024);
	buildstate->tmpCtx = AllocSetContextCreate(CurrentMemoryContext,
											   "Hnsw build temporary context",
											   ALLOCSET_DEFAULT_SIZES);

	InitAllocator(&buildstate->allocator, &HnswMemoryContextAlloc, buildstate);

	buildstate->hnswleader = NULL;
	buildstate->hnswshared = NULL;
	buildstate->hnswarea = NULL;
}

/*
 * Free resources
 */
static void
FreeBuildState(HnswBuildState * buildstate)
{
	MemoryContextDelete(buildstate->graphCtx);
	MemoryContextDelete(buildstate->tmpCtx);
}

/*
 * Within leader, wait for end of heap scan
 */
static double
ParallelHeapScan(HnswBuildState * buildstate)
{
	HnswShared *hnswshared = buildstate->hnswleader->hnswshared;
	int			nparticipanttuplesorts;
	double		reltuples;

	nparticipanttuplesorts = buildstate->hnswleader->nparticipanttuplesorts;
	for (;;)
	{
		SpinLockAcquire(&hnswshared->mutex);
		if (hnswshared->nparticipantsdone == nparticipanttuplesorts)
		{
			buildstate->graph = &hnswshared->graphData;
			buildstate->hnswarea = buildstate->hnswleader->hnswarea;
			reltuples = hnswshared->reltuples;
			SpinLockRelease(&hnswshared->mutex);
			break;
		}
		SpinLockRelease(&hnswshared->mutex);

		ConditionVariableSleep(&hnswshared->workersdonecv,
							   WAIT_EVENT_PARALLEL_CREATE_INDEX_SCAN);
	}

	ConditionVariableCancelSleep();

	return reltuples;
}

/*
 * Perform a worker's portion of a parallel insert
 */
static void
HnswParallelScanAndInsert(Relation heapRel, Relation indexRel, HnswShared * hnswshared, char *hnswarea, bool progress)
{
	HnswBuildState buildstate;
	TableScanDesc scan;
	double		reltuples;
	IndexInfo  *indexInfo;

	/* Join parallel scan */
	indexInfo = BuildIndexInfo(indexRel);
	indexInfo->ii_Concurrent = hnswshared->isconcurrent;
	InitBuildState(&buildstate, heapRel, indexRel, indexInfo, MAIN_FORKNUM);
	buildstate.graph = &hnswshared->graphData;
	buildstate.hnswarea = hnswarea;
	InitAllocator(&buildstate.allocator, &HnswSharedMemoryAlloc, &buildstate);
	scan = table_beginscan_parallel(heapRel,
									ParallelTableScanFromHnswShared(hnswshared)
#if PG_VERSION_NUM >= 190000
									,SO_NONE
#endif
		);
	reltuples = table_index_build_scan(heapRel, indexRel, indexInfo,
									   true, progress, BuildCallback,
									   (void *) &buildstate, scan);

	/* Record statistics */
	SpinLockAcquire(&hnswshared->mutex);
	hnswshared->nparticipantsdone++;
	hnswshared->reltuples += reltuples;
	SpinLockRelease(&hnswshared->mutex);

	/* Log statistics */
	if (progress)
		ereport(DEBUG1, (errmsg("leader processed " INT64_FORMAT " tuples", (int64) reltuples)));
	else
		ereport(DEBUG1, (errmsg("worker processed " INT64_FORMAT " tuples", (int64) reltuples)));

	/* Notify leader */
	ConditionVariableSignal(&hnswshared->workersdonecv);

	FreeBuildState(&buildstate);
}

/*
 * Perform work within a launched parallel process
 */
void
HnswParallelBuildMain(dsm_segment *seg, shm_toc *toc)
{
	char	   *sharedquery;
	HnswShared *hnswshared;
	char	   *hnswarea;
	Relation	heapRel;
	Relation	indexRel;
	LOCKMODE	heapLockmode;
	LOCKMODE	indexLockmode;

	/* Set debug_query_string for individual workers first */
	sharedquery = shm_toc_lookup(toc, PARALLEL_KEY_QUERY_TEXT, true);
	debug_query_string = sharedquery;

	/* Report the query string from leader */
	pgstat_report_activity(STATE_RUNNING, debug_query_string);

	/* Look up shared state */
	hnswshared = shm_toc_lookup(toc, PARALLEL_KEY_HNSW_SHARED, false);

	/* Open relations using lock modes known to be obtained by index.c */
	if (!hnswshared->isconcurrent)
	{
		heapLockmode = ShareLock;
		indexLockmode = AccessExclusiveLock;
	}
	else
	{
		heapLockmode = ShareUpdateExclusiveLock;
		indexLockmode = RowExclusiveLock;
	}

	/* Open relations within worker */
	heapRel = table_open(hnswshared->heaprelid, heapLockmode);
	indexRel = index_open(hnswshared->indexrelid, indexLockmode);

	hnswarea = shm_toc_lookup(toc, PARALLEL_KEY_HNSW_AREA, false);

	/* Perform inserts */
	HnswParallelScanAndInsert(heapRel, indexRel, hnswshared, hnswarea, false);

	/* Close relations within worker */
	index_close(indexRel, indexLockmode);
	table_close(heapRel, heapLockmode);
}

/*
 * End parallel build
 */
static void
HnswEndParallel(HnswLeader * hnswleader)
{
	/* Shutdown worker processes */
	WaitForParallelWorkersToFinish(hnswleader->pcxt);

	/* Free last reference to MVCC snapshot, if one was used */
	if (IsMVCCSnapshot(hnswleader->snapshot))
		UnregisterSnapshot(hnswleader->snapshot);
	DestroyParallelContext(hnswleader->pcxt);
	ExitParallelMode();
}

/*
 * Return size of shared memory required for parallel index build
 */
static Size
ParallelEstimateShared(Relation heap, Snapshot snapshot)
{
	return add_size(BUFFERALIGN(sizeof(HnswShared)), table_parallelscan_estimate(heap, snapshot));
}

/*
 * Within leader, participate as a parallel worker
 */
static void
HnswLeaderParticipateAsWorker(HnswBuildState * buildstate)
{
	HnswLeader *hnswleader = buildstate->hnswleader;

	/* Perform work common to all participants */
	HnswParallelScanAndInsert(buildstate->heap, buildstate->index, hnswleader->hnswshared, hnswleader->hnswarea, true);
}

/*
 * Begin parallel build
 */
static void
HnswBeginParallel(HnswBuildState * buildstate, bool isconcurrent, int request)
{
	ParallelContext *pcxt;
	Snapshot	snapshot;
	Size		esthnswshared;
	Size		esthnswarea;
	Size		estother;
	HnswShared *hnswshared;
	char	   *hnswarea;
	HnswLeader *hnswleader = palloc0_object(HnswLeader);
	bool		leaderparticipates = true;
	Size		querylen;

#ifdef DISABLE_LEADER_PARTICIPATION
	leaderparticipates = false;
#endif

	/* Enter parallel mode and create context */
	EnterParallelMode();
	Assert(request > 0);
	pcxt = CreateParallelContext("vectorfast", "HnswParallelBuildMain", request);

	/* Get snapshot for table scan */
	if (!isconcurrent)
		snapshot = SnapshotAny;
	else
		snapshot = RegisterSnapshot(GetTransactionSnapshot());

	/* Estimate size of workspaces */
	esthnswshared = ParallelEstimateShared(buildstate->heap, snapshot);
	shm_toc_estimate_chunk(&pcxt->estimator, esthnswshared);

	/* Leave space for other objects in shared memory */
	/* Docker has a default limit of 64 MB for shm_size */
	/* which happens to be the default value of maintenance_work_mem */
	esthnswarea = mul_size(maintenance_work_mem, 1024);
	estother = 3 * 1024 * 1024;
	if (esthnswarea > estother)
		esthnswarea -= estother;

	esthnswarea = Min(esthnswarea, HNSW_MAX_GRAPH_MEMORY);

	shm_toc_estimate_chunk(&pcxt->estimator, esthnswarea);
	shm_toc_estimate_keys(&pcxt->estimator, 2);

	/* Finally, estimate PARALLEL_KEY_QUERY_TEXT space */
	if (debug_query_string)
	{
		querylen = strlen(debug_query_string);
		shm_toc_estimate_chunk(&pcxt->estimator, querylen + 1);
		shm_toc_estimate_keys(&pcxt->estimator, 1);
	}
	else
		querylen = 0;			/* keep compiler quiet */

	/* Everyone's had a chance to ask for space, so now create the DSM */
	InitializeParallelDSM(pcxt);

	/* If no DSM segment was available, back out (do serial build) */
	if (pcxt->seg == NULL)
	{
		if (IsMVCCSnapshot(snapshot))
			UnregisterSnapshot(snapshot);
		DestroyParallelContext(pcxt);
		ExitParallelMode();
		return;
	}

	/* Store shared build state, for which we reserved space */
	hnswshared = (HnswShared *) shm_toc_allocate(pcxt->toc, esthnswshared);
	/* Initialize immutable state */
	hnswshared->heaprelid = RelationGetRelid(buildstate->heap);
	hnswshared->indexrelid = RelationGetRelid(buildstate->index);
	hnswshared->isconcurrent = isconcurrent;
	ConditionVariableInit(&hnswshared->workersdonecv);
	SpinLockInit(&hnswshared->mutex);
	/* Initialize mutable state */
	hnswshared->nparticipantsdone = 0;
	hnswshared->reltuples = 0;
	table_parallelscan_initialize(buildstate->heap,
								  ParallelTableScanFromHnswShared(hnswshared),
								  snapshot);

	hnswarea = (char *) shm_toc_allocate(pcxt->toc, esthnswarea);
	InitGraph(&hnswshared->graphData, hnswarea, esthnswarea);

	/*
	 * Avoid base address for relptr for Postgres < 14.5
	 * https://github.com/postgres/postgres/commit/7201cd18627afc64850537806da7f22150d1a83b
	 */
#if PG_VERSION_NUM < 140005
	hnswshared->graphData.memoryUsed = add_size(hnswshared->graphData.memoryUsed, MAXALIGN(1));
#endif

	shm_toc_insert(pcxt->toc, PARALLEL_KEY_HNSW_SHARED, hnswshared);
	shm_toc_insert(pcxt->toc, PARALLEL_KEY_HNSW_AREA, hnswarea);

	/* Store query string for workers */
	if (debug_query_string)
	{
		char	   *sharedquery;

		sharedquery = (char *) shm_toc_allocate(pcxt->toc, querylen + 1);
		memcpy(sharedquery, debug_query_string, querylen + 1);
		shm_toc_insert(pcxt->toc, PARALLEL_KEY_QUERY_TEXT, sharedquery);
	}

	/* Launch workers, saving status for leader/caller */
	LaunchParallelWorkers(pcxt);
	hnswleader->pcxt = pcxt;
	hnswleader->nparticipanttuplesorts = pcxt->nworkers_launched;
	if (leaderparticipates)
		hnswleader->nparticipanttuplesorts++;
	hnswleader->hnswshared = hnswshared;
	hnswleader->snapshot = snapshot;
	hnswleader->hnswarea = hnswarea;

	/* If no workers were successfully launched, back out (do serial build) */
	if (pcxt->nworkers_launched == 0)
	{
		HnswEndParallel(hnswleader);
		return;
	}

	/* Log participants */
	ereport(DEBUG1, (errmsg("using %d parallel workers", pcxt->nworkers_launched)));

	/* Save leader state now that it's clear build will be parallel */
	buildstate->hnswleader = hnswleader;

	/* Join heap scan ourselves */
	if (leaderparticipates)
		HnswLeaderParticipateAsWorker(buildstate);

	/* Wait for all launched workers */
	WaitForParallelWorkersToAttach(pcxt);
}

/*
 * Compute parallel workers
 */
static int
ComputeParallelWorkers(Relation heap, Relation index)
{
	int			parallel_workers;

	/* Make sure it's safe to use parallel workers */
	parallel_workers = plan_create_index_workers(RelationGetRelid(heap), RelationGetRelid(index));
	if (parallel_workers == 0)
		return 0;

	/* Use parallel_workers storage parameter on table if set */
	parallel_workers = RelationGetParallelWorkers(heap, -1);
	if (parallel_workers != -1)
		return Min(parallel_workers, max_parallel_maintenance_workers);

	return max_parallel_maintenance_workers;
}

/*
 * Graph file written by tools/pipnn_export (little-endian): this header, then
 * one HnswGraphFileTid per node, then per node a uint32 degree followed by
 * maxDegree uint32 neighbor node numbers (unused slots 0xFFFFFFFF).
 *
 * HNSW_GRAPH_FILE_NODE_ORDER in flags says the nodes are in Ginger's
 * reordered (locality) order rather than the order the rows were read. The
 * import matches nodes to rows by TID and lays elements out in heap order, so
 * it builds the same index either way.
 */
#define HNSW_GRAPH_FILE_MAGIC "PIPNNGR1"
#define HNSW_GRAPH_FILE_VERSION 1
#define HNSW_GRAPH_FILE_NODE_ORDER (1u << 0)

typedef struct HnswGraphFileHeader
{
	char		magic[8];
	uint32		version;
	uint32		maxDegree;
	uint64		numNodes;
	uint64		entryNode;
	uint32		dimensions;
	uint32		flags;
}			HnswGraphFileHeader;

typedef struct HnswGraphFileTid
{
	uint32		block;
	uint16		offset;
	uint16		pad;
}			HnswGraphFileTid;

StaticAssertDecl(sizeof(HnswGraphFileHeader) == 40, "graph file header layout");
StaticAssertDecl(sizeof(HnswGraphFileTid) == 8, "graph file TID layout");

static void
ReadGraphFile(FILE *f, void *buf, Size len, const char *path)
{
	if (fread(buf, 1, len, f) != len)
	{
		if (ferror(f))
			ereport(ERROR,
					(errcode_for_file_access(),
					 errmsg("could not read graph file \"%s\": %m", path)));
		ereport(ERROR,
				(errcode(ERRCODE_DATA_CORRUPTED),
				 errmsg("graph file \"%s\" ends early", path)));
	}
}

static int
CompareNodeTids(const void *a, const void *b, void *arg)
{
	const HnswGraphFileTid *tids = (const HnswGraphFileTid *) arg;
	const HnswGraphFileTid *ta = &tids[*(const uint32 *) a];
	const HnswGraphFileTid *tb = &tids[*(const uint32 *) b];

	if (ta->block != tb->block)
		return ta->block < tb->block ? -1 : 1;
	if (ta->offset != tb->offset)
		return ta->offset < tb->offset ? -1 : 1;
	return 0;
}

/*
 * A row collected by a graph import: its heap TID, and where its element
 * tuple and neighbor tuple went in the index
 */
typedef struct HnswImportRow
{
	ItemPointerData heaptid;
	ItemPointerData elementTid;
	ItemPointerData neighborTid;
}			HnswImportRow;

typedef struct HnswImportState
{
	HnswBuildState *buildstate;
	Buffer		buf;
	Page		page;
	HnswElementTuple etup;
	HnswNeighborTuple placeholder;
	Size		ntupSize;
	HnswImportRow *rows;
	uint64		nrows;
	uint64		maxRows;
}			HnswImportState;

static int
CompareRowTids(const void *a, const void *b, void *arg)
{
	const HnswImportRow *rows = (const HnswImportRow *) arg;

	return ItemPointerCompare((ItemPointer) &rows[*(const uint32 *) a].heaptid,
							  (ItemPointer) &rows[*(const uint32 *) b].heaptid);
}

/*
 * Callback for table_index_build_scan in a graph import: write the row's
 * element tuple and a placeholder for its neighbor tuple
 */
static void
ImportCallback(Relation index, ItemPointer tid, Datum *values,
			   bool *isnull, bool tupleIsAlive, void *state)
{
	HnswImportState *is = (HnswImportState *) state;
	HnswBuildState *buildstate = is->buildstate;
	MemoryContext oldCtx;
	Datum		value;

	/* Skip nulls */
	if (isnull[0])
		return;

	oldCtx = MemoryContextSwitchTo(buildstate->tmpCtx);

	if (HnswFormIndexValue(&value, values, isnull, buildstate->typeInfo, &buildstate->support))
	{
		Pointer		valuePtr;
		Size		etupSize;
		Size		combinedSize;
		HnswImportRow *row;
		BlockNumber blkno;
		OffsetNumber offno;

		HnswCheckDim(buildstate->dimensions, buildstate->typeInfo, buildstate->support.collation, value);
		valuePtr = DatumGetPointer(value);

		if (is->nrows >= is->maxRows)
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					 errmsg("the table has more rows to index than the graph file's " UINT64_FORMAT " nodes", is->maxRows),
					 errhint("Export the graph again if the table changed.")));

		etupSize = HNSW_ELEMENT_TUPLE_SIZE(VARSIZE_ANY(valuePtr));
		if (etupSize > HNSW_TUPLE_ALLOC_SIZE)
			ereport(ERROR,
					(errcode(ERRCODE_PROGRAM_LIMIT_EXCEEDED),
					 errmsg("index tuple too large")));
		combinedSize = etupSize + is->ntupSize + sizeof(ItemIdData);

		/* A layer-0 element with one heap TID */
		MemSet(is->etup, 0, HNSW_TUPLE_ALLOC_SIZE);
		is->etup->type = HNSW_ELEMENT_TUPLE_TYPE;
		is->etup->level = 0;
		is->etup->deleted = 0;
		is->etup->version = 1;
		is->etup->heaptids[0] = *tid;
		for (int i = 1; i < HNSW_HEAPTIDS; i++)
			ItemPointerSetInvalid(&is->etup->heaptids[i]);
		memcpy(&is->etup->data, valuePtr, VARSIZE_ANY(valuePtr));

		/* Keep element and neighbors on the same page if possible, as CreateGraphPages() */
		if (PageGetFreeSpace(is->page) < etupSize || (combinedSize <= HNSW_MAX_SIZE && PageGetFreeSpace(is->page) < combinedSize))
			HnswBuildAppendPage(index, &is->buf, &is->page, buildstate->forkNum);

		blkno = BufferGetBlockNumber(is->buf);
		offno = OffsetNumberNext(PageGetMaxOffsetNumber(is->page));
		row = &is->rows[is->nrows++];
		row->heaptid = *tid;
		ItemPointerSet(&row->elementTid, blkno, offno);
		if (combinedSize <= HNSW_MAX_SIZE)
			ItemPointerSet(&row->neighborTid, blkno, OffsetNumberNext(offno));
		else
			ItemPointerSet(&row->neighborTid, blkno + 1, FirstOffsetNumber);
		is->etup->neighbortid = row->neighborTid;

		if (PageAddItem(is->page, (Item) is->etup, etupSize, InvalidOffsetNumber, false, false) != offno)
			elog(ERROR, "failed to add index item to \"%s\"", RelationGetRelationName(index));

		if (PageGetFreeSpace(is->page) < is->ntupSize)
			HnswBuildAppendPage(index, &is->buf, &is->page, buildstate->forkNum);

		/* Placeholder for the neighbors, written once every element has a place */
		if (PageAddItem(is->page, (Item) is->placeholder, is->ntupSize, InvalidOffsetNumber, false, false) !=
			ItemPointerGetOffsetNumber(&row->neighborTid))
			elog(ERROR, "failed to add index item to \"%s\"", RelationGetRelationName(index));

		pgstat_progress_update_param(PROGRESS_CREATEIDX_TUPLES_DONE, (int64) is->nrows);
	}

	MemoryContextSwitchTo(oldCtx);
	MemoryContextReset(buildstate->tmpCtx);
}

/*
 * Build the index from the graph in hnsw.build_graph_file, without an
 * in-memory graph (see the top of this file)
 */
static void
BuildGraphFromFile(HnswBuildState * buildstate)
{
	const char *path = hnsw_build_graph_file;
	Relation	index = buildstate->index;
	ForkNumber	forkNum = buildstate->forkNum;
	int			layerM = HnswGetLayerM(buildstate->m, 0);
	HnswGraphFileHeader hdr;
	HnswGraphFileTid *tids;
	HnswImportState is;
	HnswNeighborTuple ntup;
	HnswElementData entry;
	BlockNumber insertPage;
	uint32	   *rowOrder;
	uint32	   *nodeOrder;
	uint32	   *nodeRow;
	uint32	   *fileRow;
	uint64		n;
	uint64		edges = 0;
	uint64		selfLoops = 0;
	FILE	   *f;

	f = AllocateFile(path, PG_BINARY_R);
	if (f == NULL)
		ereport(ERROR,
				(errcode_for_file_access(),
				 errmsg("could not open graph file \"%s\": %m", path)));

	ReadGraphFile(f, &hdr, sizeof(hdr), path);
	if (memcmp(hdr.magic, HNSW_GRAPH_FILE_MAGIC, sizeof(hdr.magic)) != 0 || hdr.version != HNSW_GRAPH_FILE_VERSION)
		ereport(ERROR,
				(errcode(ERRCODE_DATA_CORRUPTED),
				 errmsg("\"%s\" is not a graph file written by pipnn_export", path)));
	if ((int) hdr.dimensions != buildstate->dimensions)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("graph file has %u dimensions, but the indexed column has %d", hdr.dimensions, buildstate->dimensions)));
	if (hdr.maxDegree > (uint32) layerM)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("graph file has up to %u neighbors per node, but m = %d stores at most %d", hdr.maxDegree, buildstate->m, layerM),
				 errhint("Use m = %u or more.", (hdr.maxDegree + 1) / 2)));
	n = hdr.numNodes;
	if (n == 0 || n > PG_UINT32_MAX)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("graph file has " UINT64_FORMAT " nodes; expected 1 to %u", n, PG_UINT32_MAX)));
	if (hdr.entryNode >= n)
		ereport(ERROR,
				(errcode(ERRCODE_DATA_CORRUPTED),
				 errmsg("graph file entry node " UINT64_FORMAT " is out of range", hdr.entryNode)));

	tids = palloc_extended(mul_size(n, sizeof(HnswGraphFileTid)), MCXT_ALLOC_HUGE);
	ReadGraphFile(f, tids, n * sizeof(HnswGraphFileTid), path);

	/* 1. Element tuples and neighbor placeholders, in heap scan order */
	CreateMetaPage(buildstate);
	pgstat_progress_update_param(PROGRESS_CREATEIDX_SUBPHASE, PROGRESS_HNSW_PHASE_LOAD);

	is.buildstate = buildstate;
	is.buf = HnswNewBuffer(index, forkNum);
	is.page = BufferGetPage(is.buf);
	HnswInitPage(is.buf, is.page);
	is.etup = palloc0(HNSW_TUPLE_ALLOC_SIZE);
	is.placeholder = palloc0(HNSW_TUPLE_ALLOC_SIZE);
	is.ntupSize = HNSW_NEIGHBOR_TUPLE_SIZE(0, buildstate->m);
	is.rows = palloc_extended(mul_size(n, sizeof(HnswImportRow)), MCXT_ALLOC_HUGE);
	is.nrows = 0;
	is.maxRows = n;

	buildstate->reltuples = table_index_build_scan(buildstate->heap, index, buildstate->indexInfo,
												   true, true, ImportCallback, (void *) &is, NULL);

	insertPage = BufferGetBlockNumber(is.buf);
	MarkBufferDirty(is.buf);
	UnlockReleaseBuffer(is.buf);

	if (is.nrows != n)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("the table has " UINT64_FORMAT " rows to index, but the graph file has " UINT64_FORMAT " nodes", is.nrows, n),
				 errhint("Export the graph again if the table changed.")));
	buildstate->indtuples = n;

	/* 2. Match graph nodes to rows: sort both by heap TID and walk them */
	rowOrder = palloc_extended(mul_size(n, sizeof(uint32)), MCXT_ALLOC_HUGE);
	nodeOrder = palloc_extended(mul_size(n, sizeof(uint32)), MCXT_ALLOC_HUGE);
	for (uint64 i = 0; i < n; i++)
	{
		rowOrder[i] = (uint32) i;
		nodeOrder[i] = (uint32) i;
	}
	qsort_arg(rowOrder, n, sizeof(uint32), CompareRowTids, is.rows);
	qsort_arg(nodeOrder, n, sizeof(uint32), CompareNodeTids, tids);

	nodeRow = palloc_extended(mul_size(n, sizeof(uint32)), MCXT_ALLOC_HUGE);
	for (uint64 k = 0; k < n; k++)
	{
		uint32		node = nodeOrder[k];
		ItemPointer ht = &is.rows[rowOrder[k]].heaptid;

		if (ItemPointerGetBlockNumberNoCheck(ht) != tids[node].block ||
			ItemPointerGetOffsetNumberNoCheck(ht) != tids[node].offset)
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					 errmsg("graph file does not match the table: node %u is at TID (%u,%u), which has no row to index",
							node, tids[node].block, tids[node].offset),
					 errhint("Export the graph again if the table changed.")));
		nodeRow[node] = rowOrder[k];
	}
	pfree(rowOrder);
	pfree(nodeOrder);
	pfree(tids);

	/* 3. Neighbor tuples, streamed from the file one node at a time */
	ntup = palloc0(HNSW_TUPLE_ALLOC_SIZE);
	fileRow = palloc(mul_size(hdr.maxDegree + 1, sizeof(uint32)));
	for (uint64 i = 0; i < n; i++)
	{
		ItemPointer nt = &is.rows[nodeRow[i]].neighborTid;
		Buffer		buf;
		Page		page;
		uint32		degree;
		int			idx = 0;

		if ((i & 0xFFFF) == 0)
			CHECK_FOR_INTERRUPTS();

		ReadGraphFile(f, fileRow, (hdr.maxDegree + 1) * sizeof(uint32), path);
		degree = fileRow[0];
		if (degree > hdr.maxDegree)
			ereport(ERROR,
					(errcode(ERRCODE_DATA_CORRUPTED),
					 errmsg("graph file node " UINT64_FORMAT " has degree %u, above its maximum %u", i, degree, hdr.maxDegree)));

		MemSet(ntup, 0, is.ntupSize);
		ntup->type = HNSW_NEIGHBOR_TUPLE_TYPE;
		for (uint32 j = 0; j < degree; j++)
		{
			uint32		v = fileRow[1 + j];

			if (v >= n)
				ereport(ERROR,
						(errcode(ERRCODE_DATA_CORRUPTED),
						 errmsg("graph file node " UINT64_FORMAT " has neighbor %u, out of range", i, v)));
			if (v == i)
			{
				selfLoops++;
				continue;
			}
			ntup->indextids[idx++] = is.rows[nodeRow[v]].elementTid;
		}
		edges += idx;
		while (idx < layerM)
			ItemPointerSetInvalid(&ntup->indextids[idx++]);
		ntup->count = layerM;
		ntup->version = 1;

		buf = ReadBufferExtended(index, forkNum, ItemPointerGetBlockNumber(nt), RBM_NORMAL, NULL);
		LockBuffer(buf, BUFFER_LOCK_EXCLUSIVE);
		page = BufferGetPage(buf);
		if (!PageIndexTupleOverwrite(page, ItemPointerGetOffsetNumber(nt), (Item) ntup, is.ntupSize))
			elog(ERROR, "failed to add index item to \"%s\"", RelationGetRelationName(index));
		MarkBufferDirty(buf);
		UnlockReleaseBuffer(buf);
	}

	/* The entry point: Ginger's, on layer 0 */
	MemSet(&entry, 0, sizeof(entry));
	entry.blkno = ItemPointerGetBlockNumber(&is.rows[nodeRow[hdr.entryNode]].elementTid);
	entry.offno = ItemPointerGetOffsetNumber(&is.rows[nodeRow[hdr.entryNode]].elementTid);
	entry.level = 0;
	HnswUpdateMetaPage(index, HNSW_UPDATE_ENTRY_ALWAYS, &entry, insertPage, forkNum, true);

	if (FreeFile(f) != 0)
		ereport(ERROR,
				(errcode_for_file_access(),
				 errmsg("could not close graph file \"%s\": %m", path)));

	ereport(NOTICE,
			(errmsg("imported graph from \"%s\": " UINT64_FORMAT " elements, mean degree %.1f, entry node " UINT64_FORMAT,
					path, n, (double) edges / (double) n, hdr.entryNode),
			 selfLoops > 0 ? errdetail(UINT64_FORMAT " self-loops dropped.", selfLoops) : 0));

	pfree(fileRow);
	pfree(ntup);
	pfree(nodeRow);
	pfree(is.rows);
	pfree(is.etup);
	pfree(is.placeholder);
}

/*
 * Build graph
 */
static void
BuildGraph(HnswBuildState * buildstate)
{
	int			parallel_workers = 0;

	/* Take the graph from a file, without an in-memory graph */
	if (buildstate->importGraph)
	{
		BuildGraphFromFile(buildstate);
		return;
	}

	pgstat_progress_update_param(PROGRESS_CREATEIDX_SUBPHASE, PROGRESS_HNSW_PHASE_LOAD);

	/* Calculate parallel workers */
	if (buildstate->heap != NULL)
		parallel_workers = ComputeParallelWorkers(buildstate->heap, buildstate->index);

	/* Attempt to launch parallel worker scan when required */
	if (parallel_workers > 0)
		HnswBeginParallel(buildstate, buildstate->indexInfo->ii_Concurrent, parallel_workers);

	/* Add tuples to graph */
	if (buildstate->heap != NULL)
	{
		if (buildstate->hnswleader)
			buildstate->reltuples = ParallelHeapScan(buildstate);
		else
			buildstate->reltuples = table_index_build_scan(buildstate->heap, buildstate->index, buildstate->indexInfo,
														   true, true, BuildCallback, (void *) buildstate, NULL);

		buildstate->indtuples = buildstate->graph->indtuples;
	}

	/* Flush pages */
	if (!buildstate->graph->flushed)
		FlushPages(buildstate);

	/* End parallel build */
	if (buildstate->hnswleader)
		HnswEndParallel(buildstate->hnswleader);
}

/*
 * Build the index
 */
static void
BuildIndex(Relation heap, Relation index, IndexInfo *indexInfo,
		   HnswBuildState * buildstate, ForkNumber forkNum)
{
#ifdef HNSW_MEMORY
	SeedRandom(42);
#endif

	InitBuildState(buildstate, heap, index, indexInfo, forkNum);

	BuildGraph(buildstate);

	if (RelationNeedsWAL(index) || forkNum == INIT_FORKNUM)
		log_newpage_range(index, forkNum, 0, RelationGetNumberOfBlocksInFork(index, forkNum), true);

	FreeBuildState(buildstate);
}

/*
 * Build the index for a logged table
 */
IndexBuildResult *
hnswbuild(Relation heap, Relation index, IndexInfo *indexInfo)
{
	IndexBuildResult *result;
	HnswBuildState buildstate;

	BuildIndex(heap, index, indexInfo, &buildstate, MAIN_FORKNUM);

	result = palloc_object(IndexBuildResult);
	result->heap_tuples = buildstate.reltuples;
	result->index_tuples = buildstate.indtuples;

	return result;
}

/*
 * Build the index for an unlogged table
 */
void
hnswbuildempty(Relation index)
{
	IndexInfo  *indexInfo = BuildIndexInfo(index);
	HnswBuildState buildstate;

	BuildIndex(NULL, index, indexInfo, &buildstate, INIT_FORKNUM);
}
