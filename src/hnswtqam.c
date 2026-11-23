/*
 * TurboQuant indexes: HNSW with 4-bit codes (quantizer = 'tq4')
 *
 * The codec (hnswtq.c) is ported from Ginger. This file stores its output in
 * the index and scores with it:
 *
 * - Element tuples hold an HnswTqElementValue (4-bit codes, ||x||^2 and the
 *   4-bit scaling factor) instead of the vector. At 384 dimensions that is
 *   216 bytes instead of 1,544.
 * - With rerank = 'ntq12', each element also has 8 extra bits per dimension
 *   (a 4+8 bit code) in an HnswRerankTuple. Those are on their own pages,
 *   blocks 1 to graphHead - 1, which builds reserve ahead of the graph pages.
 * - A search navigates with 4-bit distances, then re-scores the ef_search
 *   candidates it found with the 4+8 bit codes and sorts them again.
 *
 * The graph itself is built with float distances, as for any index; only the
 * stored values change.
 *
 * Builds (CREATE INDEX, in memory, and graph-file imports) and searches are
 * supported. Inserts, vacuuming deleted rows and iterative scans are not yet.
 */
#include "postgres.h"

#include <math.h>

#include "hnsw.h"
#include "hnswio.h"
#include "hnswtq.h"
#include "storage/bufmgr.h"
#include "utils/memutils.h"
#include "utils/rel.h"
#include "vector.h"

#if PG_VERSION_NUM >= 160000
#include "varatt.h"
#endif

/* The codec's random signs and rotation come from this seed */
#define HNSW_TQ_SEED HNSW_TQ_DEFAULT_SEED

struct HnswTqState
{
	int			dim;
	int			rerank;			/* HNSW_RERANK_* */
	int			extraBits;		/* 0 without re-rank */
	HnswTqMetric metric;
	BlockNumber rerankHead;		/* first re-rank page, if any */
	BlockNumber graphHead;		/* first graph page */
	HnswTqCodec *codec;
	float	   *scratch;		/* HnswTqScratchSize(dim) bytes */
};

/* What a backend keeps per index in rd_amcache: the codec and settings */
typedef struct HnswTqCache
{
	HnswTqState state;			/* codec and scratch unset */
	Size		codecSize;
	char		codec[FLEXIBLE_ARRAY_MEMBER];
}			HnswTqCache;

/*
 * Get the quantizer of an index
 */
int
HnswGetQuantizer(Relation index)
{
	HnswOptions *opts = (HnswOptions *) index->rd_options;

	if (opts)
		return opts->quantizer;

	return HNSW_QUANTIZER_NONE;
}

/*
 * Fail for what tq4 indexes cannot do yet
 */
void
HnswTqCheckSupported(Relation index, const char *what, const char *hint)
{
	if (HnswGetQuantizer(index) != HNSW_QUANTIZER_NONE)
		ereport(ERROR,
				(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
				 errmsg("hnsw indexes with quantizer \"tq4\" do not support %s yet", what),
				 errhint("%s", hint)));
}

static int
HnswGetRerank(Relation index)
{
	HnswOptions *opts = (HnswOptions *) index->rd_options;

	if (opts)
		return opts->rerank;

	return HNSW_RERANK_NTQ12;
}

static HnswTqState *
CopyState(const HnswTqState * src, const HnswTqCodec * codec, Size codecSize)
{
	HnswTqState *tq = palloc_object(HnswTqState);

	*tq = *src;
	tq->codec = palloc(codecSize);
	memcpy(tq->codec, codec, codecSize);
	tq->scratch = palloc(HnswTqScratchSize(tq->dim));
	return tq;
}

/*
 * Create the state for building a tq4 index, and check that the index can
 * be one
 */
HnswTqState *
HnswTqCreateState(Relation index, const HnswSupport * support, int dimensions)
{
	HnswTqState state;
	HnswTqCodec *codec;
	Size		codecSize = HnswTqCodecSize(dimensions);

	if (support->vectorDistance == HNSW_VECTOR_DISTANCE_NONE)
		ereport(ERROR,
				(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
				 errmsg("quantizer \"tq4\" requires a vector column with L2, inner product or cosine distance")));

	codec = palloc0(codecSize);
	if (!HnswTqInitCodec(codec, dimensions, HNSW_TQ_SEED, NULL))
		ereport(ERROR,
				(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
				 errmsg("quantizer \"tq4\" does not support %d dimensions", dimensions),
				 errdetail("The dimensions must be an odd number of at most %d times a power of 2.",
						   HNSW_TQ_MAX_SUBSPACES)));

	MemSet(&state, 0, sizeof(state));
	state.dim = dimensions;
	state.rerank = HnswGetRerank(index);
	state.extraBits = state.rerank == HNSW_RERANK_NTQ12 ? HNSW_TQ_NTQ12_EXTRA_BITS : 0;
	state.metric = support->vectorDistance == HNSW_VECTOR_DISTANCE_L2 ? HNSW_TQ_METRIC_L2 : HNSW_TQ_METRIC_DOT;
	state.rerankHead = InvalidBlockNumber;
	state.graphHead = HNSW_HEAD_BLKNO;

	return CopyState(&state, codec, codecSize);
}

/*
 * Get the state of an existing tq4 index, from its metapage. The codec is
 * cached in rd_amcache; each caller gets its own copy, with its own scratch.
 */
HnswTqState *
HnswTqGetState(Relation index)
{
	HnswTqCache *cache = (HnswTqCache *) index->rd_amcache;

	if (cache == NULL)
	{
		Buffer		buf;
		Page		page;
		HnswMetaPage metap;
		Size		codecSize;
		HnswTqCodec *codec;

		buf = ReadBuffer(index, HNSW_METAPAGE_BLKNO);
		LockBuffer(buf, BUFFER_LOCK_SHARE);
		page = BufferGetPage(buf);
		metap = HnswPageGetMeta(page);

		if (metap->magicNumber != HNSW_MAGIC_NUMBER)
			elog(ERROR, "hnsw index is not valid");
		if (metap->version != HNSW_TQ_VERSION || metap->quantizer != HNSW_QUANTIZER_TQ4)
			elog(ERROR, "hnsw index \"%s\" has quantizer \"tq4\" but its metapage does not",
				 RelationGetRelationName(index));
		if (metap->tqSubspaces != HnswTqSubspaces(metap->dimensions) ||
			!HnswTqValidExtraBits(metap->tqExtraBits))
			elog(ERROR, "hnsw index \"%s\" has an invalid TurboQuant metapage",
				 RelationGetRelationName(index));

		codecSize = HnswTqCodecSize(metap->dimensions);
		cache = MemoryContextAllocZero(index->rd_indexcxt, offsetof(HnswTqCache, codec) + codecSize);
		codec = (HnswTqCodec *) cache->codec;
		if (!HnswTqInitCodec(codec, metap->dimensions, metap->tqSeed, metap->tqRotation))
			elog(ERROR, "hnsw index \"%s\" has an invalid TurboQuant metapage",
				 RelationGetRelationName(index));

		cache->codecSize = codecSize;
		cache->state.dim = metap->dimensions;
		cache->state.rerank = metap->rerank;
		cache->state.extraBits = metap->tqExtraBits;
		cache->state.metric = (HnswTqMetric) metap->tqMetric;
		cache->state.rerankHead = metap->rerankHead;
		cache->state.graphHead = metap->graphHead;

		UnlockReleaseBuffer(buf);

		index->rd_amcache = cache;
	}

	return CopyState(&cache->state, (HnswTqCodec *) cache->codec, cache->codecSize);
}

/*
 * Write the TurboQuant fields of a new metapage
 */
void
HnswTqSetMetaPage(Page page, const HnswTqState * tq)
{
	HnswMetaPage metap = HnswPageGetMeta(page);
	int			r = tq->codec->subspaces;

	metap->version = HNSW_TQ_VERSION;
	metap->quantizer = HNSW_QUANTIZER_TQ4;
	metap->rerank = tq->rerank;
	metap->tqExtraBits = tq->extraBits;
	metap->tqMetric = tq->metric;
	metap->tqSeed = tq->codec->seed;
	metap->tqSubspaces = r;
	metap->tqUnused = 0;
	metap->rerankHead = InvalidBlockNumber;
	metap->rerankInsertPage = InvalidBlockNumber;
	metap->graphHead = HNSW_HEAD_BLKNO;
	memcpy(metap->tqRotation, tq->codec->rotation, sizeof(float) * r * r);

	((PageHeader) page)->pd_lower =
		((char *) &metap->tqRotation[r * r]) - (char *) page;
}

/*
 * Record where the re-rank and graph pages are (during a build)
 */
void
HnswTqSetRerankPages(Relation index, BlockNumber rerankHead, BlockNumber rerankInsertPage,
					 BlockNumber graphHead, ForkNumber forkNum)
{
	Buffer		buf;
	HnswMetaPage metap;

	buf = ReadBufferExtended(index, forkNum, HNSW_METAPAGE_BLKNO, RBM_NORMAL, NULL);
	LockBuffer(buf, BUFFER_LOCK_EXCLUSIVE);
	metap = HnswPageGetMeta(BufferGetPage(buf));
	metap->rerankHead = rerankHead;
	metap->rerankInsertPage = rerankInsertPage;
	metap->graphHead = graphHead;
	MarkBufferDirty(buf);
	UnlockReleaseBuffer(buf);
}

Size
HnswTqElementValueSize(const HnswTqState * tq)
{
	return HNSW_TQ_ELEMENT_VALUE_SIZE(tq->dim);
}

/*
 * Size of a re-rank tuple, aligned as it is stored; 0 without re-rank
 */
int
HnswTqRerankTupleSize(const HnswTqState * tq)
{
	if (tq->extraBits == 0)
		return 0;
	return MAXALIGN(offsetof(HnswRerankTupleData, extra) + HnswTqExtraBytes(tq->dim, tq->extraBits));
}

/*
 * Encode a vector into an element value and, with re-rank, a re-rank tuple.
 * ev holds HnswTqElementValueSize() bytes, and rtup HnswTqRerankTupleSize()
 * zeroed bytes (or is NULL without re-rank). ev->reranktid is left invalid.
 */
void
HnswTqEncode(HnswTqState * tq, Datum value, HnswTqElementValue * ev, HnswRerankTuple rtup)
{
	Vector	   *vec = DatumGetVector(value);
	HnswTqPointInfo info;

	Assert(vec->dim == tq->dim);

	if (!HnswTqEncodePoint(tq->codec, vec->x, tq->extraBits, ev->codes,
						   rtup != NULL ? rtup->extra : NULL, &info, tq->scratch))
		ereport(ERROR,
				(errcode(ERRCODE_NUMERIC_VALUE_OUT_OF_RANGE),
				 errmsg("cannot quantize a vector whose norm overflows")));

	SET_VARSIZE(ev, HNSW_TQ_ELEMENT_VALUE_SIZE(tq->dim));
	ev->marker = HNSW_TQ_ELEMENT_MARKER;
	ev->dim = tq->dim;
	ev->sqNorm = info.sqNorm;
	ev->scale4 = info.scale4;
	ItemPointerSetInvalid(&ev->reranktid);
	ev->unused = 0;

	if (rtup != NULL)
	{
		rtup->type = HNSW_RERANK_TUPLE_TYPE;
		rtup->deleted = 0;
		rtup->unused = 0;
		rtup->scale4k = info.scale4k;
	}
}

/*
 * Make the HnswQuery value for a search of a tq4 index
 */
Datum
HnswTqMakeQuery(HnswTqState * tq, Datum value)
{
	Vector	   *vec = DatumGetVector(value);
	HnswTqQueryValue *qv = palloc0_object(HnswTqQueryValue);
	HnswTqQuery *query = HnswTqInitQuery(palloc(HnswTqQuerySize(tq->dim)), tq->dim);

	Assert(vec->dim == tq->dim);

	if (!HnswTqEncodeQuery(tq->codec, vec->x, query, tq->scratch))
		ereport(ERROR,
				(errcode(ERRCODE_NUMERIC_VALUE_OUT_OF_RANGE),
				 errmsg("cannot quantize a query vector whose norm overflows")));

	SET_VARSIZE(qv, sizeof(HnswTqQueryValue));
	qv->marker = HNSW_TQ_QUERY_MARKER;
	qv->dim = tq->dim;
	qv->metric = tq->metric;
	qv->query = query;
	return PointerGetDatum(qv);
}

/*
 * 4-bit distance from a query to a stored element value. Can read a value
 * that is being changed (optimistic reads); the caller validates afterwards.
 */
double
HnswTqQueryDistance(const HnswTqQueryValue * qv, const HnswTqElementValue * ev)
{
	HnswTqPointInfo info;

	info.sqNorm = ev->sqNorm;
	info.scale4 = ev->scale4;
	info.scale4k = ev->scale4;
	return (double) HnswTqDistance4(qv->query, ev->codes, &info, (HnswTqMetric) qv->metric);
}

/*
 * Re-rank writer. Builds reserve the re-rank pages before the graph pages
 * (so the graph's pages stay consecutive, which its placement relies on),
 * then fill them in element order.
 */
static int
RerankTuplesPerPage(Size rtupSize)
{
	return (BLCKSZ - SizeOfPageHeaderData - MAXALIGN(sizeof(HnswPageOpaqueData))) /
		(rtupSize + sizeof(ItemIdData));
}

void
HnswTqRerankWriterInit(HnswTqRerankWriter * w, HnswTqState * tq, Relation index,
					   ForkNumber forkNum, uint64 count)
{
	int			rtupSize = HnswTqRerankTupleSize(tq);
	uint64		pages;
	Buffer		prev = InvalidBuffer;

	w->index = index;
	w->forkNum = forkNum;
	w->buf = InvalidBuffer;
	w->page = NULL;
	w->head = InvalidBlockNumber;
	w->end = InvalidBlockNumber;

	if (rtupSize == 0 || count == 0)
		return;

	pages = (count + RerankTuplesPerPage(rtupSize) - 1) / RerankTuplesPerPage(rtupSize);

	for (uint64 i = 0; i < pages; i++)
	{
		Buffer		buf = HnswNewBuffer(index, forkNum);
		Page		page = BufferGetPage(buf);

		HnswInitPage(buf, page);
		HnswPageGetOpaque(page)->unused = HNSW_PAGE_RERANK;

		if (i == 0)
			w->head = BufferGetBlockNumber(buf);
		else
		{
			Page		prevPage = BufferGetPage(prev);

			HnswPageGetOpaque(prevPage)->nextblkno = BufferGetBlockNumber(buf);
			MarkBufferDirty(prev);
			UnlockReleaseBuffer(prev);
		}
		prev = buf;

		if ((i & 0xFFFF) == 0)
		{
			LockBuffer(buf, BUFFER_LOCK_UNLOCK);
			CHECK_FOR_INTERRUPTS();
			LockBuffer(buf, BUFFER_LOCK_EXCLUSIVE);
		}
	}

	w->end = BufferGetBlockNumber(prev) + 1;
	MarkBufferDirty(prev);
	UnlockReleaseBuffer(prev);

	if (w->end - w->head != pages)
		elog(ERROR, "re-rank pages of \"%s\" are not consecutive", RelationGetRelationName(index));
}

/*
 * Add a re-rank tuple (rtupSize as from HnswTqRerankTupleSize) and return
 * where it went
 */
void
HnswTqRerankWriterAdd(HnswTqRerankWriter * w, HnswRerankTuple rtup, Size rtupSize, ItemPointer tid)
{
	OffsetNumber offno;

	if (!BufferIsValid(w->buf) || PageGetFreeSpace(w->page) < rtupSize)
	{
		BlockNumber next = BufferIsValid(w->buf) ? BufferGetBlockNumber(w->buf) + 1 : w->head;

		if (BufferIsValid(w->buf))
		{
			MarkBufferDirty(w->buf);
			UnlockReleaseBuffer(w->buf);
		}
		if (!BlockNumberIsValid(next) || next >= w->end)
			elog(ERROR, "out of reserved re-rank pages in \"%s\"", RelationGetRelationName(w->index));

		w->buf = ReadBufferExtended(w->index, w->forkNum, next, RBM_NORMAL, NULL);
		LockBuffer(w->buf, BUFFER_LOCK_EXCLUSIVE);
		w->page = BufferGetPage(w->buf);
	}

	offno = PageAddItem(w->page, (Item) rtup, rtupSize, InvalidOffsetNumber, false, false);
	if (offno == InvalidOffsetNumber)
		elog(ERROR, "failed to add index item to \"%s\"", RelationGetRelationName(w->index));
	ItemPointerSet(tid, BufferGetBlockNumber(w->buf), offno);
}

/*
 * Release the writer; returns the last re-rank page written to
 */
BlockNumber
HnswTqRerankWriterFinish(HnswTqRerankWriter * w)
{
	BlockNumber last = InvalidBlockNumber;

	if (BufferIsValid(w->buf))
	{
		last = BufferGetBlockNumber(w->buf);
		MarkBufferDirty(w->buf);
		UnlockReleaseBuffer(w->buf);
		w->buf = InvalidBuffer;
	}
	return last;
}

/*
 * Re-rank: read an element's re-rank tid, then its re-rank bits
 */

/* Re-rank TID of an element, or false if it cannot be read optimistically */
static bool
ReadRerankTidOptimistic(HnswSupport * support, BlockNumber blkno, OffsetNumber offno,
						Size valueSize, ItemPointerData *out)
{
	uint64		version = 0;
	uint32		frameId = 0;
	void	   *entry = NULL;
	HnswElementTuple etup;
	const HnswTqElementValue *ev;

	if (!calico_optimistic_lookup_handle_frame(&support->read_handle, blkno, &version, &frameId, &entry))
		return false;

	etup = HnswOptimisticItem(CalicoFramePage(frameId), offno,
							  offsetof(HnswElementTupleData, data) + valueSize);
	if (etup == NULL)
		return false;

	ev = (const HnswTqElementValue *) &etup->data;
	if (etup->type != HNSW_ELEMENT_TUPLE_TYPE || ev->marker != HNSW_TQ_ELEMENT_MARKER)
		return false;
	*out = ev->reranktid;

	return calico_optimistic_validate(entry, version);
}

/*
 * The 4+k bit distance of an element, given its element value; false if its
 * re-rank tuple cannot be read optimistically
 */
static bool
RerankDistanceOptimistic(HnswSupport * support, const HnswTqQueryValue * qv, const HnswTqState * tq,
						 const HnswTqElementValue * ev, ItemPointer rtid, int rtupSize, double *distance)
{
	uint64		version = 0;
	uint32		frameId = 0;
	void	   *entry = NULL;
	HnswRerankTuple rtup;
	HnswTqPointInfo info;

	if (!calico_optimistic_lookup_handle_frame(&support->read_handle, ItemPointerGetBlockNumber(rtid),
											   &version, &frameId, &entry))
		return false;

	rtup = HnswOptimisticItem(CalicoFramePage(frameId), ItemPointerGetOffsetNumber(rtid), rtupSize);
	if (rtup == NULL || rtup->type != HNSW_RERANK_TUPLE_TYPE)
		return false;

	info.sqNorm = ev->sqNorm;
	info.scale4 = ev->scale4;
	info.scale4k = rtup->scale4k;
	*distance = HnswTqDistance4k(qv->query, ev->codes, rtup->extra, tq->extraBits, &info, (HnswTqMetric) qv->metric);

	return calico_optimistic_validate(entry, version);
}

/*
 * The 4+k bit distance of the element at (blkno, offno), with locks. Leaves
 * *distance alone if the element has no re-rank tuple.
 */
static void
RerankDistanceLocked(Relation index, const HnswTqQueryValue * qv, const HnswTqState * tq,
					 BlockNumber blkno, OffsetNumber offno, double *distance)
{
	Buffer		buf = ReadBuffer(index, blkno);
	Page		page;
	HnswElementTuple etup;
	HnswTqElementValue *ev;

	LockBuffer(buf, BUFFER_LOCK_SHARE);
	page = BufferGetPage(buf);
	etup = (HnswElementTuple) PageGetItem(page, PageGetItemId(page, offno));
	ev = (HnswTqElementValue *) &etup->data;

	if (HnswIsElementTuple(etup) && ev->marker == HNSW_TQ_ELEMENT_MARKER &&
		ItemPointerIsValid(&ev->reranktid))
	{
		/* The element page stays locked while the re-rank page is read */
		Buffer		rbuf = ReadBuffer(index, ItemPointerGetBlockNumber(&ev->reranktid));
		Page		rpage;
		HnswRerankTuple rtup;

		LockBuffer(rbuf, BUFFER_LOCK_SHARE);
		rpage = BufferGetPage(rbuf);
		rtup = (HnswRerankTuple) PageGetItem(rpage, PageGetItemId(rpage, ItemPointerGetOffsetNumber(&ev->reranktid)));
		if (rtup->type == HNSW_RERANK_TUPLE_TYPE)
		{
			HnswTqPointInfo info;

			info.sqNorm = ev->sqNorm;
			info.scale4 = ev->scale4;
			info.scale4k = rtup->scale4k;
			*distance = HnswTqDistance4k(qv->query, ev->codes, rtup->extra, tq->extraBits, &info, (HnswTqMetric) qv->metric);
		}
		UnlockReleaseBuffer(rbuf);
	}

	UnlockReleaseBuffer(buf);
}

/* Re-score one candidate: optimistically if possible, else with locks */
static void
RerankOne(Relation index, HnswSupport * support, const HnswTqQueryValue * qv,
		  BlockNumber blkno, OffsetNumber offno, const ItemPointerData *rtid, bool optimistic,
		  double *distance)
{
	const HnswTqState *tq = support->tq;
	int			rtupSize = HnswTqRerankTupleSize(tq);

	if (optimistic && ItemPointerIsValid(rtid))
	{
		Size		valueSize = HNSW_TQ_ELEMENT_VALUE_SIZE(tq->dim);
		uint64		version = 0;
		uint32		frameId = 0;
		void	   *entry = NULL;

		if (calico_optimistic_lookup_handle_frame(&support->read_handle, blkno, &version, &frameId, &entry))
		{
			HnswElementTuple etup = HnswOptimisticItem(CalicoFramePage(frameId), offno,
													   offsetof(HnswElementTupleData, data) + valueSize);

			if (etup != NULL)
			{
				const HnswTqElementValue *ev = (const HnswTqElementValue *) &etup->data;
				double		d;

				if (etup->type == HNSW_ELEMENT_TUPLE_TYPE && ev->marker == HNSW_TQ_ELEMENT_MARKER &&
					ItemPointerEquals(&((HnswTqElementValue *) ev)->reranktid, (ItemPointer) rtid) &&
					RerankDistanceOptimistic(support, qv, tq, ev, (ItemPointer) rtid, rtupSize, &d) &&
					calico_optimistic_validate(entry, version))
				{
					*distance = d;
					return;
				}
			}
		}
	}

	RerankDistanceLocked(index, qv, tq, blkno, offno, distance);
}

static bool
OptimisticReadsActive(Relation index)
{
	return calico_enable_optimistic_reads && calico_is_active() && !RelationUsesLocalBuffers(index);
}

static int
CompareFastCands(const void *a, const void *b)
{
	double		da = ((const HnswFastResultCand *) a)->distance;
	double		db = ((const HnswFastResultCand *) b)->distance;

	if (da < db)
		return -1;
	if (da > db)
		return 1;
	return 0;
}

/*
 * Re-score the candidates of a search with the 4+k bit codes, and sort them
 * nearest first again
 */
void
HnswTqRerankCands(Relation index, HnswSupport * support, HnswQuery * q, HnswFastResultCand *cands, int count)
{
	const HnswTqQueryValue *qv = (const HnswTqQueryValue *) DatumGetPointer(q->value);
	bool		optimistic = OptimisticReadsActive(index);
	Size		valueSize = HNSW_TQ_ELEMENT_VALUE_SIZE(support->tq->dim);
	ItemPointerData *rtids;

	if (support->tq->extraBits == 0 || count <= 0 || qv == NULL)
		return;

	/* First the re-rank TIDs, prefetching their pages; the elements are hot */
	rtids = palloc_array_checked(ItemPointerData, count);
	for (int i = 0; i < count; i++)
	{
		if (!optimistic ||
			!ReadRerankTidOptimistic(support, cands[i].blkno, cands[i].offno, valueSize, &rtids[i]))
		{
			ItemPointerSetInvalid(&rtids[i]);
			continue;
		}
		if (ItemPointerIsValid(&rtids[i]))
			calico_optimistic_prefetch_block(&support->read_handle, ItemPointerGetBlockNumber(&rtids[i]));
	}

	for (int i = 0; i < count; i++)
		RerankOne(index, support, qv, cands[i].blkno, cands[i].offno, &rtids[i], optimistic, &cands[i].distance);

	pfree(rtids);
	qsort(cands, count, sizeof(HnswFastResultCand), CompareFastCands);
}

static int
CompareSearchCandidatesFarthestFirst(const ListCell *a, const ListCell *b)
{
	double		da = ((HnswSearchCandidate *) lfirst(a))->distance;
	double		db = ((HnswSearchCandidate *) lfirst(b))->distance;

	if (da > db)
		return -1;
	if (da < db)
		return 1;
	return 0;
}

/*
 * The same for the list a search returns (farthest first; the scan takes
 * from the end)
 */
List *
HnswTqRerankList(Relation index, HnswSupport * support, HnswQuery * q, List *w)
{
	const HnswTqQueryValue *qv = (const HnswTqQueryValue *) DatumGetPointer(q->value);
	ListCell   *lc;

	if (support->tq->extraBits == 0 || qv == NULL)
		return w;

	foreach(lc, w)
	{
		HnswSearchCandidate *sc = lfirst(lc);
		HnswElement element = HnswPtrAccess((char *) NULL, sc->element);

		RerankDistanceLocked(index, qv, support->tq, element->blkno, element->offno, &sc->distance);
	}

	list_sort(w, CompareSearchCandidatesFarthestFirst);
	return w;
}
