/*
 * Experimental TurboQuant navigation codes for HNSW
 *
 * Based on Nested TurboQuant from "GINGER: Spicing Up Graph-Based Vector
 * Search" (arXiv:2611.08306). Each vector is normalized and randomly rotated
 * (y = sqrt(d) R x / ||x||), then every coordinate is quantized with a 4-bit
 * Gaussian Lloyd-Max codebook (sign + 3-bit index). The 4-bit code is used
 * for graph navigation. For re-ranking, each 4-bit cell is refined with k
 * more bits (a Lloyd-Max codebook within the cell), or the exact vector is
 * kept.
 *
 * Inner products are estimated as <q, x> ~= s_b(x) <R q, y~> with
 * s_b(x) = ||x|| / (sqrt(1 - D_b) ||y~||).
 */
#include "postgres.h"

#include <math.h>

#include "access/generic_xlog.h"
#include "hnsw.h"
#include "hnswtq_tables.h"
#include "hnswtq_simd.h"
#include "storage/bufmgr.h"
#include "storage/lmgr.h"
#include "utils/lsyscache.h"
#include "utils/memutils.h"
#include "utils/rel.h"
#include "vector.h"

#if PG_VERSION_NUM >= 160000
#include "varatt.h"
#endif

/* Signed 4-bit reconstruction values indexed by nibble */
static float tq4_lut[16];

/* Cached per-relation state (in rd_amcache) */
typedef struct HnswTqCache
{
	HnswTqState *tq;
}			HnswTqCache;

/*
 * Get the quantizer for the index
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
 * Get the re-rank mode for the index
 */
int
HnswGetRerank(Relation index)
{
	HnswOptions *opts = (HnswOptions *) index->rd_options;

	if (opts)
		return opts->rerank;

	return HNSW_RERANK_NTQ12;
}

/*
 * Initialize lookup tables
 */
void
HnswTqInitTables(void)
{
	for (int n = 0; n < 16; n++)
		tq4_lut[n] = (n & 8 ? -1.0f : 1.0f) * hnsw_tq4_centroids[n & 7];
}

/*
 * Deterministic PRNG (splitmix64) so all backends derive the same rotation
 */
static uint64
TqNext(uint64 *state)
{
	uint64		z = (*state += UINT64CONST(0x9E3779B97F4A7C15));

	z = (z ^ (z >> 30)) * UINT64CONST(0xBF58476D1CE4E5B9);
	z = (z ^ (z >> 27)) * UINT64CONST(0x94D049BB133111EB);
	return z ^ (z >> 31);
}

static double
TqUniform(uint64 *state)
{
	/* In (0, 1) */
	return ((double) (TqNext(state) >> 11) + 0.5) * (1.0 / 9007199254740992.0);
}

static double
TqGaussian(uint64 *state)
{
	double		u1 = TqUniform(state);
	double		u2 = TqUniform(state);

	return sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
}

/*
 * Choose padded dimensions r * 2^l >= dim with odd r <= HNSW_TQ_MAX_R
 */
static void
TqChoosePadding(int dim, int *r, int *blockSize)
{
	int			best = INT_MAX;

	for (int rr = 1; rr <= HNSW_TQ_MAX_R; rr += 2)
	{
		int			b = 1;

		while (rr * b < dim)
			b *= 2;

		if (rr * b < best)
		{
			best = rr * b;
			*r = rr;
			*blockSize = b;
		}
	}
}

/*
 * Generate random signs
 */
static float *
TqSigns(uint64 seed, int pdim)
{
	float	   *signs = palloc_array(float, pdim);
	uint64		state = seed;

	for (int i = 0; i < pdim; i++)
		signs[i] = (TqNext(&state) & 1) ? -1.0f : 1.0f;

	return signs;
}

/*
 * Generate a Haar-random r x r orthogonal matrix (Gram-Schmidt of a Gaussian matrix)
 */
static float *
TqOrthogonal(uint64 seed, int r)
{
	double	   *a = palloc_array(double, r * r);
	float	   *q = palloc_array(float, r * r);
	uint64		state = seed ^ UINT64CONST(0xD1B54A32D192ED03);

	for (int i = 0; i < r * r; i++)
		a[i] = TqGaussian(&state);

	/* Orthonormalize columns (a[i * r + j] is row i, column j) */
	for (int j = 0; j < r; j++)
	{
		double		norm = 0;

		/* Two passes for numerical stability */
		for (int pass = 0; pass < 2; pass++)
		{
			for (int p = 0; p < j; p++)
			{
				double		dot = 0;

				for (int i = 0; i < r; i++)
					dot += a[i * r + p] * a[i * r + j];
				for (int i = 0; i < r; i++)
					a[i * r + j] -= dot * a[i * r + p];
			}
		}

		for (int i = 0; i < r; i++)
			norm += a[i * r + j] * a[i * r + j];
		norm = sqrt(norm);

		for (int i = 0; i < r; i++)
			a[i * r + j] /= norm;
	}

	for (int i = 0; i < r * r; i++)
		q[i] = (float) a[i];

	pfree(a);
	return q;
}

/*
 * Get the distance kind for the opclass
 */
static int
TqDistanceKind(const HnswSupport * support)
{
	char	   *name = get_func_name(support->procinfo->fn_oid);

	if (name != NULL && strcmp(name, "vector_l2_squared_distance") == 0)
		return HNSW_TQ_DIST_L2;

	if (name != NULL && strcmp(name, "vector_negative_inner_product") == 0)
		return HNSW_TQ_DIST_NEG_IP;

	ereport(ERROR,
			(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
			 errmsg("quantizer is only supported with vector_l2_ops, vector_ip_ops, and vector_cosine_ops")));
	return 0;					/* keep compiler quiet */
}

static int
TqRefineBits(int rerank)
{
	if (rerank == HNSW_RERANK_NTQ10)
		return 6;
	if (rerank == HNSW_RERANK_NTQ12)
		return 8;
	return 0;
}

/*
 * Create state for a new index from its options
 */
HnswTqState *
HnswTqCreateState(Relation index, const HnswSupport * support, int dimensions)
{
	HnswTqState *tq;

	/* Only the vector type for now (other types have a type info proc) */
	if (HnswOptionalProcInfo(index, HNSW_TYPE_INFO_PROC) != NULL)
		ereport(ERROR,
				(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
				 errmsg("quantizer is only supported for the vector type")));

	tq = palloc0_object(HnswTqState);
	tq->distance = TqDistanceKind(support);
	tq->rerank = HnswGetRerank(index);
	tq->refineBits = TqRefineBits(tq->rerank);
	tq->dim = dimensions;
	TqChoosePadding(dimensions, &tq->r, &tq->blockSize);
	tq->pdim = tq->r * tq->blockSize;
	tq->seed = HNSW_TQ_SEED;
	tq->signs = TqSigns(tq->seed, tq->pdim);
	tq->Q = TqOrthogonal(tq->seed, tq->r);

	return tq;
}

/*
 * Write state to the metapage
 */
void
HnswTqSetMetaPage(HnswMetaPage metap, const HnswTqState * tq, Page page)
{
	metap->version = HNSW_TQ_VERSION;
	metap->quantizer = HNSW_QUANTIZER_TQ4;
	metap->rerank = tq->rerank;
	metap->tqDistance = tq->distance;
	metap->tqR = tq->r;
	metap->tqPdim = tq->pdim;
	metap->tqUnused = 0;
	metap->tqSeedLo = (uint32) tq->seed;
	metap->tqSeedHi = (uint32) (tq->seed >> 32);
	metap->rerankHeadBlkno = InvalidBlockNumber;
	metap->rerankInsertPage = InvalidBlockNumber;
	memcpy(metap->tqQ, tq->Q, sizeof(float4) * tq->r * tq->r);

	((PageHeader) page)->pd_lower =
		((char *) &metap->tqQ[tq->r * tq->r]) - (char *) page;
}

/*
 * Load state from the metapage (cached in rd_amcache)
 */
HnswTqState *
HnswTqLoadState(Relation index)
{
	Buffer		buf;
	Page		page;
	HnswMetaPage metap;
	HnswTqCache *cache;
	MemoryContext oldCtx;

	if (index->rd_amcache != NULL)
		return ((HnswTqCache *) index->rd_amcache)->tq;

	/* Index is being built */
	if (RelationGetNumberOfBlocks(index) == 0)
		return NULL;

	buf = ReadBuffer(index, HNSW_METAPAGE_BLKNO);
	LockBuffer(buf, BUFFER_LOCK_SHARE);
	page = BufferGetPage(buf);
	metap = HnswPageGetMeta(page);

	if (metap->magicNumber != HNSW_MAGIC_NUMBER)
	{
		UnlockReleaseBuffer(buf);
		return NULL;
	}

	oldCtx = MemoryContextSwitchTo(index->rd_indexcxt);

	cache = palloc0_object(HnswTqCache);

	if (metap->version >= HNSW_TQ_VERSION && metap->quantizer == HNSW_QUANTIZER_TQ4)
	{
		HnswTqState *tq = palloc0_object(HnswTqState);

		tq->rerank = metap->rerank;
		tq->distance = metap->tqDistance;
		tq->refineBits = TqRefineBits(tq->rerank);
		tq->dim = metap->dimensions;
		tq->r = metap->tqR;
		tq->pdim = metap->tqPdim;
		tq->blockSize = tq->pdim / tq->r;
		tq->seed = ((uint64) metap->tqSeedHi << 32) | metap->tqSeedLo;
		tq->signs = TqSigns(tq->seed, tq->pdim);
		tq->Q = palloc_array(float, tq->r * tq->r);
		memcpy(tq->Q, metap->tqQ, sizeof(float4) * tq->r * tq->r);

		cache->tq = tq;
	}

	MemoryContextSwitchTo(oldCtx);
	UnlockReleaseBuffer(buf);

	index->rd_amcache = cache;
	return cache->tq;
}

/*
 * Apply the rotation R (signs, blockwise Walsh-Hadamard, r x r mixing)
 */
static void
TqRotate(const HnswTqState * tq, const float *x, float *out)
{
	int			pdim = tq->pdim;
	int			r = tq->r;
	int			bs = tq->blockSize;
	float	   *tmp = palloc_array(float, pdim);
	float		hscale = (float) (1.0 / sqrt((double) bs));

	for (int i = 0; i < tq->dim; i++)
		tmp[i] = x[i] * tq->signs[i];
	for (int i = tq->dim; i < pdim; i++)
		tmp[i] = 0;

	/* Normalized Walsh-Hadamard on each block */
	for (int b = 0; b < r; b++)
	{
		float	   *blk = tmp + b * bs;

		for (int len = 1; len < bs; len <<= 1)
		{
			for (int i = 0; i < bs; i += len << 1)
			{
				for (int j = i; j < i + len; j++)
				{
					float		u = blk[j];
					float		v = blk[j + len];

					blk[j] = u + v;
					blk[j + len] = u - v;
				}
			}
		}

		for (int i = 0; i < bs; i++)
			blk[i] *= hscale;
	}

	/* Mix blocks: row i of output = sum_j Q[i][j] * block j */
	for (int i = 0; i < r; i++)
	{
		float	   *orow = out + i * bs;

		for (int t = 0; t < bs; t++)
			orow[t] = 0;

		for (int j = 0; j < r; j++)
		{
			float		qij = tq->Q[i * r + j];
			float	   *blk = tmp + j * bs;

			for (int t = 0; t < bs; t++)
				orow[t] += qij * blk[t];
		}
	}

	pfree(tmp);
}

static float
TqNorm(const Vector * vec)
{
	double		norm = 0;

	for (int i = 0; i < vec->dim; i++)
		norm += (double) vec->x[i] * (double) vec->x[i];

	return (float) sqrt(norm);
}

/*
 * Make an exact in-memory value from a vector
 */
Datum
HnswTqMakeValue(const HnswTqState * tq, Datum value, bool keepOrig)
{
	Vector	   *vec = (Vector *) DatumGetPointer(value);
	Size		size = HNSW_TQ_VALUE_SIZE(tq->pdim);
	HnswTqValue *v;

	if (HnswTqMarker(vec) < 0)
		elog(ERROR, "expected vector value");

	if (keepOrig)
		size += VARSIZE_ANY(vec);

	v = (HnswTqValue *) palloc0(size);
	SET_VARSIZE(v, size);
	v->marker = HNSW_TQ_VALUE_MARKER;
	v->pdim = tq->pdim;
	v->exact = 1;
	v->hasOrig = keepOrig;
	v->unused = tq->distance;	/* in-memory only: lets lock-free readers score codes without the state */
	v->norm = TqNorm(vec);
	TqRotate(tq, vec->x, v->r);

	if (keepOrig)
		memcpy(HnswTqValueOrig(v), vec, VARSIZE_ANY(vec));

	return PointerGetDatum(v);
}

static inline int
TqNibble(const uint8 *code, int i)
{
	return (code[i >> 1] >> ((i & 1) << 2)) & 0x0F;
}

/*
 * Decode a stored code into an in-memory value (estimate of R x)
 */
Pointer
HnswTqDecodeCode(const HnswTqCode * code)
{
	int			pdim = code->pdim;
	Size		size = HNSW_TQ_VALUE_SIZE(pdim);
	HnswTqValue *v = (HnswTqValue *) palloc0(size);

	SET_VARSIZE(v, size);
	v->marker = HNSW_TQ_VALUE_MARKER;
	v->pdim = pdim;
	v->exact = 0;
	v->hasOrig = 0;
	v->norm = code->norm;

	for (int i = 0; i < pdim; i++)
		v->r[i] = code->scale * tq4_lut[TqNibble(code->code, i)];

	return (Pointer) v;
}

static inline double
TqCombine(int distance, float qnorm, float xnorm, double ip)
{
	if (distance == HNSW_TQ_DIST_L2)
		return (double) qnorm * qnorm + (double) xnorm * xnorm - 2.0 * ip;

	return -ip;
}

/*
 * Distance between two in-memory values
 */
double
HnswTqValueDistance(const HnswSupport * support, Datum a, Datum b)
{
	Pointer		pa = DatumGetPointer(a);
	Pointer		pb = DatumGetPointer(b);
	int16		ma = HnswTqMarker(pa);
	int16		mb = HnswTqMarker(pb);

	/* Exact vectors (in-memory build) */
	if (ma >= 0 && mb >= 0)
		return DatumGetFloat8(FunctionCall2Coll(support->procinfo, support->collation, a, b));

	if (ma == HNSW_TQ_VALUE_MARKER && mb == HNSW_TQ_VALUE_MARKER)
	{
		HnswTqValue *va = (HnswTqValue *) pa;
		HnswTqValue *vb = (HnswTqValue *) pb;
		double		dot = 0;

		for (int i = 0; i < va->pdim; i++)
			dot += va->r[i] * vb->r[i];

		return TqCombine(support->tq->distance, va->norm, vb->norm, dot);
	}

	elog(ERROR, "unexpected hnsw value types for distance (%d, %d)", ma, mb);
	return 0;					/* keep compiler quiet */
}

/*
 * Inner product between a rotated query and a 4-bit code
 */
static float
TqNavDot(const float *qr, const uint8 *code, int pdim)
{
	/* AVX2 register lookup, not gathers (hnswtq_simd.h) */
	return TqNavDotFast(qr, code, pdim, hnsw_tq4_centroids, tq4_lut);
}

/*
 * Distance between an in-memory value and a stored 4-bit code
 */
double
HnswTqCodeDistance(const HnswTqState * tq, Datum q, const HnswTqCode * code)
{
	HnswTqValue *qv = (HnswTqValue *) DatumGetPointer(q);

	if (HnswTqMarker(qv) != HNSW_TQ_VALUE_MARKER || code->marker != HNSW_TQ_CODE_MARKER)
		elog(ERROR, "unexpected hnsw value types for code distance");

	return TqCombine(tq->distance, qv->norm, code->norm, code->scale * TqNavDot(qv->r, code->code, code->pdim));
}

/*
 * Distance between a query value and a code read optimistically (pgvectorfast
 * lock-free path). The caller has checked the markers, pdim and length; the
 * bytes may change underneath, and the caller validates the page version
 * afterwards. The distance kind travels in the query value.
 */
double
HnswTqCodeDistanceOptimistic(const HnswTqValue * qv, const HnswTqCode * code)
{
	return TqCombine(qv->unused, qv->norm, code->norm, code->scale * TqNavDot(qv->r, code->code, qv->pdim));
}

static inline int
TqNearest(const float *c, int n, float a)
{
	int			lo = 0;
	int			hi = n - 1;

	while (lo < hi)
	{
		int			mid = (lo + hi) / 2;

		if (c[mid] < a)
			lo = mid + 1;
		else
			hi = mid;
	}

	if (lo > 0 && (a - c[lo - 1]) <= (c[lo] - a))
		lo--;

	return lo;
}

static inline void
TqPutBits(uint8 *buf, int i, int k, uint32 v)
{
	uint32		off = (uint32) i * k;

	for (int b = 0; b < k; b++)
	{
		if ((v >> b) & 1)
			buf[(off + b) >> 3] |= (uint8) (1 << ((off + b) & 7));
	}
}

static inline uint32
TqGetBits(const uint8 *buf, int i, int k)
{
	uint32		off;
	uint32		w;

	if (k == 8)
		return buf[i];

	off = (uint32) i * k;
	w = (uint32) buf[off >> 3] | ((uint32) buf[(off >> 3) + 1] << 8);
	return (w >> (off & 7)) & ((1U << k) - 1);
}

static inline const float *
TqRefineCells(int k, int u)
{
	return k == 6 ? hnsw_tq_ref6[u] : hnsw_tq_ref8[u];
}

static Size
TqRerankSize(const HnswTqState * tq, Pointer orig)
{
	Size		size = offsetof(HnswRerankTupleData, data);

	if (tq->rerank == HNSW_RERANK_EXACT)
		size += VARSIZE_ANY(orig);
	else
		size += ((Size) tq->pdim * tq->refineBits + 7) / 8 + 1;	/* +1 for 16-bit reads */

	return MAXALIGN(size);
}

/*
 * Encode an exact value (vector or exact in-memory value) into a 4-bit code
 * and (optionally) a re-rank tuple
 */
HnswTqCode *
HnswTqEncode(const HnswTqState * tq, Datum value, HnswRerankTuple * rtupOut, Size *rtupSize)
{
	Pointer		p = DatumGetPointer(value);
	int16		marker = HnswTqMarker(p);
	int			pdim = tq->pdim;
	int			k = tq->refineBits;
	float	   *rx;
	bool		freeRx = false;
	float		norm;
	Pointer		orig = NULL;
	Size		codeSize = HNSW_TQ_CODE_SIZE(pdim);
	HnswTqCode *code;
	HnswRerankTuple rtup = NULL;
	double		scaleY;
	double		sum4 = 0;
	double		sumk = 0;

	if (marker >= 0)
	{
		Vector	   *vec = (Vector *) p;

		rx = palloc_array(float, pdim);
		freeRx = true;
		TqRotate(tq, vec->x, rx);
		norm = TqNorm(vec);
		orig = p;
	}
	else if (marker == HNSW_TQ_VALUE_MARKER && ((HnswTqValue *) p)->exact)
	{
		HnswTqValue *v = (HnswTqValue *) p;

		rx = v->r;
		norm = v->norm;
		if (v->hasOrig)
			orig = HnswTqValueOrig(v);
	}
	else
	{
		elog(ERROR, "cannot encode an approximate hnsw value");
		return NULL;			/* keep compiler quiet */
	}

	code = (HnswTqCode *) palloc0(codeSize);
	SET_VARSIZE(code, codeSize);
	code->marker = HNSW_TQ_CODE_MARKER;
	code->pdim = pdim;
	code->norm = norm;
	ItemPointerSetInvalid(&code->reranktid);

	if (tq->rerank != HNSW_RERANK_NONE)
	{
		if (tq->rerank == HNSW_RERANK_EXACT && orig == NULL)
			elog(ERROR, "exact re-rank requires the original vector");

		*rtupSize = TqRerankSize(tq, orig);
		rtup = (HnswRerankTuple) palloc0(*rtupSize);
		rtup->type = HNSW_RERANK_TUPLE_TYPE;
		rtup->deleted = 0;
	}
	else
		*rtupSize = 0;

	scaleY = norm > 0 ? sqrt((double) pdim) / norm : 0;

	for (int i = 0; i < pdim; i++)
	{
		float		y = (float) (rx[i] * scaleY);
		int			neg = y < 0;
		float		a = fabsf(y);
		int			u = 0;
		float		c;

		while (u < 7 && a >= hnsw_tq4_thresholds[u])
			u++;

		c = hnsw_tq4_centroids[u];
		sum4 += (double) c * c;
		code->code[i >> 1] |= (uint8) (((neg << 3) | u) << ((i & 1) << 2));

		if (k > 0)
		{
			const float *cells = TqRefineCells(k, u);
			int			v = TqNearest(cells, 1 << k, a);

			sumk += (double) cells[v] * cells[v];
			TqPutBits((uint8 *) rtup->data, i, k, v);
		}
	}

	code->scale = (norm > 0 && sum4 > 0) ? (float) (norm / (sqrt(1.0 - HNSW_TQ_D4) * sqrt(sum4))) : 0;

	if (k > 0)
	{
		double		dk = k == 6 ? HNSW_TQ_D10 : HNSW_TQ_D12;

		rtup->scale = (norm > 0 && sumk > 0) ? (float) (norm / (sqrt(1.0 - dk) * sqrt(sumk))) : 0;
	}
	else if (tq->rerank == HNSW_RERANK_EXACT)
		memcpy(rtup->data, orig, VARSIZE_ANY(orig));

	if (freeRx)
		pfree(rx);

	*rtupOut = rtup;
	return code;
}

/*
 * Inner product between a rotated query and a nested (4+k)-bit code
 */
static float
TqRefineDot(const float *qr, const uint8 *code, const uint8 *ref, int pdim, int k)
{
	/* k-specialized loop (hnswtq_simd.h); k is 6 or 8 */
	return TqRefineDotFast(qr, code, ref, pdim, k, k == 6 ? &hnsw_tq_ref6[0][0] : &hnsw_tq_ref8[0][0]);
}

/*
 * Update the re-rank chain info on the metapage
 */
void
HnswTqUpdateRerankMeta(Relation index, BlockNumber headBlkno, BlockNumber insertPage, ForkNumber forkNum, bool building)
{
	Buffer		buf;
	Page		page;
	GenericXLogState *state = NULL;
	HnswMetaPage metap;

	buf = ReadBufferExtended(index, forkNum, HNSW_METAPAGE_BLKNO, RBM_NORMAL, NULL);
	LockBuffer(buf, BUFFER_LOCK_EXCLUSIVE);
	if (building)
		page = BufferGetPage(buf);
	else
	{
		state = GenericXLogStart(index);
		page = GenericXLogRegisterBuffer(state, buf, 0);
	}

	metap = HnswPageGetMeta(page);
	if (BlockNumberIsValid(headBlkno))
		metap->rerankHeadBlkno = headBlkno;
	if (BlockNumberIsValid(insertPage))
		metap->rerankInsertPage = insertPage;

	if (building)
		MarkBufferDirty(buf);
	else
		GenericXLogFinish(state);
	UnlockReleaseBuffer(buf);
}

/*
 * Move the re-rank insert page back to a page with deleted slots
 *
 * Re-rank pages are always appended to the end of the relation and the
 * chain, so the chain is in block order and an earlier block is always
 * reachable from the insert page.
 */
void
HnswTqLowerRerankInsertPage(Relation index, BlockNumber blkno)
{
	Buffer		buf;
	Page		page;
	GenericXLogState *state;
	HnswMetaPage metap;

	buf = ReadBuffer(index, HNSW_METAPAGE_BLKNO);
	LockBuffer(buf, BUFFER_LOCK_EXCLUSIVE);
	state = GenericXLogStart(index);
	page = GenericXLogRegisterBuffer(state, buf, 0);
	metap = HnswPageGetMeta(page);

	if (!BlockNumberIsValid(metap->rerankInsertPage) || blkno < metap->rerankInsertPage)
	{
		metap->rerankInsertPage = blkno;
		GenericXLogFinish(state);
	}
	else
		GenericXLogAbort(state);

	UnlockReleaseBuffer(buf);
}

static void
TqInitRerankPage(Buffer buf, Page page)
{
	HnswInitPage(buf, page);
	HnswPageGetOpaque(page)->unused = HNSW_PAGE_RERANK;
}

static BlockNumber
TqGetRerankInsertPage(Relation index)
{
	Buffer		buf;
	BlockNumber blkno;

	buf = ReadBuffer(index, HNSW_METAPAGE_BLKNO);
	LockBuffer(buf, BUFFER_LOCK_SHARE);
	blkno = HnswPageGetMeta(BufferGetPage(buf))->rerankInsertPage;
	UnlockReleaseBuffer(buf);

	return blkno;
}

/*
 * Add a re-rank tuple on disk (reusing a deleted slot if possible)
 *
 * Only re-rank pages are locked here (never graph pages), so this cannot
 * deadlock with graph updates, which lock graph pages before re-rank pages.
 */
void
HnswTqAddRerankTuple(Relation index, HnswRerankTuple rtup, Size rtupSize, ItemPointer tid, bool building)
{
	BlockNumber startPage;
	BlockNumber blkno;

	startPage = TqGetRerankInsertPage(index);

	/* Create the first re-rank page if needed */
	if (!BlockNumberIsValid(startPage))
	{
		LockRelationForExtension(index, ExclusiveLock);

		startPage = TqGetRerankInsertPage(index);
		if (!BlockNumberIsValid(startPage))
		{
			Buffer		buf = HnswNewBuffer(index, MAIN_FORKNUM);
			GenericXLogState *state = NULL;
			Page		page;
			OffsetNumber offno;

			if (building)
				page = BufferGetPage(buf);
			else
			{
				state = GenericXLogStart(index);
				page = GenericXLogRegisterBuffer(state, buf, GENERIC_XLOG_FULL_IMAGE);
			}

			TqInitRerankPage(buf, page);
			offno = PageAddItem(page, (Item) rtup, rtupSize, InvalidOffsetNumber, false, false);
			if (offno == InvalidOffsetNumber)
				elog(ERROR, "failed to add index item to \"%s\"", RelationGetRelationName(index));
			ItemPointerSet(tid, BufferGetBlockNumber(buf), offno);

			if (building)
				MarkBufferDirty(buf);
			else
				GenericXLogFinish(state);

			blkno = BufferGetBlockNumber(buf);
			UnlockReleaseBuffer(buf);

			HnswTqUpdateRerankMeta(index, blkno, blkno, MAIN_FORKNUM, building);
			UnlockRelationForExtension(index, ExclusiveLock);
			return;
		}

		UnlockRelationForExtension(index, ExclusiveLock);
	}

	blkno = startPage;

	for (;;)
	{
		Buffer		buf;
		Page		page;
		GenericXLogState *state = NULL;
		OffsetNumber offno = InvalidOffsetNumber;
		OffsetNumber maxoffno;
		BlockNumber nextblkno;

		buf = ReadBuffer(index, blkno);
		LockBuffer(buf, BUFFER_LOCK_EXCLUSIVE);
		if (building)
			page = BufferGetPage(buf);
		else
		{
			state = GenericXLogStart(index);
			page = GenericXLogRegisterBuffer(state, buf, 0);
		}

		/* Reuse a deleted slot (all re-rank tuples in an index have the same size) */
		maxoffno = PageGetMaxOffsetNumber(page);
		for (OffsetNumber o = FirstOffsetNumber; o <= maxoffno; o = OffsetNumberNext(o))
		{
			ItemId		itemid = PageGetItemId(page, o);
			HnswRerankTuple existing = (HnswRerankTuple) PageGetItem(page, itemid);

			if (HnswIsRerankTuple(existing) && existing->deleted && ItemIdGetLength(itemid) == rtupSize)
			{
				if (!PageIndexTupleOverwrite(page, o, (Item) rtup, rtupSize))
					elog(ERROR, "failed to add index item to \"%s\"", RelationGetRelationName(index));
				offno = o;
				break;
			}
		}

		/* Otherwise use free space */
		if (!OffsetNumberIsValid(offno) && PageGetFreeSpace(page) >= rtupSize)
		{
			offno = PageAddItem(page, (Item) rtup, rtupSize, InvalidOffsetNumber, false, false);
			if (offno == InvalidOffsetNumber)
				elog(ERROR, "failed to add index item to \"%s\"", RelationGetRelationName(index));
		}

		if (OffsetNumberIsValid(offno))
		{
			ItemPointerSet(tid, blkno, offno);

			if (building)
				MarkBufferDirty(buf);
			else
				GenericXLogFinish(state);
			UnlockReleaseBuffer(buf);
			break;
		}

		nextblkno = HnswPageGetOpaque(page)->nextblkno;

		if (BlockNumberIsValid(nextblkno))
		{
			if (!building)
				GenericXLogAbort(state);
			UnlockReleaseBuffer(buf);
			blkno = nextblkno;
			continue;
		}

		/* Append a new page to the chain */
		{
			Buffer		nbuf;
			Page		npage;

			LockRelationForExtension(index, ExclusiveLock);
			nbuf = HnswNewBuffer(index, MAIN_FORKNUM);
			UnlockRelationForExtension(index, ExclusiveLock);

			if (building)
				npage = BufferGetPage(nbuf);
			else
				npage = GenericXLogRegisterBuffer(state, nbuf, GENERIC_XLOG_FULL_IMAGE);

			TqInitRerankPage(nbuf, npage);
			HnswPageGetOpaque(page)->nextblkno = BufferGetBlockNumber(nbuf);

			offno = PageAddItem(npage, (Item) rtup, rtupSize, InvalidOffsetNumber, false, false);
			if (offno == InvalidOffsetNumber)
				elog(ERROR, "failed to add index item to \"%s\"", RelationGetRelationName(index));

			blkno = BufferGetBlockNumber(nbuf);
			ItemPointerSet(tid, blkno, offno);

			if (building)
			{
				MarkBufferDirty(buf);
				MarkBufferDirty(nbuf);
			}
			else
				GenericXLogFinish(state);
			UnlockReleaseBuffer(nbuf);
			UnlockReleaseBuffer(buf);
			break;
		}
	}

	/* Move the insert page forward */
	if (blkno != startPage)
		HnswTqUpdateRerankMeta(index, InvalidBlockNumber, blkno, MAIN_FORKNUM, building);
}

/*
 * Compare candidates by descending distance (scans return from the end)
 */
static int
CompareCandidatesDesc(const ListCell *a, const ListCell *b)
{
	HnswSearchCandidate *ca = lfirst(a);
	HnswSearchCandidate *cb = lfirst(b);

	if (ca->distance > cb->distance)
		return -1;
	if (ca->distance < cb->distance)
		return 1;
	return 0;
}

/*
 * Re-rank scan candidates with the re-rank payload
 */
List *
HnswTqRerank(Relation index, HnswSupport * support, HnswQuery * q, List *w)
{
	HnswTqState *tq = support->tq;
	HnswTqValue *qv = (HnswTqValue *) DatumGetPointer(q->value);
	char	   *base = NULL;
	ListCell   *lc;

	if (tq->rerank == HNSW_RERANK_NONE || w == NIL)
		return w;

	foreach(lc, w)
	{
		HnswSearchCandidate *sc = lfirst(lc);
		HnswElement element = HnswPtrAccess(base, sc->element);
		Buffer		buf;
		Page		page;
		HnswElementTuple etup;

		/* Will not be returned */
		if (element->heaptidsLength == 0)
			continue;

		buf = ReadBuffer(index, element->blkno);
		LockBuffer(buf, BUFFER_LOCK_SHARE);
		page = BufferGetPage(buf);
		etup = (HnswElementTuple) PageGetItem(page, PageGetItemId(page, element->offno));

		if (HnswIsElementTuple(etup) && !etup->deleted && etup->version == element->version &&
			HnswTqMarker(&etup->data) == HNSW_TQ_CODE_MARKER)
		{
			HnswTqCode *code = (HnswTqCode *) &etup->data;

			if (ItemPointerIsValid(&code->reranktid))
			{
				Buffer		rbuf;
				Page		rpage;
				HnswRerankTuple rtup;

				/* Graph page locked before re-rank page */
				rbuf = ReadBuffer(index, ItemPointerGetBlockNumber(&code->reranktid));
				LockBuffer(rbuf, BUFFER_LOCK_SHARE);
				rpage = BufferGetPage(rbuf);
				rtup = (HnswRerankTuple) PageGetItem(rpage, PageGetItemId(rpage, ItemPointerGetOffsetNumber(&code->reranktid)));

				if (HnswIsRerankTuple(rtup) && !rtup->deleted)
				{
					if (tq->rerank == HNSW_RERANK_EXACT)
						sc->distance = DatumGetFloat8(FunctionCall2Coll(support->procinfo, support->collation,
																		PointerGetDatum(HnswTqValueOrig(qv)),
																		PointerGetDatum(rtup->data)));
					else
					{
						float		dot = TqRefineDot(qv->r, code->code, (uint8 *) rtup->data, code->pdim, tq->refineBits);

						sc->distance = TqCombine(tq->distance, qv->norm, code->norm, rtup->scale * dot);
					}
				}

				UnlockReleaseBuffer(rbuf);
			}
		}

		UnlockReleaseBuffer(buf);
	}

	list_sort(w, CompareCandidatesDesc);
	return w;
}

static int
CompareFastCands(const void *a, const void *b)
{
	double		da = ((const HnswFastResultCand *) a)->distance;
	double		db = ((const HnswFastResultCand *) b)->distance;

	return (da > db) - (da < db);
}

/*
 * Re-rank pgvectorfast's result candidates (nearest first) with the re-rank
 * payload, then restore nearest-first order. Same arithmetic as HnswTqRerank;
 * reads go through the regular (Calico-intercepted) buffer path.
 */
void
HnswTqRerankFast(Relation index, HnswSupport * support, HnswQuery * q, HnswFastResultCand *cands, int count)
{
	HnswTqState *tq = support->tq;
	HnswTqValue *qv = (HnswTqValue *) DatumGetPointer(q->value);

	if (tq->rerank == HNSW_RERANK_NONE || count <= 0)
		return;

	for (int i = 0; i < count; i++)
	{
		HnswFastResultCand *c = &cands[i];
		Buffer		buf;
		Page		page;
		HnswElementTuple etup;

		buf = ReadBuffer(index, c->blkno);
		LockBuffer(buf, BUFFER_LOCK_SHARE);
		page = BufferGetPage(buf);
		etup = (HnswElementTuple) PageGetItem(page, PageGetItemId(page, c->offno));

		if (HnswIsElementTuple(etup) && !etup->deleted &&
			HnswTqMarker(&etup->data) == HNSW_TQ_CODE_MARKER)
		{
			HnswTqCode *code = (HnswTqCode *) &etup->data;

			if (ItemPointerIsValid(&code->reranktid))
			{
				Buffer		rbuf;
				Page		rpage;
				HnswRerankTuple rtup;

				/* Graph page locked before re-rank page, as in HnswTqRerank */
				rbuf = ReadBuffer(index, ItemPointerGetBlockNumber(&code->reranktid));
				LockBuffer(rbuf, BUFFER_LOCK_SHARE);
				rpage = BufferGetPage(rbuf);
				rtup = (HnswRerankTuple) PageGetItem(rpage, PageGetItemId(rpage, ItemPointerGetOffsetNumber(&code->reranktid)));

				if (HnswIsRerankTuple(rtup) && !rtup->deleted)
				{
					if (tq->rerank == HNSW_RERANK_EXACT)
						c->distance = DatumGetFloat8(FunctionCall2Coll(support->procinfo, support->collation,
																	   PointerGetDatum(HnswTqValueOrig(qv)),
																	   PointerGetDatum(rtup->data)));
					else
					{
						float		dot = TqRefineDot(qv->r, code->code, (uint8 *) rtup->data, code->pdim, tq->refineBits);

						c->distance = TqCombine(tq->distance, qv->norm, code->norm, rtup->scale * dot);
					}
				}

				UnlockReleaseBuffer(rbuf);
			}
		}

		UnlockReleaseBuffer(buf);
	}

	qsort(cands, count, sizeof(HnswFastResultCand), CompareFastCands);
}
