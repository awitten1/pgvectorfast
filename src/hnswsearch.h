/* HNSW scan traversal entry points. */
#ifndef HNSWSEARCH_H
#define HNSWSEARCH_H

#include "hnswqueue.h"

bool HnswSearchLayerFastCands(char *base, HnswQuery *q, List *ep, int ef, int lc,
							  Relation index, HnswSupport *support, int m,
							  visited_hash *v, int64 *tuples, HnswFastResultCand *inline_buf,
							  int inline_cap, HnswFastResultCand **fast_out, int *fast_count_out);
bool HnswSearchLayerFromEntryFastCands(char *base, HnswQuery *q, HnswElement entryPoint,
									   int ef, Relation index, HnswSupport *support, int m,
									   visited_hash *v, int64 *tuples, HnswFastResultCand *inline_buf,
									   int inline_cap, HnswFastResultCand **fast_out, int *fast_count_out);
void HnswSearchIterativeFromEntry(HnswQuery *q, HnswElement entryPoint, int ef,
								  Relation index, HnswSupport *support, int m,
								  visited_hash *v, int64 *tuples, HnswIterativeState **state_ptr);
void HnswSearchIterativeBatch(HnswQuery *q, int ef, Relation index,
							  HnswSupport *support, int m, visited_hash *v, int64 *tuples,
							  HnswIterativeState *state, bool expand);

extern __attribute__((visibility("hidden"))) List *
HnswSearchLayerDiskFast(char *base, HnswQuery * q, List *ep, int ef, int lc,
						Relation index, HnswSupport * support, int m,
						visited_hash * v, bool initVisited, int64 *tuples,
						HnswFastResultCand *inline_buf, int inline_cap,
						HnswFastResultCand **fast_out, int *fast_count_out);

extern __attribute__((visibility("hidden"))) pg_noinline List *
HnswMaterializeFastList(char *base, HnswQuery * q, Relation index,
					   HnswSupport * support,
					   const HnswFastResultCand *w_heap, int wlen);

extern __attribute__((visibility("hidden"))) pg_noinline void
HnswSearchLayerUpperDiskFast(HnswQuery * q, HnswElement entryPoint,
							 Relation index, HnswSupport * support, int m,
							 bool canOptDistance, const Vector *queryVec,
							 Size optVecSize, Size optPrefetchBytes,
							 HnswFastCand *best_out);

#endif							/* HNSWSEARCH_H */
