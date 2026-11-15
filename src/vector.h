#ifndef VECTOR_H
#define VECTOR_H

#include "fmgr.h"
#include "utils/palloc.h"

#if PG_VERSION_NUM < 190000
#include "storage/shmem.h"		/* for add_size()/mul_size() in some versions */
#endif

#define VECTOR_MAX_DIM 16000

#if PG_VERSION_NUM >= 160000
#include "varatt.h"
#endif

#define VECTOR_SIZE(_dim)		add_size(offsetof(Vector, x), mul_size(sizeof(float), _dim))

static inline struct varlena *
pgvector_detoast_datum_fast(Datum x)
{
	struct varlena *v = (struct varlena *) DatumGetPointer(x);

	if (likely(!VARATT_IS_EXTENDED(v)))
		return v;
	return pg_detoast_datum(v);
}

#define DatumGetVector(x)		((Vector *) pgvector_detoast_datum_fast(x))
#define PG_GETARG_VECTOR_P(x)	DatumGetVector(PG_GETARG_DATUM(x))
#define PG_RETURN_VECTOR_P(x)	PG_RETURN_POINTER(x)

typedef struct Vector
{
	int32		vl_len_;		/* varlena header (do not touch directly!) */
	int16		dim;			/* number of dimensions */
	int16		unused;			/* reserved for future use, always zero */
	float		x[FLEXIBLE_ARRAY_MEMBER];
}			Vector;

Vector	   *InitVector(int dim);
void		PrintVector(char *msg, Vector * vector);
int			vector_cmp_internal(Vector * a, Vector * b);

/* TODO Move to better place */
#if PG_VERSION_NUM >= 160000
#define FUNCTION_PREFIX
#else
#define FUNCTION_PREFIX PGDLLEXPORT
#endif

#endif
