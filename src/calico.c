/* Calico lifecycle. The SQL package and preload preparation are independent. */
#include "postgres.h"
#include "calico_internal.h"
#include "fmgr.h"
#include "miscadmin.h"
#include "storage/ipc.h"
#include "storage/aio.h"
#include "utils/guc.h"
#include "utils/builtins.h"
#include "common/config_info.h"
#include "nodes/parsenodes.h"
#include "tcop/pquery.h"

static shmem_startup_hook_type previous_startup;
static shmem_request_hook_type previous_request;
static bool prepared;
bool calico_enable_optimistic_reads = true;

static void
calico_startup(void)
{
    if (previous_startup)
        previous_startup();
    calico_pool_startup();
    calico_lifecycle_startup();
    if (!prepared)
    {
        size_t nconfig;
        ConfigData *config = get_configdata(my_exec_path, &nconfig);
        for (size_t i = 0; i < nconfig; i++)
            if (strstr(config[i].setting, "-flto") ||
                strstr(config[i].setting, "b_lto=true"))
                elog(FATAL, "Calico does not yet qualify LTO PostgreSQL builds");
        calico_install_bufmgr();
        calico_lifecycle_install();
        calico_interception_finish();
        prepared = true;
        elog(LOG, "Calico buffer-manager adapter installed (stock translation)");
    }
}

static void
calico_request(void)
{
    if (previous_request) previous_request();
    calico_pool_request();
    calico_lifecycle_request();
}

void
CalicoInit(void)
{
    DefineCustomBoolVariable("calico.optimistic_reads",
        "Use Calico optimistic HNSW reads when the pool is active.", NULL,
        &calico_enable_optimistic_reads, true, PGC_USERSET, 0, NULL, NULL, NULL);
    if (process_shared_preload_libraries_in_progress)
    {
        if (PG_VERSION_NUM < 180000 || PG_VERSION_NUM >= 200000 ||
            strcmp(PG_VERSION, GetConfigOption("server_version", false, false)))
            elog(FATAL, "Calico requires a PostgreSQL 18 or 19 executable matching its build headers");
        DefineCustomIntVariable("calico.max_regions", "Reserved 128 MiB translation regions.",
            NULL, &calico_max_regions, 8192, 1, 65536, PGC_POSTMASTER, 0, NULL, NULL, NULL);
        calico_register_worker();
        previous_request = shmem_request_hook;
        shmem_request_hook = calico_request;
        previous_startup = shmem_startup_hook;
        shmem_startup_hook = calico_startup;
    }
}

PG_FUNCTION_INFO_V1(calico_status);
Datum
calico_status(PG_FUNCTION_ARGS)
{
    PG_RETURN_TEXT_P(cstring_to_text(prepared ? calico_cluster_status() : "not_preloaded"));
}

PG_FUNCTION_INFO_V1(calico_activate);
Datum
calico_activate(PG_FUNCTION_ARGS)
{
    PlannedStmt *primary;
    CallStmt *call;

    if (!superuser())
        ereport(ERROR, (errcode(ERRCODE_INSUFFICIENT_PRIVILEGE),
                        errmsg("Calico activation requires superuser")));
    if (!prepared)
        ereport(ERROR, (errmsg("Calico requires shared_preload_libraries = 'vectorfast'"),
                        errhint("Preload the library and restart PostgreSQL first.")));
    /* The transaction API cannot infer nested CALL/function execution when
     * given isTopLevel=true. Require this procedure to be the portal's actual
     * primary CALL, in a non-atomic native call context, before any handoff. */
    primary = ActivePortal ? PortalGetPrimaryStmt(ActivePortal) : NULL;
    call = primary && primary->utilityStmt && IsA(primary->utilityStmt, CallStmt) ?
        (CallStmt *) primary->utilityStmt : NULL;
    if (!fcinfo->context || !IsA(fcinfo->context, CallContext) ||
        ((CallContext *) fcinfo->context)->atomic || !call ||
        !call->funcexpr || call->funcexpr->funcid != fcinfo->flinfo->fn_oid)
        ereport(ERROR, (errcode(ERRCODE_ACTIVE_SQL_TRANSACTION),
                        errmsg("CALL calico.activate must be a top-level non-atomic CALL")));
    calico_native_qualify();
    calico_activate_cluster(PG_GETARG_INT32(0));
    PG_RETURN_VOID();
}

/* Group adjacent requests, stage their I/O together, then wait and unpin.
 * Portions Copyright (c) 1996-2025, PostgreSQL Global Development Group
 * Portions Copyright (c) 1994, Regents of the University of California
 */
static void
calico_read_buffers_window(SMgrRelation smgr_reln,
				 char persistence,
				 ForkNumber forkNum,
				 BlockNumber *blockNums,
				 int nblocks)
{
	ReadBuffersOperation operations[16] = {{0}};
    Buffer buffers[16] = {0};
	Buffer	   *range_buffers_array[16];
	int			range_sizes[16];
	int			num_ranges = 0;
	int			flags = 0;
	int			i = 0;

	if (nblocks <= 0)
		return;

	/* 
	 * First pass: count how many contiguous ranges we have.
	 */
	while (i < nblocks)
	{
		int range_size = 1;
		while (i + range_size < nblocks &&
			   blockNums[i + range_size] == blockNums[i] + range_size)
		{
			range_size++;
		}
		num_ranges++;
		i += range_size;
	}

	/* Second pass: prepare all operations */
	i = 0;
	for (int range_idx = 0; range_idx < num_ranges; range_idx++)
	{
		BlockNumber start_block = blockNums[i];
		int range_size = 1;
		
		/* Find the length of this consecutive range */
		while (i + range_size < nblocks &&
			   blockNums[i + range_size] == start_block + range_size)
		{
			range_size++;
		}

		/* Initialize operation for this range */
		operations[range_idx].smgr = smgr_reln;
		operations[range_idx].rel = NULL;
		operations[range_idx].persistence = persistence;
		operations[range_idx].forknum = forkNum;
		operations[range_idx].strategy = NULL;

        /* Bounded flat storage avoids a palloc/pfree for each range. */
        range_buffers_array[range_idx] = buffers + i;
        range_sizes[range_idx] = range_size;

		i += range_size;
	}

	/*
	 * Enter batch mode to stage all I/Os without submitting them.
	 * This is critical for performance - we want to submit all I/Os
	 * in one batch to the kernel.
	 */
	pgaio_enter_batchmode();

	/* Third pass: start all reads (staged but not submitted) */
    PG_TRY();
    {
	i = 0;
	for (int range_idx = 0; range_idx < num_ranges; range_idx++)
	{
		BlockNumber start_block = blockNums[i];
		int range_size = range_sizes[range_idx];
		bool need_wait;
		
		/* Start the read - this stages but doesn't submit in batch mode */
		need_wait = StartReadBuffers(&operations[range_idx],
									 range_buffers_array[range_idx],
									 start_block,
									 &range_size,
									 flags);
		
		/* 
		 * If StartReadBuffers returned false, no I/O was needed (all buffers were cache hits).
		 * Mark this operation so we know not to wait for it.
		 */
		if (!need_wait)
		{
			operations[range_idx].nblocks = 0;
		}
		
        if (range_size < range_sizes[range_idx])
        {
            Assert(range_size > 0 && num_ranges < 16);
            for (int next = num_ranges; next > range_idx + 1; next--)
            {
                range_sizes[next] = range_sizes[next - 1];
                range_buffers_array[next] = range_buffers_array[next - 1];
            }
            range_sizes[range_idx + 1] = range_sizes[range_idx] - range_size;
            range_buffers_array[range_idx + 1] = range_buffers_array[range_idx] + range_size;
            range_sizes[range_idx] = range_size;
            operations[num_ranges].smgr = smgr_reln;
            operations[num_ranges].persistence = persistence;
            operations[num_ranges].forknum = forkNum;
            num_ranges++;
        }
		i += range_sizes[range_idx];
	}
    }
    PG_FINALLY();
    {
        pgaio_exit_batchmode();
    }
    PG_END_TRY();

	/* Fourth pass: wait for all I/Os to complete */
	for (int range_idx = 0; range_idx < num_ranges; range_idx++)
	{
		/* 
		 * Only wait if I/O was actually started (need_wait was true).
		 * We marked operations with no I/O by setting nblocks to 0.
		 */
		if (operations[range_idx].nblocks > 0)
		{
			WaitReadBuffers(&operations[range_idx]);
		}
		
		/* Release the pinned buffers - caller will re-pin via mapping table */
		for (int j = 0; j < range_sizes[range_idx]; j++)
		{
			Buffer b = range_buffers_array[range_idx][j];

			if (BufferIsValid(b))
			{
				if (calico_is_active() && !BufferIsLocal(b))
					calico_mark_frame_ready(GetBufferDescriptor(b - 1));
				ReleaseBuffer(b);
			}
		}
		

	}
}

void
CalicoReadBuffersAsync(SMgrRelation smgr, char persistence, ForkNumber fork,
                       BlockNumber *blocks, int count)
{
    for (int first = 0; first < count; first += 16)
        calico_read_buffers_window(smgr, persistence, fork,
                                   blocks + first, Min(count - first, 16));
}
