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

/* Group sorted, deduplicated adjacent requests, stage their I/O together
 * in batches of up to CALICO_ASYNC_BATCH_MAX blocks, then wait and unpin.
 * Portions Copyright (c) 1996-2025, PostgreSQL Global Development Group
 * Portions Copyright (c) 1994, Regents of the University of California
 */
static void
calico_sort_unique_blocks(const BlockNumber *in, int count,
                          BlockNumber *out, int *uniq_count_out)
{
    int n = 0;
    int u;

    for (int i = 0; i < count; i++)
    {
        BlockNumber key = in[i];
        int j = n;

        while (j > 0 && out[j - 1] > key)
        {
            out[j] = out[j - 1];
            j--;
        }
        out[j] = key;
        n++;
    }

    if (n <= 1)
    {
        *uniq_count_out = n;
        return;
    }

    u = 1;
    for (int i = 1; i < n; i++)
    {
        if (out[i] != out[u - 1])
            out[u++] = out[i];
    }
    *uniq_count_out = u;
}

static void
calico_async_batch_stage(CalicoAsyncBatch *batch,
                         SMgrRelation smgr_reln,
                         char persistence,
                         ForkNumber forkNum,
                         const BlockNumber *sorted_uniq,
                         int nblocks)
{
    int num_ranges = 0;
    int i = 0;
    int staged = 0;
    bool any_io = false;

    batch->active = false;
    batch->num_ranges = 0;

    if (nblocks <= 0)
        return;

    Assert(nblocks <= CALICO_ASYNC_BATCH_MAX);
    memset(batch->buffers, 0, (Size) nblocks * sizeof(Buffer));

    /* Build contiguous ranges capped at MAX_IO_COMBINE_LIMIT */
    while (i < nblocks)
    {
        int range_size = 1;

        while (i + range_size < nblocks &&
               range_size < MAX_IO_COMBINE_LIMIT &&
               sorted_uniq[i + range_size] == sorted_uniq[i] + range_size)
        {
            range_size++;
        }

        memset(&batch->operations[num_ranges], 0, sizeof(ReadBuffersOperation));
        batch->operations[num_ranges].smgr = smgr_reln;
        batch->operations[num_ranges].rel = NULL;
        batch->operations[num_ranges].persistence = persistence;
        batch->operations[num_ranges].forknum = forkNum;
        batch->operations[num_ranges].strategy = NULL;

        batch->range_buffers[num_ranges] = batch->buffers + i;
        batch->range_sizes[num_ranges] = range_size;
        num_ranges++;
        i += range_size;
    }

    {
        bool in_batch = true;

        pgaio_enter_batchmode();

        PG_TRY();
        {
            i = 0;
            for (int range_idx = 0; range_idx < num_ranges; range_idx++)
            {
                BlockNumber start_block = sorted_uniq[i];
                int orig_size = batch->range_sizes[range_idx];
                int range_size = orig_size;
                bool need_wait;

                need_wait = StartReadBuffers(&batch->operations[range_idx],
                                             batch->range_buffers[range_idx],
                                             start_block,
                                             &range_size,
                                             0);

                if (!need_wait)
                    batch->operations[range_idx].nblocks = 0;
                else
                {
                    any_io = true;
                    staged++;
                }

                if (range_size < orig_size)
                {
                    Assert(range_size > 0 && num_ranges < CALICO_ASYNC_BATCH_MAX);
                    for (int next = num_ranges; next > range_idx + 1; next--)
                    {
                        batch->operations[next] = batch->operations[next - 1];
                        batch->range_sizes[next] = batch->range_sizes[next - 1];
                        batch->range_buffers[next] = batch->range_buffers[next - 1];
                    }
                    memset(&batch->operations[range_idx + 1], 0, sizeof(ReadBuffersOperation));
                    batch->operations[range_idx + 1].smgr = smgr_reln;
                    batch->operations[range_idx + 1].rel = NULL;
                    batch->operations[range_idx + 1].persistence = persistence;
                    batch->operations[range_idx + 1].forknum = forkNum;
                    batch->operations[range_idx + 1].strategy = NULL;
                    batch->range_sizes[range_idx + 1] = orig_size - range_size;
                    batch->range_buffers[range_idx + 1] = batch->range_buffers[range_idx] + range_size;
                    batch->range_sizes[range_idx] = range_size;
                    num_ranges++;
                }

                i += batch->range_sizes[range_idx];

                /* Respect pgaio's per-batch staged handle cap (32) */
                if (staged >= 32 && range_idx + 1 < num_ranges)
                {
                    in_batch = false;
                    pgaio_exit_batchmode();
                    staged = 0;
                    pgaio_enter_batchmode();
                    in_batch = true;
                }
            }
        }
        PG_FINALLY();
        {
            if (in_batch)
                pgaio_exit_batchmode();
        }
        PG_END_TRY();
    }

    batch->num_ranges = num_ranges;

    if (!any_io)
    {
        /* All buffers were cache hits: release immediately without marking active */
        for (int range_idx = 0; range_idx < num_ranges; range_idx++)
        {
            for (int j = 0; j < batch->range_sizes[range_idx]; j++)
            {
                Buffer b = batch->range_buffers[range_idx][j];

                if (BufferIsValid(b))
                {
                    if (calico_is_active() && !BufferIsLocal(b))
                        calico_mark_frame_ready(GetBufferDescriptor(b - 1));
                    ReleaseBuffer(b);
                }
            }
        }
        batch->num_ranges = 0;
        return;
    }

    batch->active = true;
}

void
CalicoAsyncBatchStart(CalicoAsyncBatch *batch,
                      SMgrRelation smgr_reln,
                      char persistence,
                      ForkNumber forkNum,
                      const BlockNumber *blocks,
                      int count)
{
    BlockNumber sorted[256];
    int uniq_count;
    int offset = 0;

    batch->active = false;
    batch->num_ranges = 0;

    if (count <= 0)
        return;

    while (count > 256)
    {
        CalicoAsyncBatch tmp;

        calico_sort_unique_blocks(blocks + offset, 256, sorted, &uniq_count);
        for (int pos = 0; pos < uniq_count; pos += CALICO_ASYNC_BATCH_MAX)
        {
            calico_async_batch_stage(&tmp, smgr_reln, persistence, forkNum,
                                     sorted + pos, Min(uniq_count - pos, CALICO_ASYNC_BATCH_MAX));
            CalicoAsyncBatchFinish(&tmp);
        }
        offset += 256;
        count -= 256;
    }

    calico_sort_unique_blocks(blocks + offset, count, sorted, &uniq_count);
    for (int pos = 0; pos < uniq_count; pos += CALICO_ASYNC_BATCH_MAX)
    {
        int chunk = Min(uniq_count - pos, CALICO_ASYNC_BATCH_MAX);

        if (pos + chunk < uniq_count)
        {
            CalicoAsyncBatch tmp;

            calico_async_batch_stage(&tmp, smgr_reln, persistence, forkNum,
                                     sorted + pos, chunk);
            CalicoAsyncBatchFinish(&tmp);
        }
        else
        {
            calico_async_batch_stage(batch, smgr_reln, persistence, forkNum,
                                     sorted + pos, chunk);
        }
    }
}

void
CalicoAsyncBatchFinish(CalicoAsyncBatch *batch)
{
    if (!batch->active)
        return;

    for (int range_idx = 0; range_idx < batch->num_ranges; range_idx++)
    {
        if (batch->operations[range_idx].nblocks > 0)
            WaitReadBuffers(&batch->operations[range_idx]);

        for (int j = 0; j < batch->range_sizes[range_idx]; j++)
        {
            Buffer b = batch->range_buffers[range_idx][j];

            if (BufferIsValid(b))
            {
                if (calico_is_active() && !BufferIsLocal(b))
                    calico_mark_frame_ready(GetBufferDescriptor(b - 1));
                ReleaseBuffer(b);
            }
        }
    }

    batch->active = false;
    batch->num_ranges = 0;
}

void
CalicoReadBuffersAsync(SMgrRelation smgr, char persistence, ForkNumber fork,
                       BlockNumber *blocks, int count)
{
    CalicoAsyncBatch batch;

    CalicoAsyncBatchStart(&batch, smgr, persistence, fork, blocks, count);
    CalicoAsyncBatchFinish(&batch);
}
