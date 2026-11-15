/*
 * Live mapping handoff. All code is preloaded before fork. The native process
 * barrier still performs its SMGR-release operation; our wrapper additionally
 * parks participants at safe points. No new PostgreSQL signal reason is used.
 *
 * A single coordinator publishes the mapping after every registered process
 * has parked. It never waits for acknowledgements while holding mapping locks.
 */
#include "postgres.h"
#include "calico_internal.h"
#include "access/xact.h"
#include "access/xlog.h"
#include "access/xlogrecovery.h"
#include "libpq/pqsignal.h"
#include "miscadmin.h"
#include "postmaster/bgworker.h"
#include "postmaster/interrupt.h"
#include "storage/aio.h"
#include "storage/aio_internal.h"
#include "storage/ipc.h"
#include "storage/latch.h"
#include "storage/proc.h"
#include "storage/procsignal.h"
#include "storage/shmem.h"
#include "tcop/tcopprot.h"
#include "utils/timestamp.h"
#include "utils/wait_event.h"

enum { CALICO_STOCK, CALICO_CLAIMED, CALICO_PREPARING, CALICO_QUIESCING, CALICO_ACTIVE };

typedef struct CalicoParticipant
{
    pg_atomic_uint32 pid;
    pg_atomic_uint32 ack;
} CalicoParticipant;

typedef struct CalicoControl
{
    pg_atomic_uint32 phase;
    pg_atomic_uint32 epoch;
    pg_atomic_uint32 worker;
    pg_atomic_uint32 requester;
    pg_atomic_uint32 cancelled;
    TimestampTz deadline;
    char error[256];
    CalicoParticipant participants[FLEXIBLE_ARRAY_MEMBER];
} CalicoControl;

static CalicoControl *control;
static int participant_count;
static bool inside_barrier;
static void (*original_proc_init)(const uint8 *, int);
static void (*original_barrier)(void);
static void abort_activation(const char *message);

PGDLLEXPORT void CalicoWorkerMain(Datum arg);

static void
wake_participants(void)
{
    for (int i = 0; i < participant_count; i++)
        if (pg_atomic_read_u32(&control->participants[i].pid) != 0)
            SetLatch(&ProcGlobal->allProcs[i].procLatch);
}

/* New processes join before touching buffers. They cannot inherit a stale
 * dispatch mode from the postmaster after a live activation. */
static void
participant_init(const uint8 *key, int key_len)
{
    uint32 phase;
    original_proc_init(key, key_len);
    calico_native_init();
    if (!control || MyProcNumber < 0 || MyProcNumber >= participant_count)
        elog(FATAL, "Calico process registration is unavailable");
    pg_atomic_write_u32(&control->participants[MyProcNumber].ack, 0);
    pg_atomic_write_u32(&control->participants[MyProcNumber].pid, MyProcPid);
    while ((phase = pg_atomic_read_u32(&control->phase)) == CALICO_QUIESCING)
    {
        pg_atomic_write_u32(&control->participants[MyProcNumber].ack,
                            pg_atomic_read_u32(&control->epoch));
        ResetLatch(MyLatch);
        WaitLatch(MyLatch, WL_LATCH_SET | WL_TIMEOUT | WL_EXIT_ON_PM_DEATH,
                   10, PG_WAIT_EXTENSION);
    }
    calico_local_active = phase == CALICO_ACTIVE;
}

/* Like native pgaio_shutdown, snapshot only this issuer's in-flight handles.
 * Only this backend can recycle them, and no interrupt is processed between
 * reading the generation and establishing the wait reference. get_wref is
 * deliberately unavailable after submission; it asserts for these states. */
void
calico_drain_io(void)
{
    Assert(pgaio_my_backend && !pgaio_my_backend->in_batchmode &&
           !pgaio_my_backend->handed_out_io);
    pgaio_submit_staged();
    while (!dclist_is_empty(&pgaio_my_backend->in_flight_ios))
    {
        PgAioHandle *ioh = dclist_head_element(PgAioHandle, node,
                                              &pgaio_my_backend->in_flight_ios);
        uint64 generation = ioh->generation;
        PgAioWaitRef ref;

        Assert(ioh->owner_procno == MyProcNumber && generation != 0);
        ref.aio_index = pgaio_io_get_id(ioh);
        ref.generation_upper = (uint32) (generation >> 32);
        ref.generation_lower = (uint32) generation;
        pgaio_wref_wait(&ref);
    }
}

static void
process_barrier(void)
{
    uint32 epoch;
    if (inside_barrier)
        return;
    original_barrier();
    if (!control || pg_atomic_read_u32(&control->phase) != CALICO_QUIESCING ||
        (uint32) MyProcNumber == pg_atomic_read_u32(&control->worker))
        return;
    if (InterruptHoldoffCount || CritSectionCount)
    {
        ProcSignalBarrierPending = true;
        InterruptPending = true;
        return;
    }
    inside_barrier = true;
    PG_TRY();
    {
        /* Do not strand staged operations or require a parked issuer to run
         * completion callbacks. The SMGR barrier normally drained these. */
        if (pgaio_my_backend)
        {
            if (pgaio_my_backend->in_batchmode || pgaio_my_backend->handed_out_io)
            {
                ProcSignalBarrierPending = true;
                InterruptPending = true;
            }
            else
                calico_drain_io();
        }
        if (!pgaio_my_backend ||
            (!pgaio_my_backend->in_batchmode && !pgaio_my_backend->handed_out_io))
        {
            /* Defer cancellation until we have adopted the published mode.
             * The coordinator timeout bounds this critical pause. */
            HOLD_INTERRUPTS();
            epoch = pg_atomic_read_u32(&control->epoch);
            pg_atomic_write_u32(&control->participants[MyProcNumber].ack, epoch);
            while (pg_atomic_read_u32(&control->phase) == CALICO_QUIESCING &&
                   pg_atomic_read_u32(&control->epoch) == epoch)
            {
                /* Cancellation must reach the coordinator while the requester
                 * is parked. Throw only after adopting the resulting mode. */
                if ((uint32) MyProcNumber == pg_atomic_read_u32(&control->requester) &&
                    (QueryCancelPending || ProcDiePending) &&
                    !pg_atomic_read_u32(&control->cancelled))
                    pg_atomic_write_u32(&control->cancelled, 1);
                ResetLatch(MyLatch);
                WaitLatch(MyLatch, WL_LATCH_SET | WL_TIMEOUT | WL_EXIT_ON_PM_DEATH,
                           10, PG_WAIT_EXTENSION);
            }
            calico_local_active = pg_atomic_read_u32(&control->phase) == CALICO_ACTIVE;
            RESUME_INTERRUPTS();
        }
    }
    PG_FINALLY();
    {
        inside_barrier = false;
    }
    PG_END_TRY();
}

void
calico_lifecycle_request(void)
{
    Size size = offsetof(CalicoControl, participants) +
        (Size) (MaxBackends + NUM_AUXILIARY_PROCS) * sizeof(CalicoParticipant);
    RequestAddinShmemSpace(size);
}

void
calico_lifecycle_startup(void)
{
    bool found;
    Size size;
    participant_count = MaxBackends + NUM_AUXILIARY_PROCS;
    size = offsetof(CalicoControl, participants) +
        (Size) participant_count * sizeof(CalicoParticipant);
    control = ShmemInitStruct("Calico Lifecycle", size, &found);
    if (!found)
    {
        memset(control, 0, size);
        pg_atomic_init_u32(&control->phase, CALICO_STOCK);
        pg_atomic_init_u32(&control->epoch, 0);
        pg_atomic_init_u32(&control->worker, PG_UINT32_MAX);
        pg_atomic_init_u32(&control->requester, PG_UINT32_MAX);
        pg_atomic_init_u32(&control->cancelled, 0);
        for (int i = 0; i < participant_count; i++)
        {
            pg_atomic_init_u32(&control->participants[i].pid, 0);
            pg_atomic_init_u32(&control->participants[i].ack, 0);
        }
    }
}

void
calico_lifecycle_install(void)
{
    calico_redirect("ProcSignalInit", (void *) participant_init,
                    (void **) &original_proc_init);
    calico_redirect("ProcessProcSignalBarrier", (void *) process_barrier,
                    (void **) &original_barrier);
}

void
calico_register_worker(void)
{
    BackgroundWorker worker;
    MemSet(&worker, 0, sizeof(worker));
    strlcpy(worker.bgw_name, "Calico activation coordinator", BGW_MAXLEN);
    strlcpy(worker.bgw_type, "Calico activation coordinator", BGW_MAXLEN);
    worker.bgw_flags = BGWORKER_SHMEM_ACCESS;
    worker.bgw_start_time = BgWorkerStart_ConsistentState;
    worker.bgw_restart_time = 1;
    strlcpy(worker.bgw_library_name, "vector", BGW_MAXLEN);
    strlcpy(worker.bgw_function_name, "CalicoWorkerMain", BGW_MAXLEN);
    RegisterBackgroundWorker(&worker);
}

static bool
all_parked(uint32 epoch)
{
    for (int i = 0; i < participant_count; i++)
    {
        if (i == MyProcNumber)
            continue;
        if (pg_atomic_read_u32(&control->participants[i].pid) != 0 &&
            pg_atomic_read_u32(&control->participants[i].pid) == (uint32) ProcGlobal->allProcs[i].pid &&
            pg_atomic_read_u32(&control->participants[i].ack) != epoch)
            return false;
    }
    return true;
}

static void
abort_activation(const char *message)
{
    strlcpy(control->error, message, sizeof(control->error));
    pg_write_barrier();
    pg_atomic_write_u32(&control->phase, CALICO_STOCK);
    wake_participants();
}

static void
coordinator_exit(int code, Datum arg)
{
    uint32 phase;

    /* FATAL bypasses perform_activation's ERROR handler. Drop its mapping
     * locks before waking parked backends. The native hash remains intact
     * unless publication has already made Calico authoritative. */
    LWLockReleaseAll();
    pg_atomic_write_u32(&control->worker, PG_UINT32_MAX);
    phase = pg_atomic_read_u32(&control->phase);
    if (phase == CALICO_QUIESCING)
        abort_activation("Calico activation coordinator exited before publication");
    else
        wake_participants();
    /* CLAIMED belongs to the requester. A PREPARING request can be picked up
     * by our replacement until its requester's deadline; neither state has
     * parked participants or modified the mapping. Do not race the requester
     * initializing or reclaiming that request. */
}

void
calico_check_activation_deadline(void)
{
    if (ProcDiePending || pg_atomic_read_u32(&control->cancelled) ||
        GetCurrentTimestamp() >= control->deadline)
        ereport(ERROR, (errcode(ERRCODE_QUERY_CANCELED),
                       errmsg("Calico activation deadline expired before publication")));
}

static void
perform_activation(void)
{
    uint32 expected = CALICO_PREPARING;
    uint32 epoch;
    TimestampTz last_signal = 0;
    volatile int locks = 0;
    MemoryContext caller_context = CurrentMemoryContext;
    /* A requester can time out before this worker picks up its request.
     * Claim quiescence atomically so it cannot abort a request we are seeding. */
    if (!pg_atomic_compare_exchange_u32(&control->phase, &expected, CALICO_QUIESCING))
        return;
    epoch = pg_atomic_read_u32(&control->epoch);
    PG_TRY();
    {
        elog(LOG, "Calico activation generation %u waiting for safe points", epoch);
        while (!all_parked(epoch))
        {
            TimestampTz now = GetCurrentTimestamp();
            if (pg_atomic_read_u32(&control->cancelled) || now >= control->deadline)
            {
                for (int i = 0; i < participant_count; i++)
                {
                    uint32 pid = pg_atomic_read_u32(&control->participants[i].pid);
                    if (i != MyProcNumber && pid &&
                        pid == (uint32) ProcGlobal->allProcs[i].pid &&
                        pg_atomic_read_u32(&control->participants[i].ack) != epoch)
                        elog(LOG, "Calico safe point pending: PID %u, process slot %d, wait event %u",
                             pid, i, ProcGlobal->allProcs[i].wait_event_info);
                }
                abort_activation("Calico activation cancelled or safe-point timeout expired");
                break;
            }
            if (now - last_signal > 50000)
            {
                EmitProcSignalBarrier(PROCSIGNAL_BARRIER_SMGRRELEASE);
                /* Startup waits on recoveryWakeupLatch rather than MyLatch
                 * while idle in streaming recovery. A SIGUSR1 barrier alone
                 * does not wake that wait to process its pending request. */
                if (RecoveryInProgress())
                    WakeupRecovery();
                last_signal = now;
            }
            ResetLatch(MyLatch);
            WaitLatch(MyLatch, WL_LATCH_SET | WL_TIMEOUT | WL_EXIT_ON_PM_DEATH,
                       5, PG_WAIT_EXTENSION);
            CHECK_FOR_INTERRUPTS();
        }
        if (pg_atomic_read_u32(&control->phase) == CALICO_QUIESCING)
        {
            for (int i = 0; i < NUM_BUFFER_PARTITIONS; i++)
            {
                LWLockAcquire(BufMappingPartitionLockByIndex(i), LW_EXCLUSIVE);
                locks++;
            }
            /* Seed and verify drained I/O in the same descriptor pass.
             * Neither a partial seed nor a failed check publishes Calico. */
            calico_pool_seed();
            calico_check_activation_deadline();
            pg_write_barrier();
            pg_atomic_write_u32(&control->phase, CALICO_ACTIVE);
            calico_local_active = true;
        }
        while (locks > 0)
            LWLockRelease(BufMappingPartitionLockByIndex(--locks));
        wake_participants();
    }
    PG_CATCH();
    {
        ErrorData *error;
        MemoryContextSwitchTo(caller_context);
        error = CopyErrorData();
        /* ERROR resets InterruptHoldoffCount. Native error cleanup balances
         * each release; retail LWLockRelease would underflow that counter.
         * Also release any prefix lock held by a failed hash allocation. */
        LWLockReleaseAll();
        FlushErrorState();
        if (pg_atomic_read_u32(&control->phase) != CALICO_ACTIVE)
            abort_activation(error->message);
        else
            wake_participants();
        FreeErrorData(error);
    }
    PG_END_TRY();
}

void
CalicoWorkerMain(Datum arg)
{
    before_shmem_exit(coordinator_exit, 0);
    pg_atomic_write_u32(&control->worker, MyProcNumber);
    calico_local_active = pg_atomic_read_u32(&control->phase) == CALICO_ACTIVE;
    /* Use normal deferred FATAL termination. Exit status 0 unregisters a
     * static background worker permanently; FATAL lets the postmaster restart
     * this required service. The exit callback releases a pending handoff. */
    pqsignal(SIGTERM, die);
    BackgroundWorkerUnblockSignals();
    for (;;)
    {
        ResetLatch(MyLatch);
        CHECK_FOR_INTERRUPTS();
        if (pg_atomic_read_u32(&control->phase) == CALICO_PREPARING)
            perform_activation();
        WaitLatch(MyLatch, WL_LATCH_SET | WL_TIMEOUT | WL_EXIT_ON_PM_DEATH,
                   100, PG_WAIT_EXTENSION);
        CHECK_FOR_INTERRUPTS();
        ProcessMainLoopInterrupts();
    }
}

const char *
calico_cluster_status(void)
{
    uint32 phase = control ? pg_atomic_read_u32(&control->phase) : CALICO_STOCK;
    switch (phase)
    {
        case CALICO_ACTIVE: return "calico";
        case CALICO_CLAIMED:
        case CALICO_PREPARING: return "preparing";
        case CALICO_QUIESCING: return "quiescing";
        default: return "stock";
    }
}

void
calico_append_activation_statistics(StringInfo json)
{
    uint32 phase = pg_atomic_read_u32(&control->phase);
    uint32 epoch = pg_atomic_read_u32(&control->epoch);
    uint32 worker = pg_atomic_read_u32(&control->worker);
    uint32 requester = pg_atomic_read_u32(&control->requester);
    int participants = 0;
    int parked = 0;

    /* Cold advisory observations, not a synchronized snapshot. Participant
     * slots can join/exit while this inspection query is running. */
    for (int i = 0; i < participant_count; i++)
        if (pg_atomic_read_u32(&control->participants[i].pid) != 0 &&
            pg_atomic_read_u32(&control->participants[i].pid) == (uint32) ProcGlobal->allProcs[i].pid)
        {
            participants++;
            if (phase == CALICO_QUIESCING &&
                pg_atomic_read_u32(&control->participants[i].ack) == epoch)
                parked++;
        }
    appendStringInfo(json,
        ",\"activation\":{\"generation\":%u,\"phase\":%u,\"coordinator_pid\":%d,"
        "\"requester_pid\":%d,\"participants\":%d,\"parked\":%d}",
        epoch, phase,
        worker < (uint32) participant_count ? ProcGlobal->allProcs[worker].pid : 0,
        requester < (uint32) participant_count ? ProcGlobal->allProcs[requester].pid : 0,
        participants, parked);
}

void
calico_activate_cluster(int timeout_ms)
{
    uint32 expected = CALICO_STOCK;
    uint32 epoch;
    if (timeout_ms <= 0 || timeout_ms > 600000)
        ereport(ERROR, (errcode(ERRCODE_INVALID_PARAMETER_VALUE),
                       errmsg("activation timeout must be between 1 and 600000 ms")));
    PreventInTransactionBlock(true, "CALL calico.activate");
    if (pg_atomic_read_u32(&control->phase) == CALICO_ACTIVE)
        return;
    if (!pg_atomic_compare_exchange_u32(&control->phase, &expected, CALICO_CLAIMED))
        ereport(ERROR, (errcode(ERRCODE_OBJECT_IN_USE),
                       errmsg("another Calico activation is in progress")));
    epoch = pg_atomic_fetch_add_u32(&control->epoch, 1) + 1;
    if (!epoch)
    {
        /* Keep later callers exhausted too: never issue an old generation
         * again after this error. This is a cold, restart-only boundary. */
        pg_atomic_write_u32(&control->epoch, PG_UINT32_MAX);
        pg_atomic_write_u32(&control->phase, CALICO_STOCK);
        elog(ERROR, "Calico activation generation exhausted; restart required");
    }
    control->error[0] = '\0';
    control->deadline = TimestampTzPlusMilliseconds(GetCurrentTimestamp(), timeout_ms);
    pg_atomic_write_u32(&control->cancelled, 0);
    pg_atomic_write_u32(&control->requester, MyProcNumber);
    pg_write_barrier();
    pg_atomic_write_u32(&control->phase, CALICO_PREPARING);
    wake_participants();
    PG_TRY();
    {
        while ((expected = pg_atomic_read_u32(&control->phase)) == CALICO_PREPARING ||
               expected == CALICO_QUIESCING)
        {
            if (expected == CALICO_PREPARING && GetCurrentTimestamp() >= control->deadline)
            {
                /* The coordinator can be restarting or not yet scheduled.
                 * Reclaim only a request it has not begun to quiesce. */
                if (pg_atomic_compare_exchange_u32(&control->phase, &expected, CALICO_CLAIMED))
                    abort_activation("Calico activation deadline expired waiting for coordinator");
                continue;
            }
            ResetLatch(MyLatch);
            WaitLatch(MyLatch, WL_LATCH_SET | WL_TIMEOUT | WL_EXIT_ON_PM_DEATH,
                       10, PG_WAIT_EXTENSION);
            CHECK_FOR_INTERRUPTS();
        }
    }
    PG_CATCH();
    {
        pg_atomic_write_u32(&control->cancelled, 1);
        wake_participants();
        PG_RE_THROW();
    }
    PG_END_TRY();
    calico_local_active = expected == CALICO_ACTIVE;
    if (!calico_local_active)
        ereport(ERROR, (errcode(ERRCODE_OBJECT_NOT_IN_PREREQUISITE_STATE),
                       errmsg("%s", control->error)));
}
