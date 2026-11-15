/*
 * Calico lookup, pin accounting and writer integration. Version-specific
 * PostgreSQL read and writer adapters are maintained alongside this file.
 */
#define CALICO_IN_BUFMGR 1
#include "postgres.h"
#include "port/pg_bitutils.h"
#include <dlfcn.h>

/*
 * Packaged PostgreSQL builds compiled with --enable-dtrace pull in
 * utils/probes.h via pg_trace.h, whose semaphore symbols exist only in the
 * server executable and whose header requires systemtap-sdt-dev. Stub those
 * buffer probes out before including the server buffer manager adapters.
 */
#ifndef PG_TRACE_H
#define PG_TRACE_H
#define TRACE_POSTGRESQL_BUFFER_READ_START(a0, a1, a2, a3, a4, a5) do {} while (0)
#define TRACE_POSTGRESQL_BUFFER_READ_DONE(a0, a1, a2, a3, a4, a5, a6) do {} while (0)
#define TRACE_POSTGRESQL_BUFFER_EXTEND_START(a0, a1, a2, a3, a4, a5) do {} while (0)
#define TRACE_POSTGRESQL_BUFFER_EXTEND_DONE(a0, a1, a2, a3, a4, a5, a6) do {} while (0)
#define TRACE_POSTGRESQL_BUFFER_FLUSH_START(a0, a1, a2, a3, a4) do {} while (0)
#define TRACE_POSTGRESQL_BUFFER_FLUSH_DONE(a0, a1, a2, a3, a4) do {} while (0)
#endif

#if PG_VERSION_NUM >= 180000 && PG_VERSION_NUM < 190000
#include "calico_bufmgr18.inc"
#elif PG_VERSION_NUM >= 190000 && PG_VERSION_NUM < 200000
#include "calico_bufmgr19.inc"
#else
#error "pgvectorfast requires PostgreSQL 18 or 19"
#endif

/* Keep victim selection, mapping locks and I/O setup off resident hits. */
static pg_noinline BufferDesc *
calico_lookup_and_pin_slow(BufferTag *tag,
                          CalicoTranslationEntry *entry_ptr,
                          char relpersistence, BufferAccessStrategy strategy,
                          bool *foundPtr, IOContext io_context)
{
    uint64 entry_val;
    ForkNumber forkNum = tag->forkNum;
    uint32 newHash = BufTableHashCode(tag);
    LWLock *plock = BufMappingPartitionLock(newHash);

    /* ------------- HIT PATH ------------- */
    LWLockAcquire(plock, LW_SHARED);
    entry_val = calico_load_entry(entry_ptr);

    if (CalicoIsValid(entry_val))
    {
        int buf_id = (int) CalicoGetFrameId(entry_val);
        BufferDesc *buf = GetBufferDescriptor(buf_id);
        #if PG_VERSION_NUM >= 190000
        bool valid = PinBuffer(buf, strategy, false);
#else
        bool valid = PinBuffer(buf, strategy);
#endif

        /* Validate tag while we still hold the SAME partition lock */
        if (valid)
        {
            if (unlikely(!BufferTagsEqual(&buf->tag, tag)))
            {
                /* mapping changed under us, treat as miss */
                UnpinBuffer(buf);
                LWLockRelease(plock);
                goto miss_path;
            }
        }

        LWLockRelease(plock);

        if (valid && unlikely(entry_val & CALICO_UNREADY))
            calico_mark_frame_ready(buf);

        *foundPtr = valid;

        return buf;
    }
    LWLockRelease(plock);
    // EVICTED STATE

miss_path:
    /* ------------- MISS PATH ------------- */
    {
        Buffer victim_buf = GetVictimBuffer(strategy, io_context);
        BufferDesc *victim = GetBufferDescriptor(victim_buf - 1);

        LWLockAcquire(plock, LW_EXCLUSIVE);

        entry_val = calico_load_entry(entry_ptr);
        // check invariant: evicted or (locked and valid)
        if (CalicoIsLocked(entry_val) && !CalicoIsValid(entry_val))
        {
            elog(ERROR, "Calico buffer mapping invariant violated: entry is LOCKED but not VALID");
        }
        if (CalicoIsValid(entry_val))
        {
            /* lost race: give up victim, use existing one */
            UnpinBuffer(victim);

            int buf_id = (int) CalicoGetFrameId(entry_val);
            BufferDesc *exist = GetBufferDescriptor(buf_id);
            #if PG_VERSION_NUM >= 190000
        bool valid = PinBuffer(exist, strategy, false);
#else
        bool valid = PinBuffer(exist, strategy);
#endif

            if (valid)
            {
                if (unlikely(!BufferTagsEqual(&exist->tag, tag)))
                {
                    /* Extremely unlikely; fallback to miss again */
                    UnpinBuffer(exist);
                    LWLockRelease(plock);
                    goto miss_path;
                }
            }

            LWLockRelease(plock);

            if (valid && unlikely(entry_val & CALICO_UNREADY))
                calico_mark_frame_ready(exist);

            *foundPtr = valid;
            return exist;
        }
        
        /*
         * Publish mapping via CAS under the SAME partition lock.
         * desired = {VALID | UNREADY, version+1, victim_id}
         */
        uint32 old_ver = CalicoGetVersion(entry_val);
        uint8  old_st  = (CalicoGetState(entry_val) & CALICO_STATE_NO_OPT) |
                         CALICO_STATE_VALID | CALICO_STATE_UNREADY;

        uint64 expect  = entry_val;
        uint64 desired = CalicoPackEntry(victim->buf_id, old_ver + 1, old_st);

        if (!pg_atomic_compare_exchange_u64(entry_ptr, &expect, desired))
        {
            /* someone else inserted concurrently: use theirs */
            UnpinBuffer(victim);
            uint64 now = calico_load_entry(entry_ptr);
            int buf_id = (int) CalicoGetFrameId(now);
            BufferDesc *exist = GetBufferDescriptor(buf_id);
            #if PG_VERSION_NUM >= 190000
        bool valid = PinBuffer(exist, strategy, false);
#else
        bool valid = PinBuffer(exist, strategy);
#endif
            LWLockRelease(plock);
            if (valid && unlikely(now & CALICO_UNREADY))
                calico_mark_frame_ready(exist);
            *foundPtr = valid;
            return exist;
        }

        /*
         * We "won" publishing. Now set the buffer header tag/state.
         * Keep the exact order: directory first, then buffer header.
         */
        CalicoBufferState vstate = LockBufHdr(victim);
        calico_bind_frame(victim, entry_ptr);

        /* These asserts mirror hashtable path’s invariants */
        Assert(BUF_STATE_GET_REFCOUNT(vstate) == 1);
        Assert(!(vstate & (BM_TAG_VALID | BM_VALID | BM_DIRTY | BM_IO_IN_PROGRESS)));

        victim->tag = *tag;

        vstate |= (BM_TAG_VALID | BUF_USAGECOUNT_ONE);
        if (relpersistence == RELPERSISTENCE_PERMANENT || forkNum == INIT_FORKNUM)
            vstate |= BM_PERMANENT;

#if PG_VERSION_NUM >= 190000
        UnlockBufHdrExt(victim, vstate,
            BM_TAG_VALID | BUF_USAGECOUNT_ONE |
            ((relpersistence == RELPERSISTENCE_PERMANENT || forkNum == INIT_FORKNUM) ? BM_PERMANENT : 0),
            0, 0);
#else
        UnlockBufHdr(victim, vstate);
#endif

        LWLockRelease(plock);

        *foundPtr = false;
        return victim;
    }
}

BufferDesc *
CalicoLookupAndPin(SMgrRelation smgr, char relpersistence, ForkNumber forkNum,
                   BlockNumber blockNum, BufferAccessStrategy strategy,
                   bool *foundPtr, IOContext io_context)
{
    BufferTag tag;
    CalicoTranslationEntry *entry_ptr;
    BufferDesc *buf = NULL;
    uint64 entry_val;

    InitBufferTag(&tag, &smgr->smgr_rlocator.locator, forkNum, blockNum);
    entry_ptr = calico_entry_for(&tag, true);
    entry_val = calico_load_entry(entry_ptr);
    if (CalicoIsValid(entry_val))
    {
        int frame = (int) CalicoGetFrameId(entry_val);

        Assert(frame >= 0 && frame < NBuffers);
        buf = GetBufferDescriptor(frame);
        /* Overlap descriptor and page fetches with native pin bookkeeping.
         * The mapping may change here; only the pin and tag checks below
         * establish that this is still the requested page. */
        __builtin_prefetch(buf, 0, 3);
        __builtin_prefetch(BufferGetPage(frame + 1), 0, 3);
    }

    ResourceOwnerEnlarge(CurrentResourceOwner);
    ReservePrivateRefCountEntry();
    if (buf && CalicoTryPinBuffer(buf, strategy))
    {
        if (likely(BufferTagsEqual(&buf->tag, &tag)))
        {
            if (unlikely(entry_val & CALICO_UNREADY))
                calico_mark_frame_ready(buf);
            *foundPtr = true;
            return buf;
        }
        UnpinBuffer(buf);
    }
    return calico_lookup_and_pin_slow(&tag, entry_ptr, relpersistence,
                                      strategy, foundPtr, io_context);
}


/* Native and accelerated paths use the same backend-local pin state. The
 * version adapter binds its typed aliases before any backend is forked. */
#if PG_VERSION_NUM < 190000
unsigned char *calico_writer_bits;
#else
static unsigned char *calico_writer_bits;
#endif
static uint32 calico_writer_count;

void
calico_native_init(void)
{
    if (!calico_writer_bits)
        calico_writer_bits = MemoryContextAllocZero(TopMemoryContext,
                                                   ((Size) NBuffers + 7) / 8);
}

void
calico_native_writer_begin(BufferDesc *buf)
{
    if (calico_local_active)
    {
        Assert(calico_writer_bits != NULL);
        Assert(!(calico_writer_bits[buf->buf_id / 8] & (1U << (buf->buf_id % 8))));
        calico_writer_bits[buf->buf_id / 8] |= 1U << (buf->buf_id % 8);
        calico_writer_count++;
        calico_writer_begin(buf);
    }
}

void
calico_native_writer_end(BufferDesc *buf)
{
    if (calico_local_active)
    {
        calico_writer_end(buf);
        Assert(calico_writer_count > 0);
        calico_writer_count--;
        calico_writer_bits[buf->buf_id / 8] &= ~(1U << (buf->buf_id % 8));
    }
}

void
calico_native_writer_cleanup(Buffer buffer)
{
    if (calico_local_active && buffer > 0 && calico_writer_bits &&
        (calico_writer_bits[(buffer - 1) / 8] & (1U << ((buffer - 1) % 8))))
        calico_native_writer_end(GetBufferDescriptor(buffer - 1));
}

#if PG_VERSION_NUM < 190000
void
calico_native_LWLockReleaseAll(void)
{
    /* ERROR cleanup must clear writer versions BEFORE releasing content
     * locks. Waiting writers can acquire them as soon as ReleaseAll drops
     * them; resource-owner pin cleanup alone happens too late. */
    if (calico_local_active && calico_writer_count)
        for (int i = 0; i < (NBuffers + 7) / 8 && calico_writer_count; i++)
        {
            unsigned char bits = calico_writer_bits[i];
            while (bits)
            {
                int bit = pg_rightmost_one_pos32(bits);
                calico_native_writer_end(GetBufferDescriptor(i * 8 + bit));
                bits &= bits - 1;
            }
        }
    calico_original_LWLockReleaseAll();
}

static BufferDesc *
calico_content_lock_buffer(LWLock *lock)
{
    uintptr_t offset;
    if (!calico_local_active || lock->tranche != LWTRANCHE_BUFFER_CONTENT)
        return NULL;
    offset = (uintptr_t) lock - (uintptr_t) BufferDescriptors;
    if (offset >= (Size) NBuffers * sizeof(BufferDescPadded) ||
        offset % sizeof(BufferDescPadded) != offsetof(BufferDesc, content_lock))
        return NULL;
    return GetBufferDescriptor(offset / sizeof(BufferDescPadded));
}

bool
calico_native_LWLockAcquire(LWLock *lock, LWLockMode mode)
{
    bool result = LWLockAcquire(lock, mode);
    BufferDesc *buf = mode == LW_EXCLUSIVE ? calico_content_lock_buffer(lock) : NULL;
    if (buf)
        calico_native_writer_begin(buf);
    return result;
}

bool
calico_native_LWLockConditionalAcquire(LWLock *lock, LWLockMode mode)
{
    bool result = LWLockConditionalAcquire(lock, mode);
    BufferDesc *buf = result && mode == LW_EXCLUSIVE ? calico_content_lock_buffer(lock) : NULL;
    if (buf)
        calico_native_writer_begin(buf);
    return result;
}

void
calico_native_LWLockRelease(LWLock *lock)
{
    BufferDesc *buf = calico_content_lock_buffer(lock);
    if (buf && (calico_writer_bits[buf->buf_id / 8] & (1U << (buf->buf_id % 8))))
        calico_native_writer_end(buf);
    LWLockRelease(lock);
}
#endif

/* Called after a real backend has initialized its adapted state. Exercise a
 * cross-object native caller and require it to populate OUR pin tracking.
 * Merely finding exported symbols is not sufficient to qualify LTO builds. */
#include "access/heapam.h"
#include "access/table.h"
#include "utils/snapmgr.h"
#include "utils/syscache.h"
#include "catalog/pg_type.h"

void
calico_native_qualify(void)
{
    Relation catalog = table_open(TypeRelationId, AccessShareLock);
    HeapTuple cached = SearchSysCache1(TYPEOID, ObjectIdGetDatum(INT4OID));
    HeapTupleData tuple;
    Buffer buffer = InvalidBuffer;
    void (*native_incr)(Buffer) = dlsym(RTLD_DEFAULT, "IncrBufferRefCount");
    void (*native_release)(Buffer) = dlsym(RTLD_DEFAULT, "ReleaseBuffer");
    int32 pins;
    if (!native_incr || !native_release)
        elog(FATAL, "cannot qualify native buffer pin interoperability");
    if (!HeapTupleIsValid(cached))
        elog(ERROR, "cannot locate Calico qualification tuple");
    tuple.t_self = cached->t_self;
    ReleaseSysCache(cached);
    (void) heap_fetch(catalog, SnapshotAny, &tuple, &buffer, true);
    if (!BufferIsValid(buffer) || GetPrivateRefCount(buffer) == 0)
        elog(FATAL, "Calico cannot see the native heap buffer pin");
    pins = GetPrivateRefCount(buffer);
    native_incr(buffer);
    if (GetPrivateRefCount(buffer) != pins + 1)
        elog(FATAL, "Calico and PostgreSQL do not share buffer pin state");
    native_release(buffer);
    if (GetPrivateRefCount(buffer) != pins)
        elog(FATAL, "native buffer release bypassed Calico pin state");
    ReleaseBuffer(buffer);
    table_close(catalog, AccessShareLock);
}
