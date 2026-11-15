/* pgvectorfast's internal integration contract. PostgreSQL headers are unchanged. */
#ifndef PGVECTORFAST_INTERNAL_H
#define PGVECTORFAST_INTERNAL_H

#if !defined(__linux__) || !defined(__x86_64__)
#error "pgvectorfast currently supports qualified Linux x86-64 builds only"
#endif
#include "postgres.h"
#include "lib/stringinfo.h"
#ifdef EXEC_BACKEND
#error "pgvectorfast requires fork-based PostgreSQL backend creation"
#endif
#include "storage/buf_internals.h"
#include "storage/smgr.h"
#include "utils/memutils.h"
typedef pg_atomic_uint64 CalicoTranslationEntry;

#define CALICO_FRAMEID_MASK     ((uint64)0x00000000FFFFFFFFULL)
#define CALICO_VERSION_MASK     ((uint64)0x00FFFFFF00000000ULL)
#define CALICO_STATE_MASK       ((uint64)0xFF00000000000000ULL)
#define CALICO_VERSION_SHIFT    32
#define CALICO_STATE_SHIFT      56

/* 
 * State bit flags (8-bit state field):
 * Bit 0: VALID flag - indicates entry has a valid mapping
 * Bit 1: LOCKED flag - indicates buffer has exclusive content lock
 * 
 * Valid states:
 * - 0x00 (EVICTED): No valid mapping
 * - 0x01 (VALID): Valid mapping, not locked
 * - 0x03 (VALID|LOCKED): Valid mapping with exclusive lock
 * 
 * Invariants:
 * - EVICTED (0) is mutually exclusive with VALID and LOCKED
 * - LOCKED can only be set when VALID is set
 */
#define CALICO_STATE_NO_OPT 0x04
#define CALICO_STATE_UNREADY    0x08  /* Bit 3: Mapping exists before BM_VALID */
#define CALICO_STATE_VALID      0x01  /* Bit 0: Valid mapping exists */
#define CALICO_STATE_LOCKED     0x02  /* Bit 1: Buffer has exclusive lock */
#define CALICO_STATE_EVICTED    0x00  /* No bits set: evicted state */

static inline uint64
CalicoPackEntry(uint32 frame_id, uint32 version, uint8 state)
{
    /* Saturation prevents ABA after 24-bit version exhaustion. */
    if (version > 0x00ffffffU)
    { version = 0x00ffffffU; state |= CALICO_STATE_NO_OPT; }
    return ((uint64) state << CALICO_STATE_SHIFT) |
           ((uint64) version << CALICO_VERSION_SHIFT) |
           ((uint64) frame_id);
}
static inline uint32 CalicoGetFrameId(uint64 entry) { return (uint32)(entry & CALICO_FRAMEID_MASK); }
static inline uint32 CalicoGetVersion(uint64 entry) { return (uint32)((entry & CALICO_VERSION_MASK) >> CALICO_VERSION_SHIFT); }
static inline uint8  CalicoGetState  (uint64 entry) { return (uint8 )((entry & CALICO_STATE_MASK)   >> CALICO_STATE_SHIFT); }
static inline bool   CalicoIsValid   (uint64 entry) { return (CalicoGetState(entry) & CALICO_STATE_VALID) != 0; }
static inline bool   CalicoIsLocked  (uint64 entry) { return (CalicoGetState(entry) & CALICO_STATE_LOCKED) != 0; }

typedef CalicoTranslationEntry *SecondLevelTable;
#define CALICO_REGION_MASK 0x00ffffffU
#if PG_VERSION_NUM >= 190000
typedef uint64 CalicoBufferState;
#define CALICO_READ_BUFFER_STATE(buf) calico_load_entry(&(buf)->state)
#else
typedef uint32 CalicoBufferState;
#define CALICO_READ_BUFFER_STATE(buf) __atomic_load_n(&(buf)->state.value, __ATOMIC_RELAXED)
#endif

extern void calico_redirect(const char *name, void *replacement, void **original);
extern void calico_redirect_bufmgr(const char *name, void *replacement,
                                   void **original);
extern void calico_redirect_address(void *address, void *replacement,
                                    const char *name, void **original);
extern void *calico_native_data(const char *name, size_t size, size_t alignment);
extern void calico_interception_finish(void);
extern void calico_install_bufmgr(void);
extern void CalicoInit(void);
extern __attribute__((visibility("hidden"))) bool calico_enable_optimistic_reads;
extern BufferDesc *CalicoLookupAndOptimisticPin(SMgrRelation smgr,
    ForkNumber fork, BlockNumber block, bool *found, uint64 *version,
    uint32 *frame, void **entry);
extern bool CalicoLookupAndOptimisticValidate(void *entry, uint64 version);
extern void CalicoLookupAndOptimisticPinBatch(SMgrRelation smgr,
    ForkNumber fork, const BlockNumber *blocks, int count,
    uint32 *frames, bool *found);
extern void CalicoReadBuffersAsync(SMgrRelation smgr, char persistence,
    ForkNumber fork, BlockNumber *blocks, int count);
#ifndef CALICO_IN_BUFMGR
#include "storage/bufmgr.h"
#define CALICO_ASYNC_BATCH_MAX 64

typedef struct CalicoAsyncBatch
{
    ReadBuffersOperation operations[CALICO_ASYNC_BATCH_MAX];
    Buffer      buffers[CALICO_ASYNC_BATCH_MAX];
    Buffer     *range_buffers[CALICO_ASYNC_BATCH_MAX];
    int         range_sizes[CALICO_ASYNC_BATCH_MAX];
    int         num_ranges;
    bool        active;
} CalicoAsyncBatch;

extern void CalicoAsyncBatchStart(CalicoAsyncBatch *batch,
                                  SMgrRelation smgr,
                                  char persistence,
                                  ForkNumber fork,
                                  const BlockNumber *blocks,
                                  int count);
extern void CalicoAsyncBatchFinish(CalicoAsyncBatch *batch);
#endif
extern __attribute__((visibility("hidden"))) bool calico_local_active;
static inline __attribute__((always_inline)) bool
calico_is_active(void)
{
    return calico_local_active;
}
extern int calico_max_regions;
extern void calico_pool_request(void);
extern void calico_pool_startup(void);
extern void calico_pool_seed(void);
extern void calico_install_table(void);
extern int calico_table_lookup(BufferTag *tag, uint32 hashcode);
extern int calico_table_insert(BufferTag *tag, uint32 hashcode, int frame);
extern void calico_table_delete(BufferTag *tag, uint32 hashcode);
extern void calico_table_delete_frame(BufferTag *tag, uint32 hashcode, int buf_id);
extern __attribute__((visibility("hidden"))) pg_atomic_uint64 *calico_frame_entries;

static inline __attribute__((always_inline)) void
calico_prefetch_victim_entry(int buf_id)
{
    if (calico_local_active && calico_frame_entries != NULL)
    {
        uintptr_t addr = (uintptr_t) __atomic_load_n(&calico_frame_entries[buf_id].value,
                                                     __ATOMIC_RELAXED);

        if (addr != 0)
            __builtin_prefetch((const void *) addr, 1, 3);
    }
}
typedef struct CalicoPrefix
{
    Oid tablespace;
    Oid database;
    RelFileNumber relation;
    ForkNumber fork;
    uint32 region;
} CalicoPrefix;

typedef struct CalicoPathCache
{
    CalicoPrefix prefix;
    pg_atomic_uint64 *array;
    bool valid;
} CalicoPathCache;

extern __attribute__((visibility("hidden"))) CalicoPathCache calico_path_cache;
extern pg_atomic_uint64 *calico_entry_for_miss(const BufferTag *tag, bool create);

/* Like the fork's inline TLS path cache: only resolution misses call out to
 * the shared prefix directory. Backends are single-threaded; cached regions
 * are immutable for the entire postmaster lifetime. */
static inline __attribute__((always_inline)) pg_atomic_uint64 *
calico_entry_for(const BufferTag *tag, bool create)
{
    if (calico_path_cache.valid &&
        calico_path_cache.prefix.tablespace == tag->spcOid &&
        calico_path_cache.prefix.database == tag->dbOid &&
        calico_path_cache.prefix.relation == tag->relNumber &&
        calico_path_cache.prefix.fork == tag->forkNum &&
        calico_path_cache.prefix.region == (tag->blockNum >> 24))
        return calico_path_cache.array + (tag->blockNum & CALICO_REGION_MASK);
    return calico_entry_for_miss(tag, create);
}

/* A relation's locator is stable while its operation holds the relation
 * lock. Translation arrays remain allocated until postmaster shutdown;
 * cache their addresses, never a frame or an optimistic version token. */
typedef struct CalicoReadHandle
{
    RelFileLocator locator;
    ForkNumber fork;
    uint32 region;
    CalicoTranslationEntry *array;
} CalicoReadHandle;

static inline void
calico_read_handle_init(CalicoReadHandle *handle,
                        const RelFileLocator *locator, ForkNumber fork)
{
    handle->locator = *locator;
    handle->fork = fork;
    handle->region = 0;
    handle->array = NULL;
}

static pg_noinline pg_attribute_unused() CalicoTranslationEntry *
calico_read_handle_entry_miss(CalicoReadHandle *handle, BlockNumber block)
{
    uint32 region = block >> 24;
    CalicoTranslationEntry *entry;
    BufferTag tag;

    InitBufferTag(&tag, &handle->locator, handle->fork, block);
    entry = calico_entry_for(&tag, false);
    if (entry)
    {
        handle->array = entry - (block & CALICO_REGION_MASK);
        handle->region = region;
    }
    return entry;
}

static inline __attribute__((always_inline)) CalicoTranslationEntry *
calico_read_handle_entry(CalicoReadHandle *handle, BlockNumber block)
{
    uint32 region = block >> 24;

    if (likely(handle->array && handle->region == region))
        return handle->array + (block & CALICO_REGION_MASK);

    return calico_read_handle_entry_miss(handle, block);
}

#ifndef CALICO_IN_BUFMGR
#define CalicoFramePage(frame_id) ((Page) (BufferBlocks + ((Size) (frame_id)) * BLCKSZ))
#endif
extern void calico_writer_begin(BufferDesc *buf);
extern void calico_writer_end(BufferDesc *buf);
extern void calico_native_writer_begin(BufferDesc *buf);
extern void calico_native_writer_end(BufferDesc *buf);
extern void calico_native_writer_cleanup(Buffer buffer);
extern void calico_native_init(void);
#if PG_VERSION_NUM < 190000
extern unsigned char *calico_writer_bits;

/* Public buffer APIs already have a pinned descriptor. Like the fork's
 * SetBufferLocked interface, use it directly instead of recovering it from
 * the content-lock address. The generic adapters below still cover native
 * callers that operate on lock pointers. These bits are backend-private. */
static inline __attribute__((always_inline)) void
calico_native_buffer_unlock(BufferDesc *buf)
{
    if (calico_local_active &&
        (calico_writer_bits[buf->buf_id / 8] & (1U << (buf->buf_id % 8))))
        calico_native_writer_end(buf);
    LWLockRelease(BufferDescriptorGetContentLock(buf));
}

static inline __attribute__((always_inline)) void
calico_native_buffer_lock(BufferDesc *buf, LWLockMode mode)
{
    LWLockAcquire(BufferDescriptorGetContentLock(buf), mode);
    if (mode == LW_EXCLUSIVE)
        calico_native_writer_begin(buf);
}

static inline __attribute__((always_inline)) bool
calico_native_buffer_conditional_lock(BufferDesc *buf)
{
    bool locked = LWLockConditionalAcquire(BufferDescriptorGetContentLock(buf),
                                            LW_EXCLUSIVE);
    if (locked)
        calico_native_writer_begin(buf);
    return locked;
}

extern bool calico_native_LWLockAcquire(LWLock *lock, LWLockMode mode);
extern bool calico_native_LWLockConditionalAcquire(LWLock *lock, LWLockMode mode);
extern void calico_native_LWLockRelease(LWLock *lock);
extern void calico_native_LWLockReleaseAll(void);
#endif
extern void calico_lifecycle_request(void);
extern void calico_lifecycle_startup(void);
extern void calico_lifecycle_install(void);
extern void calico_drain_io(void);
extern void calico_register_worker(void);
extern void calico_activate_cluster(int timeout_ms);
extern const char *calico_cluster_status(void);
extern void calico_append_activation_statistics(StringInfo json);
/* Backend-private counters, updated only on a failed optimistic attempt.
 * No shared RMW, fence, or accounting operation on a successful read. */
extern uint64 calico_optimistic_element_fallbacks;
extern uint64 calico_optimistic_neighbor_fallbacks;
extern void calico_native_qualify(void);
/* Internal native-derived valid-only pin. Caller reserves bookkeeping just
 * as for native PinBuffer. False acquires no shared or private pin. */
extern bool CalicoTryPinBuffer(BufferDesc *buf, BufferAccessStrategy strategy);
extern void calico_check_activation_deadline(void);
extern BufferDesc *CalicoLookupAndPin(SMgrRelation smgr, char persistence,
    ForkNumber fork, BlockNumber block, BufferAccessStrategy strategy,
    bool *found, IOContext io_context);
extern void calico_bind_frame(BufferDesc *buf, pg_atomic_uint64 *entry);

#define CALICO_VALID (UINT64CONST(1) << 56)
#define CALICO_LOCKED (UINT64CONST(2) << 56)
#define CALICO_NO_OPT (UINT64CONST(4) << 56)
#define CALICO_UNREADY (UINT64CONST(8) << 56)
extern __attribute__((visibility("hidden"))) void calico_mark_frame_ready(BufferDesc *buf);
extern __attribute__((visibility("hidden"))) void calico_mark_frame_unready(BufferDesc *buf);
extern __attribute__((visibility("hidden"), noinline)) BufferDesc *calico_try_mark_frame_ready(
    CalicoTranslationEntry *entry_ptr, uint64 *entry_val_io);

/* Qualified x86-64 only: aligned 64-bit loads/stores are single-copy atomic.
 * These intrinsics emit ordinary MOV instructions, with no locked RMW or
 * hardware fence. Alignment is established once by the arena allocator. */
static inline __attribute__((always_inline)) uint64
calico_load_entry(const pg_atomic_uint64 *entry)
{
    return __atomic_load_n(&entry->value, __ATOMIC_RELAXED);
}

static inline __attribute__((always_inline)) void
calico_store_entry(pg_atomic_uint64 *entry, uint64 word)
{
    __atomic_store_n(&entry->value, word, __ATOMIC_RELAXED);
}

static inline uint32
calico_frame(uint64 word)
{
    return (uint32) word;
}

static inline __attribute__((always_inline)) bool
calico_optimistic_lookup_frame_entry(CalicoTranslationEntry *entry_ptr,
                                     uint64 *versionOut,
                                     uint32 *frameIdOut,
                                     void **entryPtrOut)
{
    uint64 entry_val;
    uint32 buf_id;

    if (unlikely(!entry_ptr))
        goto fail;

    /* Read entry atomically */
    entry_val = calico_load_entry(entry_ptr);

    /* Fast path: VALID must be set, and LOCKED/NO_OPT/UNREADY must all be clear. */
    if (unlikely((entry_val & (CALICO_VALID | CALICO_LOCKED | CALICO_NO_OPT | CALICO_UNREADY)) != CALICO_VALID))
    {
        if ((entry_val & (CALICO_VALID | CALICO_LOCKED | CALICO_NO_OPT | CALICO_UNREADY)) ==
            (CALICO_VALID | CALICO_UNREADY))
        {
            BufferDesc *buf = calico_try_mark_frame_ready(entry_ptr, &entry_val);

            if (buf)
            {
                buf_id = CalicoGetFrameId(entry_val);
                goto ready;
            }
        }
        goto fail;
    }

    buf_id = CalicoGetFrameId(entry_val);
ready:
    pg_compiler_barrier();

    *versionOut = entry_val;
    *frameIdOut = buf_id;
    *entryPtrOut = (void *) entry_ptr;
    return true;

fail:
    *versionOut = 0;
    *frameIdOut = 0;
    *entryPtrOut = NULL;
    return false;
}

static inline __attribute__((always_inline)) BufferDesc *
calico_optimistic_lookup_entry(CalicoTranslationEntry *entry_ptr,
                                      bool *foundPtr,
                                      uint64 *versionOut,
                                      uint32 *frameIdOut,
                                      void **entryPtrOut)
{
    if (!calico_optimistic_lookup_frame_entry(entry_ptr, versionOut,
                                              frameIdOut, entryPtrOut))
    {
        *foundPtr = false;
        return NULL;
    }

    *foundPtr = true;
    return GetBufferDescriptor(*frameIdOut);
}

static inline __attribute__((always_inline)) bool
calico_optimistic_lookup_frame(SMgrRelation smgr, ForkNumber forkNum,
                               BlockNumber blockNum, uint64 *versionOut,
                               uint32 *frameIdOut, void **entryPtrOut)
{
    BufferTag tag;
    CalicoTranslationEntry *entry = NULL;

    if (calico_local_active)
    {
        Assert(!SmgrIsTemp(smgr));
        InitBufferTag(&tag, &smgr->smgr_rlocator.locator, forkNum, blockNum);
        entry = calico_entry_for(&tag, false);
    }
    return calico_optimistic_lookup_frame_entry(entry, versionOut,
                                                frameIdOut, entryPtrOut);
}

static inline __attribute__((always_inline)) bool
calico_optimistic_lookup_handle_frame(CalicoReadHandle *handle, BlockNumber block,
                                      uint64 *versionOut, uint32 *frameIdOut,
                                      void **entryPtrOut)
{
    Assert(calico_local_active);
    return calico_optimistic_lookup_frame_entry(calico_read_handle_entry(handle, block),
                                                versionOut, frameIdOut,
                                                entryPtrOut);
}

static inline __attribute__((always_inline)) BufferDesc *
calico_optimistic_lookup(SMgrRelation smgr, ForkNumber forkNum,
                         BlockNumber blockNum, bool *foundPtr,
                         uint64 *versionOut, uint32 *frameIdOut,
                         void **entryPtrOut)
{
    BufferTag tag;
    CalicoTranslationEntry *entry = NULL;

    if (calico_local_active)
    {
        Assert(!SmgrIsTemp(smgr));
        InitBufferTag(&tag, &smgr->smgr_rlocator.locator, forkNum, blockNum);
        entry = calico_entry_for(&tag, false);
    }
    return calico_optimistic_lookup_entry(entry, foundPtr, versionOut,
                                          frameIdOut, entryPtrOut);
}

/* The caller has checked active mode and excluded backend-local buffers. */
static inline __attribute__((always_inline)) BufferDesc *
calico_optimistic_lookup_handle(CalicoReadHandle *handle, BlockNumber block,
                                bool *foundPtr, uint64 *versionOut,
                                uint32 *frameIdOut, void **entryPtrOut)
{
    Assert(calico_local_active);
    return calico_optimistic_lookup_entry(calico_read_handle_entry(handle, block),
                                          foundPtr, versionOut, frameIdOut,
                                          entryPtrOut);
}

static inline __attribute__((always_inline)) void
calico_optimistic_lookup_batch_handle(CalicoReadHandle *handle,
                                      const BlockNumber *blockNums,
                                      int count,
                                      uint32 *frameIdsOut,
                                      bool *foundOut)
{
    CalicoTranslationEntry *table;
    uint32 base_mask;
    int i = 0;

    if (unlikely(!calico_local_active || count <= 0))
    {
        for (int k = 0; k < count; k++)
            foundOut[k] = false;
        return;
    }
    if (unlikely(!handle->array || handle->region != (blockNums[0] >> 24)))
    {
        if (unlikely(!calico_read_handle_entry(handle, blockNums[0])))
        {
            for (int k = 0; k < count; k++)
            {
                CalicoTranslationEntry *e = calico_read_handle_entry(handle, blockNums[k]);
                uint64 v = e ? calico_load_entry(e) : 0;

                frameIdsOut[k] = CalicoGetFrameId(v);
                foundOut[k] = (v & CALICO_VALID) != 0;
            }
            return;
        }
    }
    table = handle->array;
    base_mask = handle->region << 24;

    for (; i + 3 < count; i += 4)
    {
        BlockNumber b0 = blockNums[i + 0];
        BlockNumber b1 = blockNums[i + 1];
        BlockNumber b2 = blockNums[i + 2];
        BlockNumber b3 = blockNums[i + 3];

        if (likely((((b0 ^ base_mask) | (b1 ^ base_mask) |
                     (b2 ^ base_mask) | (b3 ^ base_mask)) >> 24) == 0))
        {
            uint64 v0 = calico_load_entry(&table[b0 & CALICO_REGION_MASK]);
            uint64 v1 = calico_load_entry(&table[b1 & CALICO_REGION_MASK]);
            uint64 v2 = calico_load_entry(&table[b2 & CALICO_REGION_MASK]);
            uint64 v3 = calico_load_entry(&table[b3 & CALICO_REGION_MASK]);

            frameIdsOut[i + 0] = CalicoGetFrameId(v0);
            frameIdsOut[i + 1] = CalicoGetFrameId(v1);
            frameIdsOut[i + 2] = CalicoGetFrameId(v2);
            frameIdsOut[i + 3] = CalicoGetFrameId(v3);
            foundOut[i + 0] = (v0 & CALICO_VALID) != 0;
            foundOut[i + 1] = (v1 & CALICO_VALID) != 0;
            foundOut[i + 2] = (v2 & CALICO_VALID) != 0;
            foundOut[i + 3] = (v3 & CALICO_VALID) != 0;
        }
        else
        {
            for (int k = 0; k < 4; k++)
            {
                CalicoTranslationEntry *e = calico_read_handle_entry(handle, blockNums[i + k]);
                uint64 v = e ? calico_load_entry(e) : 0;

                frameIdsOut[i + k] = CalicoGetFrameId(v);
                foundOut[i + k] = (v & CALICO_VALID) != 0;
            }
            if (handle->array)
            {
                table = handle->array;
                base_mask = handle->region << 24;
            }
        }
    }
    for (; i < count; i++)
    {
        BlockNumber b = blockNums[i];
        CalicoTranslationEntry *e;
        uint64 v;

        if (likely((b >> 24) == (base_mask >> 24)))
            e = &table[b & CALICO_REGION_MASK];
        else
            e = calico_read_handle_entry(handle, b);
        v = e ? calico_load_entry(e) : 0;
        frameIdsOut[i] = CalicoGetFrameId(v);
        foundOut[i] = (v & CALICO_VALID) != 0;
    }
}

typedef struct CalicoOptimisticSlot
{
    CalicoTranslationEntry *entry_ptr;
    uint64                  version;
    uint32                  frame_id;
    bool                    resident;
    bool                    opt_valid;
} CalicoOptimisticSlot;

static inline __attribute__((always_inline)) void
calico_optimistic_prefetch_block(const CalicoReadHandle *handle, BlockNumber blockNum)
{
    if (likely(handle->array != NULL && handle->region == (blockNum >> 24)))
        __builtin_prefetch(&handle->array[blockNum & CALICO_REGION_MASK], 0, 3);
}

static pg_noinline pg_attribute_unused() bool
calico_optimistic_lookup_batch_slots_slow(CalicoReadHandle *handle,
                                          const BlockNumber *blockNums,
                                          int count,
                                          CalicoOptimisticSlot *slotsOut)
{
    bool all_resident = true;

    if (unlikely(!calico_local_active || count <= 0))
    {
        for (int k = 0; k < count; k++)
        {
            slotsOut[k].entry_ptr = NULL;
            slotsOut[k].version = 0;
            slotsOut[k].frame_id = 0;
            slotsOut[k].resident = false;
            slotsOut[k].opt_valid = false;
        }
        return (count <= 0);
    }

    for (int k = 0; k < count; k++)
    {
        CalicoTranslationEntry *e = calico_read_handle_entry(handle, blockNums[k]);
        uint64 v = e ? calico_load_entry(e) : 0;
        bool   res = (v & CALICO_VALID) != 0;

        slotsOut[k].entry_ptr = e;
        slotsOut[k].version = v;
        slotsOut[k].frame_id = CalicoGetFrameId(v);
        slotsOut[k].resident = res;
        slotsOut[k].opt_valid = (v & (CALICO_VALID | CALICO_LOCKED |
                                      CALICO_NO_OPT | CALICO_UNREADY)) == CALICO_VALID;
        if (!res)
            all_resident = false;
    }
    pg_read_barrier();
    return all_resident;
}

static inline __attribute__((always_inline)) bool
calico_optimistic_lookup_batch_slots_impl(CalicoReadHandle *handle,
                                          const BlockNumber *blockNums,
                                          int count,
                                          CalicoOptimisticSlot *slotsOut,
                                          bool do_prefetch,
                                          const uint16 *linpOffsets)
{
    CalicoTranslationEntry *table;
    uint32 base_mask;
    uint64 valid_and = CALICO_VALID;
    int i = 0;

    if (unlikely(!calico_local_active || count <= 0 ||
                 !handle->array || handle->region != (blockNums[0] >> 24)))
        return calico_optimistic_lookup_batch_slots_slow(handle, blockNums, count, slotsOut);

    table = handle->array;
    base_mask = handle->region << 24;

    if (do_prefetch)
    {
        for (int k = 0; k + 3 < count; k += 4)
        {
            BlockNumber p0 = blockNums[k + 0];
            BlockNumber p1 = blockNums[k + 1];
            BlockNumber p2 = blockNums[k + 2];
            BlockNumber p3 = blockNums[k + 3];

            if (likely((((p0 ^ base_mask) | (p1 ^ base_mask) |
                         (p2 ^ base_mask) | (p3 ^ base_mask)) >> 24) == 0))
            {
                __builtin_prefetch(&table[p0 & CALICO_REGION_MASK], 0, 3);
                __builtin_prefetch(&table[p1 & CALICO_REGION_MASK], 0, 3);
                __builtin_prefetch(&table[p2 & CALICO_REGION_MASK], 0, 3);
                __builtin_prefetch(&table[p3 & CALICO_REGION_MASK], 0, 3);
            }
        }
        for (int k = (count & ~3); k < count; k++)
        {
            BlockNumber p = blockNums[k];

            if (likely((p >> 24) == (base_mask >> 24)))
                __builtin_prefetch(&table[p & CALICO_REGION_MASK], 0, 3);
        }
    }

    for (; i + 3 < count; i += 4)
    {
        BlockNumber b0 = blockNums[i + 0];
        BlockNumber b1 = blockNums[i + 1];
        BlockNumber b2 = blockNums[i + 2];
        BlockNumber b3 = blockNums[i + 3];

        if (unlikely((((b0 ^ base_mask) | (b1 ^ base_mask) |
                       (b2 ^ base_mask) | (b3 ^ base_mask)) >> 24) != 0))
        {
            bool slow_res = calico_optimistic_lookup_batch_slots_slow(handle,
                                                                      blockNums + i,
                                                                      count - i,
                                                                      slotsOut + i);
            return ((valid_and & CALICO_VALID) != 0) && slow_res;
        }

        {
            CalicoTranslationEntry *e0 = &table[b0 & CALICO_REGION_MASK];
            CalicoTranslationEntry *e1 = &table[b1 & CALICO_REGION_MASK];
            CalicoTranslationEntry *e2 = &table[b2 & CALICO_REGION_MASK];
            CalicoTranslationEntry *e3 = &table[b3 & CALICO_REGION_MASK];
            uint64 v0 = calico_load_entry(e0);
            uint64 v1 = calico_load_entry(e1);
            uint64 v2 = calico_load_entry(e2);
            uint64 v3 = calico_load_entry(e3);
            uint32 f0 = CalicoGetFrameId(v0);
            uint32 f1 = CalicoGetFrameId(v1);
            uint32 f2 = CalicoGetFrameId(v2);
            uint32 f3 = CalicoGetFrameId(v3);

            valid_and &= v0 & v1 & v2 & v3;

            slotsOut[i + 0].entry_ptr = e0;
            slotsOut[i + 0].version = v0;
            slotsOut[i + 0].frame_id = f0;
            slotsOut[i + 0].resident = (v0 & CALICO_VALID) != 0;
            slotsOut[i + 0].opt_valid = (v0 & (CALICO_VALID | CALICO_LOCKED |
                                               CALICO_NO_OPT | CALICO_UNREADY)) == CALICO_VALID;

            slotsOut[i + 1].entry_ptr = e1;
            slotsOut[i + 1].version = v1;
            slotsOut[i + 1].frame_id = f1;
            slotsOut[i + 1].resident = (v1 & CALICO_VALID) != 0;
            slotsOut[i + 1].opt_valid = (v1 & (CALICO_VALID | CALICO_LOCKED |
                                               CALICO_NO_OPT | CALICO_UNREADY)) == CALICO_VALID;

            slotsOut[i + 2].entry_ptr = e2;
            slotsOut[i + 2].version = v2;
            slotsOut[i + 2].frame_id = f2;
            slotsOut[i + 2].resident = (v2 & CALICO_VALID) != 0;
            slotsOut[i + 2].opt_valid = (v2 & (CALICO_VALID | CALICO_LOCKED |
                                               CALICO_NO_OPT | CALICO_UNREADY)) == CALICO_VALID;

            slotsOut[i + 3].entry_ptr = e3;
            slotsOut[i + 3].version = v3;
            slotsOut[i + 3].frame_id = f3;
            slotsOut[i + 3].resident = (v3 & CALICO_VALID) != 0;
            slotsOut[i + 3].opt_valid = (v3 & (CALICO_VALID | CALICO_LOCKED |
                                               CALICO_NO_OPT | CALICO_UNREADY)) == CALICO_VALID;

#ifndef CALICO_IN_BUFMGR
            if (linpOffsets != NULL)
            {
                if (likely(((v0 & v1 & v2 & v3) & CALICO_VALID) != 0))
                {
                    uint32 off0 = linpOffsets[i + 0];
                    uint32 off1 = linpOffsets[i + 1];
                    uint32 off2 = linpOffsets[i + 2];
                    uint32 off3 = linpOffsets[i + 3];

                    if (likely(off0 != 0))
                        __builtin_prefetch((const char *) CalicoFramePage(f0) + off0, 0, 3);
                    if (likely(off1 != 0))
                        __builtin_prefetch((const char *) CalicoFramePage(f1) + off1, 0, 3);
                    if (likely(off2 != 0))
                        __builtin_prefetch((const char *) CalicoFramePage(f2) + off2, 0, 3);
                    if (likely(off3 != 0))
                        __builtin_prefetch((const char *) CalicoFramePage(f3) + off3, 0, 3);
                }
                else
                {
                    if ((v0 & CALICO_VALID) != 0 && linpOffsets[i + 0] != 0)
                        __builtin_prefetch((const char *) CalicoFramePage(f0) + linpOffsets[i + 0], 0, 3);
                    if ((v1 & CALICO_VALID) != 0 && linpOffsets[i + 1] != 0)
                        __builtin_prefetch((const char *) CalicoFramePage(f1) + linpOffsets[i + 1], 0, 3);
                    if ((v2 & CALICO_VALID) != 0 && linpOffsets[i + 2] != 0)
                        __builtin_prefetch((const char *) CalicoFramePage(f2) + linpOffsets[i + 2], 0, 3);
                    if ((v3 & CALICO_VALID) != 0 && linpOffsets[i + 3] != 0)
                        __builtin_prefetch((const char *) CalicoFramePage(f3) + linpOffsets[i + 3], 0, 3);
                }
            }
#endif
        }
    }
    for (; i < count; i++)
    {
        BlockNumber b = blockNums[i];
        CalicoTranslationEntry *e;
        uint64 v;
        uint32 f;

        if (unlikely((b >> 24) != (base_mask >> 24)))
        {
            bool slow_res = calico_optimistic_lookup_batch_slots_slow(handle,
                                                                      blockNums + i,
                                                                      count - i,
                                                                      slotsOut + i);
            return ((valid_and & CALICO_VALID) != 0) && slow_res;
        }

        e = &table[b & CALICO_REGION_MASK];
        v = calico_load_entry(e);
        f = CalicoGetFrameId(v);
        valid_and &= v;
        slotsOut[i].entry_ptr = e;
        slotsOut[i].version = v;
        slotsOut[i].frame_id = f;
        slotsOut[i].resident = (v & CALICO_VALID) != 0;
        slotsOut[i].opt_valid = (v & (CALICO_VALID | CALICO_LOCKED |
                                      CALICO_NO_OPT | CALICO_UNREADY)) == CALICO_VALID;
#ifndef CALICO_IN_BUFMGR
        if (linpOffsets != NULL &&
            likely((v & CALICO_VALID) != 0 && linpOffsets[i] != 0))
            __builtin_prefetch((const char *) CalicoFramePage(f) + linpOffsets[i], 0, 3);
#endif
    }
    pg_read_barrier();
    return (valid_and & CALICO_VALID) != 0;
}

static inline __attribute__((always_inline)) void
calico_optimistic_lookup_batch_slots(CalicoReadHandle *handle,
                                     const BlockNumber *blockNums,
                                     int count,
                                     CalicoOptimisticSlot *slotsOut)
{
    (void) calico_optimistic_lookup_batch_slots_impl(handle, blockNums, count, slotsOut, true, NULL);
}

static inline __attribute__((always_inline)) bool
calico_optimistic_lookup_batch_slots_noprefetch(CalicoReadHandle *handle,
                                                const BlockNumber *blockNums,
                                                int count,
                                                CalicoOptimisticSlot *slotsOut)
{
    return calico_optimistic_lookup_batch_slots_impl(handle, blockNums, count, slotsOut, false, NULL);
}

static inline __attribute__((always_inline)) bool
calico_optimistic_lookup_batch_slots_noprefetch_linp(CalicoReadHandle *handle,
                                                     const BlockNumber *blockNums,
                                                     int count,
                                                     CalicoOptimisticSlot *slotsOut,
                                                     const uint16 *linpOffsets)
{
    return calico_optimistic_lookup_batch_slots_impl(handle, blockNums, count, slotsOut, false, linpOffsets);
}
static inline __attribute__((always_inline)) bool
calico_optimistic_validate(void *entryPtr,
                                   uint64 expectedVersion)
{
    CalicoTranslationEntry *entry_ptr = (CalicoTranslationEntry *)entryPtr;
    uint64 current_val;
    
    if (entry_ptr == NULL)
        return false;
    
    
    /* Read current entry value */
    pg_compiler_barrier();
    current_val = calico_load_entry(entry_ptr);
    
    /* Compare with expected version - exact match required */
    return (current_val == expectedVersion);
}

#endif
