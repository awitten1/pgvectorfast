/*
 * Array translation, adapted from the Calico PostgreSQL fork c1868765f1b.
 *
 * A prefix is (tablespace, database, relation, fork, block >> 24). Resolving
 * it yields an immutable 128 MiB array in an inherited MAP_SHARED arena.
 * The common path caches that pointer and indexes with the low 24 bits.
 * Frames remain PostgreSQL's frames; this module never performs data I/O.
 */
#include "postgres.h"
#include "calico_internal.h"
#include "storage/shmem.h"
#include "storage/lwlock.h"
#include "utils/hsearch.h"
#include "fmgr.h"
#include "utils/builtins.h"
#include "lib/stringinfo.h"
#include <sys/mman.h>

#define REGION_BITS 24
#define REGION_ENTRIES (UINT64CONST(1) << REGION_BITS)
#define REGION_BYTES (REGION_ENTRIES * sizeof(pg_atomic_uint64))
#define PREFIX_PARTITIONS 128

typedef struct CalicoRegion
{
    CalicoPrefix prefix;
    uint32 slot;
} CalicoRegion;

typedef struct NativeMapEntry
{
    BufferTag tag;
    int frame;
} NativeMapEntry;

static HTAB *prefixes;
static HTAB **native_map;
static int (*native_lookup)(BufferTag *, uint32);
static int (*native_insert)(BufferTag *, uint32, int);
static void (*native_delete)(BufferTag *, uint32);
static char *arena;
static Size arena_size;
static pg_atomic_uint32 *next_region;
pg_atomic_uint64 *calico_frame_entries;
static LWLockPadded *prefix_locks;
CalicoPathCache calico_path_cache;
/* Keep the inline primary hit unchanged. A bounded backend-local victim
 * cache avoids directory locks when operations alternate among relations.
 * Cache array addresses only; entries and frames are still checked afresh. */
#define RECENT_PATH_COUNT 4
static CalicoPathCache recent_paths[RECENT_PATH_COUNT];
static unsigned int next_recent_path;

/* Fixed reservation, not resident memory. Cold regions are not initialized
 * with a write loop. Physical pages are populated on demand. */
int calico_max_regions = 8192;
bool calico_local_active;
uint64 calico_optimistic_element_fallbacks;
uint64 calico_optimistic_neighbor_fallbacks;

void
calico_pool_request(void)
{
    RequestAddinShmemSpace(hash_estimate_size(calico_max_regions, sizeof(CalicoRegion)) +
                          (Size) NBuffers * sizeof(pg_atomic_uint64) + 4096);
    RequestNamedLWLockTranche("calico_prefix", PREFIX_PARTITIONS);
}

void
calico_pool_startup(void)
{
    HASHCTL info;
    bool found;
    MemSet(&info, 0, sizeof(info));
    info.keysize = sizeof(CalicoPrefix);
    info.entrysize = sizeof(CalicoRegion);
    info.num_partitions = PREFIX_PARTITIONS;
#if PG_VERSION_NUM >= 190000
    prefixes = ShmemInitHash("Calico Prefixes", calico_max_regions, &info,
                            HASH_ELEM | HASH_BLOBS | HASH_PARTITION | HASH_FIXED_SIZE);
#else
    prefixes = ShmemInitHash("Calico Prefixes", calico_max_regions,
                            calico_max_regions, &info,
                            HASH_ELEM | HASH_BLOBS | HASH_PARTITION);
#endif
    next_region = ShmemInitStruct("Calico Next Region", sizeof(*next_region), &found);
    if (!found)
        pg_atomic_init_u32(next_region, 0);
    calico_frame_entries = ShmemInitStruct("Calico Frame Entries",
                                           (Size) NBuffers * sizeof(*calico_frame_entries), &found);
    if (!found)
        memset(calico_frame_entries, 0, (Size) NBuffers * sizeof(*calico_frame_entries));
    prefix_locks = GetNamedLWLockTranche("calico_prefix");

    if (arena)
        munmap(arena, arena_size); /* only during postmaster crash restart */
    arena_size = (Size) calico_max_regions * REGION_BYTES;
    arena = mmap(NULL, arena_size, PROT_READ | PROT_WRITE,
                 MAP_SHARED | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
    if (arena == MAP_FAILED)
        elog(FATAL, "cannot reserve Calico translation arena: %m");
    if (madvise(arena, arena_size, MADV_NOHUGEPAGE))
        elog(WARNING, "cannot request small pages for Calico metadata: %m");
    calico_path_cache.valid = false;
    memset(recent_paths, 0, sizeof(recent_paths));
    next_recent_path = 0;
    calico_local_active = false;
}

void
calico_install_table(void)
{
    /* Reuse PostgreSQL's authoritative stock map, including on retry. The
     * alias follows its reset during postmaster shared-memory reinitialization. */
    native_map = calico_native_data("SharedBufHash", sizeof(*native_map),
                                    __alignof__(*native_map));
    if (!*native_map)
        elog(FATAL, "Calico cannot locate the native buffer mapping");
    calico_redirect_bufmgr("BufTableLookup", (void *) calico_table_lookup,
                    (void **) &native_lookup);
    calico_redirect_bufmgr("BufTableInsert", (void *) calico_table_insert,
                    (void **) &native_insert);
    calico_redirect_bufmgr("BufTableDelete", (void *) calico_table_delete,
                    (void **) &native_delete);
}

pg_atomic_uint64 *
calico_entry_for_miss(const BufferTag *tag, bool create)
{
    CalicoPrefix key;
    CalicoRegion *region;
    pg_atomic_uint64 *array;
    uint32 hashcode;
    LWLock *lock;
    bool found;
    key.tablespace = tag->spcOid;
    key.database = tag->dbOid;
    key.relation = tag->relNumber;
    key.fork = tag->forkNum;
    key.region = tag->blockNum >> REGION_BITS;
    for (int i = 0; i < RECENT_PATH_COUNT; i++)
    {
        if (recent_paths[i].valid &&
            memcmp(&recent_paths[i].prefix, &key, sizeof(key)) == 0)
        {
            CalicoPathCache cached = recent_paths[i];
            recent_paths[i] = calico_path_cache;
            calico_path_cache = cached;
            return cached.array + (tag->blockNum & (REGION_ENTRIES - 1));
        }
    }
    hashcode = get_hash_value(prefixes, &key);
    lock = &prefix_locks[hashcode % PREFIX_PARTITIONS].lock;
    LWLockAcquire(lock, LW_SHARED);
    region = hash_search_with_hash_value(prefixes, &key, hashcode,
                                         HASH_FIND, &found);
    if (!region && create)
    {
        LWLockRelease(lock);
        LWLockAcquire(lock, LW_EXCLUSIVE);
        /* Another backend may have created this prefix while we waited. */
        region = hash_search_with_hash_value(prefixes, &key, hashcode,
                                             HASH_ENTER_NULL, &found);
    }
    if (!region)
    {
        LWLockRelease(lock);
        if (create)
            ereport(ERROR, (errcode(ERRCODE_OUT_OF_MEMORY),
                errmsg("Calico translation region limit reached"),
                errhint("Increase calico.max_regions at server startup.")));
        return NULL;
    }
    if (!found)
    {
        uint32 slot = pg_atomic_read_u32(next_region);
        /* Prefix partitions allocate concurrently. Saturate this cold-path
         * counter so repeated allocation failures cannot wrap into reuse. */
        while (slot < (uint32) calico_max_regions &&
               !pg_atomic_compare_exchange_u32(next_region, &slot, slot + 1))
            ;
        if (slot >= (uint32) calico_max_regions)
        {
            hash_search_with_hash_value(prefixes, &key, hashcode, HASH_REMOVE, NULL);
            LWLockRelease(lock);
            ereport(ERROR, (errcode(ERRCODE_OUT_OF_MEMORY),
                            errmsg("Calico translation arena exhausted")));
        }
        region->slot = slot;
    }
    array = (pg_atomic_uint64 *) (arena + (Size) region->slot * REGION_BYTES);
    LWLockRelease(lock);
    if (calico_path_cache.valid)
    {
        recent_paths[next_recent_path] = calico_path_cache;
        next_recent_path = (next_recent_path + 1) % RECENT_PATH_COUNT;
    }
    calico_path_cache.prefix = key;
    calico_path_cache.array = array;
    calico_path_cache.valid = true;
    return array + (tag->blockNum & (REGION_ENTRIES - 1));
}

/* Saturate instead of wrapping the 24-bit version. Such an entry remains
 * usable through pinned access, but can never issue another optimistic token. */
static inline uint64
next_word(uint64 old, uint32 frame, uint64 flags)
{
    return CalicoPackEntry(frame, CalicoGetVersion(old) + 1,
        (uint8) (flags >> CALICO_STATE_SHIFT) |
        (CalicoGetState(old) & CALICO_STATE_NO_OPT));
}

void
calico_bind_frame(BufferDesc *buf, pg_atomic_uint64 *entry)
{
    calico_store_entry(&calico_frame_entries[buf->buf_id], (uintptr_t) entry);
}

int
calico_table_lookup(BufferTag *tag, uint32 hashcode)
{
    if (!calico_local_active)
    {
        return native_lookup(tag, hashcode);
    }
    else
    {
        pg_atomic_uint64 *entry = calico_entry_for(tag, false);
        uint64 value = entry ? calico_load_entry(entry) : 0;
        return value & CALICO_VALID ? (int) calico_frame(value) : -1;
    }
}

int
calico_table_insert(BufferTag *tag, uint32 hashcode, int frame)
{
    if (!calico_local_active)
    {
        return native_insert(tag, hashcode, frame);
    }
    else
    {
        pg_atomic_uint64 *entry = calico_entry_for(tag, true);
        uint64 old = calico_load_entry(entry);
        uint64 flags;
        if (old & CALICO_VALID)
            return (int) calico_frame(old);
        flags = (CALICO_READ_BUFFER_STATE(GetBufferDescriptor(frame)) & BM_VALID) ?
            CALICO_VALID : (CALICO_VALID | CALICO_UNREADY);
        calico_store_entry(&calico_frame_entries[frame], (uintptr_t) entry);
        pg_write_barrier();
        calico_store_entry(entry, next_word(old, frame, flags));
        return -1;
    }
}

void
calico_table_delete(BufferTag *tag, uint32 hashcode)
{
    if (!calico_local_active)
    {
        native_delete(tag, hashcode);
    }
    else
    {
        pg_atomic_uint64 *entry = calico_entry_for(tag, false);
        uint64 old = entry ? calico_load_entry(entry) : 0;
        if (!(old & CALICO_VALID) || (old & CALICO_LOCKED))
            elog(ERROR, "Calico mapping missing or writer locked during deletion");
        calico_store_entry(entry, next_word(old, 0, 0));
        calico_store_entry(&calico_frame_entries[calico_frame(old)], 0);
    }
}

void
calico_table_delete_frame(BufferTag *tag, uint32 hashcode, int buf_id)
{
    if (!calico_local_active)
    {
        native_delete(tag, hashcode);
    }
    else
    {
        pg_atomic_uint64 *entry = (void *) (uintptr_t)
            calico_load_entry(&calico_frame_entries[buf_id]);
        uint64 old;

        if (unlikely(entry == NULL))
            entry = calico_entry_for(tag, false);

        old = entry ? calico_load_entry(entry) : 0;
        if (!(old & CALICO_VALID) || (old & CALICO_LOCKED) ||
            (int) calico_frame(old) != buf_id)
            elog(ERROR, "Calico mapping missing or writer locked during deletion");

        calico_store_entry(entry, next_word(old, 0, 0));
        calico_store_entry(&calico_frame_entries[buf_id], 0);
    }
}

void
calico_writer_begin(BufferDesc *buf)
{
    if (calico_local_active)
    {
        pg_atomic_uint64 *entry = (void *) (uintptr_t)
            calico_load_entry(&calico_frame_entries[buf->buf_id]);
        if (entry)
        {
            uint64 old = calico_load_entry(entry);
            Assert(old & CALICO_VALID);
            Assert(!(old & CALICO_LOCKED));
            calico_store_entry(entry, next_word(old, buf->buf_id,
                                                CALICO_VALID | CALICO_LOCKED |
                                                (old & CALICO_UNREADY)));
            pg_write_barrier();
        }
    }
}

void
calico_writer_end(BufferDesc *buf)
{
    if (calico_local_active)
    {
        pg_atomic_uint64 *entry = (void *) (uintptr_t)
            calico_load_entry(&calico_frame_entries[buf->buf_id]);
        if (entry)
        {
            uint64 old = calico_load_entry(entry);
            uint64 flags;
            Assert(old & CALICO_VALID);
            Assert(old & CALICO_LOCKED);
            flags = (CALICO_READ_BUFFER_STATE(buf) & BM_VALID) ?
                CALICO_VALID : (CALICO_VALID | CALICO_UNREADY);
            pg_write_barrier();
            calico_store_entry(entry, next_word(old, buf->buf_id, flags));
        }
    }
}

pg_noinline BufferDesc *
calico_try_mark_frame_ready(CalicoTranslationEntry *entry_ptr, uint64 *entry_val_io)
{
    uint64 old = *entry_val_io;

    while ((old & (CALICO_VALID | CALICO_LOCKED | CALICO_NO_OPT | CALICO_UNREADY)) ==
           (CALICO_VALID | CALICO_UNREADY))
    {
        BufferDesc *buf = GetBufferDescriptor(CalicoGetFrameId(old));

        if (!(CALICO_READ_BUFFER_STATE(buf) & BM_VALID))
            return NULL;
        if (pg_atomic_compare_exchange_u64(entry_ptr, &old, old & ~CALICO_UNREADY))
        {
            *entry_val_io = old & ~CALICO_UNREADY;
            return buf;
        }
    }
    return NULL;
}

void
calico_mark_frame_ready(BufferDesc *buf)
{
    pg_atomic_uint64 *entry;
    uint64 old;

    if (!calico_local_active)
        return;
    entry = (void *) (uintptr_t) calico_load_entry(&calico_frame_entries[buf->buf_id]);
    if (!entry)
        return;
    old = calico_load_entry(entry);
    while ((old & (CALICO_VALID | CALICO_LOCKED | CALICO_NO_OPT | CALICO_UNREADY)) ==
           (CALICO_VALID | CALICO_UNREADY))
    {
        if (CalicoGetFrameId(old) != (uint32) buf->buf_id)
            return;
        if (!(CALICO_READ_BUFFER_STATE(buf) & BM_VALID))
            return;
        if (pg_atomic_compare_exchange_u64(entry, &old, old & ~CALICO_UNREADY))
            return;
    }
}

void
calico_mark_frame_unready(BufferDesc *buf)
{
    pg_atomic_uint64 *entry;
    uint64 old;

    if (!calico_local_active)
        return;
    entry = (void *) (uintptr_t) calico_load_entry(&calico_frame_entries[buf->buf_id]);
    if (!entry)
        return;
    old = calico_load_entry(entry);
    while ((old & CALICO_VALID) && CalicoGetFrameId(old) == (uint32) buf->buf_id)
    {
        uint64 desired = next_word(old, buf->buf_id,
                                   (old & CALICO_LOCKED) | CALICO_VALID | CALICO_UNREADY);
        if (pg_atomic_compare_exchange_u64(entry, &old, desired))
        {
            pg_write_barrier();
            return;
        }
    }
}

static inline void
calico_seed_frame(BufferDesc *buf, CalicoBufferState state)
{
    if (state & BM_IO_IN_PROGRESS)
        elog(ERROR, "Calico handoff encountered undrained buffer I/O");
    if (state & BM_TAG_VALID)
    {
        pg_atomic_uint64 *entry = calico_entry_for(&buf->tag, true);
        uint64 flags = (state & BM_VALID) ?
            CALICO_VALID : (CALICO_VALID | CALICO_UNREADY);
        calico_store_entry(&calico_frame_entries[buf->buf_id], (uintptr_t) entry);
        calico_store_entry(entry, next_word(calico_load_entry(entry),
                                            buf->buf_id, flags));
    }
}

/* Caller has parked all participants and owns every mapping partition.
 * Native insert/delete updates tag validity under these same locks. Thus
 * every tagged shared frame, including an incomplete page, is in the stock
 * map, and native shared I/O cannot belong to an unmapped frame here. */
void
calico_pool_seed(void)
{
    /* Clear any partial seed through its reverse pointers before rebuilding.
     * Reverse pointers require an allocated region. The allocation counter
     * saturates and never resets within this shared-memory lifetime, so zero
     * also excludes an incomplete earlier seed. */
    if (pg_atomic_read_u32(next_region) != 0)
    {
        for (int i = 0; i < NBuffers; i++)
        {
            if ((i & 1023) == 0)
                calico_check_activation_deadline();
            pg_atomic_uint64 *old = (void *) (uintptr_t) calico_load_entry(&calico_frame_entries[i]);
            if (old)
                calico_store_entry(old, 0);
            calico_store_entry(&calico_frame_entries[i], 0);
        }
    }

    /* A dense map favors sequential descriptors. At lower occupancy, scan
     * the smaller native bucket directory and touch only mapped descriptors.
     * This cold choice changes traversal only, never mapping semantics. */
    if (hash_get_num_entries(*native_map) >= NBuffers / 2)
    {
        for (int i = 0; i < NBuffers; i++)
        {
            BufferDesc *buf = GetBufferDescriptor(i);
            if ((i & 1023) == 0)
                calico_check_activation_deadline();
            calico_seed_frame(buf, CALICO_READ_BUFFER_STATE(buf));
        }
    }
    else
    {
        /* Mutable iterator state must survive a PG_TRY error jump. Keep it
         * outside automatic storage; the pointer itself never changes. */
        HASH_SEQ_STATUS *scan = palloc(sizeof(*scan));
        NativeMapEntry *mapping;
        volatile bool scanning = false;
        uint32 visited = 0;

        PG_TRY();
        {
            calico_check_activation_deadline();
            hash_seq_init(scan, *native_map);
            scanning = true;
            while ((mapping = hash_seq_search(scan)) != NULL)
            {
                BufferDesc *buf;
                CalicoBufferState state;
                if ((visited++ & 1023) == 0)
                    calico_check_activation_deadline();
                if (mapping->frame < 0 || mapping->frame >= NBuffers)
                    elog(ERROR, "Calico found an invalid native buffer mapping");
                buf = GetBufferDescriptor(mapping->frame);
                state = CALICO_READ_BUFFER_STATE(buf);
                if (!(state & BM_TAG_VALID) ||
                    !BufferTagsEqual(&mapping->tag, &buf->tag))
                    elog(ERROR, "Calico found an inconsistent native buffer mapping");
                calico_seed_frame(buf, state);
            }
            scanning = false;
        }
        PG_FINALLY();
        {
            /* The coordinator catches errors without transaction abort, so
             * an abandoned native hash scan needs explicit local cleanup. */
            if (scanning)
                hash_seq_term(scan);
            pfree(scan);
        }
        PG_END_TRY();
    }
}

PG_FUNCTION_INFO_V1(calico_pool_statistics);
Datum
calico_pool_statistics(PG_FUNCTION_ARGS)
{
    volatile int locked = 0;
    int mapped = 0;
    int writer_locked = 0;
    int no_optimistic = 0;
    int mismatches = 0;
    StringInfoData json;
    if (!prefixes)
        ereport(ERROR, (errmsg("Calico was not preloaded")));
    /* Verify a real native heap caller in this backend before reporting the
     * runtime pin-route check. This is outside all mapping locks and loops. */
    calico_native_qualify();
    PG_TRY();
    {
        for (int i = 0; i < NUM_BUFFER_PARTITIONS; i++)
        {
            LWLockAcquire(BufMappingPartitionLockByIndex(i), LW_EXCLUSIVE);
            locked++;
        }
        for (int i = 0; i < NBuffers; i++)
        {
            BufferDesc *buf = GetBufferDescriptor(i);
            if (CALICO_READ_BUFFER_STATE(buf) & BM_TAG_VALID)
            {
                mapped++;
                if (calico_local_active)
                {
                    pg_atomic_uint64 *entry = calico_entry_for(&buf->tag, false);
                    uint64 word = entry ? calico_load_entry(entry) : 0;
                    if (!(word & CALICO_VALID) || calico_frame(word) != (uint32) i)
                        mismatches++;
                    if (word & CALICO_LOCKED)
                        writer_locked++;
                    if (word & CALICO_NO_OPT)
                        no_optimistic++;
                }
            }
        }
        while (locked)
            LWLockRelease(BufMappingPartitionLockByIndex(--locked));
    }
    PG_CATCH();
    {
        LWLockReleaseAll();
        PG_RE_THROW();
    }
    PG_END_TRY();
    initStringInfo(&json);
    appendStringInfo(&json,
        "{\"mode\":\"%s\",\"backend_active\":%s,\"frames\":%d,"
        "\"mapped_frames\":%d,\"regions\":%u,\"virtual_bytes\":%zu,"
        "\"writer_locked_entries\":%d,\"no_optimistic_entries\":%d,\"mapping_mismatches\":%d,"
        "\"compatibility\":{\"server_version\":\"%s\",\"startup_routes_checked\":true,"
        "\"native_pin_route_canary\":true},"
        "\"optimistic_fallbacks\":{\"scope\":\"current_backend\",\"elements\":" UINT64_FORMAT
        ",\"neighbors\":" UINT64_FORMAT "}",
        calico_cluster_status(), calico_local_active ? "true" : "false",
        NBuffers, mapped, pg_atomic_read_u32(next_region), arena_size,
        writer_locked, no_optimistic, mismatches, PG_VERSION,
        calico_optimistic_element_fallbacks, calico_optimistic_neighbor_fallbacks);
    calico_append_activation_statistics(&json);
    appendStringInfoChar(&json, '}');
    PG_RETURN_DATUM(DirectFunctionCall1(jsonb_in, CStringGetDatum(json.data)));
}

BufferDesc *
CalicoLookupAndOptimisticPin(SMgrRelation smgr,
							ForkNumber forkNum,
							BlockNumber blockNum,
							bool *foundPtr,
							uint64 *versionOut,
							uint32 *frameIdOut,
							void **entryPtrOut)
{
    /* Local buffer tags have a backend namespace absent from shared tags. */
    if (SmgrIsTemp(smgr))
    {
        *foundPtr = false;
        *versionOut = 0;
        *frameIdOut = 0;
        *entryPtrOut = NULL;
        return NULL;
    }
    return calico_optimistic_lookup(smgr, forkNum, blockNum,
                                                 foundPtr, versionOut, frameIdOut,
                                                 entryPtrOut);
}
bool
CalicoLookupAndOptimisticValidate(void *entryPtr, uint64 expectedVersion)
{
    return calico_optimistic_validate(entryPtr, expectedVersion);
}
void
CalicoLookupAndOptimisticPinBatch(SMgrRelation smgr,
                                   ForkNumber forkNum,
                                   const BlockNumber *blockNums,
                                   int count,
                                   uint32 *frameIdsOut,
                                   bool *foundOut)
{
    const RelFileLocator *rlocator = &smgr->smgr_rlocator.locator;
    SecondLevelTable second_level_table;
    BufferTag tag;
    CalicoTranslationEntry *first_entry;
    uint32 base_mask;
    int i;

    if (unlikely(!calico_local_active || count <= 0 || SmgrIsTemp(smgr)))
    {
        for (i = 0; i < count; i++)
            foundOut[i] = false;
        return;
    }
    InitBufferTag(&tag, rlocator, forkNum, blockNums[0]);
    first_entry = calico_entry_for(&tag, false);
    if (unlikely(!first_entry))
    {
        for (i = 0; i < count; i++)
        {
            tag.blockNum = blockNums[i];
            CalicoTranslationEntry *te = calico_entry_for(&tag, false);
            uint64 word = te ? calico_load_entry(te) : 0;

            frameIdsOut[i] = CalicoGetFrameId(word);
            foundOut[i] = (word & CALICO_VALID) != 0;
        }
        return;
    }
    second_level_table = first_entry - (blockNums[0] & CALICO_REGION_MASK);
    base_mask = (blockNums[0] >> 24) << 24;

    /* Four-way unrolled loop for ILP/MLP without pre-scan or foundOut zeroing. */
    i = 0;
    for (; i + 3 < count; i += 4)
    {
        BlockNumber b0 = blockNums[i + 0];
        BlockNumber b1 = blockNums[i + 1];
        BlockNumber b2 = blockNums[i + 2];
        BlockNumber b3 = blockNums[i + 3];

        if (likely((((b0 ^ base_mask) | (b1 ^ base_mask) |
                     (b2 ^ base_mask) | (b3 ^ base_mask)) >> 24) == 0))
        {
            CalicoTranslationEntry *entry_ptr0 = &second_level_table[b0 & CALICO_REGION_MASK];
            CalicoTranslationEntry *entry_ptr1 = &second_level_table[b1 & CALICO_REGION_MASK];
            CalicoTranslationEntry *entry_ptr2 = &second_level_table[b2 & CALICO_REGION_MASK];
            CalicoTranslationEntry *entry_ptr3 = &second_level_table[b3 & CALICO_REGION_MASK];
            uint64 entry_val0 = calico_load_entry(entry_ptr0);
            uint64 entry_val1 = calico_load_entry(entry_ptr1);
            uint64 entry_val2 = calico_load_entry(entry_ptr2);
            uint64 entry_val3 = calico_load_entry(entry_ptr3);

            frameIdsOut[i + 0] = CalicoGetFrameId(entry_val0);
            frameIdsOut[i + 1] = CalicoGetFrameId(entry_val1);
            frameIdsOut[i + 2] = CalicoGetFrameId(entry_val2);
            frameIdsOut[i + 3] = CalicoGetFrameId(entry_val3);
            foundOut[i + 0] = (entry_val0 & CALICO_VALID) != 0;
            foundOut[i + 1] = (entry_val1 & CALICO_VALID) != 0;
            foundOut[i + 2] = (entry_val2 & CALICO_VALID) != 0;
            foundOut[i + 3] = (entry_val3 & CALICO_VALID) != 0;
        }
        else
        {
            for (int k = 0; k < 4; k++)
            {
                tag.blockNum = blockNums[i + k];
                CalicoTranslationEntry *te = calico_entry_for(&tag, false);
                uint64 word = te ? calico_load_entry(te) : 0;

                frameIdsOut[i + k] = CalicoGetFrameId(word);
                foundOut[i + k] = (word & CALICO_VALID) != 0;
            }
        }
    }

    /* Handle remaining elements (0-3 items) */
    for (; i < count; i++)
    {
        BlockNumber b = blockNums[i];
        uint64 word;

        if (likely((b >> 24) == (base_mask >> 24)))
            word = calico_load_entry(&second_level_table[b & CALICO_REGION_MASK]);
        else
        {
            tag.blockNum = b;
            CalicoTranslationEntry *te = calico_entry_for(&tag, false);
            word = te ? calico_load_entry(te) : 0;
        }
        frameIdsOut[i] = CalicoGetFrameId(word);
        foundOut[i] = (word & CALICO_VALID) != 0;
    }
}
