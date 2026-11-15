/*
 * Dynamic array-based buffer translation (Design 1: zero-preallocation extents
 * with relocatable cross-process tokens).
 *
 * A prefix is (tablespace, database, relation, fork, block >> CALICO_REGION_BITS).
 * With CALICO_REGION_BITS = 26, each second-level array has 2^26 entries
 * (512 MiB) and spans 512 GiB of relation blocks so large tables and HNSW
 * indexes stay inside a single cached array pointer with zero prefix misses.
 *
 * Memory & VM architecture:
 *  - Fixed PostgreSQL shared memory stays tiny (~1 MiB + calico_frame_entries):
 *    128 partitioned prefix LWLocks (+1 extent allocator LWLock), a 65,536-slot
 *    bucket head array (prefix_buckets[]), and CalicoPoolControl. No monolithic
 *    1 TiB arena or rigid HASH_FIXED_SIZE prefix table is reserved at startup,
 *    so vm.overcommit_memory=2 and strict RLIMIT_AS setups start cleanly with
 *    zero virtual translation bloat and no calico.max_regions cap.
 *  - Translation storage grows on demand using POSIX shared memory extents
 *    (/dev/shm via shm_open + sparse ftruncate + mmap + close):
 *      * 16 MiB metadata extents pack dynamic CalicoPrefixNode directory nodes.
 *      * 512 MiB array extents hold one second-level translation array each.
 *    Untouched pages in a 512 MiB array allocate 0 bytes of Committed_AS and
 *    0 physical tmpfs pages until read or written.
 *  - Cross-process pointers never assume identical ASLR addresses across
 *    processes. All shared structures (prefix_buckets[], CalicoPrefixNode, and
 *    calico_frame_entries[]) store 64-bit relocatable tokens:
 *      (1-based ext_id << 48) | byte_offset.
 *    Postmaster never mmaps dynamic extents, so forked children start identically
 *    and lazily attach each extent once on first touch with raw POSIX syscalls
 *    (zero palloc / zero PG locks, safe in critical sections). Backends resolve
 *    and cache their own local raw VAs in CalicoPathCache and CalicoReadHandle,
 *    preserving zero-indirection HNSW reads.
 */
#include "postgres.h"
#include "calico_internal.h"
#include "common/hashfn.h"
#include "miscadmin.h"
#include "storage/ipc.h"
#include "storage/shmem.h"
#include "storage/lwlock.h"
#include "utils/hsearch.h"
#include "fmgr.h"
#include "utils/builtins.h"
#include "utils/timestamp.h"
#include "lib/stringinfo.h"
#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <signal.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#define REGION_BITS         CALICO_REGION_BITS
#define REGION_ENTRIES      CALICO_REGION_ENTRIES
#define REGION_BYTES        CALICO_REGION_BYTES
#define PREFIX_BUCKETS      65536U
#define PREFIX_PARTITIONS   128U
#define ALLOC_LOCK_INDEX    PREFIX_PARTITIONS
#define META_EXTENT_BYTES   ((Size) (16U * 1024U * 1024U))
#define META_NODE_STRIDE    64U

/*
 * Dynamically allocated directory node stored inside 16 MiB metadata extents.
 * Next-chain and second-level array references use relocatable tokens rather
 * than virtual addresses because sibling backends mmap extents independently.
 */
typedef struct CalicoPrefixNode
{
    CalicoPrefix     key;
    uint32           hashcode;
    pg_atomic_uint64 array_tok;
    pg_atomic_uint64 next_tok;
} CalicoPrefixNode;

StaticAssertDecl(sizeof(CalicoPrefix) == 20,
                 "CalicoPrefix must have no compiler padding for hashing");
StaticAssertDecl(sizeof(CalicoPrefixNode) <= META_NODE_STRIDE,
                 "CalicoPrefixNode exceeds metadata bump stride");

/*
 * Fixed-size control header in PostgreSQL shared memory coordinating on-demand
 * /dev/shm extent creation and cross-process discovery without barriers.
 *
 * Lock hierarchy: prefix_locks[partition] -> alloc_lock (strictly one-way).
 */
typedef struct CalicoPoolControl
{
    uint64           data_dir_id;
    uint32           pm_pid;
    uint32           generation;
    uint32           cur_meta_ext;
    uint64           cur_meta_off;
    pg_atomic_uint32 num_extents;
    pg_atomic_uint32 num_arrays;
    pg_atomic_uint64 ext_bytes[CALICO_MAX_EXTENTS];
} CalicoPoolControl;

/*
 * Sorted per-process interval index mapping attached local VA ranges back to
 * extent IDs during lock/pin reverse binding (e.g., inside LockBufHdr spinlocks)
 * without allocating heap memory or taking shared locks.
 */
typedef struct CalicoLocalVaRange
{
    uintptr_t start;
    uintptr_t end;
    uint32    ext_id;
} CalicoLocalVaRange;

typedef struct NativeMapEntry
{
    BufferTag tag;
    int frame;
} NativeMapEntry;

static HTAB **native_map;
static int (*native_lookup)(BufferTag *, uint32);
static int (*native_insert)(BufferTag *, uint32, int);
static void (*native_delete)(BufferTag *, uint32);

static CalicoPoolControl *pool_ctrl;
static pg_atomic_uint64 *prefix_buckets;
static LWLock *alloc_lock;
static LWLockPadded *prefix_locks;

pg_atomic_uint64 *calico_frame_entries;
char *calico_local_extents[CALICO_MAX_EXTENTS];
static CalicoLocalVaRange local_va_ranges[CALICO_MAX_EXTENTS];
static uint32 num_local_va_ranges;
static uint32 local_restart_seq;

CalicoPathCache calico_path_cache;
/* Keep the inline primary hit unchanged. A bounded backend-local victim
 * cache avoids directory locks when operations alternate among relations.
 * Cache local VA addresses + base tokens; entries and frames are still
 * checked afresh. */
#define RECENT_PATH_COUNT 4
static CalicoPathCache recent_paths[RECENT_PATH_COUNT];
static unsigned int next_recent_path;

bool calico_local_active;
uint64 calico_optimistic_element_fallbacks;
uint64 calico_optimistic_neighbor_fallbacks;

static uint64
calico_compute_datadir_id(void)
{
    struct stat st;
    const char *path = (DataDir && DataDir[0] != '\0') ? DataDir : ".";
    uint64 vals[2];

    if (stat(path, &st) == 0)
    {
        vals[0] = (uint64) st.st_dev;
        vals[1] = (uint64) st.st_ino;
    }
    else
    {
        vals[0] = (uint64) MyProcPid;
        vals[1] = 0;
    }
    return hash_bytes_extended((const unsigned char *) vals, sizeof(vals),
                               UINT64CONST(0x43414c49434f4449));
}

static inline void
calico_format_shm_name(char *buf, Size bufsize, uint64 dir_id,
                       uint32 pm_pid, uint32 gen, uint32 ext_id)
{
    snprintf(buf, bufsize, "/calico.%016llx.%u.%u.%u",
             (unsigned long long) dir_id, pm_pid, gen, ext_id);
}

static void
calico_record_local_extent(uint32 ext_id, void *addr, Size len)
{
    uintptr_t start = (uintptr_t) addr;
    uintptr_t end = start + (uintptr_t) len;
    uint32 pos = num_local_va_ranges;

    Assert(ext_id < CALICO_MAX_EXTENTS);
    Assert(num_local_va_ranges < CALICO_MAX_EXTENTS);
    calico_local_extents[ext_id] = (char *) addr;

    /* Maintain ascending start order for lock-free binary search in bind_frame. */
    while (pos > 0 && local_va_ranges[pos - 1].start > start)
    {
        local_va_ranges[pos] = local_va_ranges[pos - 1];
        pos--;
    }
    local_va_ranges[pos].start = start;
    local_va_ranges[pos].end = end;
    local_va_ranges[pos].ext_id = ext_id;
    num_local_va_ranges++;
}

static void
calico_reset_local_mappings(void)
{
    for (uint32 i = 0; i < num_local_va_ranges; i++)
    {
        void *addr = (void *) local_va_ranges[i].start;
        Size len = (Size) (local_va_ranges[i].end - local_va_ranges[i].start);
        uint32 ext_id = local_va_ranges[i].ext_id;

        if (addr && len > 0)
            (void) munmap(addr, len);
        if (ext_id < CALICO_MAX_EXTENTS)
            calico_local_extents[ext_id] = NULL;
    }
    num_local_va_ranges = 0;
    calico_path_cache.valid = false;
    calico_path_cache.array = NULL;
    calico_path_cache.array_tok = 0;
    memset(recent_paths, 0, sizeof(recent_paths));
    next_recent_path = 0;
}

static void
calico_unlink_live_extents(void)
{
    uint32 n;

    if (!pool_ctrl)
        return;

    n = pg_atomic_read_u32(&pool_ctrl->num_extents);
    for (uint32 i = 0; i < n && i < CALICO_MAX_EXTENTS; i++)
    {
        char name[128];

        if (pg_atomic_read_u64(&pool_ctrl->ext_bytes[i]) == 0)
            continue;
        calico_format_shm_name(name, sizeof(name),
                               pool_ctrl->data_dir_id, pool_ctrl->pm_pid,
                               pool_ctrl->generation, i);
        (void) shm_unlink(name);
    }
}

static void
calico_pool_on_shmem_exit(int code, Datum arg)
{
    /*
     * Only the postmaster tears down /dev/shm extents; child backends exit
     * independently while sibling processes continue sharing active extents.
     */
    if (pool_ctrl && !IsUnderPostmaster && MyProcPid == (int) pool_ctrl->pm_pid)
    {
        calico_reset_local_mappings();
        calico_unlink_live_extents();
    }
}

/*
 * Reclaim orphaned /dev/shm/calico.* files left behind by earlier crashes
 * (including SIGKILL) or an earlier generation of this same postmaster PID.
 */
static void
calico_cleanup_stale_shm(uint64 dir_id, uint32 current_pid, uint32 current_gen)
{
    char prefix[40];
    size_t prefix_len;
    DIR *dir;
    struct dirent *de;

    snprintf(prefix, sizeof(prefix), "calico.%016llx.", (unsigned long long) dir_id);
    prefix_len = strlen(prefix);
    dir = opendir("/dev/shm");
    if (!dir)
        return;

    while ((de = readdir(dir)) != NULL)
    {
        unsigned int f_pid = 0;
        unsigned int f_gen = 0;
        unsigned int f_ext = 0;
        char full[160];

        if (strncmp(de->d_name, prefix, prefix_len) != 0)
            continue;
        if (sscanf(de->d_name + prefix_len, "%u.%u.%u", &f_pid, &f_gen, &f_ext) != 3)
            continue;
        if (f_pid == current_pid)
        {
            if (f_gen == current_gen)
                continue;
        }
        else if (kill((pid_t) f_pid, 0) == 0 || errno != ESRCH)
        {
            continue;
        }

        snprintf(full, sizeof(full), "/%s", de->d_name);
        (void) shm_unlink(full);
    }
    closedir(dir);
}

/*
 * Lazily map an already-published extent into this backend's virtual address
 * space. Uses only stack buffers and POSIX syscalls (zero PG locks, zero palloc)
 * so calling it during buffer eviction or critical sections cannot deadlock.
 */
static pg_noinline char *
calico_attach_extent(uint32 ext_id)
{
    Size len;
    char name[128];
    int fd;
    void *addr;
    int save_errno;

    if (unlikely(ext_id >= CALICO_MAX_EXTENTS))
        elog(FATAL, "Calico invalid extent identifier %u", ext_id);

    /* Ensure caller's token load completes before reading ext_bytes[ext_id]. */
    pg_read_barrier();
    if (calico_local_extents[ext_id] != NULL)
        return calico_local_extents[ext_id];

    len = (Size) pg_atomic_read_u64(&pool_ctrl->ext_bytes[ext_id]);
    if (unlikely(len == 0))
        elog(FATAL, "Calico referenced unpublished extent %u", ext_id);

    calico_format_shm_name(name, sizeof(name),
                           pool_ctrl->data_dir_id, pool_ctrl->pm_pid,
                           pool_ctrl->generation, ext_id);
    do
    {
        fd = shm_open(name, O_RDWR, 0);
    } while (fd < 0 && errno == EINTR);
    if (fd < 0)
        elog(ERROR, "Calico cannot open POSIX shared memory \"%s\": %m", name);

    addr = mmap(NULL, len, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    save_errno = errno;
    close(fd);
    if (addr == MAP_FAILED)
    {
        errno = save_errno;
        elog(ERROR, "Calico cannot map extent \"%s\" (%zu bytes): %m", name, len);
    }

    (void) madvise(addr, len, MADV_NOHUGEPAGE);
    calico_record_local_extent(ext_id, addr, len);
    return (char *) addr;
}

static inline __attribute__((always_inline)) void *
calico_resolve_token(uint64 tok)
{
    uint32 ext_id;
    char *base;

    if (unlikely(tok == 0))
        return NULL;

    ext_id = CalicoTokenExtId(tok);
    Assert(ext_id < CALICO_MAX_EXTENTS);
    base = calico_local_extents[ext_id];
    if (unlikely(base == NULL))
        base = calico_attach_extent(ext_id);
    return (void *) (base + CalicoTokenOffset(tok));
}

/*
 * Allocate and publish a fresh POSIX shm extent while holding alloc_lock.
 * ftruncate() leaves physical tmpfs pages uncommitted until actually touched;
 * MADV_NOHUGEPAGE keeps demand-fault granularity at 4 KiB.
 */
static uint32
calico_create_extent(Size bytes)
{
    uint32 ext_id = pg_atomic_read_u32(&pool_ctrl->num_extents);
    char name[128];
    int fd;
    void *addr;
    int save_errno;

    if (ext_id >= CALICO_MAX_EXTENTS)
        ereport(ERROR, (errcode(ERRCODE_OUT_OF_MEMORY),
                        errmsg("Calico translation extent table exhausted (%u extents)",
                               CALICO_MAX_EXTENTS)));

    calico_format_shm_name(name, sizeof(name),
                           pool_ctrl->data_dir_id, pool_ctrl->pm_pid,
                           pool_ctrl->generation, ext_id);
    do
    {
        fd = shm_open(name, O_RDWR | O_CREAT | O_EXCL, 0600);
    } while (fd < 0 && errno == EINTR);
    if (fd < 0)
        ereport(ERROR, (errcode(ERRCODE_OUT_OF_MEMORY),
                        errmsg("Calico cannot create POSIX shared memory \"%s\": %m",
                               name)));

    if (ftruncate(fd, (off_t) bytes) != 0)
    {
        save_errno = errno;
        close(fd);
        (void) shm_unlink(name);
        errno = save_errno;
        ereport(ERROR, (errcode(ERRCODE_OUT_OF_MEMORY),
                        errmsg("Calico cannot size extent \"%s\" to %zu bytes: %m",
                               name, bytes)));
    }

    addr = mmap(NULL, bytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    save_errno = errno;
    close(fd);
    if (addr == MAP_FAILED)
    {
        (void) shm_unlink(name);
        errno = save_errno;
        ereport(ERROR, (errcode(ERRCODE_OUT_OF_MEMORY),
                        errmsg("Calico cannot map extent \"%s\" (%zu bytes): %m",
                               name, bytes)));
    }

    (void) madvise(addr, bytes, MADV_NOHUGEPAGE);
    calico_record_local_extent(ext_id, addr, bytes);

    /*
     * Publish extent metadata before writing any token referencing ext_id or
     * advancing num_extents so concurrent attachers always see ext_bytes > 0.
     */
    pg_atomic_write_u64(&pool_ctrl->ext_bytes[ext_id], (uint64) bytes);
    pg_write_barrier();
    pg_atomic_write_u32(&pool_ctrl->num_extents, ext_id + 1);
    return ext_id;
}

/*
 * Convert a backend-local translation entry pointer back into its relocatable
 * shared token. Used when populating calico_frame_entries[buf_id], including
 * inside LockBufHdr() spinlocks (no PG locks, no syscalls, no allocations).
 */
static inline __attribute__((always_inline)) uint64
calico_token_for_local_entry(const pg_atomic_uint64 *entry)
{
    uintptr_t addr = (uintptr_t) entry;
    uint32 lo, hi;

    if (unlikely(addr == 0))
        return 0;

    if (likely(calico_path_cache.valid))
    {
        uintptr_t base = (uintptr_t) calico_path_cache.array;
        uintptr_t diff = addr - base;

        if (likely(addr >= base && diff < REGION_BYTES))
            return calico_path_cache.array_tok + (uint64) diff;
    }

    for (int i = 0; i < RECENT_PATH_COUNT; i++)
    {
        if (recent_paths[i].valid)
        {
            uintptr_t base = (uintptr_t) recent_paths[i].array;
            uintptr_t diff = addr - base;

            if (addr >= base && diff < REGION_BYTES)
                return recent_paths[i].array_tok + (uint64) diff;
        }
    }

    lo = 0;
    hi = num_local_va_ranges;
    while (lo < hi)
    {
        uint32 mid = lo + (hi - lo) / 2;

        if (addr < local_va_ranges[mid].start)
            hi = mid;
        else if (addr >= local_va_ranges[mid].end)
            lo = mid + 1;
        else
            return CalicoMakeToken(local_va_ranges[mid].ext_id,
                                   (uint64) (addr - local_va_ranges[mid].start));
    }

    elog(ERROR, "Calico could not encode translation entry address to extent token");
    return 0;
}

void
calico_pool_request(void)
{
    RequestAddinShmemSpace(MAXALIGN(sizeof(CalicoPoolControl)) +
                          MAXALIGN((Size) PREFIX_BUCKETS * sizeof(pg_atomic_uint64)) +
                          MAXALIGN((Size) NBuffers * sizeof(pg_atomic_uint64)) + 4096);
    RequestNamedLWLockTranche("calico_prefix", PREFIX_PARTITIONS + 1);
}

void
calico_pool_startup(void)
{
    bool found;
    LWLockPadded *tranche;
    uint64 dir_id = calico_compute_datadir_id();
    uint32 pm_pid = (uint32) MyProcPid;
    uint32 gen;

    /* Unmap any local mappings if reinitializing inside postmaster crash reset. */
    calico_reset_local_mappings();

    gen = ((uint32) GetCurrentTimestamp()) ^ (++local_restart_seq);
    if (gen == 0)
        gen = 1;

    pool_ctrl = ShmemInitStruct("Calico Pool Control", sizeof(*pool_ctrl), &found);
    prefix_buckets = ShmemInitStruct("Calico Prefix Buckets",
                                     (Size) PREFIX_BUCKETS * sizeof(*prefix_buckets),
                                     &found);
    calico_frame_entries = ShmemInitStruct("Calico Frame Entries",
                                           (Size) NBuffers * sizeof(*calico_frame_entries),
                                           &found);
    tranche = GetNamedLWLockTranche("calico_prefix");
    prefix_locks = tranche;
    alloc_lock = &tranche[ALLOC_LOCK_INDEX].lock;

    if (!found)
    {
        memset(pool_ctrl, 0, sizeof(*pool_ctrl));
        pool_ctrl->data_dir_id = dir_id;
        pool_ctrl->pm_pid = pm_pid;
        pool_ctrl->generation = gen;
        pool_ctrl->cur_meta_ext = PG_UINT32_MAX;
        pool_ctrl->cur_meta_off = 0;
        pg_atomic_init_u32(&pool_ctrl->num_extents, 0);
        pg_atomic_init_u32(&pool_ctrl->num_arrays, 0);
        for (uint32 i = 0; i < CALICO_MAX_EXTENTS; i++)
            pg_atomic_init_u64(&pool_ctrl->ext_bytes[i], 0);
        for (uint32 i = 0; i < PREFIX_BUCKETS; i++)
            pg_atomic_init_u64(&prefix_buckets[i], 0);
        memset(calico_frame_entries, 0, (Size) NBuffers * sizeof(*calico_frame_entries));
    }

    calico_cleanup_stale_shm(dir_id, pm_pid, gen);
    /*
     * PostgreSQL drains and clears on_shmem_exit_list during postmaster crash
     * recovery before re-running shmem_startup_hook, so register unconditionally
     * on each postmaster shared-memory startup instead of gating on a static flag.
     */
    if (!found && !IsUnderPostmaster)
        on_shmem_exit(calico_pool_on_shmem_exit, 0);
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

static inline void
calico_cache_path(const CalicoPrefix *key, pg_atomic_uint64 *array, uint64 array_tok)
{
    if (calico_path_cache.valid)
    {
        recent_paths[next_recent_path] = calico_path_cache;
        next_recent_path = (next_recent_path + 1) % RECENT_PATH_COUNT;
    }
    calico_path_cache.prefix = *key;
    calico_path_cache.array = array;
    calico_path_cache.array_tok = array_tok;
    calico_path_cache.valid = true;
}

static CalicoPrefixNode *
calico_find_prefix_in_bucket(uint32 bucket, const CalicoPrefix *key, uint32 hashcode)
{
    uint64 node_tok = pg_atomic_read_u64(&prefix_buckets[bucket]);

    while (node_tok != 0)
    {
        CalicoPrefixNode *node = (CalicoPrefixNode *) calico_resolve_token(node_tok);

        if (node->hashcode == hashcode &&
            memcmp(&node->key, key, sizeof(*key)) == 0)
            return node;
        node_tok = pg_atomic_read_u64(&node->next_tok);
    }
    return NULL;
}

pg_atomic_uint64 *
calico_entry_for_miss(const BufferTag *tag, bool create)
{
    CalicoPrefix key;
    CalicoPrefixNode *node;
    pg_atomic_uint64 *array;
    uint64 array_tok = 0;
    uint32 hashcode;
    uint32 bucket;
    LWLock *lock;

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

    hashcode = hash_bytes((const unsigned char *) &key, sizeof(key));
    bucket = hashcode & (PREFIX_BUCKETS - 1);
    lock = &prefix_locks[bucket & (PREFIX_PARTITIONS - 1)].lock;

    LWLockAcquire(lock, LW_SHARED);
    node = calico_find_prefix_in_bucket(bucket, &key, hashcode);
    if (node)
        array_tok = pg_atomic_read_u64(&node->array_tok);
    LWLockRelease(lock);

    if (!node && create)
    {
        LWLockAcquire(lock, LW_EXCLUSIVE);
        node = calico_find_prefix_in_bucket(bucket, &key, hashcode);
        if (node)
        {
            array_tok = pg_atomic_read_u64(&node->array_tok);
        }
        else
        {
            uint32 meta_ext;
            uint64 meta_off;
            uint32 arr_ext;
            uint64 head_tok;
            uint64 node_tok;
            CalicoPrefixNode *new_node;

            LWLockAcquire(alloc_lock, LW_EXCLUSIVE);
            meta_ext = pool_ctrl->cur_meta_ext;
            meta_off = pool_ctrl->cur_meta_off;
            if (meta_ext == PG_UINT32_MAX ||
                meta_off + META_NODE_STRIDE > META_EXTENT_BYTES)
            {
                meta_ext = calico_create_extent(META_EXTENT_BYTES);
                meta_off = 0;
                pool_ctrl->cur_meta_ext = meta_ext;
                pool_ctrl->cur_meta_off = 0;
            }

            /* Create the 512 MiB array extent before advancing meta_off. */
            arr_ext = calico_create_extent(REGION_BYTES);
            pool_ctrl->cur_meta_off = meta_off + META_NODE_STRIDE;
            pg_atomic_fetch_add_u32(&pool_ctrl->num_arrays, 1);
            LWLockRelease(alloc_lock);

            array_tok = CalicoMakeToken(arr_ext, 0);
            node_tok = CalicoMakeToken(meta_ext, meta_off);
            /* Attach meta_ext if a sibling backend created this metadata extent earlier. */
            new_node = (CalicoPrefixNode *) calico_resolve_token(node_tok);
            head_tok = pg_atomic_read_u64(&prefix_buckets[bucket]);

            new_node->key = key;
            new_node->hashcode = hashcode;
            pg_atomic_init_u64(&new_node->array_tok, array_tok);
            pg_atomic_init_u64(&new_node->next_tok, head_tok);
            pg_write_barrier();
            pg_atomic_write_u64(&prefix_buckets[bucket], node_tok);
        }
        LWLockRelease(lock);
    }

    if (array_tok == 0)
        return NULL;

    array = (pg_atomic_uint64 *) calico_resolve_token(array_tok);
    calico_cache_path(&key, array, array_tok);
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
    calico_store_entry(&calico_frame_entries[buf->buf_id],
                       calico_token_for_local_entry(entry));
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
        calico_store_entry(&calico_frame_entries[frame],
                           calico_token_for_local_entry(entry));
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
        uint64 tok = calico_load_entry(&calico_frame_entries[buf_id]);
        pg_atomic_uint64 *entry = tok ? calico_resolve_token(tok) : NULL;
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
        uint64 tok = calico_load_entry(&calico_frame_entries[buf->buf_id]);
        pg_atomic_uint64 *entry = tok ? calico_resolve_token(tok) : NULL;
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
        uint64 tok = calico_load_entry(&calico_frame_entries[buf->buf_id]);
        pg_atomic_uint64 *entry = tok ? calico_resolve_token(tok) : NULL;
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
    uint64 tok;
    pg_atomic_uint64 *entry;
    uint64 old;

    if (!calico_local_active)
        return;
    tok = calico_load_entry(&calico_frame_entries[buf->buf_id]);
    entry = tok ? calico_resolve_token(tok) : NULL;
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
    uint64 tok;
    pg_atomic_uint64 *entry;
    uint64 old;

    if (!calico_local_active)
        return;
    tok = calico_load_entry(&calico_frame_entries[buf->buf_id]);
    entry = tok ? calico_resolve_token(tok) : NULL;
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
        calico_store_entry(&calico_frame_entries[buf->buf_id],
                           calico_token_for_local_entry(entry));
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
    /* Clear any partial seed through its reverse tokens before rebuilding.
     * Allocated extents stay registered for this postmaster lifetime, so
     * num_extents == 0 guarantees no earlier partial seed ran. */
    if (pg_atomic_read_u32(&pool_ctrl->num_extents) != 0)
    {
        for (int i = 0; i < NBuffers; i++)
        {
            uint64 tok;
            if ((i & 1023) == 0)
                calico_check_activation_deadline();
            tok = calico_load_entry(&calico_frame_entries[i]);
            if (tok)
            {
                pg_atomic_uint64 *old = (pg_atomic_uint64 *) calico_resolve_token(tok);
                if (old)
                    calico_store_entry(old, 0);
            }
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
    uint32 num_extents;
    uint32 num_arrays;
    uint64 virtual_bytes = 0;
    StringInfoData json;
    if (!pool_ctrl || !prefix_buckets)
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
    num_extents = pg_atomic_read_u32(&pool_ctrl->num_extents);
    num_arrays = pg_atomic_read_u32(&pool_ctrl->num_arrays);
    for (uint32 ext = 0; ext < num_extents && ext < CALICO_MAX_EXTENTS; ext++)
        virtual_bytes += pg_atomic_read_u64(&pool_ctrl->ext_bytes[ext]);
    initStringInfo(&json);
    appendStringInfo(&json,
        "{\"mode\":\"%s\",\"backend_active\":%s,\"frames\":%d,"
        "\"mapped_frames\":%d,\"regions\":%u,\"extents\":%u,"
        "\"region_bits\":%u,\"array_bytes\":%zu,\"virtual_bytes\":%zu,"
        "\"writer_locked_entries\":%d,\"no_optimistic_entries\":%d,\"mapping_mismatches\":%d,"
        "\"compatibility\":{\"server_version\":\"%s\",\"startup_routes_checked\":true,"
        "\"native_pin_route_canary\":true},"
        "\"optimistic_fallbacks\":{\"scope\":\"current_backend\",\"elements\":" UINT64_FORMAT
        ",\"neighbors\":" UINT64_FORMAT "}",
        calico_cluster_status(), calico_local_active ? "true" : "false",
        NBuffers, mapped, num_arrays, num_extents,
        REGION_BITS, (size_t) REGION_BYTES, (size_t) virtual_bytes,
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
    base_mask = (blockNums[0] >> REGION_BITS) << REGION_BITS;

    /* Four-way unrolled loop for ILP/MLP without pre-scan or foundOut zeroing. */
    i = 0;
    for (; i + 3 < count; i += 4)
    {
        BlockNumber b0 = blockNums[i + 0];
        BlockNumber b1 = blockNums[i + 1];
        BlockNumber b2 = blockNums[i + 2];
        BlockNumber b3 = blockNums[i + 3];

        if (likely((((b0 ^ base_mask) | (b1 ^ base_mask) |
                     (b2 ^ base_mask) | (b3 ^ base_mask)) >> REGION_BITS) == 0))
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

        if (likely((b >> REGION_BITS) == (base_mask >> REGION_BITS)))
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
