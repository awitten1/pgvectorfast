/*-------------------------------------------------------------------------
 *
 * calico_bufmgr19.inc
 *	  buffer manager interface routines
 *
 * Portions Copyright (c) 1996-2026, PostgreSQL Global Development Group
 * Portions Copyright (c) 1994, Regents of the University of California
 *
 *
 * IDENTIFICATION
 *	  src/calico_bufmgr19.inc
 *
 *-------------------------------------------------------------------------
 */
/* Read, extension and content-lock paths used by Calico. PostgreSQL provides
 * the other public buffer APIs, native pin storage and AIO callbacks. */
#include "postgres.h"
/* Own only redirected entry points; other public calls use PostgreSQL. */
#define ReadBuffer calico_core_ReadBuffer
#define ReadBufferExtended calico_core_ReadBufferExtended
#define ReadBufferWithoutRelcache calico_core_ReadBufferWithoutRelcache
#define ExtendBufferedRel calico_core_ExtendBufferedRel
#define ExtendBufferedRelBy calico_core_ExtendBufferedRelBy
#define ExtendBufferedRelTo calico_core_ExtendBufferedRelTo
#define StartReadBuffers calico_core_StartReadBuffers
#define StartReadBuffer calico_core_StartReadBuffer
#define ReleaseAndReadBuffer calico_core_ReleaseAndReadBuffer
#define CreateAndCopyRelationData calico_core_CreateAndCopyRelationData
#define FlushOneBuffer calico_core_FlushOneBuffer
#define UnlockReleaseBuffer calico_core_UnlockReleaseBuffer
#define UnlockBuffer calico_core_UnlockBuffer
#define LockBufferInternal calico_core_LockBufferInternal
#define ConditionalLockBuffer calico_core_ConditionalLockBuffer
#define LockBufferForCleanup calico_core_LockBufferForCleanup
#define ConditionalLockBufferForCleanup calico_core_ConditionalLockBufferForCleanup
#define MarkDirtyUnpinnedBuffer calico_core_MarkDirtyUnpinnedBuffer
#define MarkDirtyRelUnpinnedBuffers calico_core_MarkDirtyRelUnpinnedBuffers
#define MarkDirtyAllUnpinnedBuffers calico_core_MarkDirtyAllUnpinnedBuffers
#define BufTableLookup calico_table_lookup
#define BufTableInsert calico_table_insert
#define BufTableDelete calico_table_delete
#include "calico_internal.h"

#include <sys/file.h>
#include <unistd.h>

#include "access/tableam.h"
#include "access/xloginsert.h"
#include "access/xlogutils.h"
#ifdef USE_ASSERT_CHECKING
#include "catalog/pg_tablespace_d.h"
#endif
#include "catalog/storage.h"
#include "catalog/storage_xlog.h"
#include "common/hashfn.h"
#include "executor/instrument.h"
#include "miscadmin.h"
#include "pg_trace.h"
#include "pgstat.h"
#include "postmaster/bgwriter.h"
#include "storage/aio.h"
#include "storage/buf_internals.h"
#include "storage/bufmgr.h"
#include "storage/fd.h"
#include "storage/ipc.h"
#include "storage/lmgr.h"
#include "storage/proc.h"
#include "storage/proclist.h"
#include "storage/procsignal.h"
#include "storage/read_stream.h"
#include "storage/smgr.h"
#include "storage/standby.h"
#include "utils/memdebug.h"
#include "utils/ps_status.h"
#include "utils/rel.h"
#include "utils/resowner.h"
#include "utils/timestamp.h"

/* Inactive backends retain PostgreSQL's read, extension and lock paths. */
static __typeof__(ReadBuffer) *native_ReadBuffer;
static __typeof__(ReadBufferExtended) *native_ReadBufferExtended;
static __typeof__(ReadBufferWithoutRelcache) *native_ReadBufferWithoutRelcache;
static __typeof__(ExtendBufferedRel) *native_ExtendBufferedRel;
static __typeof__(ExtendBufferedRelBy) *native_ExtendBufferedRelBy;
static __typeof__(ExtendBufferedRelTo) *native_ExtendBufferedRelTo;
static __typeof__(StartReadBuffers) *native_StartReadBuffers;
static __typeof__(StartReadBuffer) *native_StartReadBuffer;
static __typeof__(ReleaseAndReadBuffer) *native_ReleaseAndReadBuffer;
static __typeof__(CreateAndCopyRelationData) *native_CreateAndCopyRelationData;
static __typeof__(FlushOneBuffer) *native_FlushOneBuffer;
static __typeof__(UnlockReleaseBuffer) *native_UnlockReleaseBuffer;
static __typeof__(UnlockBuffer) *native_UnlockBuffer;
static __typeof__(LockBufferInternal) *native_LockBufferInternal;
static __typeof__(ConditionalLockBuffer) *native_ConditionalLockBuffer;
static __typeof__(LockBufferForCleanup) *native_LockBufferForCleanup;
static __typeof__(ConditionalLockBufferForCleanup) *native_ConditionalLockBufferForCleanup;
static __typeof__(MarkDirtyUnpinnedBuffer) *native_MarkDirtyUnpinnedBuffer;
static __typeof__(MarkDirtyRelUnpinnedBuffers) *native_MarkDirtyRelUnpinnedBuffers;
static __typeof__(MarkDirtyAllUnpinnedBuffers) *native_MarkDirtyAllUnpinnedBuffers;
#include "utils/wait_event.h"

/* Note: these two macros only work on shared buffers, not local ones! */
#define BufHdrGetBlock(bufHdr)	((Block) (BufferBlocks + ((Size) (bufHdr)->buf_id) * BLCKSZ))
#define BufferGetLSN(bufHdr)	(PageGetLSN(BufHdrGetBlock(bufHdr)))

/* Note: this macro only works on local buffers, not shared ones! */
#define LocalBufHdrGetBlock(bufHdr) \
	LocalBufferBlockPointers[-((bufHdr)->buf_id + 2)]

/*
 * This is separated out from PrivateRefCountEntry to allow for copying all
 * the data members via struct assignment.
 */
typedef struct PrivateRefCountData
{
	/*
	 * How many times has the buffer been pinned by this backend.
	 */
	int32		refcount;

	/*
	 * Is the buffer locked by this backend? BUFFER_LOCK_UNLOCK indicates that
	 * the buffer is not locked.
	 */
	BufferLockMode lockmode;
} PrivateRefCountData;

typedef struct PrivateRefCountEntry
{
	/*
	 * Note that this needs to be same as the entry's corresponding
	 * PrivateRefCountArrayKeys[i], if the entry is stored in the array. We
	 * store it in both places as this is used for the hashtable key and
	 * because it is more convenient (passing around a PrivateRefCountEntry
	 * suffices to identify the buffer) and faster (checking the keys array is
	 * faster when checking many entries, checking the entry is faster if just
	 * checking a single entry).
	 */
	Buffer		buffer;

	char		status;

	PrivateRefCountData data;
} PrivateRefCountEntry;

#define SH_PREFIX refcount
#define SH_ELEMENT_TYPE PrivateRefCountEntry
#define SH_KEY_TYPE Buffer
#define SH_KEY buffer
#define SH_HASH_KEY(tb, key) murmurhash32((uint32) (key))
#define SH_EQUAL(tb, a, b) ((a) == (b))
#define SH_SCOPE static inline
#define SH_DECLARE
#define SH_DEFINE
#include "lib/simplehash.h"

/* 64 bytes, about the size of a cache line on common systems */
#define REFCOUNT_ARRAY_ENTRIES 8

/* GUC variables */
extern bool zero_damaged_pages;
extern bool track_io_timing;

/* local state for LockBufferForCleanup */
static BufferDesc * *native_PinCountWaitBuf;
#define PinCountWaitBuf (*native_PinCountWaitBuf)

/*
 * Backend-Private refcount management:
 *
 * Each buffer also has a private refcount that keeps track of the number of
 * times the buffer is pinned in the current process.  This is so that the
 * shared refcount needs to be modified only once if a buffer is pinned more
 * than once by an individual backend.  It's also used to check that no
 * buffers are still pinned at the end of transactions and when exiting. We
 * also use this mechanism to track whether this backend has a buffer locked,
 * and, if so, in what mode.
 *
 *
 * To avoid - as we used to - requiring an array with NBuffers entries to keep
 * track of local buffers, we use a small sequentially searched array
 * (PrivateRefCountArrayKeys, with the corresponding data stored in
 * PrivateRefCountArray) and an overflow hash table (PrivateRefCountHash) to
 * keep track of backend local pins.
 *
 * Until no more than REFCOUNT_ARRAY_ENTRIES buffers are pinned at once, all
 * refcounts are kept track of in the array; after that, new array entries
 * displace old ones into the hash table. That way a frequently used entry
 * can't get "stuck" in the hashtable while infrequent ones clog the array.
 *
 * Note that in most scenarios the number of pinned buffers will not exceed
 * REFCOUNT_ARRAY_ENTRIES.
 *
 *
 * To enter a buffer into the refcount tracking mechanism first reserve a free
 * entry using ReservePrivateRefCountEntry() and then later, if necessary,
 * fill it with NewPrivateRefCountEntry(). That split lets us avoid doing
 * memory allocations in NewPrivateRefCountEntry() which can be important
 * because in some scenarios it's called with a spinlock held...
 */
static Buffer (*native_PrivateRefCountArrayKeys)[REFCOUNT_ARRAY_ENTRIES];
#define PrivateRefCountArrayKeys (*native_PrivateRefCountArrayKeys)
static struct PrivateRefCountEntry (*native_PrivateRefCountArray)[REFCOUNT_ARRAY_ENTRIES];
#define PrivateRefCountArray (*native_PrivateRefCountArray)
static refcount_hash * *native_PrivateRefCountHash;
#define PrivateRefCountHash (*native_PrivateRefCountHash)
static int32 *native_PrivateRefCountOverflowed;
#define PrivateRefCountOverflowed (*native_PrivateRefCountOverflowed)
static uint32 *native_PrivateRefCountClock;
#define PrivateRefCountClock (*native_PrivateRefCountClock)
static int *native_ReservedRefCountSlot;
#define ReservedRefCountSlot (*native_ReservedRefCountSlot)
static int *native_PrivateRefCountEntryLast;
#define PrivateRefCountEntryLast (*native_PrivateRefCountEntryLast)

static uint32 *native_MaxProportionalPins;
#define MaxProportionalPins (*native_MaxProportionalPins)

static void ReservePrivateRefCountEntry(void);
static PrivateRefCountEntry *GetPrivateRefCountEntry(Buffer buffer, bool do_move);
static inline int32 GetPrivateRefCount(Buffer buffer);
static void ForgetPrivateRefCountEntry(PrivateRefCountEntry *ref);

/* ResourceOwner callbacks to hold in-progress I/Os and buffer pins */
static void ResOwnerReleaseBuffer(Datum res);

/* buffer_io_resowner_desc and its storage remain owned by PostgreSQL. */

/* buffer_resowner_desc and its storage remain owned by PostgreSQL. */

/*
 * Ensure that the PrivateRefCountArray has sufficient space to store one more
 * entry. This has to be called before using NewPrivateRefCountEntry() to fill
 * a new entry - but it's perfectly fine to not use a reserved entry.
 */
static void
ReservePrivateRefCountEntry(void)
{
	/* Already reserved (or freed), nothing to do */
	if (ReservedRefCountSlot != -1)
		return;

	/*
	 * First search for a free entry the array, that'll be sufficient in the
	 * majority of cases.
	 */
	{
		int			i;

		for (i = 0; i < REFCOUNT_ARRAY_ENTRIES; i++)
		{
			if (PrivateRefCountArrayKeys[i] == InvalidBuffer)
			{
				ReservedRefCountSlot = i;

				/*
				 * We could return immediately, but iterating till the end of
				 * the array allows compiler-autovectorization.
				 */
			}
		}

		if (ReservedRefCountSlot != -1)
			return;
	}

	/*
	 * No luck. All array entries are full. Move one array entry into the hash
	 * table.
	 */
	{
		/*
		 * Move entry from the current clock position in the array into the
		 * hashtable. Use that slot.
		 */
		int			victim_slot;
		PrivateRefCountEntry *victim_entry;
		PrivateRefCountEntry *hashent;
		bool		found;

		/* select victim slot */
		victim_slot = PrivateRefCountClock++ % REFCOUNT_ARRAY_ENTRIES;
		victim_entry = &PrivateRefCountArray[victim_slot];
		ReservedRefCountSlot = victim_slot;

		/* Better be used, otherwise we shouldn't get here. */
		Assert(PrivateRefCountArrayKeys[victim_slot] != InvalidBuffer);
		Assert(PrivateRefCountArray[victim_slot].buffer != InvalidBuffer);
		Assert(PrivateRefCountArrayKeys[victim_slot] == PrivateRefCountArray[victim_slot].buffer);

		/* enter victim array entry into hashtable */
		hashent = refcount_insert(PrivateRefCountHash,
								  PrivateRefCountArrayKeys[victim_slot],
								  &found);
		Assert(!found);
		/* move data from the entry in the array to the hash entry */
		hashent->data = victim_entry->data;

		/* clear the now free array slot */
		PrivateRefCountArrayKeys[victim_slot] = InvalidBuffer;
		victim_entry->buffer = InvalidBuffer;

		/* clear the whole data member, just for future proofing */
		memset(&victim_entry->data, 0, sizeof(victim_entry->data));
		victim_entry->data.refcount = 0;
		victim_entry->data.lockmode = BUFFER_LOCK_UNLOCK;

		PrivateRefCountOverflowed++;
	}
}

/*
 * Slow-path for GetPrivateRefCountEntry(). This is big enough to not be worth
 * inlining. This particularly seems to be true if the compiler is capable of
 * auto-vectorizing the code, as that imposes additional stack-alignment
 * requirements etc.
 */
static pg_noinline PrivateRefCountEntry *
GetPrivateRefCountEntrySlow(Buffer buffer, bool do_move)
{
	PrivateRefCountEntry *res;
	int			match = -1;
	int			i;

	/*
	 * First search for references in the array, that'll be sufficient in the
	 * majority of cases.
	 */
	for (i = 0; i < REFCOUNT_ARRAY_ENTRIES; i++)
	{
		if (PrivateRefCountArrayKeys[i] == buffer)
		{
			match = i;
			/* see ReservePrivateRefCountEntry() for why we don't return */
		}
	}

	if (likely(match != -1))
	{
		/* update cache for the next lookup */
		PrivateRefCountEntryLast = match;

		return &PrivateRefCountArray[match];
	}

	/*
	 * By here we know that the buffer, if already pinned, isn't residing in
	 * the array.
	 *
	 * Only look up the buffer in the hashtable if we've previously overflowed
	 * into it.
	 */
	if (PrivateRefCountOverflowed == 0)
		return NULL;

	res = refcount_lookup(PrivateRefCountHash, buffer);

	if (res == NULL)
		return NULL;
	else if (!do_move)
	{
		/* caller doesn't want us to move the hash entry into the array */
		return res;
	}
	else
	{
		/* move buffer from hashtable into the free array slot */
		PrivateRefCountEntry *free;
		PrivateRefCountData data;

		/* Save data and delete from hashtable while res is still valid */
		data = res->data;
		refcount_delete_item(PrivateRefCountHash, res);
		Assert(PrivateRefCountOverflowed > 0);
		PrivateRefCountOverflowed--;

		/* Ensure there's a free array slot */
		ReservePrivateRefCountEntry();

		/* Use up the reserved slot */
		Assert(ReservedRefCountSlot != -1);
		free = &PrivateRefCountArray[ReservedRefCountSlot];
		Assert(PrivateRefCountArrayKeys[ReservedRefCountSlot] == free->buffer);
		Assert(free->buffer == InvalidBuffer);

		/* and fill it */
		free->buffer = buffer;
		free->data = data;
		PrivateRefCountArrayKeys[ReservedRefCountSlot] = buffer;
		/* update cache for the next lookup */
		PrivateRefCountEntryLast = ReservedRefCountSlot;

		ReservedRefCountSlot = -1;

		return free;
	}
}

/*
 * Return the PrivateRefCount entry for the passed buffer.
 *
 * Returns NULL if a buffer doesn't have a refcount entry. Otherwise, if
 * do_move is true, and the entry resides in the hashtable the entry is
 * optimized for frequent access by moving it to the array.
 */
static inline PrivateRefCountEntry *
GetPrivateRefCountEntry(Buffer buffer, bool do_move)
{
	Assert(BufferIsValid(buffer));
	Assert(!BufferIsLocal(buffer));

	/*
	 * It's very common to look up the same buffer repeatedly. To make that
	 * fast, we have a one-entry cache.
	 *
	 * In contrast to the loop in GetPrivateRefCountEntrySlow(), here it
	 * faster to check PrivateRefCountArray[].buffer, as in the case of a hit
	 * fewer addresses are computed and fewer cachelines are accessed. Whereas
	 * in GetPrivateRefCountEntrySlow()'s case, checking
	 * PrivateRefCountArrayKeys saves a lot of memory accesses.
	 */
	if (likely(PrivateRefCountEntryLast != -1) &&
		likely(PrivateRefCountArray[PrivateRefCountEntryLast].buffer == buffer))
	{
		return &PrivateRefCountArray[PrivateRefCountEntryLast];
	}

	/*
	 * The code for the cached lookup is small enough to be worth inlining
	 * into the caller. In the miss case however, that empirically doesn't
	 * seem worth it.
	 */
	return GetPrivateRefCountEntrySlow(buffer, do_move);
}

/*
 * Returns how many times the passed buffer is pinned by this backend.
 *
 * Only works for shared memory buffers!
 */
static inline int32
GetPrivateRefCount(Buffer buffer)
{
	PrivateRefCountEntry *ref;

	Assert(BufferIsValid(buffer));
	Assert(!BufferIsLocal(buffer));

	/*
	 * Not moving the entry - that's ok for the current users, but we might
	 * want to change this one day.
	 */
	ref = GetPrivateRefCountEntry(buffer, false);

	if (ref == NULL)
		return 0;
	return ref->data.refcount;
}

/*
 * Release resources used to track the reference count of a buffer which we no
 * longer have pinned and don't want to pin again immediately.
 */
static void
ForgetPrivateRefCountEntry(PrivateRefCountEntry *ref)
{
	Assert(ref->data.refcount == 0);
	Assert(ref->data.lockmode == BUFFER_LOCK_UNLOCK);

	if (ref >= &PrivateRefCountArray[0] &&
		ref < &PrivateRefCountArray[REFCOUNT_ARRAY_ENTRIES])
	{
		ref->buffer = InvalidBuffer;
		PrivateRefCountArrayKeys[ref - PrivateRefCountArray] = InvalidBuffer;

		/*
		 * Mark the just used entry as reserved - in many scenarios that
		 * allows us to avoid ever having to search the array/hash for free
		 * entries.
		 */
		ReservedRefCountSlot = ref - PrivateRefCountArray;
	}
	else
	{
		refcount_delete_item(PrivateRefCountHash, ref);
		Assert(PrivateRefCountOverflowed > 0);
		PrivateRefCountOverflowed--;
	}
}

/*
 * BufferIsPinned
 *		True iff the buffer is pinned (also checks for valid buffer number).
 *
 *		NOTE: what we check here is that *this* backend holds a pin on
 *		the buffer.  We do not care whether some other backend does.
 */
#define BufferIsPinned(bufnum) \
( \
	!BufferIsValid(bufnum) ? \
		false \
	: \
		BufferIsLocal(bufnum) ? \
			(LocalRefCount[-(bufnum) - 1] > 0) \
		: \
	(GetPrivateRefCount(bufnum) > 0) \
)

static Buffer ReadBuffer_common(Relation rel,
								SMgrRelation smgr, char smgr_persistence,
								ForkNumber forkNum, BlockNumber blockNum,
								ReadBufferMode mode, BufferAccessStrategy strategy);
static BlockNumber ExtendBufferedRelCommon(BufferManagerRelation bmr,
										   ForkNumber fork,
										   BufferAccessStrategy strategy,
										   uint32 flags,
										   uint32 extend_by,
										   BlockNumber extend_upto,
										   Buffer *buffers,
										   uint32 *extended_by);
static BlockNumber ExtendBufferedRelShared(BufferManagerRelation bmr,
										   ForkNumber fork,
										   BufferAccessStrategy strategy,
										   uint32 flags,
										   uint32 extend_by,
										   BlockNumber extend_upto,
										   Buffer *buffers,
										   uint32 *extended_by);
static bool PinBuffer(BufferDesc *buf, BufferAccessStrategy strategy,
					  bool skip_if_not_valid);
static void PinBuffer_Locked(BufferDesc *buf);
static void UnpinBuffer(BufferDesc *buf);
static void UnpinBufferNoOwner(BufferDesc *buf);
static void shared_buffer_write_error_callback(void *arg);
static inline BufferDesc *BufferAlloc(SMgrRelation smgr,
									  char relpersistence,
									  ForkNumber forkNum,
									  BlockNumber blockNum,
									  BufferAccessStrategy strategy,
									  bool *foundPtr, IOContext io_context);
static bool AsyncReadBuffers(ReadBuffersOperation *operation, int *nblocks_progress);
static void CheckReadBuffersOperation(ReadBuffersOperation *operation, bool is_complete);

static pg_always_inline void TrackBufferHit(IOObject io_object,
											IOContext io_context,
											Relation rel, char persistence, SMgrRelation smgr,
											ForkNumber forknum, BlockNumber blocknum);
static Buffer GetVictimBuffer(BufferAccessStrategy strategy, IOContext io_context);
static void FlushBuffer(BufferDesc *buf, SMgrRelation reln,
						IOObject io_object, IOContext io_context);
static void RelationCopyStorageUsingBuffer(RelFileLocator srclocator,
										   RelFileLocator dstlocator,
										   ForkNumber forkNum, bool permanent);

static void BufferLockAcquire(Buffer buffer, BufferDesc *buf_hdr, BufferLockMode mode);
static void BufferLockUnlock(Buffer buffer, BufferDesc *buf_hdr);
static bool BufferLockConditional(Buffer buffer, BufferDesc *buf_hdr, BufferLockMode mode);
static bool BufferLockHeldByMeInMode(BufferDesc *buf_hdr, BufferLockMode mode);
static bool BufferLockHeldByMe(BufferDesc *buf_hdr);
static inline int BufferLockDisownInternal(Buffer buffer, BufferDesc *buf_hdr);
static inline bool BufferLockAttempt(BufferDesc *buf_hdr, BufferLockMode mode);
static void BufferLockQueueSelf(BufferDesc *buf_hdr, BufferLockMode mode);
static void BufferLockDequeueSelf(BufferDesc *buf_hdr);
static void BufferLockWakeup(BufferDesc *buf_hdr, bool wake_exclusive);
static void BufferLockProcessRelease(BufferDesc *buf_hdr, BufferLockMode mode, uint64 lockstate);
static inline uint64 BufferLockReleaseSub(BufferLockMode mode);

static pg_noinline Buffer calico_active_ReadBufferExtended(Relation reln,
	ForkNumber forkNum, BlockNumber blockNum, ReadBufferMode mode,
	BufferAccessStrategy strategy);

/*
 * ReadBuffer -- a shorthand for ReadBufferExtended, for reading from main
 *		fork with RBM_NORMAL mode and default strategy.
 */
Buffer
ReadBuffer(Relation reln, BlockNumber blockNum)
{
	if (unlikely(!calico_local_active))
		return native_ReadBuffer(reln, blockNum);

	return calico_active_ReadBufferExtended(reln, MAIN_FORKNUM, blockNum,
										 RBM_NORMAL, NULL);
}

/*
 * ReadBufferExtended -- returns a buffer containing the requested
 *		block of the requested relation.  If the blknum
 *		requested is P_NEW, extend the relation file and
 *		allocate a new block.  (Caller is responsible for
 *		ensuring that only one backend tries to extend a
 *		relation at the same time!)
 *
 * Returns: the buffer number for the buffer containing
 *		the block read.  The returned buffer has been pinned.
 *		Does not return on error --- elog's instead.
 *
 * Assume when this function is called, that reln has been opened already.
 *
 * In RBM_NORMAL mode, the page is read from disk, and the page header is
 * validated.  An error is thrown if the page header is not valid.  (But
 * note that an all-zero page is considered "valid"; see
 * PageIsVerified().)
 *
 * RBM_ZERO_ON_ERROR is like the normal mode, but if the page header is not
 * valid, the page is zeroed instead of throwing an error. This is intended
 * for non-critical data, where the caller is prepared to repair errors.
 *
 * In RBM_ZERO_AND_LOCK mode, if the page isn't in buffer cache already, it's
 * filled with zeros instead of reading it from disk.  Useful when the caller
 * is going to fill the page from scratch, since this saves I/O and avoids
 * unnecessary failure if the page-on-disk has corrupt page headers.
 * The page is returned locked to ensure that the caller has a chance to
 * initialize the page before it's made visible to others.
 * Caution: do not use this mode to read a page that is beyond the relation's
 * current physical EOF; that is likely to cause problems in md.c when
 * the page is modified and written out. P_NEW is OK, though.
 *
 * RBM_ZERO_AND_CLEANUP_LOCK is the same as RBM_ZERO_AND_LOCK, but acquires
 * a cleanup-strength lock on the page.
 *
 * RBM_NORMAL_NO_LOG mode is treated the same as RBM_NORMAL here.
 *
 * If strategy is not NULL, a nondefault buffer access strategy is used.
 * See buffer/README for details.
 */
Buffer
ReadBufferExtended(Relation reln, ForkNumber forkNum, BlockNumber blockNum,
				   ReadBufferMode mode, BufferAccessStrategy strategy)
{
	if (unlikely(!calico_local_active))
		return native_ReadBufferExtended(reln, forkNum, blockNum, mode, strategy);

	return calico_active_ReadBufferExtended(reln, forkNum, blockNum, mode, strategy);
}

/* Native forwarding must not build the active read's operation stack frame. */
static pg_noinline Buffer
calico_active_ReadBufferExtended(Relation reln, ForkNumber forkNum,
								BlockNumber blockNum, ReadBufferMode mode,
								BufferAccessStrategy strategy)
{
	Buffer		buf;

	/*
	 * Read the buffer, and update pgstat counters to reflect a cache hit or
	 * miss.  The other-session temp-relation check is enforced by
	 * ReadBuffer_common().
	 */
	buf = ReadBuffer_common(reln, RelationGetSmgr(reln), 0,
							forkNum, blockNum, mode, strategy);

	return buf;
}

/*
 * ReadBufferWithoutRelcache -- like ReadBufferExtended, but doesn't require
 *		a relcache entry for the relation.
 *
 * Pass permanent = true for a RELPERSISTENCE_PERMANENT relation, and
 * permanent = false for a RELPERSISTENCE_UNLOGGED relation. This function
 * cannot be used for temporary relations (and making that work might be
 * difficult, unless we only want to read temporary relations for our own
 * ProcNumber).
 */
Buffer
ReadBufferWithoutRelcache(RelFileLocator rlocator, ForkNumber forkNum,
						  BlockNumber blockNum, ReadBufferMode mode,
						  BufferAccessStrategy strategy, bool permanent)
{
	SMgrRelation smgr;

	if (unlikely(!calico_local_active))
		return native_ReadBufferWithoutRelcache(rlocator, forkNum, blockNum, mode, strategy, permanent);

	smgr = smgropen(rlocator, INVALID_PROC_NUMBER);

	return ReadBuffer_common(NULL, smgr,
							 permanent ? RELPERSISTENCE_PERMANENT : RELPERSISTENCE_UNLOGGED,
							 forkNum, blockNum,
							 mode, strategy);
}

/*
 * Convenience wrapper around ExtendBufferedRelBy() extending by one block.
 */
Buffer
ExtendBufferedRel(BufferManagerRelation bmr,
				  ForkNumber forkNum,
				  BufferAccessStrategy strategy,
				  uint32 flags)
{
	Buffer		buf;
	uint32		extend_by = 1;

	if (unlikely(!calico_local_active))
		return native_ExtendBufferedRel(bmr, forkNum, strategy, flags);

	ExtendBufferedRelBy(bmr, forkNum, strategy, flags, extend_by,
						&buf, &extend_by);

	return buf;
}

/*
 * Extend relation by multiple blocks.
 *
 * Tries to extend the relation by extend_by blocks. Depending on the
 * availability of resources the relation may end up being extended by a
 * smaller number of pages (unless an error is thrown, always by at least one
 * page). *extended_by is updated to the number of pages the relation has been
 * extended to.
 *
 * buffers needs to be an array that is at least extend_by long. Upon
 * completion, the first extend_by array elements will point to a pinned
 * buffer.
 *
 * If EB_LOCK_FIRST is part of flags, the first returned buffer is
 * locked. This is useful for callers that want a buffer that is guaranteed to
 * be empty.
 */
BlockNumber
ExtendBufferedRelBy(BufferManagerRelation bmr,
					ForkNumber fork,
					BufferAccessStrategy strategy,
					uint32 flags,
					uint32 extend_by,
					Buffer *buffers,
					uint32 *extended_by)
{
	if (unlikely(!calico_local_active))
		return native_ExtendBufferedRelBy(bmr, fork, strategy, flags, extend_by, buffers, extended_by);

	Assert((bmr.rel != NULL) != (bmr.smgr != NULL));
	Assert(bmr.smgr == NULL || bmr.relpersistence != '\0');
	Assert(extend_by > 0);

	if (bmr.relpersistence == '\0')
		bmr.relpersistence = bmr.rel->rd_rel->relpersistence;

	return ExtendBufferedRelCommon(bmr, fork, strategy, flags,
								   extend_by, InvalidBlockNumber,
								   buffers, extended_by);
}

/*
 * Extend the relation so it is at least extend_to blocks large, return buffer
 * (extend_to - 1).
 *
 * This is useful for callers that want to write a specific page, regardless
 * of the current size of the relation (e.g. useful for visibilitymap and for
 * crash recovery).
 */
Buffer
ExtendBufferedRelTo(BufferManagerRelation bmr,
					ForkNumber fork,
					BufferAccessStrategy strategy,
					uint32 flags,
					BlockNumber extend_to,
					ReadBufferMode mode)
{
	BlockNumber current_size;
	uint32		extended_by = 0;
	Buffer		buffer = InvalidBuffer;
	Buffer		buffers[64];

	if (unlikely(!calico_local_active))
		return native_ExtendBufferedRelTo(bmr, fork, strategy, flags, extend_to, mode);

	Assert((bmr.rel != NULL) != (bmr.smgr != NULL));
	Assert(bmr.smgr == NULL || bmr.relpersistence != '\0');
	Assert(extend_to != InvalidBlockNumber && extend_to > 0);

	if (bmr.relpersistence == '\0')
		bmr.relpersistence = bmr.rel->rd_rel->relpersistence;

	/*
	 * If desired, create the file if it doesn't exist.  If
	 * smgr_cached_nblocks[fork] is positive then it must exist, no need for
	 * an smgrexists call.
	 */
	if ((flags & EB_CREATE_FORK_IF_NEEDED) &&
		(BMR_GET_SMGR(bmr)->smgr_cached_nblocks[fork] == 0 ||
		 BMR_GET_SMGR(bmr)->smgr_cached_nblocks[fork] == InvalidBlockNumber) &&
		!smgrexists(BMR_GET_SMGR(bmr), fork))
	{
		LockRelationForExtension(bmr.rel, ExclusiveLock);

		/* recheck, fork might have been created concurrently */
		if (!smgrexists(BMR_GET_SMGR(bmr), fork))
			smgrcreate(BMR_GET_SMGR(bmr), fork, flags & EB_PERFORMING_RECOVERY);

		UnlockRelationForExtension(bmr.rel, ExclusiveLock);
	}

	/*
	 * If requested, invalidate size cache, so that smgrnblocks asks the
	 * kernel.
	 */
	if (flags & EB_CLEAR_SIZE_CACHE)
		BMR_GET_SMGR(bmr)->smgr_cached_nblocks[fork] = InvalidBlockNumber;

	/*
	 * Estimate how many pages we'll need to extend by. This avoids acquiring
	 * unnecessarily many victim buffers.
	 */
	current_size = smgrnblocks(BMR_GET_SMGR(bmr), fork);

	/*
	 * Since no-one else can be looking at the page contents yet, there is no
	 * difference between an exclusive lock and a cleanup-strength lock. Note
	 * that we pass the original mode to ReadBuffer_common() below, when
	 * falling back to reading the buffer to a concurrent relation extension.
	 */
	if (mode == RBM_ZERO_AND_LOCK || mode == RBM_ZERO_AND_CLEANUP_LOCK)
		flags |= EB_LOCK_TARGET;

	while (current_size < extend_to)
	{
		uint32		num_pages = lengthof(buffers);
		BlockNumber first_block;

		if ((uint64) current_size + num_pages > extend_to)
			num_pages = extend_to - current_size;

		first_block = ExtendBufferedRelCommon(bmr, fork, strategy, flags,
											  num_pages, extend_to,
											  buffers, &extended_by);

		current_size = first_block + extended_by;
		Assert(num_pages != 0 || current_size >= extend_to);

		for (uint32 i = 0; i < extended_by; i++)
		{
			if (first_block + i != extend_to - 1)
				ReleaseBuffer(buffers[i]);
			else
				buffer = buffers[i];
		}
	}

	/*
	 * It's possible that another backend concurrently extended the relation.
	 * In that case read the buffer.
	 *
	 * XXX: Should we control this via a flag?
	 */
	if (buffer == InvalidBuffer)
	{
		Assert(extended_by == 0);
		buffer = ReadBuffer_common(bmr.rel, BMR_GET_SMGR(bmr), bmr.relpersistence,
								   fork, extend_to - 1, mode, strategy);
	}

	return buffer;
}

/*
 * Lock and optionally zero a buffer, as part of the implementation of
 * RBM_ZERO_AND_LOCK or RBM_ZERO_AND_CLEANUP_LOCK.  The buffer must be already
 * pinned.  If the buffer is not already valid, it is zeroed and made valid.
 */
static void
ZeroAndLockBuffer(Buffer buffer, ReadBufferMode mode, bool already_valid)
{
	BufferDesc *bufHdr;
	bool		need_to_zero;
	bool		isLocalBuf = BufferIsLocal(buffer);
	StartBufferIOResult sbres;

	Assert(mode == RBM_ZERO_AND_LOCK || mode == RBM_ZERO_AND_CLEANUP_LOCK);

	if (already_valid)
	{
		/*
		 * If the caller already knew the buffer was valid, we can skip some
		 * header interaction.  The caller just wants to lock the buffer.
		 */
		need_to_zero = false;
	}
	else
	{
		if (isLocalBuf)
		{
			/* Simple case for non-shared buffers. */
			bufHdr = GetLocalBufferDescriptor(-buffer - 1);
			sbres = StartLocalBufferIO(bufHdr, true, true, NULL);
		}
		else
		{
			/*
			 * Take BM_IO_IN_PROGRESS, or discover that BM_VALID has been set
			 * concurrently.  Even though we aren't doing I/O, that ensures
			 * that we don't zero a page that someone else has pinned.  An
			 * exclusive content lock wouldn't be enough, because readers are
			 * allowed to drop the content lock after determining that a tuple
			 * is visible (see buffer access rules in README).
			 */
			bufHdr = GetBufferDescriptor(buffer - 1);
			sbres = StartSharedBufferIO(bufHdr, true, true, NULL);
		}

		Assert(sbres != BUFFER_IO_IN_PROGRESS);
		need_to_zero = sbres == BUFFER_IO_READY_FOR_IO;
	}

	if (need_to_zero)
	{
		memset(BufferGetPage(buffer), 0, BLCKSZ);

		/*
		 * Grab the buffer content lock before marking the page as valid, to
		 * make sure that no other backend sees the zeroed page before the
		 * caller has had a chance to initialize it.
		 *
		 * Since no-one else can be looking at the page contents yet, there is
		 * no difference between an exclusive lock and a cleanup-strength
		 * lock. (Note that we cannot use LockBuffer() or
		 * LockBufferForCleanup() here, because they assert that the buffer is
		 * already valid.)
		 */
		if (!isLocalBuf)
			LockBuffer(buffer, BUFFER_LOCK_EXCLUSIVE);

		/* Set BM_VALID, terminate IO, and wake up any waiters */
		if (isLocalBuf)
			TerminateLocalBufferIO(bufHdr, false, BM_VALID, false);
		else
		{
			TerminateBufferIO(bufHdr, false, BM_VALID, true, false);
			calico_mark_frame_ready(bufHdr);
		}
	}
	else if (!isLocalBuf)
	{
		/*
		 * The buffer is valid, so we can't zero it.  The caller still expects
		 * the page to be locked on return.
		 */
		if (mode == RBM_ZERO_AND_LOCK)
			LockBuffer(buffer, BUFFER_LOCK_EXCLUSIVE);
		else
			LockBufferForCleanup(buffer);
	}
}

/*
 * Pin a buffer for a given block.  *foundPtr is set to true if the block was
 * already present, or false if more work is required to either read it in or
 * zero it.
 */
static pg_always_inline Buffer
PinBufferForBlock(Relation rel,
				  SMgrRelation smgr,
				  char persistence,
				  ForkNumber forkNum,
				  BlockNumber blockNum,
				  BufferAccessStrategy strategy,
				  IOObject io_object,
				  IOContext io_context,
				  bool *foundPtr)
{
	BufferDesc *bufHdr;

	Assert(blockNum != P_NEW);

	/* Persistence should be set before */
	Assert((persistence == RELPERSISTENCE_TEMP ||
			persistence == RELPERSISTENCE_PERMANENT ||
			persistence == RELPERSISTENCE_UNLOGGED));

	TRACE_POSTGRESQL_BUFFER_READ_START(forkNum, blockNum,
									   smgr->smgr_rlocator.locator.spcOid,
									   smgr->smgr_rlocator.locator.dbOid,
									   smgr->smgr_rlocator.locator.relNumber,
									   smgr->smgr_rlocator.backend);

	if (persistence == RELPERSISTENCE_TEMP)
		bufHdr = LocalBufferAlloc(smgr, forkNum, blockNum, foundPtr);
	else
		bufHdr = BufferAlloc(smgr, persistence, forkNum, blockNum,
							 strategy, foundPtr, io_context);

	if (*foundPtr)
		TrackBufferHit(io_object, io_context, rel, persistence, smgr, forkNum, blockNum);

	if (rel)
	{
		/*
		 * While pgBufferUsage's "read" counter isn't bumped unless we reach
		 * WaitReadBuffers() (so, not for hits, and not for buffers that are
		 * zeroed instead), the per-relation stats always count them.
		 */
		pgstat_count_buffer_read(rel);
	}

	return BufferDescriptorGetBuffer(bufHdr);
}

/*
 * ReadBuffer_common -- common logic for all ReadBuffer variants
 *
 * smgr is required, rel is optional unless using P_NEW.
 */
static pg_always_inline Buffer
ReadBuffer_common(Relation rel, SMgrRelation smgr, char smgr_persistence,
				  ForkNumber forkNum,
				  BlockNumber blockNum, ReadBufferMode mode,
				  BufferAccessStrategy strategy)
{
	ReadBuffersOperation operation;
	Buffer		buffer;
	int			flags;
	char		persistence;

	/*
	 * Reject attempts to read non-local temporary relations; we would be
	 * likely to get wrong data since we have no visibility into the owning
	 * session's local buffers.  This is the canonical place for the check,
	 * covering the ReadBufferExtended() entry point and any other caller that
	 * supplies a Relation.
	 */
	if (rel && RELATION_IS_OTHER_TEMP(rel))
		ereport(ERROR,
				(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
				 errmsg("cannot access temporary tables of other sessions")));

	/*
	 * Backward compatibility path, most code should use ExtendBufferedRel()
	 * instead, as acquiring the extension lock inside ExtendBufferedRel()
	 * scales a lot better.
	 */
	if (unlikely(blockNum == P_NEW))
	{
		uint32		flags = EB_SKIP_EXTENSION_LOCK;

		/*
		 * Since no-one else can be looking at the page contents yet, there is
		 * no difference between an exclusive lock and a cleanup-strength
		 * lock.
		 */
		if (mode == RBM_ZERO_AND_LOCK || mode == RBM_ZERO_AND_CLEANUP_LOCK)
			flags |= EB_LOCK_FIRST;

		return ExtendBufferedRel(BMR_REL(rel), forkNum, strategy, flags);
	}

	if (rel)
		persistence = rel->rd_rel->relpersistence;
	else
		persistence = smgr_persistence;

	if (unlikely(mode == RBM_ZERO_AND_CLEANUP_LOCK ||
				 mode == RBM_ZERO_AND_LOCK))
	{
		bool		found;
		IOContext	io_context;
		IOObject	io_object;

		if (persistence == RELPERSISTENCE_TEMP)
		{
			io_context = IOCONTEXT_NORMAL;
			io_object = IOOBJECT_TEMP_RELATION;
		}
		else
		{
			io_context = IOContextForStrategy(strategy);
			io_object = IOOBJECT_RELATION;
		}

		buffer = PinBufferForBlock(rel, smgr, persistence,
								   forkNum, blockNum, strategy,
								   io_object, io_context, &found);
		ZeroAndLockBuffer(buffer, mode, found);
		return buffer;
	}

	/*
	 * Signal that we are going to immediately wait. If we're immediately
	 * waiting, there is no benefit in actually executing the IO
	 * asynchronously, it would just add dispatch overhead.
	 */
	flags = READ_BUFFERS_SYNCHRONOUSLY;
	if (mode == RBM_ZERO_ON_ERROR)
		flags |= READ_BUFFERS_ZERO_ON_ERROR;
	operation.smgr = smgr;
	operation.rel = rel;
	operation.persistence = persistence;
	operation.forknum = forkNum;
	operation.strategy = strategy;
	if (StartReadBuffer(&operation,
						&buffer,
						blockNum,
						flags))
	{
		WaitReadBuffers(&operation);
		if (BufferIsValid(buffer) && !BufferIsLocal(buffer))
			calico_mark_frame_ready(GetBufferDescriptor(buffer - 1));
	}

	return buffer;
}

static pg_always_inline bool
StartReadBuffersImpl(ReadBuffersOperation *operation,
					 Buffer *buffers,
					 BlockNumber blockNum,
					 int *nblocks,
					 int flags,
					 bool allow_forwarding)
{
	int			actual_nblocks = *nblocks;
	int			maxcombine = 0;
	bool		did_start_io;
	IOContext	io_context;
	IOObject	io_object;

	Assert(*nblocks == 1 || allow_forwarding);
	Assert(*nblocks > 0);
	Assert(*nblocks <= MAX_IO_COMBINE_LIMIT);

	/* see comments in ReadBuffer_common */
	if (operation->rel && RELATION_IS_OTHER_TEMP(operation->rel))
		ereport(ERROR,
				(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
				 errmsg("cannot access temporary tables of other sessions")));

	if (operation->persistence == RELPERSISTENCE_TEMP)
	{
		io_context = IOCONTEXT_NORMAL;
		io_object = IOOBJECT_TEMP_RELATION;
	}
	else
	{
		io_context = IOContextForStrategy(operation->strategy);
		io_object = IOOBJECT_RELATION;
	}

	for (int i = 0; i < actual_nblocks; ++i)
	{
		bool		found;

		if (allow_forwarding && buffers[i] != InvalidBuffer)
		{
			BufferDesc *bufHdr;

			/*
			 * This is a buffer that was pinned by an earlier call to
			 * StartReadBuffers(), but couldn't be handled in one operation at
			 * that time.  The operation was split, and the caller has passed
			 * an already pinned buffer back to us to handle the rest of the
			 * operation.  It must continue at the expected block number.
			 */
			Assert(BufferGetBlockNumber(buffers[i]) == blockNum + i);

			/*
			 * It might be an already valid buffer (a hit) that followed the
			 * final contiguous block of an earlier I/O (a miss) marking the
			 * end of it, or a buffer that some other backend has since made
			 * valid by performing the I/O for us, in which case we can handle
			 * it as a hit now.  It is safe to check for a BM_VALID flag with
			 * a relaxed load, because we got a fresh view of it while pinning
			 * it in the previous call.
			 *
			 * On the other hand if we don't see BM_VALID yet, it must be an
			 * I/O that was split by the previous call and we need to try to
			 * start a new I/O from this block.  We're also racing against any
			 * other backend that might start the I/O or even manage to mark
			 * it BM_VALID after this check, but StartBufferIO() will handle
			 * those cases.
			 */
			if (BufferIsLocal(buffers[i]))
				bufHdr = GetLocalBufferDescriptor(-buffers[i] - 1);
			else
				bufHdr = GetBufferDescriptor(buffers[i] - 1);
			Assert(pg_atomic_read_u64(&bufHdr->state) & BM_TAG_VALID);
			found = pg_atomic_read_u64(&bufHdr->state) & BM_VALID;
			if (found && !BufferIsLocal(buffers[i]))
				calico_mark_frame_ready(bufHdr);
		}
		else
		{
			buffers[i] = PinBufferForBlock(operation->rel,
										   operation->smgr,
										   operation->persistence,
										   operation->forknum,
										   blockNum + i,
										   operation->strategy,
										   io_object, io_context,
										   &found);
		}

		if (found)
		{
			/*
			 * We have a hit.  If it's the first block in the requested range,
			 * we can return it immediately and report that WaitReadBuffers()
			 * does not need to be called.  If the initial value of *nblocks
			 * was larger, the caller will have to call again for the rest.
			 */
			if (i == 0)
			{
				*nblocks = 1;

#ifdef USE_ASSERT_CHECKING

				/*
				 * Initialize enough of ReadBuffersOperation to make
				 * CheckReadBuffersOperation() work. Outside of assertions
				 * that's not necessary when no IO is issued.
				 */
				operation->buffers = buffers;
				operation->blocknum = blockNum;
				operation->nblocks = 1;
				operation->nblocks_done = 1;
				CheckReadBuffersOperation(operation, true);
#endif
				return false;
			}

			/*
			 * Otherwise we already have an I/O to perform, but this block
			 * can't be included as it is already valid.  Split the I/O here.
			 * There may or may not be more blocks requiring I/O after this
			 * one, we haven't checked, but they can't be contiguous with this
			 * one in the way.  We'll leave this buffer pinned, forwarding it
			 * to the next call, avoiding the need to unpin it here and re-pin
			 * it in the next call.
			 */
			actual_nblocks = i;
			break;
		}
		else
		{
			/*
			 * Check how many blocks we can cover with the same IO. The smgr
			 * implementation might e.g. be limited due to a segment boundary.
			 */
			if (i == 0 && actual_nblocks > 1)
			{
				maxcombine = smgrmaxcombine(operation->smgr,
											operation->forknum,
											blockNum);
				if (unlikely(maxcombine < actual_nblocks))
				{
					elog(DEBUG2, "limiting nblocks at %u from %u to %u",
						 blockNum, actual_nblocks, maxcombine);
					actual_nblocks = maxcombine;
				}
			}
		}
	}
	*nblocks = actual_nblocks;

	/* Populate information needed for I/O. */
	operation->buffers = buffers;
	operation->blocknum = blockNum;
	operation->flags = flags;
	operation->nblocks = actual_nblocks;
	operation->nblocks_done = 0;
	pgaio_wref_clear(&operation->io_wref);

	/*
	 * When using AIO, start the IO in the background. If not, issue prefetch
	 * requests if desired by the caller.
	 *
	 * The reason we have a dedicated path for IOMETHOD_SYNC here is to
	 * de-risk the introduction of AIO somewhat. It's a large architectural
	 * change, with lots of chances for unanticipated performance effects.
	 *
	 * Use of IOMETHOD_SYNC already leads to not actually performing IO
	 * asynchronously, but without the check here we'd execute IO earlier than
	 * we used to. Eventually this IOMETHOD_SYNC specific path should go away.
	 */
	if (io_method != IOMETHOD_SYNC)
	{
		/*
		 * Try to start IO asynchronously. It's possible that no IO needs to
		 * be started, if another backend already performed the IO.
		 *
		 * Note that if an IO is started, it might not cover the entire
		 * requested range, e.g. because an intermediary block has been read
		 * in by another backend.  In that case any "trailing" buffers we
		 * already pinned above will be "forwarded" by read_stream.c to the
		 * next call to StartReadBuffers().
		 *
		 * This is signalled to the caller by decrementing *nblocks *and*
		 * reducing operation->nblocks. The latter is done here, but not below
		 * WaitReadBuffers(), as in WaitReadBuffers() we can't "shorten" the
		 * overall read size anymore, we need to retry until done in its
		 * entirety or until failed.
		 */
		did_start_io = AsyncReadBuffers(operation, nblocks);

		operation->nblocks = *nblocks;
	}
	else
	{
		operation->flags |= READ_BUFFERS_SYNCHRONOUSLY;

		if (flags & READ_BUFFERS_ISSUE_ADVICE)
		{
			/*
			 * In theory we should only do this if PinBufferForBlock() had to
			 * allocate new buffers above.  That way, if two calls to
			 * StartReadBuffers() were made for the same blocks before
			 * WaitReadBuffers(), only the first would issue the advice.
			 * That'd be a better simulation of true asynchronous I/O, which
			 * would only start the I/O once, but isn't done here for
			 * simplicity.
			 */
			smgrprefetch(operation->smgr,
						 operation->forknum,
						 blockNum,
						 actual_nblocks);
		}

		/*
		 * Indicate that WaitReadBuffers() should be called. WaitReadBuffers()
		 * will initiate the necessary IO.
		 */
		did_start_io = true;
	}

	CheckReadBuffersOperation(operation, !did_start_io);

	return did_start_io;
}

/*
 * Begin reading a range of blocks beginning at blockNum and extending for
 * *nblocks.  *nblocks and the buffers array are in/out parameters.  On entry,
 * the buffers elements covered by *nblocks must hold either InvalidBuffer or
 * buffers forwarded by an earlier call to StartReadBuffers() that was split
 * and is now being continued.  On return, *nblocks holds the number of blocks
 * accepted by this operation.  If it is less than the original number then
 * this operation has been split, but buffer elements up to the original
 * requested size may hold forwarded buffers to be used for a continuing
 * operation.  The caller must either start a new I/O beginning at the block
 * immediately following the blocks accepted by this call and pass those
 * buffers back in, or release them if it chooses not to.  It shouldn't make
 * any other use of or assumptions about forwarded buffers.
 *
 * If false is returned, no I/O is necessary and the buffers covered by
 * *nblocks on exit are valid and ready to be accessed.  If true is returned,
 * an I/O has been started, and WaitReadBuffers() must be called with the same
 * operation object before the buffers covered by *nblocks on exit can be
 * accessed.  Along with the operation object, the caller-supplied array of
 * buffers must remain valid until WaitReadBuffers() is called, and any
 * forwarded buffers must also be preserved for a continuing call unless
 * they are explicitly released.
 */
bool
StartReadBuffers(ReadBuffersOperation *operation,
				 Buffer *buffers,
				 BlockNumber blockNum,
				 int *nblocks,
				 int flags)
{
	if (unlikely(!calico_local_active))
		return native_StartReadBuffers(operation, buffers, blockNum, nblocks, flags);

	return StartReadBuffersImpl(operation, buffers, blockNum, nblocks, flags,
								true /* expect forwarded buffers */ );
}

/*
 * Single block version of the StartReadBuffers().  This might save a few
 * instructions when called from another translation unit, because it is
 * specialized for nblocks == 1.
 *
 * This version does not support "forwarded" buffers: they cannot be created
 * by reading only one block and *buffer is ignored on entry.
 */
static pg_noinline bool
calico_active_StartReadBuffer(ReadBuffersOperation *operation,
				Buffer *buffer,
				BlockNumber blocknum,
				int flags)
{
	int			nblocks = 1;
	bool		result;

	result = StartReadBuffersImpl(operation, buffer, blocknum, &nblocks, flags,
								  false /* single block, no forwarding */ );
	Assert(nblocks == 1);		/* single block can't be short */

	return result;
}

/* Keep native forwarding outside the active read's stack frame. */
bool
StartReadBuffer(ReadBuffersOperation *operation,
				Buffer *buffer,
				BlockNumber blocknum,
				int flags)
{
	if (unlikely(!calico_local_active))
		return native_StartReadBuffer(operation, buffer, blocknum, flags);
	return calico_active_StartReadBuffer(operation, buffer, blocknum, flags);
}

/*
 * Perform sanity checks on the ReadBuffersOperation.
 */
static void
CheckReadBuffersOperation(ReadBuffersOperation *operation, bool is_complete)
{
#ifdef USE_ASSERT_CHECKING
	Assert(operation->nblocks_done <= operation->nblocks);
	Assert(!is_complete || operation->nblocks == operation->nblocks_done);

	for (int i = 0; i < operation->nblocks; i++)
	{
		Buffer		buffer = operation->buffers[i];
		BufferDesc *buf_hdr = BufferIsLocal(buffer) ?
			GetLocalBufferDescriptor(-buffer - 1) :
			GetBufferDescriptor(buffer - 1);

		Assert(BufferGetBlockNumber(buffer) == operation->blocknum + i);
		Assert(pg_atomic_read_u64(&buf_hdr->state) & BM_TAG_VALID);

		if (i < operation->nblocks_done)
			Assert(pg_atomic_read_u64(&buf_hdr->state) & BM_VALID);
	}
#endif
}

/*
 * We track various stats related to buffer hits. Because this is done in a
 * few separate places, this helper exists for convenience.
 */
static pg_always_inline void
TrackBufferHit(IOObject io_object, IOContext io_context,
			   Relation rel, char persistence, SMgrRelation smgr,
			   ForkNumber forknum, BlockNumber blocknum)
{
	TRACE_POSTGRESQL_BUFFER_READ_DONE(forknum,
									  blocknum,
									  smgr->smgr_rlocator.locator.spcOid,
									  smgr->smgr_rlocator.locator.dbOid,
									  smgr->smgr_rlocator.locator.relNumber,
									  smgr->smgr_rlocator.backend,
									  true);

	if (persistence == RELPERSISTENCE_TEMP)
		pgBufferUsage.local_blks_hit += 1;
	else
		pgBufferUsage.shared_blks_hit += 1;

	pgstat_count_io_op(io_object, io_context, IOOP_HIT, 1, 0);

	if (VacuumCostActive)
		VacuumCostBalance += VacuumCostPageHit;

	if (rel)
		pgstat_count_buffer_hit(rel);
}

/*
 * Initiate IO for the ReadBuffersOperation
 *
 * This function only starts a single IO at a time. The size of the IO may be
 * limited to below the to-be-read blocks, if one of the buffers has
 * concurrently been read in. If the first to-be-read buffer is already valid,
 * no IO will be issued.
 *
 * To support retries after partial reads, the first operation->nblocks_done
 * buffers are skipped.
 *
 * On return *nblocks_progress is updated to reflect the number of buffers
 * affected by the call. If the first buffer is valid, *nblocks_progress is
 * set to 1 and operation->nblocks_done is incremented.
 *
 * Returns true if IO was initiated or is already in progress (foreign IO),
 * false if the buffer was already valid.
 */
static bool
AsyncReadBuffers(ReadBuffersOperation *operation, int *nblocks_progress)
{
	Buffer	   *buffers = &operation->buffers[0];
	int			flags = operation->flags;
	ForkNumber	forknum = operation->forknum;
	char		persistence = operation->persistence;
	int16		nblocks_done = operation->nblocks_done;
	BlockNumber blocknum = operation->blocknum + nblocks_done;
	Buffer	   *io_buffers = &operation->buffers[nblocks_done];
	int			io_buffers_len = 0;
	PgAioHandle *ioh;
	uint32		ioh_flags = 0;
	void	   *io_pages[MAX_IO_COMBINE_LIMIT];
	IOContext	io_context;
	IOObject	io_object;
	instr_time	io_start;
	StartBufferIOResult status;

	if (persistence == RELPERSISTENCE_TEMP)
	{
		io_context = IOCONTEXT_NORMAL;
		io_object = IOOBJECT_TEMP_RELATION;
	}
	else
	{
		io_context = IOContextForStrategy(operation->strategy);
		io_object = IOOBJECT_RELATION;
	}

	/*
	 * When this IO is executed synchronously, either because the caller will
	 * immediately block waiting for the IO or because IOMETHOD_SYNC is used,
	 * the AIO subsystem needs to know.
	 */
	if (flags & READ_BUFFERS_SYNCHRONOUSLY)
		ioh_flags |= PGAIO_HF_SYNCHRONOUS;

	if (persistence == RELPERSISTENCE_TEMP)
		ioh_flags |= PGAIO_HF_REFERENCES_LOCAL;

	/*
	 * If zero_damaged_pages is enabled, add the READ_BUFFERS_ZERO_ON_ERROR
	 * flag. The reason for that is that, hopefully, zero_damaged_pages isn't
	 * set globally, but on a per-session basis. The completion callback,
	 * which may be run in other processes, e.g. in IO workers, may have a
	 * different value of the zero_damaged_pages GUC.
	 *
	 * XXX: We probably should eventually use a different flag for
	 * zero_damaged_pages, so we can report different log levels / error codes
	 * for zero_damaged_pages and ZERO_ON_ERROR.
	 */
	if (zero_damaged_pages)
		flags |= READ_BUFFERS_ZERO_ON_ERROR;

	/*
	 * For the same reason as with zero_damaged_pages we need to use this
	 * backend's ignore_checksum_failure value.
	 */
	if (ignore_checksum_failure)
		flags |= READ_BUFFERS_IGNORE_CHECKSUM_FAILURES;

	/*
	 * To be allowed to report stats in the local completion callback we need
	 * to prepare to report stats now. This ensures we can safely report the
	 * checksum failure even in a critical section.
	 */
	pgstat_prepare_report_checksum_failure(operation->smgr->smgr_rlocator.locator.dbOid);

	/*
	 * We must get an IO handle before StartBufferIO(), as pgaio_io_acquire()
	 * might block, which we don't want after setting IO_IN_PROGRESS. If we
	 * don't need to do the IO, we'll release the handle.
	 *
	 * If we need to wait for IO before we can get a handle, submit
	 * already-staged IO first, so that other backends don't need to wait.
	 * There wouldn't be a deadlock risk, as pgaio_io_acquire() just needs to
	 * wait for already submitted IO, which doesn't require additional locks,
	 * but it could still cause undesirable waits.
	 *
	 * A secondary benefit is that this would allow us to measure the time in
	 * pgaio_io_acquire() without causing undue timer overhead in the common,
	 * non-blocking, case.  However, currently the pgstats infrastructure
	 * doesn't really allow that because it doesn't have an API to report
	 * "accumulated" time.
	 */
	ioh = pgaio_io_acquire_nb(CurrentResourceOwner, &operation->io_return);
	if (unlikely(!ioh))
	{
		pgaio_submit_staged();
		ioh = pgaio_io_acquire(CurrentResourceOwner, &operation->io_return);
	}

	operation->foreign_io = false;
	pgaio_wref_clear(&operation->io_wref);

	/*
	 * Try to start IO on the first buffer in a new run of blocks. If AIO is
	 * in progress, be it in this backend or another backend, we just
	 * associate the wait reference with the operation and wait in
	 * WaitReadBuffers(). This turns out to be important for performance in
	 * two workloads:
	 *
	 * 1) A read stream that has to read the same block multiple times within
	 * the readahead distance. This can happen e.g. for the table accesses of
	 * an index scan.
	 *
	 * 2) Concurrent scans by multiple backends on the same relation.
	 *
	 * If we were to synchronously wait for the in-progress IO, we'd not be
	 * able to keep enough I/O in flight.
	 *
	 * If we do find there is ongoing I/O for the buffer, we set up a 1-block
	 * ReadBuffersOperation that WaitReadBuffers then can wait on.
	 *
	 * It's possible that another backend has started IO on the buffer but not
	 * yet set its wait reference. In this case, we have no choice but to wait
	 * for either the wait reference to be valid or the IO to be done.
	 */
	status = StartBufferIO(buffers[nblocks_done], true, true,
						   &operation->io_wref);
	if (status != BUFFER_IO_READY_FOR_IO)
	{
		pgaio_io_release(ioh);
		*nblocks_progress = 1;
		if (status == BUFFER_IO_ALREADY_DONE)
		{
			/*
			 * Someone has already completed this block, we're done.
			 *
			 * When IO is necessary, ->nblocks_done is updated in
			 * ProcessReadBuffersResult(), but that is not called if no IO is
			 * necessary. Thus update here.
			 */
			operation->nblocks_done += 1;
			Assert(operation->nblocks_done <= operation->nblocks);

			Assert(!pgaio_wref_valid(&operation->io_wref));

			/*
			 * Report and track this as a 'hit' for this backend, even though
			 * it must have started out as a miss in PinBufferForBlock(). The
			 * other backend will track this as a 'read'.
			 */
			TrackBufferHit(io_object, io_context,
						   operation->rel, operation->persistence,
						   operation->smgr, operation->forknum,
						   blocknum);
			if (!BufferIsLocal(buffers[nblocks_done]))
				calico_mark_frame_ready(GetBufferDescriptor(buffers[nblocks_done] - 1));
			return false;
		}

		/* The IO is already in-progress */
		Assert(status == BUFFER_IO_IN_PROGRESS);
		Assert(pgaio_wref_valid(&operation->io_wref));
		operation->foreign_io = true;

		return true;
	}

	Assert(io_buffers[0] == buffers[nblocks_done]);
	io_pages[0] = BufferGetBlock(buffers[nblocks_done]);
	io_buffers_len = 1;

	/*
	 * NB: As little code as possible should be added between the
	 * StartBufferIO() above, the further StartBufferIO()s below and the
	 * smgrstartreadv(), as some of the buffers are now marked as
	 * IO_IN_PROGRESS and will thus cause other backends to wait.
	 */

	/*
	 * How many neighboring-on-disk blocks can we scatter-read into other
	 * buffers at the same time?  In this case we don't wait if we see an I/O
	 * already in progress (see comment above).
	 */
	for (int i = nblocks_done + 1; i < operation->nblocks; i++)
	{
		/* Must be consecutive block numbers. */
		Assert(BufferGetBlockNumber(buffers[i - 1]) ==
			   BufferGetBlockNumber(buffers[i]) - 1);

		status = StartBufferIO(buffers[i], true, false, NULL);
		if (status != BUFFER_IO_READY_FOR_IO)
			break;

		Assert(io_buffers[io_buffers_len] == buffers[i]);

		io_pages[io_buffers_len++] = BufferGetBlock(buffers[i]);
	}

	/* get a reference to wait for in WaitReadBuffers() */
	pgaio_io_get_wref(ioh, &operation->io_wref);

	/* provide the list of buffers to the completion callbacks */
	pgaio_io_set_handle_data_32(ioh, (uint32 *) io_buffers, io_buffers_len);

	pgaio_io_register_callbacks(ioh,
								persistence == RELPERSISTENCE_TEMP ?
								PGAIO_HCB_LOCAL_BUFFER_READV :
								PGAIO_HCB_SHARED_BUFFER_READV,
								flags);

	pgaio_io_set_flag(ioh, ioh_flags);

	/* ---
	 * Even though we're trying to issue IO asynchronously, track the time
	 * in smgrstartreadv():
	 * - if io_method == IOMETHOD_SYNC, we will always perform the IO
	 *   immediately
	 * - the io method might not support the IO (e.g. worker IO for a temp
	 *   table)
	 * ---
	 */
	io_start = pgstat_prepare_io_time(track_io_timing);
	smgrstartreadv(ioh, operation->smgr, forknum,
				   blocknum,
				   io_pages, io_buffers_len);
	pgstat_count_io_op_time(io_object, io_context, IOOP_READ,
							io_start, 1, io_buffers_len * BLCKSZ);

	if (persistence == RELPERSISTENCE_TEMP)
		pgBufferUsage.local_blks_read += io_buffers_len;
	else
		pgBufferUsage.shared_blks_read += io_buffers_len;

	/*
	 * Track vacuum cost when issuing IO, not after waiting for it. Otherwise
	 * we could end up issuing a lot of IO in a short timespan, despite a low
	 * cost limit.
	 */
	if (VacuumCostActive)
		VacuumCostBalance += VacuumCostPageMiss * io_buffers_len;

	*nblocks_progress = io_buffers_len;

	return true;
}

/*
 * BufferAlloc -- subroutine for PinBufferForBlock.  Handles lookup of a shared
 *		buffer.  If no buffer exists already, selects a replacement victim and
 *		evicts the old page, but does NOT read in new page.
 *
 * "strategy" can be a buffer replacement strategy object, or NULL for
 * the default strategy.  The selected buffer's usage_count is advanced when
 * using the default strategy, but otherwise possibly not (see PinBuffer).
 *
 * The returned buffer is pinned and is already marked as holding the
 * desired page.  If it already did have the desired page, *foundPtr is
 * set true.  Otherwise, *foundPtr is set false.
 *
 * io_context is passed as an output parameter to avoid calling
 * IOContextForStrategy() when there is a shared buffers hit and no IO
 * statistics need be captured.
 *
 * No locks are held either at entry or exit.
 */
static pg_noinline BufferDesc *
calico_stock_BufferAlloc(SMgrRelation smgr, char relpersistence, ForkNumber forkNum,
			BlockNumber blockNum,
			BufferAccessStrategy strategy,
			bool *foundPtr, IOContext io_context)
{
	BufferTag	newTag;			/* identity of requested block */
	uint32		newHash;		/* hash value for newTag */
	LWLock	   *newPartitionLock;	/* buffer partition lock for it */
	int			existing_buf_id;
	Buffer		victim_buffer;
	BufferDesc *victim_buf_hdr;
	uint64		victim_buf_state;
	uint64		set_bits = 0;

	/* Make sure we will have room to remember the buffer pin */
	ResourceOwnerEnlarge(CurrentResourceOwner);
	ReservePrivateRefCountEntry();

	/* create a tag so we can lookup the buffer */
	InitBufferTag(&newTag, &smgr->smgr_rlocator.locator, forkNum, blockNum);

	/* determine its hash code and partition lock ID */
	newHash = BufTableHashCode(&newTag);
	newPartitionLock = BufMappingPartitionLock(newHash);

	/* see if the block is in the buffer pool already */
	LWLockAcquire(newPartitionLock, LW_SHARED);
	existing_buf_id = BufTableLookup(&newTag, newHash);
	if (existing_buf_id >= 0)
	{
		BufferDesc *buf;
		bool		valid;

		/*
		 * Found it.  Now, pin the buffer so no one can steal it from the
		 * buffer pool, and check to see if the correct data has been loaded
		 * into the buffer.
		 */
		buf = GetBufferDescriptor(existing_buf_id);

		valid = PinBuffer(buf, strategy, false);

		/* Can release the mapping lock as soon as we've pinned it */
		LWLockRelease(newPartitionLock);

		*foundPtr = true;

		if (!valid)
		{
			/*
			 * We can only get here if (a) someone else is still reading in
			 * the page, (b) a previous read attempt failed, or (c) someone
			 * called StartReadBuffers() but not yet WaitReadBuffers().
			 */
			*foundPtr = false;
		}

		return buf;
	}

	/*
	 * Didn't find it in the buffer pool.  We'll have to initialize a new
	 * buffer.  Remember to unlock the mapping lock while doing the work.
	 */
	LWLockRelease(newPartitionLock);

	/*
	 * Acquire a victim buffer. Somebody else might try to do the same, we
	 * don't hold any conflicting locks. If so we'll have to undo our work
	 * later.
	 */
	victim_buffer = GetVictimBuffer(strategy, io_context);
	victim_buf_hdr = GetBufferDescriptor(victim_buffer - 1);

	/*
	 * Try to make a hashtable entry for the buffer under its new tag. If
	 * somebody else inserted another buffer for the tag, we'll release the
	 * victim buffer we acquired and use the already inserted one.
	 */
	LWLockAcquire(newPartitionLock, LW_EXCLUSIVE);
	existing_buf_id = BufTableInsert(&newTag, newHash, victim_buf_hdr->buf_id);
	if (existing_buf_id >= 0)
	{
		BufferDesc *existing_buf_hdr;
		bool		valid;

		/*
		 * Got a collision. Someone has already done what we were about to do.
		 * We'll just handle this as if it were found in the buffer pool in
		 * the first place.  First, give up the buffer we were planning to
		 * use.
		 *
		 * We could do this after releasing the partition lock, but then we'd
		 * have to call ResourceOwnerEnlarge() & ReservePrivateRefCountEntry()
		 * before acquiring the lock, for the rare case of such a collision.
		 */
		UnpinBuffer(victim_buf_hdr);

		/* remaining code should match code at top of routine */

		existing_buf_hdr = GetBufferDescriptor(existing_buf_id);

		valid = PinBuffer(existing_buf_hdr, strategy, false);

		/* Can release the mapping lock as soon as we've pinned it */
		LWLockRelease(newPartitionLock);

		*foundPtr = true;

		if (!valid)
		{
			/*
			 * We can only get here if (a) someone else is still reading in
			 * the page, (b) a previous read attempt failed, or (c) someone
			 * called StartReadBuffers() but not yet WaitReadBuffers().
			 */
			*foundPtr = false;
		}

		return existing_buf_hdr;
	}

	/*
	 * Need to lock the buffer header too in order to change its tag.
	 */
	victim_buf_state = LockBufHdr(victim_buf_hdr);

	/* some sanity checks while we hold the buffer header lock */
	Assert(BUF_STATE_GET_REFCOUNT(victim_buf_state) == 1);
	Assert(!(victim_buf_state & (BM_TAG_VALID | BM_VALID | BM_DIRTY | BM_IO_IN_PROGRESS)));

	victim_buf_hdr->tag = newTag;

	/*
	 * Make sure BM_PERMANENT is set for buffers that must be written at every
	 * checkpoint.  Unlogged buffers only need to be written at shutdown
	 * checkpoints, except for their "init" forks, which need to be treated
	 * just like permanent relations.
	 */
	set_bits |= BM_TAG_VALID | BUF_USAGECOUNT_ONE;
	if (relpersistence == RELPERSISTENCE_PERMANENT || forkNum == INIT_FORKNUM)
		set_bits |= BM_PERMANENT;

	UnlockBufHdrExt(victim_buf_hdr, victim_buf_state,
					set_bits, 0, 0);

	LWLockRelease(newPartitionLock);

	/*
	 * Buffer contents are currently invalid.
	 */
	*foundPtr = false;

	return victim_buf_hdr;
}

static pg_always_inline BufferDesc *
BufferAlloc(SMgrRelation smgr, char relpersistence, ForkNumber forkNum,
			BlockNumber blockNum,
			BufferAccessStrategy strategy,
			bool *foundPtr, IOContext io_context)
{
    if (likely(calico_local_active))
        return CalicoLookupAndPin(smgr, relpersistence, forkNum, blockNum,
                                 strategy, foundPtr, io_context);
    return calico_stock_BufferAlloc(smgr, relpersistence, forkNum, blockNum,
                                    strategy, foundPtr, io_context);
}

/*
 * Helper routine for GetVictimBuffer()
 *
 * Needs to be called on a buffer with a valid tag, pinned, but without the
 * buffer header spinlock held.
 *
 * Returns true if the buffer can be reused, in which case the buffer is only
 * pinned by this backend and marked as invalid, false otherwise.
 */
static bool
InvalidateVictimBuffer(BufferDesc *buf_hdr)
{
	uint64		buf_state;
	uint32		hash;
	LWLock	   *partition_lock;
	BufferTag	tag;

	Assert(GetPrivateRefCount(BufferDescriptorGetBuffer(buf_hdr)) == 1);

	/* have buffer pinned, so it's safe to read tag without lock */
	tag = buf_hdr->tag;
	calico_prefetch_victim_entry(buf_hdr->buf_id);

	hash = BufTableHashCode(&tag);
	partition_lock = BufMappingPartitionLock(hash);

	LWLockAcquire(partition_lock, LW_EXCLUSIVE);

	/* lock the buffer header */
	buf_state = LockBufHdr(buf_hdr);

	/*
	 * We have the buffer pinned nobody else should have been able to unset
	 * this concurrently.
	 */
	Assert(buf_state & BM_TAG_VALID);
	Assert(BUF_STATE_GET_REFCOUNT(buf_state) > 0);
	Assert(BufferTagsEqual(&buf_hdr->tag, &tag));

	/*
	 * If somebody else pinned the buffer since, or even worse, dirtied it,
	 * give up on this buffer: It's clearly in use.
	 */
	if (BUF_STATE_GET_REFCOUNT(buf_state) != 1 || (buf_state & BM_DIRTY))
	{
		Assert(BUF_STATE_GET_REFCOUNT(buf_state) > 0);

		UnlockBufHdr(buf_hdr);
		LWLockRelease(partition_lock);

		return false;
	}

	/*
	 * An invalidated buffer should not have any backends waiting to lock the
	 * buffer, therefore BM_LOCK_WAKE_IN_PROGRESS should not be set.
	 */
	Assert(!(buf_state & BM_LOCK_WAKE_IN_PROGRESS));

	/*
	 * Clear out the buffer's tag and flags and usagecount.  This is not
	 * strictly required, as BM_TAG_VALID/BM_VALID needs to be checked before
	 * doing anything with the buffer. But currently it's beneficial, as the
	 * cheaper pre-check for several linear scans of shared buffers use the
	 * tag (see e.g. FlushDatabaseBuffers()).
	 */
	ClearBufferTag(&buf_hdr->tag);
	UnlockBufHdrExt(buf_hdr, buf_state,
					0,
					BUF_FLAG_MASK | BUF_USAGECOUNT_MASK,
					0);

	Assert(BUF_STATE_GET_REFCOUNT(buf_state) > 0);

	/* finally delete buffer from the buffer mapping table */
	calico_table_delete_frame(&tag, hash, buf_hdr->buf_id);

	LWLockRelease(partition_lock);

	buf_state = pg_atomic_read_u64(&buf_hdr->state);
	Assert(!(buf_state & (BM_DIRTY | BM_VALID | BM_TAG_VALID)));
	Assert(BUF_STATE_GET_REFCOUNT(buf_state) > 0);
	Assert(BUF_STATE_GET_REFCOUNT(pg_atomic_read_u64(&buf_hdr->state)) > 0);

	return true;
}

static Buffer
GetVictimBuffer(BufferAccessStrategy strategy, IOContext io_context)
{
	BufferDesc *buf_hdr;
	Buffer		buf;
	uint64		buf_state;
	bool		from_ring;

	/*
	 * Ensure, before we pin a victim buffer, that there's a free refcount
	 * entry and resource owner slot for the pin.
	 */
	ReservePrivateRefCountEntry();
	ResourceOwnerEnlarge(CurrentResourceOwner);

	/* we return here if a prospective victim buffer gets used concurrently */
again:

	/*
	 * Select a victim buffer.  The buffer is returned pinned and owned by
	 * this backend.
	 */
	buf_hdr = StrategyGetBuffer(strategy, &buf_state, &from_ring);
	if (calico_is_active() && (buf_state & BM_TAG_VALID) && calico_frame_entries != NULL)
		__builtin_prefetch(&calico_frame_entries[buf_hdr->buf_id], 0, 3);
	buf = BufferDescriptorGetBuffer(buf_hdr);

	/*
	 * We shouldn't have any other pins for this buffer.
	 */
	CheckBufferIsPinnedOnce(buf);

	/*
	 * If the buffer was dirty, try to write it out.  There is a race
	 * condition here, another backend could dirty the buffer between
	 * StrategyGetBuffer() checking that it is not in use and invalidating the
	 * buffer below. That's addressed by InvalidateVictimBuffer() verifying
	 * that the buffer is not dirty.
	 */
	if (buf_state & BM_DIRTY)
	{
		Assert(buf_state & BM_TAG_VALID);
		Assert(buf_state & BM_VALID);

		/*
		 * We need a share-exclusive lock on the buffer contents to write it
		 * out (else we might write invalid data, eg because someone else is
		 * compacting the page contents while we write).  We must use a
		 * conditional lock acquisition here to avoid deadlock.  Even though
		 * the buffer was not pinned (and therefore surely not locked) when
		 * StrategyGetBuffer returned it, someone else could have pinned and
		 * (share-)exclusive-locked it by the time we get here. If we try to
		 * get the lock unconditionally, we'd block waiting for them; if they
		 * later block waiting for us, deadlock ensues. (This has been
		 * observed to happen when two backends are both trying to split btree
		 * index pages, and the second one just happens to be trying to split
		 * the page the first one got from StrategyGetBuffer.)
		 */
		if (!BufferLockConditional(buf, buf_hdr, BUFFER_LOCK_SHARE_EXCLUSIVE))
		{
			/*
			 * Someone else has locked the buffer, so give it up and loop back
			 * to get another one.
			 */
			UnpinBuffer(buf_hdr);
			goto again;
		}

		/*
		 * If using a nondefault strategy, and this victim came from the
		 * strategy ring, let the strategy decide whether to reject it when
		 * reusing it would require a WAL flush.  This only applies to
		 * permanent buffers; unlogged buffers can have fake LSNs, so
		 * XLogNeedsFlush() is not meaningful for them.
		 *
		 * We need to hold the content lock in at least share-exclusive mode
		 * to safely inspect the page LSN, so this couldn't have been done
		 * inside StrategyGetBuffer().
		 */
		if (strategy && from_ring &&
			buf_state & BM_PERMANENT &&
			XLogNeedsFlush(BufferGetLSN(buf_hdr)) &&
			StrategyRejectBuffer(strategy, buf_hdr, from_ring))
		{
			UnlockReleaseBuffer(buf);
			goto again;
		}

		/* OK, do the I/O */
		FlushBuffer(buf_hdr, NULL, IOOBJECT_RELATION, io_context);
		LockBuffer(buf, BUFFER_LOCK_UNLOCK);

		ScheduleBufferTagForWriteback(&BackendWritebackContext, io_context,
									  &buf_hdr->tag);
	}

	if (buf_state & BM_VALID)
	{
		/*
		 * When a BufferAccessStrategy is in use, blocks evicted from shared
		 * buffers are counted as IOOP_EVICT in the corresponding context
		 * (e.g. IOCONTEXT_BULKWRITE). Shared buffers are evicted by a
		 * strategy in two cases: 1) while initially claiming buffers for the
		 * strategy ring 2) to replace an existing strategy ring buffer
		 * because it is pinned or in use and cannot be reused.
		 *
		 * Blocks evicted from buffers already in the strategy ring are
		 * counted as IOOP_REUSE in the corresponding strategy context.
		 *
		 * At this point, we can accurately count evictions and reuses,
		 * because we have successfully claimed the valid buffer. Previously,
		 * we may have been forced to release the buffer due to concurrent
		 * pinners or erroring out.
		 */
		pgstat_count_io_op(IOOBJECT_RELATION, io_context,
						   from_ring ? IOOP_REUSE : IOOP_EVICT, 1, 0);
	}

	/*
	 * If the buffer has an entry in the buffer mapping table, delete it. This
	 * can fail because another backend could have pinned or dirtied the
	 * buffer.
	 */
	if ((buf_state & BM_TAG_VALID) && !InvalidateVictimBuffer(buf_hdr))
	{
		UnpinBuffer(buf_hdr);
		goto again;
	}

	/* a final set of sanity checks */
#ifdef USE_ASSERT_CHECKING
	buf_state = pg_atomic_read_u64(&buf_hdr->state);

	Assert(BUF_STATE_GET_REFCOUNT(buf_state) == 1);
	Assert(!(buf_state & (BM_TAG_VALID | BM_VALID | BM_DIRTY)));

	CheckBufferIsPinnedOnce(buf);
#endif

	return buf;
}

/*
 * Logic shared between ExtendBufferedRelBy(), ExtendBufferedRelTo(). Just to
 * avoid duplicating the tracing and relpersistence related logic.
 */
static BlockNumber
ExtendBufferedRelCommon(BufferManagerRelation bmr,
						ForkNumber fork,
						BufferAccessStrategy strategy,
						uint32 flags,
						uint32 extend_by,
						BlockNumber extend_upto,
						Buffer *buffers,
						uint32 *extended_by)
{
	BlockNumber first_block;

	TRACE_POSTGRESQL_BUFFER_EXTEND_START(fork,
										 BMR_GET_SMGR(bmr)->smgr_rlocator.locator.spcOid,
										 BMR_GET_SMGR(bmr)->smgr_rlocator.locator.dbOid,
										 BMR_GET_SMGR(bmr)->smgr_rlocator.locator.relNumber,
										 BMR_GET_SMGR(bmr)->smgr_rlocator.backend,
										 extend_by);

	if (bmr.relpersistence == RELPERSISTENCE_TEMP)
	{
		/*
		 * Reject attempts to extend non-local temporary relations; we have no
		 * ability to transfer about-to-be-created local buffers into the
		 * owning session's local buffers.  This is the canonical place for
		 * the check, covering any attempt to extend a non-local temporary
		 * relation.
		 */
		if (bmr.rel && RELATION_IS_OTHER_TEMP(bmr.rel))
			ereport(ERROR,
					(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
					 errmsg("cannot access temporary tables of other sessions")));

		first_block = ExtendBufferedRelLocal(bmr, fork, flags,
											 extend_by, extend_upto,
											 buffers, &extend_by);
	}
	else
		first_block = ExtendBufferedRelShared(bmr, fork, strategy, flags,
											  extend_by, extend_upto,
											  buffers, &extend_by);
	*extended_by = extend_by;

	TRACE_POSTGRESQL_BUFFER_EXTEND_DONE(fork,
										BMR_GET_SMGR(bmr)->smgr_rlocator.locator.spcOid,
										BMR_GET_SMGR(bmr)->smgr_rlocator.locator.dbOid,
										BMR_GET_SMGR(bmr)->smgr_rlocator.locator.relNumber,
										BMR_GET_SMGR(bmr)->smgr_rlocator.backend,
										*extended_by,
										first_block);

	return first_block;
}

/*
 * Implementation of ExtendBufferedRelBy() and ExtendBufferedRelTo() for
 * shared buffers.
 */
static BlockNumber
ExtendBufferedRelShared(BufferManagerRelation bmr,
						ForkNumber fork,
						BufferAccessStrategy strategy,
						uint32 flags,
						uint32 extend_by,
						BlockNumber extend_upto,
						Buffer *buffers,
						uint32 *extended_by)
{
	BlockNumber first_block;
	IOContext	io_context = IOContextForStrategy(strategy);
	instr_time	io_start;

	LimitAdditionalPins(&extend_by);

	/*
	 * Acquire victim buffers for extension without holding extension lock.
	 * Writing out victim buffers is the most expensive part of extending the
	 * relation, particularly when doing so requires WAL flushes. Zeroing out
	 * the buffers is also quite expensive, so do that before holding the
	 * extension lock as well.
	 *
	 * These pages are pinned by us and not valid. While we hold the pin they
	 * can't be acquired as victim buffers by another backend.
	 */
	for (uint32 i = 0; i < extend_by; i++)
	{
		Block		buf_block;

		buffers[i] = GetVictimBuffer(strategy, io_context);
		buf_block = BufHdrGetBlock(GetBufferDescriptor(buffers[i] - 1));

		/* new buffers are zero-filled */
		MemSet(buf_block, 0, BLCKSZ);
	}

	/*
	 * Lock relation against concurrent extensions, unless requested not to.
	 *
	 * We use the same extension lock for all forks. That's unnecessarily
	 * restrictive, but currently extensions for forks don't happen often
	 * enough to make it worth locking more granularly.
	 *
	 * Note that another backend might have extended the relation by the time
	 * we get the lock.
	 */
	if (!(flags & EB_SKIP_EXTENSION_LOCK))
		LockRelationForExtension(bmr.rel, ExclusiveLock);

	/*
	 * If requested, invalidate size cache, so that smgrnblocks asks the
	 * kernel.
	 */
	if (flags & EB_CLEAR_SIZE_CACHE)
		BMR_GET_SMGR(bmr)->smgr_cached_nblocks[fork] = InvalidBlockNumber;

	first_block = smgrnblocks(BMR_GET_SMGR(bmr), fork);

	/*
	 * Now that we have the accurate relation size, check if the caller wants
	 * us to extend to only up to a specific size. If there were concurrent
	 * extensions, we might have acquired too many buffers and need to release
	 * them.
	 */
	if (extend_upto != InvalidBlockNumber)
	{
		uint32		orig_extend_by = extend_by;

		if (first_block > extend_upto)
			extend_by = 0;
		else if ((uint64) first_block + extend_by > extend_upto)
			extend_by = extend_upto - first_block;

		for (uint32 i = extend_by; i < orig_extend_by; i++)
		{
			BufferDesc *buf_hdr = GetBufferDescriptor(buffers[i] - 1);

			UnpinBuffer(buf_hdr);
		}

		if (extend_by == 0)
		{
			if (!(flags & EB_SKIP_EXTENSION_LOCK))
				UnlockRelationForExtension(bmr.rel, ExclusiveLock);
			*extended_by = extend_by;
			return first_block;
		}
	}

	/* Fail if relation is already at maximum possible length */
	if ((uint64) first_block + extend_by >= MaxBlockNumber)
		ereport(ERROR,
				(errcode(ERRCODE_PROGRAM_LIMIT_EXCEEDED),
				 errmsg("cannot extend relation %s beyond %u blocks",
						relpath(BMR_GET_SMGR(bmr)->smgr_rlocator, fork).str,
						MaxBlockNumber)));

	/*
	 * Insert buffers into buffer table, mark as IO_IN_PROGRESS.
	 *
	 * This needs to happen before we extend the relation, because as soon as
	 * we do, other backends can start to read in those pages.
	 */
	for (uint32 i = 0; i < extend_by; i++)
	{
		Buffer		victim_buf = buffers[i];
		BufferDesc *victim_buf_hdr = GetBufferDescriptor(victim_buf - 1);
		BufferTag	tag;
		uint32		hash;
		LWLock	   *partition_lock;
		int			existing_id;

		/* in case we need to pin an existing buffer below */
		ResourceOwnerEnlarge(CurrentResourceOwner);
		ReservePrivateRefCountEntry();

		InitBufferTag(&tag, &BMR_GET_SMGR(bmr)->smgr_rlocator.locator, fork,
					  first_block + i);
		hash = BufTableHashCode(&tag);
		partition_lock = BufMappingPartitionLock(hash);

		LWLockAcquire(partition_lock, LW_EXCLUSIVE);

		existing_id = BufTableInsert(&tag, hash, victim_buf_hdr->buf_id);

		/*
		 * We get here only in the corner case where we are trying to extend
		 * the relation but we found a pre-existing buffer. This can happen
		 * because a prior attempt at extending the relation failed, and
		 * because mdread doesn't complain about reads beyond EOF (when
		 * zero_damaged_pages is ON) and so a previous attempt to read a block
		 * beyond EOF could have left a "valid" zero-filled buffer.
		 *
		 * This has also been observed when relation was overwritten by
		 * external process. Since the legitimate cases should always have
		 * left a zero-filled buffer, complain if not PageIsNew.
		 */
		if (existing_id >= 0)
		{
			BufferDesc *existing_hdr = GetBufferDescriptor(existing_id);
			Block		buf_block;
			bool		valid;

			/*
			 * Pin the existing buffer before releasing the partition lock,
			 * preventing it from being evicted.
			 */
			valid = PinBuffer(existing_hdr, strategy, false);

			LWLockRelease(partition_lock);
			UnpinBuffer(victim_buf_hdr);

			buffers[i] = BufferDescriptorGetBuffer(existing_hdr);
			buf_block = BufHdrGetBlock(existing_hdr);

			if (valid && !PageIsNew((Page) buf_block))
				ereport(ERROR,
						(errmsg("unexpected data beyond EOF in block %u of relation \"%s\"",
								existing_hdr->tag.blockNum,
								relpath(BMR_GET_SMGR(bmr)->smgr_rlocator, fork).str)));

			/*
			 * We *must* do smgr[zero]extend before succeeding, else the page
			 * will not be reserved by the kernel, and the next P_NEW call
			 * will decide to return the same page.  Clear the BM_VALID bit,
			 * do StartSharedBufferIO() and proceed.
			 *
			 * Loop to handle the very small possibility that someone re-sets
			 * BM_VALID between our clearing it and StartSharedBufferIO
			 * inspecting it.
			 */
			while (true)
			{
				StartBufferIOResult sbres;

				pg_atomic_fetch_and_u64(&existing_hdr->state, ~BM_VALID);
				calico_mark_frame_unready(existing_hdr);

				sbres = StartSharedBufferIO(existing_hdr, true, true, NULL);

				if (sbres != BUFFER_IO_ALREADY_DONE)
					break;
			}
		}
		else
		{
			uint64		buf_state;
			uint64		set_bits = 0;

			buf_state = LockBufHdr(victim_buf_hdr);

			/* some sanity checks while we hold the buffer header lock */
			Assert(!(buf_state & (BM_VALID | BM_TAG_VALID | BM_DIRTY)));
			Assert(BUF_STATE_GET_REFCOUNT(buf_state) == 1);

			victim_buf_hdr->tag = tag;

			set_bits |= BM_TAG_VALID | BUF_USAGECOUNT_ONE;
			if (bmr.relpersistence == RELPERSISTENCE_PERMANENT || fork == INIT_FORKNUM)
				set_bits |= BM_PERMANENT;

			UnlockBufHdrExt(victim_buf_hdr, buf_state,
							set_bits, 0,
							0);

			LWLockRelease(partition_lock);

			/* XXX: could combine the locked operations in it with the above */
			StartSharedBufferIO(victim_buf_hdr, true, true, NULL);
		}
	}

	io_start = pgstat_prepare_io_time(track_io_timing);

	/*
	 * Note: if smgrzeroextend fails, we will end up with buffers that are
	 * allocated but not marked BM_VALID.  The next relation extension will
	 * still select the same block number (because the relation didn't get any
	 * longer on disk) and so future attempts to extend the relation will find
	 * the same buffers (if they have not been recycled) but come right back
	 * here to try smgrzeroextend again.
	 *
	 * We don't need to set checksum for all-zero pages.
	 */
	smgrzeroextend(BMR_GET_SMGR(bmr), fork, first_block, extend_by, false);

	/*
	 * Release the file-extension lock; it's now OK for someone else to extend
	 * the relation some more.
	 *
	 * We remove IO_IN_PROGRESS after this, as waking up waiting backends can
	 * take noticeable time.
	 */
	if (!(flags & EB_SKIP_EXTENSION_LOCK))
		UnlockRelationForExtension(bmr.rel, ExclusiveLock);

	pgstat_count_io_op_time(IOOBJECT_RELATION, io_context, IOOP_EXTEND,
							io_start, 1, extend_by * BLCKSZ);

	/* Set BM_VALID, terminate IO, and wake up any waiters */
	for (uint32 i = 0; i < extend_by; i++)
	{
		Buffer		buf = buffers[i];
		BufferDesc *buf_hdr = GetBufferDescriptor(buf - 1);
		bool		lock = false;

		if (flags & EB_LOCK_FIRST && i == 0)
			lock = true;
		else if (flags & EB_LOCK_TARGET)
		{
			Assert(extend_upto != InvalidBlockNumber);
			if (first_block + i + 1 == extend_upto)
				lock = true;
		}

		if (lock)
			LockBuffer(buf, BUFFER_LOCK_EXCLUSIVE);

		TerminateBufferIO(buf_hdr, false, BM_VALID, true, false);
		calico_mark_frame_ready(buf_hdr);
	}

	pgBufferUsage.shared_blks_written += extend_by;

	*extended_by = extend_by;

	return first_block;
}

/*
 * ReleaseAndReadBuffer -- combine ReleaseBuffer() and ReadBuffer()
 *
 * Formerly, this saved one cycle of acquiring/releasing the BufMgrLock
 * compared to calling the two routines separately.  Now it's mainly just
 * a convenience function.  However, if the passed buffer is valid and
 * already contains the desired block, we just return it as-is; and that
 * does save considerable work compared to a full release and reacquire.
 *
 * Note: it is OK to pass buffer == InvalidBuffer, indicating that no old
 * buffer actually needs to be released.  This case is the same as ReadBuffer,
 * but can save some tests in the caller.
 */
Buffer
ReleaseAndReadBuffer(Buffer buffer,
					 Relation relation,
					 BlockNumber blockNum)
{
	ForkNumber	forkNum = MAIN_FORKNUM;
	BufferDesc *bufHdr;

	if (unlikely(!calico_local_active))
		return native_ReleaseAndReadBuffer(buffer, relation, blockNum);

	if (BufferIsValid(buffer))
	{
		Assert(BufferIsPinned(buffer));
		if (BufferIsLocal(buffer))
		{
			bufHdr = GetLocalBufferDescriptor(-buffer - 1);
			if (bufHdr->tag.blockNum == blockNum &&
				BufTagMatchesRelFileLocator(&bufHdr->tag, &relation->rd_locator) &&
				BufTagGetForkNum(&bufHdr->tag) == forkNum)
				return buffer;
			UnpinLocalBuffer(buffer);
		}
		else
		{
			bufHdr = GetBufferDescriptor(buffer - 1);
			/* we have pin, so it's ok to examine tag without spinlock */
			if (bufHdr->tag.blockNum == blockNum &&
				BufTagMatchesRelFileLocator(&bufHdr->tag, &relation->rd_locator) &&
				BufTagGetForkNum(&bufHdr->tag) == forkNum)
				return buffer;
			UnpinBuffer(bufHdr);
		}
	}

	return ReadBuffer(relation, blockNum);
}

/*
 * PinBuffer -- make buffer unavailable for replacement.
 *
 * For the default access strategy, the buffer's usage_count is incremented
 * when we first pin it; for other strategies we just make sure the usage_count
 * isn't zero.  (The idea of the latter is that we don't want synchronized
 * heap scans to inflate the count, but we need it to not be zero to discourage
 * other backends from stealing buffers from our ring.  As long as we cycle
 * through the ring faster than the global clock-sweep cycles, buffers in
 * our ring won't be chosen as victims for replacement by other backends.)
 *
 * This should be applied only to shared buffers, never local ones.
 *
 * Since buffers are pinned/unpinned very frequently, pin buffers without
 * taking the buffer header lock; instead update the state variable in loop of
 * CAS operations. Hopefully it's just a single CAS.
 *
 * Note that ResourceOwnerEnlarge() and ReservePrivateRefCountEntry()
 * must have been done already.
 *
 * Returns true if buffer is BM_VALID, else false.  This provision allows
 * some callers to avoid an extra spinlock cycle.  If skip_if_not_valid is
 * true, then a false return value also indicates that the buffer was
 * (recently) invalid and has not been pinned.
 */
static bool
PinBuffer(BufferDesc *buf, BufferAccessStrategy strategy,
		  bool skip_if_not_valid)
{
	Buffer		b = BufferDescriptorGetBuffer(buf);
	bool		result;
	PrivateRefCountEntry *ref;

	Assert(!BufferIsLocal(b));
	Assert(ReservedRefCountSlot != -1);

	ref = GetPrivateRefCountEntry(b, true);

	if (ref == NULL)
	{
		uint64		buf_state;
		uint64		old_buf_state;

		old_buf_state = pg_atomic_read_u64(&buf->state);
		for (;;)
		{
			if (unlikely(skip_if_not_valid && !(old_buf_state & BM_VALID)))
				return false;

			/*
			 * We're not allowed to increase the refcount while the buffer
			 * header spinlock is held. Wait for the lock to be released.
			 */
			if (unlikely(old_buf_state & BM_LOCKED))
			{
				old_buf_state = WaitBufHdrUnlocked(buf);

				/* perform checks at the top of the loop again */
				continue;
			}

			buf_state = old_buf_state;

			/* increase refcount */
			buf_state += BUF_REFCOUNT_ONE;

			if (strategy == NULL)
			{
				/* Default case: increase usagecount unless already max. */
				if (BUF_STATE_GET_USAGECOUNT(buf_state) < BM_MAX_USAGE_COUNT)
					buf_state += BUF_USAGECOUNT_ONE;
			}
			else
			{
				/*
				 * Ring buffers shouldn't evict others from pool.  Thus we
				 * don't make usagecount more than 1.
				 */
				if (BUF_STATE_GET_USAGECOUNT(buf_state) == 0)
					buf_state += BUF_USAGECOUNT_ONE;
			}

			if (pg_atomic_compare_exchange_u64(&buf->state, &old_buf_state,
											   buf_state))
			{
				result = (buf_state & BM_VALID) != 0;

				TrackNewBufferPin(b);
				break;
			}
		}
	}
	else
	{
		/*
		 * If we previously pinned the buffer, it is likely to be valid, but
		 * it may not be if StartReadBuffers() was called and
		 * WaitReadBuffers() hasn't been called yet.  We'll check by loading
		 * the flags without locking.  This is racy, but it's OK to return
		 * false spuriously: when WaitReadBuffers() calls StartBufferIO(),
		 * it'll see that it's now valid.
		 *
		 * Note: We deliberately avoid a Valgrind client request here.
		 * Individual access methods can optionally superimpose buffer page
		 * client requests on top of our client requests to enforce that
		 * buffers are only accessed while locked (and pinned).  It's possible
		 * that the buffer page is legitimately non-accessible here.  We
		 * cannot meddle with that.
		 */
		result = (pg_atomic_read_u64(&buf->state) & BM_VALID) != 0;

		Assert(ref->data.refcount > 0);
		ref->data.refcount++;
		ResourceOwnerRememberBuffer(CurrentResourceOwner, b);
	}

	return result;
}

bool
CalicoTryPinBuffer(BufferDesc *buf, BufferAccessStrategy strategy)
{
	Buffer		b = BufferDescriptorGetBuffer(buf);
	bool		result;
	PrivateRefCountEntry *ref;

	Assert(!BufferIsLocal(b));
	Assert(ReservedRefCountSlot != -1);

	ref = GetPrivateRefCountEntry(b, true);

	if (ref == NULL)
	{
		uint64		buf_state;
		uint64		old_buf_state;

		old_buf_state = pg_atomic_read_u64(&buf->state);
		for (;;)
		{
			if (unlikely(!(old_buf_state & BM_VALID)))
				return false;

			/*
			 * We're not allowed to increase the refcount while the buffer
			 * header spinlock is held. Wait for the lock to be released.
			 */
			if (unlikely(old_buf_state & BM_LOCKED))
			{
				old_buf_state = WaitBufHdrUnlocked(buf);

				/* perform checks at the top of the loop again */
				continue;
			}

			buf_state = old_buf_state;

			/* increase refcount */
			buf_state += BUF_REFCOUNT_ONE;

			if (strategy == NULL)
			{
				/* Default case: increase usagecount unless already max. */
				if (BUF_STATE_GET_USAGECOUNT(buf_state) < BM_MAX_USAGE_COUNT)
					buf_state += BUF_USAGECOUNT_ONE;
			}
			else
			{
				/*
				 * Ring buffers shouldn't evict others from pool.  Thus we
				 * don't make usagecount more than 1.
				 */
				if (BUF_STATE_GET_USAGECOUNT(buf_state) == 0)
					buf_state += BUF_USAGECOUNT_ONE;
			}

			if (pg_atomic_compare_exchange_u64(&buf->state, &old_buf_state,
											   buf_state))
			{
				result = (buf_state & BM_VALID) != 0;

				TrackNewBufferPin(b);
				break;
			}
		}
	}
	else
	{
		/*
		 * If we previously pinned the buffer, it is likely to be valid, but
		 * it may not be if StartReadBuffers() was called and
		 * WaitReadBuffers() hasn't been called yet.  We'll check by loading
		 * the flags without locking.  This is racy, but it's OK to return
		 * false spuriously: when WaitReadBuffers() calls StartBufferIO(),
		 * it'll see that it's now valid.
		 *
		 * Note: We deliberately avoid a Valgrind client request here.
		 * Individual access methods can optionally superimpose buffer page
		 * client requests on top of our client requests to enforce that
		 * buffers are only accessed while locked (and pinned).  It's possible
		 * that the buffer page is legitimately non-accessible here.  We
		 * cannot meddle with that.
		 */
		result = (pg_atomic_read_u64(&buf->state) & BM_VALID) != 0;
		if (!result)
			return false;

		Assert(ref->data.refcount > 0);
		ref->data.refcount++;
		ResourceOwnerRememberBuffer(CurrentResourceOwner, b);
	}

	return result;
}

/*
 * PinBuffer_Locked -- as above, but caller already locked the buffer header.
 * The spinlock is released before return.
 *
 * As this function is called with the spinlock held, the caller has to
 * previously call ReservePrivateRefCountEntry() and
 * ResourceOwnerEnlarge(CurrentResourceOwner);
 *
 * Currently, no callers of this function want to modify the buffer's
 * usage_count at all, so there's no need for a strategy parameter.
 * Also we don't bother with a BM_VALID test (the caller could check that for
 * itself).
 *
 * Also all callers only ever use this function when it's known that the
 * buffer can't have a preexisting pin by this backend. That allows us to skip
 * searching the private refcount array & hash, which is a boon, because the
 * spinlock is still held.
 *
 * Note: use of this routine is frequently mandatory, not just an optimization
 * to save a spin lock/unlock cycle, because we need to pin a buffer before
 * its state can change under us.
 */
static void
PinBuffer_Locked(BufferDesc *buf)
{
	uint64		old_buf_state;

	/*
	 * As explained, We don't expect any preexisting pins. That allows us to
	 * manipulate the PrivateRefCount after releasing the spinlock
	 */
	Assert(GetPrivateRefCountEntry(BufferDescriptorGetBuffer(buf), false) == NULL);

	/*
	 * Since we hold the buffer spinlock, we can update the buffer state and
	 * release the lock in one operation.
	 */
	old_buf_state = pg_atomic_read_u64(&buf->state);

	UnlockBufHdrExt(buf, old_buf_state,
					0, 0, 1);

	TrackNewBufferPin(BufferDescriptorGetBuffer(buf));
}

/*
 * Support for waking up another backend that is waiting for the cleanup lock
 * to be released using BM_PIN_COUNT_WAITER.
 *
 * See LockBufferForCleanup().
 *
 * Expected to be called just after releasing a buffer pin (in a BufferDesc,
 * not just reducing the backend-local pincount for the buffer).
 */
static void
WakePinCountWaiter(BufferDesc *buf)
{
	/*
	 * Acquire the buffer header lock, re-check that there's a waiter. Another
	 * backend could have unpinned this buffer, and already woken up the
	 * waiter.
	 *
	 * There's no danger of the buffer being replaced after we unpinned it
	 * above, as it's pinned by the waiter. The waiter removes
	 * BM_PIN_COUNT_WAITER if it stops waiting for a reason other than this
	 * backend waking it up.
	 */
	uint64		buf_state = LockBufHdr(buf);

	if ((buf_state & BM_PIN_COUNT_WAITER) &&
		BUF_STATE_GET_REFCOUNT(buf_state) == 1)
	{
		/* we just released the last pin other than the waiter's */
		int			wait_backend_pgprocno = buf->wait_backend_pgprocno;

		UnlockBufHdrExt(buf, buf_state,
						0, BM_PIN_COUNT_WAITER,
						0);
		ProcSendSignal(wait_backend_pgprocno);
	}
	else
		UnlockBufHdr(buf);
}

/*
 * UnpinBuffer -- make buffer available for replacement.
 *
 * This should be applied only to shared buffers, never local ones.  This
 * always adjusts CurrentResourceOwner.
 */
static void
UnpinBuffer(BufferDesc *buf)
{
	Buffer		b = BufferDescriptorGetBuffer(buf);

	ResourceOwnerForgetBuffer(CurrentResourceOwner, b);
	UnpinBufferNoOwner(buf);
}

static void
UnpinBufferNoOwner(BufferDesc *buf)
{
	PrivateRefCountEntry *ref;
	Buffer		b = BufferDescriptorGetBuffer(buf);

	Assert(!BufferIsLocal(b));

	/* not moving as we're likely deleting it soon anyway */
	ref = GetPrivateRefCountEntry(b, false);
	Assert(ref != NULL);
	Assert(ref->data.refcount > 0);
	ref->data.refcount--;
	if (ref->data.refcount == 0)
	{
		uint64		old_buf_state;

		/*
		 * Mark buffer non-accessible to Valgrind.
		 *
		 * Note that the buffer may have already been marked non-accessible
		 * within access method code that enforces that buffers are only
		 * accessed while a buffer lock is held.
		 */
		VALGRIND_MAKE_MEM_NOACCESS(BufHdrGetBlock(buf), BLCKSZ);

		/*
		 * I'd better not still hold the buffer content lock. Can't use
		 * BufferIsLockedByMe(), as that asserts the buffer is pinned.
		 */
		Assert(!BufferLockHeldByMe(buf));

		/* decrement the shared reference count */
		old_buf_state = pg_atomic_fetch_sub_u64(&buf->state, BUF_REFCOUNT_ONE);

		/* Support LockBufferForCleanup() */
		if (old_buf_state & BM_PIN_COUNT_WAITER)
			WakePinCountWaiter(buf);

		ForgetPrivateRefCountEntry(ref);
	}
}

/*
 * FlushBuffer
 *		Physically write out a shared buffer.
 *
 * NOTE: this actually just passes the buffer contents to the kernel; the
 * real write to disk won't happen until the kernel feels like it.  This
 * is okay from our point of view since we can redo the changes from WAL.
 * However, we will need to force the changes to disk via fsync before
 * we can checkpoint WAL.
 *
 * The caller must hold a pin on the buffer and have
 * (share-)exclusively-locked the buffer contents.
 *
 * If the caller has an smgr reference for the buffer's relation, pass it
 * as the second parameter.  If not, pass NULL.
 */
static void
FlushBuffer(BufferDesc *buf, SMgrRelation reln, IOObject io_object,
			IOContext io_context)
{
	XLogRecPtr	recptr;
	ErrorContextCallback errcallback;
	instr_time	io_start;
	Block		bufBlock;

	Assert(BufferLockHeldByMeInMode(buf, BUFFER_LOCK_EXCLUSIVE) ||
		   BufferLockHeldByMeInMode(buf, BUFFER_LOCK_SHARE_EXCLUSIVE));

	/*
	 * Try to start an I/O operation.  If StartBufferIO returns false, then
	 * someone else flushed the buffer before we could, so we need not do
	 * anything.
	 */
	if (StartSharedBufferIO(buf, false, true, NULL) == BUFFER_IO_ALREADY_DONE)
		return;

	/* Setup error traceback support for ereport() */
	errcallback.callback = shared_buffer_write_error_callback;
	errcallback.arg = buf;
	errcallback.previous = error_context_stack;
	error_context_stack = &errcallback;

	/* Find smgr relation for buffer */
	if (reln == NULL)
		reln = smgropen(BufTagGetRelFileLocator(&buf->tag), INVALID_PROC_NUMBER);

	TRACE_POSTGRESQL_BUFFER_FLUSH_START(BufTagGetForkNum(&buf->tag),
										buf->tag.blockNum,
										reln->smgr_rlocator.locator.spcOid,
										reln->smgr_rlocator.locator.dbOid,
										reln->smgr_rlocator.locator.relNumber);

	/*
	 * As we hold at least a share-exclusive lock on the buffer, the LSN
	 * cannot change during the flush (and thus can't be torn).
	 */
	recptr = BufferGetLSN(buf);

	/*
	 * Force XLOG flush up to buffer's LSN.  This implements the basic WAL
	 * rule that log updates must hit disk before any of the data-file changes
	 * they describe do.
	 *
	 * However, this rule does not apply to unlogged relations, which will be
	 * lost after a crash anyway.  Most unlogged relation pages do not bear
	 * LSNs since we never emit WAL records for them, and therefore flushing
	 * up through the buffer LSN would be useless, but harmless.  However,
	 * some index AMs use LSNs internally to detect concurrent page
	 * modifications, and therefore unlogged index pages bear "fake" LSNs
	 * generated by XLogGetFakeLSN.  It is unlikely but possible that the fake
	 * LSN counter could advance past the WAL insertion point; and if it did
	 * happen, attempting to flush WAL through that location would fail, with
	 * disastrous system-wide consequences.  To make sure that can't happen,
	 * skip the flush if the buffer isn't permanent.
	 */
	if (pg_atomic_read_u64(&buf->state) & BM_PERMANENT)
		XLogFlush(recptr);

	/*
	 * Now it's safe to write the buffer to disk. Note that no one else should
	 * have been able to write it, while we were busy with log flushing,
	 * because we got the exclusive right to perform I/O by setting the
	 * BM_IO_IN_PROGRESS bit.
	 */
	bufBlock = BufHdrGetBlock(buf);

	/* Update page checksum if desired. */
	PageSetChecksum((Page) bufBlock, buf->tag.blockNum);

	io_start = pgstat_prepare_io_time(track_io_timing);

	smgrwrite(reln,
			  BufTagGetForkNum(&buf->tag),
			  buf->tag.blockNum,
			  bufBlock,
			  false);

	/*
	 * When a strategy is in use, only flushes of dirty buffers already in the
	 * strategy ring are counted as strategy writes (IOCONTEXT
	 * [BULKREAD|BULKWRITE|VACUUM] IOOP_WRITE) for the purpose of IO
	 * statistics tracking.
	 *
	 * If a shared buffer initially added to the ring must be flushed before
	 * being used, this is counted as an IOCONTEXT_NORMAL IOOP_WRITE.
	 *
	 * If a shared buffer which was added to the ring later because the
	 * current strategy buffer is pinned or in use or because all strategy
	 * buffers were dirty and rejected (for BAS_BULKREAD operations only)
	 * requires flushing, this is counted as an IOCONTEXT_NORMAL IOOP_WRITE
	 * (from_ring will be false).
	 *
	 * When a strategy is not in use, the write can only be a "regular" write
	 * of a dirty shared buffer (IOCONTEXT_NORMAL IOOP_WRITE).
	 */
	pgstat_count_io_op_time(io_object, io_context,
							IOOP_WRITE, io_start, 1, BLCKSZ);

	pgBufferUsage.shared_blks_written++;

	/*
	 * Mark the buffer as clean and end the BM_IO_IN_PROGRESS state.
	 */
	TerminateBufferIO(buf, true, 0, true, false);

	TRACE_POSTGRESQL_BUFFER_FLUSH_DONE(BufTagGetForkNum(&buf->tag),
									   buf->tag.blockNum,
									   reln->smgr_rlocator.locator.spcOid,
									   reln->smgr_rlocator.locator.dbOid,
									   reln->smgr_rlocator.locator.relNumber);

	/* Pop the error context stack */
	error_context_stack = errcallback.previous;
}

/* ---------------------------------------------------------------------
 *		RelationCopyStorageUsingBuffer
 *
 *		Copy fork's data using bufmgr.  Same as RelationCopyStorage but instead
 *		of using smgrread and smgrextend this will copy using bufmgr APIs.
 *
 *		Refer comments atop CreateAndCopyRelationData() for details about
 *		'permanent' parameter.
 * --------------------------------------------------------------------
 */
static void
RelationCopyStorageUsingBuffer(RelFileLocator srclocator,
							   RelFileLocator dstlocator,
							   ForkNumber forkNum, bool permanent)
{
	Buffer		srcBuf;
	Buffer		dstBuf;
	Page		srcPage;
	Page		dstPage;
	bool		use_wal;
	BlockNumber nblocks;
	BlockNumber blkno;
	PGIOAlignedBlock buf;
	BufferAccessStrategy bstrategy_src;
	BufferAccessStrategy bstrategy_dst;
	BlockRangeReadStreamPrivate p;
	ReadStream *src_stream;
	SMgrRelation src_smgr;

	/*
	 * In general, we want to write WAL whenever wal_level > 'minimal', but we
	 * can skip it when copying any fork of an unlogged relation other than
	 * the init fork.
	 */
	use_wal = XLogIsNeeded() && (permanent || forkNum == INIT_FORKNUM);

	/* Get number of blocks in the source relation. */
	nblocks = smgrnblocks(smgropen(srclocator, INVALID_PROC_NUMBER),
						  forkNum);

	/* Nothing to copy; just return. */
	if (nblocks == 0)
		return;

	/*
	 * Bulk extend the destination relation of the same size as the source
	 * relation before starting to copy block by block.
	 */
	memset(buf.data, 0, BLCKSZ);
	smgrextend(smgropen(dstlocator, INVALID_PROC_NUMBER), forkNum, nblocks - 1,
			   buf.data, true);

	/* This is a bulk operation, so use buffer access strategies. */
	bstrategy_src = GetAccessStrategy(BAS_BULKREAD);
	bstrategy_dst = GetAccessStrategy(BAS_BULKWRITE);

	/* Initialize streaming read */
	p.current_blocknum = 0;
	p.last_exclusive = nblocks;
	src_smgr = smgropen(srclocator, INVALID_PROC_NUMBER);

	/*
	 * It is safe to use batchmode as block_range_read_stream_cb takes no
	 * locks.
	 */
	src_stream = read_stream_begin_smgr_relation(READ_STREAM_FULL |
												 READ_STREAM_USE_BATCHING,
												 bstrategy_src,
												 src_smgr,
												 permanent ? RELPERSISTENCE_PERMANENT : RELPERSISTENCE_UNLOGGED,
												 forkNum,
												 block_range_read_stream_cb,
												 &p,
												 0);

	/* Iterate over each block of the source relation file. */
	for (blkno = 0; blkno < nblocks; blkno++)
	{
		CHECK_FOR_INTERRUPTS();

		/* Read block from source relation. */
		srcBuf = read_stream_next_buffer(src_stream, NULL);
		LockBuffer(srcBuf, BUFFER_LOCK_SHARE);
		srcPage = BufferGetPage(srcBuf);

		dstBuf = ReadBufferWithoutRelcache(dstlocator, forkNum,
										   BufferGetBlockNumber(srcBuf),
										   RBM_ZERO_AND_LOCK, bstrategy_dst,
										   permanent);
		dstPage = BufferGetPage(dstBuf);

		START_CRIT_SECTION();

		/* Copy page data from the source to the destination. */
		memcpy(dstPage, srcPage, BLCKSZ);
		MarkBufferDirty(dstBuf);

		/* WAL-log the copied page. */
		if (use_wal)
			log_newpage_buffer(dstBuf, true);

		END_CRIT_SECTION();

		UnlockReleaseBuffer(dstBuf);
		UnlockReleaseBuffer(srcBuf);
	}
	Assert(read_stream_next_buffer(src_stream, NULL) == InvalidBuffer);
	read_stream_end(src_stream);

	FreeAccessStrategy(bstrategy_src);
	FreeAccessStrategy(bstrategy_dst);
}

/* ---------------------------------------------------------------------
 *		CreateAndCopyRelationData
 *
 *		Create destination relation storage and copy all forks from the
 *		source relation to the destination.
 *
 *		Pass permanent as true for permanent relations and false for
 *		unlogged relations.  Currently this API is not supported for
 *		temporary relations.
 * --------------------------------------------------------------------
 */
void
CreateAndCopyRelationData(RelFileLocator src_rlocator,
						  RelFileLocator dst_rlocator, bool permanent)
{
	char		relpersistence;
	SMgrRelation src_rel;
	SMgrRelation dst_rel;

	if (unlikely(!calico_local_active))
	{
		native_CreateAndCopyRelationData(src_rlocator, dst_rlocator, permanent);
		return;
	}

	/* Set the relpersistence. */
	relpersistence = permanent ?
		RELPERSISTENCE_PERMANENT : RELPERSISTENCE_UNLOGGED;

	src_rel = smgropen(src_rlocator, INVALID_PROC_NUMBER);
	dst_rel = smgropen(dst_rlocator, INVALID_PROC_NUMBER);

	/*
	 * Create and copy all forks of the relation.  During create database we
	 * have a separate cleanup mechanism which deletes complete database
	 * directory.  Therefore, each individual relation doesn't need to be
	 * registered for cleanup.
	 */
	RelationCreateStorage(dst_rlocator, relpersistence, false);

	/* copy main fork. */
	RelationCopyStorageUsingBuffer(src_rlocator, dst_rlocator, MAIN_FORKNUM,
								   permanent);

	/* copy those extra forks that exist */
	for (ForkNumber forkNum = MAIN_FORKNUM + 1;
		 forkNum <= MAX_FORKNUM; forkNum++)
	{
		if (smgrexists(src_rel, forkNum))
		{
			smgrcreate(dst_rel, forkNum, false);

			/*
			 * WAL log creation if the relation is persistent, or this is the
			 * init fork of an unlogged relation.
			 */
			if (permanent || forkNum == INIT_FORKNUM)
				log_smgrcreate(&dst_rlocator, forkNum);

			/* Copy a fork's data, block by block. */
			RelationCopyStorageUsingBuffer(src_rlocator, dst_rlocator, forkNum,
										   permanent);
		}
	}
}

/*
 * Flush a previously, share-exclusively or exclusively, locked and pinned
 * buffer to the OS.
 */
void
FlushOneBuffer(Buffer buffer)
{
	BufferDesc *bufHdr;

	if (unlikely(!calico_local_active))
	{
		native_FlushOneBuffer(buffer);
		return;
	}

	/* currently not needed, but no fundamental reason not to support */
	Assert(!BufferIsLocal(buffer));

	Assert(BufferIsPinned(buffer));

	bufHdr = GetBufferDescriptor(buffer - 1);

	Assert(BufferIsLockedByMe(buffer));

	FlushBuffer(bufHdr, NULL, IOOBJECT_RELATION, IOCONTEXT_NORMAL);
}

/*
 * UnlockReleaseBuffer -- release the content lock and pin on a buffer
 *
 * This is just a, more efficient, shorthand for a common combination.
 */
void
UnlockReleaseBuffer(Buffer buffer)
{
	int			mode;
	BufferDesc *buf;
	PrivateRefCountEntry *ref;
	uint64		sub;
	uint64		lockstate;

	if (unlikely(!calico_local_active))
	{
		native_UnlockReleaseBuffer(buffer);
		return;
	}

	Assert(BufferIsPinned(buffer));

	if (BufferIsLocal(buffer))
	{
		UnpinLocalBuffer(buffer);
		return;
	}

	ResourceOwnerForgetBuffer(CurrentResourceOwner, buffer);

	buf = GetBufferDescriptor(buffer - 1);

	mode = BufferLockDisownInternal(buffer, buf);
    if (mode == BUFFER_LOCK_EXCLUSIVE) calico_native_writer_end(buf);

	/* compute state modification for lock release */
	sub = BufferLockReleaseSub(mode);

	/* compute state modification for pin release */
	ref = GetPrivateRefCountEntry(buffer, false);
	Assert(ref != NULL);
	Assert(ref->data.refcount > 0);
	ref->data.refcount--;

	/* no more backend local pins, reduce shared pin count */
	if (likely(ref->data.refcount == 0))
	{
		/* See comment in UnpinBufferNoOwner() */
		VALGRIND_MAKE_MEM_NOACCESS(BufHdrGetBlock(buf), BLCKSZ);

		sub |= BUF_REFCOUNT_ONE;
		ForgetPrivateRefCountEntry(ref);
	}

	/* perform the lock and pin release in one atomic op */
	lockstate = pg_atomic_sub_fetch_u64(&buf->state, sub);

	/* wake up waiters for the lock */
	BufferLockProcessRelease(buf, mode, lockstate);

	/* wake up waiter for the pin release */
	if (lockstate & BM_PIN_COUNT_WAITER)
		WakePinCountWaiter(buf);

	/*
	 * Now okay to allow cancel/die interrupts again, which were held when the
	 * lock was acquired.
	 */
	RESUME_INTERRUPTS();
}

/*
 * Acquire the buffer content lock in the specified mode
 *
 * If the lock is not available, sleep until it is.
 *
 * Side effect: cancel/die interrupts are held off until lock release.
 *
 * This uses almost the same locking approach as lwlock.c's
 * LWLockAcquire(). See documentation at the top of lwlock.c for a more
 * detailed discussion.
 *
 * The reason that this, and most of the other BufferLock* functions, get both
 * the Buffer and BufferDesc* as parameters, is that looking up one from the
 * other repeatedly shows up noticeably in profiles.
 *
 * Callers should provide a constant for mode, for more efficient code
 * generation.
 */
static inline void
BufferLockAcquire(Buffer buffer, BufferDesc *buf_hdr, BufferLockMode mode)
{
	PrivateRefCountEntry *entry;
	int			extraWaits = 0;

	/*
	 * Get reference to the refcount entry before we hold the lock, it seems
	 * better to do before holding the lock.
	 */
	entry = GetPrivateRefCountEntry(buffer, true);

	/*
	 * We better not already hold a lock on the buffer.
	 */
	Assert(entry->data.lockmode == BUFFER_LOCK_UNLOCK);

	/*
	 * Lock out cancel/die interrupts until we exit the code section protected
	 * by the content lock.  This ensures that interrupts will not interfere
	 * with manipulations of data structures in shared memory.
	 */
	HOLD_INTERRUPTS();

	for (;;)
	{
		uint32		wait_event = 0; /* initialized to avoid compiler warning */
		bool		mustwait;

		/*
		 * Try to grab the lock the first time, we're not in the waitqueue
		 * yet/anymore.
		 */
		mustwait = BufferLockAttempt(buf_hdr, mode);

		if (likely(!mustwait))
		{
			break;
		}

		/*
		 * Ok, at this point we couldn't grab the lock on the first try. We
		 * cannot simply queue ourselves to the end of the list and wait to be
		 * woken up because by now the lock could long have been released.
		 * Instead add us to the queue and try to grab the lock again. If we
		 * succeed we need to revert the queuing and be happy, otherwise we
		 * recheck the lock. If we still couldn't grab it, we know that the
		 * other locker will see our queue entries when releasing since they
		 * existed before we checked for the lock.
		 */

		/* add to the queue */
		BufferLockQueueSelf(buf_hdr, mode);

		/* we're now guaranteed to be woken up if necessary */
		mustwait = BufferLockAttempt(buf_hdr, mode);

		/* ok, grabbed the lock the second time round, need to undo queueing */
		if (!mustwait)
		{
			BufferLockDequeueSelf(buf_hdr);
			break;
		}

		switch (mode)
		{
			case BUFFER_LOCK_EXCLUSIVE:
				wait_event = WAIT_EVENT_BUFFER_EXCLUSIVE;
				break;
			case BUFFER_LOCK_SHARE_EXCLUSIVE:
				wait_event = WAIT_EVENT_BUFFER_SHARE_EXCLUSIVE;
				break;
			case BUFFER_LOCK_SHARE:
				wait_event = WAIT_EVENT_BUFFER_SHARED;
				break;
			case BUFFER_LOCK_UNLOCK:
				pg_unreachable();

		}
		pgstat_report_wait_start(wait_event);

		/*
		 * Wait until awakened.
		 *
		 * It is possible that we get awakened for a reason other than being
		 * signaled by BufferLockWakeup().  If so, loop back and wait again.
		 * Once we've gotten the lock, re-increment the sema by the number of
		 * additional signals received.
		 */
		for (;;)
		{
			PGSemaphoreLock(MyProc->sem);
			if (MyProc->lwWaiting == LW_WS_NOT_WAITING)
				break;
			extraWaits++;
		}

		pgstat_report_wait_end();

		/* Retrying, allow BufferLockReleaseSub to release waiters again. */
		pg_atomic_fetch_and_u64(&buf_hdr->state, ~BM_LOCK_WAKE_IN_PROGRESS);
	}

	/* Remember that we now hold this lock */
	entry->data.lockmode = mode;
    if (mode == BUFFER_LOCK_EXCLUSIVE) calico_native_writer_begin(buf_hdr);

	/*
	 * Fix the process wait semaphore's count for any absorbed wakeups.
	 */
	while (unlikely(extraWaits-- > 0))
		PGSemaphoreUnlock(MyProc->sem);
}

/*
 * Release a previously acquired buffer content lock.
 */
static void
BufferLockUnlock(Buffer buffer, BufferDesc *buf_hdr)
{
	BufferLockMode mode;
	uint64		oldstate;
	uint64		sub;

	mode = BufferLockDisownInternal(buffer, buf_hdr);
    if (mode == BUFFER_LOCK_EXCLUSIVE) calico_native_writer_end(buf_hdr);

	/*
	 * Release my hold on lock, after that it can immediately be acquired by
	 * others, even if we still have to wakeup other waiters.
	 */
	sub = BufferLockReleaseSub(mode);

	oldstate = pg_atomic_sub_fetch_u64(&buf_hdr->state, sub);

	BufferLockProcessRelease(buf_hdr, mode, oldstate);

	/*
	 * Now okay to allow cancel/die interrupts.
	 */
	RESUME_INTERRUPTS();
}

/*
 * Acquire the content lock for the buffer, but only if we don't have to wait.
 *
 * It is allowed to try to conditionally acquire a lock on a buffer that this
 * backend has already locked, but the lock acquisition will always fail, even
 * if the new lock acquisition does not conflict with an already held lock
 * (e.g. two share locks). This is because we currently do not have space to
 * track multiple lock ownerships of the same buffer within one backend.  That
 * is ok for the current uses of BufferLockConditional().
 */
static bool
BufferLockConditional(Buffer buffer, BufferDesc *buf_hdr, BufferLockMode mode)
{
	PrivateRefCountEntry *entry = GetPrivateRefCountEntry(buffer, true);
	bool		mustwait;

	/*
	 * As described above, if we're trying to lock a buffer this backend
	 * already has locked, return false, independent of the existing and
	 * desired lock level.
	 */
	if (entry->data.lockmode != BUFFER_LOCK_UNLOCK)
		return false;

	/*
	 * Lock out cancel/die interrupts until we exit the code section protected
	 * by the content lock.  This ensures that interrupts will not interfere
	 * with manipulations of data structures in shared memory.
	 */
	HOLD_INTERRUPTS();

	/* Check for the lock */
	mustwait = BufferLockAttempt(buf_hdr, mode);

	if (mustwait)
	{
		/* Failed to get lock, so release interrupt holdoff */
		RESUME_INTERRUPTS();
	}
	else
	{
		entry->data.lockmode = mode;
    if (mode == BUFFER_LOCK_EXCLUSIVE) calico_native_writer_begin(buf_hdr);
	}

	return !mustwait;
}

/*
 * Internal function that tries to atomically acquire the content lock in the
 * passed in mode.
 *
 * This function will not block waiting for a lock to become free - that's the
 * caller's job.
 *
 * Similar to LWLockAttemptLock().
 */
static inline bool
BufferLockAttempt(BufferDesc *buf_hdr, BufferLockMode mode)
{
	uint64		old_state;

	/*
	 * Read once outside the loop, later iterations will get the newer value
	 * via compare & exchange.
	 */
	old_state = pg_atomic_read_u64(&buf_hdr->state);

	/* loop until we've determined whether we could acquire the lock or not */
	while (true)
	{
		uint64		desired_state;
		bool		lock_free;

		desired_state = old_state;

		if (mode == BUFFER_LOCK_EXCLUSIVE)
		{
			lock_free = (old_state & BM_LOCK_MASK) == 0;
			if (lock_free)
				desired_state += BM_LOCK_VAL_EXCLUSIVE;
		}
		else if (mode == BUFFER_LOCK_SHARE_EXCLUSIVE)
		{
			lock_free = (old_state & (BM_LOCK_VAL_EXCLUSIVE | BM_LOCK_VAL_SHARE_EXCLUSIVE)) == 0;
			if (lock_free)
				desired_state += BM_LOCK_VAL_SHARE_EXCLUSIVE;
		}
		else
		{
			lock_free = (old_state & BM_LOCK_VAL_EXCLUSIVE) == 0;
			if (lock_free)
				desired_state += BM_LOCK_VAL_SHARED;
		}

		/*
		 * Attempt to swap in the state we are expecting. If we didn't see
		 * lock to be free, that's just the old value. If we saw it as free,
		 * we'll attempt to mark it acquired. The reason that we always swap
		 * in the value is that this doubles as a memory barrier. We could try
		 * to be smarter and only swap in values if we saw the lock as free,
		 * but benchmark haven't shown it as beneficial so far.
		 *
		 * Retry if the value changed since we last looked at it.
		 */
		if (likely(pg_atomic_compare_exchange_u64(&buf_hdr->state,
												  &old_state, desired_state)))
		{
			if (lock_free)
			{
				/* Great! Got the lock. */
				return false;
			}
			else
				return true;	/* somebody else has the lock */
		}
	}

	pg_unreachable();
}

/*
 * Add ourselves to the end of the content lock's wait queue.
 */
static void
BufferLockQueueSelf(BufferDesc *buf_hdr, BufferLockMode mode)
{
	/*
	 * If we don't have a PGPROC structure, there's no way to wait. This
	 * should never occur, since MyProc should only be null during shared
	 * memory initialization.
	 */
	if (MyProc == NULL)
		elog(PANIC, "cannot wait without a PGPROC structure");

	if (MyProc->lwWaiting != LW_WS_NOT_WAITING)
		elog(PANIC, "queueing for lock while waiting on another one");

	LockBufHdr(buf_hdr);

	/* setting the flag is protected by the spinlock */
	pg_atomic_fetch_or_u64(&buf_hdr->state, BM_LOCK_HAS_WAITERS);

	/*
	 * These are currently used both for lwlocks and buffer content locks,
	 * which is acceptable, although not pretty, because a backend can't wait
	 * for both types of locks at the same time.
	 */
	MyProc->lwWaiting = LW_WS_WAITING;
	MyProc->lwWaitMode = mode;

	proclist_push_tail(&buf_hdr->lock_waiters, MyProcNumber, lwWaitLink);

	/* Can release the mutex now */
	UnlockBufHdr(buf_hdr);
}

/*
 * Remove ourselves from the waitlist.
 *
 * This is used if we queued ourselves because we thought we needed to sleep
 * but, after further checking, we discovered that we don't actually need to
 * do so.
 */
static void
BufferLockDequeueSelf(BufferDesc *buf_hdr)
{
	bool		on_waitlist;

	LockBufHdr(buf_hdr);

	on_waitlist = MyProc->lwWaiting == LW_WS_WAITING;
	if (on_waitlist)
		proclist_delete(&buf_hdr->lock_waiters, MyProcNumber, lwWaitLink);

	if (proclist_is_empty(&buf_hdr->lock_waiters) &&
		(pg_atomic_read_u64(&buf_hdr->state) & BM_LOCK_HAS_WAITERS) != 0)
	{
		pg_atomic_fetch_and_u64(&buf_hdr->state, ~BM_LOCK_HAS_WAITERS);
	}

	/* XXX: combine with fetch_and above? */
	UnlockBufHdr(buf_hdr);

	/* clear waiting state again, nice for debugging */
	if (on_waitlist)
		MyProc->lwWaiting = LW_WS_NOT_WAITING;
	else
	{
		int			extraWaits = 0;

		/*
		 * Somebody else dequeued us and has or will wake us up. Deal with the
		 * superfluous absorption of a wakeup.
		 */

		/*
		 * Clear BM_LOCK_WAKE_IN_PROGRESS if somebody woke us before we
		 * removed ourselves - they'll have set it.
		 */
		pg_atomic_fetch_and_u64(&buf_hdr->state, ~BM_LOCK_WAKE_IN_PROGRESS);

		/*
		 * Now wait for the scheduled wakeup, otherwise our ->lwWaiting would
		 * get reset at some inconvenient point later. Most of the time this
		 * will immediately return.
		 */
		for (;;)
		{
			PGSemaphoreLock(MyProc->sem);
			if (MyProc->lwWaiting == LW_WS_NOT_WAITING)
				break;
			extraWaits++;
		}

		/*
		 * Fix the process wait semaphore's count for any absorbed wakeups.
		 */
		while (extraWaits-- > 0)
			PGSemaphoreUnlock(MyProc->sem);
	}
}

/*
 * Stop treating lock as held by current backend.
 *
 * This is the code that can be shared between actually releasing a lock
 * (BufferLockUnlock()) and just not tracking ownership of the lock anymore
 * without releasing the lock (BufferLockDisown()).
 */
static inline int
BufferLockDisownInternal(Buffer buffer, BufferDesc *buf_hdr)
{
	BufferLockMode mode;
	PrivateRefCountEntry *ref;

	ref = GetPrivateRefCountEntry(buffer, false);
	if (ref == NULL)
		elog(ERROR, "lock %d is not held", buffer);
	mode = ref->data.lockmode;
	ref->data.lockmode = BUFFER_LOCK_UNLOCK;

	return mode;
}

/*
 * Wakeup all the lockers that currently have a chance to acquire the lock.
 *
 * wake_exclusive indicates whether exclusive lock waiters should be woken up.
 */
static void
BufferLockWakeup(BufferDesc *buf_hdr, bool wake_exclusive)
{
	bool		new_wake_in_progress = false;
	bool		wake_share_exclusive = true;
	proclist_head wakeup;
	proclist_mutable_iter iter;

	proclist_init(&wakeup);

	/* lock wait list while collecting backends to wake up */
	LockBufHdr(buf_hdr);

	proclist_foreach_modify(iter, &buf_hdr->lock_waiters, lwWaitLink)
	{
		PGPROC	   *waiter = GetPGProcByNumber(iter.cur);

		/*
		 * Already woke up a conflicting lock, so skip over this wait list
		 * entry.
		 */
		if (!wake_exclusive && waiter->lwWaitMode == BUFFER_LOCK_EXCLUSIVE)
			continue;
		if (!wake_share_exclusive && waiter->lwWaitMode == BUFFER_LOCK_SHARE_EXCLUSIVE)
			continue;

		proclist_delete(&buf_hdr->lock_waiters, iter.cur, lwWaitLink);
		proclist_push_tail(&wakeup, iter.cur, lwWaitLink);

		/*
		 * Prevent additional wakeups until retryer gets to run. Backends that
		 * are just waiting for the lock to become free don't retry
		 * automatically.
		 */
		new_wake_in_progress = true;

		/*
		 * Signal that the process isn't on the wait list anymore. This allows
		 * BufferLockDequeueSelf() to remove itself from the waitlist with a
		 * proclist_delete(), rather than having to check if it has been
		 * removed from the list.
		 */
		Assert(waiter->lwWaiting == LW_WS_WAITING);
		waiter->lwWaiting = LW_WS_PENDING_WAKEUP;

		/*
		 * Don't wakeup further waiters after waking a conflicting waiter.
		 */
		if (waiter->lwWaitMode == BUFFER_LOCK_SHARE)
		{
			/*
			 * Share locks conflict with exclusive locks.
			 */
			wake_exclusive = false;
		}
		else if (waiter->lwWaitMode == BUFFER_LOCK_SHARE_EXCLUSIVE)
		{
			/*
			 * Share-exclusive locks conflict with share-exclusive and
			 * exclusive locks.
			 */
			wake_exclusive = false;
			wake_share_exclusive = false;
		}
		else if (waiter->lwWaitMode == BUFFER_LOCK_EXCLUSIVE)
		{
			/*
			 * Exclusive locks conflict with all other locks, there's no point
			 * in waking up anybody else.
			 */
			break;
		}
	}

	Assert(proclist_is_empty(&wakeup) || pg_atomic_read_u64(&buf_hdr->state) & BM_LOCK_HAS_WAITERS);

	/* unset required flags, and release lock, in one fell swoop */
	{
		uint64		old_state;
		uint64		desired_state;

		old_state = pg_atomic_read_u64(&buf_hdr->state);
		while (true)
		{
			desired_state = old_state;

			/* compute desired flags */

			if (new_wake_in_progress)
				desired_state |= BM_LOCK_WAKE_IN_PROGRESS;
			else
				desired_state &= ~BM_LOCK_WAKE_IN_PROGRESS;

			if (proclist_is_empty(&buf_hdr->lock_waiters))
				desired_state &= ~BM_LOCK_HAS_WAITERS;

			desired_state &= ~BM_LOCKED;	/* release lock */

			if (pg_atomic_compare_exchange_u64(&buf_hdr->state, &old_state,
											   desired_state))
				break;
		}
	}

	/* Awaken any waiters I removed from the queue. */
	proclist_foreach_modify(iter, &wakeup, lwWaitLink)
	{
		PGPROC	   *waiter = GetPGProcByNumber(iter.cur);

		proclist_delete(&wakeup, iter.cur, lwWaitLink);

		/*
		 * Guarantee that lwWaiting being unset only becomes visible once the
		 * unlink from the link has completed. Otherwise the target backend
		 * could be woken up for other reason and enqueue for a new lock - if
		 * that happens before the list unlink happens, the list would end up
		 * being corrupted.
		 *
		 * The barrier pairs with the LockBufHdr() when enqueuing for another
		 * lock.
		 */
		pg_write_barrier();
		waiter->lwWaiting = LW_WS_NOT_WAITING;
		PGSemaphoreUnlock(waiter->sem);
	}
}

/*
 * Compute subtraction from buffer state for a release of a held lock in
 * `mode`.
 *
 * This is separated from BufferLockUnlock() as we want to combine the lock
 * release with other atomic operations when possible, leading to the lock
 * release being done in multiple places, each needing to compute what to
 * subtract from the lock state.
 */
static inline uint64
BufferLockReleaseSub(BufferLockMode mode)
{
	/*
	 * Turns out that a switch() leads gcc to generate sufficiently worse code
	 * for this to show up in profiles...
	 */
	if (mode == BUFFER_LOCK_EXCLUSIVE)
		return BM_LOCK_VAL_EXCLUSIVE;
	else if (mode == BUFFER_LOCK_SHARE_EXCLUSIVE)
		return BM_LOCK_VAL_SHARE_EXCLUSIVE;
	else
	{
		Assert(mode == BUFFER_LOCK_SHARE);
		return BM_LOCK_VAL_SHARED;
	}

	return 0;					/* keep compiler quiet */
}

/*
 * Handle work that needs to be done after releasing a lock that was held in
 * `mode`, where `lockstate` is the result of the atomic operation modifying
 * the state variable.
 *
 * This is separated from BufferLockUnlock() as we want to combine the lock
 * release with other atomic operations when possible, leading to the lock
 * release being done in multiple places.
 */
static void
BufferLockProcessRelease(BufferDesc *buf_hdr, BufferLockMode mode, uint64 lockstate)
{
	bool		check_waiters = false;
	bool		wake_exclusive = false;

	/* nobody else can have that kind of lock */
	Assert(!(lockstate & BM_LOCK_VAL_EXCLUSIVE));

	/*
	 * If we're still waiting for backends to get scheduled, don't wake them
	 * up again. Otherwise check if we need to look through the waitqueue to
	 * wake other backends.
	 */
	if ((lockstate & BM_LOCK_HAS_WAITERS) &&
		!(lockstate & BM_LOCK_WAKE_IN_PROGRESS))
	{
		if ((lockstate & BM_LOCK_MASK) == 0)
		{
			/*
			 * We released a lock and the lock was, in that moment, free. We
			 * therefore can wake waiters for any kind of lock.
			 */
			check_waiters = true;
			wake_exclusive = true;
		}
		else if (mode == BUFFER_LOCK_SHARE_EXCLUSIVE)
		{
			/*
			 * We released the lock, but another backend still holds a lock.
			 * We can't have released an exclusive lock, as there couldn't
			 * have been other lock holders. If we released a share lock, no
			 * waiters need to be woken up, as there must be other share
			 * lockers. However, if we held a share-exclusive lock, another
			 * backend now could acquire a share-exclusive lock.
			 */
			check_waiters = true;
			wake_exclusive = false;
		}
	}

	/*
	 * As waking up waiters requires the spinlock to be acquired, only do so
	 * if necessary.
	 */
	if (check_waiters)
		BufferLockWakeup(buf_hdr, wake_exclusive);
}

/*
 * BufferLockHeldByMeInMode - test whether my process holds the content lock
 * in the specified mode
 *
 * This is meant as debug support only.
 */
static bool
BufferLockHeldByMeInMode(BufferDesc *buf_hdr, BufferLockMode mode)
{
	PrivateRefCountEntry *entry =
		GetPrivateRefCountEntry(BufferDescriptorGetBuffer(buf_hdr), false);

	if (!entry)
		return false;
	else
		return entry->data.lockmode == mode;
}

/*
 * BufferLockHeldByMe - test whether my process holds the content lock in any
 * mode
 *
 * This is meant as debug support only.
 */
static bool
BufferLockHeldByMe(BufferDesc *buf_hdr)
{
	PrivateRefCountEntry *entry =
		GetPrivateRefCountEntry(BufferDescriptorGetBuffer(buf_hdr), false);

	if (!entry)
		return false;
	else
		return entry->data.lockmode != BUFFER_LOCK_UNLOCK;
}

/*
 * Release the content lock for the buffer.
 */
static pg_noinline void
calico_active_UnlockBuffer(Buffer buffer)
{
	BufferDesc *buf_hdr;

	Assert(BufferIsPinned(buffer));
	if (BufferIsLocal(buffer))
		return;					/* local buffers need no lock */

	buf_hdr = GetBufferDescriptor(buffer - 1);
	BufferLockUnlock(buffer, buf_hdr);
}

void
UnlockBuffer(Buffer buffer)
{
	if (unlikely(!calico_local_active))
	{
		native_UnlockBuffer(buffer);
		return;
	}
	calico_active_UnlockBuffer(buffer);
}

/*
 * Acquire the content_lock for the buffer.
 */
static pg_noinline void
calico_active_LockBufferInternal(Buffer buffer, BufferLockMode mode)
{
	BufferDesc *buf_hdr;

	/*
	 * We can't wait if we haven't got a PGPROC.  This should only occur
	 * during bootstrap or shared memory initialization.  Put an Assert here
	 * to catch unsafe coding practices.
	 */
	Assert(!(MyProc == NULL && IsUnderPostmaster));

	/* handled in LockBuffer() wrapper */
	Assert(mode != BUFFER_LOCK_UNLOCK);

	Assert(BufferIsPinned(buffer));
	if (BufferIsLocal(buffer))
		return;					/* local buffers need no lock */

	buf_hdr = GetBufferDescriptor(buffer - 1);

	/*
	 * Test the most frequent lock modes first. While a switch (mode) would be
	 * nice, at least gcc generates considerably worse code for it.
	 *
	 * Call BufferLockAcquire() with a constant argument for mode, to generate
	 * more efficient code for the different lock modes.
	 */
	if (mode == BUFFER_LOCK_SHARE)
		BufferLockAcquire(buffer, buf_hdr, BUFFER_LOCK_SHARE);
	else if (mode == BUFFER_LOCK_EXCLUSIVE)
		BufferLockAcquire(buffer, buf_hdr, BUFFER_LOCK_EXCLUSIVE);
	else if (mode == BUFFER_LOCK_SHARE_EXCLUSIVE)
		BufferLockAcquire(buffer, buf_hdr, BUFFER_LOCK_SHARE_EXCLUSIVE);
	else
		elog(ERROR, "unrecognized buffer lock mode: %d", mode);
}

void
LockBufferInternal(Buffer buffer, BufferLockMode mode)
{
	if (unlikely(!calico_local_active))
	{
		native_LockBufferInternal(buffer, mode);
		return;
	}
	calico_active_LockBufferInternal(buffer, mode);
}

/*
 * Acquire the content_lock for the buffer, but only if we don't have to wait.
 *
 * This assumes the caller wants BUFFER_LOCK_EXCLUSIVE mode.
 */
bool
ConditionalLockBuffer(Buffer buffer)
{
	BufferDesc *buf;

	if (unlikely(!calico_local_active))
		return native_ConditionalLockBuffer(buffer);

	Assert(BufferIsPinned(buffer));
	if (BufferIsLocal(buffer))
		return true;			/* act as though we got it */

	buf = GetBufferDescriptor(buffer - 1);

	return BufferLockConditional(buffer, buf, BUFFER_LOCK_EXCLUSIVE);
}

/*
 * LockBufferForCleanup - lock a buffer in preparation for deleting items
 *
 * Items may be deleted from a disk page only when the caller (a) holds an
 * exclusive lock on the buffer and (b) has observed that no other backend
 * holds a pin on the buffer.  If there is a pin, then the other backend
 * might have a pointer into the buffer (for example, a heapscan reference
 * to an item --- see README for more details).  It's OK if a pin is added
 * after the cleanup starts, however; the newly-arrived backend will be
 * unable to look at the page until we release the exclusive lock.
 *
 * To implement this protocol, a would-be deleter must pin the buffer and
 * then call LockBufferForCleanup().  LockBufferForCleanup() is similar to
 * LockBuffer(buffer, BUFFER_LOCK_EXCLUSIVE), except that it loops until
 * it has successfully observed pin count = 1.
 */
void
LockBufferForCleanup(Buffer buffer)
{
	BufferDesc *bufHdr;
	TimestampTz waitStart = 0;
	bool		waiting = false;
	bool		logged_recovery_conflict = false;

	if (unlikely(!calico_local_active))
	{
		native_LockBufferForCleanup(buffer);
		return;
	}

	Assert(BufferIsPinned(buffer));
	Assert(PinCountWaitBuf == NULL);

	CheckBufferIsPinnedOnce(buffer);

	/*
	 * We do not yet need to be worried about in-progress AIOs holding a pin,
	 * as we, so far, only support doing reads via AIO and this function can
	 * only be called once the buffer is valid (i.e. no read can be in
	 * flight).
	 */

	/* Nobody else to wait for */
	if (BufferIsLocal(buffer))
		return;

	bufHdr = GetBufferDescriptor(buffer - 1);

	for (;;)
	{
		uint64		buf_state;
		uint64		unset_bits = 0;

		/* Try to acquire lock */
		LockBuffer(buffer, BUFFER_LOCK_EXCLUSIVE);
		buf_state = LockBufHdr(bufHdr);

		Assert(BUF_STATE_GET_REFCOUNT(buf_state) > 0);
		if (BUF_STATE_GET_REFCOUNT(buf_state) == 1)
		{
			/* Successfully acquired exclusive lock with pincount 1 */
			UnlockBufHdr(bufHdr);
			goto cleanup_lock_acquired;
		}
		/* Failed, so mark myself as waiting for pincount 1 */
		if (buf_state & BM_PIN_COUNT_WAITER)
		{
			UnlockBufHdr(bufHdr);
			LockBuffer(buffer, BUFFER_LOCK_UNLOCK);
			elog(ERROR, "multiple backends attempting to wait for pincount 1");
		}
		bufHdr->wait_backend_pgprocno = MyProcNumber;
		PinCountWaitBuf = bufHdr;

		/*
		 * Publish BM_PIN_COUNT_WAITER while retaining the buffer header lock.
		 * The shared refcount can be decremented while BM_LOCKED is set, so
		 * use an atomic operation that preserves concurrent refcount changes.
		 */
		pg_atomic_fetch_or_u64(&bufHdr->state, BM_PIN_COUNT_WAITER);

		/*
		 * Recheck the refcount after publishing the waiter flag, while shared
		 * refcount increments are still prevented by BM_LOCKED.  If only our
		 * pin remains, the cleanup-lock condition has already been satisfied,
		 * so remove the waiter state and return without sleeping.
		 */
		buf_state = pg_atomic_read_u64(&bufHdr->state);

		if (BUF_STATE_GET_REFCOUNT(buf_state) == 1)
		{
			UnlockBufHdrExt(bufHdr, buf_state,
							0, BM_PIN_COUNT_WAITER,
							0);
			PinCountWaitBuf = NULL;
			goto cleanup_lock_acquired;
		}

		UnlockBufHdr(bufHdr);
		LockBuffer(buffer, BUFFER_LOCK_UNLOCK);

		/* Wait to be signaled by UnpinBuffer() */
		if (InHotStandby)
		{
			if (!waiting)
			{
				/* adjust the process title to indicate that it's waiting */
				set_ps_display_suffix("waiting");
				waiting = true;
			}

			/*
			 * Emit the log message if the startup process is waiting longer
			 * than deadlock_timeout for recovery conflict on buffer pin.
			 *
			 * Skip this if first time through because the startup process has
			 * not started waiting yet in this case. So, the wait start
			 * timestamp is set after this logic.
			 */
			if (waitStart != 0 && !logged_recovery_conflict)
			{
				TimestampTz now = GetCurrentTimestamp();

				if (TimestampDifferenceExceeds(waitStart, now,
											   DeadlockTimeout))
				{
					LogRecoveryConflict(RECOVERY_CONFLICT_BUFFERPIN,
										waitStart, now, NULL, true);
					logged_recovery_conflict = true;
				}
			}

			/*
			 * Set the wait start timestamp if logging is enabled and first
			 * time through.
			 */
			if (log_recovery_conflict_waits && waitStart == 0)
				waitStart = GetCurrentTimestamp();

			/* Publish the bufid that Startup process waits on */
			SetStartupBufferPinWaitBufId(buffer - 1);
			/* Set alarm and then wait to be signaled by UnpinBuffer() */
			ResolveRecoveryConflictWithBufferPin();
			/* Reset the published bufid */
			SetStartupBufferPinWaitBufId(-1);
		}
		else
			ProcWaitForSignal(WAIT_EVENT_BUFFER_CLEANUP);

		/*
		 * Remove flag marking us as waiter. Normally this will not be set
		 * anymore, but ProcWaitForSignal() can return for other signals as
		 * well.  We take care to only reset the flag if we're the waiter, as
		 * theoretically another backend could have started waiting. That's
		 * impossible with the current usages due to table level locking, but
		 * better be safe.
		 */
		buf_state = LockBufHdr(bufHdr);
		if ((buf_state & BM_PIN_COUNT_WAITER) != 0 &&
			bufHdr->wait_backend_pgprocno == MyProcNumber)
			unset_bits |= BM_PIN_COUNT_WAITER;

		UnlockBufHdrExt(bufHdr, buf_state,
						0, unset_bits,
						0);

		PinCountWaitBuf = NULL;
		/* Loop back and try again */
	}

cleanup_lock_acquired:

	/*
	 * Emit the log message if recovery conflict on buffer pin was resolved
	 * but the startup process waited longer than deadlock_timeout for it.
	 */
	if (logged_recovery_conflict)
		LogRecoveryConflict(RECOVERY_CONFLICT_BUFFERPIN,
							waitStart, GetCurrentTimestamp(),
							NULL, false);

	if (waiting)
	{
		/* reset ps display to remove the suffix if we added one */
		set_ps_display_remove_suffix();
		waiting = false;
	}

	return;
}

/*
 * ConditionalLockBufferForCleanup - as above, but don't wait to get the lock
 *
 * We won't loop, but just check once to see if the pin count is OK.  If
 * not, return false with no lock held.
 */
bool
ConditionalLockBufferForCleanup(Buffer buffer)
{
	BufferDesc *bufHdr;
	uint64		buf_state,
				refcount;

	if (unlikely(!calico_local_active))
		return native_ConditionalLockBufferForCleanup(buffer);

	Assert(BufferIsValid(buffer));

	/* see AIO related comment in LockBufferForCleanup() */

	if (BufferIsLocal(buffer))
	{
		refcount = LocalRefCount[-buffer - 1];
		/* There should be exactly one pin */
		Assert(refcount > 0);
		if (refcount != 1)
			return false;
		/* Nobody else to wait for */
		return true;
	}

	/* There should be exactly one local pin */
	refcount = GetPrivateRefCount(buffer);
	Assert(refcount);
	if (refcount != 1)
		return false;

	/* Try to acquire lock */
	if (!ConditionalLockBuffer(buffer))
		return false;

	bufHdr = GetBufferDescriptor(buffer - 1);
	buf_state = LockBufHdr(bufHdr);
	refcount = BUF_STATE_GET_REFCOUNT(buf_state);

	Assert(refcount > 0);
	if (refcount == 1)
	{
		/* Successfully acquired exclusive lock with pincount 1 */
		UnlockBufHdr(bufHdr);
		return true;
	}

	/* Failed, so release the lock */
	UnlockBufHdr(bufHdr);
	LockBuffer(buffer, BUFFER_LOCK_UNLOCK);
	return false;
}

/*
 *	Functions for buffer I/O handling
 *
 *	Also note that these are used only for shared buffers, not local ones.
 */

/*
 * Error context callback for errors occurring during shared buffer writes.
 */
static void
shared_buffer_write_error_callback(void *arg)
{
	BufferDesc *bufHdr = (BufferDesc *) arg;

	/* Buffer is pinned, so we can read the tag without locking the spinlock */
	if (bufHdr != NULL)
		errcontext("writing block %u of relation \"%s\"",
				   bufHdr->tag.blockNum,
				   relpathperm(BufTagGetRelFileLocator(&bufHdr->tag),
							   BufTagGetForkNum(&bufHdr->tag)).str);
}

/*
 * Release buffer as part of resource owner cleanup. This will only be called
 * if the buffer is pinned. If this backend held the content lock at the time
 * of the error we also need to release that (note that it is not possible to
 * hold a content lock without a pin).
 */
static void
ResOwnerReleaseBuffer(Datum res)
{
	Buffer		buffer = DatumGetInt32(res);

	/* Like ReleaseBuffer, but don't call ResourceOwnerForgetBuffer */
	if (!BufferIsValid(buffer))
		elog(ERROR, "bad buffer ID: %d", buffer);

	if (BufferIsLocal(buffer))
		UnpinLocalBufferNoOwner(buffer);
	else
	{
		PrivateRefCountEntry *ref;

		ref = GetPrivateRefCountEntry(buffer, false);

		/* not having a private refcount would imply resowner corruption */
		Assert(ref != NULL);

		/*
		 * If the buffer was locked at the time of the resowner release,
		 * release the lock now. This should only happen after errors.
		 */
		if (ref->data.lockmode != BUFFER_LOCK_UNLOCK)
		{
			BufferDesc *buf = GetBufferDescriptor(buffer - 1);

			HOLD_INTERRUPTS();	/* match the upcoming RESUME_INTERRUPTS */
			BufferLockUnlock(buffer, buf);
		}

		UnpinBufferNoOwner(GetBufferDescriptor(buffer - 1));
	}
}

/*
 * Helper function to mark unpinned buffer dirty whose buffer header lock is
 * already acquired.
 */
static bool
MarkDirtyUnpinnedBufferInternal(Buffer buf, BufferDesc *desc,
								bool *buffer_already_dirty)
{
	uint64		buf_state;
	bool		result = false;

	*buffer_already_dirty = false;

	buf_state = pg_atomic_read_u64(&(desc->state));
	Assert(buf_state & BM_LOCKED);

	if ((buf_state & BM_VALID) == 0)
	{
		UnlockBufHdr(desc);
		return false;
	}

	/* Check that it's not pinned already. */
	if (BUF_STATE_GET_REFCOUNT(buf_state) > 0)
	{
		UnlockBufHdr(desc);
		return false;
	}

	/* Pin the buffer and then release the buffer spinlock */
	PinBuffer_Locked(desc);

	/* If it was not already dirty, mark it as dirty. */
	if (!(buf_state & BM_DIRTY))
	{
		BufferLockAcquire(buf, desc, BUFFER_LOCK_EXCLUSIVE);
		MarkBufferDirty(buf);
		result = true;
		BufferLockUnlock(buf, desc);
	}
	else
		*buffer_already_dirty = true;

	UnpinBuffer(desc);

	return result;
}

/*
 * Try to mark the provided shared buffer as dirty.
 *
 * This function is intended for testing/development use only!
 *
 * Same as EvictUnpinnedBuffer() but with MarkBufferDirty() call inside.
 *
 * The buffer_already_dirty parameter is mandatory and indicate if the buffer
 * could not be dirtied because it is already dirty.
 *
 * Returns true if the buffer has successfully been marked as dirty.
 */
bool
MarkDirtyUnpinnedBuffer(Buffer buf, bool *buffer_already_dirty)
{
	BufferDesc *desc;
	bool		buffer_dirtied = false;

	if (unlikely(!calico_local_active))
		return native_MarkDirtyUnpinnedBuffer(buf, buffer_already_dirty);

	Assert(!BufferIsLocal(buf));

	/* Make sure we can pin the buffer. */
	ResourceOwnerEnlarge(CurrentResourceOwner);
	ReservePrivateRefCountEntry();

	desc = GetBufferDescriptor(buf - 1);
	LockBufHdr(desc);

	buffer_dirtied = MarkDirtyUnpinnedBufferInternal(buf, desc, buffer_already_dirty);
	/* Both can not be true at the same time */
	Assert(!(buffer_dirtied && *buffer_already_dirty));

	return buffer_dirtied;
}

/*
 * Try to mark all the shared buffers containing provided relation's pages as
 * dirty.
 *
 * This function is intended for testing/development use only! See
 * MarkDirtyUnpinnedBuffer().
 *
 * The buffers_* parameters are mandatory and indicate the total count of
 * buffers that:
 * - buffers_dirtied - were dirtied
 * - buffers_already_dirty - were already dirty
 * - buffers_skipped - could not be dirtied because of a reason different
 * than a buffer being already dirty.
 */
void
MarkDirtyRelUnpinnedBuffers(Relation rel,
							int32 *buffers_dirtied,
							int32 *buffers_already_dirty,
							int32 *buffers_skipped)
{
	if (unlikely(!calico_local_active))
	{
		native_MarkDirtyRelUnpinnedBuffers(rel, buffers_dirtied, buffers_already_dirty, buffers_skipped);
		return;
	}

	Assert(!RelationUsesLocalBuffers(rel));

	*buffers_dirtied = 0;
	*buffers_already_dirty = 0;
	*buffers_skipped = 0;

	for (int buf = 1; buf <= NBuffers; buf++)
	{
		BufferDesc *desc = GetBufferDescriptor(buf - 1);
		uint64		buf_state = pg_atomic_read_u64(&(desc->state));
		bool		buffer_already_dirty;

		CHECK_FOR_INTERRUPTS();

		/* An unlocked precheck should be safe and saves some cycles. */
		if ((buf_state & BM_VALID) == 0 ||
			!BufTagMatchesRelFileLocator(&desc->tag, &rel->rd_locator))
			continue;

		/* Make sure we can pin the buffer. */
		ResourceOwnerEnlarge(CurrentResourceOwner);
		ReservePrivateRefCountEntry();

		buf_state = LockBufHdr(desc);

		/* recheck, could have changed without the lock */
		if ((buf_state & BM_VALID) == 0 ||
			!BufTagMatchesRelFileLocator(&desc->tag, &rel->rd_locator))
		{
			UnlockBufHdr(desc);
			continue;
		}

		if (MarkDirtyUnpinnedBufferInternal(buf, desc, &buffer_already_dirty))
			(*buffers_dirtied)++;
		else if (buffer_already_dirty)
			(*buffers_already_dirty)++;
		else
			(*buffers_skipped)++;
	}
}

/*
 * Try to mark all the shared buffers as dirty.
 *
 * This function is intended for testing/development use only! See
 * MarkDirtyUnpinnedBuffer().
 *
 * See MarkDirtyRelUnpinnedBuffers() above for details about the buffers_*
 * parameters.
 */
void
MarkDirtyAllUnpinnedBuffers(int32 *buffers_dirtied,
							int32 *buffers_already_dirty,
							int32 *buffers_skipped)
{
	if (unlikely(!calico_local_active))
	{
		native_MarkDirtyAllUnpinnedBuffers(buffers_dirtied, buffers_already_dirty, buffers_skipped);
		return;
	}

	*buffers_dirtied = 0;
	*buffers_already_dirty = 0;
	*buffers_skipped = 0;

	for (int buf = 1; buf <= NBuffers; buf++)
	{
		BufferDesc *desc = GetBufferDescriptor(buf - 1);
		uint64		buf_state;
		bool		buffer_already_dirty;

		CHECK_FOR_INTERRUPTS();

		buf_state = pg_atomic_read_u64(&desc->state);
		if (!(buf_state & BM_VALID))
			continue;

		ResourceOwnerEnlarge(CurrentResourceOwner);
		ReservePrivateRefCountEntry();

		LockBufHdr(desc);

		if (MarkDirtyUnpinnedBufferInternal(buf, desc, &buffer_already_dirty))
			(*buffers_dirtied)++;
		else if (buffer_already_dirty)
			(*buffers_already_dirty)++;
		else
			(*buffers_skipped)++;
	}
}

void
calico_install_bufmgr(void)
{
    native_PinCountWaitBuf = calico_native_data("PinCountWaitBuf", sizeof(PinCountWaitBuf), __alignof__(PinCountWaitBuf));
    native_PrivateRefCountArray = calico_native_data("PrivateRefCountArray", sizeof(PrivateRefCountArray), __alignof__(PrivateRefCountArray));
    native_PrivateRefCountHash = calico_native_data("PrivateRefCountHash", sizeof(PrivateRefCountHash), __alignof__(PrivateRefCountHash));
    native_PrivateRefCountOverflowed = calico_native_data("PrivateRefCountOverflowed", sizeof(PrivateRefCountOverflowed), __alignof__(PrivateRefCountOverflowed));
    native_PrivateRefCountClock = calico_native_data("PrivateRefCountClock", sizeof(PrivateRefCountClock), __alignof__(PrivateRefCountClock));
    native_MaxProportionalPins = calico_native_data("MaxProportionalPins", sizeof(MaxProportionalPins), __alignof__(MaxProportionalPins));
    native_PrivateRefCountArrayKeys = calico_native_data("PrivateRefCountArrayKeys", sizeof(PrivateRefCountArrayKeys), __alignof__(PrivateRefCountArrayKeys));
    native_ReservedRefCountSlot = calico_native_data("ReservedRefCountSlot", sizeof(ReservedRefCountSlot), __alignof__(ReservedRefCountSlot));
    native_PrivateRefCountEntryLast = calico_native_data("PrivateRefCountEntryLast", sizeof(PrivateRefCountEntryLast), __alignof__(PrivateRefCountEntryLast));

    calico_redirect_bufmgr("ReadBuffer", (void *) calico_core_ReadBuffer,
                            (void **) &native_ReadBuffer);
    calico_redirect_bufmgr("ReadBufferExtended", (void *) calico_core_ReadBufferExtended,
                            (void **) &native_ReadBufferExtended);
    calico_redirect_bufmgr("ReadBufferWithoutRelcache", (void *) calico_core_ReadBufferWithoutRelcache,
                            (void **) &native_ReadBufferWithoutRelcache);
    calico_redirect_bufmgr("ExtendBufferedRel", (void *) calico_core_ExtendBufferedRel,
                            (void **) &native_ExtendBufferedRel);
    calico_redirect_bufmgr("ExtendBufferedRelBy", (void *) calico_core_ExtendBufferedRelBy,
                            (void **) &native_ExtendBufferedRelBy);
    calico_redirect_bufmgr("ExtendBufferedRelTo", (void *) calico_core_ExtendBufferedRelTo,
                            (void **) &native_ExtendBufferedRelTo);
    calico_redirect_bufmgr("StartReadBuffers", (void *) calico_core_StartReadBuffers,
                            (void **) &native_StartReadBuffers);
    calico_redirect_bufmgr("StartReadBuffer", (void *) calico_core_StartReadBuffer,
                            (void **) &native_StartReadBuffer);
    calico_redirect_bufmgr("ReleaseAndReadBuffer", (void *) calico_core_ReleaseAndReadBuffer,
                            (void **) &native_ReleaseAndReadBuffer);
    calico_redirect_bufmgr("CreateAndCopyRelationData", (void *) calico_core_CreateAndCopyRelationData,
                            (void **) &native_CreateAndCopyRelationData);
    calico_redirect_bufmgr("FlushOneBuffer", (void *) calico_core_FlushOneBuffer,
                            (void **) &native_FlushOneBuffer);
    calico_redirect_bufmgr("UnlockReleaseBuffer", (void *) calico_core_UnlockReleaseBuffer,
                            (void **) &native_UnlockReleaseBuffer);
    calico_redirect_bufmgr("UnlockBuffer", (void *) calico_core_UnlockBuffer,
                            (void **) &native_UnlockBuffer);
    calico_redirect_bufmgr("LockBufferInternal", (void *) calico_core_LockBufferInternal,
                            (void **) &native_LockBufferInternal);
    calico_redirect_bufmgr("ConditionalLockBuffer", (void *) calico_core_ConditionalLockBuffer,
                            (void **) &native_ConditionalLockBuffer);
    calico_redirect_bufmgr("LockBufferForCleanup", (void *) calico_core_LockBufferForCleanup,
                            (void **) &native_LockBufferForCleanup);
    calico_redirect_bufmgr("ConditionalLockBufferForCleanup", (void *) calico_core_ConditionalLockBufferForCleanup,
                            (void **) &native_ConditionalLockBufferForCleanup);
    calico_redirect_bufmgr("MarkDirtyUnpinnedBuffer", (void *) calico_core_MarkDirtyUnpinnedBuffer,
                            (void **) &native_MarkDirtyUnpinnedBuffer);
    calico_redirect_bufmgr("MarkDirtyRelUnpinnedBuffers", (void *) calico_core_MarkDirtyRelUnpinnedBuffers,
                            (void **) &native_MarkDirtyRelUnpinnedBuffers);
    calico_redirect_bufmgr("MarkDirtyAllUnpinnedBuffers", (void *) calico_core_MarkDirtyAllUnpinnedBuffers,
                            (void **) &native_MarkDirtyAllUnpinnedBuffers);
    calico_redirect_address((void *) buffer_resowner_desc.ReleaseResource, (void *) ResOwnerReleaseBuffer, "buffer_resowner_desc.ReleaseResource", NULL);
    calico_install_table();
}
