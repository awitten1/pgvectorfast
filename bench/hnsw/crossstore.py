#!/usr/bin/env python3
"""Recall-calibrated, single-client HNSW comparison on the existing 10M datasets.

Builds may use several cores; queries use one fixed server core and one client
core. Docker limits include anonymous memory AND file cache. Swap is disabled
except in the explicit paging mode, which requires host swap to be available.
Store data must live on a real NVMe filesystem, not a loop device. This script
never changes system-wide caches, swap settings, or another application's data.
"""
import argparse
import hashlib
import getpass
import importlib.metadata
import json
import os
import pickletools
from pathlib import Path
import struct
import subprocess
import time

import h5py
import numpy as np

IMAGES = {"chroma": "chromadb/chroma:1.5.9", "qdrant": "qdrant/qdrant:v1.20.1",
          "pgvector": "pgvector-upstream-bench:0.8.7",
          "pgvectorfast": "pgvectorfast-021-test:trixie"}
TARGETS = {"sift10m": 0.9336, "deep10m": 0.8712}
QUERY_SEED = 20261115
CALIBRATION_QUERIES = 4000


def shell(*cmd, check=True):
    return subprocess.run(list(map(str, cmd)), check=check, capture_output=True, text=True)


def save(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(data, indent=2) + '\n')
    temp.replace(path)


def inputs(root, dataset):
    if dataset == 'sift10m':
        with h5py.File(root / 'SIFT10M/SIFT10Mfeatures.mat') as f:
            q = np.asarray(f['fea'][10_000_000:10_010_000], dtype=np.float32)
        gt = np.load(root / 'SIFT10M/ground_truth_indices.npy')[:, :10]
    else:
        p = root / 'deep10m_data/query.public.10K.fbin'
        with p.open('rb') as f:
            n, d = struct.unpack('<II', f.read(8))
            q = np.fromfile(f, dtype='<f4').reshape(n, d)
        p = root / 'deep10m_data/groundtruth.ibin'
        with p.open('rb') as f:
            n, d = struct.unpack('<II', f.read(8))
            gt = np.fromfile(f, dtype='<i4').reshape(n, d)[:, :10]
    return q, gt


def batches(root, dataset, count, size, start=0):
    if dataset == 'sift10m':
        with h5py.File(root / 'SIFT10M/SIFT10Mfeatures.mat') as f:
            for i in range(start, count, size):
                yield i, np.asarray(f['fea'][i:min(i + size, count)], dtype=np.float32)
    else:
        p = root / 'deep10m_data/base.10M.fbin'
        with p.open('rb') as f:
            n, d = struct.unpack('<II', f.read(8))
        assert n >= count
        a = np.memmap(p, dtype='<f4', mode='r', offset=8, shape=(n, d))
        for i in range(start, count, size):
            yield i, np.asarray(a[i:min(i + size, count)])


class Store:
    def __init__(self, args):
        self.a = args
        self.name = f'pgvf-bench-{args.store}-{args.dataset}'
        self.data = args.work_dir / args.store / args.dataset
        self.data.mkdir(parents=True, exist_ok=True)
        self.collection = args.dataset
        self.client = None
        self.col = None
        self.ef = None
        self.swap_bytes = 0
        offset = 10 if args.dataset == 'deep10m' else 0
        self.http_port = 8800 + offset if args.store == 'chroma' else 8643 + offset
        self.grpc_port = 8644 + offset

    def pause_gate(self):
        flag = self.a.output / f'{self.a.store}-{self.a.dataset}.pause'
        ready = self.a.output / f'{self.a.store}-{self.a.dataset}.paused.json'
        if flag.exists():
            # At this boundary no ingestion RPC is outstanding. The runner
            # may freeze the container while timing a different ready store,
            # then unfreeze it before removing the flag. This avoids client
            # request timeouts and competing build CPU/I/O during timings.
            save(ready, {'pid': os.getpid(), 'container': self.name})
            while flag.exists():
                time.sleep(.25)
            ready.unlink(missing_ok=True)

    def state(self):
        r = shell('docker', 'inspect', self.name, check=False)
        return json.loads(r.stdout)[0] if r.returncode == 0 else None

    def stop(self):
        state = self.state()
        if state:
            assert state['Config']['Labels'].get('pgvf.benchmark') == 'crossstore'
            shell('docker', 'stop', '-t', '30', self.name, check=False)
            shell('docker', 'rm', self.name)
        self.client = None
        self.col = None

    def memory_swap_limit(self, memory, build=False):
        if build or not self.swap_bytes:
            return memory
        assert memory.endswith('g')
        return str(int(memory[:-1]) * 1024**3 + self.swap_bytes)

    def start(self, memory='16g', build=False):
        self.stop()
        self.current_memory = memory
        if not build:
            # File-cache pages are charged to the cgroup that first populated
            # them. Evict this stopped store's files before launching a new
            # cgroup, otherwise cache left by ingestion/copying could bypass
            # the next container's memory limit. Never drop host-wide caches.
            for path in self.data.rglob('*'):
                if path.is_file():
                    fd = os.open(path, os.O_RDONLY)
                    try:
                        os.fsync(fd)
                        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
                    finally:
                        os.close(fd)
        cmd = ['docker', 'run', '-d', '--name', self.name,
               '--label', 'pgvf.benchmark=crossstore', '--memory', memory,
               '--memory-swap', self.memory_swap_limit(memory, build),
               '--cpuset-cpus', self.a.build_cpus if build else str(self.a.server_cpu),
               '--log-opt', 'max-size=10m', '--log-opt', 'max-file=2']
        if self.a.store == 'chroma':
            cmd += ['-p', f'127.0.0.1:{self.http_port}:8000', '-v', f'{self.data}:/data',
                    '-e', f'RAYON_NUM_THREADS={6 if build else 1}']
        else:
            cmd += ['-p', f'127.0.0.1:{self.http_port}:6333', '-p', f'127.0.0.1:{self.grpc_port}:6334',
                    '-v', f'{self.data}:/qdrant/storage',
                    '-e', 'QDRANT__TELEMETRY_DISABLED=true',
                    '-e', 'QDRANT__SERVICE__MAX_WORKERS=1']
        cmd.append(IMAGES[self.a.store])
        shell(*cmd)
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline:
            state = self.state()
            if not state['State']['Running']:
                raise RuntimeError(f'server exited: {state["State"]}')
            try:
                if self.a.store == 'chroma':
                    import chromadb
                    from chromadb.config import Settings
                    self.client = chromadb.HttpClient(host='127.0.0.1', port=self.http_port,
                        settings=Settings(anonymized_telemetry=False,
                            chroma_query_request_timeout_seconds=600))
                    self.client.heartbeat()
                else:
                    from qdrant_client import QdrantClient
                    self.client = QdrantClient(host='127.0.0.1', port=self.http_port,
                        grpc_port=self.grpc_port, prefer_grpc=True, timeout=120)
                    self.client.get_collections()
                return
            except Exception:
                time.sleep(1)
        raise TimeoutError('server did not become ready within 120 seconds')

    def attach(self):
        if self.a.store == 'chroma':
            self.col = self.client.get_collection(self.collection, embedding_function=None)

    def build(self):
        self.start('24g', build=True)
        t0 = time.monotonic()
        dim = 128 if self.a.dataset == 'sift10m' else 96
        if self.a.store == 'chroma':
            self.a.batch_size = min(self.a.batch_size, self.client.get_max_batch_size())
            self.col = self.client.get_or_create_collection(self.collection,
                embedding_function=None, configuration={'hnsw': {
                    'space': 'l2', 'ef_construction': 64, 'max_neighbors': 16,
                    'ef_search': 40, 'num_threads': 6, 'batch_size': 1000,
                    'sync_threshold': 100000}})
            count = self.col.count()
        else:
            from qdrant_client import models as m
            if not self.client.collection_exists(self.collection):
                self.client.create_collection(self.collection,
                    vectors_config=m.VectorParams(size=dim, distance=m.Distance.EUCLID,
                        memory=m.Memory.CACHED), shard_number=1,
                    hnsw_config=m.HnswConfigDiff(m=16, ef_construct=64,
                        memory=m.Memory.CACHED, max_indexing_threads=6),
                    optimizers_config=m.OptimizersConfigDiff(indexing_threshold=0,
                        default_segment_number=1, max_segment_size=50000000,
                        max_optimization_threads=1))
            count = self.client.count(self.collection, exact=True).count
        assert 0 <= count <= self.a.count
        # An interrupted request can have persisted only part of its final
        # batch. Repair that boundary before resuming by count; otherwise a
        # hole could silently survive while later IDs are reinserted.
        resumed_from = count
        if count:
            left = max(0, count - 2 * self.a.batch_size)
            if self.a.store == 'chroma':
                tail = self.col.get(ids=[str(i) for i in range(left,
                    min(count + 2 * self.a.batch_size, self.a.count))], include=[])
                present = set(map(int, tail['ids']))
            else:
                tail, _ = self.client.scroll(self.collection, offset=left,
                    limit=4 * self.a.batch_size, with_payload=False, with_vectors=False)
                present = {p.id for p in tail}
            if present:
                right = max(present) + 1
                assert right <= self.a.count
                missing = sorted(set(range(left, right)) - present)
                if missing:
                    _, values = next(batches(self.a.data_root, self.a.dataset,
                        right, right-left, left))
                    values = values[np.asarray(missing) - left]
                    if self.a.store == 'chroma':
                        self.col.upsert(ids=list(map(str, missing)), embeddings=values)
                    else:
                        self.client.upsert(self.collection,
                            points=m.Batch(ids=missing, vectors=values.tolist()), wait=True)
                    print(f'repaired {len(missing)} interrupted-batch records', flush=True)
                count = right
        print(f'{self.a.store} {self.a.dataset}: resume at {count:,}', flush=True)
        last = time.monotonic()
        for i, a in batches(self.a.data_root, self.a.dataset, self.a.count,
                            self.a.batch_size, count):
            self.pause_gate()
            ids = list(range(i, i + len(a)))
            if self.a.store == 'chroma':
                self.col.add(ids=list(map(str, ids)), embeddings=a)
            else:
                self.client.upsert(self.collection,
                    points=m.Batch(ids=ids, vectors=a.tolist()), wait=True)
            if time.monotonic() - last >= 30 or i + len(a) == self.a.count:
                print(f'inserted {i + len(a):,}/{self.a.count:,}; elapsed {time.monotonic()-t0:.0f}s', flush=True)
                last = time.monotonic()
        self.pause_gate()
        actual = self.col.count() if self.a.store == 'chroma' else self.client.count(self.collection, exact=True).count
        assert actual == self.a.count, f'expected {self.a.count} vectors, found {actual}'
        if self.a.store == 'qdrant':
            self.client.update_collection(self.collection,
                optimizers_config=m.OptimizersConfigDiff(indexing_threshold=10000))
            deadline = time.monotonic() + 14400
            while time.monotonic() < deadline:
                info = self.client.get_collection(self.collection)
                print(f'indexing: {info.indexed_vectors_count:,}/{info.points_count:,} {info.status}', flush=True)
                if (info.indexed_vectors_count >= self.a.count * 0.995 and
                        str(info.status).lower().endswith('green')):
                    break
                time.sleep(30)
            else:
                raise TimeoutError('index build did not complete within 4 hours')
        metadata = {'store': self.a.store, 'dataset': self.a.dataset,
            'count': self.a.count, 'dimension': dim, 'm': 16, 'ef_construction': 64,
            'image': IMAGES[self.a.store], 'build_seconds': time.monotonic()-t0,
            'resumed_from': resumed_from,
            'image_digest': shell('docker', 'image', 'inspect', IMAGES[self.a.store],
                '--format', '{{json .RepoDigests}}').stdout.strip(),
            'collection_config': self.col.configuration if self.col else
                self.client.get_collection(self.collection).model_dump(mode='json')}
        save(self.a.output / f'{self.a.store}-{self.a.dataset}-build.json', metadata)
        self.stop()
        if self.a.store == 'chroma':
            # Chroma checkpoints the HNSW graph and its ID map independently
            # of the SQL record count. Persist the final partial import batch
            # once, so calibration trials reload an identical complete graph
            # instead of rebuilding a different WAL tail on every restart.
            pending = self.a.count - self.persisted_chroma_count()
            if pending:
                assert pending > 0
                self.start('24g', build=True)
                self.attach()
                self.col.modify(configuration={'hnsw': {'sync_threshold': max(2, pending)}})
                if pending == 1:
                    i, a = next(batches(self.a.data_root, self.a.dataset,
                        self.a.count, 1, self.a.count-1))
                    self.col.upsert(ids=[str(i)], embeddings=a)
                q, _ = inputs(self.a.data_root, self.a.dataset)
                self.query(q[0].tolist())
                self.stop()
            persisted = self.persisted_chroma_count()
            assert persisted == self.a.count, f'persisted HNSW contains {persisted} records'
            metadata['persisted_hnsw_count'] = persisted
            save(self.a.output / f'{self.a.store}-{self.a.dataset}-build.json', metadata)

    def persisted_chroma_count(self):
        paths = list(self.data.rglob('index_metadata.pickle'))
        if not paths:
            return 0
        assert len(paths) == 1
        # Read only the scalar near the front of Chroma's Rust-generated
        # pickle; loading two complete 10M-entry ID maps would waste RAM.
        found = False
        with paths[0].open('rb') as f:
            for op, arg, _ in pickletools.genops(f):
                if arg == 'total_elements_added':
                    found = True
                elif found and op.name in ('INT', 'BININT', 'BININT1', 'BININT2', 'LONG', 'LONG1', 'LONG4'):
                    return int(arg)
        raise RuntimeError('could not read persisted Chroma graph count')

    def set_ef(self, ef):
        self.ef = ef
        if self.a.store == 'chroma':
            self.col.modify(configuration={'hnsw': {'ef_search': ef, 'num_threads': 1}})
            # This server version caches loaded HNSW objects. Reload between
            # parameter trials to ensure ef_search is actually applied.
            memory = self.current_memory
            self.stop()
            self.start(memory)
            self.attach()

    def query(self, q):
        if self.a.store == 'chroma':
            r = self.col.query(query_embeddings=[q], n_results=10, include=[])
            return list(map(int, r['ids'][0]))
        from qdrant_client import models as m
        r = self.client.query_points(self.collection, query=q,
            search_params=m.SearchParams(hnsw_ef=self.ef, exact=False),
            limit=10, with_payload=False, with_vectors=False)
        return [p.id for p in r.points]

    def calibrate(self, queries, gt):
        target = TARGETS[self.a.dataset]
        results = {}

        def recall(ef):
            if ef not in results:
                self.set_ef(ef)
                matches = sum(len(set(self.query(q)) & set(g))
                    for q, g in zip(queries[:CALIBRATION_QUERIES], gt[:CALIBRATION_QUERIES]))
                results[ef] = matches / (10 * min(len(queries), CALIBRATION_QUERIES))
                print(f'calibration ef={ef}: recall@10={results[ef]:.5f}; target={target:.4f}', flush=True)
                save(self.a.output / f'{self.a.store}-{self.a.dataset}-calibration.json',
                    {'target': target, 'query_range': [0, CALIBRATION_QUERIES],
                     'query_order_seed': QUERY_SEED, 'trials': results})
            return results[ef]

        lo, hi = 10, 40
        while recall(hi) < target and hi < 512:
            hi = min(hi * 2, 512)
        if recall(lo) >= target:
            chosen = lo
        elif recall(hi) < target:
            raise RuntimeError('target recall not reachable at ef=512')
        else:
            while hi - lo > 1:
                mid = (hi + lo) // 2
                if recall(mid) < target:
                    lo = mid
                else:
                    hi = mid
            chosen = hi
        if abs(results[chosen]-target) > .005:
            raise RuntimeError(f'could not match target recall: {results}')
        save(self.a.output / f'{self.a.store}-{self.a.dataset}-tuned.json',
            {'ef_search': chosen, 'recall': results[chosen], 'target': target,
             'calibration_query_range': [0, CALIBRATION_QUERIES],
             'query_order_seed': QUERY_SEED, 'selection': 'at_least_target',
             'trials': results})
        return chosen

    def memory_stats(self):
        state = self.state()
        pid = state['State']['Pid']
        cg = Path(f'/proc/{pid}/cgroup').read_text().strip().split('::')[-1]
        root = Path('/sys/fs/cgroup') / cg.lstrip('/')
        out = {}
        for name in ('memory.current', 'memory.peak', 'memory.max', 'memory.swap.max',
                     'memory.swap.current', 'memory.swap.peak', 'memory.swap.events',
                     'memory.stat', 'memory.events', 'memory.pressure', 'io.stat', 'cpu.stat'):
            p = root / name
            if p.exists():
                out[name] = p.read_text()
        return out

    @staticmethod
    def read_bytes(stats):
        total = 0
        for line in stats.get('io.stat', '').splitlines():
            for value in line.split()[1:]:
                if value.startswith('rbytes='):
                    total += int(value.split('=')[1])
        return total

    def measure(self, mode, queries, gt):
        memory = '16g' if mode == 'inmem' else '2g'
        self.swap_bytes = self.a.swap_gib * 1024**3 if mode == 'paging' else 0
        tuned = json.loads((self.a.output / f'{self.a.store}-{self.a.dataset}-tuned.json').read_text())
        assert tuned['query_order_seed'] == QUERY_SEED
        result = {'store': self.a.store, 'dataset': self.a.dataset, 'mode': mode,
            'memory_limit': memory, 'swap': bool(self.swap_bytes),
            'swap_allowance_bytes': self.swap_bytes,
            'server_cpus': str(self.a.server_cpu), 'client_cpu': self.a.client_cpu,
            'image': IMAGES[self.a.store], 'count': self.a.count, 'ef_search': tuned['ef_search'],
            'calibration': tuned, 'query_order_seed': QUERY_SEED,
            'query_range': [CALIBRATION_QUERIES, 10000 if mode == 'inmem' else CALIBRATION_QUERIES+1000]}
        try:
            self.start(memory)
            self.attach()
            self.set_ef(tuned['ef_search'])
            host_config = self.state()['HostConfig']
            result['resource_limits'] = {key: host_config[key]
                for key in ('Memory', 'MemorySwap', 'CpusetCpus')}
            result['memory_at_start'] = self.memory_stats()
            if mode == 'paging':
                assert int(result['memory_at_start']['memory.max']) == 2 * 1024**3
                assert int(result['memory_at_start']['memory.swap.max']) == self.swap_bytes
            print(f'warming {self.a.store} {self.a.dataset} {mode}: 0/2000', flush=True)
            for i, q in enumerate(queries[:2000]):
                self.query(q)
                if (i + 1) % 100 == 0:
                    print(f'warmup {i + 1}/2000', flush=True)
            before = self.memory_stats()
            repeats = []
            start, end = result['query_range']
            for rep in range(self.a.repeats):
                times, matches = [], 0
                fingerprint = hashlib.sha256()
                for q, expected in zip(queries[start:end], gt[start:end]):
                    t = time.perf_counter_ns()
                    ids = self.query(q)
                    times.append((time.perf_counter_ns()-t) / 1e6)
                    matches += len(set(ids) & set(expected))
                    fingerprint.update(np.asarray(ids, dtype='<i8').tobytes())
                arr = np.asarray(times)
                row = {'repeat': rep, 'queries': len(arr), 'qps': 1000 / arr.mean(),
                    'mean_ms': float(arr.mean()), 'median_ms': float(np.median(arr)),
                    'p95_ms': float(np.percentile(arr, 95)), 'p99_ms': float(np.percentile(arr, 99)),
                    'recall_at_10': matches / (10 * len(arr)), 'fingerprint': fingerprint.hexdigest()}
                repeats.append(row)
                print(json.dumps(row), flush=True)
                result.update(outcome='ok', repeats=repeats,
                    memory_before=before, memory_after=self.memory_stats())
                result['timed_read_bytes'] = self.read_bytes(result['memory_after']) - self.read_bytes(before)
                save(self.a.output / f'{self.a.store}-{self.a.dataset}-{mode}.json', result)
            if mode in ('oom', 'paging') and result['timed_read_bytes'] < 64 * 1024**2:
                raise RuntimeError('less than 64 MiB read during larger-than-memory timing; invalid OOM run')
            if mode == 'paging' and int(result['memory_after']['memory.swap.current']) == 0:
                raise RuntimeError('paging run did not use swap')
        except Exception as exc:
            state = self.state()
            log = shell('docker', 'logs', '--tail', '60', self.name, check=False)
            result.update(outcome='failed', error=repr(exc),
                container_state=state['State'] if state else None,
                server_log_tail=(log.stdout + log.stderr)[-8000:])
            save(self.a.output / f'{self.a.store}-{self.a.dataset}-{mode}.json', result)
            if mode != 'oom':
                raise
            print(json.dumps(result), flush=True)
        finally:
            self.stop()


class Postgres(Store):
    def __init__(self, args):
        super().__init__(args)
        # Both PostgreSQL variants query the same physical index/heap fixture.
        self.data = args.work_dir / 'postgres' / args.dataset
        self.data.mkdir(parents=True, exist_ok=True)

    def build(self):
        template = self.a.data_root / 'perf_work' / f'{self.a.dataset}_template'
        if (self.data / 'PG_VERSION').exists():
            raise RuntimeError(f'fixture already exists: {self.data}; reuse with --action calibrate/measure')
        shell('cp', '-a', '--reflink=auto', str(template) + '/.', self.data)
        for name in ('postmaster.pid', 'postmaster.opts', 'phase.conf'):
            (self.data / name).unlink(missing_ok=True)
        (self.data / 'postgresql.auto.conf').write_text('# Benchmark configuration is supplied on the command line.\n')
        with (self.data / 'pg_hba.conf').open('a') as f:
            f.write('\nhost all all 172.16.0.0/12 trust\n')

    def start(self, memory='16g', build=False):
        self.stop()
        self.current_memory = memory
        # As with mmap stores, remove only this stopped fixture's cached pages
        # so the next cgroup owns its cache and the cap measures real memory.
        for path in self.data.rglob('*'):
            if path.is_file():
                fd = os.open(path, os.O_RDONLY)
                try:
                    os.fsync(fd)
                    os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
                finally:
                    os.close(fd)
        active = self.a.store == 'pgvectorfast'
        cmd = ['docker', 'run', '-d', '--name', self.name,
            '--label', 'pgvf.benchmark=crossstore', '--memory', memory,
            '--memory-swap', self.memory_swap_limit(memory),
            '--cpuset-cpus', str(self.a.server_cpu), '--shm-size', '20g',
            '--security-opt', 'seccomp=unconfined',
            '--user', f'{os.getuid()}:{os.getgid()}',
            '-p', '127.0.0.1:8842:5432', '-v', f'{self.data}:/benchdata',
            '--entrypoint', '/usr/lib/postgresql/18/bin/postgres', IMAGES[self.a.store],
            '-D', '/benchdata']
        settings = {'port': '5432', 'listen_addresses': '*', 'unix_socket_directories': '/tmp',
            'shared_preload_libraries': 'vectorfast' if active else '',
            'shared_buffers': '14GB' if memory == '16g' else '1536MB',
            'huge_pages': 'off', 'max_parallel_workers_per_gather': '0',
            'max_connections': '20', 'max_worker_processes': '16', 'autovacuum': 'off',
            'checkpoint_timeout': '1h', 'max_wal_size': '16GB', 'track_io_timing': 'on',
            'io_method': 'io_uring', 'debug_io_direct': 'data'}
        for k, v in settings.items():
            cmd += ['-c', f'{k}={v}']
        shell(*cmd)
        import psycopg
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline:
            state = self.state()
            if not state['State']['Running']:
                log = shell('docker', 'logs', self.name, check=False)
                raise RuntimeError(f'PostgreSQL exited: {log.stderr[-4000:]}')
            try:
                self.client = psycopg.connect(host='127.0.0.1', port=8842,
                    user=self.a.pg_user, dbname='postgres', autocommit=True, connect_timeout=3)
                break
            except psycopg.OperationalError:
                time.sleep(1)
        else:
            raise TimeoutError('PostgreSQL readiness timeout')
        from pgvector.psycopg import register_vector
        register_vector(self.client)
        if active:
            self.client.execute('CREATE EXTENSION IF NOT EXISTS vectorfast')
        self.client.execute('SET enable_seqscan = off')
        self.table = f'{self.a.dataset}_vectors'
        self.server_settings = settings
        self.query_sql = f'SELECT id FROM {self.table} ORDER BY embedding <-> %s LIMIT 10'

    def attach(self):
        if self.current_memory == '16g':
            self.client.execute('CREATE EXTENSION IF NOT EXISTS pg_prewarm')
            # Both relations fit in the 14 GiB buffer pool. Populate the
            # whole working set, not only pages reached by calibration queries.
            for relation in (self.table, f'{self.a.dataset}_hnsw_idx'):
                self.client.execute('SELECT pg_prewarm(%s::regclass)', (relation,)).fetchone()

    def set_ef(self, ef):
        self.ef = ef
        self.client.execute(f'SET hnsw.ef_search = {int(ef)}')

    def query(self, q):
        cur = self.client.execute(self.query_sql,
            (np.asarray(q, dtype=np.float32),), prepare=True, binary=True)
        return [row[0] - 1 for row in cur.fetchall()]

    def memory_stats(self):
        out = super().memory_stats()
        out['postgres_settings'] = self.server_settings
        out['postgres_version'] = self.client.execute('SELECT version()').fetchone()[0]
        out['pg_stat_io'] = self.client.execute('''SELECT sum(reads), sum(hits), sum(evictions)
            FROM pg_stat_io''').fetchone()
        out['pg_stat_io'] = [int(n or 0) for n in out['pg_stat_io']]
        if self.a.store == 'pgvectorfast':
            out['calico_pool'] = self.client.execute('SELECT calico.pool_statistics()').fetchone()[0]
            assert out['calico_pool']['mode'] == 'calico'
            assert out['calico_pool']['mapping_mismatches'] == 0
        return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--store', choices=IMAGES, required=True)
    p.add_argument('--dataset', choices=TARGETS, required=True)
    p.add_argument('--data-root', type=Path, required=True)
    p.add_argument('--work-dir', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--action', choices=['build', 'calibrate', 'measure', 'all'], default='all')
    p.add_argument('--mode', choices=['inmem', 'oom', 'paging', 'both'], default='both')
    p.add_argument('--swap-gib', type=int, default=16,
                   help='Additional swap allowance in paging mode (default: 16 GiB)')
    p.add_argument('--count', type=int, default=10_000_000)
    p.add_argument('--batch-size', type=int, default=2000)
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--build-cpus', default='2-31')
    p.add_argument('--server-cpu', type=int, default=2)
    p.add_argument('--client-cpu', type=int, default=0)
    p.add_argument('--pg-user', default=getpass.getuser())
    a = p.parse_args()
    if a.mode == 'paging' and a.swap_gib <= 0:
        p.error('--swap-gib must be positive in paging mode')
    a.work_dir.mkdir(parents=True, exist_ok=True)
    a.output.mkdir(parents=True, exist_ok=True)
    source = shell('findmnt', '-n', '-o', 'SOURCE', '-T', a.work_dir).stdout.strip()
    if a.action in ('measure', 'all') and not source.startswith('/dev/nvme'):
        raise RuntimeError(f'benchmark storage must be real NVMe, got {source}')
    store = Postgres(a) if a.store.startswith('pgvector') else Store(a)
    if a.action in ('build', 'all'):
        store.build()
    if a.action == 'build':
        return
    os.sched_setaffinity(0, {a.client_cpu})
    q, gt = inputs(a.data_root, a.dataset)
    assert len(q) == len(gt) == 10000
    order = np.random.default_rng(QUERY_SEED).permutation(10000)
    save(a.output / f'{a.dataset}-query-split.json', {
        'seed': QUERY_SEED, 'calibration_indices': order[:CALIBRATION_QUERIES].tolist(),
        'warmup_indices': order[:2000].tolist(),
        'inmem_indices': order[CALIBRATION_QUERIES:].tolist(),
        'oom_indices': order[CALIBRATION_QUERIES:CALIBRATION_QUERIES+1000].tolist()})
    q, gt = q[order], gt[order]
    queries = q.tolist()
    if a.action in ('calibrate', 'all'):
        try:
            store.start('16g')
            store.attach()
            store.calibrate(queries, gt)
        finally:
            store.stop()
    if a.action in ('measure', 'all'):
        for mode in ('inmem', 'oom') if a.mode == 'both' else (a.mode,):
            store.measure(mode, queries, gt)


if __name__ == '__main__':
    main()
