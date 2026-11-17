#!/usr/bin/env python3
"""Build one shared PostgreSQL fixture for both filtered-search implementations."""
import argparse
import getpass
import json
import os
from pathlib import Path
import subprocess
import time

import psycopg
from pgvector.psycopg import register_vector
from datasets import rows, vectors


def run(*args):
    r = subprocess.run(list(map(str, args)), capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError(f'{args[0]} failed: {r.stderr}\n{r.stdout}')
    return r.stdout


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', choices=['arxiv', 'yfcc10m'], required=True)
    p.add_argument('--data-root', type=Path, required=True)
    p.add_argument('--work-dir', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    n, d = vectors(a.data_root, a.dataset).shape
    assert (n, d) == ((2138591, 384) if a.dataset == 'arxiv' else (10000000, 192))
    data = a.work_dir/a.dataset
    assert not data.exists(), f'Fixture already exists: {data}'
    data.mkdir(parents=True)
    a.output.mkdir(parents=True, exist_ok=True)
    image = 'pgvectorfast-021-test:trixie'
    name = f'pgvf-filtered-build-{a.dataset}'
    user = getpass.getuser()
    common = ['--user', f'{os.getuid()}:{os.getgid()}', '-v', f'{data}:/benchdata']
    print(run('docker', 'run', '--rm', *common, '-v', '/etc/passwd:/etc/passwd:ro', '--entrypoint', '/usr/lib/postgresql/18/bin/initdb',
              image, '-D', '/benchdata', '-U', user, '--no-locale', '-E', 'UTF8', '--auth=trust'), flush=True)
    with (data/'pg_hba.conf').open('a') as f:
        f.write('\nhost all all 172.16.0.0/12 trust\n')
    started = time.time()
    run('docker', 'run', '-d', '--name', name, '--label', 'pgvf.benchmark=filtered',
        '--memory', '32g', '--memory-swap', '48g', '--shm-size', '24g', '--cpuset-cpus', '2-9',
        '-p', '127.0.0.1:8852:5432', *common, '--entrypoint', '/usr/lib/postgresql/18/bin/postgres',
        image, '-D', '/benchdata', '-c', 'shared_preload_libraries=', '-c', 'shared_buffers=4GB',
        '-c', 'listen_addresses=*', '-c', 'unix_socket_directories=/tmp', '-c', 'autovacuum=off',
        '-c', 'wal_level=minimal', '-c', 'max_wal_senders=0', '-c', 'max_wal_size=2GB',
        '-c', 'maintenance_work_mem=20GB', '-c', 'max_parallel_maintenance_workers=4')
    try:
        for _ in range(60):
            try:
                conn = psycopg.connect(host='127.0.0.1', port=8852, user=user,
                                       dbname='postgres', autocommit=True)
                break
            except psycopg.OperationalError:
                time.sleep(1)
        else:
            raise TimeoutError('PostgreSQL startup')
        conn.execute('CREATE EXTENSION vector')
        register_vector(conn)
        meta = 'labels text[], updated bigint, submitter text' if a.dataset == 'arxiv' else 'tags integer[]'
        conn.execute(f'CREATE TABLE items (id bigint PRIMARY KEY, embedding vector({d}), {meta})')
        with conn.cursor().copy('COPY items FROM STDIN (FORMAT BINARY)') as cp:
            cp.set_types(['int8', 'vector', 'text[]', 'int8', 'text'] if a.dataset == 'arxiv'
                         else ['int8', 'vector', 'int4[]'])
            for i, row in enumerate(rows(a.data_root, a.dataset), 1):
                cp.write_row(row)
                if i % 100000 == 0:
                    print(f'{a.dataset} loaded {i:,}/{n:,}', flush=True)
        assert i == n
        print('Building HNSW', flush=True)
        t = time.time()
        ops = 'vector_cosine_ops' if a.dataset == 'arxiv' else 'vector_l2_ops'
        conn.execute(f'CREATE INDEX items_hnsw ON items USING hnsw (embedding {ops}) WITH (m=16, ef_construction=64)')
        build_seconds = time.time()-t
        print(f'HNSW built in {build_seconds:.1f}s; building metadata indexes', flush=True)
        col = 'labels' if a.dataset == 'arxiv' else 'tags'
        conn.execute(f'CREATE INDEX items_filter ON items USING gin ({col})')
        if a.dataset == 'arxiv':
            conn.execute('CREATE INDEX items_updated ON items (updated)')
            conn.execute('CREATE INDEX items_submitter ON items (submitter)')
        conn.execute('VACUUM (FREEZE, ANALYZE) items')
        conn.execute('CHECKPOINT')
        count = conn.execute('SELECT count(*) FROM items').fetchone()[0]
        assert count == n
        manifest = dict(dataset=a.dataset, count=count, dimensions=d, image=image,
                        image_id=run('docker','image','inspect',image,'--format','{{.Id}}').strip(),
                        hnsw_build_seconds=build_seconds, total_seconds=time.time()-started,
                        relations=conn.execute("SELECT relname, pg_relation_size(oid) FROM pg_class WHERE relname LIKE 'items%' ORDER BY relname").fetchall())
        (a.output/f'{a.dataset}-fixture.json').write_text(json.dumps(manifest, indent=2)+'\n')
        conn.close()
        print(json.dumps(manifest), flush=True)
    finally:
        run('docker', 'stop', '-t', '120', name)
        run('docker', 'rm', name)


if __name__ == '__main__':
    main()
