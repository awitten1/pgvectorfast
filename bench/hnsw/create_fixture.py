#!/usr/bin/env python3
"""Create a PostgreSQL 18 snapshot from the raw benchmark vectors.

Use a PostgreSQL installation with pgvectorfast installed. The snapshot uses
the vector SQL interface without preloading Calico. Both PostgreSQL variants
will subsequently query this same heap and HNSW index.
"""
import argparse
import getpass
import re
from pathlib import Path
import subprocess

import psycopg
from pgvector.psycopg import register_vector
from crossstore import batches


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', choices=['sift10m', 'deep10m'], required=True)
    p.add_argument('--data-root', type=Path, required=True)
    p.add_argument('--pg-bin', type=Path, required=True)
    p.add_argument('--data-dir', type=Path, required=True)
    p.add_argument('--count', type=int, default=10_000_000)
    p.add_argument('--port', type=int, default=55498)
    p.add_argument('--shared-buffers', default='16GB')
    p.add_argument('--maintenance-work-mem', default='24GB')
    a = p.parse_args()
    assert re.search(r'PostgreSQL\)?\s+18\.', subprocess.check_output(
        [str(a.pg_bin/'postgres'), '--version'], text=True))
    if a.data_dir.exists():
        raise RuntimeError(f'refusing to overwrite {a.data_dir}')
    a.data_dir.parent.mkdir(parents=True, exist_ok=True)
    socket = a.data_dir.parent / (a.data_dir.name + '-socket')
    socket.mkdir(exist_ok=True)
    run = lambda name, *args: subprocess.run(
        [str(a.pg_bin/name), *map(str, args)], check=True)
    run('initdb', '-D', a.data_dir, '--no-locale', '-E', 'UTF8', '--auth=trust')
    with (a.data_dir/'postgresql.conf').open('a') as f:
        f.write(f"\nlisten_addresses=''\nunix_socket_directories='{socket}'\nport={a.port}\n"
                f"shared_buffers='{a.shared_buffers}'\nshared_preload_libraries=''\n")
    run('pg_ctl', '-D', a.data_dir, '-l', a.data_dir.parent/(a.data_dir.name+'.log'), '-w', 'start')
    try:
        with psycopg.connect(host=str(socket), port=a.port, user=getpass.getuser(),
                             dbname='postgres', autocommit=True) as conn:
            conn.execute('CREATE EXTENSION vector')
            register_vector(conn)
            dim = 128 if a.dataset == 'sift10m' else 96
            table = f'{a.dataset}_vectors'
            index = f'{a.dataset}_hnsw_idx'
            conn.execute(f'CREATE TABLE {table} (id bigint PRIMARY KEY, embedding vector({dim}))')
            with conn.cursor().copy(f'COPY {table} (id, embedding) FROM STDIN (FORMAT BINARY)') as cp:
                cp.set_types(['int8', 'vector'])
                for start, values in batches(a.data_root, a.dataset, a.count, 10000):
                    for offset, vector in enumerate(values):
                        cp.write_row((start+offset+1, vector))
                    if (start+len(values)) % 1_000_000 == 0:
                        print(f'loaded {start+len(values):,}', flush=True)
            conn.execute('SET max_parallel_maintenance_workers = 0')
            conn.execute("SELECT set_config('maintenance_work_mem', %s, false)",
                         (a.maintenance_work_mem,))
            conn.execute(f'CREATE INDEX {index} ON {table} USING hnsw '
                         '(embedding vector_l2_ops) WITH (m=16, ef_construction=64)')
            conn.execute(f'VACUUM (FREEZE, ANALYZE) {table}')
            conn.execute('CHECKPOINT')
            actual = conn.execute(f'SELECT count(*) FROM {table}').fetchone()[0]
            assert actual == a.count
            print(f'created {a.dataset} snapshot with {actual:,} vectors', flush=True)
    finally:
        run('pg_ctl', '-D', a.data_dir, '-m', 'fast', '-w', 'stop')


if __name__ == '__main__':
    main()
