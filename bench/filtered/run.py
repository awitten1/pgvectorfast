#!/usr/bin/env python3
"""Matched-recall filtered SQL benchmark using published queries and ground truth."""
import argparse
from collections import defaultdict
import getpass
import json
import os
from pathlib import Path
import statistics
import sys
import time

import numpy as np
import psycopg
from pgvector.psycopg import register_vector
from datasets import queries

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'hnsw'))
from crossstore import Store, IMAGES, save, shell

SEED = 20261117
TARGET = .95


def predicate(q, dataset):
    if dataset == 'yfcc10m':
        return 'tags @> %s::integer[]', [q['tags']], f"{len(q['tags'])}-tag"
    clauses, params, fields = [], [], []
    assert set(q['conditions']) == {'and'}
    for term in q['conditions']['and']:
        for key, condition in term.items():
            fields.append(key)
            if key == 'labels':
                clauses.append('labels @> %s::text[]')
                params.append([condition['match']['value']])
            elif key == 'submitter':
                clauses.append('submitter = %s')
                params.append(condition['match']['value'])
            elif key == 'update_date_ts':
                for op, value in condition['range'].items():
                    symbol = {'gt': '>', 'gte': '>=', 'lt': '<', 'lte': '<='}[op]
                    clauses.append(f'updated {symbol} %s')
                    params.append(value)
            else:
                raise ValueError(f'Unsupported predicate: {term}')
    return ' AND '.join(clauses), params, '+'.join(sorted(fields))


def split_queries(a):
    data = queries(a.data_root, a.dataset)
    groups = defaultdict(list)
    for i, q in enumerate(data):
        groups[predicate(q, a.dataset)[2]].append(i)
    rng = np.random.default_rng(SEED)
    calibration, timed = [], []
    for group in sorted(groups):
        order = rng.permutation(groups[group]).tolist()
        assert len(order) >= a.calibration_per_group + a.timed_per_group
        calibration += order[:a.calibration_per_group]
        timed += order[a.calibration_per_group:a.calibration_per_group+a.timed_per_group]
    rng.shuffle(calibration)
    rng.shuffle(timed)
    manifest = dict(seed=SEED, calibration=calibration, timed=timed,
                    population_by_filter={k:len(v) for k,v in groups.items()})
    path = a.output/f'{a.dataset}-split.json'
    if path.exists():
        assert json.loads(path.read_text()) == manifest
    else:
        save(path, manifest)
    return data, calibration, timed


def per_filter_prefix(data, indices, dataset, count):
    seen=defaultdict(int)
    chosen=[]
    for i in indices:
        group=predicate(data[i],dataset)[2]
        if seen[group]<count:
            chosen.append(i)
            seen[group]+=1
    return chosen


class PostgreSQL(Store):
    def __init__(self, a):
        super().__init__(a)
        self.name = f'pgvf-filtered-{a.store}-{a.dataset}'
        self.data = a.work_dir/a.dataset
        assert (self.data/'PG_VERSION').read_text().strip() == '18'

    def start(self, memory):
        self.stop()
        for path in self.data.rglob('*'):
            if path.is_file():
                fd = os.open(path, os.O_RDONLY)
                try:
                    os.fsync(fd)
                    os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
                finally:
                    os.close(fd)
        active = self.a.store == 'pgvectorfast'
        buffers = {2:'1536MB', 16:'14GB', 32:'28GB'}[memory]
        settings = dict(port='5432', listen_addresses='*', unix_socket_directories='/tmp',
                        shared_preload_libraries='vectorfast' if active else '',
                        shared_buffers=buffers, huge_pages='off', max_parallel_workers_per_gather='0',
                        max_connections='20', max_worker_processes='16', autovacuum='off',
                        checkpoint_timeout='1h', max_wal_size='2GB', track_io_timing='on',
                        io_method='io_uring', debug_io_direct='data')
        cmd = ['docker','run','-d','--name',self.name,'--label','pgvf.benchmark=crossstore',
               '--memory',f'{memory}g','--memory-swap',f'{memory+self.a.swap_gib}g',
               '--cpuset-cpus',str(self.a.server_cpu),'--shm-size','32g',
               '--security-opt','seccomp=unconfined','--user',f'{os.getuid()}:{os.getgid()}',
               '-p','127.0.0.1:8852:5432','-v',f'{self.data}:/benchdata',
               '--entrypoint','/usr/lib/postgresql/18/bin/postgres',IMAGES[self.a.store],
               '-D','/benchdata']
        for k,v in settings.items():
            cmd += ['-c',f'{k}={v}']
        shell(*cmd)
        for _ in range(120):
            if not self.state()['State']['Running']:
                raise RuntimeError(shell('docker','logs',self.name,check=False).stderr)
            try:
                self.client = psycopg.connect(host='127.0.0.1',port=8852,user=getpass.getuser(),
                                               dbname='postgres',autocommit=True,connect_timeout=2)
                break
            except psycopg.OperationalError:
                time.sleep(1)
        else:
            raise TimeoutError('PostgreSQL startup')
        register_vector(self.client)
        if active:
            self.client.execute('CREATE EXTENSION IF NOT EXISTS vectorfast')
        self.client.execute("SET plan_cache_mode = force_custom_plan")
        self.client.execute("SET work_mem = '64MB'")
        self.client.execute("SET hnsw.scan_mem_multiplier = 2")
        self.client.execute("SET hnsw.iterative_scan = relaxed_order")
        self.client.execute("SET statement_timeout = '120s'")
        if self.a.plan == 'hnsw':
            for name in ('enable_seqscan','enable_bitmapscan','enable_sort'):
                self.client.execute(f'SET {name} = off')
        self.settings = settings
        state = self.memory_stats()
        assert int(state['memory.max']) == memory*1024**3
        assert int(state['memory.swap.max']) == self.a.swap_gib*1024**3
        if memory >= 16:
            self.client.execute('CREATE EXTENSION IF NOT EXISTS pg_prewarm')
            for (relation,) in self.client.execute("SELECT oid::regclass::text FROM pg_class WHERE relname LIKE 'items%' AND relkind IN ('r','i') ORDER BY oid").fetchall():
                self.client.execute('SELECT pg_prewarm(%s::regclass)',(relation,)).fetchone()

    def memory_stats(self):
        result = super().memory_stats()
        if self.a.store == 'pgvectorfast' and self.client:
            result['calico_pool'] = self.client.execute('SELECT calico.pool_statistics()').fetchone()[0]
            assert result['calico_pool']['mode'] == 'calico'
            assert result['calico_pool']['mapping_mismatches'] == 0
        return result

    def configure(self, ef, scan):
        self.client.execute('SELECT set_config(%s,%s,false)', ('hnsw.ef_search',str(ef)))
        self.client.execute('SELECT set_config(%s,%s,false)', ('hnsw.max_scan_tuples',str(scan)))

    def sql(self, q, exact=False):
        where, params, _ = predicate(q, self.a.dataset)
        op = '<=>' if self.a.dataset == 'arxiv' else '<->'
        expr = f'(embedding {op} %s)' + (' + 0.0' if exact else '')
        # +0 disables the ANN ORDER BY path for exact reference checks.
        order = 'distance'
        limit = 10 if exact else 50
        sql = (f'WITH candidates AS MATERIALIZED (SELECT id, {expr} AS distance FROM items '
               f'WHERE {where} ORDER BY {order} LIMIT {limit}) '
               'SELECT id, distance FROM candidates ORDER BY distance + 0.0 LIMIT 10')
        return sql, [np.asarray(q['query'],dtype=np.float32), *params]

    def query(self, q, exact=False):
        sql, params = self.sql(q, exact)
        return self.client.execute(sql, params, prepare=True, binary=True).fetchall()


def evaluate(db, data, indices, verbose=True):
    records = []
    for j, i in enumerate(indices):
        q = data[i]
        start = time.perf_counter_ns()
        returned = db.query(q)
        ms = (time.perf_counter_ns()-start)/1e6
        ids = [int(r[0]) for r in returned]
        expected = set(q['closest_ids'][:10])
        assert len(expected) == 10
        recall = len(set(ids)&expected)/10
        records.append(dict(query_index=i, filter=predicate(q,db.a.dataset)[2],
                            latency_ms=ms, recall_at_10=recall, returned=len(ids), ids=ids))
        interval=25 if len(indices)<=300 else 100
        if verbose and (j+1)%interval == 0:
            print(f'{db.a.store} {db.a.dataset} {j+1}/{len(indices)}',flush=True)
    return records


def aggregate(records):
    groups = defaultdict(list)
    for r in records:
        groups[r['filter']].append(r)
    groups['all'] = records
    return {k:dict(queries=len(rs), recall_at_10=statistics.mean(r['recall_at_10'] for r in rs),
                   complete_fraction=statistics.mean(r['returned']==10 for r in rs),
                   qps=1000/statistics.mean(r['latency_ms'] for r in rs),
                   mean_ms=statistics.mean(r['latency_ms'] for r in rs),
                   p50_ms=float(np.percentile([r['latency_ms'] for r in rs],50)),
                   p95_ms=float(np.percentile([r['latency_ms'] for r in rs],95)))
            for k,rs in groups.items()}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',choices=['arxiv','yfcc10m'],required=True)
    p.add_argument('--store',choices=['pgvector','pgvectorfast'],required=True)
    p.add_argument('--action',choices=['validate','calibrate','measure'],required=True)
    p.add_argument('--data-root',type=Path,required=True)
    p.add_argument('--work-dir',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--ram-gib',type=int,choices=[2,16,32],default=16)
    p.add_argument('--swap-gib',type=int,default=16)
    p.add_argument('--image',help='Override the Docker image for implementation comparisons')
    p.add_argument('--plan',choices=['auto','hnsw'],default='auto')
    p.add_argument('--server-cpu',type=int,default=2)
    p.add_argument('--client-cpu',type=int,default=0)
    p.add_argument('--calibration-per-group',type=int,default=100)
    p.add_argument('--timed-per-group',type=int,default=300)
    p.add_argument('--repeats',type=int,default=3)
    a=p.parse_args()
    if a.image:
        IMAGES[a.store]=a.image
    a.output.mkdir(parents=True,exist_ok=True)
    assert shell('findmnt','-n','-o','SOURCE','-T',a.work_dir).stdout.strip().startswith('/dev/nvme')
    os.sched_setaffinity(0,{a.client_cpu})
    data, calibration, timed = split_queries(a)
    warmup=calibration
    if a.ram_gib==2 and a.action=='measure':
        timed=per_filter_prefix(data,timed,a.dataset,100)
        warmup=per_filter_prefix(data,calibration,a.dataset,30)
    db=PostgreSQL(a)
    stem=f'{a.store}-{a.dataset}-{a.plan}'
    try:
        db.start(a.ram_gib)
        if a.action=='validate':
            checked=[]
            groups=defaultdict(list)
            for i in calibration: groups[predicate(data[i],a.dataset)[2]].append(i)
            for indices in groups.values():
                for i in indices[:3]:
                    returned=db.query(data[i],exact=True)
                    expected=set(data[i]['closest_ids'][:10])
                    match=len(expected&{r[0] for r in returned})/10
                    checked.append(dict(index=i,recall=match,ids=[r[0] for r in returned]))
                    assert match>=.9, (i,match,returned,data[i]['closest_ids'][:10])
            save(a.output/f'{a.dataset}-reference-check.json',checked)
            print('Exact filtered reference checks passed',flush=True)
        elif a.action=='calibrate':
            trials=[]
            for ef,scan in [(40,20000),(100,100000),(200,200000),(400,500000),(800,1000000),(1000,2000000)]:
                db.configure(ef,scan)
                result=aggregate(evaluate(db,data,calibration))
                trial=dict(ef_search=ef,max_scan_tuples=scan,candidate_limit=50,metrics=result)
                trials.append(trial)
                save(a.output/f'{stem}-calibration.json',trials)
                print(json.dumps(trial),flush=True)
                if min(r['recall_at_10'] for r in result.values())>=TARGET and min(r['complete_fraction'] for r in result.values())>=.99:
                    save(a.output/f'{stem}-tuned.json',trial)
                    break
            else:
                raise RuntimeError('No configuration met filtered recall/completeness targets')
        else:
            tuned=json.loads((a.output/f'{stem}-tuned.json').read_text())
            db.configure(tuned['ef_search'],tuned['max_scan_tuples'])
            print(f'Warming with {len(warmup)} calibration queries',flush=True)
            evaluate(db,data,warmup)
            print(f'Timing {len(timed)} held-out queries, {a.repeats} repeats',flush=True)
            before=db.memory_stats()
            reps=[]
            for rep in range(a.repeats):
                records=evaluate(db,data,timed)
                reps.append(dict(metrics=aggregate(records),queries=records))
                save(a.output/f'{stem}-{a.ram_gib}g-progress.json',reps)
                print('repeat',rep+1,json.dumps(reps[-1]['metrics']),flush=True)
            after=db.memory_stats()
            plans={}
            plan_counts=defaultdict(int)
            query_plans=[]
            def index_names(node):
                names=([node['Index Name']] if 'Index Name' in node else [])
                for child in node.get('Plans',[]): names.extend(index_names(child))
                return names
            for i in timed:
                group=predicate(data[i],a.dataset)[2]
                sql,params=db.sql(data[i])
                plan=db.client.execute('EXPLAIN (FORMAT JSON) '+sql,params).fetchone()[0]
                names=index_names(plan[0]['Plan'])
                kind='hnsw' if 'items_hnsw' in names else ('metadata_index' if names else 'sequential_scan')
                plan_counts[f'{group}/{kind}']+=1
                query_plans.append(dict(query_index=i,plan_kind=kind,indexes=names))
                key=f'{group}/{kind}'
                if key not in plans:
                    plans[key]=db.client.execute('EXPLAIN (ANALYZE, BUFFERS, SETTINGS, FORMAT JSON) '+sql,params).fetchone()[0]
            result=dict(dataset=a.dataset,store=a.store,plan=a.plan,ram_gib=a.ram_gib,
                        swap_gib=a.swap_gib,seed=SEED,configuration=tuned,repeats=reps,
                        memory_before=before,memory_after=after,plans=plans,
                        plan_counts=dict(plan_counts),query_plans=query_plans,
                        timed_read_bytes=db.read_bytes(after)-db.read_bytes(before),
                        postgres_settings=db.settings,image=IMAGES[a.store],
                        image_id=db.state()['Image'],server_cpu=a.server_cpu,client_cpu=a.client_cpu)
            result['warmup_indices']=warmup
            result['timed_indices']=timed
            result['postgres_version']=db.client.execute('SELECT version()').fetchone()[0]
            result['system_identifier']=db.client.execute('SELECT system_identifier::text FROM pg_control_system()').fetchone()[0]
            result['extensions']=db.client.execute('SELECT extname, extversion FROM pg_extension ORDER BY extname').fetchall()
            result['query_settings']={name:db.client.execute('SELECT current_setting(%s,true)',(name,)).fetchone()[0]
                for name in ['hnsw.ef_search','hnsw.max_scan_tuples','hnsw.iterative_scan',
                             'hnsw.scan_mem_multiplier','work_mem','plan_cache_mode',
                             'enable_seqscan','enable_bitmapscan','enable_sort',
                             'calico.optimistic_reads','hnsw.enable_prefetch']}
            save(a.output/f'{stem}-{a.ram_gib}g.json',result)
    finally:
        if db.client: db.client.close()
        db.stop()


if __name__=='__main__':
    main()
