#!/usr/bin/env python3
"""Export median-of-three HNSW results and the two README charts."""
import argparse
import csv
import json
from pathlib import Path
import statistics
import sys

import matplotlib
matplotlib.use('Agg')
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from plot_style import PALETTE, apply_style, new_chart, add_legend, save_chart

STORES = ['pgvector', 'pgvectorfast', 'chroma', 'qdrant']
NAMES = {'pgvector': 'pgvector', 'pgvectorfast': 'pgvectorfast',
         'chroma': 'Chroma', 'qdrant': 'Qdrant'}
COLORS = PALETTE[:4]
DATASETS = ['sift10m', 'deep10m']


def summarize(root):
    rows = []
    for dataset in DATASETS:
        split = json.loads((root / f'{dataset}-query-split.json').read_text())
        assert split['seed'] == 20261115
        assert len(split['calibration_indices']) == 4000
        assert len(split['inmem_indices']) == 6000
        assert set(split['calibration_indices']).isdisjoint(split['inmem_indices'])
        assert set(split['calibration_indices'] + split['inmem_indices']) == set(range(10000))
        assert split['warmup_indices'] == split['calibration_indices'][:2000]
        assert split['oom_indices'] == split['inmem_indices'][:1000]
    for mode in ('inmem', 'oom'):
        for dataset in DATASETS:
            for store in STORES:
                p = root / f'{store}-{dataset}-{mode}.json'
                r = json.loads(p.read_text())
                assert r['count'] == 10_000_000
                assert r['query_order_seed'] == 20261115
                assert r['calibration']['query_order_seed'] == 20261115
                assert r['calibration']['target'] <= r['calibration']['recall'] <= r['calibration']['target'] + .005
                assert r['calibration']['calibration_query_range'] == [0, 4000]
                assert r['query_range'] == [4000, 10000 if mode == 'inmem' else 5000]
                row = {'dataset': dataset, 'mode': mode, 'store': store,
                       'ef_search': r['ef_search'], 'outcome': r['outcome']}
                if r['outcome'] == 'ok':
                    assert len(r['repeats']) == 3
                    assert {x['queries'] for x in r['repeats']} == {6000 if mode == 'inmem' else 1000}
                    recalls = [x['recall_at_10'] for x in r['repeats']]
                    assert max(recalls) - min(recalls) <= .001
                    for stats in (r['memory_before'], r['memory_after']):
                        assert int(stats['memory.max']) == (16 if mode == 'inmem' else 2) * 1024**3
                        assert int(stats['memory.swap.max']) == 0
                        if store == 'pgvectorfast':
                            assert stats['calico_pool']['mode'] == 'calico'
                            assert stats['calico_pool']['mapping_mismatches'] == 0
                    for field in ('qps', 'mean_ms', 'median_ms', 'p95_ms', 'p99_ms', 'recall_at_10'):
                        row[field] = statistics.median(x[field] for x in r['repeats'])
                    row['qps_min'] = min(x['qps'] for x in r['repeats'])
                    row['qps_max'] = max(x['qps'] for x in r['repeats'])
                    row['timed_read_bytes'] = r['timed_read_bytes']
                    if mode == 'oom':
                        assert r['timed_read_bytes'] >= 64 * 1024**2
                else:
                    assert store == 'chroma' and mode == 'oom', r
                    assert r.get('container_state', {}).get('OOMKilled'), r
                    row['outcome'] = 'OOM killed at 2 GiB'
                rows.append(row)
    for dataset in DATASETS:
        for mode in ('inmem', 'oom'):
            recalls = [r['recall_at_10'] for r in rows if r['dataset'] == dataset
                       and r['mode'] == mode and r['outcome'] == 'ok']
            assert max(recalls)-min(recalls) <= .01, f'recall mismatch: {dataset} {mode}: {recalls}'
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results', type=Path, required=True)
    p.add_argument('--assets', type=Path, required=True)
    a = p.parse_args()
    rows = summarize(a.results)
    fields = ['dataset', 'mode', 'store', 'ef_search', 'outcome', 'recall_at_10',
              'qps', 'mean_ms', 'median_ms', 'p95_ms', 'p99_ms', 'qps_min', 'qps_max',
              'timed_read_bytes']
    with (a.results / 'summary.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields, lineterminator='\n')
        w.writeheader()
        w.writerows(rows)
    lines = []
    for mode in ('inmem', 'oom'):
        lines += [f'## {mode}', '', '| Dataset | System | ef_search | Recall@10 | QPS | Mean (ms) | P95 (ms) |',
                  '| --- | --- | ---: | ---: | ---: | ---: | ---: |']
        for r in rows:
            if r['mode'] != mode:
                continue
            if r['outcome'] == 'ok':
                cells = [r['dataset'].upper(), NAMES[r['store']], str(r['ef_search']),
                    f"{r['recall_at_10']:.4f}", f"{r['qps']:,.1f}",
                    f"{r['mean_ms']:.3f}", f"{r['p95_ms']:.3f}"]
            else:
                cells = [r['dataset'].upper(), NAMES[r['store']], str(r['ef_search']),
                         '—', 'OOM at 2 GiB', '—', '—']
            lines += ['| ' + ' | '.join(cells) + ' |']
        lines += ['']
    (a.results / 'tables.md').write_text('\n'.join(lines))
    a.assets.mkdir(parents=True, exist_ok=True)
    apply_style()
    for mode, title, budget in [('inmem', 'In-Memory HNSW', '16 GiB per service'),
                                ('oom', 'Larger-Than-Memory HNSW', '2 GiB per service, OS swap disabled')]:
        fig, ax = new_chart(title, budget)
        xpos = np.arange(2)
        width = .19
        peak = max(r.get('qps', 0) for r in rows if r['mode'] == mode)
        for j, store in enumerate(STORES):
            selected = [next(r for r in rows if r['mode'] == mode
                        and r['dataset'] == ds and r['store'] == store) for ds in DATASETS]
            values = [r.get('qps', float('nan')) for r in selected]
            ranges = [[r['qps'] - r['qps_min'] if r['outcome'] == 'ok' else 0 for r in selected],
                      [r['qps_max'] - r['qps'] if r['outcome'] == 'ok' else 0 for r in selected]]
            xs = xpos + (j-1.5)*width
            bars = ax.bar(xs, values, width*.92, label=NAMES[store], color=COLORS[j],
                          yerr=ranges, error_kw={'capsize': 3, 'elinewidth': 1}, zorder=3)
            for x, bar, r in zip(xs, bars, selected):
                if r['outcome'] == 'ok':
                    ax.text(x, bar.get_height()+peak*.018, f"{r['qps']:,.0f}",
                            ha='center', va='bottom', fontsize=10)
                else:
                    ax.text(x, peak*.025, 'OOM', ha='center', va='bottom',
                            color=COLORS[j], weight='bold', fontsize=11)
        ax.set_ylabel('QPS (median of 3 repeats)')
        ax.set_xticks(xpos, ['SIFT10M', 'DEEP10M'])
        ax.set_ylim(0, peak*1.18)
        add_legend(ax, 4)
        save_chart(fig, a.assets / f'hnsw_{mode}.png')


if __name__ == '__main__':
    main()
