#!/usr/bin/env python3
"""Restyle the historical IVFFlat and SQL charts without rerunning benchmarks."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import numpy as np

from plot_style import (PALETTE, BASELINE_COLOR, FAST_COLOR, apply_style,
                        new_chart, add_legend, save_chart)


def plot_ivfflat(data, assets):
    fig, ax = new_chart('IVFFlat Vector Search Latency',
                        f"shared_buffers = {data['shared_buffers_gb']} GB")
    datasets = data['datasets']
    xs = np.arange(len(datasets))
    width = .34
    peak = max(r['pgvector_median_ms'] for r in datasets)
    for offset, field, label, color in [
        (-.5, 'pgvector_median_ms', 'pgvector', BASELINE_COLOR),
        (.5, 'pgvectorfast_median_ms', 'pgvectorfast', FAST_COLOR),
    ]:
        bars = ax.bar(xs + offset*width, [r[field] for r in datasets],
                      width*.92, label=label, color=color, zorder=3)
        for bar, row in zip(bars, datasets):
            value = row[field]
            text = f'{value:.2f}'
            if field == 'pgvectorfast_median_ms':
                reduction = 100 * (1 - value / row['pgvector_median_ms'])
                text += f'\n(-{reduction:.1f}%)'
            ax.text(bar.get_x() + bar.get_width()/2, value + peak*.018,
                    text, ha='center', va='bottom', fontsize=12)
    ax.set_xticks(xs, [f"{r['name']} ({r['dimensions']}D)\n"
                      f"Recall@3 = {r['recall_at_3']:.2f}" for r in datasets])
    ax.set_ylabel('Median query latency (ms)')
    ax.set_ylim(0, peak*1.22)
    add_legend(ax, 2)
    save_chart(fig, assets/'ivfflat_latency.png')


def plot_analytical(data, assets):
    fig, ax = new_chart('PostgreSQL Analytical Query Speedup',
                        f"shared_buffers = {data['shared_buffers_gb']} GB")
    sizes = data['row_sizes_bytes']
    xs = np.arange(len(sizes))
    series = data['series']
    width = .78 / len(series)
    for i, (row, color) in enumerate(zip(series, PALETTE)):
        bars = ax.bar(xs + (i-(len(series)-1)/2)*width, row['speedups'],
                      width*.92, label=row['label'], color=color, zorder=3)
        last = bars[-1]
        ax.text(last.get_x()+last.get_width()/2, last.get_height()+.05,
                f'{last.get_height():.2f}x', ha='center', va='bottom',
                rotation=90, fontsize=12)
    ax.axhline(1, color='#555555', linestyle='--', linewidth=1.1,
               label='Stock PostgreSQL (1.00x)', zorder=4)
    ax.set_xticks(xs, [str(n) for n in sizes])
    ax.set_xlabel('Row size (bytes)')
    ax.set_ylabel('Speedup vs. stock PostgreSQL (x)')
    ax.set_ylim(.85, 3.6)
    add_legend(ax, 3)
    save_chart(fig, assets/'olap_speedup.png')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path,
                        default=Path(__file__).with_name('figure_data.json'))
    parser.add_argument('--assets', type=Path,
                        default=Path(__file__).resolve().parents[1]/'assets')
    args = parser.parse_args()
    data = json.loads(args.data.read_text())
    args.assets.mkdir(parents=True, exist_ok=True)
    apply_style()
    plot_ivfflat(data['ivfflat'], args.assets)
    plot_analytical(data['analytical'], args.assets)


if __name__ == '__main__':
    main()
