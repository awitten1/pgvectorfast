#!/usr/bin/env python3
"""Render the filtered-search, IVFFlat, and SQL charts from saved measurements."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
from matplotlib.ticker import FuncFormatter
import numpy as np

from plot_style import (PALETTE, BASELINE_COLOR, FAST_COLOR, apply_style,
                        new_chart, add_legend, save_chart)


def plot_filtered_arxiv(data, assets):
    for scenario in data['scenarios']:
        ram = scenario['ram_gib']
        groups = scenario['groups']
        xs = np.arange(len(groups))
        fig, ax = new_chart('Filtered Vector Search on ArXiv', f'{ram} GiB RAM')
        qps = {store: np.array([group[f'{store}_qps'] for group in groups])
               for store in ('pgvector', 'pgvectorfast')}
        medians = {store: np.median(repeats, axis=1) for store, repeats in qps.items()}
        width = .34
        peak = max(repeats.max() for repeats in qps.values())
        if ram == 2:
            ax.set_yscale('log')
            ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f'{value:g}'))
            ax.set_ylim(min(repeats.min() for repeats in qps.values()) / 2, peak * 2)
            ax.set_ylabel('QPS (median of 3 repeats, log scale)')
        else:
            ax.set_ylim(0, peak * 1.18)
            ax.set_ylabel('QPS (median of 3 repeats)')
        for store, offset, color in [('pgvector', -.5, BASELINE_COLOR),
                                     ('pgvectorfast', .5, FAST_COLOR)]:
            repeats = qps[store]
            median = medians[store]
            low, high = repeats.min(axis=1), repeats.max(axis=1)
            positions = xs + offset * width
            ax.bar(positions, median, width * .92, label=store, color=color,
                   yerr=[median - low, high - median],
                   error_kw={'capsize': 3, 'elinewidth': 1}, zorder=3)
            for x, value, top in zip(positions, median, high):
                digits = 2 if value < 100 else 0
                ax.annotate(f'{value:,.{digits}f}', (x, top),
                            xytext=(0, 4), textcoords='offset points',
                            ha='center', va='bottom', fontsize=12)
        ax.set_xticks(xs, [group['filter'] for group in groups])
        add_legend(ax, 2)
        save_chart(fig, assets / f'filtered_arxiv_{ram}g.png')


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
    if 'filtered_arxiv' in data:
        plot_filtered_arxiv(data['filtered_arxiv'], args.assets)
    plot_ivfflat(data['ivfflat'], args.assets)
    plot_analytical(data['analytical'], args.assets)


if __name__ == '__main__':
    main()
