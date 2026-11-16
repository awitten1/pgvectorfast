"""Shared appearance for the four benchmark figures in the README."""
import matplotlib.pyplot as plt

PALETTE = ['#4c78a8', '#f58518', '#b279a2', '#54a24b', '#e45756']
BASELINE_COLOR, FAST_COLOR = PALETTE[:2]


def apply_style():
    plt.rcParams.update({
        'font.family': 'DejaVu Sans', 'font.size': 12,
        'axes.titlesize': 17, 'axes.titleweight': 'normal',
        'axes.labelsize': 12, 'axes.labelweight': 'normal',
        'xtick.labelsize': 12, 'ytick.labelsize': 12,
        'legend.fontsize': 12, 'legend.frameon': False,
        'figure.facecolor': 'white', 'axes.facecolor': 'white',
        'axes.edgecolor': 'black', 'axes.linewidth': .8,
        'axes.spines.top': False, 'axes.spines.right': False,
        'text.color': 'black', 'axes.labelcolor': 'black',
        'xtick.color': 'black', 'ytick.color': 'black',
        'grid.color': '#b0b0b0', 'grid.linestyle': '-',
        'grid.linewidth': .8,
    })


def new_chart(title, subtitle):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_title(f'{title}\n{subtitle}', fontsize=17, pad=18)
    ax.grid(axis='y', alpha=.18, zorder=0)
    return fig, ax


def add_legend(ax, columns):
    ax.legend(ncol=columns, loc='upper center',
              bbox_to_anchor=(.5, -.11), frameon=False)


def save_chart(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches='tight', facecolor='white')
    plt.close(fig)
