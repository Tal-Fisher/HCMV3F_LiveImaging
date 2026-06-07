#!/usr/bin/env python3
"""
B2R comparison figure: top-20 embedding dims vs tabular reference.
Regression (delay_blue_to_red) + classification (b2r early/late labels, cut=1094 min).
Values from scripts 08_b2r_analysis.py and BluetoRed_analysis tabular results.
Matches style of embedding_comparison_figure.png.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

FIGURES_DIR = Path('/home/labs/ginossar/talfis/LiveImaging/CellposeEmbedding/figures')

# Results from 08_b2r_analysis.py (top-20 dims, 5-fold CV)
# Tabular: BluetoRed_analysis A2+A3 TabICL
labels     = ['Top-20 dims\n(embedding)', 'Tabular\n(reference)']
regr_r     = [0.297,  0.382]
clf_auc    = [0.708,  0.678]

x          = np.arange(len(labels))
width      = 0.45
EMB_CLR    = '#1565C0'
TAB_CLR    = '#EF6C00'
BAR_COLORS = [EMB_CLR, TAB_CLR]

fig, axes = plt.subplots(1, 2, figsize=(8, 5))
fig.subplots_adjust(wspace=0.40)

for ax, vals, ylabel, title, chance, ref_line, ref_label in [
    (axes[0], regr_r,  'Pearson r', 'Regression\n(delay BFP→mCherry)',
     None,  0.382, 'Tabular r = 0.382'),
    (axes[1], clf_auc, 'AUC',       'Classification\n(early vs med+late, b2r labels)',
     0.500, 0.678, 'Tabular AUC = 0.678'),
]:
    bars = ax.bar(x, vals, width=width, color=BAR_COLORS, zorder=3,
                  edgecolor='white', linewidth=0.6)

    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.005,
                f'{v:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

    if chance is not None:
        ax.axhline(chance, color='#9E9E9E', lw=1.2, ls='--', zorder=2,
                   label=f'Chance AUC = {chance:.3f}')
    ax.axhline(ref_line, color=TAB_CLR, lw=1.5, ls=':', zorder=2, alpha=0.8,
               label=ref_label)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=12, fontweight='bold', pad=8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', lw=0.5, alpha=0.4, zorder=0)
    ax.legend(fontsize=8, loc='lower right' if chance else 'upper left')

    # highlight winner
    best = int(np.argmax(vals))
    bars[best].set_edgecolor('#FFD600')
    bars[best].set_linewidth(2.5)

    ymax = max(vals) * 1.12
    ymin = (min(vals) - 0.04) if chance is None else min(chance - 0.05, min(vals) - 0.04)
    ax.set_ylim(max(0, ymin), ymax)

emb_patch = mpatches.Patch(color=EMB_CLR, label='Top-20 embedding dims (neck hook, A2)')
tab_patch  = mpatches.Patch(color=TAB_CLR, label='Tabular features (TabICL, A2+A3)')
fig.legend(handles=[emb_patch, tab_patch], loc='lower center',
           ncol=2, fontsize=9, frameon=False, bbox_to_anchor=(0.5, -0.04))

pass  # no title

out = FIGURES_DIR / 'embedding_comparison_b2r.png'
fig.savefig(str(out), dpi=180, bbox_inches='tight')
plt.close(fig)
print(f'Saved {out}')
