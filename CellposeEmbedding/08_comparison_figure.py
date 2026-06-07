#!/usr/bin/env python3
"""
Summary comparison figure: embedding dimensionality reduction vs tabular baseline.
Two panels: regression (Pearson r) and classification (AUC).
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

FIGURES_DIR = Path('/home/labs/ginossar/talfis/LiveImaging/CellposeEmbedding/figures')

# ── Data ──────────────────────────────────────────────────────────────────────
labels   = ['256 dims\n(raw)', '116 dims\n(subset)', '20 dims\n(top-20)', 'Tabular\n(reference)']
regr_r   = [0.277,  0.271,  0.297,  0.359]
clf_auc  = [0.637,  0.668,  0.742,  0.674]

x       = np.arange(len(labels))
width   = 0.55

EMBD_CLR  = ['#5C6BC0', '#42A5F5', '#1565C0']   # blue family for embedding models
TAB_CLR   = '#EF6C00'                             # orange for tabular
BAR_COLORS = EMBD_CLR + [TAB_CLR]

# ── Figure ────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
fig.subplots_adjust(wspace=0.35)

for ax, vals, ylabel, title, chance, ref_line, ref_label in [
    (axes[0], regr_r,  "Pearson r",  "Regression\n(delay BFP→mCherry)",
     None,   0.359,  "Tabular r = 0.359"),
    (axes[1], clf_auc, "AUC",        "Classification\n(early vs med+late)",
     0.500,  0.674,  "Tabular AUC = 0.674"),
]:
    bars = ax.bar(x, vals, width=width, color=BAR_COLORS, zorder=3,
                  edgecolor='white', linewidth=0.6)

    # value labels on bars
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.005,
                f'{v:.3f}', ha='center', va='bottom', fontsize=9.5, fontweight='bold')

    # reference lines
    if chance is not None:
        ax.axhline(chance, color='#9E9E9E', lw=1.2, ls='--', zorder=2,
                   label=f'Chance AUC = {chance:.3f}')
    ax.axhline(ref_line, color=TAB_CLR, lw=1.5, ls=':', zorder=2, alpha=0.8,
               label=ref_label)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=12, fontweight='bold', pad=8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', lw=0.5, alpha=0.4, zorder=0)
    ax.legend(fontsize=8, loc='lower right' if chance else 'upper left')

    # highlight the best embedding bar
    best_emb = int(np.argmax(vals[:3]))
    bars[best_emb].set_edgecolor('#FFD600')
    bars[best_emb].set_linewidth(2.5)

    ymax = max(vals) * 1.12
    ymin = (min(vals) - 0.04) if chance is None else min(chance - 0.05, min(vals) - 0.04)
    ax.set_ylim(max(0, ymin), ymax)

# shared legend patch for embedding vs tabular
emb_patch = mpatches.Patch(color='#42A5F5', label='Embedding (neck hook, 256-dim)')
tab_patch  = mpatches.Patch(color=TAB_CLR,  label='Tabular features (ElasticNet / LogReg)')
fig.legend(handles=[emb_patch, tab_patch], loc='lower center',
           ncol=2, fontsize=9, frameon=False,
           bbox_to_anchor=(0.5, -0.04))

fig.suptitle('Cellpose Neck Embeddings — A2 Dataset\nMorphology at GFP onset frame',
             fontsize=13, fontweight='bold', y=1.02)

out = FIGURES_DIR / 'embedding_comparison_figure.png'
fig.savefig(str(out), dpi=180, bbox_inches='tight')
plt.close(fig)
print(f'Saved {out}')
