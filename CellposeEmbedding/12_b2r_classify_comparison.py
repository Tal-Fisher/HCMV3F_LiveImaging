#!/usr/bin/env python3
"""
Bar chart comparing b2r classification AUC:
  - Embedding top-20 dims: ElasticNet, XGBoost
  - Tabular (A2+A3): ElasticNet, XGBoost, TabICL
  (BluetoRed_analysis results, no half-movie filter, GMM cut=1094 min)
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

FIGURES = Path('/home/labs/ginossar/talfis/LiveImaging/CellposeEmbedding/figures')

labels = [
    'ElasticNet\n(embedding)',
    'XGBoost\n(embedding)',
    'ElasticNet\n(tabular)',
    'XGBoost\n(tabular)',
    'TabICL\n(tabular)',
]
aucs = [0.708, 0.609, 0.651, 0.668, 0.678]

EMB_COLORS = ['#1565C0', '#1E88E5']
TAB_COLORS = ['#E65100', '#EF6C00', '#FF8F00']
colors = EMB_COLORS + TAB_COLORS

x     = np.arange(len(labels))
width = 0.55

fig, ax = plt.subplots(figsize=(8, 5))

bars = ax.bar(x, aucs, width=width, color=colors, zorder=3,
              edgecolor='white', linewidth=0.6)

for bar, v in zip(bars, aucs):
    ax.text(bar.get_x() + bar.get_width() / 2, v + 0.004,
            f'{v:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

ax.axhline(0.500, color='#9E9E9E', lw=1.2, ls='--', zorder=2, label='Chance AUC = 0.500')

# highlight best bar
best = int(np.argmax(aucs))
bars[best].set_edgecolor('#FFD600')
bars[best].set_linewidth(2.5)

ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=10)
ax.set_ylabel('AUC', fontsize=12)
ax.set_ylim(0.45, max(aucs) * 1.10)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(axis='y', lw=0.5, alpha=0.4, zorder=0)
ax.legend(fontsize=9, loc='lower right')

# vertical separator between embedding and tabular
ax.axvline(1.5, color='#BDBDBD', lw=1.2, ls=':', zorder=2)
ax.text(0.5, max(aucs) * 1.07, 'Embedding\n(top-20 dims, A2)',
        ha='center', fontsize=9, color='#1565C0', fontweight='bold')
ax.text(3.0, max(aucs) * 1.07, 'Tabular features\n(A2+A3)',
        ha='center', fontsize=9, color='#E65100', fontweight='bold')

emb_patch = mpatches.Patch(color='#1565C0', label='GFP embedding (neck hook)')
tab_patch  = mpatches.Patch(color='#EF6C00', label='Tabular features')
fig.legend(handles=[emb_patch, tab_patch], loc='lower center',
           ncol=2, fontsize=9, frameon=False, bbox_to_anchor=(0.5, -0.06))

out = FIGURES / 'b2r_classify_comparison.png'
fig.savefig(str(out), dpi=180, bbox_inches='tight')
plt.close(fig)
print(f'Saved {out}')
