#!/usr/bin/env python3
"""
Bar chart comparing b2r regression Pearson r:
  - Embedding top-20 dims: ElasticNet (r=0.297), XGBoost (r=0.238)
  - Tabular (A2+A3): ElasticNet (r=0.285), XGBoost (r=0.340), TabICL (r=0.334)
  (BluetoRed_analysis results, no half-movie filter)
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
rs = [0.297, 0.238, 0.285, 0.340, 0.334]

EMB_COLORS = ['#1565C0', '#1E88E5']
TAB_COLORS = ['#E65100', '#EF6C00', '#FF8F00']
colors = EMB_COLORS + TAB_COLORS

x     = np.arange(len(labels))
width = 0.55

fig, ax = plt.subplots(figsize=(8, 5))

bars = ax.bar(x, rs, width=width, color=colors, zorder=3,
              edgecolor='white', linewidth=0.6)

for bar, v in zip(bars, rs):
    ax.text(bar.get_x() + bar.get_width() / 2, v + 0.004,
            f'{v:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

# highlight best bar
best = int(np.argmax(rs))
bars[best].set_edgecolor('#FFD600')
bars[best].set_linewidth(2.5)

ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=10)
ax.set_ylabel('Pearson r', fontsize=12)
ax.set_ylim(0, max(rs) * 1.18)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(axis='y', lw=0.5, alpha=0.4, zorder=0)

ax.axvline(1.5, color='#BDBDBD', lw=1.2, ls=':', zorder=2)
ax.text(0.5, max(rs) * 1.12, 'Embedding\n(top-20 dims, A2)',
        ha='center', fontsize=9, color='#1565C0', fontweight='bold')
ax.text(3.0, max(rs) * 1.12, 'Tabular features\n(A2+A3)',
        ha='center', fontsize=9, color='#E65100', fontweight='bold')

emb_patch = mpatches.Patch(color='#1565C0', label='GFP embedding (neck hook)')
tab_patch  = mpatches.Patch(color='#EF6C00', label='Tabular features')
fig.legend(handles=[emb_patch, tab_patch], loc='lower center',
           ncol=2, fontsize=9, frameon=False, bbox_to_anchor=(0.5, -0.06))

out = FIGURES / 'b2r_regression_comparison.png'
fig.savefig(str(out), dpi=180, bbox_inches='tight')
plt.close(fig)
print(f'Saved {out}')
