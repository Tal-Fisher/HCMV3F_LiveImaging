#!/usr/bin/env python3
"""
pca_onset_embeddings.py

PCA visualization of 256-dim Cellpose embeddings at GFP onset.
Three feature sets: BF, GFP, BF+GFP (concatenated).
Cells colored by fast vs slow (b2r <= 1094 min).

Output: dim_reduction/pca_all.png  (1x3 panel)
"""

import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
sys.path.insert(0, str(Path(__file__).parent))
from _load_data import load_embeddings_and_labels, BASE

OUT_DIR = BASE / 'dim_reduction'
OUT_DIR.mkdir(exist_ok=True)

COLOR_FAST = '#e06c75'
COLOR_SLOW = '#61afef'
ALPHA = 0.55
S = 18

X_GFP, X_BF, y = load_embeddings_and_labels()

feature_sets = [
    ('BF at GFP Onset', X_BF),
    ('GFP at Onset',    X_GFP),
    ('BF + GFP',        np.hstack([X_BF, X_GFP])),
]


def run_pca(X):
    X_sc = StandardScaler().fit_transform(X)
    pca  = PCA(n_components=2, random_state=42)
    Z    = pca.fit_transform(X_sc)
    return Z, pca.explained_variance_ratio_


def make_pca_ax(ax, Z, y, evr, title):
    mask_fast = y == 1
    mask_slow = y == 0
    ax.scatter(Z[mask_slow, 0], Z[mask_slow, 1],
               c=COLOR_SLOW, alpha=ALPHA, s=S, linewidths=0,
               label=f'Slow (n={mask_slow.sum()})')
    ax.scatter(Z[mask_fast, 0], Z[mask_fast, 1],
               c=COLOR_FAST, alpha=ALPHA, s=S, linewidths=0,
               label=f'Fast (n={mask_fast.sum()})')
    ax.set_xlabel(f'PC1 ({evr[0]*100:.1f}%)', fontsize=11)
    ax.set_ylabel(f'PC2 ({evr[1]*100:.1f}%)', fontsize=11)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.legend(fontsize=9, markerscale=1.4, framealpha=0.7)
    ax.spines[['top', 'right']].set_visible(False)


fig_all, axes = plt.subplots(1, 3, figsize=(15, 4.5))

for idx, (title, X) in enumerate(feature_sets):
    Z, evr = run_pca(X)
    print(f'{title}: PC1={evr[0]*100:.1f}%  PC2={evr[1]*100:.1f}%')
    make_pca_ax(axes[idx], Z, y, evr, title)

fig_all.suptitle('PCA of Cellpose Embeddings at GFP Onset', fontsize=13, fontweight='bold', y=1.02)
fig_all.tight_layout()
fig_all.savefig(OUT_DIR / 'pca_all.png', dpi=300, bbox_inches='tight')
plt.close(fig_all)
print('saved pca_all.png')
