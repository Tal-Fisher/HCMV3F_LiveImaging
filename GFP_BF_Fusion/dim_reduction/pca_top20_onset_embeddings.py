#!/usr/bin/env python3
"""
pca_top20_onset_embeddings.py

PCA on the pre-selected top-20 embedding dims per channel (A2+A3, HMF).
Dim indices from 07_top40_network_hmf.py log (job 516213), A2+A3 full-dataset fit.

  GFP top-20: [204,5,237,18,85,168,95,148,66,118,77,78,127,64,6,59,241,0,94,217]
  BF  top-20: [158,40,171,65,28,60,0,123,172,85,68,14,165,21,37,162,168,113,167,164]

Three feature sets: BF (20), GFP (20), BF+GFP (40).
Output: dim_reduction/pca_top20_all.png  (1x3 panel)
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

GFP_TOP20 = [204, 5, 237, 18, 85, 168, 95, 148, 66, 118, 77, 78, 127, 64, 6, 59, 241, 0, 94, 217]
BF_TOP20  = [158, 40, 171, 65, 28, 60, 0, 123, 172, 85, 68, 14, 165, 21, 37, 162, 168, 113, 167, 164]

COLOR_FAST = '#e06c75'
COLOR_SLOW = '#61afef'
ALPHA = 0.55
S = 18

X_GFP, X_BF, y = load_embeddings_and_labels()

X_GFP_t  = X_GFP[:, GFP_TOP20]
X_BF_t   = X_BF[:, BF_TOP20]
X_CONCAT = np.hstack([X_BF_t, X_GFP_t])

feature_sets = [
    ('BF at GFP Onset (top-20)', X_BF_t),
    ('GFP at Onset (top-20)',    X_GFP_t),
    ('BF + GFP (top-40)',        X_CONCAT),
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

fig_all.suptitle('PCA of Top-20 Cellpose Embedding Dims at GFP Onset', fontsize=13, fontweight='bold', y=1.02)
fig_all.tight_layout()
fig_all.savefig(OUT_DIR / 'pca_top20_all.png', dpi=300, bbox_inches='tight')
plt.close(fig_all)
print('saved pca_top20_all.png')
