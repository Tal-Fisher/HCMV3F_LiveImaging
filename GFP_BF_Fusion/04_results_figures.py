#!/usr/bin/env python3
"""
04_results_figures.py

Generate summary figures and results table from already-computed OOF results.
Uses classify_top20_metrics.csv outputs (no re-fitting needed).
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

FIGURES_DIR = Path('/home/labs/ginossar/talfis/LiveImaging/GFP_BF_Fusion/figures')
RESULTS_DIR = Path('/home/labs/ginossar/talfis/LiveImaging/GFP_BF_Fusion/results')
FIGURES_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

# ── Compiled results from already-run CV ───────────────────────────────────────
records = [
    # dataset,  model,          n,   n_fast, auc,   ap,    sens,  spec,  bal_acc, mcc
    ('A2',      'GFP alone',    274, 54,     0.647, 0.326, 0.500, 0.691, 0.595,   0.160),
    ('A2',      'BF alone',     274, 54,     0.686, 0.352, 0.537, 0.741, 0.639,   0.238),
    ('A2',      'GFP+BF',       274, 54,     0.712, 0.398, 0.519, 0.764, 0.641,   0.247),
    ('A3',      'GFP alone',    223, 51,     0.569, 0.340, 0.471, 0.686, 0.578,   0.138),
    ('A3',      'BF alone',     223, 51,     0.783, 0.498, 0.706, 0.715, 0.710,   0.364),
    ('A3',      'GFP+BF',       223, 51,     0.745, 0.427, 0.588, 0.756, 0.672,   0.309),
    ('A2+A3',   'GFP alone',    497, 105,    0.635, 0.306, 0.495, 0.735, 0.615,   0.202),
    ('A2+A3',   'BF alone',     497, 105,    0.675, 0.368, 0.581, 0.648, 0.614,   0.191),
    ('A2+A3',   'GFP+BF',       497, 105,    0.710, 0.403, 0.600, 0.679, 0.639,   0.234),
]

cols = ['dataset', 'model', 'n', 'n_fast', 'auc', 'ap', 'sens', 'spec', 'bal_acc', 'mcc']
df   = pd.DataFrame(records, columns=cols)
df.to_csv(RESULTS_DIR / 'results_all_conditions.csv', index=False)
print('Saved results/results_all_conditions.csv')

# ── Figure 1: AUC bar chart ────────────────────────────────────────────────────
ds_labels    = ['A2', 'A3', 'A2+A3']
model_labels = ['GFP alone', 'BF alone', 'GFP+BF']
colours      = {'GFP alone': '#4CAF50', 'BF alone': '#2196F3', 'GFP+BF': '#FF9800'}

fig, ax = plt.subplots(figsize=(8, 5))
x       = np.arange(len(ds_labels))
width   = 0.25
offsets = [-width, 0, width]
ns      = {ds: df.loc[df['dataset'] == ds, 'n'].iloc[0] for ds in ds_labels}

for offset, model in zip(offsets, model_labels):
    aucs = [df.loc[(df['dataset'] == ds) & (df['model'] == model), 'auc'].values[0]
            for ds in ds_labels]
    bars = ax.bar(x + offset, aucs, width * 0.9, label=model,
                  color=colours[model], alpha=0.85, edgecolor='white', linewidth=0.5)
    for bar, auc in zip(bars, aucs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f'{auc:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

ax.axhline(0.5, color='gray', lw=1, ls='--', alpha=0.6, label='Random (AUC=0.500)')
ax.set_xticks(x)
ax.set_xticklabels([f'{ds}\n(n={ns[ds]})' for ds in ds_labels], fontsize=11)
ax.set_ylabel('AUC (5-fold OOF)', fontsize=11)
ax.set_title('GFP vs BF vs Fusion — b2r classification\n'
             'top-20 dims per embedding  |  ElasticNet LogReg  |  cut=1094 min', fontsize=11)
ax.set_ylim(0.4, 0.87)
ax.legend(fontsize=9, loc='upper right')
ax.spines[['top', 'right']].set_visible(False)
plt.tight_layout()
fig.savefig(str(FIGURES_DIR / 'results_auc_comparison.png'), dpi=150, bbox_inches='tight')
plt.close(fig)
print('Saved figures/results_auc_comparison.png')

# ── Figure 2: multi-metric heatmap ────────────────────────────────────────────
metrics  = ['auc', 'ap', 'sens', 'spec', 'bal_acc', 'mcc']
mlabels  = ['AUC', 'AP', 'Sensitivity', 'Specificity', 'Bal. Acc.', 'MCC']

fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
for ax, ds in zip(axes, ds_labels):
    sub  = df[df['dataset'] == ds].set_index('model')
    mat  = sub.loc[model_labels, metrics].values.astype(float)
    im   = ax.imshow(mat, aspect='auto', cmap='RdYlGn', vmin=0.1, vmax=0.9)
    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels(mlabels, rotation=35, ha='right', fontsize=9)
    ax.set_yticks(range(len(model_labels)))
    ax.set_yticklabels(model_labels, fontsize=10)
    n_f = sub.loc['GFP alone', 'n_fast']
    n   = sub.loc['GFP alone', 'n']
    ax.set_title(f'{ds}  (n={n}, {n_f} fast)', fontsize=11)
    for i in range(len(model_labels)):
        for j in range(len(metrics)):
            ax.text(j, i, f'{mat[i, j]:.3f}', ha='center', va='center',
                    fontsize=8.5, color='black', fontweight='bold')

fig.colorbar(im, ax=axes[-1], fraction=0.03, pad=0.04, label='Score')
fig.suptitle('Classification metrics — b2r (fast ≤ 1094 min)  |  5-fold OOF',
             fontsize=12, fontweight='bold')
plt.tight_layout()
fig.savefig(str(FIGURES_DIR / 'results_metrics_heatmap.png'), dpi=150, bbox_inches='tight')
plt.close(fig)
print('Saved figures/results_metrics_heatmap.png')

# ── Print summary table ────────────────────────────────────────────────────────
print(f'\n{"─"*70}')
print(f'  {"Dataset":<8}  {"Model":<12}  {"n":>4}  {"Fast":>4}  '
      f'{"AUC":>6}  {"AP":>6}  {"Sens":>6}  {"Spec":>6}  {"BalAcc":>7}')
print(f'{"─"*70}')
for _, row in df.iterrows():
    print(f'  {row["dataset"]:<8}  {row["model"]:<12}  {row["n"]:>4}  '
          f'{row["n_fast"]:>4}  {row["auc"]:>6.3f}  {row["ap"]:>6.3f}  '
          f'{row["sens"]:>6.3f}  {row["spec"]:>6.3f}  {row["bal_acc"]:>7.3f}')
print(f'{"─"*70}')
print('Done.')
