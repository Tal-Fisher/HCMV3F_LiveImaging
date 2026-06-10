#!/usr/bin/env python3
"""
06_architecture_diagram.py

Architecture diagram for the GFP+BF classification pipeline.
Pure matplotlib — no data loading needed. Run locally (CPU).

Layout: 2 rows × 3 columns
  Row 1 (GLM):      GFP top-20      | BF top-20      | GFP+BF top-40
  Row 2 (Network):  GFP full (256)  | BF full (256)  | GFP+BF concat (512)

Output
------
  figures/architecture_diagram.png
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
from pathlib import Path

FIGURES_DIR = Path('/home/labs/ginossar/talfis/LiveImaging/GFP_BF_Fusion/figures')
FIGURES_DIR.mkdir(exist_ok=True)

# ── Colour palette ─────────────────────────────────────────────────────────────
C_GFP    = '#388E3C'   # dark green    — GFP embedding
C_BF     = '#455A64'   # dark blue-grey — BF embedding
C_SEL    = '#6A1B9A'   # purple        — top-k selection
C_FC     = '#1565C0'   # dark blue     — FC layer
C_FC2    = '#0277BD'   # lighter blue  — output FC
C_CONCAT = '#E65100'   # deep orange   — concatenation
C_OUT    = '#B71C1C'   # dark red      — output node
C_ANNO   = '#546E7A'   # muted label colour


# ── Drawing helpers ────────────────────────────────────────────────────────────

def box(ax, cx, cy, w, h, text, fc, tc='white', fs=8.0, bold=True, sub=None):
    """Rounded rectangle box centred at (cx, cy)."""
    rect = FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                          boxstyle='round,pad=0.025',
                          facecolor=fc, edgecolor='white', linewidth=1.4, zorder=2)
    ax.add_patch(rect)
    fw = 'bold' if bold else 'normal'
    if sub:
        ax.text(cx, cy + h * 0.18, text, ha='center', va='center',
                fontsize=fs, color=tc, fontweight=fw, zorder=3)
        ax.text(cx, cy - h * 0.22, sub, ha='center', va='center',
                fontsize=fs - 1.5, color=tc, alpha=0.82, zorder=3)
    else:
        ax.text(cx, cy, text, ha='center', va='center',
                fontsize=fs, color=tc, fontweight=fw, zorder=3)


def arrow(ax, x1, y1, x2, y2, lw=1.5, c='#444444'):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=c, lw=lw, mutation_scale=11),
                zorder=1)


def italic(ax, cx, cy, text, fs=6.8):
    ax.text(cx, cy, text, ha='center', va='center',
            fontsize=fs, color=C_ANNO, style='italic', zorder=3)


# ── Panel drawing functions ────────────────────────────────────────────────────

def panel_glm_single(ax, feat_name, feat_color, top_k=20):
    """GLM panel for a single embedding modality."""
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')
    bw, bh = 0.62, 0.095
    cx     = 0.50

    box(ax, cx, 0.870, bw, bh, f'{feat_name} Embedding', feat_color,
        sub='256 dims')
    arrow(ax, cx, 0.823, cx, 0.763)
    box(ax, cx, 0.718, bw, bh, f'Select top-{top_k}', C_SEL,
        sub='ElasticNet + LogReg rank')
    arrow(ax, cx, 0.671, cx, 0.611)
    box(ax, cx, 0.566, bw, bh, 'Logistic Regression', C_FC,
        sub=f'{top_k} features  ·  5-fold CV')
    arrow(ax, cx, 0.519, cx, 0.449)
    box(ax, cx, 0.404, bw * 0.78, bh * 0.85, 'Fast / Slow', C_OUT)


def panel_glm_dual(ax, top_k=20):
    """GLM panel for GFP+BF concatenated (top-20 each → 40)."""
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')
    bw, bh = 0.42, 0.090

    # Two inputs
    box(ax, 0.27, 0.900, bw, bh, 'GFP Embedding', C_GFP, sub='256 dims', fs=7.5)
    box(ax, 0.73, 0.900, bw, bh, 'BF Embedding',  C_BF,  sub='256 dims', fs=7.5)

    # Select top-20 from each
    arrow(ax, 0.27, 0.855, 0.27, 0.795)
    arrow(ax, 0.73, 0.855, 0.73, 0.795)
    box(ax, 0.27, 0.750, bw, bh * 0.88, f'Select top-{top_k}', C_SEL,
        sub='GFP dims', fs=7.5)
    box(ax, 0.73, 0.750, bw, bh * 0.88, f'Select top-{top_k}', C_SEL,
        sub='BF dims',  fs=7.5)

    # Converge to concat
    arrow(ax, 0.27, 0.706, 0.41, 0.646)
    arrow(ax, 0.73, 0.706, 0.59, 0.646)
    box(ax, 0.50, 0.601, 0.46, bh * 0.88, f'Concatenate  [{top_k*2}]', C_CONCAT, fs=7.5)

    arrow(ax, 0.50, 0.557, 0.50, 0.497)
    box(ax, 0.50, 0.452, 0.62, bh, 'Logistic Regression', C_FC,
        sub=f'{top_k*2} features  ·  5-fold CV', fs=7.5)
    arrow(ax, 0.50, 0.407, 0.50, 0.337)
    box(ax, 0.50, 0.292, 0.48 * 0.78, bh * 0.85, 'Fast / Slow', C_OUT, fs=7.5)


def panel_net_single(ax, feat_name, feat_color, in_dim=256):
    """Network panel for a single embedding modality."""
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')
    bw, bh = 0.66, 0.085
    cx     = 0.50
    h1     = 128 if in_dim > 300 else 64

    box(ax, cx, 0.915, bw, bh, f'{feat_name} Embedding', feat_color,
        sub=f'{in_dim} dims')
    arrow(ax, cx, 0.873, cx, 0.815)

    box(ax, cx, 0.771, bw, bh, f'FC  [{in_dim} → {h1}]', C_FC)
    italic(ax, cx, 0.732, 'ReLU  ·  Dropout(0.4)')
    arrow(ax, cx, 0.717, cx, 0.659)

    box(ax, cx, 0.615, bw, bh, f'FC  [{h1} → 32]', C_FC)
    italic(ax, cx, 0.576, 'ReLU  ·  Dropout(0.3)')
    arrow(ax, cx, 0.561, cx, 0.501)

    box(ax, cx, 0.457, bw, bh, 'FC  [32 → 1]  →  σ', C_FC2)
    arrow(ax, cx, 0.415, cx, 0.347)
    box(ax, cx, 0.303, bw * 0.78, bh * 0.85, 'Fast / Slow', C_OUT)


def panel_net_dual(ax):
    """Network panel for GFP+BF full 512-dim concatenation."""
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')
    bw, bh = 0.42, 0.085

    # Two inputs
    box(ax, 0.27, 0.945, bw, bh, 'GFP Embedding', C_GFP, sub='256 dims', fs=7.5)
    box(ax, 0.73, 0.945, bw, bh, 'BF Embedding',  C_BF,  sub='256 dims', fs=7.5)

    # Converge to concat
    arrow(ax, 0.27, 0.903, 0.42, 0.843)
    arrow(ax, 0.73, 0.903, 0.58, 0.843)
    box(ax, 0.50, 0.800, 0.46, bh * 0.88, 'Concatenate  [512]', C_CONCAT, fs=7.5)

    arrow(ax, 0.50, 0.758, 0.50, 0.700)
    box(ax, 0.50, 0.656, 0.66, bh, 'FC  [512 → 128]', C_FC, fs=7.5)
    italic(ax, 0.50, 0.617, 'ReLU  ·  Dropout(0.4)')
    arrow(ax, 0.50, 0.602, 0.50, 0.542)

    box(ax, 0.50, 0.498, 0.66, bh, 'FC  [128 → 32]', C_FC, fs=7.5)
    italic(ax, 0.50, 0.459, 'ReLU  ·  Dropout(0.3)')
    arrow(ax, 0.50, 0.444, 0.50, 0.382)

    box(ax, 0.50, 0.338, 0.66, bh, 'FC  [32 → 1]  →  σ', C_FC2, fs=7.5)
    arrow(ax, 0.50, 0.296, 0.50, 0.228)
    box(ax, 0.50, 0.184, 0.46 * 0.78, bh * 0.85, 'Fast / Slow', C_OUT, fs=7.5)


# ── Build figure ───────────────────────────────────────────────────────────────

fig, axes = plt.subplots(2, 3, figsize=(13, 9.5))
fig.patch.set_facecolor('#F5F5F5')

# GLM row
panel_glm_single(axes[0, 0], 'GFP', C_GFP)
panel_glm_single(axes[0, 1], 'BF',  C_BF)
panel_glm_dual(axes[0, 2])

# Network row
panel_net_single(axes[1, 0], 'GFP', C_GFP, in_dim=256)
panel_net_single(axes[1, 1], 'BF',  C_BF,  in_dim=256)
panel_net_dual(axes[1, 2])

# ── Column titles ──────────────────────────────────────────────────────────────
col_titles = ['GFP only', 'BF only', 'GFP + BF']
for col_i, title in enumerate(col_titles):
    axes[0, col_i].set_title(title, fontsize=11, fontweight='bold', pad=8)

# ── Row labels (left side) ─────────────────────────────────────────────────────
row_labels = [
    'GLM\n(Logistic Regression)',
    'FC Network\n(2 hidden layers)',
]
for row_i, lbl in enumerate(row_labels):
    axes[row_i, 0].text(-0.12, 0.50, lbl, transform=axes[row_i, 0].transAxes,
                         ha='center', va='center', fontsize=10,
                         fontweight='bold', color='#333333', rotation=90,
                         bbox=dict(boxstyle='round,pad=0.35', fc='#E8EAF6',
                                   ec='#9FA8DA', linewidth=1.2))

# ── Legend ─────────────────────────────────────────────────────────────────────
legend_items = [
    mpatches.Patch(facecolor=C_GFP,    label='GFP Embedding'),
    mpatches.Patch(facecolor=C_BF,     label='BF Embedding'),
    mpatches.Patch(facecolor=C_SEL,    label='Top-k dim selection'),
    mpatches.Patch(facecolor=C_CONCAT, label='Concatenation'),
    mpatches.Patch(facecolor=C_FC,     label='Fully-connected layer'),
    mpatches.Patch(facecolor=C_FC2,    label='FC output + sigmoid'),
    mpatches.Patch(facecolor=C_OUT,    label='Fast / Slow prediction'),
]
fig.legend(handles=legend_items, loc='lower center', ncol=4,
           fontsize=8.5, frameon=True, framealpha=0.9,
           bbox_to_anchor=(0.50, -0.04))

# ── Suptitle ──────────────────────────────────────────────────────────────────
fig.suptitle(
    'Classification architecture: GFP & BF Cellpose embeddings at GFP onset\n'
    'Predicting fast vs slow delay_blue_to_red  ·  5-fold stratified CV  ·  A2 + A3',
    fontsize=11.5, fontweight='bold', y=1.02)

plt.tight_layout(rect=[0.07, 0.05, 1.0, 1.0])
out_path = FIGURES_DIR / 'architecture_diagram.png'
fig.savefig(str(out_path), dpi=200, bbox_inches='tight', facecolor=fig.get_facecolor())
plt.close(fig)
print(f'Saved {out_path}', flush=True)
