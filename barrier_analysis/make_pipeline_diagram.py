"""
make_pipeline_diagram.py — Barrier analysis pipeline as a matplotlib flowchart.
Rendered with matplotlib because mmdc (Mermaid CLI) is not available on this cluster.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from pathlib import Path

OUT = Path("/home/labs/ginossar/talfis/LiveImaging/barrier_analysis/analysis_pipeline.png")

# ── color scheme ──────────────────────────────────────────────────────────────
C_DATA   = "#dce8f5"   # light blue — data files
C_R      = "#fde8d0"   # light orange — R scripts
C_PY     = "#e3f5e1"   # light green — Python scripts
C_FIG    = "#f0e6f6"   # light purple — output figures
C_EDGE   = "#555555"

def box(ax, x, y, w, h, text, color, fontsize=8.5, bold=False):
    patch = FancyBboxPatch((x - w/2, y - h/2), w, h,
                           boxstyle="round,pad=0.04",
                           facecolor=color, edgecolor=C_EDGE, linewidth=0.9)
    ax.add_patch(patch)
    weight = "bold" if bold else "normal"
    ax.text(x, y, text, ha="center", va="center",
            fontsize=fontsize, fontweight=weight, wrap=True,
            multialignment="center")

def arrow(ax, x0, y0, x1, y1, label=""):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", color=C_EDGE, lw=1.1))
    if label:
        mx, my = (x0 + x1) / 2, (y0 + y1) / 2
        ax.text(mx + 0.02, my, label, fontsize=7, color="#444", va="center")

fig, ax = plt.subplots(figsize=(10, 13))
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")
fig.patch.set_facecolor("white")

fig.suptitle("Barrier Analysis Pipeline — HCMV Live Imaging",
             fontsize=12, fontweight="bold", y=0.98)

# ── raw data row ──────────────────────────────────────────────────────────────
box(ax, 0.25, 0.91, 0.32, 0.055,
    "spots_clean.rds\n(cell spots, ch2_corrected)\n81 MB — 343k rows",
    C_DATA)
box(ax, 0.72, 0.91, 0.32, 0.055,
    "nuc_assigned.rds\n(nucleus spots, assigned to cells)\n53 MB — 261k rows",
    C_DATA)

# ── R export ──────────────────────────────────────────────────────────────────
box(ax, 0.50, 0.775, 0.36, 0.055,
    "scripts/export_timeseries.R\nmerge cell + nucleus | apply first-half filter\nexport per-frame time series",
    C_R, bold=True)

arrow(ax, 0.25, 0.882, 0.38, 0.803)
arrow(ax, 0.72, 0.882, 0.62, 0.803)

# ── onset / model_df inputs to R ──────────────────────────────────────────────
box(ax, 0.10, 0.775, 0.15, 0.045,
    "onset_df.rds\nred/GFP onset times",
    C_DATA, fontsize=7.5)
box(ax, 0.88, 0.775, 0.18, 0.045,
    "model_df.csv\nmovie_half_min filter",
    C_DATA, fontsize=7.5)

arrow(ax, 0.175, 0.775, 0.32, 0.775)
arrow(ax, 0.79, 0.775, 0.68, 0.775)

# ── exported CSV ──────────────────────────────────────────────────────────────
box(ax, 0.50, 0.660, 0.38, 0.055,
    "cache/python_export/timeseries_data.csv\n809 cells · 148k rows · 20 columns\n(Track.ID, Frame, T_min, ch2_corrected, …)",
    C_DATA, bold=True)

arrow(ax, 0.50, 0.747, 0.50, 0.688)

# ── Python script ─────────────────────────────────────────────────────────────
box(ax, 0.50, 0.555, 0.38, 0.065,
    "barrier_analysis/barrier_analysis.py\nalign to GFP onset · label productive cells\nassign timing groups · compute peak values\nROC thresholds · crossing timing",
    C_PY, bold=True)

arrow(ax, 0.50, 0.632, 0.50, 0.588)

# ── model_df input to Python ──────────────────────────────────────────────────
box(ax, 0.88, 0.555, 0.18, 0.050,
    "model_df.csv\nproductive labels\ntiming groups",
    C_DATA, fontsize=7.5)
arrow(ax, 0.79, 0.555, 0.69, 0.555)

# ── output figures ────────────────────────────────────────────────────────────
out_y = [0.400, 0.295, 0.190, 0.085]
out_labels = [
    "Fig 1: barrier_trajectories_gfp.png\nMedian ± IQR trajectories from GFP onset\n(early / medium / late / non-productive)",
    "Fig 2: barrier_trajectories_red.png\nMedian ± IQR trajectories before red onset\n(productive only; aligned to mCherry onset)",
    "Fig 3: barrier_thresholds.png\nPeak values: productive vs non-productive\nROC-optimal threshold per feature (AUC)",
    "Fig 4: barrier_crossing.png\nWhen do productive cells first cross threshold?\nDistribution of crossing lag before red onset",
]

for y, lbl in zip(out_y, out_labels):
    box(ax, 0.50, y, 0.50, 0.072, lbl, C_FIG)
    arrow(ax, 0.50, 0.522, 0.50, y + 0.036)

# ── legend ────────────────────────────────────────────────────────────────────
legend_items = [
    mpatches.Patch(facecolor=C_DATA,  edgecolor=C_EDGE, label="Data files"),
    mpatches.Patch(facecolor=C_R,     edgecolor=C_EDGE, label="R script"),
    mpatches.Patch(facecolor=C_PY,    edgecolor=C_EDGE, label="Python script"),
    mpatches.Patch(facecolor=C_FIG,   edgecolor=C_EDGE, label="Output figure"),
]
ax.legend(handles=legend_items, loc="lower right", fontsize=8,
          bbox_to_anchor=(0.98, 0.01), framealpha=0.9)

plt.tight_layout(rect=[0, 0, 1, 0.97])
fig.savefig(OUT, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved {OUT}")
