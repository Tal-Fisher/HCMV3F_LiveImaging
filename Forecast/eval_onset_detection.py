"""
eval_onset_detection.py
Onset-detection evaluation for H=10 (all-windows) and H=5 (all-windows).

Crossing window:   max(actual_future) > RED_THRESHOLD
Predicted onset:   any predicted frame > RED_THRESHOLD
Sensitivity:       on crossing windows only
Specificity:       on non-crossing, pre-onset windows only (remaining_min >= 0)

Baselines:
  Random:       predict onset randomly at base rate p = n_crossing / (n_crossing + n_noncrossing)
  Last context: predict onset if ctx_last > RED_THRESHOLD
"""

import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

OUT           = Path("/home/labs/ginossar/talfis/LiveImaging/Forecast")
RED_THRESHOLD = 2.302
RNG_SEED      = 42

datasets = {
    "H=10 all-windows": OUT / "predictions_allwindows_h010.csv",
    "H=5  all-windows": OUT / "predictions_allwindows_h005.csv",
}

# ── Combined figure: 2 rows (sens / spec) × 2 cols (H10 / H5) ────────────────
fig, axes = plt.subplots(2, 2, figsize=(13, 9))
fig.suptitle(
    f"Onset detection — sensitivity & specificity  (RED_THRESHOLD={RED_THRESHOLD})\n"
    "Model vs Random baseline vs Last-context baseline",
    fontsize=11, fontweight="bold")

summary_rows = []

for col, (label, csv_path) in enumerate(datasets.items()):
    ax_sens = axes[0, col]
    ax_spec = axes[1, col]

    if not csv_path.exists():
        for ax in [ax_sens, ax_spec]:
            ax.set_title(f"{label}\n(file not found)")
            ax.axis("off")
        continue

    df = pd.read_csv(csv_path)
    df["actual_list"] = df["actual_traj"].apply(json.loads)
    df["pred_list"]   = df["pred_traj"].apply(json.loads)
    df["actual_max"]  = df["actual_list"].apply(max)
    df["is_crossing"] = df["actual_max"] > RED_THRESHOLD

    crossing     = df[df["is_crossing"]].copy()
    non_crossing = df[~df["is_crossing"] & (df["remaining_min"] >= 0)].copy()
    n_cross      = len(crossing)
    n_noncross   = len(non_crossing)
    base_rate    = n_cross / (n_cross + n_noncross)   # for random baseline

    print(f"\n{'='*55}", flush=True)
    print(f"{label}", flush=True)
    print(f"  Crossing windows:     {n_cross:,}", flush=True)
    print(f"  Non-crossing pre-onset: {n_noncross:,}", flush=True)
    print(f"  Base rate:            {base_rate:.3f}", flush=True)

    # ── Model predictions ────────────────────────────────────────────────────
    crossing["pred_model"]     = crossing["pred_list"].apply(
        lambda p: any(v > RED_THRESHOLD for v in p))
    non_crossing["pred_model"] = non_crossing["pred_list"].apply(
        lambda p: any(v > RED_THRESHOLD for v in p))

    # ── Random baseline (fixed seed for reproducibility) ────────────────────
    rng = np.random.default_rng(RNG_SEED)
    crossing["pred_random"]     = rng.random(n_cross)     < base_rate
    non_crossing["pred_random"] = rng.random(n_noncross)  < base_rate

    # ── Last-context baseline: predict onset if ctx_last > RED_THRESHOLD ────
    crossing["pred_ctx"]     = crossing["ctx_last"]     > RED_THRESHOLD
    non_crossing["pred_ctx"] = non_crossing["ctx_last"] > RED_THRESHOLD

    groups = ["early", "medium", "late", "all"]
    colors = ["#e67e22", "#2980b9", "#27ae60", "#7f8c8d"]

    sens_model, sens_rand, sens_ctx = [], [], []
    spec_model, spec_rand, spec_ctx = [], [], []
    n_cross_g, n_noncross_g         = [], []

    print(f"\n  {'Group':8s}  {'n_cross':>8}  {'sens_model':>10}  "
          f"{'sens_rand':>9}  {'sens_ctx':>8}  "
          f"{'n_nonc':>7}  {'spec_model':>10}  "
          f"{'spec_rand':>9}  {'spec_ctx':>8}", flush=True)

    for g in groups:
        sc = crossing     if g == "all" else crossing[crossing["group"]     == g]
        sn = non_crossing if g == "all" else non_crossing[non_crossing["group"] == g]

        sm  = sc["pred_model"].mean()  if len(sc) > 0 else 0
        sr  = sc["pred_random"].mean() if len(sc) > 0 else 0
        sct = sc["pred_ctx"].mean()    if len(sc) > 0 else 0

        pm  = 1 - sn["pred_model"].mean()  if len(sn) > 0 else 0
        pr  = 1 - sn["pred_random"].mean() if len(sn) > 0 else 0
        pct = 1 - sn["pred_ctx"].mean()    if len(sn) > 0 else 0

        sens_model.append(sm);  sens_rand.append(sr);  sens_ctx.append(sct)
        spec_model.append(pm);  spec_rand.append(pr);  spec_ctx.append(pct)
        n_cross_g.append(len(sc)); n_noncross_g.append(len(sn))

        print(f"  {g:8s}  {len(sc):8,}  {sm:10.3f}  {sr:9.3f}  {sct:8.3f}  "
              f"{len(sn):7,}  {pm:10.3f}  {pr:9.3f}  {pct:8.3f}", flush=True)

        summary_rows.append({
            "model": label, "group": g,
            "n_crossing": len(sc), "n_noncrossing": len(sn),
            "sens_model": round(sm, 3),  "sens_random": round(sr, 3),  "sens_ctx": round(sct, 3),
            "spec_model": round(pm, 3),  "spec_random": round(pr, 3),  "spec_ctx": round(pct, 3),
        })

    # ── Plot sensitivity ──────────────────────────────────────────────────────
    x      = np.arange(len(groups))
    width  = 0.25
    ax     = ax_sens
    b1 = ax.bar(x - width, sens_model, width, label="Model",        color=colors, alpha=0.85, edgecolor="white")
    b2 = ax.bar(x,          sens_rand,  width, label="Random",       color="lightgrey", alpha=0.85, edgecolor="white")
    b3 = ax.bar(x + width,  sens_ctx,   width, label="Last context", color="white",     alpha=0.85,
                edgecolor="grey", linewidth=1.2, linestyle="--")
    ax.set_xticks(x); ax.set_xticklabels(groups)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Sensitivity", fontsize=10)
    ax.set_title(f"{label}\nSensitivity on {n_cross:,} crossing windows", fontsize=9)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    for bars, vals in [(b1, sens_model), (b2, sens_rand), (b3, sens_ctx)]:
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, v + 0.02,
                    f"{v:.2f}", ha="center", va="bottom", fontsize=7)

    # ── Plot specificity ──────────────────────────────────────────────────────
    ax     = ax_spec
    b1 = ax.bar(x - width, spec_model, width, label="Model",        color=colors, alpha=0.85, edgecolor="white")
    b2 = ax.bar(x,          spec_rand,  width, label="Random",       color="lightgrey", alpha=0.85, edgecolor="white")
    b3 = ax.bar(x + width,  spec_ctx,   width, label="Last context", color="white",     alpha=0.85,
                edgecolor="grey", linewidth=1.2, linestyle="--")
    ax.set_xticks(x); ax.set_xticklabels(groups)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Specificity", fontsize=10)
    ax.set_title(f"{label}\nSpecificity on {n_noncross:,} non-crossing pre-onset windows", fontsize=9)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(axis="y", alpha=0.3)
    for bars, vals in [(b1, spec_model), (b2, spec_rand), (b3, spec_ctx)]:
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, v + 0.02,
                    f"{v:.2f}", ha="center", va="bottom", fontsize=7)

fig.tight_layout()
fig.savefig(OUT / "eval_onset_detection.png", dpi=150)
plt.close(fig)
print(f"\nSaved eval_onset_detection.png", flush=True)

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(OUT / "eval_onset_detection.csv", index=False)
print(f"Saved eval_onset_detection.csv", flush=True)
print("\nAll done.", flush=True)
