"""
future_infection.py  —  How many non-productive cells would turn red if observed longer?

We have cells that turned GFP-positive (IE gene expression) but never turned mCherry-positive
(late gene expression) during the movie.  These non-productive cells are right-censored:
we observed them for a limited window after GFP onset and saw no red transition.

This script estimates how many of the 312 non-productive cells would have turned red if
the movie had continued, using the green-to-red delay distribution of productive cells.

Run:  python3 future_infection/future_infection.py
Outputs saved to:  future_infection/
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import lognorm, weibull_min
from scipy.stats import gaussian_kde

BASE    = Path("/home/labs/ginossar/talfis/LiveImaging")
MD_CSV  = BASE / "cache" / "python_export" / "model_df.csv"
OUT_DIR = BASE / "future_infection"
OUT_DIR.mkdir(exist_ok=True)

EARLY_CUT = 911
LATE_CUT  = 2163

GROUP_COLORS = {
    "early":          "#e67e22",
    "medium":         "#2980b9",
    "late":           "#27ae60",
    "non-productive": "#888888",
}

# ══════════════════════════════════════════════════════════════════════════════
# Load & prepare data
# ══════════════════════════════════════════════════════════════════════════════
print("Loading data ...", flush=True)
md = pd.read_csv(MD_CSV)
md = md[md["abs_gfp_onset_min"] <= md["movie_half_min"]].copy()   # first-half filter

md["productive"]   = np.isfinite(md["delay_green_to_red"])
md["censor_time"]  = np.where(
    md["productive"],
    md["delay_green_to_red"],
    md["movie_half_min"] * 2 - md["abs_gfp_onset_min"],
)
md["event"] = md["productive"].astype(int)

def assign_group(row):
    if not row["productive"]:
        return "non-productive"
    d = row["delay_green_to_red"]
    return "early" if d <= EARLY_CUT else ("medium" if d <= LATE_CUT else "late")

md["group"] = md.apply(assign_group, axis=1)

prod   = md[md["productive"]].copy()
nonprod = md[~md["productive"]].copy()

n_all  = len(md)
n_prod = len(prod)
n_np   = len(nonprod)
print(f"  {n_all} cells  ({n_prod} productive, {n_np} non-productive)", flush=True)

delays_prod  = prod["delay_green_to_red"].values
censor_np    = nonprod["censor_time"].values

# ══════════════════════════════════════════════════════════════════════════════
# Kaplan-Meier estimator
# ══════════════════════════════════════════════════════════════════════════════
def kaplan_meier(times, events):
    """
    Standard KM estimator with Greenwood 95% CI (log-log transform).
    Ties: events counted before censored observations at the same time.
    Returns arrays (t, S, lower_95, upper_95).
    """
    times  = np.asarray(times,  dtype=float)
    events = np.asarray(events, dtype=int)
    # sort by time; among ties, events first (lexsort: secondary key first)
    order  = np.lexsort((-events, times))
    times, events = times[order], events[order]
    n = len(times)

    t_list, S_list, lo_list, hi_list = [], [], [], []
    S, gw, i = 1.0, 0.0, 0

    while i < n:
        t = times[i]
        j = i
        while j < n and times[j] == t:
            j += 1
        n_risk = n - i
        d = int(events[i:j].sum())
        if d > 0:
            S = S * (1 - d / n_risk)
            if n_risk > d:
                gw += d / (n_risk * (n_risk - d))
            if 0 < S < 1 and gw > 0:
                se_ll = np.sqrt(gw) / abs(np.log(S))
                lo = float(S ** np.exp( 1.96 * se_ll))
                hi = float(S ** np.exp(-1.96 * se_ll))
            else:
                lo = hi = float(S)
            t_list.append(t)
            S_list.append(float(S))
            lo_list.append(lo)
            hi_list.append(hi)
        i = j

    return (np.array(t_list), np.array(S_list),
            np.array(lo_list), np.array(hi_list))


print("Fitting Kaplan-Meier ...", flush=True)
km_t, km_S, km_lo, km_hi = kaplan_meier(md["censor_time"], md["event"])

def km_eval(t_arr):
    """Vectorised KM step-function evaluation."""
    t_arr  = np.asarray(t_arr, dtype=float)
    result = np.ones(t_arr.shape)
    for i, t in enumerate(t_arr.flat):
        idx = int(np.searchsorted(km_t, t, side="right")) - 1
        result.flat[i] = km_S[idx] if idx >= 0 else 1.0
    return result

km_plateau = km_S[-1]   # S at last event ≈ estimated cure fraction
print(f"  KM at last event (t={km_t[-1]:.0f} min): S={km_plateau:.4f}  "
      f"→ estimated max eventual productive = {(1-km_plateau)*100:.1f}%", flush=True)

# ══════════════════════════════════════════════════════════════════════════════
# Parametric fit — log-normal and Weibull
# ══════════════════════════════════════════════════════════════════════════════
print("Fitting parametric distributions ...", flush=True)
shape_ln, loc_ln, scale_ln = lognorm.fit(delays_prod, floc=0)
shape_wb, loc_wb, scale_wb = weibull_min.fit(delays_prod, floc=0)

ll_ln = lognorm.logpdf(delays_prod, s=shape_ln, loc=0, scale=scale_ln).sum()
ll_wb = weibull_min.logpdf(delays_prod, c=shape_wb, loc=0, scale=scale_wb).sum()
aic_ln = -2 * ll_ln + 4
aic_wb = -2 * ll_wb + 4

mu_ln    = np.log(scale_ln)
sigma_ln = shape_ln
print(f"  Log-normal: mu={mu_ln:.3f}  sigma={sigma_ln:.3f}  AIC={aic_ln:.1f}", flush=True)
print(f"  Weibull:    k={shape_wb:.3f}  lambda={scale_wb:.1f}  AIC={aic_wb:.1f}", flush=True)

def ln_surv(t_arr):
    return lognorm.sf(np.asarray(t_arr, dtype=float), s=shape_ln, loc=0, scale=scale_ln)

# ══════════════════════════════════════════════════════════════════════════════
# Extrapolation: expected additional red cells vs extra observation time
# ══════════════════════════════════════════════════════════════════════════════
print("Computing extrapolation ...", flush=True)

dt_hours = np.linspace(0, 120, 500)   # 0 to 5 days
dt_mins  = dt_hours * 60

def expected_additional(dt_mins_arr, censor_times, surv_fn):
    """
    For each extra ΔT, sum over non-productive cells:
      P(turns red in (c_i, c_i+ΔT] | not red by c_i) = [S(c_i) - S(c_i+ΔT)] / S(c_i)
    """
    c = censor_times
    out = np.zeros(len(dt_mins_arr))
    s_c = surv_fn(c)
    for k, dt in enumerate(dt_mins_arr):
        s_ct = surv_fn(c + dt)
        probs = np.where(s_c > 1e-10, (s_c - s_ct) / s_c, 0.0)
        out[k] = np.clip(probs, 0, 1).sum()
    return out

# Empirical survival function — direct from observed productive cell delays
sorted_delays = np.sort(delays_prod)
n_prod_d      = len(sorted_delays)

def emp_surv(t_arr):
    """Fraction of productive cells with delay > t (empirical, no parametric assumption)."""
    t_arr = np.asarray(t_arr, dtype=float)
    idx   = np.searchsorted(sorted_delays, t_arr, side="right")
    return (n_prod_d - idx) / n_prod_d

exp_ln  = expected_additional(dt_mins, censor_np, ln_surv)
exp_km  = expected_additional(dt_mins, censor_np, km_eval)
exp_emp = expected_additional(dt_mins, censor_np, emp_surv)

# Summary table
key_hours = [0, 6, 12, 24, 48, 72, 120]
rows = []
for h in key_hours:
    idx = int(np.argmin(np.abs(dt_hours - h)))
    rows.append({
        "extra_time_h":                h,
        "expected_additional_empirical": round(float(exp_emp[idx]), 1),
        "expected_additional_lognorm":   round(float(exp_ln[idx]), 1),
        "expected_additional_km":        round(float(exp_km[idx]), 1),
        "pct_nonprod_empirical":         round(float(exp_emp[idx]) / n_np * 100, 1),
        "pct_nonprod_lognorm":           round(float(exp_ln[idx]) / n_np * 100, 1),
        "expected_total_prod":           round(n_prod + float(exp_emp[idx]), 1),
        "expected_pct_all_prod":         round((n_prod + float(exp_emp[idx])) / n_all * 100, 1),
    })
summary_df = pd.DataFrame(rows)
summary_df.to_csv(OUT_DIR / "extrapolation_summary.csv", index=False)
print("\nSummary table:", flush=True)
print(summary_df.to_string(index=False), flush=True)

# ══════════════════════════════════════════════════════════════════════════════
# Figure 1 — Kaplan-Meier survival curve
# ══════════════════════════════════════════════════════════════════════════════
print("\nFigure 1: KM curve ...", flush=True)

fig, ax = plt.subplots(figsize=(9, 5.5))

# build step function arrays: prepend (0, 1) so the curve starts from origin
km_t_h = km_t / 60
t_plot  = np.concatenate([[0], km_t_h])
S_plot  = np.concatenate([[1.0], km_S])
lo_plot = np.concatenate([[1.0], km_lo])
hi_plot = np.concatenate([[1.0], km_hi])

ax.fill_between(t_plot, lo_plot, hi_plot, step="post",
                alpha=0.2, color="#2980b9", label="95% CI")
ax.step(t_plot, S_plot, where="post", color="#2980b9", lw=2,
        label="Kaplan-Meier S(t)")

# mark non-productive censoring percentiles
for p, ls, lbl in zip([25, 50, 75], [":", "--", ":"], ["Q1", "median", "Q3"]):
    pct = float(np.percentile(censor_np, p)) / 60
    ax.axvline(pct, color="tomato", linestyle=ls, lw=1.2,
               label=f"non-prod {lbl} obs. ({pct:.0f} h)")

# current observed fraction
ax.axhline(1 - n_prod / n_all, color="grey", linestyle="--", lw=1,
           label=f"current productive fraction ({n_prod}/{n_all} = {n_prod/n_all*100:.1f}%)")

ax.set_xlabel("Time from GFP onset (hours)", fontsize=12)
ax.set_ylabel("Fraction not yet mCherry-positive  S(t)", fontsize=12)
ax.set_title("Kaplan-Meier estimate — time from GFP onset to mCherry onset\n"
             "(non-productive cells censored at movie end)", fontsize=11)
ax.set_xlim(0, km_t_h[-1] * 1.05)
ax.set_ylim(-0.02, 1.05)
ax.legend(fontsize=8, loc="upper right")
ax.annotate(f"KM at t={km_t[-1]/60:.0f} h: S={km_plateau:.3f}\n"
            f"→ ≤{km_plateau*100:.1f}% may never turn red",
            xy=(km_t_h[-1], km_plateau),
            xytext=(km_t_h[-1] * 0.7, km_plateau + 0.12),
            arrowprops=dict(arrowstyle="->", color="black", lw=1),
            fontsize=9, ha="center")
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(OUT_DIR / "km_curve.png", dpi=150)
plt.close(fig)
print("  Saved km_curve.png", flush=True)

# ══════════════════════════════════════════════════════════════════════════════
# Figure 2 — Parametric fit quality
# ══════════════════════════════════════════════════════════════════════════════
print("Figure 2: parametric fit ...", flush=True)

t_grid = np.linspace(0, delays_prod.max() * 1.1, 500)

fig, axes = plt.subplots(1, 2, figsize=(11, 5))

# --- Left: KDE + fitted PDFs ---
ax = axes[0]
kde = gaussian_kde(delays_prod, bw_method=0.2)
ax.plot(t_grid / 60, kde(t_grid) * 60, color="black", lw=1.5, label="Observed KDE")
ax.plot(t_grid / 60,
        lognorm.pdf(t_grid, s=shape_ln, loc=0, scale=scale_ln) * 60,
        color="#e67e22", lw=2,
        label=f"Log-normal  AIC={aic_ln:.0f}\n(μ={mu_ln:.2f}, σ={sigma_ln:.2f})")
ax.plot(t_grid / 60,
        weibull_min.pdf(t_grid, c=shape_wb, loc=0, scale=scale_wb) * 60,
        color="#2980b9", lw=2, linestyle="--",
        label=f"Weibull  AIC={aic_wb:.0f}\n(k={shape_wb:.2f}, λ={scale_wb/60:.1f} h)")
ax.set_xlabel("GFP → mCherry delay (hours)", fontsize=11)
ax.set_ylabel("Density (per hour)", fontsize=11)
ax.set_title(f"Productive cell delay distribution\n(n={n_prod})", fontsize=11)
ax.legend(fontsize=8)
ax.grid(axis="y", alpha=0.3)

# --- Right: QQ plot vs log-normal ---
ax = axes[1]
sorted_d = np.sort(delays_prod)
n_d = len(sorted_d)
probs = (np.arange(1, n_d + 1) - 0.5) / n_d
theoretical = lognorm.ppf(probs, s=shape_ln, loc=0, scale=scale_ln)
ax.scatter(theoretical / 60, sorted_d / 60, s=8, alpha=0.5, color="#e67e22")
ref_max = max(theoretical.max(), sorted_d.max()) / 60
ax.plot([0, ref_max], [0, ref_max], "k--", lw=1, label="y = x")
ax.set_xlabel("Log-normal theoretical quantiles (hours)", fontsize=11)
ax.set_ylabel("Observed delay (hours)", fontsize=11)
ax.set_title("QQ plot: productive delays vs log-normal fit", fontsize=11)
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

fig.suptitle("Parametric fit to productive cell GFP→mCherry delays", fontsize=12, y=1.01)
fig.tight_layout()
fig.savefig(OUT_DIR / "parametric_fit.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("  Saved parametric_fit.png", flush=True)

# ══════════════════════════════════════════════════════════════════════════════
# Figure 3 — Expected additional red cells vs extra observation time (main)
# ══════════════════════════════════════════════════════════════════════════════
print("Figure 3: extrapolation curves ...", flush=True)

fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(dt_hours, exp_ln, color="#e74c3c", lw=2, linestyle="--",
        label="Log-normal parametric fit")

# right y-axis: % of non-productive — sync after setting main ylim
y_max = exp_ln.max() * 1.08
ax.set_ylim(0, y_max)
ax2 = ax.twinx()
ax2.set_ylim(0, y_max / n_np * 100)
ax2.set_ylabel(f"% of {n_np} non-productive cells", fontsize=11, color="grey")
ax2.tick_params(axis="y", colors="grey")

# horizontal reference lines
for frac, lbl in zip([0.10, 0.25, 0.50], ["10%", "25%", "50%"]):
    ax.axhline(frac * n_np, color="grey", linestyle=":", lw=1, alpha=0.6)
    ax.text(dt_hours[-1] * 0.99, frac * n_np + 1, lbl,
            ha="right", va="bottom", fontsize=8, color="grey")

# vertical annotations using empirical curve as reference
for h in [6, 12, 24, 48, 72]:
    idx   = int(np.argmin(np.abs(dt_hours - h)))
    n_add = exp_ln[idx]
    ax.axvline(h, color="black", lw=0.7, linestyle=":", alpha=0.4)
    ax.text(h, y_max * 0.96, f"+{h}h\n~{n_add:.0f}",
            ha="center", va="top", fontsize=8)

ax.set_xlabel("Additional observation time beyond movie end (hours)", fontsize=12)
ax.set_ylabel(f"Expected additional mCherry-positive cells  (out of {n_np})", fontsize=12)
ax.set_title("Estimated additional productive cells if movie continued\n"
             "(assumes non-productive delay distribution = productive cells)", fontsize=11)
ax.set_xlim(0, dt_hours[-1])
ax.legend(fontsize=10, loc="lower right")
ax.grid(axis="y", alpha=0.3)

fig.tight_layout()
fig.savefig(OUT_DIR / "extrapolation_curves.png", dpi=150)
plt.close(fig)
print("  Saved extrapolation_curves.png", flush=True)

# ══════════════════════════════════════════════════════════════════════════════
# Figure 4 — Breakdown by GFP onset timing group
# ══════════════════════════════════════════════════════════════════════════════
print("Figure 4: onset group breakdown ...", flush=True)

groups = ["early", "medium", "late"]
group_labels = {
    "early":  f"Early onset\n(GFP ≤ {EARLY_CUT//60:.0f} h)",
    "medium": f"Medium onset\n({EARLY_CUT//60:.0f}–{LATE_CUT//60:.0f} h)",
    "late":   f"Late onset\n(GFP > {LATE_CUT//60:.0f} h)",
}

fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharey=False)

for ax, grp in zip(axes, groups):
    # non-productive cells in this group (defined by abs_gfp_onset_min)
    if grp == "early":
        mask = nonprod["abs_gfp_onset_min"] <= EARLY_CUT
    elif grp == "medium":
        mask = (nonprod["abs_gfp_onset_min"] > EARLY_CUT) & \
               (nonprod["abs_gfp_onset_min"] <= LATE_CUT)
    else:
        mask = nonprod["abs_gfp_onset_min"] > LATE_CUT

    c_grp = nonprod.loc[mask, "censor_time"].values
    n_grp = len(c_grp)

    if n_grp == 0:
        ax.set_title(group_labels[grp] + "\n(n=0)")
        continue

    # histogram of censoring times
    ax2 = ax.twinx()
    ax2.hist(c_grp / 60, bins=15, color="lightgrey", alpha=0.6, edgecolor="white")
    ax2.set_ylabel("# non-productive cells", fontsize=9, color="grey")
    ax2.tick_params(axis="y", colors="grey")

    # extrapolation for this subgroup — both empirical and log-normal
    exp_grp_ln  = expected_additional(dt_mins, c_grp,
                      lambda t: lognorm.sf(np.asarray(t), s=shape_ln, loc=0, scale=scale_ln))
    exp_grp_km  = expected_additional(dt_mins, c_grp, km_eval)
    ax.plot(dt_hours, exp_grp_ln / n_grp * 100, color=GROUP_COLORS[grp], lw=2,
            linestyle="--", label="Log-normal")
    ax.plot(dt_hours, exp_grp_km / n_grp * 100, color=GROUP_COLORS[grp], lw=1.5,
            linestyle=":", alpha=0.8, label="KM-based")
    ax.legend(fontsize=7)
    exp_grp = exp_grp_ln  # use log-normal for annotations
    ax.set_xlabel("Extra observation time (hours)", fontsize=10)
    ax.set_ylabel("% of group turning red", fontsize=10)
    ax.set_title(f"{group_labels[grp]}\n(n={n_grp} non-productive)", fontsize=10)
    ax.set_xlim(0, dt_hours[-1])
    ax.set_ylim(0, None)
    ax.grid(axis="y", alpha=0.3)

    # annotate at 24h and 72h
    for h in [24, 72]:
        idx = int(np.argmin(np.abs(dt_hours - h)))
        pct = exp_grp[idx] / n_grp * 100
        ax.annotate(f"+{h}h: {pct:.0f}%",
                    xy=(h, pct), xytext=(h + 3, pct + 2),
                    fontsize=8, color=GROUP_COLORS[grp],
                    arrowprops=dict(arrowstyle="->", color=GROUP_COLORS[grp], lw=0.8))

fig.suptitle("Expected fraction turning red — by GFP onset timing group", fontsize=12)
fig.tight_layout()
fig.savefig(OUT_DIR / "extrapolation_by_group.png", dpi=150)
plt.close(fig)
print("  Saved extrapolation_by_group.png", flush=True)

# ══════════════════════════════════════════════════════════════════════════════
# Methodology text
# ══════════════════════════════════════════════════════════════════════════════
idx_24 = int(np.argmin(np.abs(dt_hours - 24)))
idx_72 = int(np.argmin(np.abs(dt_hours - 72)))

meth_txt = f"""Future Infection — Extrapolation Analysis
==========================================
Generated by: future_infection/future_infection.py

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. BIOLOGICAL RATIONALE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Cells that turn GFP-positive (IE gene expression) but never turn mCherry-positive
(late gene expression) during the movie are called non-productive. However, some of
these cells may simply have been censored by the finite movie duration: they might
have turned red given more observation time.

This analysis uses the distribution of green-to-red delays in productive cells as a
reference to estimate how many non-productive cells would have turned red if the movie
had continued.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
2. DATA
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Source: A2 + A3, first-half filter (abs_gfp_onset_min ≤ movie_half_min)
  Total:  {n_all} cells
  Productive (turned red):      {n_prod}  (delay: min={delays_prod.min():.0f}, "
f"median={np.median(delays_prod):.0f}, max={delays_prod.max():.0f} min)
  Non-productive (censored):    {n_np}  (obs. window after GFP: "
f"median={np.median(censor_np):.0f}, max={np.max(censor_np):.0f} min)

Key imbalance: median non-productive observation window ({np.median(censor_np):.0f} min)
is less than the median productive delay ({np.median(delays_prod):.0f} min), so many
non-productive cells were simply not watched long enough.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
3. METHODS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Step 1 — Kaplan-Meier (KM) estimator on all {n_all} cells:
  Productive cells: event = 1, survival time = delay_green_to_red
  Non-productive:   event = 0, survival time = movie_max_min − abs_gfp_onset_min
                   (= 2 × movie_half_min − abs_gfp_onset_min; full movie, not half)
  95% CI computed via Greenwood formula with log-log transform.
  KM survival at last event (t={km_t[-1]/60:.0f} h): S={km_plateau:.4f}
  Interpretation: at most {km_plateau*100:.1f}% of cells may never turn red.

Step 2 — Log-normal parametric fit to productive cell delays (scipy.stats.lognorm):
  Parameters: μ={mu_ln:.3f}, σ={sigma_ln:.3f} (log-scale)
  AIC: log-normal={aic_ln:.1f}, Weibull={aic_wb:.1f}
  {"Log-normal preferred (lower AIC)" if aic_ln < aic_wb else "Weibull slightly preferred (lower AIC); log-normal used for interpretability"}
  The log-normal fit allows extrapolation beyond the last observed productive delay.

Step 3 — Conditional probability of turning red in extra time ΔT:
  For each non-productive cell i with observation window c_i:
    P(turns red in (c_i, c_i+ΔT] | not red by c_i) = [S(c_i) − S(c_i+ΔT)] / S(c_i)
  Three survival functions S(t) are compared:
    A. Empirical: S(t) = fraction of the {n_prod} productive cells with delay > t
       (no parametric assumption; naturally caps at t={delays_prod.max():.0f} min)
    B. Log-normal: parametric fit from Step 2 (allows extrapolation beyond last event)
    C. KM: Kaplan-Meier on all cells (productive + non-productive as censored)
  Summing over all {n_np} non-productive cells gives the expected additional count.
  Primary result uses the empirical distribution (Method A).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
4. KEY RESULTS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Current productive fraction: {n_prod}/{n_all} = {n_prod/n_all*100:.1f}%

  If movie continued (empirical distribution):
    +24 hours: ~{exp_emp[idx_24]:.0f} additional cells expected to turn red "
f"({exp_emp[idx_24]/n_np*100:.1f}% of non-productive)
    +72 hours: ~{exp_emp[idx_72]:.0f} additional cells expected to turn red "
f"({exp_emp[idx_72]/n_np*100:.1f}% of non-productive)

  Upper bound (all non-productive eventually turn red):
    {n_np} additional cells → total productive fraction = {(n_prod+n_np)/n_all*100:.1f}%

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
5. KEY ASSUMPTION AND CAVEAT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
The log-normal delay distribution from productive cells is assumed to apply to
non-productive cells that would eventually turn red. This is an UPPER BOUND because:
  a) Some non-productive cells may not be infected (no actual HCMV entry despite GFP)
  b) Some may be in a dead-end infection state and will never produce late antigen
  c) The cure fraction from the KM ({km_plateau*100:.1f}%) places an upper bound on the
     fraction that truly never turns red, but this estimate is noisy at long follow-up times.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
6. FIGURES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
km_curve.png:
  Kaplan-Meier survival curve (S(t) = fraction not yet red, from GFP onset).
  Vertical dashed lines = 25th/50th/75th percentile of non-productive observation windows.

parametric_fit.png:
  Left: observed KDE of productive delays + log-normal and Weibull fitted PDFs.
  Right: QQ plot of productive delays vs log-normal (should follow y=x if fit is good).

extrapolation_curves.png (MAIN RESULT):
  Expected additional mCherry-positive cells vs. extra observation time.
  Solid red = log-normal extrapolation; dashed blue = KM-based (plateaus at last event).
  Right y-axis shows percentage of the {n_np} non-productive cells.

extrapolation_by_group.png:
  Same extrapolation split by GFP onset timing group (early/medium/late).
  Grey histogram = distribution of observation window lengths per group.
"""

(OUT_DIR / "methodology.txt").write_text(meth_txt)
print("  Saved methodology.txt", flush=True)
print("\nAll done.", flush=True)
