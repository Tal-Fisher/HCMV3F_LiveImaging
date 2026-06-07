"""
make_assignment_gifs.py
Render 3 GIF movies of random 1000×1000-px windows, frames 100–130.
Assigned cell-nucleus pairs share a color; large circle = cell, small = nucleus.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image, ImageDraw
import random

CACHE  = Path("/home/labs/ginossar/talfis/LiveImaging/BFP_onset_analysis/cache")
FIGS   = Path("/home/labs/ginossar/talfis/LiveImaging/BFP_onset_analysis/figures")
FIGS.mkdir(exist_ok=True)

cells = pd.read_csv(CACHE / "gif_cells.csv")
nuclei = pd.read_csv(CACHE / "gif_nuclei.csv")
asgn   = pd.read_csv(CACHE / "gif_asgn.csv")

# rename nucleus Track.ID to cell_track so we can join
nuclei = nuclei.rename(columns={"Track.ID": "cell_track"})
cells  = cells.rename(columns={"Track.ID": "cell_track"})

FRAMES     = list(range(100, 131))   # 31 frames
WIN        = 1000                    # window size in pixels
CELL_R     = 14                      # cell circle radius
NUC_R      = 6                       # nucleus circle radius
BG_COLOR   = (20, 20, 20)
UNASSIGNED = (120, 120, 120)
DURATION   = 200                     # ms per frame in GIF
N_MOVIES   = 3
SEED       = 42

random.seed(SEED)
np.random.seed(SEED)

# ── build color palette: one distinct color per assigned cell_track ──────────
assigned_ids = asgn["cell_track"].unique()
rng = np.random.default_rng(SEED)

def distinct_colors(n, rng):
    """HSV-spaced colors, bright and saturated."""
    hues = np.linspace(0, 1, n, endpoint=False)
    rng.shuffle(hues)
    colors = []
    for h in hues:
        import colorsys
        r, g, b = colorsys.hsv_to_rgb(h, 0.85, 0.95)
        colors.append((int(r*255), int(g*255), int(b*255)))
    return colors

palette = dict(zip(assigned_ids, distinct_colors(len(assigned_ids), rng)))

# ── choose 3 windows that are populated with cells ───────────────────────────
# Use frame 115 (middle of range) to find dense areas
mid_cells = cells[cells["frame_idx"] == 115]

def pick_windows(mid_cells, win, n, margin=50):
    """Pick n random win×win boxes that each contain ≥ 5 cells."""
    x_max = mid_cells["X"].max()
    y_max = mid_cells["Y"].max()
    attempts, chosen = 0, []
    while len(chosen) < n and attempts < 10000:
        ox = random.uniform(margin, x_max - win - margin)
        oy = random.uniform(margin, y_max - win - margin)
        inside = mid_cells[
            (mid_cells["X"] >= ox) & (mid_cells["X"] < ox + win) &
            (mid_cells["Y"] >= oy) & (mid_cells["Y"] < oy + win)
        ]
        if len(inside) >= 5:
            # check not too overlapping with existing windows
            ok = True
            for (ex, ey) in chosen:
                if abs(ox - ex) < win * 0.5 and abs(oy - ey) < win * 0.5:
                    ok = False
                    break
            if ok:
                chosen.append((ox, oy))
        attempts += 1
    return chosen

windows = pick_windows(mid_cells, WIN, N_MOVIES)
print(f"Selected {len(windows)} windows: {[(int(x), int(y)) for x,y in windows]}")

# ── render ────────────────────────────────────────────────────────────────────
def draw_frame(ox, oy, frame_idx, win=WIN):
    img = Image.new("RGB", (win, win), BG_COLOR)
    draw = ImageDraw.Draw(img)

    fc = cells[(cells["frame_idx"] == frame_idx) &
               (cells["X"] >= ox) & (cells["X"] < ox + win) &
               (cells["Y"] >= oy) & (cells["Y"] < oy + win)]
    fn = nuclei[(nuclei["frame_idx"] == frame_idx) &
                (nuclei["X"] >= ox) & (nuclei["X"] < ox + win) &
                (nuclei["Y"] >= oy) & (nuclei["Y"] < oy + win)]

    # draw cells
    for _, row in fc.iterrows():
        px = int(row["X"] - ox)
        py = int(row["Y"] - oy)
        color = palette.get(row["cell_track"], UNASSIGNED)
        draw.ellipse([px - CELL_R, py - CELL_R, px + CELL_R, py + CELL_R],
                     outline=color, width=2)

    # draw nuclei (same color as assigned cell)
    for _, row in fn.iterrows():
        px = int(row["X"] - ox)
        py = int(row["Y"] - oy)
        color = palette.get(row["cell_track"], UNASSIGNED)
        draw.ellipse([px - NUC_R, py - NUC_R, px + NUC_R, py + NUC_R],
                     fill=color)

    # frame label
    draw.text((8, 6), f"frame {frame_idx}", fill=(200, 200, 200))
    return img

for movie_idx, (ox, oy) in enumerate(windows, 1):
    print(f"Rendering movie {movie_idx}  origin=({int(ox)}, {int(oy)}) ...", flush=True)
    frames_imgs = [draw_frame(ox, oy, f) for f in FRAMES]
    out_path = FIGS / f"assignment_movie_{movie_idx}.gif"
    frames_imgs[0].save(
        out_path,
        save_all=True,
        append_images=frames_imgs[1:],
        duration=DURATION,
        loop=0,
    )
    print(f"  Saved {out_path}")

print("Done.")
