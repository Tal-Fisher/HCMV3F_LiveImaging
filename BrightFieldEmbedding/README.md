# BrightFieldEmbedding

## Goal
Predict cell fate from brightfield morphology at **movie frame 0** — before any fluorescent marker is visible. Uses Cellpose embeddings extracted from the brightfield channel to predict `delay_green_to_red`, early/late classification, and related outcomes.

## Dataset
A2 only.

## Biological question
Can the morphology of a cell, as seen in brightfield at the very start of the movie (before GFP onset), predict how quickly it will proceed through the HCMV infection program?

## Pipeline overview
1. **01_bf_gfp_overlap.py** — Validate that brightfield TrackMate tracks correspond to the correct GFP-tracked cells. Matches BF centroids to GFP centroids at GFP onset, checks temporal consistency at 3 later frames, and confirms BF tracks reach frame 0.
2. *(planned)* Extract Cellpose latent embeddings from BF images at frame 0 for confidently matched cells.
3. *(planned)* Predict delay_green_to_red and early/late class from those embeddings.
4. *(planned)* Compare with GFP-channel embedding baseline (AUC = 0.742 from CellposeEmbedding/).

## Files
| File | Description |
|------|-------------|
| `bf_gfp_matches.csv` | Matched BF↔GFP pairs with distance and consistency metrics |
| `summary.txt` | Console summary from step 1 |
| `figures/01_distance_histogram.png` | Distribution of BF–GFP centroid distances at onset |
| `figures/02_centroid_scatter.png` | Spatial map of GFP cells coloured by match quality |
| `figures/03_temporal_consistency.png` | BF–GFP distance over 3 later timepoints |
| `figures/04_backtrace_bar.png` | % of matched BF tracks reaching frame 0 |

## Input data (CompleteImage/)
| File | Role |
|------|------|
| `A2_Merged_spots.csv` | GFP/merged channel TrackMate tracks |
| `A2_gfp_onset.csv` | Precomputed GFP onset frame + position per cell |
| `A2_BrightField_spots.csv` | Brightfield channel TrackMate tracks |

## Constants
- `PIXEL_SCALE = 0.2871 µm/px`
- All coordinates in CSV files are in **µm**
