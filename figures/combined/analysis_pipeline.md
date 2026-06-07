# HCMV Live Imaging — Analysis Pipeline

```mermaid
flowchart TD
    %% ── Raw data ──────────────────────────────────────────
    RAW["📁 Raw data\nA2_Merged_spots.csv\nA2_nuclei_spots.csv\n(+ A3 equivalents)"]

    %% ── R pipeline ────────────────────────────────────────
    R01["01_preprocess.R\nBleedthrough correction (α=0.05)\nFilter early-GFP cells"]
    R02["02_nuclei.R\nAssign nuclei to cell tracks\n(spatial nearest-neighbour)"]
    R03["03_onset.R\nDetect GFP / BFP / mCherry onsets\ncompute delay_green_to_red"]
    R04["04_features.R\n16-frame feature window at GFP onset\n29 features (excl. 2 all-NA SNR cols)"]
    R07["07_combine.R\nMerge A2 + A3\nElasticNet + Cox on combined data"]
    R09["09_export_for_python.R\nExport model_df.csv"]
    RSLOPE["export_red_slope.R  ★NEW\nmCherry slope (10 frames post-onset)\nfrom spots_clean.rds → red_slope.csv"]

    %% ── Exported data ─────────────────────────────────────
    CSV["📄 cache/python_export/\nmodel_df.csv  (861 cells × 40 cols)\nred_slope.csv  (622 productive cells)"]

    %% ── Python ML ─────────────────────────────────────────
    P10["10_tabpfn.py  (baseline)\nTabPFN v2, 3-fold CV"]
    P11["11_xgboost.py  ★FIXED\nXGBoost + Optuna nested CV\n3-fold outer, 50 trials/fold\nBUG FIXED: now retrains on full tr"]
    P12["12_feature_analysis.py  ★NEW\nSpearman corr matrix (29 feats)\nPer-group feature–delay bar plots\nKDE distributions by timing group"]
    P13["13_elasticnet_early_vs_rest.py  ★NEW\nElastic net, early vs medium+late\n5-fold nested CV, class_weight=balanced"]
    P14["14_general_statistics.py  ★NEW\nDescriptive stats + distribution plots\nAll 861 cells, no filter"]

    %% ── Results ───────────────────────────────────────────
    RES10["TabPFN results\nRegression R²=0.105  ρ=0.315\nBinary AUC=0.647\n3-class acc=0.481 (chance 0.485)"]
    RES11["XGBoost results (pending)\nPrevious bug: R²=0.001\nFixed: training on full fold"]
    RES12["Feature analysis\nCorrelation matrix PNG\nPer-group bar plots PNG\nDistributions PDF (5 pages)"]
    RES13["Early vs Rest\nAUC=0.684\nSens=0.597  Spec=0.710\n(77 early / 420 med+late)"]
    RES14["General statistics\nGFP→Red median 1785 min (~30h)\nProductive 61%\nTiming + dynamics PNGs"]

    %% ── Flow ──────────────────────────────────────────────
    RAW --> R01 --> R02 --> R03 --> R04 --> R07 --> R09 --> CSV
    R03 --> RSLOPE --> CSV

    CSV --> P10 --> RES10
    CSV --> P11 --> RES11
    CSV --> P12 --> RES12
    CSV --> P13 --> RES13
    CSV --> P14 --> RES14

    %% ── Styling ───────────────────────────────────────────
    classDef rscript  fill:#d4edda,stroke:#28a745,color:#000
    classDef pyscript fill:#cce5ff,stroke:#004085,color:#000
    classDef result   fill:#fff3cd,stroke:#856404,color:#000
    classDef data     fill:#f8d7da,stroke:#721c24,color:#000
    classDef new      fill:#e2d9f3,stroke:#6f42c1,color:#000

    class R01,R02,R03,R04,R07,R09 rscript
    class RSLOPE new
    class P10 pyscript
    class P11,P12,P13,P14 new
    class RES10,RES11,RES12,RES13,RES14 result
    class RAW,CSV data
```

## Legend
- 🟢 Green boxes — existing R pipeline scripts
- 🟣 Purple boxes — new scripts written this session (★NEW) or fixed (★FIXED)
- 🔵 Blue — existing Python baseline (TabPFN)
- 🟡 Yellow — results
- 🔴 Red — data files
