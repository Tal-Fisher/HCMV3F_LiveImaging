#!/bin/bash
#BSUB -J xgb_33_perm
#BSUB -q short
#BSUB -n 4
#BSUB -R "rusage[mem=8000]"
#BSUB -W 6:00
#BSUB -o /home/labs/ginossar/talfis/LiveImaging/logs/xgb_33feat_permtest_%J.out
#BSUB -e /home/labs/ginossar/talfis/LiveImaging/logs/xgb_33feat_permtest_%J.err

echo "Job started: $(date)"
echo "Host: $(hostname)"
python3 /home/labs/ginossar/talfis/LiveImaging/scripts/20d_xgboost_early_vs_rest_33feat_permtest.py
echo "Job finished: $(date)"
