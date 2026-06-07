#!/bin/bash
#BSUB -J xgboost_early
#BSUB -q short
#BSUB -n 8
#BSUB -R "rusage[mem=8000]"
#BSUB -W 4:00
#BSUB -o /home/labs/ginossar/talfis/LiveImaging/logs/xgboost_early_%J.out
#BSUB -e /home/labs/ginossar/talfis/LiveImaging/logs/xgboost_early_%J.err

cd /home/labs/ginossar/talfis/LiveImaging
source /apps/easybd/programs/miniconda/26.1.1_environmentally/etc/profile.d/conda.sh 2>/dev/null || true

echo "Job started: $(date)"
python3 scripts/20_xgboost_early_vs_rest.py
echo "Job finished: $(date)"
