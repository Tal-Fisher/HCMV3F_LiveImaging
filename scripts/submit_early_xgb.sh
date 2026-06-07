#!/bin/bash
#BSUB -J early_xgb
#BSUB -q short
#BSUB -n 4
#BSUB -R "rusage[mem=8000]"
#BSUB -W 4:00
#BSUB -o /home/labs/ginossar/talfis/LiveImaging/logs/early_xgb_%J.out
#BSUB -e /home/labs/ginossar/talfis/LiveImaging/logs/early_xgb_%J.err

PYTHON=python3
SCRIPT=/home/labs/ginossar/talfis/LiveImaging/scripts/20_xgboost_early_vs_rest.py

echo "Job started: $(date)"
echo "Host: $(hostname)"

$PYTHON $SCRIPT

echo "Job finished: $(date)"
