#!/bin/bash
#BSUB -J regr_33feat
#BSUB -q short
#BSUB -n 4
#BSUB -R "rusage[mem=8000]"
#BSUB -W 4:00
#BSUB -o /home/labs/ginossar/talfis/LiveImaging/logs/regression_33feat_%J.out
#BSUB -e /home/labs/ginossar/talfis/LiveImaging/logs/regression_33feat_%J.err

echo "Job started: $(date)"
echo "Host: $(hostname)"
python3 /home/labs/ginossar/talfis/LiveImaging/scripts/25d_regression_33feat.py
echo "Job finished: $(date)"
