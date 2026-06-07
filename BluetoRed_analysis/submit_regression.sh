#!/bin/bash
#BSUB -q short
#BSUB -J b2r_regression
#BSUB -o /home/labs/ginossar/talfis/LiveImaging/BluetoRed_analysis/logs/regression_%J.out
#BSUB -e /home/labs/ginossar/talfis/LiveImaging/BluetoRed_analysis/logs/regression_%J.err
#BSUB -n 8
#BSUB -R "rusage[mem=4096]"

echo "Job started : $(date)"
echo "Host        : $(hostname)"
python3 /home/labs/ginossar/talfis/LiveImaging/BluetoRed_analysis/01_regression_en_xgb.py
echo "Job finished: $(date)"
