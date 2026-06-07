#!/bin/bash
#BSUB -q short
#BSUB -J b2r_tabicl_reg
#BSUB -o /home/labs/ginossar/talfis/LiveImaging/BluetoRed_analysis/logs/tabicl_reg_%J.out
#BSUB -e /home/labs/ginossar/talfis/LiveImaging/BluetoRed_analysis/logs/tabicl_reg_%J.err
#BSUB -n 8
#BSUB -R "rusage[mem=4096]"

PYTHON=/home/labs/ginossar/talfis/envs/tabicl_forecast/bin/python3.12

echo "Job started : $(date)"
echo "Host        : $(hostname)"
$PYTHON /home/labs/ginossar/talfis/LiveImaging/BluetoRed_analysis/02_regression_tabicl.py
echo "Job finished: $(date)"
