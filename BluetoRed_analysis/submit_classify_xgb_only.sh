#!/bin/bash
#BSUB -q short
#BSUB -J b2r_classify_xgb
#BSUB -o /home/labs/ginossar/talfis/LiveImaging/BluetoRed_analysis/logs/classify_xgb_%J.out
#BSUB -e /home/labs/ginossar/talfis/LiveImaging/BluetoRed_analysis/logs/classify_xgb_%J.err
#BSUB -n 8
#BSUB -R "rusage[mem=4096]"

echo "Job started : $(date)"
echo "Host        : $(hostname)"
python3 /home/labs/ginossar/talfis/LiveImaging/BluetoRed_analysis/03b_classify_xgb_only.py
echo "Job finished: $(date)"
