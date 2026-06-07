#!/bin/bash
#BSUB -J tabicl_b2r
#BSUB -q long-gpu
#BSUB -gpu num=1:gmem=8G
#BSUB -n 4
#BSUB -R "rusage[mem=16000]"
#BSUB -o /home/labs/ginossar/talfis/LiveImaging/logs/tabicl_b2r_%J.out
#BSUB -e /home/labs/ginossar/talfis/LiveImaging/logs/tabicl_b2r_%J.err

PYTHON=/home/labs/ginossar/talfis/envs/tabicl_forecast/bin/python3.12
SCRIPT=/home/labs/ginossar/talfis/LiveImaging/scripts/25b_blue_to_red_tabicl.py

echo "Job started: $(date)"
echo "Host: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'none')"

$PYTHON $SCRIPT

echo "Job finished: $(date)"
