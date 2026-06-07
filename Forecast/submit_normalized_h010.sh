#!/bin/bash
#BSUB -q long-gpu
#BSUB -gpu num=1:gmem=8G
#BSUB -J tabicl_norm10
#BSUB -o /home/labs/ginossar/talfis/LiveImaging/Forecast/normalized_h010.log
#BSUB -e /home/labs/ginossar/talfis/LiveImaging/Forecast/normalized_h010.err
#BSUB -n 4

PYTHON=/home/labs/ginossar/talfis/envs/tabicl_forecast/bin/python3.12
SCRIPT=/home/labs/ginossar/talfis/LiveImaging/Forecast/05_run_forecaster_normalized_h010.py

echo "Job started: $(date)"
echo "Host: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')"

$PYTHON $SCRIPT

echo "Job finished: $(date)"
