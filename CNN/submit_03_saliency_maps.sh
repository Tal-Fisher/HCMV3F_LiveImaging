#!/bin/bash
#BSUB -q short-gpu
#BSUB -gpu num=1:gmem=4G
#BSUB -J saliency_maps
#BSUB -o /home/labs/ginossar/talfis/LiveImaging/CNN/logs/saliency_%J.out
#BSUB -e /home/labs/ginossar/talfis/LiveImaging/CNN/logs/saliency_%J.err
#BSUB -n 4
#BSUB -R "rusage[mem=16384]"

PYTHON=/home/labs/ginossar/talfis/envs/cellpose_embed/bin/python
SCRIPT=/home/labs/ginossar/talfis/LiveImaging/CNN/03_saliency_maps.py

echo "Job started : $(date)"
echo "Host        : $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "GPU info unavailable"

$PYTHON $SCRIPT

echo "Job finished: $(date)"
