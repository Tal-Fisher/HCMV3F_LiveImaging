#!/bin/bash
#BSUB -q short-gpu
#BSUB -gpu num=1:gmem=8G
#BSUB -J inf_vs_uninf_extract
#BSUB -o /home/labs/ginossar/talfis/LiveImaging/InfectedvsUninfected/logs/extract_%J.out
#BSUB -e /home/labs/ginossar/talfis/LiveImaging/InfectedvsUninfected/logs/extract_%J.err
#BSUB -n 4
#BSUB -R "rusage[mem=32768]"

PYTHON=/home/labs/ginossar/talfis/envs/cellpose_embed/bin/python
SCRIPT=/home/labs/ginossar/talfis/LiveImaging/InfectedvsUninfected/01_extract_embeddings.py

echo "Job started : $(date)"
echo "Host        : $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "GPU info unavailable"

$PYTHON $SCRIPT

echo "Job finished: $(date)"
