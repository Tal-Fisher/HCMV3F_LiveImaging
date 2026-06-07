#!/bin/bash
#BSUB -q short-gpu
#BSUB -gpu num=1:gmem=8G
#BSUB -J gfp_bf_fusion
#BSUB -o /home/labs/ginossar/talfis/LiveImaging/GFP_BF_Fusion/logs/fusion_%J.out
#BSUB -e /home/labs/ginossar/talfis/LiveImaging/GFP_BF_Fusion/logs/fusion_%J.err
#BSUB -n 8
#BSUB -R "rusage[mem=4096]"

PYTHON=/home/labs/ginossar/talfis/envs/cellpose_embed/bin/python
SCRIPT=/home/labs/ginossar/talfis/LiveImaging/GFP_BF_Fusion/02_gfp_bf_fusion_network.py

echo "Job started : $(date)"
echo "Host        : $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "GPU info unavailable"

$PYTHON $SCRIPT

echo "Job finished: $(date)"
