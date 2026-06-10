#!/bin/bash
#BSUB -q short-gpu
#BSUB -gpu num=1:gmem=8G
#BSUB -J gfp_embed_a3
#BSUB -o /home/labs/ginossar/talfis/LiveImaging/GFP_BF_Fusion/logs/gfp_embed_a3_%J.out
#BSUB -e /home/labs/ginossar/talfis/LiveImaging/GFP_BF_Fusion/logs/gfp_embed_a3_%J.err
#BSUB -n 4
#BSUB -R "rusage[mem=16384]"

PYTHON=/home/labs/ginossar/talfis/envs/cellpose_embed/bin/python
SCRIPT=/home/labs/ginossar/talfis/LiveImaging/CellposeEmbedding/01_extract_embeddings.py

echo "Job started : $(date)"
echo "Host        : $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "GPU info unavailable"

$PYTHON $SCRIPT --dataset A3

echo "Job finished: $(date)"
