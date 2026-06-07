#!/bin/bash
#BSUB -q short
#BSUB -J cellpose_classify
#BSUB -o /home/labs/ginossar/talfis/LiveImaging/CellposeEmbedding/logs/classify_%J.out
#BSUB -e /home/labs/ginossar/talfis/LiveImaging/CellposeEmbedding/logs/classify_%J.err
#BSUB -n 8
#BSUB -R "rusage[mem=4096]"

PYTHON=/home/labs/ginossar/talfis/envs/cellpose_embed/bin/python
SCRIPT=/home/labs/ginossar/talfis/LiveImaging/CellposeEmbedding/04_classify_early_vs_rest.py

echo "Job started : $(date)"
echo "Host        : $(hostname)"

$PYTHON $SCRIPT

echo "Job finished: $(date)"
