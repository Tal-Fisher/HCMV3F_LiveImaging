#!/bin/bash
#BSUB -q short
#BSUB -J cellpose_combined
#BSUB -o /home/labs/ginossar/talfis/LiveImaging/CellposeEmbedding/logs/combined_%J.out
#BSUB -e /home/labs/ginossar/talfis/LiveImaging/CellposeEmbedding/logs/combined_%J.err
#BSUB -n 4
#BSUB -R "rusage[mem=4096]"

PYTHON=/home/labs/ginossar/talfis/envs/cellpose_embed/bin/python
SCRIPT=/home/labs/ginossar/talfis/LiveImaging/CellposeEmbedding/03_combined_features.py

echo "Job started : $(date)"
echo "Host        : $(hostname)"

$PYTHON $SCRIPT

echo "Job finished: $(date)"
