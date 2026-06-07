#!/bin/bash
#BSUB -q short
#BSUB -J subset_analysis
#BSUB -o /home/labs/ginossar/talfis/LiveImaging/CellposeEmbedding/logs/subset_%J.out
#BSUB -e /home/labs/ginossar/talfis/LiveImaging/CellposeEmbedding/logs/subset_%J.err
#BSUB -n 8
#BSUB -R "rusage[mem=4096]"

PYTHON=/home/labs/ginossar/talfis/envs/cellpose_embed/bin/python
SCRIPT=/home/labs/ginossar/talfis/LiveImaging/CellposeEmbedding/06_subset_analysis.py

echo "Job started : $(date)"
echo "Host        : $(hostname)"
$PYTHON $SCRIPT
echo "Job finished: $(date)"
