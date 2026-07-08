#!/bin/bash
#BSUB -q short
#BSUB -J cnn_extract_patches
#BSUB -o /home/labs/ginossar/talfis/LiveImaging/CNN/logs/extract_patches_%J.out
#BSUB -e /home/labs/ginossar/talfis/LiveImaging/CNN/logs/extract_patches_%J.err
#BSUB -n 4
#BSUB -R "rusage[mem=16384]"

PYTHON=/home/labs/ginossar/talfis/envs/cellpose_embed/bin/python
SCRIPT=/home/labs/ginossar/talfis/LiveImaging/CNN/01_extract_raw_patches.py

echo "Job started : $(date)"
echo "Host        : $(hostname)"

$PYTHON $SCRIPT --dataset A2
$PYTHON $SCRIPT --dataset A3

echo "Job finished: $(date)"
