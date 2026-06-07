#!/bin/bash
#BSUB -q short
#BSUB -J inf_vs_uninf_classify
#BSUB -o /home/labs/ginossar/talfis/LiveImaging/InfectedvsUninfected/logs/classify_%J.out
#BSUB -e /home/labs/ginossar/talfis/LiveImaging/InfectedvsUninfected/logs/classify_%J.err
#BSUB -n 8
#BSUB -R "rusage[mem=8192]"

PYTHON=/home/labs/ginossar/talfis/envs/cellpose_embed/bin/python
SCRIPT=/home/labs/ginossar/talfis/LiveImaging/InfectedvsUninfected/02_classify.py

echo "Job started : $(date)"
echo "Host        : $(hostname)"

$PYTHON $SCRIPT

echo "Job finished: $(date)"
