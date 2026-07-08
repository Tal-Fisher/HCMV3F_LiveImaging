#!/bin/bash
#BSUB -q short
#BSUB -J corrected_glm
#BSUB -o /home/labs/ginossar/talfis/LiveImaging/GFP_BF_Fusion/logs/corrected_glm_%J.out
#BSUB -e /home/labs/ginossar/talfis/LiveImaging/GFP_BF_Fusion/logs/corrected_glm_%J.err
#BSUB -n 8
#BSUB -R "rusage[mem=4096]"

PYTHON=/home/labs/ginossar/talfis/envs/cellpose_embed/bin/python
SCRIPT=/home/labs/ginossar/talfis/LiveImaging/GFP_BF_Fusion/11_corrected_onset_glm.py

mkdir -p /home/labs/ginossar/talfis/LiveImaging/GFP_BF_Fusion/logs

echo "Job started : $(date)"
echo "Host        : $(hostname)"

$PYTHON $SCRIPT

echo "Job finished: $(date)"
