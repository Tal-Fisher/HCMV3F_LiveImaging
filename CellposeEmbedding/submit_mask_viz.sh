#!/bin/bash
#BSUB -J mask_viz
#BSUB -q short-gpu
#BSUB -n 2
#BSUB -R "rusage[mem=20000]"
#BSUB -gpu "num=1:gmodel=NVIDIAA40:j_exclusive=yes"
#BSUB -o /home/labs/ginossar/talfis/LiveImaging/CellposeEmbedding/logs/mask_viz_%J.out
#BSUB -e /home/labs/ginossar/talfis/LiveImaging/CellposeEmbedding/logs/mask_viz_%J.err

source /home/labs/ginossar/talfis/envs/cellpose_embed/bin/activate

python /home/labs/ginossar/talfis/LiveImaging/CellposeEmbedding/14_mask_overlay_viz.py
