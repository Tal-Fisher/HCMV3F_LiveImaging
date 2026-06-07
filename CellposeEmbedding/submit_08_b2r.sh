#!/bin/bash
#BSUB -J gfp_b2r
#BSUB -q short
#BSUB -n 4
#BSUB -R "rusage[mem=8000]"
#BSUB -o /home/labs/ginossar/talfis/LiveImaging/CellposeEmbedding/logs/gfp_b2r_%J.out
#BSUB -e /home/labs/ginossar/talfis/LiveImaging/CellposeEmbedding/logs/gfp_b2r_%J.err

source /home/labs/ginossar/talfis/envs/cellpose_embed/bin/activate

python /home/labs/ginossar/talfis/LiveImaging/CellposeEmbedding/08_b2r_analysis.py
