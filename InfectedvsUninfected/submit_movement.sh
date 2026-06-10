#!/bin/bash
#BSUB -J movement_features
#BSUB -q short
#BSUB -n 4
#BSUB -R "rusage[mem=16000]"
#BSUB -o logs/movement_%J.out
#BSUB -e logs/movement_%J.err

source ~/.bashrc
conda activate cellpose

cd /home/labs/ginossar/talfis/LiveImaging/InfectedvsUninfected
python3 03_movement_features.py
