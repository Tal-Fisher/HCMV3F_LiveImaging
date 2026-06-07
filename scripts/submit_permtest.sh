#!/bin/bash
#BSUB -q short
#BSUB -J perm_test
#BSUB -n 4
#BSUB -o /home/labs/ginossar/talfis/LiveImaging/scripts/permtest_job.log
#BSUB -e /home/labs/ginossar/talfis/LiveImaging/scripts/permtest_job.err

RSCRIPT=/apps/easybd/easybuild/amd/software/R/4.4.1-gfbf-2023b/bin/Rscript
SCRIPT=/home/labs/ginossar/talfis/LiveImaging/scripts/05b_permutation_test.R

echo "Job started: $(date)"
echo "Host: $(hostname)"
$RSCRIPT $SCRIPT
echo "Job finished: $(date)"
