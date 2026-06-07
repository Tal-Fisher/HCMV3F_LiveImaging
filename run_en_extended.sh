#!/bin/bash
source /apps/RHEL9/modules/lmod/lmod/init/bash 2>/dev/null || \
source /etc/profile.d/lmod.sh 2>/dev/null || true

module load R/4.4.2-gfbf-2024a 2>/dev/null || true

# fall back to full path if module didn't add Rscript to PATH
RSCRIPT=$(which Rscript 2>/dev/null || \
          echo "/apps/easybd/easybuild/amd/software/R/4.4.2-gfbf-2024a/bin/Rscript")

cd /home/labs/ginossar/talfis/LiveImaging
"$RSCRIPT" scripts/16_elasticnet_extended_features.R
