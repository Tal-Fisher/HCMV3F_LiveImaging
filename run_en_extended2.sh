#!/bin/bash
source /apps/RHEL9/modules/lmod/lmod/init/bash 2>/dev/null || \
source /etc/profile.d/lmod.sh 2>/dev/null || true

module load R/4.4.2-gfbf-2024a 2>/dev/null || true

# explicitly set library paths for R 4.4.2-gfbf-2024a dependencies
# (needed on nodes where Lmod module load fails to set LD_LIBRARY_PATH)
R_BASE=/apps/easybd/easybuild/amd/software
export LD_LIBRARY_PATH=\
${R_BASE}/R/4.4.2-gfbf-2024a/lib/R/lib:\
${R_BASE}/libdeflate/1.19-GCCcore-13.2.0/lib64:\
${R_BASE}/libiconv/1.17-GCCcore-13.3.0/lib:\
${R_BASE}/GCCcore/13.3.0/lib64:\
${R_BASE}/bzip2/1.0.8-GCCcore-13.3.0/lib:\
${R_BASE}/XZ/5.4.5-GCCcore-13.3.0/lib:\
${R_BASE}/PCRE2/10.43-GCCcore-13.3.0/lib:\
${R_BASE}/zlib/1.3.1-GCCcore-13.3.0/lib:\
${LD_LIBRARY_PATH}

RSCRIPT=$(which Rscript 2>/dev/null || \
          echo "/apps/easybd/easybuild/amd/software/R/4.4.2-gfbf-2024a/bin/Rscript")

cd /home/labs/ginossar/talfis/LiveImaging
"$RSCRIPT" scripts/18_elasticnet_extended2.R
