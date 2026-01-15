#!/usr/bin/env bash
set -euo pipefail

BASE_DIR="/home/fr/fr_fr/fr_rl1038/awebox/examples/Leuthold_2025_RLL_paper_scripts/with_ineq_at_all_but_integral_and_tf_fraction"
SBATCH_SCRIPT="${BASE_DIR}/run_one_mu_hippo.sbatch"

MU_VALUES=(1e-1 1e-2 1e-3 1e-4 1e-5 1e-6 1e-7)

for mu in "${MU_VALUES[@]}"; do
  sbatch "${SBATCH_SCRIPT}" "${mu}"
done

