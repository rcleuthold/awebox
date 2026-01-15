#!/usr/bin/env bash
set -euo pipefail

MU_VALUES=(1e-1 1e-2 1e-3 1e-4 1e-5 1e-6 1e-7)

for mu in "${MU_VALUES[@]}"; do
  sbatch run_one_mu_hippo.sbatch "$mu"
done

