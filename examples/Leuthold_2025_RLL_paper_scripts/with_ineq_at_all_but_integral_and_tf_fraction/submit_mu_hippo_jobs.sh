#!/usr/bin/env bash
set -euo pipefail

# Directory where THIS script lives
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

SBATCH_SCRIPT="${HERE}/run_one_mu_hippo.sbatch"

MU_VALUES=(1e-1 1e-2 1e-3 1e-4 1e-5 1e-6 1e-7)

for mu in "${MU_VALUES[@]}"; do
  sbatch "${SBATCH_SCRIPT}" "$mu"
done

