#!/bin/bash
#SBATCH --partition=cpu
#SBATCH --job-name=veri_dyn_test
#SBATCH --time=72:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=50G
#SBATCH --output=veri_dyn_test.out
#SBATCH --error=veri_dyn_test.err

echo "veri dyn test!"

set -euo pipefail

module purge
module load devel/miniforge   # bwHPC module :contentReference[oaicite:1]{index=1}

# Make "conda activate" available in this non-interactive shell:
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ocp

python -c "import casadi, numpy as np; print('casadi', casadi.__version__, 'numpy', np.__version__)"


python /pfs/data6/home/fr/fr_fr/fr_rl1038/awebox/examples/Leuthold_2025_RLL_paper_scripts/with_ineq_at_all_but_integral_and_tf_fraction/verification_dynamic_conditions_from_haas2019_test.py
