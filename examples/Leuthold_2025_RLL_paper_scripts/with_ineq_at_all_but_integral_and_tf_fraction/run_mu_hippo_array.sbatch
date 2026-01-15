#!/bin/bash
#SBATCH --partition=cpu
#SBATCH --job-name=muHippo
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=24G
#SBATCH --time=72:00:00
#SBATCH --threads-per-core=1
#SBATCH --output=muHippo_%j.out
#SBATCH --error=muHippo_%j.err

set -euo pipefail

MU_HIPPO="${1:?ERROR: pass mu_hippo as first argument, e.g. sbatch run_one_mu_hippo.sbatch 1e-3}"

echo "==== Slurm ===="
echo "JobID:        ${SLURM_JOB_ID}"
echo "Node:         $(hostname)"
echo "CPUs/task:    ${SLURM_CPUS_PER_TASK}"
echo "Mem:          24G"
echo "mu_hippo:     ${MU_HIPPO}"
echo "============="

# --- Headless plotting (keep your figures!) ---
export MPLBACKEND=Agg

# --- Threading: OpenMP uses the reserved CPUs; BLAS stays single-threaded ---
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OMP_PROC_BIND=spread
export OMP_PLACES=cores

# Prevent oversubscription (this was a big issue before)
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# Optional: reduce malloc arena bloat (helps with many threads)
export MALLOC_ARENA_MAX=2

# --- Environment (copied from your previous generator, but corrected) ---
module purge
module load devel/miniforge
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ocp

# Ensure local helper modules (helpful_operations etc.) are importable
MODULE_PY="/home/fr/fr_fr/fr_rl1038/awebox/examples/Leuthold_2025_RLL_paper_scripts/with_ineq_at_all_but_integral_and_tf_fraction/convergence_and_expense.py"
SCRIPT_DIR="$(dirname "$MODULE_PY")"
export PYTHONPATH="${SCRIPT_DIR}:$(dirname "${SCRIPT_DIR}"):${PYTHONPATH:-}"

# --- Run exactly one trial ---
# Use srun + cpu binding so threads actually land on the allocated cores.
srun --cpu-bind=cores python - <<PY
from convergence_and_expense import call_by_pt

call_by_pt(
    n_k=20,
    pt=1.0,
    ipopt_tol=1e-8,
    mu_hippo=float("${MU_HIPPO}")
)
PY

echo "DONE job ${SLURM_JOB_ID}"

