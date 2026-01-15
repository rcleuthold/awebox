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
echo "Memory:       24G"
echo "mu_hippo:     ${MU_HIPPO}"
echo "============="

# Headless plotting backend (keeps figures in batch)
export MPLBACKEND=Agg

# OpenMP uses allocated CPUs; BLAS stays single-threaded (prevents oversubscription)
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OMP_PROC_BIND=spread
export OMP_PLACES=cores
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# Helps avoid glibc malloc arena blow-up with many threads
export MALLOC_ARENA_MAX=2

# ---- Environment setup (adjust conda env name if needed) ----
module purge
module load devel/miniforge
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ocp

# Ensure we can import the local script and its local helpers
BASE_DIR="/home/fr/fr_fr/fr_rl1038/awebox/examples/Leuthold_2025_RLL_paper_scripts/with_ineq_at_all_but_integral_and_tf_fraction"
export PYTHONPATH="${BASE_DIR}:${PYTHONPATH:-}"

# Run exactly one trial
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

