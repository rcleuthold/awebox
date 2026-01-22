#!/usr/bin/env bash
set -euo pipefail

# make_run_scripts.sh
#
# Generates AND SUBMITS SLURM sbatch-ready run scripts, one per (n_k, memory_gb) pair.
# Each generated script calls:
#   convergence_and_expense.call_by_memory(n_k, memory_gb)
# exactly once.
#
# IMPORTANT CPU EFFICIENCY FIXES (for 48 CPUs):
# - Give ALL allocated CPUs to OpenMP (Ipopt/HSL MA86/MA97): OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
# - FORCE BLAS/NumExpr to 1 thread to avoid oversubscription (this was the big problem)
# - Use srun --cpu-bind=cores so threads land on the allocated cores
# - (Recommended) avoid hyperthreads: #SBATCH --threads-per-core=1

# ---- CONFIG ----------------------------------------------------------------

MODULE_PY="/home/fr/fr_fr/fr_rl1038/awebox/examples/Leuthold_2025_RLL_paper_scripts/with_ineq_at_all_but_integral_and_tf_fraction/convergence_and_expense.py"
OUTDIR="$HOME/generated_runs"

# SLURM defaults (edit as needed)
PARTITION="cpu"
TIME_LIMIT="72:00:00"
NODES=1
NTASKS=1
CPUS_PER_TASK=48

# Conda/miniforge
MINIFORGE_MODULE="devel/miniforge"
CONDA_ENV_NAME="ocp"

# A) nk options
NK_LIST=(15) # e.g. (15 20 30 40)

# B) planned memory targets (GB) based on fractions of 128GB
#    (rounded up where needed to integer GB)
MEM_LIST=(13) # e.g. (32 64 77 90 103)

# If you want "generate only" sometimes, set SUBMIT=0 when running:
#   SUBMIT=0 ./make_run_scripts.sh
SUBMIT="${SUBMIT:-1}"

# ---------------------------------------------------------------------------

mkdir -p "$OUTDIR"

if [[ ! -f "$MODULE_PY" ]]; then
  echo "ERROR: MODULE_PY not found: $MODULE_PY" >&2
  exit 1
fi

gen_one() {
  local nk="$1"
  local mem="$2"

  # Request 30% extra memory, rounded up: ceil(1.3 * mem)
  local mem_req=$(( (13*mem + 9) / 10 ))

  # Job name reflects planned memory (mem), not requested (mem_req)
  local jobname="nk${nk}_mem${mem}G"
  local fname="${OUTDIR}/run_${jobname}.sh"

  cat > "$fname" <<EOF
#!/bin/bash
#SBATCH --partition=${PARTITION}
#SBATCH --job-name=${jobname}
#SBATCH --time=${TIME_LIMIT}
#SBATCH --nodes=${NODES}
#SBATCH --ntasks=${NTASKS}
#SBATCH --cpus-per-task=${CPUS_PER_TASK}
#SBATCH --threads-per-core=1
#SBATCH --mem=${mem_req}G
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

# Prevent accidental login-node execution
if [[ -z "\${SLURM_JOB_ID:-}" ]]; then
  echo "ERROR: Run with: sbatch $fname" >&2
  exit 1
fi

set -euo pipefail

echo "starting! job=\${SLURM_JOB_NAME} id=\${SLURM_JOB_ID} host=\$(hostname)"
echo "planned memory (GB) = ${mem}, requested memory (GB) = ${mem_req}"
echo "cpus-per-task = \${SLURM_CPUS_PER_TASK}"

# Headless plotting (keep figures, avoid GUI backends)
export MPLBACKEND=Agg

# ---- Threading controls (CRITICAL FOR 48-CPU EFFICIENCY) ----
# OpenMP (Ipopt/HSL MA86/MA97) uses all allocated CPUs:
export OMP_NUM_THREADS="\${SLURM_CPUS_PER_TASK}"
export OMP_PLACES=cores
export OMP_PROC_BIND=spread

# Force BLAS/NumExpr to single-thread to avoid OpenMP×BLAS oversubscription:
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# Helps avoid glibc malloc arena bloat with many threads:
export MALLOC_ARENA_MAX=2

module purge
module load ${MINIFORGE_MODULE}
source "\$(conda info --base)/etc/profile.d/conda.sh"
conda activate ${CONDA_ENV_NAME}

# Optional: quick sanity print
srun --cpu-bind=cores python -c "import casadi, numpy as np; print('casadi', casadi.__version__, 'numpy', np.__version__)"

# Make local helper modules importable
MODULE_PY="${MODULE_PY}"
SCRIPT_DIR="\$(dirname "\$MODULE_PY")"
export PYTHONPATH="\${SCRIPT_DIR}:\$(dirname "\${SCRIPT_DIR}"):\${PYTHONPATH:-}"

# Run (use srun + core binding)
srun --cpu-bind=cores python - <<'PY'
import importlib.util

module_path = r"""${MODULE_PY}"""
spec = importlib.util.spec_from_file_location("convergence_and_expense", module_path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

# Call exactly once:
mod.call_by_memory(${nk}, ${mem})
PY

echo "done! job=\${SLURM_JOB_NAME} id=\${SLURM_JOB_ID}"
EOF

  chmod +x "$fname"
  echo "Wrote: $fname"
}

# ---- GENERATE (+ OPTIONAL SUBMIT) ------------------------------------------

echo "Generating run scripts..."
generated=0
submitted=0

for nk in "${NK_LIST[@]}"; do
  for mem in "${MEM_LIST[@]}"; do
    gen_one "$nk" "$mem"
    generated=$((generated + 1))

    script="${OUTDIR}/run_nk${nk}_mem${mem}G.sh"
    if [[ "$SUBMIT" == "1" ]]; then
      jobid=$(sbatch "$script" | awk '{print $4}')
      echo "Submitted: nk=${nk} mem=${mem}G -> JobID ${jobid}"
      submitted=$((submitted + 1))
    else
      echo "Generated (not submitted): $script"
    fi
  done
done

echo
echo "Generated ${generated} scripts in: ${OUTDIR}"
if [[ "$SUBMIT" == "1" ]]; then
  echo "Submitted ${submitted} jobs."
  echo
  echo "Current jobs for user $USER:"
  squeue -u "$USER" -o "%.18i %.20j %.8T %.10M %.9l %R"
fi

