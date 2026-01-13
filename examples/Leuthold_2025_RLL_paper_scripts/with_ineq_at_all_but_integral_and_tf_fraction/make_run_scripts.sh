#!/usr/bin/env bash
set -euo pipefail

# make_run_scripts.sh
#
# Generates AND SUBMITS SLURM sbatch-ready run scripts, one per (n_k, memory_gb) pair.
# Each generated script calls:
#   convergence_and_expense.call_by_memory(n_k, memory_gb)
# exactly once.
#
# Planned memory_gb values (derived from 128GB):
#   1/4 * 128 = 32
#   0.5 * 128 = 64
#   0.6 * 128 = 76.8  -> 77 (rounded up to integer GB)
#   0.7 * 128 = 89.6  -> 90
#   0.8 * 128 = 102.4 -> 103
#
# SLURM requested memory rule:
#   #SBATCH --mem = ceil(1.3 * planned_memory_gb)G

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
NK_LIST=(30 40 20)

# B) planned memory targets (GB) based on fractions of 128GB
#    (rounded up where needed to integer GB)
MEM_LIST=(32 64 77 90 103)

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
#SBATCH --mem=${mem_req}G
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

# Prevent accidental login-node execution
if [[ -z "\${SLURM_JOB_ID:-}" ]]; then
  echo "ERROR: Run with sbatch $fname" >&2
  exit 1
fi

echo "starting! job=\${SLURM_JOB_NAME} id=\${SLURM_JOB_ID} host=\$(hostname)"
echo "planned memory (GB) = ${mem}, requested memory (GB) = ${mem_req}"
echo "cpus-per-task = ${CPUS_PER_TASK}"
set -euo pipefail

# Headless plotting
export MPLBACKEND=Agg

# Threading controls
export OMP_NUM_THREADS="\${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="\${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="\${SLURM_CPUS_PER_TASK}"
export NUMEXPR_NUM_THREADS="\${SLURM_CPUS_PER_TASK}"

module purge
module load ${MINIFORGE_MODULE}
source "\$(conda info --base)/etc/profile.d/conda.sh"
conda activate ${CONDA_ENV_NAME}

python -c "import casadi, numpy as np; print('casadi', casadi.__version__, 'numpy', np.__version__)"

# Make local helper modules importable
MODULE_PY="${MODULE_PY}"
SCRIPT_DIR="\$(dirname "\$MODULE_PY")"
export PYTHONPATH="\${SCRIPT_DIR}:\$(dirname "\${SCRIPT_DIR}"):\${PYTHONPATH:-}"

python - <<'PY'
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

# ---- GENERATE + SUBMIT ALL JOBS --------------------------------------------

echo "Generating and submitting jobs..."
submitted=0

for nk in "${NK_LIST[@]}"; do
  for mem in "${MEM_LIST[@]}"; do
    gen_one "$nk" "$mem"
    script="${OUTDIR}/run_nk${nk}_mem${mem}G.sh"
    jobid=$(sbatch "$script" | awk '{print $4}')
    echo "Submitted: nk=${nk} mem=${mem}G -> JobID ${jobid}"
    submitted=$((submitted + 1))
  done
done

echo
echo "Submitted ${submitted} jobs."
echo
echo "Current jobs for user $USER:"
squeue -u "$USER" -o "%.18i %.20j %.8T %.10M %.9l %R"

