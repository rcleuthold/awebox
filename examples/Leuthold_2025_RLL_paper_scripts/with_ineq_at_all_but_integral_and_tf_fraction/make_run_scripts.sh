#!/usr/bin/env bash
set -euo pipefail

# make_run_scripts.sh (BwUniCluster3)
# Generates and optionally submits SLURM run scripts.
# Fixes:
# - Avoid slow/hanging `conda activate` by using `conda run`
# - Correct threading for 48 CPUs with HSL/Ipopt (OpenMP) + no BLAS oversubscription
# - srun CPU binding
# - unbuffered python output

# ---------------- CONFIG ----------------
BASE_DIR="/home/fr/fr_fr/fr_rl1038/awebox/examples/Leuthold_2025_RLL_paper_scripts/with_ineq_at_all_but_integral_and_tf_fraction"
MODULE_PY="${BASE_DIR}/convergence_and_expense.py"

OUTDIR="$HOME/generated_runs"
mkdir -p "$OUTDIR"

PARTITION="cpu"
TIME_LIMIT="72:00:00"
CPUS_PER_TASK=48
NTASKS=1
NODES=1

# Your conda env (do NOT activate; we will use conda run)
MINIFORGE_MODULE="devel/miniforge"
CONDA_ENV_NAME="ocp"

# Experiment grid (edit as needed)
NK_LIST=(15)
MEM_LIST=(13)

# SUBMIT=1 to submit, SUBMIT=0 to only generate
SUBMIT="${SUBMIT:-1}"

# --------------- FUNCTIONS --------------
gen_one() {
  local nk="$1"
  local mem="$2"

  # request 30% extra, ceil(1.3*mem)
  local mem_req=$(( (13*mem + 9) / 10 ))

  local jobname="nk${nk}_mem${mem}G"
  local script="${OUTDIR}/run_${jobname}.sbatch"

  cat > "$script" <<EOF
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

set -euo pipefail

echo "starting! job=\${SLURM_JOB_NAME} id=\${SLURM_JOB_ID} host=\$(hostname)"
echo "planned memory (GB) = ${mem}, requested memory (GB) = ${mem_req}"
echo "cpus-per-task = \${SLURM_CPUS_PER_TASK}"
echo "timestamp: \$(date)"

# Headless plotting
export MPLBACKEND=Agg

# ---- Threading (CRITICAL for 48 CPU efficiency) ----
export OMP_NUM_THREADS="\${SLURM_CPUS_PER_TASK}"
export OMP_PLACES=cores
export OMP_PROC_BIND=spread

# Prevent OpenMP×BLAS oversubscription
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

export MALLOC_ARENA_MAX=2

echo "timestamp before module load: \$(date)"
module purge
module load ${MINIFORGE_MODULE}
echo "timestamp after module load: \$(date)"

# IMPORTANT: avoid 'conda activate' (can hang). Use conda run instead.
PYTHON="conda run -n ${CONDA_ENV_NAME} python"

# Ensure local modules import cleanly
export PYTHONPATH="${BASE_DIR}:\${PYTHONPATH:-}"

echo "timestamp before python: \$(date)"
srun --cpu-bind=cores \$PYTHON -u - <<'PY'
import importlib.util
import time

print("Entered python, starting dynamic import...", flush=True)

module_path = r"""${MODULE_PY}"""
spec = importlib.util.spec_from_file_location("convergence_and_expense", module_path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

print("Imported convergence_and_expense. Calling call_by_memory...", flush=True)
mod.call_by_memory(${nk}, ${mem})
print("Finished call_by_memory.", flush=True)
PY
echo "timestamp after python: \$(date)"
echo "done!"
EOF

  chmod 644 "$script"
  echo "Wrote: $script"

  if [[ "$SUBMIT" == "1" ]]; then
    sbatch "$script"
  fi
}

# --------------- MAIN -------------------
if [[ ! -f "$MODULE_PY" ]]; then
  echo "ERROR: Cannot find python module: $MODULE_PY" >&2
  exit 1
fi

echo "Generating scripts into: $OUTDIR"
echo "SUBMIT=$SUBMIT"

for nk in "${NK_LIST[@]}"; do
  for mem in "${MEM_LIST[@]}"; do
    gen_one "$nk" "$mem"
  done
done

