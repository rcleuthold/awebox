#!/usr/bin/env bash
set -euo pipefail

# make_run_scripts.sh (BwUniCluster3-safe)
#
# What this version fixes (based on what you observed):
# - DOES NOT use `module load ...` inside the job (your jobs were hanging there)
# - DOES NOT use `conda activate ...` inside the job (previously hung in conda shell hook)
# - Uses Miniforge by absolute path (auto-detect newest under /opt/bwhpc/common/devel/miniforge/*)
# - Uses `conda run -n <env> python` to run inside your env
# - Keeps 48 CPUs and sets threading correctly for Ipopt/HSL (OpenMP)
# - Prevents BLAS/NumExpr oversubscription
# - Uses `srun --cpu-bind=cores`
# - Adds timestamp checkpoints so you ALWAYS know where it is stuck

# ---------------- CONFIG ----------------
BASE_DIR="/home/fr/fr_fr/fr_rl1038/awebox/examples/Leuthold_2025_RLL_paper_scripts/with_ineq_at_all_but_integral_and_tf_fraction"
MODULE_PY="${BASE_DIR}/convergence_and_expense.py"

OUTDIR="$HOME/generated_runs"
mkdir -p "$OUTDIR"

PARTITION="cpu"
TIME_LIMIT="72:00:00"
NODES=1
NTASKS=1
CPUS_PER_TASK=48

# Conda env name (used with: conda run -n <env> python)
CONDA_ENV_NAME="ocp"

# Grid of runs (edit as needed)
NK_LIST=(15)
MEM_LIST=(13)

# SUBMIT=1 to submit, SUBMIT=0 to only generate
SUBMIT="${SUBMIT:-1}"
# --------------------------------------

if [[ ! -f "$MODULE_PY" ]]; then
  echo "ERROR: Cannot find python module: $MODULE_PY" >&2
  exit 1
fi

gen_one() {
  local nk="$1"
  local mem="$2"

  # Request 30% extra memory, rounded up: ceil(1.3 * mem)
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

# Headless plotting (keeps your figures safe in batch)
export MPLBACKEND=Agg

# ---- Threading (CRITICAL for 48 CPU efficiency with HSL/Ipopt OpenMP) ----
export OMP_NUM_THREADS="\${SLURM_CPUS_PER_TASK}"
export OMP_PLACES=cores
export OMP_PROC_BIND=spread

# Prevent OpenMP×BLAS oversubscription (this killed your efficiency before)
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# Helps avoid glibc malloc arena blow-up with many threads
export MALLOC_ARENA_MAX=2

echo "timestamp before locating miniforge: \$(date)"

# ---- IMPORTANT: avoid module load and conda activate inside jobs ----
# Find newest Miniforge installation on BwUniCluster:
MINIFORGE_ROOT="\$(ls -d /opt/bwhpc/common/devel/miniforge/* 2>/dev/null | sort -V | tail -1 || true)"
if [[ -z "\${MINIFORGE_ROOT}" || ! -x "\${MINIFORGE_ROOT}/bin/conda" ]]; then
  echo "ERROR: Could not find usable miniforge under /opt/bwhpc/common/devel/miniforge/*" >&2
  exit 2
fi
export PATH="\${MINIFORGE_ROOT}/bin:\${PATH}"
echo "Using MINIFORGE_ROOT=\${MINIFORGE_ROOT}"
echo "timestamp after locating miniforge: \$(date)"

# Run python inside env WITHOUT conda activate (no shell hook)
PYTHON="conda run -n ${CONDA_ENV_NAME} python"

# Ensure local scripts are importable
export PYTHONPATH="${BASE_DIR}:\${PYTHONPATH:-}"

echo "timestamp before python smoke test: \$(date)"
srun --cpu-bind=cores \$PYTHON -u -c "import sys; print('python:', sys.executable); import numpy; print('numpy ok')"
echo "timestamp after python smoke test: \$(date)"

echo "timestamp before main python: \$(date)"
srun --cpu-bind=cores \$PYTHON -u - <<'PY'
import importlib.util

print("Entered python, starting dynamic import...", flush=True)

module_path = r"""${MODULE_PY}"""
spec = importlib.util.spec_from_file_location("convergence_and_expense", module_path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

print("Imported convergence_and_expense. Calling call_by_memory...", flush=True)
mod.call_by_memory(${nk}, ${mem})
print("Finished call_by_memory.", flush=True)
PY
echo "timestamp after main python: \$(date)"
echo "done!"
EOF

  chmod 644 "$script"
  echo "Wrote: $script"

  if [[ "$SUBMIT" == "1" ]]; then
    sbatch "$script"
  fi
}

echo "Generating scripts into: $OUTDIR"
echo "SUBMIT=$SUBMIT"

for nk in "${NK_LIST[@]}"; do
  for mem in "${MEM_LIST[@]}"; do
    gen_one "$nk" "$mem"
  done
done

echo "All done."

