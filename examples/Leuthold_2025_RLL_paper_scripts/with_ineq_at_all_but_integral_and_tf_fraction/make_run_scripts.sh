#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# make_run_scripts.sh
#
# bwUniCluster3.0 robust approach:
# 1) On LOGIN NODE: load devel/miniforge and activate env (official way)
# 2) Discover absolute path to env python
# 3) Generate SBATCH scripts that run that python directly
#
# This avoids batch-time hangs in `module load` / `module purge`
# that you observed on compute nodes.
# ============================================================

# -------------------- PATHS --------------------
AWEBOX_ROOT="/pfs/data6/home/fr/fr_fr/fr_rl1038/awebox"
MODULE_PY="${AWEBOX_ROOT}/examples/Leuthold_2025_RLL_paper_scripts/with_ineq_at_all_but_integral_and_tf_fraction/convergence_and_expense.py"
OUTDIR="$HOME/generated_runs"
mkdir -p "$OUTDIR"

# -------------------- SLURM --------------------
PARTITION="cpu"
TIME_LIMIT="72:00:00"
NODES=1
NTASKS=1
CPUS_PER_TASK=48
THREADS_PER_CORE=1

# -------------------- CONDA --------------------
MINIFORGE_MODULE="devel/miniforge"
CONDA_ENV_NAME="ocp"

# -------------------- JOB GRID -----------------
# Your requested test case (extend later)
NK_LIST=(15)
MEM_LIST=(13)

# SUBMIT=1 submits jobs; SUBMIT=0 only writes scripts
SUBMIT="${SUBMIT:-1}"

# -------------------- CHECKS -------------------
[[ -d "$AWEBOX_ROOT" ]] || { echo "ERROR: AWEBOX_ROOT not found: $AWEBOX_ROOT" >&2; exit 1; }
[[ -f "$MODULE_PY"   ]] || { echo "ERROR: MODULE_PY not found: $MODULE_PY" >&2; exit 1; }

ceil_13_over_10() {  # ceil(1.3*x)
  local mem="$1"
  echo $(( (13*mem + 9) / 10 ))
}

echo "[GEN] Preparing Python executable from conda env '${CONDA_ENV_NAME}' using '${MINIFORGE_MODULE}'..."

# --- IMPORTANT: Do module/conda ONLY HERE on login node ---
module purge || true
module load "${MINIFORGE_MODULE}"
# Make conda activate available
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_NAME}"

PYTHON_EXE="$(python -c 'import sys; print(sys.executable)')"
if [[ -z "${PYTHON_EXE}" || ! -x "${PYTHON_EXE}" ]]; then
  echo "ERROR: Could not determine executable python for env '${CONDA_ENV_NAME}'." >&2
  echo "Got: ${PYTHON_EXE}" >&2
  exit 2
fi

echo "[GEN] Using env python: ${PYTHON_EXE}"
echo "[GEN] Using module file: ${MODULE_PY}"
echo "[GEN] Writing jobs to: ${OUTDIR}"

gen_one() {
  local nk="$1"
  local mem="$2"
  local mem_req
  mem_req="$(ceil_13_over_10 "$mem")"

  local jobname="nk${nk}_mem${mem}G"
  local script="${OUTDIR}/run_${jobname}.sh"

  cat > "$script" <<EOF
#!/bin/bash
#SBATCH --partition=${PARTITION}
#SBATCH --job-name=${jobname}
#SBATCH --time=${TIME_LIMIT}
#SBATCH --nodes=${NODES}
#SBATCH --ntasks=${NTASKS}
#SBATCH --cpus-per-task=${CPUS_PER_TASK}
#SBATCH --threads-per-core=${THREADS_PER_CORE}
#SBATCH --mem=${mem_req}G
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail

echo "[START] job=\${SLURM_JOB_NAME} id=\${SLURM_JOB_ID} host=\$(hostname)"
echo "[START] nk=${nk} mem=${mem}G (requested mem=${mem_req}G)"
echo "[START] cpus-per-task=\${SLURM_CPUS_PER_TASK}"
echo "[START] timestamp: \$(date)"

# Trace to stderr (so we can always see where we are)
export PS4='+ \$(date "+%F %T") \${BASH_SOURCE}:\${LINENO}: '
set -x

export MPLBACKEND=Agg
export PYTHONNOUSERSITE=1

# ---- Threading (use allocated CPUs) ----
export OMP_NUM_THREADS="\${SLURM_CPUS_PER_TASK}"
export OMP_PLACES=cores
export OMP_PROC_BIND=spread
export OMP_DYNAMIC=FALSE

export OPENBLAS_NUM_THREADS="\${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="\${SLURM_CPUS_PER_TASK}"
export BLIS_NUM_THREADS="\${SLURM_CPUS_PER_TASK}"
export NUMEXPR_NUM_THREADS="\${SLURM_CPUS_PER_TASK}"
export MKL_DYNAMIC=FALSE

export MALLOC_ARENA_MAX=2

# Ensure awebox import works
export PYTHONPATH="${AWEBOX_ROOT}:\${PYTHONPATH:-}"

PYTHON="${PYTHON_EXE}"

# Quick sanity
"\${PYTHON}" -u -c "import sys; print('[ENV] python=', sys.executable, flush=True)"
"\${PYTHON}" -u -c "import awebox, casadi, numpy as np; print('[ENV] awebox ok;', 'casadi', casadi.__version__, 'numpy', np.__version__, flush=True)"

echo "[RUN] timestamp before call_by_memory: \$(date)"

# Bind python to allocated cores
srun --cpu-bind=cores "\${PYTHON}" -u - <<'PY'
import importlib.util

module_path = r"""${MODULE_PY}"""
spec = importlib.util.spec_from_file_location("convergence_and_expense", module_path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

print("[PY] calling call_by_memory(${nk}, ${mem})", flush=True)
mod.call_by_memory(${nk}, ${mem})
print("[PY] done call_by_memory", flush=True)
PY

echo "[DONE] timestamp: \$(date)"
EOF

  chmod +x "$script"
  echo "[GEN] Wrote: $script"

  if [[ "$SUBMIT" == "1" ]]; then
    sbatch "$script"
  fi
}

for nk in "${NK_LIST[@]}"; do
  for mem in "${MEM_LIST[@]}"; do
    gen_one "$nk" "$mem"
  done
done

echo "[GEN] Done."

