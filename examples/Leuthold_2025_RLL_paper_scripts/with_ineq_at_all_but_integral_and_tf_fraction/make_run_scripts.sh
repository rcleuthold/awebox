#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# make_run_scripts.sh  (bwUniCluster3.0 robust)
#
# - Generates one sbatch script per (nk, mem) pair
# - Submits each with sbatch
# - DOES NOT use "module" anywhere (avoids your eval/hang issues)
# - Uses conda-run to locate the env's python on the login node
# - Batch jobs call that python directly
# - Sets thread env vars so BLAS/OpenMP can use 48 CPUs
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

# ---- Locate env python WITHOUT module/activate ----
if ! command -v conda >/dev/null 2>&1; then
  echo "ERROR: 'conda' not found in PATH." >&2
  echo "Fix: in your login shell first run: module load devel/miniforge (or ensure conda is available), then rerun this generator." >&2
  exit 2
fi

echo "[GEN] Locating python for conda env '${CONDA_ENV_NAME}' via conda run..."
PYTHON_EXE="$(conda run -n "${CONDA_ENV_NAME}" python -c 'import sys; print(sys.executable)' 2>/dev/null || true)"

if [[ -z "${PYTHON_EXE}" ]]; then
  echo "ERROR: Could not run python in conda env '${CONDA_ENV_NAME}' via 'conda run'." >&2
  echo "Try: conda run -n ${CONDA_ENV_NAME} python -c 'import sys; print(sys.executable)'" >&2
  exit 3
fi

echo "[GEN] Using env python: ${PYTHON_EXE}"

ceil_13_over_10() {  # ceil(1.3*x)
  local mem="$1"
  echo $(( (13*mem + 9) / 10 ))
}

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

# Trace to stderr for visibility
export PS4='+ \$(date "+%F %T") \${BASH_SOURCE}:\${LINENO}: '
set -x

export MPLBACKEND=Agg
export PYTHONNOUSERSITE=1

# ---- Threading (48 CPUs) ----
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

# Make awebox importable
export PYTHONPATH="${AWEBOX_ROOT}:\${PYTHONPATH:-}"

PYTHON="${PYTHON_EXE}"

"\${PYTHON}" -u -c "import sys; print('[ENV] python=', sys.executable, flush=True)"
"\${PYTHON}" -u -c "import awebox, casadi, numpy as np; print('[ENV] awebox ok;', 'casadi', casadi.__version__, 'numpy', np.__version__, flush=True)"

echo "[RUN] timestamp before call_by_memory: \$(date)"

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
    echo "[GEN] Submitting: $script"
    sbatch "$script"
  fi
}

echo "[GEN] OUTDIR=$OUTDIR"
echo "[GEN] NK_LIST=${NK_LIST[*]}"
echo "[GEN] MEM_LIST=${MEM_LIST[*]}"
echo "[GEN] SUBMIT=$SUBMIT"

for nk in "${NK_LIST[@]}"; do
  for mem in "${MEM_LIST[@]}"; do
    gen_one "$nk" "$mem"
  done
done

echo "[GEN] Done."

