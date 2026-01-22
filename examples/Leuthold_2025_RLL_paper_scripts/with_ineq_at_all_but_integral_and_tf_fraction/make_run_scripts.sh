#!/usr/bin/env bash
set -euo pipefail

# ---------------- USER CONFIG ----------------
AWEBOX_ROOT="/pfs/data6/home/fr/fr_fr/fr_rl1038/awebox"
MODULE_PY="${AWEBOX_ROOT}/examples/Leuthold_2025_RLL_paper_scripts/with_ineq_at_all_but_integral_and_tf_fraction/convergence_and_expense.py"

OUTDIR="$HOME/generated_runs"
mkdir -p "$OUTDIR"

PARTITION="cpu"
TIME_LIMIT="72:00:00"
CPUS_PER_TASK=48
THREADS_PER_CORE=1

MINIFORGE_MODULE="devel/miniforge"
CONDA_ENV_NAME="ocp"

# ---- Your requested test case (expand later) ----
NK_LIST=(15)
MEM_LIST=(13)

# SUBMIT=1 -> sbatch each generated script
# SUBMIT=0 -> only generate scripts
SUBMIT="${SUBMIT:-1}"
# -----------------------------------------------

[[ -d "$AWEBOX_ROOT" ]] || { echo "ERROR: AWEBOX_ROOT not found: $AWEBOX_ROOT" >&2; exit 1; }
[[ -f "$MODULE_PY" ]] || { echo "ERROR: MODULE_PY not found: $MODULE_PY" >&2; exit 1; }

gen_one () {
  local nk="$1"
  local mem="$2"

  # request 30% extra memory, rounded up: ceil(1.3*mem)
  local mem_req=$(( (13*mem + 9) / 10 ))

  local jobname="nk${nk}_mem${mem}G"
  local script="${OUTDIR}/run_${jobname}.sh"

  cat > "$script" <<EOF
#!/bin/bash
#SBATCH --partition=${PARTITION}
#SBATCH --job-name=${jobname}
#SBATCH --time=${TIME_LIMIT}
#SBATCH --nodes=1
#SBATCH --ntasks=1
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

export MPLBACKEND=Agg

# ---- Threading: allow linear algebra to use all allocated CPUs ----
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

module purge
module load ${MINIFORGE_MODULE}

source "\$(conda info --base)/etc/profile.d/conda.sh"
conda activate ${CONDA_ENV_NAME}

# Make awebox importable (this is the key fix you just validated)
export AWEBOX_ROOT="${AWEBOX_ROOT}"
export PYTHONPATH="\${AWEBOX_ROOT}:\${PYTHONPATH:-}"

# Quick sanity prints
python -u -c "import awebox, casadi, numpy as np; print('[ENV] awebox ok;', 'casadi', casadi.__version__, 'numpy', np.__version__, flush=True)"

echo "[RUN] timestamp before call_by_memory: \$(date)"

# Bind the process to the allocated cores
srun --cpu-bind=cores python -u - <<'PY'
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
  echo "Wrote: $script"

  if [[ "$SUBMIT" == "1" ]]; then
    sbatch "$script"
  fi
}

echo "OUTDIR=$OUTDIR"
echo "NK_LIST=${NK_LIST[*]}"
echo "MEM_LIST=${MEM_LIST[*]}"
echo "SUBMIT=$SUBMIT"

for nk in "${NK_LIST[@]}"; do
  for mem in "${MEM_LIST[@]}"; do
    gen_one "$nk" "$mem"
  done
done

echo "All generated."

