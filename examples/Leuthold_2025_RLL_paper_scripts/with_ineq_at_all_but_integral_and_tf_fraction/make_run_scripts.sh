#!/usr/bin/env bash
set -euo pipefail

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
NK_LIST=(15)
MEM_LIST=(13)

# SUBMIT=1 submits jobs; SUBMIT=0 only writes scripts
SUBMIT="${SUBMIT:-1}"

# -------------------- CHECKS -------------------
[[ -d "$AWEBOX_ROOT" ]] || { echo "ERROR: AWEBOX_ROOT not found: $AWEBOX_ROOT" >&2; exit 1; }
[[ -f "$MODULE_PY" ]]   || { echo "ERROR: MODULE_PY not found: $MODULE_PY" >&2; exit 1; }

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

# Trace to stderr (helps if anything blocks)
export PS4='+ \$(date "+%F %T") \${BASH_SOURCE}:\${LINENO}: '
set -x

export MPLBACKEND=Agg

# ---- Threading: allow linear algebra to use allocated CPUs ----
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

# ---- Locate Miniforge/conda WITHOUT module load (module load hangs in batch) ----
echo "[STEP] locating conda.sh"
CANDIDATES=(
  "/opt/bwhpc/common/devel/miniforge"
  "/opt/bwhpc/common/devel/miniforge3"
  "/opt/bwhpc/common/devel/Miniforge3"
  "/opt/bwhpc/common/devel"
)

CONDA_SH=""
for base in "\${CANDIDATES[@]}"; do
  # direct candidate
  if [[ -f "\${base}/etc/profile.d/conda.sh" ]]; then
    CONDA_SH="\${base}/etc/profile.d/conda.sh"
    break
  fi
  # versioned subdirs
  if [[ -d "\${base}" ]]; then
    found=\$(find "\${base}" -maxdepth 3 -type f -path "*/etc/profile.d/conda.sh" 2>/dev/null | head -n 1 || true)
    if [[ -n "\${found}" ]]; then
      CONDA_SH="\${found}"
      break
    fi
  fi
done

if [[ -z "\${CONDA_SH}" ]]; then
  echo "ERROR: Could not find conda.sh under /opt/bwhpc/common/devel (module load devel/miniforge hangs in batch)." >&2
  echo "Tried candidates: \${CANDIDATES[*]}" >&2
  exit 2
fi

echo "[STEP] using conda.sh: \${CONDA_SH}"
source "\${CONDA_SH}"

echo "[STEP] conda base: \$(conda info --base)"
conda activate ${CONDA_ENV_NAME}
echo "[STEP] conda active; python=\$(which python)"

# Make awebox importable
export PYTHONPATH="${AWEBOX_ROOT}:\${PYTHONPATH:-}"

python -u -c "import awebox, casadi, numpy as np; print('[ENV] awebox ok;', 'casadi', casadi.__version__, 'numpy', np.__version__, flush=True)"

echo "[RUN] timestamp before call_by_memory: \$(date)"

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

echo "All done."

