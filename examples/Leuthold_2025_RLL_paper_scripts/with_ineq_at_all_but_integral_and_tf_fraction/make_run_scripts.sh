#!/usr/bin/env bash
set -euo pipefail

# make_run_scripts.sh
# Generator script (currently 1 test case: nk=15, mem=13)

# ---------------- CONFIG ----------------
MODULE_PY="/home/fr/fr_fr/fr_rl1038/awebox/examples/Leuthold_2025_RLL_paper_scripts/with_ineq_at_all_but_integral_and_tf_fraction/convergence_and_expense.py"
OUTDIR="$HOME/generated_runs"

PARTITION="cpu"
TIME_LIMIT="72:00:00"
NODES=1
NTASKS=1
CPUS_PER_TASK=48
THREADS_PER_CORE=1

MINIFORGE_MODULE="devel/miniforge"
CONDA_ENV_NAME="ocp"

# >>> EXACT TEST VALUES (as requested) <<<
NK_LIST=(15)
MEM_LIST=(13)

SUBMIT="${SUBMIT:-1}"
# --------------------------------------

mkdir -p "$OUTDIR"

if [[ ! -f "$MODULE_PY" ]]; then
  echo "ERROR: MODULE_PY not found: $MODULE_PY" >&2
  exit 1
fi

gen_one() {
  local nk="$1"
  local mem="$2"

  # Request 30% extra memory, rounded up
  local mem_req=$(( (13*mem + 9) / 10 ))

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

if [[ -z "\${SLURM_JOB_ID:-}" ]]; then
  echo "ERROR: Run with: sbatch $script" >&2
  exit 1
fi

echo "starting! job=\${SLURM_JOB_NAME} id=\${SLURM_JOB_ID} host=\$(hostname)"
echo "planned memory (GB) = ${mem}, requested memory (GB) = ${mem_req}"
echo "cpus-per-task = \${SLURM_CPUS_PER_TASK}"
echo "timestamp: \$(date)"

export MPLBACKEND=Agg

# ---- Threading (same logic as your working job) ----
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

python -c "import casadi, numpy as np; print('casadi', casadi.__version__, 'numpy', np.__version__)"

# Make local helper modules importable
SCRIPT_DIR="\$(dirname "${MODULE_PY}")"
export PYTHONPATH="\${SCRIPT_DIR}:\$(dirname "\${SCRIPT_DIR}"):\${PYTHONPATH:-}"

echo "timestamp before main python: \$(date)"
srun --cpu-bind=cores python -u - <<'PY'
import importlib.util

module_path = r"""${MODULE_PY}"""
spec = importlib.util.spec_from_file_location("convergence_and_expense", module_path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

mod.call_by_memory(${nk}, ${mem})
PY
echo "timestamp after main python: \$(date)"

echo "done! job=\${SLURM_JOB_NAME} id=\${SLURM_JOB_ID}"
EOF

  chmod +x "$script"
  echo "Wrote: $script"

  if [[ "$SUBMIT" == "1" ]]; then
    sbatch "$script"
  fi
}

echo "Generating job scripts into: $OUTDIR"
echo "NK_LIST=${NK_LIST[*]}"
echo "MEM_LIST=${MEM_LIST[*]}"
echo "SUBMIT=$SUBMIT"

for nk in "${NK_LIST[@]}"; do
  for mem in "${MEM_LIST[@]}"; do
    gen_one "$nk" "$mem"
  done
done

echo "All done."

