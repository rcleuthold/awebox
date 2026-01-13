#!/usr/bin/env bash
set -euo pipefail

# make_run_scripts.sh (TEST VERSION)
#
# Generates SLURM sbatch-ready run scripts, one per (n_k, memory_gb) pair.
# Submits ONLY the smallest case (min nk, min mem), then prints squeue.

# ---- CONFIG ----------------------------------------------------------------

MODULE_PY="/home/fr/fr_fr/fr_rl1038/awebox/examples/Leuthold_2025_RLL_paper_scripts/with_ineq_at_all_but_integral_and_tf_fraction/convergence_and_expense.py"
OUTDIR="$HOME/generated_runs"

# SLURM defaults (TEST)
PARTITION="cpu"
TIME_LIMIT="72:00:00"
NODES=1
NTASKS=1
CPUS_PER_TASK=48

# Conda/miniforge
MINIFORGE_MODULE="devel/miniforge"
CONDA_ENV_NAME="ocp"

# TEST VALUES
NK_LIST=(20 30)
MEM_LIST=(8 10)   # planned memory_gb

# ---------------------------------------------------------------------------

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

if [[ -z "\${SLURM_JOB_ID:-}" ]]; then
  echo "ERROR: Run with sbatch $fname" >&2
  exit 1
fi

echo "starting! job=\${SLURM_JOB_NAME} id=\${SLURM_JOB_ID} host=\$(hostname)"
echo "planned memory (GB) = ${mem}, requested memory (GB) = ${mem_req}"
set -euo pipefail

export MPLBACKEND=Agg
export OMP_NUM_THREADS="\${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="\${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="\${SLURM_CPUS_PER_TASK}"
export NUMEXPR_NUM_THREADS="\${SLURM_CPUS_PER_TASK}"

module purge
module load ${MINIFORGE_MODULE}
source "\$(conda info --base)/etc/profile.d/conda.sh"
conda activate ${CONDA_ENV_NAME}

python -c "import casadi, numpy as np; print('casadi', casadi.__version__, 'numpy', np.__version__)"

MODULE_PY="${MODULE_PY}"
SCRIPT_DIR="\$(dirname "\$MODULE_PY")"
export PYTHONPATH="\${SCRIPT_DIR}:\$(dirname "\${SCRIPT_DIR}"):\${PYTHONPATH:-}"

python - <<'PY'
import importlib.util

module_path = r"""${MODULE_PY}"""
spec = importlib.util.spec_from_file_location("convergence_and_expense", module_path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

mod.call_by_memory(${nk}, ${mem})
PY

echo "done! job=\${SLURM_JOB_NAME} id=\${SLURM_JOB_ID}"
EOF

  chmod +x "$fname"
  echo "Wrote: $fname"
}

# ---- GENERATE ALL SCRIPTS --------------------------------------------------

for nk in "${NK_LIST[@]}"; do
  for mem in "${MEM_LIST[@]}"; do
    gen_one "$nk" "$mem"
  done
done

# ---- SUBMIT ONLY THE SMALLEST CASE -----------------------------------------

SMALLEST_NK="${NK_LIST[0]}"
SMALLEST_MEM="${MEM_LIST[0]}"
SMALLEST_SCRIPT="${OUTDIR}/run_nk${SMALLEST_NK}_mem${SMALLEST_MEM}G.sh"

echo
echo "Submitting smallest test job:"
echo "  $SMALLEST_SCRIPT"
JOBID=$(sbatch "$SMALLEST_SCRIPT" | awk '{print $4}')
echo "Submitted as JobID $JOBID"

# ---- SHOW QUEUE ------------------------------------------------------------

echo
echo "Current jobs for user $USER:"
squeue -u "$USER" -o "%.18i %.20j %.8T %.10M %.9l %R"

