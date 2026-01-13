#!/usr/bin/env bash
set -euo pipefail

# make_run_scripts.sh
#
# Generates SLURM sbatch-ready run scripts, one per (n_k, memory_gb) pair.
# Each generated script calls:
#   convergence_and_expense.call_by_memory(n_k, memory_gb)
# exactly once.
#
# IMPORTANT: We escape $ in the heredoc so variables like $SCRIPT_DIR expand
# in the *generated* script (runtime), not in this generator (build-time).

# ---- CONFIG ----------------------------------------------------------------

MODULE_PY="/home/fr/fr_fr/fr_rl1038/awebox/examples/Leuthold_2025_RLL_paper_scripts/with_ineq_at_all_but_integral_and_tf_fraction/convergence_and_expense.py"
OUTDIR="$HOME/generated_runs"

# SLURM defaults (edit as needed)
PARTITION="dev_cpu"
TIME_LIMIT="00:30:00"
NTASKS=1

# Memory is set per-script from memory_gb (e.g., 4 -> 4G)

# Conda/miniforge on bwUniCluster
MINIFORGE_MODULE="devel/miniforge"
CONDA_ENV_NAME="ocp"

# Choose ONE mode:

# Mode A: grid (cross-product of NK_LIST x MEM_LIST)
NK_LIST=(10 20 40)
MEM_LIST=(4 8 16)

# Mode B: explicit pairs (uncomment to use; then comment out Mode A loop)
# PAIRS=(
#   "10 4"
#   "20 8"
#   "40 16"
# )

# ---------------------------------------------------------------------------

mkdir -p "$OUTDIR"

if [[ ! -f "$MODULE_PY" ]]; then
  echo "ERROR: MODULE_PY not found: $MODULE_PY" >&2
  exit 1
fi

gen_one() {
  local nk="$1"
  local mem="$2"

  local jobname="nk${nk}_mem${mem}G"
  local fname="${OUTDIR}/run_${jobname}.sh"

  cat > "$fname" <<EOF
#!/bin/bash
#SBATCH --partition=${PARTITION}
#SBATCH --job-name=${jobname}
#SBATCH --time=${TIME_LIMIT}
#SBATCH --ntasks=${NTASKS}
#SBATCH --mem=${mem}G
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

# Refuse to run interactively (prevents accidental login-node runs)
if [[ -z "\${SLURM_JOB_ID:-}" ]]; then
  echo "ERROR: Run this script with: sbatch $fname" >&2
  exit 1
fi

echo "starting! job=\${SLURM_JOB_NAME} id=\${SLURM_JOB_ID} host=\$(hostname)"
set -euo pipefail

# Headless-safe plotting on compute nodes
export MPLBACKEND=Agg

module purge
module load ${MINIFORGE_MODULE}

# Make "conda activate" available in this non-interactive shell:
source "\$(conda info --base)/etc/profile.d/conda.sh"
conda activate ${CONDA_ENV_NAME}

python -c "import casadi, numpy as np; print('casadi', casadi.__version__, 'numpy', np.__version__)"

# Ensure local helper modules next to convergence_and_expense.py are importable
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

# ---- GENERATE --------------------------------------------------------------

# Mode A: grid
for nk in "${NK_LIST[@]}"; do
  for mem in "${MEM_LIST[@]}"; do
    gen_one "$nk" "$mem"
  done
done

# Mode B: explicit pairs
# for p in "${PAIRS[@]}"; do
#   read -r nk mem <<<"$p"
#   gen_one "$nk" "$mem"
# done

echo
echo "Done. Scripts in: $OUTDIR"
echo "Submit one with: sbatch ${OUTDIR}/run_nk${NK_LIST[0]}_mem${MEM_LIST[0]}G.sh"
echo "Submit all with: for f in ${OUTDIR}/run_*.sh; do sbatch \"\$f\"; done"

