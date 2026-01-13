#!/usr/bin/env bash
set -euo pipefail

# ---- CONFIG ---------------------------------------------------------------

MODULE_PY="/home/fr/fr_fr/fr_rl1038/awebox/examples/Leuthold_2025_RLL_paper_scripts/with_ineq_at_all_but_integral_and_tf_fraction/convergence_and_expense.py"
OUTDIR="./generated_runs"

# SLURM defaults
PARTITION="dev_cpu"
TIME_LIMIT="00:30:00"
NTASKS=1
CONDA_ENV_NAME="ocp"
MINIFORGE_MODULE="devel/miniforge"

# Choose one mode:

# Mode A: grid (cross-product)
NK_LIST=(10 20) #10 20 40)
MEM_LIST=(4 8) # 16)

# Mode B: explicit pairs (uncomment and use instead of Mode A loop)
# PAIRS=(
#   "10 4"
#   "20 8"
#   "40 16"
# )

# --------------------------------------------------------------------------

mkdir -p "$OUTDIR"

if [[ ! -f "$MODULE_PY" ]]; then
  echo "ERROR: MODULE_PY not found: $MODULE_PY" >&2
  exit 1
fi

gen_one() {
  local nk="$1"
  local mem="$2"

  # Job name and filename derived from nk/mem
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

echo "testing! job=\${SLURM_JOB_NAME} id=\${SLURM_JOB_ID} host=\$(hostname)"
set -euo pipefail

# Headless-safe plotting on compute nodes
export MPLBACKEND=Agg

module purge
module load ${MINIFORGE_MODULE}

# Make "conda activate" available in this non-interactive shell:
source "\$(conda info --base)/etc/profile.d/conda.sh"
conda activate ${CONDA_ENV_NAME}

python -c "import casadi, numpy as np; print('casadi', casadi.__version__, 'numpy', np.__version__)"

python - <<'PY'
import importlib.util

module_path = r"""${MODULE_PY}"""
spec = importlib.util.spec_from_file_location("convergence_and_expense", module_path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

# Call exactly once:
mod.call_by_memory(${nk}, ${mem})
PY
EOF

  chmod +x "$fname"
  echo "Wrote: $fname"
}

# ---- GENERATE -------------------------------------------------------------

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
echo "Done. Submit one with: sbatch ${OUTDIR}/run_nk${NK_LIST[0]}_mem${MEM_LIST[0]}G.sh"

