#!/usr/bin/env bash
set -euo pipefail

# make_run_scripts_fixed.sh
#
# Generates sbatch scripts that run:
#   convergence_and_expense.call_by_memory(nk, mem)
# using 48 CPUs effectively (incl. BLAS/LAPACK threading),
# and submits them by default (SUBMIT=1).
#
# Based on your working "safe" generator :contentReference[oaicite:4]{index=4}
# and avoids the module/conda-activate hangs by using `conda run`,
# but adds robust Miniforge discovery + correct threading for linear algebra.

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

# Headless plotting
export MPLBACKEND=Agg

# ---------- CPU binding / threading ----------
# OpenMP threads (bwUniCluster docs: OMP_NUM_THREADS defaults to 1 unless set) :contentReference[oaicite:5]{index=5}
export OMP_NUM_THREADS="\${SLURM_CPUS_PER_TASK}"
export OMP_PLACES=cores
export OMP_PROC_BIND=spread

# Make BLAS/LAPACK use the allocated CPUs (this is what your "safe" script prevented)
# (NumPy/SciPy/OpenBLAS/MKL/BLIS respect these) :contentReference[oaicite:6]{index=6}
export OPENBLAS_NUM_THREADS="\${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="\${SLURM_CPUS_PER_TASK}"
export BLIS_NUM_THREADS="\${SLURM_CPUS_PER_TASK}"
export NUMEXPR_NUM_THREADS="\${SLURM_CPUS_PER_TASK}"

# Keep libraries from changing thread counts dynamically
export MKL_DYNAMIC=FALSE
export OMP_DYNAMIC=FALSE

# Helps avoid glibc malloc arena blow-up with many threads
export MALLOC_ARENA_MAX=2

echo "timestamp before locating conda: \$(date)"

# ---------- Conda / Miniforge ----------
# Preferred: detect newest Miniforge installation on bwUniCluster by absolute path
MINIFORGE_ROOT="\$(ls -d /opt/bwhpc/common/devel/miniforge/* 2>/dev/null | sort -V | tail -1 || true)"

# Fallback: use the documented module on bwUniCluster if path detection fails :contentReference[oaicite:7]{index=7}
if [[ -z "\${MINIFORGE_ROOT}" || ! -x "\${MINIFORGE_ROOT}/bin/conda" ]]; then
  echo "Miniforge not found under /opt/bwhpc/common/devel/miniforge/*, trying: module load devel/miniforge"
  if command -v module >/dev/null 2>&1; then
    module purge
    module load devel/miniforge
  fi
else
  export PATH="\${MINIFORGE_ROOT}/bin:\${PATH}"
fi

if ! command -v conda >/dev/null 2>&1; then
  echo "ERROR: conda not found after Miniforge setup." >&2
  echo "Tried MINIFORGE_ROOT=\${MINIFORGE_ROOT}" >&2
  exit 2
fi

echo "conda: \$(command -v conda)"
echo "timestamp after locating conda: \$(date)"

# Run python inside env WITHOUT conda activate (no shell hook / no interactive activation)
PYTHON=(conda run -n ${CONDA_ENV_NAME} python)

# Ensure local scripts are importable
export PYTHONPATH="${BASE_DIR}:\${PYTHONPATH:-}"

echo "timestamp before python smoke test: \$(date)"
srun --cpu-bind=cores "\${PYTHON[@]}" -u -c "import sys; import numpy as np; print('python:', sys.executable); print('numpy:', np.__version__)"
echo "timestamp after python smoke test: \$(date)"

echo "timestamp before main python: \$(date)"
srun --cpu-bind=cores "\${PYTHON[@]}" -u - <<'PY'
import importlib.util

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

