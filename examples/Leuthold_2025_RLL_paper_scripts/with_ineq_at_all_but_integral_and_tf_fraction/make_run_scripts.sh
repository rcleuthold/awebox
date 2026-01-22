#!/usr/bin/env bash
# make_run_scripts.sh
#
# bwUniCluster3.0-safe generator: writes sbatch scripts and optionally submits them.
# Key fixes:
#   - NEVER use: eval $(...)
#   - Always use: eval "$(...)"  (quoted)
#   - Force Lmod pagers to cat (non-interactive safety)
#   - Wrap modulecmd safely and neutralize leading "conda deactivate;" in module output
#
# Usage examples:
#   ./make_run_scripts.sh
#   ./make_run_scripts.sh --submit
#   ./make_run_scripts.sh --nk 15 --mem 17G --time 06:00:00 --cpus 48 --partition cpu --submit
#   ./make_run_scripts.sh --nk 15 --nk 25 --mem 13G --mem 17G --submit
#
set -euo pipefail

# ------------------------ defaults you can edit ------------------------
NK_LIST_DEFAULT=(15)
MEM_LIST_DEFAULT=(17G)           # you can add more, e.g. (13G 17G 25G)
CPUS_DEFAULT="48"
TIME_DEFAULT="72:00:00"
PARTITION_DEFAULT="cpu"
CONDA_ENV_DEFAULT="ocp"

AWEBOX_ROOT_DEFAULT="/pfs/data6/home/fr/fr_fr/fr_rl1038/awebox"
PY_ENTRY_DEFAULT="/pfs/data6/home/fr/fr_fr/fr_rl1038/awebox/examples/Leuthold_2025_RLL_paper_scripts/with_ineq_at_all_but_integral_and_tf_fraction/convergence_and_expense_zero.py"

OUTDIR_DEFAULT="./run_scripts"
LOGDIR_DEFAULT="./logs"
# ----------------------------------------------------------------------

SUBMIT=0
NK_LIST=()
MEM_LIST=()
CPUS="$CPUS_DEFAULT"
TIME="$TIME_DEFAULT"
PARTITION="$PARTITION_DEFAULT"
CONDA_ENV="$CONDA_ENV_DEFAULT"
AWEBOX_ROOT="$AWEBOX_ROOT_DEFAULT"
PY_ENTRY="$PY_ENTRY_DEFAULT"
OUTDIR="$OUTDIR_DEFAULT"
LOGDIR="$LOGDIR_DEFAULT"

usage() {
  cat <<'USAGE'
make_run_scripts.sh [options]

Options:
  --nk <n>            Add one NK value (repeatable). If omitted, uses default list.
  --mem <mem>         Add one mem value (repeatable). If omitted, uses default list.
                      Example: --mem 13G --mem 17G
  --cpus <n>          Slurm --cpus-per-task. Default: 48
  --time <hh:mm:ss>   Slurm --time. Default: 72:00:00
  --partition <p>     Slurm --partition. Default: cpu
  --conda-env <name>  Conda environment to activate. Default: ocp
  --awebox-root <p>   AWEBOX_ROOT and added to PYTHONPATH.
  --py-entry <p>      Python entry script to run in each job.
  --outdir <dir>      Where to write sbatch scripts. Default: ./run_scripts
  --logdir <dir>      Where to write logs. Default: ./logs
  --submit            Immediately submit each generated script with sbatch.
  -h|--help           Show this help.

Examples:
  ./make_run_scripts.sh --nk 15 --mem 17G --cpus 48 --time 06:00:00 --partition cpu --submit
  ./make_run_scripts.sh --nk 15 --nk 25 --mem 13G --mem 17G --submit
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --nk)           NK_LIST+=("$2"); shift 2;;
    --mem)          MEM_LIST+=("$2"); shift 2;;
    --cpus)         CPUS="$2"; shift 2;;
    --time)         TIME="$2"; shift 2;;
    --partition)    PARTITION="$2"; shift 2;;
    --conda-env)    CONDA_ENV="$2"; shift 2;;
    --awebox-root)  AWEBOX_ROOT="$2"; shift 2;;
    --py-entry)     PY_ENTRY="$2"; shift 2;;
    --outdir)       OUTDIR="$2"; shift 2;;
    --logdir)       LOGDIR="$2"; shift 2;;
    --submit)       SUBMIT=1; shift;;
    -h|--help)      usage; exit 0;;
    *) echo "Unknown arg: $1" >&2; usage; exit 2;;
  esac
done

if [[ ${#NK_LIST[@]} -eq 0 ]]; then
  NK_LIST=("${NK_LIST_DEFAULT[@]}")
fi
if [[ ${#MEM_LIST[@]} -eq 0 ]]; then
  MEM_LIST=("${MEM_LIST_DEFAULT[@]}")
fi

mkdir -p "$OUTDIR" "$LOGDIR"

echo "[GEN] Writing sbatch scripts to: $OUTDIR"
echo "[GEN] Logs will go to:         $LOGDIR"
echo "[GEN] conda env:              $CONDA_ENV"
echo "[GEN] entrypoint:             $PY_ENTRY"
echo "[GEN] nk list:                ${NK_LIST[*]}"
echo "[GEN] mem list:               ${MEM_LIST[*]}"
echo

# -----------------------------------------------------------------------------
# Safe module wrapper snippet to embed into each sbatch script.
# IMPORTANT: NO eval $(...) (unquoted) anywhere.
# -----------------------------------------------------------------------------
read -r -d '' MODULE_SNIPPET <<'BASH' || true
# ---- bwUniCluster3.0 safe module init (non-interactive) ----
export PAGER=cat MODULES_PAGER=cat LMOD_PAGER=cat

# Define a safe module() wrapper for batch scripts.
# - uses eval "$(...)" (quoted) to avoid breaking on function definitions "() { ... }"
# - neutralizes leading "conda deactivate;" injected by the miniforge module
module() {
  local _out
  _out="$(
    /opt/bwhpc/common/admin/modules/module-wrapper/modulecmd bash "$@" 2>&1
  )" || {
    echo "[FATAL] modulecmd failed for: module $*" >&2
    echo "$_out" >&2
    return 1
  }

  # If module output begins with "conda deactivate;" we must ensure it cannot fail
  # in a fresh batch shell (where conda might not exist yet).
  if [[ "$_out" == conda\ deactivate\;* ]]; then
    conda() { :; }
  fi

  eval "$_out"
}
# ------------------------------------------------------------
BASH

# -----------------------------------------------------------------------------
# This is the "sub-task" function you asked for.
# One call = one sbatch script (and optional sbatch submission).
# -----------------------------------------------------------------------------
call_by_memory() {
  local nk="$1"
  local mem="$2"

  local jobname="nk${nk}_mem${mem}"
  local sbatch_file="${OUTDIR}/${jobname}.sbatch"

  cat > "$sbatch_file" <<EOF
#!/bin/bash -l
#SBATCH --job-name=${jobname}
#SBATCH --partition=${PARTITION}
#SBATCH --time=${TIME}
#SBATCH --cpus-per-task=${CPUS}
#SBATCH --mem=${mem}
#SBATCH --output=${LOGDIR}/${jobname}.out
#SBATCH --error=${LOGDIR}/${jobname}.err

set -euo pipefail

echo "[START] job=${jobname} id=\${SLURM_JOB_ID:-NA} host=\$(hostname)"
echo "[START] nk=${nk} mem=${mem}"
echo "[START] cpus-per-task=${CPUS}"
echo "[START] timestamp: \$(date)"
echo

${MODULE_SNIPPET}

echo "[STEP] module load devel/miniforge"
# NOTE: module purge is optional; keep it if you want a clean environment.
# On some sites purge can be noisy; if it ever causes trouble, delete this line.
module purge
module load devel/miniforge

# Derive base from CONDA_EXE set by the module
CONDA_BASE="\${CONDA_EXE%/bin/conda}"
CONDA_SH="\${CONDA_BASE}/etc/profile.d/conda.sh"
if [[ ! -f "\$CONDA_SH" ]]; then
  echo "[FATAL] conda.sh not found at: \$CONDA_SH" >&2
  echo "[FATAL] CONDA_EXE=\${CONDA_EXE:-<unset>}" >&2
  exit 2
fi

echo "[STEP] activate conda env: ${CONDA_ENV}"
source "\$CONDA_SH"
conda activate "${CONDA_ENV}"

# Threading sanity
export MPLBACKEND=Agg
export OMP_NUM_THREADS=\${SLURM_CPUS_PER_TASK:-1}
export OMP_PLACES=cores
export OMP_PROC_BIND=spread
export OMP_DYNAMIC=FALSE
export OPENBLAS_NUM_THREADS=\${SLURM_CPUS_PER_TASK:-1}
export MKL_NUM_THREADS=\${SLURM_CPUS_PER_TASK:-1}
export BLIS_NUM_THREADS=\${SLURM_CPUS_PER_TASK:-1}
export NUMEXPR_NUM_THREADS=\${SLURM_CPUS_PER_TASK:-1}
export MKL_DYNAMIC=FALSE
export MALLOC_ARENA_MAX=2

export AWEBOX_ROOT="${AWEBOX_ROOT}"
export PYTHONPATH="\${AWEBOX_ROOT}:\${PYTHONPATH:-}"

echo "[STEP] sanity imports"
python -c "import casadi, numpy as np; import awebox; print('OK', casadi.__version__, np.__version__)"

echo "[STEP] run"
python "${PY_ENTRY}"

echo
echo "[END] timestamp: \$(date)"
EOF

  chmod +x "$sbatch_file"
  echo "[GEN] wrote: $sbatch_file"

  if [[ "$SUBMIT" -eq 1 ]]; then
    echo "[SUBMIT] sbatch $sbatch_file"
    sbatch "$sbatch_file"
  fi
}

# -----------------------------------------------------------------------------
# Generate all cases (each case calls call_by_memory)
# -----------------------------------------------------------------------------
for nk in "${NK_LIST[@]}"; do
  for mem in "${MEM_LIST[@]}"; do
    call_by_memory "$nk" "$mem"
  done
done

echo
echo "[GEN] done."

