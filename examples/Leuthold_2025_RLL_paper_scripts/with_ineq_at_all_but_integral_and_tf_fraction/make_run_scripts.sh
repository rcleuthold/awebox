#!/usr/bin/env bash
# make_run_scripts.sh
#
# Generates one sbatch script per case and SUBMITS them (default).
# Job-name uses the *target* memory (e.g. 13G), while Slurm --mem reserves
# a safety-factor-adjusted amount (e.g. ceil(13*1.30)=17G).
#
# Usage:
#   ./make_run_scripts.sh
#   ./make_run_scripts.sh --no-submit
#   ./make_run_scripts.sh --nk 15 --mem 13G --safety 1.30
#   ./make_run_scripts.sh --nk 15 --nk 25 --mem 13G --submit
#
set -euo pipefail

# ------------------------ defaults you can edit ------------------------
NK_LIST_DEFAULT=(15)
MEM_TARGET_DEFAULT="13G"      # <-- what you want to IDENTIFY jobs by
SAFETY_DEFAULT="1.30"         # <-- what you want Slurm to RESERVE
CPUS_DEFAULT="48"
TIME_DEFAULT="72:00:00"
PARTITION_DEFAULT="cpu"
CONDA_ENV_DEFAULT="ocp"

AWEBOX_ROOT_DEFAULT="/pfs/data6/home/fr/fr_fr/fr_rl1038/awebox"
PY_ENTRY_DEFAULT="/pfs/data6/home/fr/fr_fr/fr_rl1038/awebox/examples/Leuthold_2025_RLL_paper_scripts/with_ineq_at_all_but_integral_and_tf_fraction/convergence_and_expense_zero.py"

OUTDIR_DEFAULT="./run_scripts"
LOGDIR_DEFAULT="./logs"
# ----------------------------------------------------------------------

SUBMIT=1
NK_LIST=()
MEM_TARGET="$MEM_TARGET_DEFAULT"
SAFETY="$SAFETY_DEFAULT"
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
  --nk <n>              Add one NK value (repeatable). If omitted, uses default list.
  --mem <mem>           TARGET memory for naming/logic (e.g. 13G). Default: 13G
  --safety <factor>     Safety factor for Slurm reservation. Default: 1.30
  --cpus <n>            Slurm --cpus-per-task. Default: 48
  --time <hh:mm:ss>     Slurm --time. Default: 72:00:00
  --partition <p>       Slurm --partition. Default: cpu
  --conda-env <name>    Conda environment to activate. Default: ocp
  --awebox-root <path>  AWEBOX_ROOT and added to PYTHONPATH.
  --py-entry <path>     Python file that provides call_by_memory(...) (preferred).
  --outdir <dir>        Where to write sbatch scripts. Default: ./run_scripts
  --logdir <dir>        Where to write logs. Default: ./logs
  --submit              Submit each generated script with sbatch (default).
  --no-submit           Only generate scripts, do not submit.
  -h|--help             Show this help.

Notes:
  - Job name uses the TARGET mem (e.g. mem13G).
  - Slurm reserves ceil(TARGET_GB * safety) as --mem (e.g. 17G).

USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --nk)           NK_LIST+=("$2"); shift 2;;
    --mem)          MEM_TARGET="$2"; shift 2;;
    --safety)       SAFETY="$2"; shift 2;;
    --cpus)         CPUS="$2"; shift 2;;
    --time)         TIME="$2"; shift 2;;
    --partition)    PARTITION="$2"; shift 2;;
    --conda-env)    CONDA_ENV="$2"; shift 2;;
    --awebox-root)  AWEBOX_ROOT="$2"; shift 2;;
    --py-entry)     PY_ENTRY="$2"; shift 2;;
    --outdir)       OUTDIR="$2"; shift 2;;
    --logdir)       LOGDIR="$2"; shift 2;;
    --submit)       SUBMIT=1; shift;;
    --no-submit)    SUBMIT=0; shift;;
    -h|--help)      usage; exit 0;;
    *) echo "Unknown arg: $1" >&2; usage; exit 2;;
  esac
done

if [[ ${#NK_LIST[@]} -eq 0 ]]; then
  NK_LIST=("${NK_LIST_DEFAULT[@]}")
fi

mkdir -p "$OUTDIR" "$LOGDIR"

# ---- helpers ----------------------------------------------------------

die() { echo "[FATAL] $*" >&2; exit 2; }

# Parse mem like "13G" or "13000M" to integer GB (rounding UP for M)
mem_to_gb_int() {
  local m="${1^^}"  # uppercase
  if [[ "$m" =~ ^([0-9]+)(G|GB|GIB)$ ]]; then
    echo "${BASH_REMATCH[1]}"
    return 0
  fi
  if [[ "$m" =~ ^([0-9]+)(M|MB|MIB)$ ]]; then
    local mb="${BASH_REMATCH[1]}"
    # ceil(mb/1024)
    awk -v mb="$mb" 'BEGIN{ printf("%d\n", int((mb+1023)/1024)) }'
    return 0
  fi
  die "Unsupported --mem format '$1' (use e.g. 13G or 13000M)."
}

# Compute ceil(target_gb * safety) as integer GB
reserve_gb_int() {
  local target_gb="$1"
  local safety="$2"
  awk -v t="$target_gb" -v s="$safety" 'BEGIN{
    x=t*s;
    r=int(x);
    if (x>r) r=r+1;
    if (r<1) r=1;
    printf("%d\n", r);
  }'
}

# Safe module wrapper snippet for bwUniCluster3.0 non-interactive shells.
# Critical detail: modulecmd may prepend "conda deactivate;" -> we no-op conda during eval.
read -r -d '' MODULE_SNIPPET <<'BASH' || true
export PAGER=cat MODULES_PAGER=cat LMOD_PAGER=cat
module() {
  conda() { :; }
  eval "$(/opt/bwhpc/common/admin/modules/module-wrapper/modulecmd bash "$@")"
  unset -f conda >/dev/null 2>&1 || true
}
BASH

# ----------------------------------------------------------------------

MEM_TARGET_GB="$(mem_to_gb_int "$MEM_TARGET")"
MEM_RESERVE_GB="$(reserve_gb_int "$MEM_TARGET_GB" "$SAFETY")"
MEM_RESERVE="${MEM_RESERVE_GB}G"

echo "[GEN] Writing sbatch scripts to: $OUTDIR"
echo "[GEN] Logs will go to:         $LOGDIR"
echo "[GEN] conda env:              $CONDA_ENV"
echo "[GEN] entrypoint:             $PY_ENTRY"
echo "[GEN] nk list:                ${NK_LIST[*]}"
echo "[GEN] mem target:             ${MEM_TARGET}  (=${MEM_TARGET_GB}G)"
echo "[GEN] safety factor:           ${SAFETY}"
echo "[GEN] mem reserved (Slurm):    ${MEM_RESERVE}"
echo "[GEN] submit:                 ${SUBMIT}"
echo

for NK in "${NK_LIST[@]}"; do
  # Job name should reflect TARGET memory (what you asked for)
  JOBNAME="nk${NK}_mem${MEM_TARGET}"
  SBATCH_FILE="${OUTDIR}/${JOBNAME}.sbatch"

  cat > "$SBATCH_FILE" <<EOF
#!/bin/bash -l
#SBATCH --job-name=${JOBNAME}
#SBATCH --partition=${PARTITION}
#SBATCH --time=${TIME}
#SBATCH --cpus-per-task=${CPUS}
#SBATCH --mem=${MEM_RESERVE}
#SBATCH --output=${LOGDIR}/${JOBNAME}.out
#SBATCH --error=${LOGDIR}/${JOBNAME}.err

set -euo pipefail

echo "[START] job=${JOBNAME} id=\${SLURM_JOB_ID:-NA} host=\$(hostname)"
echo "[START] timestamp: \$(date)"
echo "[START] nk=${NK}"
echo "[START] mem_target=${MEM_TARGET} (=${MEM_TARGET_GB}G)"
echo "[START] mem_reserved=${MEM_RESERVE} (safety=${SAFETY})"
echo "[START] cpus-per-task=${CPUS}"
echo

${MODULE_SNIPPET}

echo "[STEP] module load devel/miniforge"
module purge
module load devel/miniforge

# Miniforge module sets CONDA_EXE; derive base from it
CONDA_BASE="\${CONDA_EXE%/bin/conda}"
if [[ ! -f "\${CONDA_BASE}/etc/profile.d/conda.sh" ]]; then
  echo "[FATAL] conda.sh not found at: \${CONDA_BASE}/etc/profile.d/conda.sh" >&2
  echo "[FATAL] CONDA_EXE=\${CONDA_EXE}" >&2
  exit 2
fi

echo "[STEP] activate conda env: ${CONDA_ENV}"
source "\${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

# Threading sanity
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

echo "[STEP] run call_by_memory(nk=${NK}, mem_target_gb=${MEM_TARGET_GB}) if available"
python - <<'PY'
import os, sys, importlib.util

py_entry = os.environ.get("PY_ENTRY_FILE") or "${PY_ENTRY}"
nk = int(os.environ.get("NK_VAL") or "${NK}")
mem_gb = int(os.environ.get("MEM_TARGET_GB") or "${MEM_TARGET_GB}")

spec = importlib.util.spec_from_file_location("entrymod", py_entry)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)  # type: ignore

if hasattr(mod, "call_by_memory"):
    print(f"[PY] calling call_by_memory({nk=}, {mem_gb=})")
    mod.call_by_memory(nk=nk, mem_gb=mem_gb)  # try kwargs first
else:
    print("[PY] No call_by_memory found; executing file as script instead.")
    import runpy
    runpy.run_path(py_entry, run_name="__main__")
PY

echo
echo "[END] timestamp: \$(date)"
EOF

  chmod +x "$SBATCH_FILE"
  echo "[GEN] wrote: $SBATCH_FILE"

  if [[ "$SUBMIT" -eq 1 ]]; then
    echo "[SUBMIT] sbatch $SBATCH_FILE"
    sbatch "$SBATCH_FILE"
  fi
done

echo
echo "[GEN] done."

