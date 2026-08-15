#!/bin/bash
#SBATCH --job-name=icemix
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=4
#SBATCH --mem=0
#SBATCH --time=72:00:00
#SBATCH --output=slurm/logs/%x-%j.out
# IceMix Slurm launcher. Set DATA_ROOT; site-specific allocation flags stay on
# `sbatch`. Stage-2 and resume behavior are selected with the training flags.

set -euo pipefail

# sbatch runs a spooled copy, so locate the checkout from the submission path.
REPO_ROOT="${GRAPHNET_REPO_ROOT:-$PWD}"
while [ "$REPO_ROOT" != "/" ] && [ ! -f "$REPO_ROOT/setup.py" ]; do
    REPO_ROOT="$(dirname "$REPO_ROOT")"
done
if [ ! -f "$REPO_ROOT/setup.py" ]; then
    echo "Could not locate the graphnet checkout; set GRAPHNET_REPO_ROOT."
    exit 1
fi
cd "$REPO_ROOT"
mkdir -p slurm/logs
echo "repo root: $REPO_ROOT"

DATA_ROOT="${DATA_ROOT:?set DATA_ROOT to the directory holding the input databases}"
CONDA_ENV="${CONDA_ENV:-graphnet-gpu}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_ROOT/outputs}"

# Atomically claim a run directory; RUN_NAME deliberately reuses one.
if [ -n "${RUN_NAME:-}" ]; then
    mkdir -p "$OUTPUT_ROOT/$RUN_NAME"
else
    RUN_NAME="${SLURM_JOB_NAME:-icemix}-${SLURM_JOB_ID:-local}"
    mkdir -p "$OUTPUT_ROOT"
    candidate="$RUN_NAME"
    attempt=2
    until mkdir "$OUTPUT_ROOT/$candidate" 2>/dev/null; do
        candidate="$RUN_NAME-r$attempt"
        attempt=$((attempt + 1))
    done
    RUN_NAME="$candidate"
fi
echo "run directory: $OUTPUT_ROOT/$RUN_NAME"

# Shared by staging and prewarming.
DB_NEED_KB=$(du -ck "$DATA_ROOT"/nue.db "$DATA_ROOT"/numu.db \
    "$DATA_ROOT"/nutau.db 2>/dev/null | tail -1 | awk '{print $1}')

# --- Optional node-local staging ---------------------------------------------
# rsync verifies the file copy; a non-empty WAL means a file copy is unsafe.
if [ "${LOCAL_STAGE:-1}" = "1" ]; then
    STAGE_DIR="${GRAPHNET_STAGE_DIR:-${TMPDIR:-/tmp}/icemix-stage-$USER}"

    # Avoid concurrent rsyncs into the shared per-user staging directory.
    exec 9>"$STAGE_DIR.lock"
    flock -x 9

    stage_ok=1

    for flavour in nue numu nutau; do
        wal="$DATA_ROOT/$flavour.db-wal"
        if [ -s "$wal" ]; then
            echo "[WARN] $wal is non-empty ($(stat -c%s "$wal") bytes)."
            echo "       A file copy would drop un-checkpointed frames."
            stage_ok=0
        fi
    done

    mkdir -p "$STAGE_DIR" 2>/dev/null || stage_ok=0
    if [ "$stage_ok" = "1" ]; then
        avail_kb=$(df -Pk "$STAGE_DIR" | awk 'NR==2 {print $4}')
        if [ "$avail_kb" -lt $((DB_NEED_KB + 5242880)) ]; then
            echo "[WARN] $STAGE_DIR has $((avail_kb / 1048576)) GiB free, need"
            echo "       $((DB_NEED_KB / 1048576)) GiB + headroom. Not staging."
            stage_ok=0
        fi
    fi

    if [ "$stage_ok" = "1" ]; then
        echo "staging $((DB_NEED_KB / 1048576)) GiB -> $STAGE_DIR"
        for flavour in nue numu nutau; do
            rsync -ah --info=progress2 \
                "$DATA_ROOT/$flavour.db" "$STAGE_DIR/$flavour.db" || stage_ok=0
            src_bytes=$(stat -c%s "$DATA_ROOT/$flavour.db")
            dst_bytes=$(stat -c%s "$STAGE_DIR/$flavour.db" 2>/dev/null || echo 0)
            if [ "$src_bytes" != "$dst_bytes" ]; then
                echo "[WARN] $flavour.db staged $dst_bytes of $src_bytes bytes"
                stage_ok=0
            fi
        done
    fi

    if [ "$stage_ok" = "1" ]; then
        DATA_ROOT="$STAGE_DIR"
        echo "staged. DATA_ROOT is now $DATA_ROOT"
    else
        echo "[WARN] staging skipped or incomplete; reading from $DATA_ROOT."
        echo "       Correct, just slower."
    fi
    exec 9>&-
fi
export DATA_ROOT

# --- Optional page-cache warmup ----------------------------------------------
if [ "${PREWARM:-1}" = "1" ]; then
    avail_kb=$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)

    # Leave half of the allocation for training processes.
    if [ -n "${SLURM_MEM_PER_NODE:-}" ]; then
        budget_kb=$(( SLURM_MEM_PER_NODE * 1024 / 2 ))
        if [ "$budget_kb" -lt "$avail_kb" ]; then
            avail_kb="$budget_kb"
            echo "pre-warm budget capped at half of --mem"\
"($((SLURM_MEM_PER_NODE / 1024)) GiB)"
        fi
    fi

    if [ -z "$avail_kb" ] || [ "$avail_kb" -lt "$DB_NEED_KB" ]; then
        echo "[WARN] pre-warm skipped: need $((DB_NEED_KB / 1048576)) GiB of"
        echo "       page cache, budget is $((${avail_kb:-0} / 1048576)) GiB."
        echo "       Reading cold is correct, just slower."
    else
        echo "pre-warming $((DB_NEED_KB / 1048576)) GiB into the page cache"
        prewarm_start=$SECONDS
        for flavour in nue numu nutau; do
            dd if="$DATA_ROOT/$flavour.db" of=/dev/null bs=16M \
                status=none 2>/dev/null &
        done
        wait
        echo "pre-warmed in $((SECONDS - prewarm_start)) s"
    fi
fi

# --- Environment --------------------------------------------------------------
module load anaconda3/2023.03 2>/dev/null || true
# shellcheck disable=SC1091
conda activate "$CONDA_ENV"

# Avoid packages leaking in from ~/.local.
export PYTHONNOUSERSITE=1

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export TORCH_CUDNN_V8_API_ENABLED=1

# Permit W&B's service process to start on loaded nodes.
export WANDB__SERVICE_WAIT="${WANDB__SERVICE_WAIT:-300}"

# Stream logs to the Slurm output file.
export PYTHONUNBUFFERED=1

# --- Select healthy GPUs ------------------------------------------------------
# Pass physical indices to Lightning: Slurm resets CUDA_VISIBLE_DEVICES per task.
HEALTHY=""
if command -v nvidia-smi >/dev/null 2>&1; then
    if HEALTHY="$(python slurm/healthy_gpus.py)"; then
        echo "Healthy GPUs: $HEALTHY"
    else
        echo "[FATAL] no healthy GPUs on $(hostname)"
        exit 42
    fi
fi

# Fail before training if the activated environment cannot use CUDA.
python - <<'PYTHON' || exit 1
import sys

import torch

print(
    f"torch {torch.__version__} | cuda build {torch.version.cuda} | "
    f"{torch.cuda.device_count()} device(s) visible"
)
if not torch.cuda.is_available():
    print("[FATAL] no CUDA device. Either this env is a CPU-only build")
    print("        (check torch.version.cuda) or the job has no GPU.")
    sys.exit(1)
for index in range(torch.cuda.device_count()):
    print(f"  device {index}: {torch.cuda.get_device_name(index)}")
PYTHON

# One srun task uses each selected physical GPU.
if [ -n "$HEALTHY" ]; then
    GPU_LIST="${HEALTHY//,/ }"
else
    NGPU_ALL="$(python -c 'import torch; print(torch.cuda.device_count())')"
    GPU_LIST="$(seq -s' ' 0 $((NGPU_ALL - 1)))"
fi
NGPU="$(echo "$GPU_LIST" | wc -w)"
echo "Training on $NGPU GPU(s): $GPU_LIST"

# Size workers per rank; an explicit --num-workers still wins.
CPUS="${SLURM_CPUS_ON_NODE:-}"
if [ -z "$CPUS" ]; then
    CPUS="$(python -c 'import os; print(len(os.sched_getaffinity(0)))')"
fi
CPUS_PER_TASK=$((CPUS / NGPU))
[ "$CPUS_PER_TASK" -lt 1 ] && CPUS_PER_TASK=1
NUM_WORKERS=$((CPUS_PER_TASK > 1 ? CPUS_PER_TASK - 1 : 1))
echo "$CPUS core(s) over $NGPU rank(s) -> --cpus-per-task $CPUS_PER_TASK," \
     "--num-workers $NUM_WORKERS"

# --- Verify split-query indexes ----------------------------------------------
python - <<'PYTHON'
import os
import sqlite3

for flavour in ("nue", "numu", "nutau"):
    path = os.path.join(os.environ["DATA_ROOT"], f"{flavour}.db")
    with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as connection:
        found = None
        for row in connection.execute("PRAGMA index_list(truth)"):
            columns = [
                info[2]
                for info in connection.execute(f"PRAGMA index_info({row[1]})")
            ]
            if columns[:1] == ["event_no"]:
                found = row[1]
                break
    print(
        f"  {flavour}.db: "
        + (f"indexed ({found})" if found else "NO INDEX - expect a slow scan")
    )
PYTHON

# --- Run ----------------------------------------------------------------------
echo "Job ${SLURM_JOB_ID:-local} started at $(date) on ${SLURM_JOB_NODELIST:-$(hostname)}"
echo "Run name: $RUN_NAME"
echo "Extra arguments: $*"

# One srun task per GPU gives samplers a rank before Trainer.fit starts.
export SRUN_CPUS_PER_TASK="$CPUS_PER_TASK"
EXIT_CODE=0
# $GPU_LIST must word-split into one --gpus value per device.
# shellcheck disable=SC2086
srun --ntasks="$NGPU" --ntasks-per-node="$NGPU" \
     --cpus-per-task="$CPUS_PER_TASK" \
     python examples/04_training/09_train_icemix_joint_reconstruction.py \
    --dataset-config configs/datasets/icecube86_joint_reconstruction.yml \
    --run-name "$RUN_NAME" \
    --output-dir "$OUTPUT_ROOT/$RUN_NAME" \
    --gpus $GPU_LIST \
    --num-workers "$NUM_WORKERS" \
    "$@" || EXIT_CODE=$?

# set -e is active, so a bare `EXIT_CODE=$?` after a failure would never
# run -- the shell exits on the failing python first, so the `|| EXIT_CODE=$?`
# above is what actually captures it.
echo "Job finished with code $EXIT_CODE at $(date)"
exit $EXIT_CODE
