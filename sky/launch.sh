#!/usr/bin/env bash
# Run a command on the GCP cluster and copy the task's results back.
#
#   sky/launch.sh <task_dir> "<command>" [extra sky launch args]
#
#   sky/launch.sh tasks/05_ddp_conversion "python -m common.train --steps 20"
#   sky/launch.sh tasks/01_nccl_bandwidth "torchrun --nproc_per_node 8 tasks/01_nccl_bandwidth/bench.py" --gpus A100:8
#
# The cluster is reused between calls (setup is skipped if nothing changed) and
# auto-stops after IDLE minutes. Tear it down explicitly with:  sky down $CLUSTER
set -euo pipefail
cd "$(dirname "$0")/.."

TASK=${1:?task dir, e.g. tasks/01_nccl_bandwidth}
CMD=${2:?command to run from the repo root}
shift 2
CLUSTER=${CLUSTER:-dshw}
IDLE=${IDLE:-30}

# Use sky from PATH if the venv is activated, otherwise the one in .venv.
SKY=$(command -v sky || true)
[ -n "$SKY" ] || SKY="$PWD/.venv/bin/sky"
[ -x "$SKY" ] || { echo "sky not found; run: uv sync --extra cloud" >&2; exit 1; }

"$SKY" launch -y -c "$CLUSTER" --idle-minutes-to-autostop "$IDLE" \
  --env TASK="$TASK" --env CMD="$CMD" "$@" sky/task.yaml

mkdir -p "$TASK/results"
rsync -az --progress "$CLUSTER:~/sky_workdir/$TASK/results/" "$TASK/results/"
echo "results synced to $TASK/results/"
