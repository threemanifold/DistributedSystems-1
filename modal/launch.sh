#!/usr/bin/env bash
# Run a command in a Modal GPU container and copy the task's results back.
#
#   modal/launch.sh <task_dir> "<command>" [--gpus GPU:N] [--timeout SECONDS]
#   modal/launch.sh tasks/01_nccl_bandwidth "torchrun --nproc_per_node 8 tasks/01_nccl_bandwidth/bench.py" --gpus A100-80GB:8
set -euo pipefail
cd "$(dirname "$0")/.."

TASK=${1:?task dir, e.g. tasks/01_nccl_bandwidth}
CMD=${2:?command to run from the repo root}
shift 2

MODAL=$(command -v modal || true)
[ -n "$MODAL" ] || MODAL="$PWD/.venv/bin/modal"
[ -x "$MODAL" ] || { echo "modal not found; run: uv sync --extra cloud" >&2; exit 1; }

exec "$MODAL" run modal/app.py --task "$TASK" --cmd "$CMD" "$@"
