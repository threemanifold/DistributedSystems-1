# Getting started

The assignment is in `README.md`. This file describes the repo layout and the shared pieces.

## Layout

```
common/         shared code every task starts from (not a task; tasks may import it)
  model.py        MLP of identical Linear+GELU blocks, deterministic synthetic data
  train.py        single-device timed training recipe: the baseline for tasks 5-11
  sweeps.py       shared sweep constants (message sizes, model sizes, batch sizes, GPU counts)
tasks/NN_name/  one folder per task, with its own README (the checklist)
  results/        raw numbers (JSON) + env.txt written by the sky run
  plots/          PNGs, each accompanied by a few sentences in notes.md
sky/            GCP and RunPod launch pipeline (SkyPilot)
modal/          Modal launch pipeline, same interface as sky/launch.sh
```

Rules:

- A task may import `common`, never another task. Copy what you need.
- Keep dtype, sizes, GPU counts and iteration counts from `common/sweeps.py` so plots are comparable across tasks.
- Do not change the model, loss, optimizer or seed in `common/train.py`; agreement checks rely on them.
- Tasks 3 and 6 contain parts that must be written by hand, without AI assistance. See their READMEs.
- Save every plot to `plots/` and write a few sentences about it in `plots/notes.md`.

## Local setup and smoke test (CPU)

```
uv sync                                   # Python 3.12 venv with torch, numpy, matplotlib
uv run python -m common.train --device cpu --hidden 256 --depth 4 --steps 5
bash sky/record_env.sh /tmp/env.txt       # works without a GPU too
```

The task 1 CPU variant (gloo backend) also runs locally:
`uv run torchrun --nproc_per_node 4 <your script>`.

## Running on GCP

One-time:

```
uv sync --extra cloud
gcloud auth login
gcloud auth application-default login     # tick every permission box on the consent page
gcloud config set project <project-id>
gcloud auth application-default set-quota-project <project-id>
uv run sky check gcp
```

Then (no need to activate the venv, the script finds `sky` in `.venv`):

```
# 1-GPU smoke test of the recipe on a cheap L4
CLUSTER=l4 sky/launch.sh tasks/05_ddp_conversion \
  "python -m common.train --steps 20 --out tasks/05_ddp_conversion/results/smoke_l4.json" \
  --gpus L4:1 --retry-until-up

# 8-GPU measurement on A100s
sky/launch.sh tasks/01_nccl_bandwidth \
  "torchrun --nproc_per_node 8 tasks/01_nccl_bandwidth/bench.py" \
  --gpus A100:8 --retry-until-up

sky status
sky down l4        # when finished; clusters also auto-stop after 30 idle minutes
```

`sky/launch.sh` syncs the repo to the VM, runs `setup` (uv env, builds `nccl-tests` against the
NCCL shipped with torch), records hardware and software versions to `<task>/results/env.txt`,
runs the command, and rsyncs `<task>/results/` back to your machine. Nothing is stored in a
bucket; results live in this repo. `all_reduce_perf` and the other `nccl-tests` binaries are on
the PATH inside the run.

## RunPod and Modal

RunPod (one-time: `uv run runpod config` with an API key, then `uv run sky check runpod`): prefix
the commands above with `INFRA=runpod` and use RunPod GPU names, e.g. `--gpus A100-80GB-SXM:8`
(SXM for NVLink; plain `A100-80GB` is PCIe). Pods are on-demand and cannot be stopped, so the
cluster `dshw-runpod` is deleted after `IDLE` idle minutes; `sky down dshw-runpod` when done.

Modal (one-time: `uv run modal setup`): `modal/launch.sh` takes the same arguments as
`sky/launch.sh` with Modal GPU names, e.g. `--gpus A100-80GB:8`, plus `--timeout SECONDS`
(default 3600). Each run is a fresh container, so there is nothing to tear down.

Both use a CUDA 13 container, which needs a host driver >= 580. Check `nvidia-smi topo` in
`env.txt` for NVLink before comparing with the A100 spec.
