#!/usr/bin/env bash
# Record hardware and software versions next to the results of a run.
# Usage: sky/record_env.sh <output file>
# Works on a CPU-only machine too (GPU sections are skipped).
out=${1:-env.txt}
{
  echo "== date";       date -u +"%Y-%m-%dT%H:%M:%SZ"
  echo "== host";       hostname; uname -a
  echo "== python";     python --version 2>&1
  echo "== torch"
  python - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
print("cuda", torch.version.cuda)
print("cudnn", torch.backends.cudnn.version())
try:
    print("nccl", ".".join(map(str, torch.cuda.nccl.version())))
except Exception as e:
    print("nccl", "n/a", e)
if torch.cuda.is_available():
    print("gpu_count", torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        print(f"gpu{i}", p.name, f"{p.total_memory/2**30:.1f}GiB", f"sm_{p.major}{p.minor}")
PY
  if command -v nvidia-smi >/dev/null; then
    echo "== nvidia-smi";      nvidia-smi
    echo "== nvidia-smi topo"; nvidia-smi topo -m
  else
    echo "== nvidia-smi: not available (CPU machine)"
  fi
  if command -v nvcc >/dev/null; then echo "== nvcc"; nvcc --version; fi
  echo "== pip"; python -m pip list 2>/dev/null | grep -iE "^(torch|numpy|nvidia-nccl)" || true
} > "$out" 2>&1
echo "environment recorded to $out"
