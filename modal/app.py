"""Run a task command in a Modal GPU container and copy <task>/results back.
Use modal/launch.sh (see GETTING_STARTED.md). The image mirrors sky/task.yaml."""

import os
import subprocess
from pathlib import Path

import modal

LOCAL_REPO = Path(__file__).resolve().parent.parent
REPO, VENV = Path("/root/repo"), "/opt/venv"

image = (
    modal.Image.from_registry("nvidia/cuda:13.0.2-devel-ubuntu22.04", add_python="3.12")
    .apt_install("git", "build-essential", "curl")
    .run_commands("curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/usr/local/bin sh")
    .add_local_file(LOCAL_REPO / "pyproject.toml", "/opt/deps/pyproject.toml", copy=True)
    .add_local_file(LOCAL_REPO / "uv.lock", "/opt/deps/uv.lock", copy=True)
    .add_local_file(LOCAL_REPO / ".python-version", "/opt/deps/.python-version", copy=True)
    # Cached until the lock changes. nccl-tests is built against the NCCL shipped with torch.
    .run_commands(
        f"cd /opt/deps && UV_PROJECT_ENVIRONMENT={VENV} uv sync --frozen --no-install-project",
        "git clone --depth 1 https://github.com/NVIDIA/nccl-tests.git /opt/nccl-tests",
        f'NCCL_HOME=$({VENV}/bin/python -c "import nvidia.nccl; print(nvidia.nccl.__path__[0])")'
        ' && { [ -e "$NCCL_HOME/lib/libnccl.so" ] || ln -s libnccl.so.2 "$NCCL_HOME/lib/libnccl.so"; }'
        ' && make -C /opt/nccl-tests -j MPI=0 CUDA_HOME=/usr/local/cuda NCCL_HOME="$NCCL_HOME"',
    )
    # Copied (not mounted) so the repo is writable; old results are not uploaded.
    .add_local_dir(LOCAL_REPO, str(REPO), copy=True,
                   ignore=[".git", ".venv", "nccl-tests", "tasks/*/results", "**/__pycache__"])
)
app = modal.App("ds-hw", image=image)


@app.function(timeout=3600)
def run(task: str, cmd: str) -> tuple[int, dict[str, bytes]]:
    results = REPO / task / "results"
    results.mkdir(parents=True, exist_ok=True)
    # Like `uv run` on the sky VMs, but only for the command: Modal itself needs the image's python.
    nccl = subprocess.check_output([f"{VENV}/bin/python", "-c",
                                    "import nvidia.nccl; print(nvidia.nccl.__path__[0])"], text=True).strip()
    env = {**os.environ, "PATH": f"{VENV}/bin:/opt/nccl-tests/build:{os.environ['PATH']}",
           "PYTHONPATH": str(REPO), "VIRTUAL_ENV": VENV,
           "LD_LIBRARY_PATH": f"{nccl}/lib:{os.environ.get('LD_LIBRARY_PATH', '')}"}

    subprocess.run(["bash", "sky/record_env.sh", str(results / "env.txt")], cwd=REPO, env=env, check=True)
    print(f"+ {cmd}", flush=True)
    rc = subprocess.run(["bash", "-c", cmd], cwd=REPO, env=env).returncode
    return rc, {str(p.relative_to(results)): p.read_bytes() for p in results.rglob("*") if p.is_file()}


@app.local_entrypoint()
def main(task: str, cmd: str, gpus: str = "A100-80GB:1", timeout: int = 3600):
    rc, files = run.with_options(gpu=gpus, timeout=timeout).remote(task, cmd)
    for rel, data in files.items():  # returned even if the command failed
        dest = LOCAL_REPO / task / "results" / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
    print(f"{len(files)} files synced to {task}/results/")
    if rc != 0:
        raise SystemExit(f"command exited with status {rc}")
