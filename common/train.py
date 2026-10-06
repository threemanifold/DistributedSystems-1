"""Single-device training recipe. This is the baseline every task diffs against.

Fixed choices (do not change them in task code, or agreement checks break):
  - model:     common.model.MLP
  - loss:      MSE
  - optimizer: Adam, lr 1e-3
  - data:      common.model.synthetic_batch (deterministic per step)
  - seed:      0

Timing here is intentionally simple (synchronize + perf_counter). The
CUDA-event harness is part of the tasks themselves.

Run locally on CPU as a smoke test:
    uv run python -m common.train --device cpu --hidden 256 --depth 4 --steps 5
Run on a GPU:
    uv run python -m common.train --steps 50
"""
import argparse
import json
import statistics
import time

import torch
import torch.nn.functional as F

from common.model import MLP, ModelConfig, num_params, synthetic_batch


def make_model_and_opt(cfg: ModelConfig, device, seed: int = 0, lr: float = 1e-3):
    torch.manual_seed(seed)
    model = MLP(cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    return model, opt


def train_step(model, opt, x, y):
    """One optimizer step. Returns the loss tensor (not synced)."""
    opt.zero_grad(set_to_none=True)
    loss = F.mse_loss(model(x), y)
    loss.backward()
    opt.step()
    return loss


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def timed_step(model, opt, x, y, device):
    """Same as train_step, but returns per-phase wall times in milliseconds."""
    t = {}
    _sync(device)
    t0 = time.perf_counter()
    opt.zero_grad(set_to_none=True)
    loss = F.mse_loss(model(x), y)
    _sync(device)
    t1 = time.perf_counter()
    loss.backward()
    _sync(device)
    t2 = time.perf_counter()
    opt.step()
    _sync(device)
    t3 = time.perf_counter()
    t["forward_ms"] = (t1 - t0) * 1e3
    t["backward_ms"] = (t2 - t1) * 1e3
    t["optimizer_ms"] = (t3 - t2) * 1e3
    t["step_ms"] = (t3 - t0) * 1e3
    return loss, t


def summarize(samples):
    return {
        "median": statistics.median(samples),
        "min": min(samples),
        "max": max(samples),
        "n": len(samples),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--hidden", type=int, default=1024)
    p.add_argument("--depth", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", help="write a JSON summary to this path")
    args = p.parse_args()

    device = torch.device(args.device)
    cfg = ModelConfig(hidden=args.hidden, depth=args.depth)
    model, opt = make_model_and_opt(cfg, device, args.seed, args.lr)
    print(f"device={device} params={num_params(model):,} batch={args.batch_size}")

    for step in range(args.warmup):
        x, y = synthetic_batch(step, args.batch_size, cfg, device, seed=args.seed)
        train_step(model, opt, x, y)

    times = {k: [] for k in ("forward_ms", "backward_ms", "optimizer_ms", "step_ms")}
    loss = None
    for step in range(args.warmup, args.warmup + args.steps):
        x, y = synthetic_batch(step, args.batch_size, cfg, device, seed=args.seed)
        loss, t = timed_step(model, opt, x, y, device)
        for k, v in t.items():
            times[k].append(v)

    summary = {k: summarize(v) for k, v in times.items()}
    result = {
        "config": vars(args),
        "params": num_params(model),
        "final_loss": loss.item(),
        "timing_ms": summary,
    }
    for k, s in summary.items():
        print(f"{k:13s} median={s['median']:9.3f}  min={s['min']:9.3f}  max={s['max']:9.3f}")
    print(f"final_loss={result['final_loss']:.6f}")
    if args.out:
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
