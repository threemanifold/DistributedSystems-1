# 5. Turn a timed training loop into PyTorch DDP

Start from `common/train.py` and measure what changes when training with PyTorch DDP.

- [ ] Use GPU-resident synthetic data (`common.model.synthetic_batch`). Time forward, backward, optimizer step, and full step.
- [ ] Convert the loop to DDP and verify agreement under equivalent global batches and update rules (same `global_batch_size`, sliced per rank by `synthetic_batch`).
- [ ] Sweep model size (`MODEL_HIDDEN_SWEEP`); plot step time and separately measured all-reduce time against parameter count. Do not treat isolated all-reduce time as the overhead added to a DDP step, since communication can overlap backward.
- [ ] Sweep batch size (`BATCH_SIZE_SWEEP`) separately; explain how increased computation changes the communication fraction.
