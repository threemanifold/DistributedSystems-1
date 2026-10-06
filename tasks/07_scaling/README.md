# 7. Find when adding GPUs stops helping

Connect the collective and overlap measurements to training throughput.

- [ ] Run PyTorch DDP with the available GPU counts (`GPU_COUNTS`).
- [ ] Measure strong scaling (fixed global batch) and weak scaling (fixed batch per GPU).
- [ ] Repeat for a small and a compute-heavy model (small and large end of `MODEL_HIDDEN_SWEEP`).
- [ ] Plot samples per second, step time, and scaling efficiency; explain the main limit in each case.
