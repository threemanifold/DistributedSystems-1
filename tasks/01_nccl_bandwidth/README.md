# 1. Reproduce an NCCL bandwidth curve

Benchmark all-reduce with `nccl-tests`, then write an independent PyTorch benchmark that calls `torch.distributed.all_reduce` with the NCCL backend. Compare both with each other and with the A100 NVLink specification.

- [ ] Run `nccl-tests` (`all_reduce_perf`, on the PATH inside the sky run) across small to large message sizes and available GPU counts. Plot latency, algorithm bandwidth, and bus bandwidth.
- [ ] Hand implement the benchmark harness: one process per GPU, create tensors, warm up, time repeated all-reduces with CUDA events, calculate throughput. The collective itself may call `torch.distributed.all_reduce`.
- [ ] Check reduced values for correctness. Plot your measurements alongside `nccl-tests` for matching sizes, dtype, and GPU counts.
- [ ] Explain differences between measurement methods and bandwidth definitions. Record topology and distinguish advertised bidirectional NVLink bandwidth from measured all-reduce throughput before comparing with the datasheet.

Extra runs agreed for this repo:

- [ ] **NVLink removed**: repeat the PyTorch benchmark with `NCCL_P2P_DISABLE=1` (traffic goes through PCIe / host memory, everything else identical). This isolates what NVLink contributes.
- [ ] **CPU variant**: run the same harness with the `gloo` backend on CPU processes. Do this first, locally, as a prep step before paying for GPUs. It is a curiosity number, not the NVLink comparison: it changes backend, algorithm and memory system at once.

Message sizes come from `common/sweeps.py`.
