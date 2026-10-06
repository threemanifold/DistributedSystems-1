# 3. Implement a ring all-gather

Build one collective yourself and compare it with NCCL's all-gather.

> **No AI-generated code for the first working version** of the ring all-gather.

- [ ] Implement ring all-gather using peer sends and receives (`dist.send` / `dist.recv` / `dist.isend` / `dist.irecv`).
- [ ] Check that every rank receives the correct data for several tensor sizes and rank counts.
- [ ] Benchmark your implementation and NCCL `all_gather` across message sizes.
- [ ] Plot their latency ratio and investigate the largest performance gaps.
