# 2. Find the ring/tree crossover

Determine where NCCL's ring and tree all-reduce algorithms each perform better on this hardware.

- [ ] Force ring and tree in separate runs using `NCCL_ALGO=Ring` / `NCCL_ALGO=Tree`.
- [ ] Sweep message sizes and plot their latency curves together.
- [ ] Repeat with at least two GPU counts, keeping other conditions comparable.
- [ ] Identify the crossover region, or report that no crossover appeared in the tested range.

Reuses the task 1 harness idea but must not import from `tasks/01_*`; copy what you need.
