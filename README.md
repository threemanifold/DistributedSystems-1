# NCCL, DDP, and ZeRO: implementation and measurement checklist

The goal is to understand distributed training by implementing small pieces, measuring their behavior, and explaining the results. Start with a single multi-GPU node. Tasks **1–3 and 5–9** form the core path; tasks **4, 10, and 11** are extensions. In tasks 5 and 7, use PyTorch DDP. Task 6 includes a manual synchronization experiment and a comparison with DDP.

## Measurement rules for every applicable task

- [ ] Record GPU model, GPU count, interconnect topology, and CUDA, NCCL, and PyTorch versions.
- [ ] Warm up before timing; run repeated measurements and report median and spread.
- [ ] Check numerical correctness before interpreting speed. Make synchronization boundaries explicit in timings.
- [ ] Keep data type, tensor size, GPU count, and batch semantics consistent when comparing implementations.
- [ ] Save each plot and write a few sentences explaining its main pattern and any unexpected result.



## NCCL



### 1. Reproduce an NCCL bandwidth curve

**Task description:** Benchmark all-reduce with `nccl-tests`, then write an independent PyTorch benchmark that calls `torch.distributed.all_reduce` with the NCCL backend. Compare both measurements with each other and with the A100 NVLink specification.

- [ ] Run `nccl-tests` across small to large message sizes and available GPU counts. Plot latency, algorithm bandwidth, and bus bandwidth.
- [ ] **Hand implement the benchmark harness:** launch one process per GPU, create tensors, warm up, time repeated all-reduces with CUDA events, and calculate throughput. The collective itself may call `torch.distributed.all_reduce`; implementing a collective algorithm belongs to task 3.
- [ ] Check reduced values for correctness. Plot your measurements alongside `nccl-tests` for matching tensor sizes, data types, and GPU counts.
- [ ] Explain differences between measurement methods and bandwidth definitions. Record topology and distinguish advertised bidirectional NVLink bandwidth from measured all-reduce throughput before comparing with the datasheet.



### 2. Find the ring/tree crossover

**Task description:** Determine where NCCL's ring and tree all-reduce algorithms each perform better on your hardware.

- [ ] Force ring and tree in separate runs using `NCCL_ALGO`.
- [ ] Sweep message sizes and plot their latency curves together.
- [ ] Repeat with at least two GPU counts if available, keeping other conditions comparable.
- [ ] Identify the crossover region, or report that no crossover appeared in the tested range.



### 3. Implement a ring all-gather

**Task description:** Build one collective yourself and compare it with NCCL's all-gather.

- [ ] Implement ring all-gather using peer sends and receives. **No AI-generated code for the first working version.**
- [ ] Check that every rank receives the correct data for several tensor sizes and rank counts.
- [ ] Benchmark your implementation and NCCL across message sizes.
- [ ] Plot their latency ratio and investigate the largest performance gaps.



### 4. Compose an all-reduce from two collectives (extension)

**Task description:** Measure the cost of implementing all-reduce as reduce-scatter followed by all-gather.

- [ ] Implement the composition and check its output against direct all-reduce.
- [ ] Benchmark both versions across message sizes.
- [ ] Plot latency and bandwidth, highlighting small messages.
- [ ] Explain where the composed version pays additional overhead.



## DDP



### 5. Turn a timed training loop into PyTorch DDP

**Task description:** Start with a single-GPU training loop and measure what changes when training with PyTorch DDP.

- [ ] Use GPU-resident synthetic data initially. Time forward, backward, optimizer step, and full step.
- [ ] Convert the loop to PyTorch DDP and verify agreement under equivalent global batches and update rules.
- [ ] Sweep **model size**; plot step time and separately measured all-reduce time against parameter count. Do not treat isolated all-reduce time as the overhead added to a DDP step, since communication can overlap backward.
- [ ] Sweep **batch size** separately; explain how increased computation changes the communication fraction.



### 6. Measure how much all-reduce can overlap with backward

**Task description:** Build a minimal manual baseline that synchronizes gradients after backward, then launch reductions as layer gradients become ready. Compare both with PyTorch DDP.

- [ ] Implement the after-backward baseline and check its synchronized gradients.
- [ ] Implement early reductions for a simple layered model. **No AI-generated code for the manual scheduling portion.**
- [ ] Make the number of layers grouped per reduction configurable; also sweep DDP's `bucket_cap_mb`.
- [ ] Plot full step time, separately measured communication time, and an estimate of exposed communication time for each setting. State how you obtained the estimate and keep the noncommunication work comparable.



### 7. Find when adding GPUs stops helping

**Task description:** Connect the collective and overlap measurements to training throughput.

- [ ] Run PyTorch DDP with the available GPU counts.
- [ ] Measure **strong scaling** with a fixed global batch and **weak scaling** with a fixed batch per GPU.
- [ ] Repeat for a small and a compute-heavy model.
- [ ] Plot samples per second, step time, and scaling efficiency; explain the main limit in each case.



## ZeRO



### 8. Implement minimal ZeRO-1 with Adam

**Task description:** Shard Adam's optimizer state while retaining replicated parameters and gradients. Compare with DDP plus replicated Adam.

- [ ] Assign each rank a disjoint parameter shard to update, and distribute updated parameters after each step.
- [ ] Check that parameters agree across ranks and with the DDP baseline after several steps.
- [ ] Sweep model size and plot peak GPU memory and step time for both implementations.
- [ ] Record memory after Adam's optimizer state has been initialized; otherwise the comparison misses the state ZeRO-1 is intended to save.



### 9. Account for the measured memory

**Task description:** Check whether ZeRO-1 memory savings match what the implementation should save.

- [ ] Calculate expected bytes for parameters, gradients, Adam state, and activations.
- [ ] Plot predicted and measured memory against model size.
- [ ] Repeat at two batch sizes to expose the effect of activations.
- [ ] Investigate substantial gaps between prediction and measurement.



### 10. Extend the implementation to ZeRO-2 (extension)

**Task description:** Shard gradients as well as optimizer state, then measure the additional memory saving and communication cost.

- [ ] Replace replicated gradient synchronization with reduce-scatter into owned shards.
- [ ] Check updates against ZeRO-1 on the same batches.
- [ ] Confirm that full gradients are not retained for the entire step, so the intended memory saving is actually realized.
- [ ] Plot peak memory and step time against model size for DDP, ZeRO-1, and ZeRO-2.



### 11. Study ZeRO-3 behavior with PyTorch FSDP (extension)

**Task description:** Use FSDP to measure the memory and speed tradeoff of parameter sharding, and use `torch.profiler` to inspect how a training step executes.

- [ ] Train the same model under DDP and FSDP; check numerical agreement.
- [ ] Vary FSDP wrapping size and plot peak allocated GPU memory and step time against model size.
- [ ] After warm-up, capture one or a few steps with `torch.profiler`, recording CPU and CUDA activity. Export a **separate timeline trace per rank**; label forward, backward, and optimizer regions so they are easy to locate.
- [ ] In the timeline, identify parameter all-gathers and gradient reduce-scatters. Check when they run relative to computation and whether communication overlaps it. Compare traces from coarse and fine wrapping.
- [ ] Use memory measurements or a memory snapshot alongside the timeline to investigate when full parameter copies are present. Report whether the memory saved by finer wrapping is accompanied by additional communication time.