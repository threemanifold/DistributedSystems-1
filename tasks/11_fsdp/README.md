# 11. Study ZeRO-3 behavior with PyTorch FSDP (extension)

Use FSDP to measure the memory and speed tradeoff of parameter sharding, and use `torch.profiler` to inspect how a training step executes.

- [ ] Train the same model under DDP and FSDP; check numerical agreement.
- [ ] Vary FSDP wrapping size and plot peak allocated GPU memory and step time against model size.
- [ ] After warm-up, capture one or a few steps with `torch.profiler`, recording CPU and CUDA activity. Export a separate timeline trace per rank; label forward, backward, and optimizer regions.
- [ ] In the timeline, identify parameter all-gathers and gradient reduce-scatters. Check when they run relative to computation and whether communication overlaps it. Compare traces from coarse and fine wrapping.
- [ ] Use memory measurements or a memory snapshot alongside the timeline to investigate when full parameter copies are present. Report whether the memory saved by finer wrapping is accompanied by additional communication time.
