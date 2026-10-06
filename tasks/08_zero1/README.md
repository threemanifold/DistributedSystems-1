# 8. Implement minimal ZeRO-1 with Adam

Shard Adam's optimizer state while retaining replicated parameters and gradients. Compare with DDP plus replicated Adam.

- [ ] Assign each rank a disjoint parameter shard to update, and distribute updated parameters after each step.
- [ ] Check that parameters agree across ranks and with the DDP baseline after several steps.
- [ ] Sweep model size and plot peak GPU memory and step time for both implementations.
- [ ] Record memory after Adam's optimizer state has been initialized (after the first step), otherwise the comparison misses the state ZeRO-1 is intended to save.
