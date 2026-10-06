# 10. Extend the implementation to ZeRO-2 (extension)

Shard gradients as well as optimizer state, then measure the additional memory saving and communication cost.

- [ ] Replace replicated gradient synchronization with reduce-scatter into owned shards.
- [ ] Check updates against ZeRO-1 on the same batches.
- [ ] Confirm that full gradients are not retained for the entire step, so the intended memory saving is actually realized.
- [ ] Plot peak memory and step time against model size for DDP, ZeRO-1, and ZeRO-2.
