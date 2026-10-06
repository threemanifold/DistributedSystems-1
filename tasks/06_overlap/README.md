# 6. Measure how much all-reduce can overlap with backward

Build a minimal manual baseline that synchronizes gradients after backward, then launch reductions as layer gradients become ready. Compare both with PyTorch DDP.

> **No AI-generated code for the manual scheduling portion** (the early-reduction logic).

- [ ] Implement the after-backward baseline and check its synchronized gradients.
- [ ] Implement early reductions for the layered `common.model.MLP` (each `Block` is one layer).
- [ ] Make the number of layers grouped per reduction configurable; also sweep DDP's `bucket_cap_mb`.
- [ ] Plot full step time, separately measured communication time, and an estimate of exposed communication time for each setting. State how you obtained the estimate and keep the noncommunication work comparable.
