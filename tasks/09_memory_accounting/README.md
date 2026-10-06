# 9. Account for the measured memory

Check whether ZeRO-1 memory savings match what the implementation should save.

- [ ] Calculate expected bytes for parameters, gradients, Adam state, and activations.
- [ ] Plot predicted and measured memory against model size.
- [ ] Repeat at two batch sizes to expose the effect of activations.
- [ ] Investigate substantial gaps between prediction and measurement.
