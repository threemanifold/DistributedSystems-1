# 4. Compose an all-reduce from two collectives (extension)

Measure the cost of implementing all-reduce as reduce-scatter followed by all-gather.

- [ ] Implement the composition and check its output against direct all-reduce.
- [ ] Benchmark both versions across message sizes.
- [ ] Plot latency and bandwidth, highlighting small messages.
- [ ] Explain where the composed version pays additional overhead.
