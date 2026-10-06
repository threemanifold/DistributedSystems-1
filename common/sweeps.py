"""Shared sweep constants.

Every task that sweeps message size, model size, batch size or GPU count
should pick its values from here, so plots from different tasks are directly
comparable (same sizes, same dtype, same iteration counts).
"""

DTYPE = "float32"

# All-reduce / all-gather message sizes in bytes, 1 KiB .. 1 GiB.
MESSAGE_SIZES_BYTES = [2**i for i in range(10, 31)]

# Model-size sweep: vary hidden width at fixed depth (see common/model.py).
MODEL_DEPTH = 8
MODEL_HIDDEN_SWEEP = [256, 512, 1024, 2048, 4096]

# Per-GPU batch sizes for the batch-size sweep.
BATCH_SIZE_SWEEP = [32, 64, 128, 256, 512, 1024]

GPU_COUNTS = [1, 2, 4, 8]

WARMUP_ITERS = 10
TIMED_ITERS = 50
