"""
Small benchmark configuration - suitable for quick tests
"""

BATCH_SIZE = 32
NUM_WORKERS = 1
PREFETCH_FACTOR = 2
NUM_BATCHES = 100
WARMUP = 10
SHUFFLE = True

# HTCondor job settings
REQUEST_CPUS = 1
REQUEST_MEMORY = "8GB"
REQUEST_DISK = "10GB"

# Job naming
CONFIG_NAME = "small"
DESCRIPTION = "Small benchmark for quick testing"
