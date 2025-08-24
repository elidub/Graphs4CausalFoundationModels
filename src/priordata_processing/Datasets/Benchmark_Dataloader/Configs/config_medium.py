"""
Medium benchmark configuration - standard workload
"""

BATCH_SIZE = 128
NUM_WORKERS = 2
PREFETCH_FACTOR = 2
NUM_BATCHES = 1000
WARMUP = 100
SHUFFLE = True

# HTCondor job settings
REQUEST_CPUS = 2
REQUEST_MEMORY = "16GB"
REQUEST_DISK = "20GB"

# Job naming
CONFIG_NAME = "medium"
DESCRIPTION = "Medium benchmark for standard workload testing"
