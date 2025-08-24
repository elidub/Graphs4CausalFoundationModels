"""
Large benchmark configuration - intensive workload
"""

BATCH_SIZE = 256
NUM_WORKERS = 4
PREFETCH_FACTOR = 4
NUM_BATCHES = 2000
WARMUP = 200
SHUFFLE = True

# HTCondor job settings
REQUEST_CPUS = 4
REQUEST_MEMORY = "32GB"
REQUEST_DISK = "30GB"

# Job naming
CONFIG_NAME = "large"
DESCRIPTION = "Large benchmark for intensive workload testing"
